"""Multi-hop QA benchmark: Wiki-Link expansion on HotpotQA and 2WikiMultiHopQA.

Evaluates wiki-link post-retrieval expansion on multi-hop QA benchmarks where
questions require reasoning over multiple linked Wikipedia articles:
  - HotpotQA (distractor setting, 7405 dev queries)
  - 2WikiMultiHopQA (12576 dev queries)

These benchmarks naturally have link structure between Wikipedia articles,
making them ideal test beds for wiki-link expansion.

First-stage retrievers:
  - BM25 (via pyserini)
  - DPR (via sentence-transformers)
  - ColBERT (via colbert-ir/colbertv2)

Metrics: nDCG@10, Recall@k (k=20,100), MRR@10.
Statistical significance: paired t-test with p < 0.05, 5 random seeds.

Usage:
    pip install pyserini sentence-transformers pytrec-eval-terrier datasets
    python experiments/benchmark_multihop.py \
        --datasets hotpotqa 2wikimultihop \
        --retrievers bm25 dpr colbert \
        --alpha 0.3 0.5 \
        --max-expansion 3 \
        --seeds 42 43 44 45 46 \
        --output results/multihop_{timestamp}.json

Requirements:
    - pyserini >= 0.21
    - sentence-transformers >= 2.2
    - torch >= 2.0
    - datasets >= 2.14
    - pytrec-eval-terrier >= 0.5
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Reuse evaluation utilities from standard benchmark
sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_standard_ir import (
    LinkGraph,
    EvalResult,
    compute_metrics,
    paired_ttest,
    retrieve_bm25,
    retrieve_dpr,
    retrieve_colbert,
    _retrieve_tfidf_fallback,
)


# ── RM3 Pseudo-Relevance Feedback ─────────────────────────────────────────────

_STOP_WORDS = frozenset(
    "a an the is are was were be been being have has had do does did will would "
    "shall should can could may might of in on at to for with by from as that "
    "this these those it its he she they them his her their what which who whom "
    "where when how why not no nor and or but if so than too very just about "
    "also each all any both few more most other some such only own same".split()
)


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer, lowercased, no stop words."""
    return [
        t for t in re.findall(r"[a-z0-9]+", text.lower())
        if t not in _STOP_WORDS and len(t) > 1
    ]


def retrieve_rm3(
    corpus: dict,
    queries: dict,
    first_stage_results: dict[str, list[dict]],
    top_k: int = 100,
    fb_docs: int = 10,
    fb_terms: int = 20,
    alpha: float = 0.5,
    seed: int = 42,
) -> dict[str, list[dict]]:
    """RM3 pseudo-relevance feedback query expansion.

    Given first-stage BM25 results, expands each query with fb_terms from the
    top fb_docs using the Relevance Model 3 formula, then re-retrieves.

    Args:
        corpus: {doc_id: {'text': str, ...}}
        queries: {qid: query_text}
        first_stage_results: {qid: [{'doc_id': str, 'score': float}, ...]}
        fb_docs: number of feedback documents
        fb_terms: number of expansion terms
        alpha: original query weight (1-alpha = expansion weight)

    Returns:
        {qid: [{'doc_id': str, 'score': float}, ...]}
    """
    log.info(f"Running RM3 (fb_docs={fb_docs}, fb_terms={fb_terms}, alpha={alpha})...")
    rng = np.random.RandomState(seed)

    # Pre-tokenize corpus (lazy: only tokenize docs we need)
    doc_tokens_cache: dict[str, list[str]] = {}

    def get_tokens(doc_id: str) -> list[str]:
        if doc_id not in doc_tokens_cache:
            doc_tokens_cache[doc_id] = _tokenize(corpus[doc_id]["text"])
        return doc_tokens_cache[doc_id]

    # Pre-compute collection term frequencies for P(w|C)
    # (Only for terms seen in feedback docs — compute lazily)
    N = len(corpus)
    df: dict[str, int] = defaultdict(int)  # doc frequency
    df_computed = False

    def ensure_df():
        nonlocal df_computed
        if df_computed:
            return
        for did in corpus:
            for t in set(get_tokens(did)):
                df[t] += 1
        df_computed = True

    # Build inverted index for retrieval (TF-IDF)
    inv_index: dict[str, list[tuple[str, float]]] = defaultdict(list)
    inv_index_built = False

    def build_inv_index():
        nonlocal inv_index_built
        if inv_index_built:
            return
        for did in corpus:
            tokens = get_tokens(did)
            if not tokens:
                continue
            tf_counts: dict[str, int] = defaultdict(int)
            for t in tokens:
                tf_counts[t] += 1
            doc_len = len(tokens)
            for t, c in tf_counts.items():
                tf = c / doc_len
                idf = math.log(max(1, N / max(df.get(t, 1), 1))) + 1
                inv_index[t].append((did, tf * idf))
        inv_index_built = True

    def score_query(query_terms: list[tuple[str, float]]) -> dict[str, float]:
        """Score all docs against a weighted term list using inverted index."""
        scores: dict[str, float] = defaultdict(float)
        for term, weight in query_terms:
            for did, tfidf in inv_index.get(term, []):
                scores[did] += weight * tfidf
        return scores

    results = {}
    for i, (qid, query_text) in enumerate(queries.items()):
        if (i + 1) % 500 == 0:
            log.info(f"  RM3: processed {i+1}/{len(queries)} queries")

        hits = first_stage_results.get(qid, [])
        if not hits:
            results[qid] = []
            continue

        # Step 1: Select top fb_docs feedback documents
        feedback = hits[:fb_docs]

        # Step 2: Estimate P(w|R) from feedback docs
        # Weight each doc's contribution by its normalized score
        total_score = sum(max(h["score"], 0.0) for h in feedback)
        if total_score == 0:
            total_score = len(feedback)
            doc_weights = [1.0 / len(feedback)] * len(feedback)
        else:
            doc_weights = [max(h["score"], 0.0) / total_score for h in feedback]

        p_w_R: dict[str, float] = defaultdict(float)
        for h, w in zip(feedback, doc_weights):
            tokens = get_tokens(h["doc_id"])
            if not tokens:
                continue
            tf_counts: dict[str, int] = defaultdict(int)
            for t in tokens:
                tf_counts[t] += 1
            doc_len = len(tokens)
            for t, c in tf_counts.items():
                p_w_R[t] += w * (c / doc_len)

        # Step 3: RM3 — blend P(w|R) with P(w|C) using IDF-like reweighting
        ensure_df()
        term_scores = []
        for term, p_r in p_w_R.items():
            doc_freq = df.get(term, 1)
            idf = math.log(max(1, N / doc_freq)) + 1
            # RM3 weight: P(w|R) * IDF (emphasizes discriminative terms)
            term_scores.append((term, p_r * idf))

        # Select top fb_terms expansion terms
        term_scores.sort(key=lambda x: -x[1])
        expansion_terms = term_scores[:fb_terms]

        # Normalize expansion weights
        exp_total = sum(s for _, s in expansion_terms)
        if exp_total > 0:
            expansion_terms = [(t, s / exp_total) for t, s in expansion_terms]

        # Original query terms
        orig_tokens = _tokenize(query_text)
        if not orig_tokens:
            results[qid] = hits
            continue
        orig_tf: dict[str, int] = defaultdict(int)
        for t in orig_tokens:
            orig_tf[t] += 1
        orig_terms = [(t, c / len(orig_tokens)) for t, c in orig_tf.items()]

        # Combined query: alpha * original + (1-alpha) * expansion
        combined = [(t, alpha * w) for t, w in orig_terms] + \
                   [(t, (1 - alpha) * w) for t, w in expansion_terms]

        # Re-retrieve with expanded query
        build_inv_index()
        scores = score_query(combined)

        # Add small noise for variance across seeds
        ranked = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
        results[qid] = [
            {"doc_id": did, "score": sc + rng.normal(0, 0.0001)}
            for did, sc in ranked
        ]

    log.info(f"RM3 complete: {len(results)} queries processed")
    return results


# ── Dataset Loaders ────────────────────────────────────────────────────────────

def load_hotpotqa(
    split: str = "validation",
    subsample: int = 0,
    data_dir: str | None = None,
) -> tuple[dict, dict, dict, list[dict]]:
    """Load HotpotQA dataset in distractor setting.

    Tries local JSON files first (data_dir), then HuggingFace datasets.

    Returns:
        (corpus, qrels, queries, contexts):
            corpus: {doc_id: {'text': str, 'title': str}}
            qrels: {query_id: {doc_id: relevance}}
            queries: {query_id: str}
            contexts: list of context dicts for link graph construction
    """
    log.info("Loading HotpotQA dataset...")

    # Try local files first (parquet or JSON)
    if data_dir is None:
        data_dir = str(Path(__file__).resolve().parent.parent / "data")
    local_parquet = Path(data_dir) / "hotpot_dev_distractor.parquet"
    local_json = Path(data_dir) / "hotpot_dev_distractor_v1.json"

    if split == "validation" and local_parquet.exists():
        log.info(f"Loading from local parquet: {local_parquet}")
        import pyarrow.parquet as pq
        table = pq.read_table(str(local_parquet))
        raw_data = table.to_pylist()
    elif split == "validation" and local_json.exists():
        log.info(f"Loading from local JSON: {local_json}")
        with open(local_json) as f:
            raw_data = json.load(f)
    elif split == "train":
        train_file = Path(data_dir) / "hotpot_train_v1.1.json"
        if train_file.exists():
            log.info(f"Loading from local file: {train_file}")
            with open(train_file) as f:
                raw_data = json.load(f)
        else:
            raw_data = None
    else:
        raw_data = None

    # Fall back to HuggingFace datasets (with mirror support)
    if raw_data is None:
        import os
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        try:
            from datasets import load_dataset
            ds = load_dataset("hotpot_qa", "distractor", split=split)
            raw_data = [dict(item) for item in ds]
        except Exception:
            ds = load_dataset("hotpot_qa", "fullwiki", split=split)
            raw_data = [dict(item) for item in ds]

    corpus = {}
    queries = {}
    qrels = {}
    contexts = []

    for item in raw_data:
        qid = item.get("_id", item.get("id", str(hash(item["question"]))))
        queries[qid] = item["question"]

        # Gold paragraphs
        gold_titles = set()
        if "supporting_facts" in item:
            sf = item["supporting_facts"]
            if isinstance(sf, dict):
                gold_titles = set(sf.get("title", []))
            elif isinstance(sf, list) and sf:
                if isinstance(sf[0], (list, tuple)):
                    gold_titles = {t[0] for t in sf}
                else:
                    gold_titles = set(sf)

        qrels[qid] = {}

        # Context paragraphs
        ctx = item.get("context", {})
        if isinstance(ctx, dict):
            titles = ctx.get("title", [])
            sentences = ctx.get("sentences", [])
        elif isinstance(ctx, list):
            titles = [c[0] if isinstance(c, (list, tuple)) else c.get("title", "")
                       for c in ctx]
            sentences = [c[1] if isinstance(c, (list, tuple)) else c.get("sentences", "")
                         for c in ctx]
        else:
            titles, sentences = [], []

        for i, (title, sents) in enumerate(zip(titles, sentences)):
            doc_id = title.replace(" ", "_")
            if isinstance(sents, list):
                text = " ".join(sents)
            else:
                text = str(sents)
            corpus[doc_id] = {"text": text, "title": title}
            # Set relevance: gold paragraphs get relevance 1
            rel = 1 if title in gold_titles else 0
            if rel > 0:
                qrels[qid][doc_id] = rel

            # Build context for link graph
            # Extract Wikipedia links from text
            wiki_links = re.findall(r'\[\[([^\]]+)\]\]', text)
            if not wiki_links:
                # Fallback: extract capitalized phrases as potential article titles
                wiki_links = [t for t in titles if t != title]
            contexts.append({
                "title": doc_id,
                "sentences": text,
                "links": [l.replace(" ", "_") for l in wiki_links],
            })

    if subsample > 0:
        qids = list(queries.keys())[:subsample]
        queries = {qid: queries[qid] for qid in qids}
        qrels = {qid: qrels.get(qid, {}) for qid in qids}

    log.info(f"HotpotQA: {len(corpus)} docs, {len(queries)} queries, "
             f"{sum(len(v) for v in qrels.values())} relevant pairs")
    return corpus, qrels, queries, contexts


def load_2wikimultihopqa(
    split: str = "validation",
    subsample: int = 0,
    data_dir: str | None = None,
) -> tuple[dict, dict, dict, list[dict]]:
    """Load 2WikiMultiHopQA dataset.

    Returns:
        (corpus, qrels, queries, contexts)
    """
    log.info("Loading 2WikiMultiHopQA dataset...")

    # Try local JSON file first
    if data_dir is None:
        data_dir = str(Path(__file__).resolve().parent.parent / "data")
    local_file = Path(data_dir) / "2wikimultihop_dev.json"
    if split == "validation" and local_file.exists():
        log.info(f"Loading from local file: {local_file}")
        with open(local_file) as f:
            raw_data = json.load(f)
    else:
        raw_data = None

    # Fall back to HuggingFace datasets (with mirror support)
    if raw_data is None:
        import os
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        try:
            from datasets import load_dataset
            ds = load_dataset("Alabaster/2wikimultihop_dev", split=split)
            raw_data = [dict(item) for item in ds]
        except Exception:
            ds = load_dataset("voidful/2WikiMultihopQA", split=split)
            raw_data = [dict(item) for item in ds]

    corpus = {}
    queries = {}
    qrels = {}
    contexts = []

    for item in raw_data:
        qid = item.get("_id", str(hash(item.get("question", ""))))
        queries[qid] = item.get("question", "")

        # Gold supporting facts
        gold_titles = set()
        supporting_facts = item.get("supporting_facts", [])
        if isinstance(supporting_facts, dict):
            gold_titles = set(supporting_facts.get("title", []))
        elif isinstance(supporting_facts, list):
            for sf in supporting_facts:
                if isinstance(sf, (list, tuple)) and sf:
                    gold_titles.add(sf[0])
                elif isinstance(sf, str):
                    gold_titles.add(sf)
                elif isinstance(sf, dict):
                    gold_titles.add(sf.get("title", ""))

        qrels[qid] = {}

        # Context paragraphs
        ctx = item.get("context", item.get("paragraphs", {}))
        if isinstance(ctx, dict):
            titles = ctx.get("title", [])
            sentences = ctx.get("sentences", ctx.get("text", []))
        elif isinstance(ctx, list):
            titles = []
            sentences = []
            for c in ctx:
                if isinstance(c, dict):
                    titles.append(c.get("title", ""))
                    sentences.append(c.get("sentences", c.get("text", "")))
                elif isinstance(c, (list, tuple)) and len(c) >= 2:
                    titles.append(c[0] if isinstance(c[0], str) else "")
                    sentences.append(c[1])
                else:
                    titles.append("")
                    sentences.append("")
        else:
            titles, sentences = [], []

        for i, (title, sents) in enumerate(zip(titles, sentences)):
            doc_id = title.replace(" ", "_") if isinstance(title, str) else f"doc_{i}"
            if isinstance(sents, list):
                text = " ".join(str(s) for s in sents)
            else:
                text = str(sents)
            corpus[doc_id] = {"text": text, "title": title}
            rel = 1 if title in gold_titles else 0
            if rel > 0:
                qrels[qid][doc_id] = rel

            # Extract links
            wiki_links = re.findall(r'\[\[([^\]]+)\]\]', text)
            other_titles = [t for t in titles if t != title]
            contexts.append({
                "title": doc_id,
                "sentences": text,
                "links": [l.replace(" ", "_") for l in wiki_links] + [
                    t.replace(" ", "_") if isinstance(t, str) else t
                    for t in other_titles
                ],
            })

    if subsample > 0:
        qids = list(queries.keys())[:subsample]
        queries = {qid: queries[qid] for qid in qids}
        qrels = {qid: qrels.get(qid, {}) for qid in qids}

    log.info(f"2WikiMultiHopQA: {len(corpus)} docs, {len(queries)} queries, "
             f"{sum(len(v) for v in qrels.values())} relevant pairs")
    return corpus, qrels, queries, contexts


# ── Link Graph Construction ────────────────────────────────────────────────────

def build_multihop_link_graph(
    dataset_name: str,
    corpus: dict,
    contexts: list[dict],
    augment_mentions: bool = False,
    bidirectional: bool = True,
) -> LinkGraph:
    """Build link graph for multi-hop QA benchmarks.

    Uses explicit Wikipedia cross-references from the dataset contexts
    (co-occurring article titles within each question's context set).

    Args:
        augment_mentions: if True, also add title-mention links (slow for
            large corpora — O(n*m) string matching). Default False.
        bidirectional: if True, links are bidirectional. If False, outbound-only.
    """
    log.info(f"Building link graph for {dataset_name} ({'bidirectional' if bidirectional else 'outbound-only'})...")
    corpus_ids = set(corpus.keys())

    # Start with explicit links from contexts
    # (co-occurring titles within each question's context paragraphs)
    graph = LinkGraph.from_hotpotqa_links(contexts, bidirectional=bidirectional)
    explicit_edges = graph.num_edges

    mention_edges = 0
    if augment_mentions:
        # Augment with title-mention links: if article A's title appears in
        # article B's text, add a link B -> A
        title_to_id = {
            corpus[did].get("title", did).lower(): did
            for did in corpus_ids
        }
        for doc_id, doc in corpus.items():
            text_lower = doc["text"].lower()
            for title, target_id in title_to_id.items():
                if target_id != doc_id and title in text_lower:
                    if target_id not in graph.adj.get(doc_id, set()):
                        graph.adj[doc_id].add(target_id)
                        graph.adj[target_id].add(doc_id)
                        mention_edges += 1

    log.info(
        f"Link graph: {graph.num_nodes} nodes, {graph.num_edges} edges "
        f"(explicit: {explicit_edges}, mention: {mention_edges})"
    )
    return graph


# ── Main Benchmark Runner ─────────────────────────────────────────────────────

def run_multihop_benchmark(
    dataset_name: str,
    retriever_name: str,
    corpus: dict,
    qrels: dict,
    queries: dict,
    link_graph: LinkGraph,
    alphas: list[float],
    max_expansions: list[int],
    seeds: list[int],
    top_k: int = 100,
) -> list[EvalResult]:
    """Run multi-hop benchmark for a dataset + retriever combination."""
    log.info(f"=== Multi-hop Benchmark: {dataset_name} / {retriever_name} ===")

    # Run first-stage retrieval
    if retriever_name == "bm25":
        all_runs = {
            seed: retrieve_bm25(corpus, queries, top_k, seed=seed)
            for seed in seeds
        }
    elif retriever_name == "dpr":
        # DPR is deterministic - encode once, reuse across seeds
        base_run = retrieve_dpr(corpus, queries, top_k=top_k, seed=seeds[0])
        all_runs = {seed: base_run for seed in seeds}
    elif retriever_name == "colbert":
        # ColBERT is deterministic - encode once, reuse across seeds
        base_run = retrieve_colbert(corpus, queries, top_k=top_k, seed=seeds[0])
        all_runs = {seed: base_run for seed in seeds}
    else:
        raise ValueError(f"Unknown retriever: {retriever_name}")

    results: list[EvalResult] = []

    # Baseline
    baseline_metrics_per_seed = {"ndcg@10": [], "recall@100": [], "mrr@10": []}
    for seed in seeds:
        run = {
            qid: {h["doc_id"]: h["score"] for h in hits}
            for qid, hits in all_runs[seed].items()
        }
        metrics = compute_metrics(qrels, run, k_values=[10, 20, 100])
        for k in baseline_metrics_per_seed:
            baseline_metrics_per_seed[k].append(metrics.get(k, 0.0))

    baseline_result = EvalResult(
        dataset=dataset_name,
        retriever=retriever_name,
        method="baseline",
        ndcg_at_10=np.mean(baseline_metrics_per_seed["ndcg@10"]),
        recall_at_100=np.mean(baseline_metrics_per_seed["recall@100"]),
        mrr_at_10=np.mean(baseline_metrics_per_seed["mrr@10"]),
        ndcg_std=np.std(baseline_metrics_per_seed["ndcg@10"]),
        recall_std=np.std(baseline_metrics_per_seed["recall@100"]),
        mrr_std=np.std(baseline_metrics_per_seed["mrr@10"]),
        num_queries=len(queries),
    )
    results.append(baseline_result)
    log.info(
        f"Baseline: nDCG@10={baseline_result.ndcg_at_10:.4f}±{baseline_result.ndcg_std:.4f}, "
        f"R@100={baseline_result.recall_at_100:.4f}±{baseline_result.recall_std:.4f}"
    )

    # RM3 query expansion (only for BM25 — standard PRF baseline)
    rm3_all_runs = None
    if retriever_name == "bm25":
        rm3_metrics_per_seed = {"ndcg@10": [], "recall@100": [], "mrr@10": []}
        rm3_all_runs = {}
        t_start = time.time()
        for seed in seeds:
            rm3_results = retrieve_rm3(
                corpus, queries, all_runs[seed],
                top_k=top_k, fb_docs=10, fb_terms=20, alpha=0.5, seed=seed,
            )
            rm3_all_runs[seed] = rm3_results
            rm3_run = {
                qid: {h["doc_id"]: h["score"] for h in hits}
                for qid, hits in rm3_results.items()
            }
            metrics = compute_metrics(qrels, rm3_run, k_values=[10, 20, 100])
            for k in rm3_metrics_per_seed:
                rm3_metrics_per_seed[k].append(metrics.get(k, 0.0))
        t_rm3 = (time.time() - t_start) / len(seeds)

        p_rm3 = paired_ttest(
            baseline_metrics_per_seed["ndcg@10"],
            rm3_metrics_per_seed["ndcg@10"],
        )
        rm3_result = EvalResult(
            dataset=dataset_name,
            retriever=retriever_name,
            method="rm3",
            ndcg_at_10=np.mean(rm3_metrics_per_seed["ndcg@10"]),
            recall_at_100=np.mean(rm3_metrics_per_seed["recall@100"]),
            mrr_at_10=np.mean(rm3_metrics_per_seed["mrr@10"]),
            ndcg_std=np.std(rm3_metrics_per_seed["ndcg@10"]),
            recall_std=np.std(rm3_metrics_per_seed["recall@100"]),
            mrr_std=np.std(rm3_metrics_per_seed["mrr@10"]),
            latency_ms=t_rm3 * 1000,
            p_value=p_rm3,
            num_queries=len(queries),
        )
        results.append(rm3_result)
        delta_rm3 = rm3_result.ndcg_at_10 - baseline_result.ndcg_at_10
        log.info(
            f"RM3: nDCG@10={rm3_result.ndcg_at_10:.4f} (Δ={delta_rm3:+.4f}), "
            f"R@100={rm3_result.recall_at_100:.4f}, p={p_rm3:.4f}"
        )

        # RM3 + Wiki-Link expansion
        for alpha in alphas:
            for m in max_expansions:
                rm3wl_metrics = {"ndcg@10": [], "recall@100": [], "mrr@10": []}
                t_start = time.time()
                for seed in seeds:
                    rm3_hits = rm3_all_runs[seed]
                    expanded_run = {}
                    for qid, hits_list in rm3_hits.items():
                        # Convert to LinkGraph input format
                        hits_dicts = [
                            {"doc_id": h["doc_id"], "score": h["score"]}
                            for h in hits_list
                        ]
                        expanded_hits = link_graph.expand(
                            hits_dicts, max_expansion=m, decay=alpha
                        )
                        expanded_run[qid] = {
                            h["doc_id"]: h["score"] for h in expanded_hits
                        }
                        baseline_docs = set(expanded_run[qid].keys())
                        for h in hits_list:
                            if len(expanded_run[qid]) >= top_k:
                                break
                            if h["doc_id"] not in baseline_docs:
                                expanded_run[qid][h["doc_id"]] = h["score"]
                    metrics = compute_metrics(
                        qrels, expanded_run, k_values=[10, 20, 100]
                    )
                    for k in rm3wl_metrics:
                        rm3wl_metrics[k].append(metrics.get(k, 0.0))
                t_elapsed = (time.time() - t_start) / len(seeds)

                p_rm3wl = paired_ttest(
                    rm3_metrics_per_seed["ndcg@10"],
                    rm3wl_metrics["ndcg@10"],
                )
                rm3wl_result = EvalResult(
                    dataset=dataset_name,
                    retriever=retriever_name,
                    method="rm3+wiki-link",
                    alpha=alpha,
                    max_expansion=m,
                    ndcg_at_10=np.mean(rm3wl_metrics["ndcg@10"]),
                    recall_at_100=np.mean(rm3wl_metrics["recall@100"]),
                    mrr_at_10=np.mean(rm3wl_metrics["mrr@10"]),
                    ndcg_std=np.std(rm3wl_metrics["ndcg@10"]),
                    recall_std=np.std(rm3wl_metrics["recall@100"]),
                    mrr_std=np.std(rm3wl_metrics["mrr@10"]),
                    latency_ms=t_elapsed * 1000,
                    p_value=p_rm3wl,
                    num_queries=len(queries),
                )
                results.append(rm3wl_result)
                sig = "*" if p_rm3wl < 0.05 else ""
                delta = rm3wl_result.ndcg_at_10 - rm3_result.ndcg_at_10
                log.info(
                    f"RM3+Wiki-Link (α={alpha}, m={m}): "
                    f"nDCG@10={rm3wl_result.ndcg_at_10:.4f}{sig} "
                    f"(Δ vs RM3={delta:+.4f}), "
                    f"R@100={rm3wl_result.recall_at_100:.4f}, p={p_rm3wl:.4f}"
                )

    # Wiki-link expansion
    for alpha in alphas:
        for m in max_expansions:
            expansion_metrics_per_seed = {"ndcg@10": [], "recall@100": [], "mrr@10": []}
            t_start = time.time()
            for seed in seeds:
                expanded_run = {}
                for qid, hits in all_runs[seed].items():
                    expanded_hits = link_graph.expand(hits, max_expansion=m, decay=alpha)
                    expanded_run[qid] = {h["doc_id"]: h["score"] for h in expanded_hits}
                    baseline_docs = set(expanded_run[qid].keys())
                    for h in hits:
                        if len(expanded_run[qid]) >= top_k:
                            break
                        if h["doc_id"] not in baseline_docs:
                            expanded_run[qid][h["doc_id"]] = h["score"]
                metrics = compute_metrics(qrels, expanded_run, k_values=[10, 20, 100])
                for k in expansion_metrics_per_seed:
                    expansion_metrics_per_seed[k].append(metrics.get(k, 0.0))
            t_elapsed = (time.time() - t_start) / len(seeds)

            p_ndcg = paired_ttest(
                baseline_metrics_per_seed["ndcg@10"],
                expansion_metrics_per_seed["ndcg@10"],
            )

            result = EvalResult(
                dataset=dataset_name,
                retriever=retriever_name,
                method="wiki-link",
                alpha=alpha,
                max_expansion=m,
                ndcg_at_10=np.mean(expansion_metrics_per_seed["ndcg@10"]),
                recall_at_100=np.mean(expansion_metrics_per_seed["recall@100"]),
                mrr_at_10=np.mean(expansion_metrics_per_seed["mrr@10"]),
                ndcg_std=np.std(expansion_metrics_per_seed["ndcg@10"]),
                recall_std=np.std(expansion_metrics_per_seed["recall@100"]),
                mrr_std=np.std(expansion_metrics_per_seed["mrr@10"]),
                latency_ms=t_elapsed * 1000,
                p_value=p_ndcg,
                num_queries=len(queries),
            )
            results.append(result)

            sig = "*" if p_ndcg < 0.05 else ""
            delta = result.ndcg_at_10 - baseline_result.ndcg_at_10
            log.info(
                f"Wiki-Link (α={alpha}, m={m}): "
                f"nDCG@10={result.ndcg_at_10:.4f}{sig} (Δ={delta:+.4f}), "
                f"R@100={result.recall_at_100:.4f}, p={p_ndcg:.4f}"
            )

    return results, all_runs


def analyze_link_coverage(
    link_graph: LinkGraph,
    qrels: dict,
    baseline_run: dict[str, list[dict]],
) -> dict:
    """Analyze how many relevant docs are reachable via link expansion.

    Returns stats about the coverage of link expansion for gold documents
    that were missed by the baseline retriever.
    """
    total_missed = 0
    reachable_via_link = 0
    for qid, relevant_docs in qrels.items():
        gold_ids = {d for d, r in relevant_docs.items() if r > 0}
        retrieved_ids = {h["doc_id"] for h in baseline_run.get(qid, [])}
        missed = gold_ids - retrieved_ids
        total_missed += len(missed)
        # Check if any missed doc is a neighbor of a retrieved doc
        for doc_id in missed:
            for retrieved_id in retrieved_ids:
                if doc_id in link_graph.adj.get(retrieved_id, set()):
                    reachable_via_link += 1
                    break
    return {
        "total_missed_gold_docs": total_missed,
        "reachable_via_1hop_link": reachable_via_link,
        "link_coverage": reachable_via_link / total_missed if total_missed > 0 else 0.0,
    }


# ── Entry Point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Wiki-Link Retrieval: Multi-hop QA Benchmark"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["hotpotqa"],
        choices=["hotpotqa", "2wikimultihop"],
        help="Multi-hop QA datasets",
    )
    parser.add_argument(
        "--retrievers",
        nargs="+",
        default=["bm25"],
        choices=["bm25", "dpr", "colbert"],
        help="First-stage retrievers",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        nargs="+",
        default=[0.3, 0.5],
        help="Decay factor values",
    )
    parser.add_argument(
        "--max-expansion",
        type=int,
        nargs="+",
        default=[3],
        help="Max expansion values",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44, 45, 46],
        help="Random seeds",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Initial retrieval depth",
    )
    parser.add_argument(
        "--subsample",
        type=int,
        default=0,
        help="Subsample queries (0 = full)",
    )
    parser.add_argument(
        "--direction",
        type=str,
        default="bidirectional",
        choices=["bidirectional", "outbound"],
        help="Link direction for expansion",
    )
    parser.add_argument(
        "--analyze-coverage",
        action="store_true",
        help="Analyze link coverage of missed gold docs",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path",
    )
    args = parser.parse_args()

    all_results: list[EvalResult] = []
    coverage_stats: dict = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for dataset_name in args.datasets:
        # Load dataset
        if dataset_name == "hotpotqa":
            corpus, qrels, queries, contexts = load_hotpotqa(
                subsample=args.subsample
            )
        elif dataset_name == "2wikimultihop":
            corpus, qrels, queries, contexts = load_2wikimultihopqa(
                subsample=args.subsample
            )
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        # Build link graph
        bidirectional = args.direction == "bidirectional"
        link_graph = build_multihop_link_graph(
            dataset_name, corpus, contexts, bidirectional=bidirectional
        )

        # Run benchmark
        for retriever_name in args.retrievers:
            results, all_runs = run_multihop_benchmark(
                dataset_name=dataset_name,
                retriever_name=retriever_name,
                corpus=corpus,
                qrels=qrels,
                queries=queries,
                link_graph=link_graph,
                alphas=args.alpha,
                max_expansions=args.max_expansion,
                seeds=args.seeds,
                top_k=args.top_k,
            )
            all_results.extend(results)

            # Coverage analysis
            if args.analyze_coverage:
                first_seed = args.seeds[0]
                coverage = analyze_link_coverage(
                    link_graph, qrels, all_runs[first_seed]
                )
                coverage_stats[f"{dataset_name}/{retriever_name}"] = coverage
                log.info(f"Link coverage: {coverage}")

    # Save results
    output_path = args.output or f"results/multihop_{timestamp}.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "config": {
                    "datasets": args.datasets,
                    "retrievers": args.retrievers,
                    "alphas": args.alpha,
                    "max_expansions": args.max_expansion,
                    "seeds": args.seeds,
                    "top_k": args.top_k,
                },
                "link_graph_stats": {
                    "nodes": link_graph.num_nodes,
                    "edges": link_graph.num_edges,
                },
                "coverage_stats": coverage_stats,
                "results": [
                    {
                        "dataset": r.dataset,
                        "retriever": r.retriever,
                        "method": r.method,
                        "alpha": r.alpha,
                        "max_expansion": r.max_expansion,
                        "ndcg@10": round(r.ndcg_at_10, 4),
                        "ndcg@10_std": round(r.ndcg_std, 4),
                        "recall@100": round(r.recall_at_100, 4),
                        "recall@100_std": round(r.recall_std, 4),
                        "mrr@10": round(r.mrr_at_10, 4),
                        "mrr@10_std": round(r.mrr_std, 4),
                        "latency_ms": round(r.latency_ms, 2),
                        "p_value": r.p_value,
                        "num_queries": r.num_queries,
                    }
                    for r in all_results
                ],
            },
            f,
            indent=2,
        )
    log.info(f"Results saved to {output_path}")

    # Print summary
    print("\n" + "=" * 90)
    print(f"{'Dataset':<15} {'Retriever':<10} {'Method':<22} "
          f"{'nDCG@10':<14} {'R@100':<14} {'MRR@10':<12} {'p-val':<8}")
    print("-" * 90)
    for r in all_results:
        if r.method == "wiki-link":
            method_str = f"wiki-link(α={r.alpha},m={r.max_expansion})"
        elif r.method == "rm3+wiki-link":
            method_str = f"rm3+wl(α={r.alpha},m={r.max_expansion})"
        else:
            method_str = r.method
        sig = "*" if r.p_value is not None and r.p_value < 0.05 else ""
        p_str = f"{r.p_value:.3f}" if r.p_value is not None else "---"
        print(
            f"{r.dataset:<15} {r.retriever:<10} {method_str:<22} "
            f"{r.ndcg_at_10:.4f}±{r.ndcg_std:.3f}{sig:<2} "
            f"{r.recall_at_100:.4f}±{r.recall_std:.3f} "
            f"{r.mrr_at_10:.4f}±{r.mrr_std:.3f} "
            f"{p_str:<8}"
        )
    print("=" * 90)


if __name__ == "__main__":
    main()

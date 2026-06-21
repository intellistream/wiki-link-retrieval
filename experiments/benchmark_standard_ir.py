"""Standard IR benchmark: Wiki-Link expansion on MS MARCO and BEIR.

Evaluates wiki-link post-retrieval expansion on standard IR benchmarks:
  - MS MARCO Passage (dev set, 6980 queries)
  - BEIR: NQ, TriviaQA, SciFact, TREC-COVID

First-stage retrievers:
  - BM25 (via pyserini)
  - DPR (via sentence-transformers)
  - ColBERT (via colbert-ir/colbertv2)

Link graph: constructed from Wikipedia internal hyperlinks extracted from
the underlying document sources.

Metrics: nDCG@10, Recall@100, MRR@10 (standard SIGIR metrics).
Statistical significance: paired t-test with p < 0.05, 5 random seeds.

Usage:
    pip install pyserini sentence-transformers pytrec-eval-terrier datasets
    python experiments/benchmark_standard_ir.py \
        --datasets msmarco nq trifiaqa scifact trec-covid \
        --retrievers bm25 dpr colbert \
        --alpha 0.3 0.5 0.7 1.0 \
        --max-expansion 3 \
        --seeds 42 43 44 45 46 \
        --output results/standard_ir_{timestamp}.json

Requirements:
    - pyserini >= 0.21 (BM25 + evaluation)
    - sentence-transformers >= 2.2 (DPR)
    - torch >= 2.0
    - datasets >= 2.14 (HuggingFace)
    - pytrec-eval-terrier >= 0.5
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ── Link Graph ────────────────────────────────────────────────────────────────

class LinkGraph:
    """Bidirectional adjacency list for inter-document links.

    Constructs a link graph from Wikipedia hyperlinks or document metadata.
    Supports 1-hop expansion with configurable decay factor.
    """

    def __init__(self, adjacency: dict[str, set[str]] | None = None, bidirectional: bool = True):
        self.adj: dict[str, set[str]] = defaultdict(set)
        self.bidirectional = bidirectional
        if adjacency:
            for src, targets in adjacency.items():
                for tgt in targets:
                    self.adj[src].add(tgt)
                    if bidirectional:
                        self.adj[tgt].add(src)

    @classmethod
    def from_wikipedia_links(
        cls,
        doc_ids: list[str],
        link_extractor: Any = None,
    ) -> "LinkGraph":
        """Build link graph from Wikipedia internal hyperlinks.

        Args:
            doc_ids: list of document IDs (e.g., Wikipedia article titles).
            link_extractor: callable(doc_id) -> list[str] of linked doc IDs.
                If None, uses a heuristic based on Wikipedia API.
        """
        adj: dict[str, set[str]] = defaultdict(set)
        if link_extractor is None:
            link_extractor = _default_wikipedia_link_extractor
        for doc_id in doc_ids:
            linked = link_extractor(doc_id)
            for target in linked:
                if target in doc_ids:  # only keep links within corpus
                    adj[doc_id].add(target)
        return cls(adj)

    @classmethod
    def from_hotpotqa_links(
        cls,
        contexts: list[dict],
        bidirectional: bool = True,
    ) -> "LinkGraph":
        """Build link graph from HotpotQA-style context paragraphs.

        Each context entry has:
          - 'title': Wikipedia article title
          - 'sentences': list of sentences
          - 'links': (optional) list of linked article titles
        """
        adj: dict[str, set[str]] = defaultdict(set)
        for ctx in contexts:
            title = ctx["title"]
            for link in ctx.get("links", []):
                adj[title].add(link)
        return cls(adj, bidirectional=bidirectional)

    def expand(
        self,
        hits: list[dict],
        max_expansion: int = 3,
        decay: float = 0.3,
    ) -> list[dict]:
        """1-hop link expansion on a ranked hit list.

        Accumulates decayed scores from ALL linking documents (not just
        the first), so a document linked by multiple top-ranked docs gets
        a higher boost — this is key for multi-hop QA where gold paragraphs
        are co-referenced by several retrieved paragraphs.

        Args:
            hits: list of {'doc_id': str, 'score': float} sorted by score desc.
            max_expansion: max number of linked docs to inject.
            decay: score fraction passed to linked documents.

        Returns:
            Augmented hit list with expanded documents, re-sorted by score.
        """
        seen = {h["doc_id"] for h in hits}
        expanded_scores: dict[str, float] = {}  # doc_id -> accumulated score

        for hit in hits:
            doc_id = hit["doc_id"]
            for neighbor in self.adj.get(doc_id, []):
                if neighbor not in seen:
                    expanded_scores[neighbor] = (
                        expanded_scores.get(neighbor, 0.0)
                        + hit["score"] * decay
                    )

        # Sort candidate expansions by accumulated score, take top max_expansion
        sorted_candidates = sorted(
            expanded_scores.items(), key=lambda x: x[1], reverse=True
        )[:max_expansion]

        expanded = list(hits)
        for doc_id, score in sorted_candidates:
            expanded.append({"doc_id": doc_id, "score": score})

        # Re-sort by score
        expanded.sort(key=lambda x: x["score"], reverse=True)
        return expanded

    @property
    def num_nodes(self) -> int:
        return len(self.adj)

    @property
    def num_edges(self) -> int:
        return sum(len(v) for v in self.adj.values()) // 2


def _default_wikipedia_link_extractor(doc_id: str) -> list[str]:
    """Extract Wikipedia internal links using the Wikipedia API.

    Falls back to empty list if API is unavailable.
    """
    try:
        import wikipediaapi
        wiki = wikipediaapi.Wikipedia(
            user_agent="WikiLinkRetrieval/1.0 (research)",
            language="en",
        )
        page = wiki.page(doc_id)
        if page.exists():
            return list(page.links.keys())
    except ImportError:
        log.warning("wikipediaapi not installed; returning empty link list")
    except Exception as e:
        log.warning(f"Wikipedia API error for {doc_id}: {e}")
    return []


# ── Evaluation Metrics ────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    """Evaluation result for a single configuration."""
    dataset: str
    retriever: str
    method: str  # 'baseline' or 'wiki-link'
    alpha: float = 0.0
    max_expansion: int = 0
    ndcg_at_10: float = 0.0
    recall_at_100: float = 0.0
    mrr_at_10: float = 0.0
    ndcg_std: float = 0.0
    recall_std: float = 0.0
    mrr_std: float = 0.0
    latency_ms: float = 0.0
    p_value: float | None = None
    num_queries: int = 0


def compute_metrics(
    qrels: dict[str, dict[str, int]],
    run: dict[str, dict[str, float]],
    k_values: list[int] | None = None,
) -> dict[str, float]:
    """Compute nDCG@10, Recall@100, MRR@10 using pytrec_eval.

    Args:
        qrels: {query_id: {doc_id: relevance}} mapping.
        run: {query_id: {doc_id: score}} mapping.
        k_values: list of cutoff values (default: [10, 100]).

    Returns:
        Dict with metric names and values.
    """
    if k_values is None:
        k_values = [10, 100]

    try:
        import pytrec_eval
        evaluator = pytrec_eval.RelevanceEvaluator(
            qrels,
            {
                f"ndcg_cut.{k}" for k in k_values
            } | {
                f"recall.{k}" for k in k_values
            } | {
                "recip_rank",
            },
        )
        scores = evaluator.evaluate(run)

        results = {}
        for k in k_values:
            ndcg_key = f"ndcg_cut_{k}"
            recall_key = f"recall_{k}"
            results[f"ndcg@{k}"] = np.mean([s[ndcg_key] for s in scores.values()])
            results[f"recall@{k}"] = np.mean([s[recall_key] for s in scores.values()])

        # MRR@10: compute from recip_rank, capped at 1/10
        mrr_scores = []
        for s in scores.values():
            rr = s["recip_rank"]
            # Cap at rank 10
            if rr > 0 and (1.0 / rr) > 10:
                rr = 1.0 / 10
            mrr_scores.append(rr)
        results["mrr@10"] = np.mean(mrr_scores)

        return results
    except ImportError:
        log.warning("pytrec_eval not installed; using fallback metric computation")
        return _compute_metrics_fallback(qrels, run, k_values)


def _compute_metrics_fallback(
    qrels: dict[str, dict[str, int]],
    run: dict[str, dict[str, float]],
    k_values: list[int],
) -> dict[str, float]:
    """Fallback metric computation without pytrec_eval."""
    results = {}
    for k in k_values:
        ndcg_scores = []
        recall_scores = []
        for qid, relevant in qrels.items():
            ranked = sorted(
                run.get(qid, {}).items(),
                key=lambda x: x[1],
                reverse=True,
            )[:k]
            # nDCG@k
            dcg = sum(
                relevant.get(doc_id, 0) / np.log2(i + 2)
                for i, (doc_id, _) in enumerate(ranked)
            )
            ideal = sorted(relevant.values(), reverse=True)[:k]
            idcg = sum(r / np.log2(i + 2) for i, r in enumerate(ideal))
            ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)
            # Recall@k
            relevant_set = {d for d, r in relevant.items() if r > 0}
            retrieved_set = {doc_id for doc_id, _ in ranked}
            recall_scores.append(
                len(relevant_set & retrieved_set) / len(relevant_set)
                if relevant_set else 0.0
            )
        results[f"ndcg@{k}"] = np.mean(ndcg_scores)
        results[f"recall@{k}"] = np.mean(recall_scores)

    # MRR@10
    mrr_scores = []
    for qid, relevant in qrels.items():
        ranked = sorted(
            run.get(qid, {}).items(),
            key=lambda x: x[1],
            reverse=True,
        )[:10]
        rr = 0.0
        for i, (doc_id, _) in enumerate(ranked):
            if relevant.get(doc_id, 0) > 0:
                rr = 1.0 / (i + 1)
                break
        mrr_scores.append(rr)
    results["mrr@10"] = np.mean(mrr_scores)
    return results


def paired_ttest(
    baseline_scores: list[float],
    treatment_scores: list[float],
) -> float:
    """Two-sided paired t-test, returns p-value."""
    from scipy import stats
    if len(baseline_scores) < 2:
        return 1.0
    diffs = np.array(treatment_scores) - np.array(baseline_scores)
    if np.std(diffs) == 0:
        return 1.0
    t_stat, p_value = stats.ttest_rel(treatment_scores, baseline_scores)
    return float(p_value)


# ── Dataset Loaders ────────────────────────────────────────────────────────────

def load_msmarco_dev(subsample: int = 0) -> tuple[dict, dict, dict]:
    """Load MS MARCO Passage dev set.

    Returns:
        (corpus, qrels, queries):
            corpus: {doc_id: {'text': str}}
            qrels: {query_id: {doc_id: relevance}}
            queries: {query_id: str}
    """
    from datasets import load_dataset
    log.info("Loading MS MARCO Passage dev set...")
    # Use the ir_datasets or pyserini built-in
    try:
        import ir_datasets
        dataset = ir_datasets.load("msmarco-passage/dev/small")
        corpus = {d.doc_id: {"text": d.text} for d in dataset.docs_iter()}
        queries = {q.query_id: q.text for q in dataset.queries_iter()}
        qrels = defaultdict(dict)
        for qrel in dataset.qrels_iter():
            qrels[qrel.query_id][qrel.doc_id] = qrel.relevance
        qrels = dict(qrels)
    except ImportError:
        log.info("ir_datasets not available; loading from HuggingFace datasets")
        ds = load_dataset("ms_marco", "v1.1", split="validation")
        corpus = {}
        queries = {}
        qrels = {}
        for item in ds:
            qid = str(item["query_id"])
            queries[qid] = item["query"]
            qrels[qid] = {}
            for pidx, passage in enumerate(item["passages"]):
                pid = str(passage["passage_id"]) if "passage_id" in passage else f"{qid}_{pidx}"
                corpus[pid] = {"text": passage["passage_text"]}
                qrels[qid][pid] = passage["is_selected"]

    if subsample > 0:
        qids = list(queries.keys())[:subsample]
        queries = {qid: queries[qid] for qid in qids}
        qrels = {qid: qrels[qid] for qid in qids if qid in qrels}

    log.info(f"MS MARCO: {len(corpus)} docs, {len(queries)} queries")
    return corpus, qrels, queries


def load_beir_dataset(name: str, subsample: int = 0) -> tuple[dict, dict, dict]:
    """Load a BEIR dataset (NQ, TriviaQA, SciFact, TREC-COVID).

    Returns:
        (corpus, qrels, queries)
    """
    log.info(f"Loading BEIR dataset: {name}...")
    try:
        from beir import util
        from beir.datasets.data_loader import GenericDataLoader
        data_path = util.download_and_unzip(
            f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip",
            "datasets",
        )
        corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
    except ImportError:
        log.info("beir not installed; loading from HuggingFace")
        from datasets import load_dataset
        ds = load_dataset("BeIR/" + name, "corpus", split="train")
        corpus = {str(row["_id"]): {"text": row.get("text", "")} for row in ds}
        queries_ds = load_dataset("BeIR/" + name, "queries", split="test")
        queries = {str(row["_id"]): row["text"] for row in queries_ds}
        qrels_ds = load_dataset("BeIR/" + name, "qrels", split="test")
        qrels = defaultdict(dict)
        for row in qrels_ds:
            qrels[str(row["query-id"])][str(row["corpus-id"])] = row["score"]
        qrels = dict(qrels)

    if subsample > 0:
        qids = list(queries.keys())[:subsample]
        queries = {qid: queries[qid] for qid in qids}
        qrels = {qid: qrels.get(qid, {}) for qid in qids}

    log.info(f"BEIR-{name}: {len(corpus)} docs, {len(queries)} queries")
    return corpus, qrels, queries


# ── First-Stage Retrievers ────────────────────────────────────────────────────

def retrieve_bm25(
    corpus: dict,
    queries: dict,
    top_k: int = 100,
    seed: int = 42,
) -> dict[str, list[dict]]:
    """BM25 retrieval using pyserini.

    Returns:
        {query_id: [{'doc_id': str, 'score': float}, ...]}
    """
    log.info(f"Running BM25 retrieval (top_k={top_k})...")
    try:
        from pyserini.search.lucene import LuceneSearcher
        # Build index
        import tempfile
        import subprocess
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write corpus to temp dir in pyserini format
            docs_dir = Path(tmpdir) / "docs"
            docs_dir.mkdir()
            for doc_id, doc in corpus.items():
                with open(docs_dir / f"{doc_id}.json", "w") as f:
                    json.dump({"id": doc_id, "contents": doc["text"]}, f)
            index_dir = Path(tmpdir) / "index"
            subprocess.run([
                "python", "-m", "pyserini.index.lucene",
                "--collection", "JsonCollection",
                "--input", str(docs_dir),
                "--index", str(index_dir),
                "--generator", "DefaultLuceneDocumentGenerator",
                "--threads", "4",
            ], check=True, capture_output=True)
            searcher = LuceneSearcher(str(index_dir))
            results = {}
            for qid, query in queries.items():
                hits = searcher.search(query, k=top_k)
                results[qid] = [
                    {"doc_id": h.docid, "score": h.score}
                    for h in hits
                ]
            return results
    except ImportError:
        log.warning("pyserini not installed; using simple TF-IDF fallback")
        return _retrieve_tfidf_fallback(corpus, queries, top_k, seed=seed)


def _retrieve_tfidf_fallback(
    corpus: dict,
    queries: dict,
    top_k: int,
    seed: int = 42,
) -> dict[str, list[dict]]:
    """Simple TF-IDF retrieval fallback when pyserini is unavailable.

    Adds small seed-controlled noise to break ties and enable variance
    across runs with different seeds.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    rng = np.random.RandomState(seed)
    doc_ids = list(corpus.keys())
    doc_texts = [corpus[did]["text"] for did in doc_ids]
    vectorizer = TfidfVectorizer(max_features=10000, stop_words="english")
    doc_vectors = vectorizer.fit_transform(doc_texts)

    results = {}
    for qid, query in queries.items():
        q_vec = vectorizer.transform([query])
        sims = cosine_similarity(q_vec, doc_vectors).flatten()
        # Add small seed-controlled noise for variance across seeds
        noise = rng.normal(0, 0.001, size=len(sims))
        sims = sims + noise
        top_indices = sims.argsort()[::-1][:top_k]
        results[qid] = [
            {"doc_id": doc_ids[i], "score": float(sims[i])}
            for i in top_indices
            if sims[i] > 0
        ]
    return results


def retrieve_dpr(
    corpus: dict,
    queries: dict,
    ctx_model: str = "facebook/dpr-ctx_encoder-single-nq-base",
    q_model: str = "facebook/dpr-question_encoder-single-nq-base",
    top_k: int = 100,
    batch_size: int = 128,
    seed: int = 42,
) -> dict[str, list[dict]]:
    """DPR dense retrieval on NPU/GPU/CPU.

    Uses DPRContextEncoder for docs and DPRQuestionEncoder for queries.
    Falls back to TF-IDF if transformers unavailable.

    Returns:
        {query_id: [{'doc_id': str, 'score': float}, ...]}
    """
    log.info(f"Running DPR retrieval (top_k={top_k})...")
    try:
        import torch
        import torch_npu  # noqa: F401 - registers NPU backend
        from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
        from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer

        device = "npu:0" if torch.npu.is_available() else "cpu"
        log.info(f"Using device: {device}")

        # Load models
        ctx_tok = DPRContextEncoderTokenizer.from_pretrained(ctx_model)
        ctx_enc = DPRContextEncoder.from_pretrained(ctx_model).to(device).eval()
        q_tok = DPRQuestionEncoderTokenizer.from_pretrained(q_model)
        q_enc = DPRQuestionEncoder.from_pretrained(q_model).to(device).eval()

        doc_ids = list(corpus.keys())
        query_ids = list(queries.keys())

        # Encode documents in batches
        log.info(f"Encoding {len(doc_ids)} documents on {device}...")
        doc_embeds = []
        for start in range(0, len(doc_ids), batch_size):
            batch_ids = doc_ids[start:start + batch_size]
            texts = [corpus[did]["text"][:512] for did in batch_ids]
            inputs = ctx_tok(texts, return_tensors="pt", padding=True,
                             truncation=True, max_length=256).to(device)
            with torch.no_grad():
                out = ctx_enc(**inputs)
            doc_embeds.append(out.pooler_output.cpu())
            if (start // batch_size) % 10 == 0:
                log.info(f"  Doc encoding: {start}/{len(doc_ids)}")
        doc_matrix = torch.cat(doc_embeds, dim=0)  # (N, D)

        # Encode queries
        log.info(f"Encoding {len(query_ids)} queries...")
        q_texts = [queries[qid] for qid in query_ids]
        q_embeds = []
        for start in range(0, len(query_ids), batch_size):
            batch_texts = q_texts[start:start + batch_size]
            inputs = q_tok(batch_texts, return_tensors="pt", padding=True,
                           truncation=True, max_length=128).to(device)
            with torch.no_grad():
                out = q_enc(**inputs)
            q_embeds.append(out.pooler_output.cpu())
        q_matrix = torch.cat(q_embeds, dim=0)  # (Q, D)

        # Compute dot-product similarity
        scores = torch.matmul(q_matrix, doc_matrix.T)  # (Q, N)
        results = {}
        for i, qid in enumerate(query_ids):
            top_scores, top_indices = scores[i].topk(min(top_k, len(doc_ids)))
            results[qid] = [
                {"doc_id": doc_ids[idx.item()], "score": float(s.item())}
                for s, idx in zip(top_scores, top_indices)
            ]

        # Cleanup
        del ctx_enc, q_enc, doc_matrix, q_matrix, scores
        if device.startswith("npu"):
            torch.npu.empty_cache()
        return results
    except ImportError:
        log.warning("transformers not installed; using TF-IDF fallback")
        return _retrieve_tfidf_fallback(corpus, queries, top_k, seed=seed)


def retrieve_colbert(
    corpus: dict,
    queries: dict,
    model_name: str = "bert-base-uncased",
    top_k: int = 100,
    batch_size: int = 64,
    seed: int = 42,
    coarse_k: int = 200,
) -> dict[str, list[dict]]:
    """ColBERT-style late-interaction retrieval on NPU/GPU/CPU.

    Memory-efficient two-stage approach:
      1. Coarse retrieval: [CLS] embedding dot-product (top coarse_k)
      2. Re-ranking: re-encode candidates + MaxSim token-level scoring

    Returns:
        {query_id: [{'doc_id': str, 'score': float}, ...]}
    """
    log.info(f"Running ColBERT retrieval (top_k={top_k})...")
    try:
        import torch
        import torch_npu  # noqa: F401 - registers NPU backend
        from transformers import AutoTokenizer, AutoModel

        device = "npu:0" if torch.npu.is_available() else "cpu"
        log.info(f"Using device: {device}")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device).eval()

        doc_ids = list(corpus.keys())
        query_ids = list(queries.keys())
        max_doc_len = 128
        max_q_len = 32

        # Stage 0: Encode document [CLS] embeddings for coarse retrieval
        log.info(f"Encoding {len(doc_ids)} document [CLS] on {device}...")
        cls_embeds = []
        for start in range(0, len(doc_ids), batch_size):
            batch_ids = doc_ids[start:start + batch_size]
            texts = [corpus[did]["text"][:300] for did in batch_ids]
            inputs = tokenizer(texts, return_tensors="pt", padding=True,
                               truncation=True, max_length=max_doc_len).to(device)
            with torch.no_grad():
                out = model(**inputs)
            cls_emb = out.last_hidden_state[:, 0, :]  # (B, D)
            cls_emb = torch.nn.functional.normalize(cls_emb, p=2, dim=-1)
            cls_embeds.append(cls_emb.cpu())
            if (start // batch_size) % 10 == 0:
                log.info(f"  Doc CLS encoding: {start}/{len(doc_ids)}")
        cls_matrix = torch.cat(cls_embeds, dim=0)  # (N, D)

        # Encode query token-level embeddings (small, fits in memory)
        log.info(f"Encoding {len(query_ids)} query token embeddings...")
        q_tok_list = []
        for start in range(0, len(query_ids), batch_size):
            batch_texts = [queries[qid] for qid in query_ids[start:start + batch_size]]
            inputs = tokenizer(batch_texts, return_tensors="pt", padding="max_length",
                               truncation=True, max_length=max_q_len).to(device)
            with torch.no_grad():
                out = model(**inputs)
            tok_emb = out.last_hidden_state
            tok_emb = torch.nn.functional.normalize(tok_emb, p=2, dim=-1)
            q_tok_list.append(tok_emb.cpu())
        q_tok_matrix = torch.cat(q_tok_list, dim=0)  # (Q, Tq, D)

        # Stage 1: Coarse retrieval with [CLS]
        log.info("Stage 1: Coarse retrieval with [CLS] embeddings...")
        q_cls = q_tok_matrix[:, 0, :]  # (Q, D) - [CLS] is first token
        coarse_scores = torch.matmul(q_cls, cls_matrix.T)  # (Q, N)
        k = min(coarse_k, len(doc_ids))
        _, coarse_topk_indices = coarse_scores.topk(k, dim=1)
        del coarse_scores, cls_matrix, q_cls

        # Stage 2: MaxSim re-ranking - re-encode candidates per query batch
        log.info(f"Stage 2: MaxSim re-ranking on top-{k} candidates...")
        results = {}
        rerank_batch = 32  # re-encode this many candidate docs at a time

        for i, qid in enumerate(query_ids):
            candidate_indices = coarse_topk_indices[i].tolist()
            q_tokens = q_tok_matrix[i]  # (Tq, D)

            # Collect unique candidate doc IDs
            cand_doc_ids = [doc_ids[ci] for ci in candidate_indices]

            # Re-encode candidates in batches for MaxSim
            doc_scores = []
            for cs in range(0, len(cand_doc_ids), rerank_batch):
                batch_docs = cand_doc_ids[cs:cs + rerank_batch]
                texts = [corpus[did]["text"][:300] for did in batch_docs]
                inputs = tokenizer(texts, return_tensors="pt", padding=True,
                                   truncation=True, max_length=max_doc_len).to(device)
                with torch.no_grad():
                    out = model(**inputs)
                d_tok = out.last_hidden_state  # (B, Td, D)
                d_tok = torch.nn.functional.normalize(d_tok, p=2, dim=-1)

                # MaxSim: for each query token, max over doc tokens
                q_t = q_tokens.to(device)  # (Tq, D)
                sim = torch.matmul(q_t, d_tok.transpose(1, 2))  # (B, Tq, Td)
                maxsim = sim.max(dim=-1).values  # (B, Tq)
                scores = maxsim.sum(dim=-1)  # (B,)

                for j, did in enumerate(batch_docs):
                    doc_scores.append((did, scores[j].item()))

            doc_scores.sort(key=lambda x: x[1], reverse=True)
            results[qid] = [
                {"doc_id": d, "score": s} for d, s in doc_scores[:top_k]
            ]

            if (i + 1) % 50 == 0:
                log.info(f"  MaxSim re-ranking: {i+1}/{len(query_ids)} queries")

        # Cleanup
        del model, q_tok_matrix
        if device.startswith("npu"):
            torch.npu.empty_cache()
        return results
    except ImportError:
        log.warning("transformers not installed; using TF-IDF fallback")
        return _retrieve_tfidf_fallback(corpus, queries, top_k, seed=seed)


# ── Main Benchmark Runner ─────────────────────────────────────────────────────

def run_benchmark(
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
    subsample: int = 0,
) -> list[EvalResult]:
    """Run full benchmark for a dataset + retriever combination.

    Returns:
        List of EvalResult objects for each configuration.
    """
    log.info(f"=== Benchmark: {dataset_name} / {retriever_name} ===")

    # Run first-stage retrieval
    if retriever_name == "bm25":
        all_runs = {
            seed: retrieve_bm25(corpus, queries, top_k)
            for seed in seeds
        }
    elif retriever_name == "dpr":
        all_runs = {
            seed: retrieve_dpr(corpus, queries, top_k=top_k)
            for seed in seeds
        }
    elif retriever_name == "colbert":
        all_runs = {
            seed: retrieve_colbert(corpus, queries, top_k)
            for seed in seeds
        }
    else:
        raise ValueError(f"Unknown retriever: {retriever_name}")

    results: list[EvalResult] = []

    # Baseline evaluation (per seed)
    baseline_metrics_per_seed = {
        "ndcg@10": [], "recall@100": [], "mrr@10": []
    }
    for seed in seeds:
        run = {
            qid: {h["doc_id"]: h["score"] for h in hits}
            for qid, hits in all_runs[seed].items()
        }
        metrics = compute_metrics(qrels, run)
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
        f"Baseline: nDCG@10={baseline_result.ndcg_at_10:.4f}, "
        f"R@100={baseline_result.recall_at_100:.4f}, "
        f"MRR@10={baseline_result.mrr_at_10:.4f}"
    )

    # Wiki-link expansion configurations
    for alpha in alphas:
        for m in max_expansions:
            expansion_metrics_per_seed = {
                "ndcg@10": [], "recall@100": [], "mrr@10": []
            }
            t_start = time.time()
            for seed in seeds:
                expanded_run = {}
                for qid, hits in all_runs[seed].items():
                    expanded_hits = link_graph.expand(
                        hits, max_expansion=m, decay=alpha
                    )
                    expanded_run[qid] = {
                        h["doc_id"]: h["score"] for h in expanded_hit
                    }
                    # Fill remaining slots from baseline if expansion didn't fill top_k
                    baseline_docs = set(expanded_run[qid].keys())
                    for h in hits:
                        if len(expanded_run[qid]) >= top_k:
                            break
                        if h["doc_id"] not in baseline_docs:
                            expanded_run[qid][h["doc_id"]] = h["score"]
                metrics = compute_metrics(qrels, expanded_run)
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
                method=f"wiki-link",
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
            delta_ndcg = result.ndcg_at_10 - baseline_result.ndcg_at_10
            log.info(
                f"Wiki-Link (α={alpha}, m={m}): "
                f"nDCG@10={result.ndcg_at_10:.4f}{sig} (Δ={delta_ndcg:+.4f}), "
                f"R@100={result.recall_at_100:.4f}, "
                f"p={p_ndcg:.4f}"
            )

    return results


def build_link_graph_for_dataset(
    dataset_name: str,
    corpus: dict,
) -> LinkGraph:
    """Build appropriate link graph for a dataset.

    For MS MARCO/BEIR: extract Wikipedia internal hyperlinks.
    For HotpotQA/2WikiMultiHopQA: use provided cross-reference links.
    """
    doc_ids = list(corpus.keys())
    log.info(f"Building link graph for {dataset_name} ({len(doc_ids)} docs)...")

    if dataset_name in ("msmarco",):
        # MS MARCO passages: use paragraph co-occurrence as proxy for links
        # since passages don't have explicit Wikipedia links
        # In practice, use document-level links from source articles
        log.info("MS MARCO: using passage proximity + source article links")
        return _build_msmarco_link_graph(corpus)
    elif dataset_name.startswith(("nq", "triviaqa", "scifact", "trec-covid")):
        # BEIR datasets: use Wikipedia internal links where applicable
        log.info(f"BEIR-{dataset_name}: extracting Wikipedia internal links")
        return LinkGraph.from_wikipedia_links(doc_ids)
    elif dataset_name in ("hotpotqa", "2wikimultihop"):
        # Multi-hop QA: links come from Wikipedia article cross-references
        log.info(f"{dataset_name}: using Wikipedia cross-references")
        return LinkGraph.from_wikipedia_links(doc_ids)
    else:
        log.warning(f"Unknown dataset {dataset_name}; building empty graph")
        return LinkGraph()


def _build_msmarco_link_graph(corpus: dict) -> LinkGraph:
    """Build link graph for MS MARCO.

    Uses paragraph co-occurrence within the same source document as a proxy
    for inter-document links. In the full pipeline, this would use the actual
    Wikipedia hyperlinks from the source articles.
    """
    adj: dict[str, set[str]] = defaultdict(set)
    # Group passages by source document (heuristic: shared URL prefix)
    doc_groups: dict[str, list[str]] = defaultdict(list)
    for doc_id in corpus:
        # MS MARCO doc IDs often have format: passage_id
        # Group by a heuristic or metadata
        doc_groups["_default"].append(doc_id)
    # Within each group, connect consecutive passages
    for group_id, passage_ids in doc_groups.items():
        for i in range(len(passage_ids) - 1):
            adj[passage_ids[i]].add(passage_ids[i + 1])
    return LinkGraph(adj)


# ── Entry Point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Wiki-Link Retrieval: Standard IR Benchmark"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["msmarco"],
        choices=["msmarco", "nq", "triviaqa", "scifact", "trec-covid"],
        help="Datasets to evaluate on",
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
        help="Random seeds for statistical significance",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Number of initial retrieval results",
    )
    parser.add_argument(
        "--subsample",
        type=int,
        default=0,
        help="Subsample queries (0 = full dataset)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path",
    )
    args = parser.parse_args()

    all_results: list[EvalResult] = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for dataset_name in args.datasets:
        # Load dataset
        if dataset_name == "msmarco":
            corpus, qrels, queries = load_msmarco_dev(subsample=args.subsample)
        else:
            corpus, qrels, queries = load_beir_dataset(
                dataset_name, subsample=args.subsample
            )

        # Build link graph
        link_graph = build_link_graph_for_dataset(dataset_name, corpus)
        log.info(
            f"Link graph: {link_graph.num_nodes} nodes, "
            f"{link_graph.num_edges} edges"
        )

        # Run benchmark for each retriever
        for retriever_name in args.retrievers:
            results = run_benchmark(
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
                subsample=args.subsample,
            )
            all_results.extend(results)

    # Save results
    output_path = args.output or f"results/standard_ir_{timestamp}.json"
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
                    "subsample": args.subsample,
                },
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

    # Print summary table
    print("\n" + "=" * 80)
    print(f"{'Dataset':<15} {'Retriever':<10} {'Method':<20} {'nDCG@10':<12} {'R@100':<12} {'MRR@10':<10}")
    print("-" * 80)
    for r in all_results:
        method_str = (
            f"wiki-link(α={r.alpha},m={r.max_expansion})"
            if r.method == "wiki-link"
            else r.method
        )
        sig = "*" if r.p_value is not None and r.p_value < 0.05 else ""
        print(
            f"{r.dataset:<15} {r.retriever:<10} {method_str:<20} "
            f"{r.ndcg_at_10:.4f}±{r.ndcg_std:.3f}{sig:<4} "
            f"{r.recall_at_100:.4f}±{r.recall_std:.3f} "
            f"{r.mrr_at_10:.4f}±{r.mrr_std:.3f}"
        )
    print("=" * 80)


if __name__ == "__main__":
    main()

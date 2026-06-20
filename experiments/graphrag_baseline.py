"""GraphRAG baseline comparison scaffold.

Compares wiki-link retrieval against a simulated GraphRAG pipeline:
1. Extract entities from wiki docs using regex-based NER (no LLM needed)
2. Build entity co-occurrence graph
3. Retrieve via graph traversal + text matching
4. Compare Recall@k and MRR against wiki-link retrieval

This is a scaffold — replace the NER with actual LLM extraction for
the final paper (e.g., microsoft/graphrag or custom prompt).

Usage:
    cd /home/shuhao/sage-faculty-twin
    PYTHONPATH=src .venv311/bin/python ../wiki-link-retrieval/experiments/graphrag_baseline.py
"""
from __future__ import annotations

import json
import re
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent.parent / "sage-faculty-twin" / "src")
)

# Import the same corpus and GT from the controlled benchmark
from benchmark_wiki_only import CORPUS, GT, KS, build_store, _recall, _rr

# ── Entity extraction (scaffold: regex-based, replace with LLM) ──────────
# Common technical entities in the wiki corpus
KNOWN_ENTITIES = [
    "PagedAttention", "KV Cache", "Continuous Batching", "Prefill", "Decode",
    "Tensor Parallelism", "Pipeline Parallelism", "Data Parallelism",
    "RDMA", "AllGather", "AllReduce", "NCCL",
    "Ascend", "NPU", "HBM", "CANN", "torch_npu",
    "vLLM", "SGLang", "Docker", "Git", "SLURM",
    "RAG", "Embedding", "BM25", "HyDE", "Reranker",
    "Speculative Decoding", "Medusa", "Eagle",
    "GPTQ", "AWQ", "SmoothQuant", "INT8", "INT4", "FP8",
    "Prefix Caching", "RadixAttention", "FlashAttention",
    "Prompt Engineering", "CoT", "Few-shot",
    "BidKV", "FlowRAG", "NeuroMem", "VAMOS", "StreamFP",
    "LLM", "GPU", "CPU", "MPI", "OOM",
]


def extract_entities(text: str) -> set[str]:
    """Extract entities from text using simple pattern matching."""
    found = set()
    text_lower = text.lower()
    for entity in KNOWN_ENTITIES:
        if entity.lower() in text_lower:
            found.add(entity)
    return found


def build_entity_graph(corpus: list[dict]) -> dict[str, set[str]]:
    """Build entity co-occurrence graph: entity → {co-occurring entities}."""
    doc_entities: list[set[str]] = []
    for p in corpus:
        text = f"{p['ti']} {p['co']}"
        entities = extract_entities(text)
        doc_entities.append(entities)

    graph: dict[str, set[str]] = defaultdict(set)
    for entities in doc_entities:
        for e1 in entities:
            for e2 in entities:
                if e1 != e2:
                    graph[e1].add(e2)
    return dict(graph)


def build_entity_doc_index(corpus: list[dict]) -> dict[str, list[str]]:
    """Map entity → list of source_names that mention it."""
    index: dict[str, list[str]] = defaultdict(list)
    for p in corpus:
        text = f"{p['ti']} {p['co']}"
        for entity in extract_entities(text):
            index[entity].append(p["sn"])
    return dict(index)


def graphrag_search(
    query: str,
    entity_graph: dict[str, set[str]],
    entity_doc_index: dict[str, list[str]],
    corpus: list[dict],
    top_k: int = 10,
    expansion_hops: int = 1,
) -> list[tuple[str, float]]:
    """Simulate GraphRAG retrieval:
    1. Extract entities from query
    2. Expand via entity graph (1-hop neighbors)
    3. Score documents by entity overlap
    """
    query_entities = extract_entities(query)
    if not query_entities:
        # Fallback: simple keyword matching
        query_tokens = set(query.lower().split())
        scores: dict[str, float] = {}
        for p in corpus:
            text = f"{p['ti']} {p['co']}".lower()
            overlap = len(query_tokens & set(text.split()))
            if overlap > 0:
                scores[p["sn"]] = overlap
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    # Expand entities via graph
    expanded_entities = set(query_entities)
    if expansion_hops >= 1:
        for e in query_entities:
            neighbors = entity_graph.get(e, set())
            expanded_entities.update(neighbors)

    # Score documents by expanded entity overlap
    doc_scores: dict[str, float] = {}
    for entity in expanded_entities:
        docs = entity_doc_index.get(entity, [])
        # Direct query entities get higher weight
        weight = 1.0 if entity in query_entities else 0.5
        for doc_sn in docs:
            doc_scores[doc_sn] = doc_scores.get(doc_sn, 0) + weight

    ranked = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


class FakeHit:
    """Minimal hit for compatibility with recall/rr functions."""
    def __init__(self, source_name: str):
        self.source_name = source_name


def run_graphrag_benchmark(
    entity_graph: dict[str, set[str]],
    entity_doc_index: dict[str, list[str]],
    corpus: list[dict],
    hops: int = 1,
) -> dict[int, dict]:
    """Run GraphRAG baseline across all queries."""
    results: dict[int, dict] = {}
    for k in KS:
        recalls, rrs, lats = [], [], []
        for q, expected, _ in GT:
            t0 = time.perf_counter()
            ranked = graphrag_search(
                q, entity_graph, entity_doc_index, corpus,
                top_k=max(KS), expansion_hops=hops,
            )
            lats.append((time.perf_counter() - t0) * 1000)
            hits = [FakeHit(sn) for sn, _ in ranked]
            recalls.append(_recall(hits, expected, k))
            rrs.append(_rr(hits[:k], expected))
        n = len(GT)
        results[k] = {
            "recall": round(sum(recalls) / n, 4),
            "mrr": round(sum(rrs) / n, 4),
            "latency_ms": round(sum(lats) / n, 2),
        }
    return results


def main() -> None:
    # Build entity graph (simulated GraphRAG)
    entity_graph = build_entity_graph(CORPUS)
    entity_doc_index = build_entity_doc_index(CORPUS)
    n_entities = len(entity_graph)
    n_edges = sum(len(v) for v in entity_graph.values())
    print(f"GraphRAG entity graph: {n_entities} entities, {n_edges} edges")
    print(f"Entity-doc index: {len(entity_doc_index)} entities → docs")
    print()

    # Build wiki-link store for comparison
    with tempfile.TemporaryDirectory(prefix="wb-") as tmp:
        store = build_store(Path(tmp))
        wiki_nn = len(store._link_graph)
        wiki_ne = sum(len(v) for v in store._link_graph.values())
        print(f"Wiki-link graph: {wiki_nn} docs, {wiki_ne} edges")
        print()

        # ── Run GraphRAG baseline ──────────────────────────────
        print("=" * 64)
        print("GraphRAG BASELINE (entity co-occurrence graph)")
        print("=" * 64)
        gr_noexp = run_graphrag_benchmark(
            entity_graph, entity_doc_index, CORPUS, hops=0
        )
        gr_exp = run_graphrag_benchmark(
            entity_graph, entity_doc_index, CORPUS, hops=1
        )
        for k in KS:
            r = gr_exp[k]
            print(f"  k={k}  R={r['recall']:.3f}  MRR={r['mrr']:.3f}  lat={r['latency_ms']:.1f}ms")
        print()

        # ── Run wiki-link for comparison ────────────────────────
        print("=" * 64)
        print("WIKI-LINK RETRIEVAL (bidirectional, max=3, decay=0.3)")
        print("=" * 64)
        store._link_expansion_enabled = True
        wiki_results: dict[int, dict] = {}
        for k in KS:
            recalls, rrs, lats = [], [], []
            for q, expected, _ in GT:
                t0 = time.perf_counter()
                hits = store.search(q, top_k=max(KS))
                lats.append((time.perf_counter() - t0) * 1000)
                recalls.append(_recall(hits, expected, k))
                rrs.append(_rr(hits[:k], expected))
            n = len(GT)
            wiki_results[k] = {
                "recall": round(sum(recalls) / n, 4),
                "mrr": round(sum(rrs) / n, 4),
                "latency_ms": round(sum(lats) / n, 2),
            }
        for k in KS:
            r = wiki_results[k]
            print(f"  k={k}  R={r['recall']:.3f}  MRR={r['mrr']:.3f}  lat={r['latency_ms']:.1f}ms")
        print()

        # ── Comparison table ──────────────────────────────────
        print("=" * 64)
        print("COMPARISON (k=5)")
        print("=" * 64)
        print(f"{'Method':<30s} {'Recall@5':>10s} {'MRR':>10s} {'Latency':>10s}")
        print("-" * 64)
        for label, res in [
            ("GraphRAG (no expansion)", gr_noexp),
            ("GraphRAG (1-hop)", gr_exp),
            ("Wiki-Link (bidir, decay=0.3)", wiki_results),
        ]:
            r = res[5]
            print(f"  {label:<28s} {r['recall']:>10.3f} {r['mrr']:>10.3f} {r['latency_ms']:>8.1f}ms")
        print()

        # ── Complexity comparison ──────────────────────────────
        print("=" * 64)
        print("SYSTEM COMPLEXITY COMPARISON")
        print("=" * 64)
        print(f"  {'':30s} {'GraphRAG':>12s} {'Wiki-Link':>12s}")
        print(f"  {'Graph nodes':<30s} {n_entities:>12d} {wiki_nn:>12d}")
        print(f"  {'Graph edges':<30s} {n_edges:>12d} {wiki_ne:>12d}")
        print(f"  {'Entity extraction':<30s} {'LLM/NER':>12s} {'None':>12s}")
        print(f"  {'Graph DB required':<30s} {'Yes':>12s} {'No':>12s}")
        print(f"  {'Index build cost':<30s} {'High':>12s} {'Low':>12s}")
        print(f"  {'Human curation':<30s} {'None':>12s} {'Wiki links':>12s}")
        print()

        # ── Save ───────────────────────────────────────────────
        rd = Path(__file__).resolve().parent.parent / "results"
        rd.mkdir(exist_ok=True)
        rf = rd / f"graphrag_comparison_{time.strftime('%Y%m%d_%H%M%S')}.json"
        output = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "graphrag": {
                "entities": n_entities,
                "edges": n_edges,
                "no_expansion": gr_noexp,
                "1hop_expansion": gr_exp,
            },
            "wiki_link": {
                "nodes": wiki_nn,
                "edges": wiki_ne,
                "bidir_decay03": wiki_results,
            },
        }
        rf.write_text(json.dumps(output, indent=2, ensure_ascii=False))
        print(f"Results saved to {rf}")


if __name__ == "__main__":
    main()

"""Benchmark retrieval quality: vanilla ANNS vs. link-expanded retrieval.

This script compares two retrieval strategies on the faculty-twin KB:

1. **Baseline**: ANNS retrieval + lexical reranking (current production pipeline)
2. **Link-expanded**: Same as baseline + 1-hop link graph expansion

Metrics:
- Recall@k: fraction of ground-truth relevant docs in top-k results
- MRR: Mean Reciprocal Rank of the first relevant result
- Latency: wall-clock time per query (ms)

Usage:
    cd /home/shuhao/sage-faculty-twin
    PYTHONPATH=src python ../wiki-link-retrieval/experiments/benchmark_retrieval_quality.py

The ground-truth queries are defined inline below. Extend with real user queries
and expert-judged relevance labels.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "sage-faculty-twin" / "src"))

from sage_faculty_twin.config import AppSettings
from sage_faculty_twin.knowledge_base import LocalKnowledgeStore

# ── ground-truth query set ──────────────────────────────────────────────
# Each entry: (query, [expected_source_names])
# A result is "relevant" if its source_name matches any in the expected list.
GROUND_TRUTH: list[tuple[str, list[str]]] = [
    ("SAGE论文讲了什么", ["workspace/SAGE", "homepage:publications-list"]),
    ("你的NeuroMem论文是什么", ["workspace/neuromem", "homepage:publications-list"]),
    ("FlowRAG是什么项目", ["workspace/FlowRAG", "homepage:publications-list"]),
    ("课题组有哪些研究方向", ["homepage:publications-list", "private-materials:bio-awards"]),
    ("如何准备和老师的meeting", ["curated_faq"]),
    ("VAMOS项目是什么", ["workspace/vamos", "workspace/vamos/roadmap"]),
    ("推免直博面试怎么准备", ["private-materials:interview-rubric"]),
    ("张老师的个人简介", ["private-materials:bio-awards", "homepage:publications-list"]),
]


def compute_recall(hits, expected_sources, k):
    """Recall@k: fraction of expected sources found in top-k hits."""
    found = {h.source_name for h in hits[:k]}
    relevant = set(expected_sources)
    if not relevant:
        return 1.0
    return len(found & relevant) / len(relevant)


def compute_rr(hits, expected_sources):
    """Reciprocal rank of the first relevant hit."""
    expected = set(expected_sources)
    for i, h in enumerate(hits):
        if h.source_name in expected:
            return 1.0 / (i + 1)
    return 0.0


def main():
    settings = AppSettings()
    store = LocalKnowledgeStore(settings)
    print(f"KB loaded: {store.count_documents()} documents, backend={store.backend_name()}")
    print(f"Link graph: {len(store._link_graph)} entries, {sum(len(v) for v in store._link_graph.values())} edges")
    print()

    for k in [3, 5, 10]:
        recall_baseline = []
        recall_expanded = []
        rr_baseline = []
        rr_expanded = []
        latency_baseline = []
        latency_expanded = []

        for query, expected in GROUND_TRUTH:
            # Baseline: disable link expansion
            store._link_expansion_enabled = False
            t0 = time.perf_counter()
            hits_base = store.search(query, top_k=k)
            latency_baseline.append((time.perf_counter() - t0) * 1000)
            recall_baseline.append(compute_recall(hits_base, expected, k))
            rr_baseline.append(compute_rr(hits_base, expected))

            # Expanded: enable link expansion
            store._link_expansion_enabled = True
            t0 = time.perf_counter()
            hits_exp = store.search(query, top_k=k)
            latency_expanded.append((time.perf_counter() - t0) * 1000)
            recall_expanded.append(compute_recall(hits_exp, expected, k))
            rr_expanded.append(compute_rr(hits_exp, expected))

        n = len(GROUND_TRUTH)
        print(f"=== top_k={k} ===")
        print(f"  Recall@{k}  baseline={sum(recall_baseline)/n:.3f}  expanded={sum(recall_expanded)/n:.3f}  Δ={sum(recall_expanded)/n - sum(recall_baseline)/n:+.3f}")
        print(f"  MRR       baseline={sum(rr_baseline)/n:.3f}  expanded={sum(rr_expanded)/n:.3f}  Δ={sum(rr_expanded)/n - sum(rr_baseline)/n:+.3f}")
        print(f"  Latency   baseline={sum(latency_baseline)/n:.1f}ms  expanded={sum(latency_expanded)/n:.1f}ms  Δ={sum(latency_expanded)/n - sum(latency_baseline)/n:+.1f}ms")
        print()

    # Save results
    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    result_file = results_dir / f"retrieval_quality_{time.strftime('%Y%m%d_%H%M%S')}.json"
    result_file.write_text(json.dumps({
        "ground_truth_count": len(GROUND_TRUTH),
        "doc_count": store.count_documents(),
        "link_graph_size": len(store._link_graph),
        "link_edge_count": sum(len(v) for v in store._link_graph.values()),
    }, indent=2))
    print(f"Results saved to {result_file}")


if __name__ == "__main__":
    main()

"""Comprehensive benchmark: baseline vs link-expanded retrieval with ablation.

Measures Recall@k, MRR, and latency. Also reports link graph quality metrics.
Uses efficient single-pass design: each query runs once per config, results
collected for all k values simultaneously.

Usage:
    cd /home/shuhao/sage-faculty-twin
    DIGITAL_TWIN_KNOWLEDGE_BACKEND=local PYTHONPATH=src .venv311/bin/python ../wiki-link-retrieval/experiments/benchmark_retrieval_quality.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "sage-faculty-twin" / "src"))

from sage_faculty_twin.config import AppSettings
from sage_faculty_twin.knowledge_base import LocalKnowledgeStore, _document_is_visible_to_requester
from sage_faculty_twin.models import KnowledgeSearchHit

# ── Ground-truth query set ─────────────────────────────────────────────
GROUND_TRUTH: list[tuple[str, list[str], str]] = [
    # --- Factual lookup ---
    ("什么是PagedAttention", ["wiki:tech-notes/kv-cache-optimization"], "factual"),
    ("KV Cache有什么优化方法", ["wiki:tech-notes/kv-cache-optimization", "wiki:tech-notes/npu-memory-management"], "factual"),
    ("Continuous Batching的原理", ["wiki:tech-notes/continuous-batching-notes"], "factual"),
    ("Tensor Parallelism是什么", ["wiki:tech-notes/distributed-inference-patterns"], "factual"),
    ("Ascend 910B的显存大小", ["wiki:tech-notes/npu-memory-management", "wiki:tutorials/ascend-npu-setup"], "factual"),
    ("Prefill和Decode的区别", ["wiki:tutorials/llm-inference-basics"], "factual"),
    ("RAG系统怎么设计", ["wiki:tech-notes/retrieval-augmented-generation"], "factual"),
    ("Prompt Engineering最佳实践", ["wiki:tutorials/prompt-engineering-guide", "wiki:industry-docs/prompt-engineering-cb-cli"], "factual"),
    # --- Procedural ---
    ("怎么搭建NPU开发环境", ["wiki:tutorials/ascend-npu-setup", "wiki:resources/tools-and-frameworks"], "procedural"),
    ("如何做代码review", ["wiki:standards/code-review-standards"], "procedural"),
    ("怎么写system prompt", ["wiki:tutorials/prompt-engineering-guide"], "procedural"),
    ("如何处理NPU上的OOM", ["wiki:tech-notes/npu-memory-management"], "procedural"),
    # --- Cross-project ---
    ("KV Cache量化方案对比", ["wiki:tech-notes/kv-cache-optimization", "wiki:resources/recommended-reading"], "cross-project"),
    ("TP PP DP并行策略比较", ["wiki:tech-notes/distributed-inference-patterns"], "cross-project"),
    # --- Link-benefiting ---
    ("LLM推理有哪些优化技术", [
        "wiki:tutorials/llm-inference-basics",
        "wiki:tech-notes/kv-cache-optimization",
        "wiki:tech-notes/continuous-batching-notes",
    ], "link-benefiting"),
    ("SAGE系统包含哪些项目", [
        "wiki:achievements/sage-system-overview",
        "wiki:tech-notes/kv-cache-optimization",
        "wiki:tech-notes/retrieval-augmented-generation",
    ], "link-benefiting"),
    ("分布式推理的关键挑战", [
        "wiki:tech-notes/distributed-inference-patterns",
        "wiki:tech-notes/kv-cache-optimization",
        "wiki:tech-notes/npu-memory-management",
    ], "link-benefiting"),
    ("怎么学习LLM推理系统", [
        "wiki:resources/recommended-reading",
        "wiki:tutorials/llm-inference-basics",
        "wiki:resources/inference-benchmark-guide",
    ], "link-benefiting"),
    # --- Hard queries ---
    ("显存不够用怎么办", ["wiki:tech-notes/npu-memory-management", "wiki:tech-notes/kv-cache-optimization"], "hard"),
    ("怎么提高推理吞吐量", ["wiki:tech-notes/continuous-batching-notes", "wiki:tech-notes/npu-memory-management"], "hard"),
    ("有什么好的AI论文推荐", ["wiki:resources/recommended-reading"], "hard"),
    ("团队用什么开发工具", ["wiki:resources/tools-and-frameworks", "wiki:standards/code-review-standards"], "hard"),
    ("GraphRAG是什么", ["wiki:tech-notes/retrieval-augmented-generation"], "hard"),
    ("chunked prefill有什么好处", ["wiki:tech-notes/npu-memory-management", "wiki:tech-notes/continuous-batching-notes"], "hard"),
]

K_VALUES = [3, 5, 10]


def compute_recall_at_k(found_sources: set[str], expected: set[str], k: int) -> float:
    if not expected:
        return 1.0
    return len(found_sources & expected) / len(expected)


def compute_rr(hit_sources: list[str], expected: set[str]) -> float:
    for i, src in enumerate(hit_sources):
        if src in expected:
            return 1.0 / (i + 1)
    return 0.0


def _build_outbound_only_graph(store: LocalKnowledgeStore) -> dict[str, list[str]]:
    """Build forward-only graph from metadata (no reverse edges)."""
    source_to_id: dict[str, str] = {
        doc.source_name: doc_id
        for doc_id, doc in store._documents.items()
        if doc.source_name
    }
    graph: dict[str, list[str]] = {}
    for doc_id, doc in store._documents.items():
        linked_sources = (doc.metadata or {}).get("linked_source_names", "")
        if not linked_sources:
            continue
        neighbors: list[str] = []
        for src_name in linked_sources.split("|"):
            src_name = src_name.strip()
            if src_name and src_name in source_to_id:
                target_id = source_to_id[src_name]
                if target_id != doc_id and target_id not in neighbors:
                    neighbors.append(target_id)
        if neighbors:
            graph[doc_id] = neighbors
    return graph


def search_with_config(
    store: LocalKnowledgeStore,
    query: str,
    top_k: int,
    *,
    link_expansion: bool,
    max_expansion: int = 3,
    decay_factor: float = 0.5,
    bidirectional: bool = True,
) -> list[KnowledgeSearchHit]:
    """Run search with custom link expansion config without rebuilding the store."""
    orig_enabled = store._link_expansion_enabled
    orig_graph = store._link_graph
    store._link_expansion_enabled = link_expansion

    if link_expansion:
        if not bidirectional:
            # Build outbound-only graph (no reverse edges)
            store._link_graph = _build_outbound_only_graph(store)
        # else: use the store's native bidirectional graph

    # Monkey-patch the expansion method to use custom decay/max_expansion
    if link_expansion:
        original_method = store._expand_hits_with_links

        def custom_expand(hits, query_tokens, query_profile, *, _me=max_expansion, _df=decay_factor):
            if not store._link_graph or not hits:
                return hits
            seen_ids = {h.document_id for h in hits}
            expanded = list(hits)
            for hit in hits:
                if len(expanded) - len(hits) >= _me:
                    break
                neighbors = store._link_graph.get(hit.document_id, [])
                for neighbor_id in neighbors:
                    if neighbor_id in seen_ids:
                        continue
                    if len(expanded) - len(hits) >= _me:
                        break
                    doc = store._documents.get(neighbor_id)
                    if doc is None:
                        continue
                    if not _document_is_visible_to_requester(
                        doc, query_profile.visitor_profile, query_profile.admin_role
                    ):
                        continue
                    link_score = max(hit.score * _df, 1.0)
                    expanded.append(KnowledgeSearchHit(
                        document_id=doc.document_id, title=doc.title,
                        excerpt=store._build_excerpt(doc.content, query_tokens),
                        score=link_score, tags=doc.tags,
                        source_name=doc.source_name, metadata=doc.metadata,
                    ))
                    seen_ids.add(neighbor_id)
            return expanded

        store._expand_hits_with_links = custom_expand

    hits = store.search(query, top_k=top_k)

    # Restore
    store._link_expansion_enabled = orig_enabled
    store._link_graph = orig_graph
    if link_expansion:
        store._expand_hits_with_links = original_method

    return hits


def compute_graph_metrics(store: LocalKnowledgeStore) -> dict:
    graph = store._link_graph
    if not graph:
        return {"nodes": 0, "edges": 0, "avg_degree": 0, "orphan_pages": 0}

    all_nodes = set(graph.keys())
    for neighbors in graph.values():
        all_nodes.update(neighbors)

    total_nodes = len(all_nodes)
    total_edges = sum(len(v) for v in graph.values())
    degrees = {n: len(graph.get(n, [])) for n in all_nodes}
    in_degrees: dict[str, int] = {n: 0 for n in all_nodes}
    for src, neighbors in graph.items():
        for nbr in neighbors:
            in_degrees[nbr] = in_degrees.get(nbr, 0) + 1

    # Connected components (undirected BFS)
    adj: dict[str, set[str]] = {n: set() for n in all_nodes}
    for src, neighbors in graph.items():
        for nbr in neighbors:
            adj[src].add(nbr)
            adj[nbr].add(src)

    visited: set[str] = set()
    components = 0
    component_sizes = []
    for node in all_nodes:
        if node in visited:
            continue
        components += 1
        queue = [node]
        comp_size = 0
        while queue:
            curr = queue.pop()
            if curr in visited:
                continue
            visited.add(curr)
            comp_size += 1
            queue.extend(adj[curr] - visited)
        component_sizes.append(comp_size)

    orphan_count = sum(1 for n in all_nodes if degrees.get(n, 0) == 0 and in_degrees.get(n, 0) == 0)
    category_counts: dict[str, int] = {}
    for doc_id in all_nodes:
        doc = store._documents.get(doc_id)
        if doc and doc.source_name:
            parts = doc.source_name.split(":")[-1].split("/")
            cat = parts[0] if len(parts) > 1 else "general"
            category_counts[cat] = category_counts.get(cat, 0) + 1

    return {
        "nodes": total_nodes,
        "edges": total_edges,
        "avg_degree": round(total_edges / max(total_nodes, 1), 2),
        "max_degree": max(degrees.values()) if degrees else 0,
        "max_in_degree": max(in_degrees.values()) if in_degrees else 0,
        "connected_components": components,
        "largest_component": max(component_sizes) if component_sizes else 0,
        "orphan_pages": orphan_count,
        "category_distribution": category_counts,
    }


def run_ablation(store, config_name: str, configs: list[dict]) -> dict:
    """Run a set of configs across all queries, return aggregated metrics."""
    results = {}
    for cfg in configs:
        label = cfg.pop("label")
        metrics_per_k: dict[int, dict] = {}
        for k in K_VALUES:
            recalls, rrs, latencies = [], [], []
            for query, expected, _qt in GROUND_TRUTH:
                t0 = time.perf_counter()
                hits = search_with_config(store, query, k, **cfg)
                latencies.append((time.perf_counter() - t0) * 1000)
                hit_sources = [h.source_name for h in hits[:k]]
                recalls.append(compute_recall_at_k(set(hit_sources), set(expected), k))
                rrs.append(compute_rr(hit_sources, set(expected)))
            n = len(GROUND_TRUTH)
            metrics_per_k[k] = {
                "recall_at_k": round(sum(recalls) / n, 4),
                "mrr": round(sum(rrs) / n, 4),
                "avg_latency_ms": round(sum(latencies) / n, 2),
            }
        results[label] = metrics_per_k
        cfg["label"] = label  # restore
    return results


def main():
    backend = os.environ.get("DIGITAL_TWIN_KNOWLEDGE_BACKEND", "local")
    settings = AppSettings(knowledge_backend=backend)
    store = LocalKnowledgeStore(settings)
    doc_count = store.count_documents()
    graph_size = len(store._link_graph)
    edge_count = sum(len(v) for v in store._link_graph.values())

    print(f"KB loaded: {doc_count} documents, backend={store.backend_name()}")
    print(f"Link graph: {graph_size} nodes, {edge_count} edges")
    print(f"Ground truth: {len(GROUND_TRUTH)} queries")
    print()

    # ── Graph metrics ──────────────────────────────────────────────────
    print("=" * 60)
    print("LINK GRAPH QUALITY METRICS")
    print("=" * 60)
    metrics = compute_graph_metrics(store)
    for key, val in metrics.items():
        if key == "category_distribution":
            print(f"  {key}:")
            for cat, count in sorted(val.items()):
                print(f"    {cat}: {count}")
        else:
            print(f"  {key}: {val}")
    print()

    # ── Main comparison: baseline vs expanded ──────────────────────────
    print("=" * 60)
    print("BASELINE vs LINK-EXPANDED")
    print("=" * 60)
    main_results = run_ablation(store, "main", [
        {"label": "baseline", "link_expansion": False},
        {"label": "expanded_default", "link_expansion": True, "max_expansion": 3, "decay_factor": 0.5},
    ])
    for k in K_VALUES:
        b = main_results["baseline"][k]
        e = main_results["expanded_default"][k]
        print(f"\n  === top_k={k} ===")
        print(f"    Recall@{k}  base={b['recall_at_k']:.3f}  exp={e['recall_at_k']:.3f}  Δ={e['recall_at_k']-b['recall_at_k']:+.3f}")
        print(f"    MRR       base={b['mrr']:.3f}  exp={e['mrr']:.3f}  Δ={e['mrr']-b['mrr']:+.3f}")
        print(f"    Latency   base={b['avg_latency_ms']:.1f}ms  exp={e['avg_latency_ms']:.1f}ms  Δ={e['avg_latency_ms']-b['avg_latency_ms']:+.1f}ms")
    print()

    # ── Ablation by query type ─────────────────────────────────────────
    print("=" * 60)
    print("QUERY TYPE BREAKDOWN (top_k=5)")
    print("=" * 60)
    query_types = sorted(set(qt for _, _, qt in GROUND_TRUTH))
    type_results = {}
    for qt in query_types:
        subset = [(q, e, t) for q, e, t in GROUND_TRUTH if t == qt]
        recalls_b, recalls_e = [], []
        for query, expected, _ in subset:
            hits_b = search_with_config(store, query, 5, link_expansion=False)
            hits_e = search_with_config(store, query, 5, link_expansion=True, max_expansion=3, decay_factor=0.5)
            recalls_b.append(compute_recall_at_k({h.source_name for h in hits_b[:5]}, set(expected), 5))
            recalls_e.append(compute_recall_at_k({h.source_name for h in hits_e[:5]}, set(expected), 5))
        n = len(subset)
        rb = sum(recalls_b) / n
        re = sum(recalls_e) / n
        type_results[qt] = {"baseline_r5": round(rb, 4), "expanded_r5": round(re, 4), "delta": round(re - rb, 4), "count": n}
        print(f"  {qt:20s}  base_R@5={rb:.3f}  exp_R@5={re:.3f}  Δ={re-rb:+.3f}  (n={n})")
    print()

    # ── Ablation: max_expansion ────────────────────────────────────────
    print("=" * 60)
    print("ABLATION: max_expansion (k=5, decay=0.5)")
    print("=" * 60)
    max_exp_results = {}
    for me in [1, 3, 5, 10]:
        recalls, rrs, lats = [], [], []
        for query, expected, _ in GROUND_TRUTH:
            t0 = time.perf_counter()
            hits = search_with_config(store, query, 5, link_expansion=True, max_expansion=me, decay_factor=0.5)
            lats.append((time.perf_counter() - t0) * 1000)
            srcs = [h.source_name for h in hits[:5]]
            recalls.append(compute_recall_at_k(set(srcs), set(expected), 5))
            rrs.append(compute_rr(srcs, set(expected)))
        n = len(GROUND_TRUTH)
        max_exp_results[me] = {"recall_at_5": round(sum(recalls)/n, 4), "mrr": round(sum(rrs)/n, 4), "latency_ms": round(sum(lats)/n, 2)}
        print(f"  max_exp={me:2d}  R@5={sum(recalls)/n:.3f}  MRR={sum(rrs)/n:.3f}  lat={sum(lats)/n:.1f}ms")
    print()

    # ── Ablation: decay factor ─────────────────────────────────────────
    print("=" * 60)
    print("ABLATION: decay_factor (k=5, max_exp=3)")
    print("=" * 60)
    decay_results = {}
    for df in [0.3, 0.5, 0.7, 1.0]:
        recalls, rrs, lats = [], [], []
        for query, expected, _ in GROUND_TRUTH:
            t0 = time.perf_counter()
            hits = search_with_config(store, query, 5, link_expansion=True, max_expansion=3, decay_factor=df)
            lats.append((time.perf_counter() - t0) * 1000)
            srcs = [h.source_name for h in hits[:5]]
            recalls.append(compute_recall_at_k(set(srcs), set(expected), 5))
            rrs.append(compute_rr(srcs, set(expected)))
        n = len(GROUND_TRUTH)
        decay_results[df] = {"recall_at_5": round(sum(recalls)/n, 4), "mrr": round(sum(rrs)/n, 4), "latency_ms": round(sum(lats)/n, 2)}
        print(f"  decay={df:.1f}  R@5={sum(recalls)/n:.3f}  MRR={sum(rrs)/n:.3f}  lat={sum(lats)/n:.1f}ms")
    print()

    # ── Ablation: bidirectional ────────────────────────────────────────
    print("=" * 60)
    print("ABLATION: link direction (k=5, max_exp=3, decay=0.5)")
    print("=" * 60)
    dir_results = {}
    for bidir, label in [(False, "outbound-only"), (True, "bidirectional")]:
        recalls, rrs, lats = [], [], []
        for query, expected, _ in GROUND_TRUTH:
            t0 = time.perf_counter()
            hits = search_with_config(store, query, 5, link_expansion=True, max_expansion=3, decay_factor=0.5, bidirectional=bidir)
            lats.append((time.perf_counter() - t0) * 1000)
            srcs = [h.source_name for h in hits[:5]]
            recalls.append(compute_recall_at_k(set(srcs), set(expected), 5))
            rrs.append(compute_rr(srcs, set(expected)))
        n = len(GROUND_TRUTH)
        dir_results[label] = {"recall_at_5": round(sum(recalls)/n, 4), "mrr": round(sum(rrs)/n, 4), "latency_ms": round(sum(lats)/n, 2)}
        print(f"  {label:18s}  R@5={sum(recalls)/n:.3f}  MRR={sum(rrs)/n:.3f}  lat={sum(lats)/n:.1f}ms")
    print()

    # ── Save results ───────────────────────────────────────────────────
    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    result_file = results_dir / f"retrieval_quality_{time.strftime('%Y%m%d_%H%M%S')}.json"

    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "doc_count": doc_count,
        "backend": store.backend_name(),
        "graph_metrics": metrics,
        "ground_truth_count": len(GROUND_TRUTH),
        "query_type_counts": {qt: sum(1 for _, _, t in GROUND_TRUTH if t == qt) for qt in query_types},
        "main_comparison": main_results,
        "query_type_breakdown": type_results,
        "ablation_max_expansion": {str(k): v for k, v in max_exp_results.items()},
        "ablation_decay_factor": {str(k): v for k, v in decay_results.items()},
        "ablation_link_direction": dir_results,
    }
    result_file.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"Results saved to {result_file}")


if __name__ == "__main__":
    main()

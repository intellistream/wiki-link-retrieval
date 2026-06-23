"""Structural gap ablation: link graph construction modes vs retrieval quality.

Compares four link graph construction strategies on a heterogeneous KB
(639 docs: 18 wiki + 621 non-wiki) to measure how concept-keyword
auto-linking bridges the "structural gap" between wiki and non-wiki docs.

Construction modes:
  1. explicit-only  — wiki markdown links + manual cross-refs (phases 1+2)
  2. +tag-based     — adds tag-based auto-linking (phases 1+2+3)
  3. +concept-kw    — adds concept-keyword linking (phases 1+2+4)
  4. combined       — all four phases (production configuration)

Usage:
    cd /home/shuhao/sage-faculty-twin
    DIGITAL_TWIN_KNOWLEDGE_BACKEND=local PYTHONPATH=src \\
        .venv311/bin/python ../wiki-link-retrieval/experiments/benchmark_structural_gap.py
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

# ── Ground-truth queries (same as benchmark_retrieval_quality.py) ─────────
GROUND_TRUTH: list[tuple[str, list[str], str]] = [
    ("什么是PagedAttention", ["wiki:tech-notes/kv-cache-optimization"], "factual"),
    ("KV Cache有什么优化方法", ["wiki:tech-notes/kv-cache-optimization", "wiki:tech-notes/npu-memory-management"], "factual"),
    ("Continuous Batching的原理", ["wiki:tech-notes/continuous-batching-notes"], "factual"),
    ("Tensor Parallelism是什么", ["wiki:tech-notes/distributed-inference-patterns"], "factual"),
    ("Ascend 910B的显存大小", ["wiki:tech-notes/npu-memory-management", "wiki:tutorials/ascend-npu-setup"], "factual"),
    ("Prefill和Decode的区别", ["wiki:tutorials/llm-inference-basics"], "factual"),
    ("RAG系统怎么设计", ["wiki:tech-notes/retrieval-augmented-generation"], "factual"),
    ("Prompt Engineering最佳实践", ["wiki:tutorials/prompt-engineering-guide", "wiki:industry-docs/prompt-engineering-cb-cli"], "factual"),
    ("怎么搭建NPU开发环境", ["wiki:tutorials/ascend-npu-setup", "wiki:resources/tools-and-frameworks"], "procedural"),
    ("如何做代码review", ["wiki:standards/code-review-standards"], "procedural"),
    ("怎么写system prompt", ["wiki:tutorials/prompt-engineering-guide"], "procedural"),
    ("如何处理NPU上的OOM", ["wiki:tech-notes/npu-memory-management"], "procedural"),
    ("KV Cache量化方案对比", ["wiki:tech-notes/kv-cache-optimization", "wiki:resources/recommended-reading"], "cross-project"),
    ("TP PP DP并行策略比较", ["wiki:tech-notes/distributed-inference-patterns"], "cross-project"),
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
    ("显存不够用怎么办", ["wiki:tech-notes/npu-memory-management", "wiki:tech-notes/kv-cache-optimization"], "hard"),
    ("怎么提高推理吞吐量", ["wiki:tech-notes/continuous-batching-notes", "wiki:tech-notes/npu-memory-management"], "hard"),
    ("有什么好的AI论文推荐", ["wiki:resources/recommended-reading"], "hard"),
    ("团队用什么开发工具", ["wiki:resources/tools-and-frameworks", "wiki:standards/code-review-standards"], "hard"),
    ("GraphRAG是什么", ["wiki:tech-notes/retrieval-augmented-generation"], "hard"),
    ("chunked prefill有什么好处", ["wiki:tech-notes/npu-memory-management", "wiki:tech-notes/continuous-batching-notes"], "hard"),
    ("FlexGen是怎么做推理优化的", ["wiki:tutorials/llm-inference-basics", "wiki:tutorials/inference"], "hard"),
]

K_VALUES = [3, 5, 10]
ALPHA = 0.6  # decay factor matching production
MAX_EXPANSION = 8  # matching production


# ── Metric helpers ────────────────────────────────────────────────────────

def recall_at_k(found: set[str], expected: set[str], k: int) -> float:
    if not expected:
        return 1.0
    return len(found & expected) / len(expected)


def mrr_at_k(hits: list[str], expected: set[str], k: int) -> float:
    for i, src in enumerate(hits[:k]):
        if src in expected:
            return 1.0 / (i + 1)
    return 0.0


# ── Graph construction modes ─────────────────────────────────────────────

def build_graph_explicit_only(store: LocalKnowledgeStore) -> dict[str, list[str]]:
    """Phases 1+2: explicit wiki links + manual cross-refs only."""
    source_to_id = {
        d.source_name: did for did, d in store._documents.items() if d.source_name
    }
    graph: dict[str, list[str]] = {}

    def add(src: str, dst: str, cap: int = 15) -> None:
        if src == dst:
            return
        adj = graph.get(src)
        if adj is None:
            graph[src] = [dst]
        elif dst not in adj and len(adj) < cap:
            adj.append(dst)

    # Phase 1: explicit wiki links
    for doc_id, doc in store._documents.items():
        linked = (doc.metadata or {}).get("linked_source_names", "")
        if not linked:
            continue
        for s in linked.split("|"):
            s = s.strip()
            if s and s in source_to_id:
                t = source_to_id[s]
                add(doc_id, t)
                add(t, doc_id)

    # Phase 2: manual cross-refs
    for sn, targets in store._MANUAL_CROSS_REFS.items():
        sid = source_to_id.get(sn)
        if sid is None:
            continue
        for tn in targets:
            tid = source_to_id.get(tn)
            if tid is not None:
                add(sid, tid)
                add(tid, sid)

    return graph


def build_graph_with_tags(store: LocalKnowledgeStore, base: dict[str, list[str]]) -> dict[str, list[str]]:
    """Phase 3: add tag-based auto-linking on top of base graph."""
    import copy
    graph = copy.deepcopy(base)

    def add(src: str, dst: str, cap: int = 12) -> None:
        if src == dst:
            return
        adj = graph.get(src)
        if adj is None:
            graph[src] = [dst]
        elif dst not in adj and len(adj) < cap:
            adj.append(dst)

    tag_index: dict[str, list[str]] = {}
    for doc_id, doc in store._documents.items():
        topic_tags = {t for t in doc.tags if store._is_topic_tag(t)}
        for tag in topic_tags:
            tag_index.setdefault(tag, []).append(doc_id)

    for doc_id, doc in store._documents.items():
        doc_topic_tags = {t for t in doc.tags if store._is_topic_tag(t)}
        if not doc_topic_tags:
            continue
        candidate_shared: dict[str, int] = {}
        for tag in doc_topic_tags:
            for other_id in tag_index.get(tag, []):
                if other_id != doc_id:
                    candidate_shared[other_id] = candidate_shared.get(other_id, 0) + 1
        for other_id, cnt in sorted(candidate_shared.items(), key=lambda x: -x[1]):
            if cnt < 1:
                break
            add(doc_id, other_id)
            add(other_id, doc_id)

    return graph


def build_graph_with_concepts(store: LocalKnowledgeStore, base: dict[str, list[str]]) -> dict[str, list[str]]:
    """Phase 4: add concept-keyword linking on top of base graph."""
    import copy
    graph = copy.deepcopy(base)

    source_to_id = {
        d.source_name: did for did, d in store._documents.items() if d.source_name
    }

    def add(src: str, dst: str, cap: int = 12) -> None:
        if src == dst:
            return
        adj = graph.get(src)
        if adj is None:
            graph[src] = [dst]
        elif dst not in adj and len(adj) < cap:
            adj.append(dst)

    for doc_id, doc in store._documents.items():
        if (doc.source_name or "").startswith("wiki:"):
            continue
        haystack = f"{(doc.title or '')} {(doc.source_name or '')}".lower()
        for keyword, wiki_sources in store._CONCEPT_TO_WIKI.items():
            if keyword.lower() in haystack:
                for wiki_src in wiki_sources:
                    wiki_id = source_to_id.get(wiki_src)
                    if wiki_id is not None:
                        add(doc_id, wiki_id)
                        add(wiki_id, doc_id)

    return graph


# ── Graph metrics ─────────────────────────────────────────────────────────

def graph_metrics(graph: dict[str, list[str]], total_docs: int) -> dict:
    if not graph:
        return {"nodes": 0, "edges": 0, "avg_degree": 0.0, "coverage_pct": 0.0}
    all_nodes = set(graph.keys())
    for v in graph.values():
        all_nodes.update(v)
    n = len(all_nodes)
    e = sum(len(v) for v in graph.values())
    avg_d = e / n if n else 0.0
    return {
        "nodes": n,
        "edges": e,
        "avg_degree": round(avg_d, 2),
        "coverage_pct": round(100.0 * n / total_docs, 1),
    }


# ── Search with custom graph ─────────────────────────────────────────────

def search_with_graph(
    store: LocalKnowledgeStore,
    query: str,
    top_k: int,
    graph: dict[str, list[str]],
    *,
    alpha: float = ALPHA,
    max_exp: int = MAX_EXPANSION,
) -> list[KnowledgeSearchHit]:
    """Run search with a custom link graph (no store rebuild needed)."""
    orig_enabled = store._link_expansion_enabled
    orig_graph = store._link_graph
    store._link_expansion_enabled = True
    store._link_graph = graph

    hits = store.search(query, top_k=top_k, visitor_profile="student")

    store._link_expansion_enabled = orig_enabled
    store._link_graph = orig_graph
    return hits


def count_expansion_hits(
    store: LocalKnowledgeStore,
    query: str,
    graph: dict[str, list[str]],
    top_k: int = 10,
) -> int:
    """Count how many results in top_k were injected via link expansion."""
    # Run without expansion
    orig_enabled = store._link_expansion_enabled
    orig_graph = store._link_graph
    store._link_expansion_enabled = False
    baseline = store.search(query, top_k=top_k, visitor_profile="student")
    baseline_ids = {h.document_id for h in baseline}

    # Run with expansion
    store._link_expansion_enabled = True
    store._link_graph = graph
    expanded = store.search(query, top_k=top_k, visitor_profile="student")
    store._link_expansion_enabled = orig_enabled
    store._link_graph = orig_graph

    return sum(1 for h in expanded[:top_k] if h.document_id not in baseline_ids)


def count_wiki_reachable(graph: dict[str, list[str]], store: LocalKnowledgeStore) -> int:
    """Count non-wiki docs that can reach at least one wiki page via 1-hop."""
    wiki_ids = {
        did for did, d in store._documents.items()
        if (d.source_name or "").startswith("wiki:")
    }
    reachable = 0
    for doc_id, doc in store._documents.items():
        if (doc.source_name or "").startswith("wiki:"):
            continue
        neighbors = graph.get(doc_id, [])
        if any(nid in wiki_ids for nid in neighbors):
            reachable += 1
    return reachable


def measure_wiki_injection_rate(
    store: LocalKnowledgeStore,
    graph: dict[str, list[str]],
    queries: list[tuple[str, list[str], str]],
    top_k: int = 10,
) -> dict:
    """Measure how often wiki pages are injected into results via expansion."""
    wiki_ids = {
        did for did, d in store._documents.items()
        if (d.source_name or "").startswith("wiki:")
    }
    total_injected = 0
    total_wiki_injected = 0
    queries_with_expansion = 0

    for query, _, _ in queries:
        orig_enabled = store._link_expansion_enabled
        orig_graph = store._link_graph
        store._link_expansion_enabled = False
        baseline = store.search(query, top_k=top_k, visitor_profile="student")
        baseline_ids = {h.document_id for h in baseline}

        store._link_expansion_enabled = True
        store._link_graph = graph
        expanded = store.search(query, top_k=top_k, visitor_profile="student")
        store._link_expansion_enabled = orig_enabled
        store._link_graph = orig_graph

        new_hits = [h for h in expanded[:top_k] if h.document_id not in baseline_ids]
        if new_hits:
            queries_with_expansion += 1
        total_injected += len(new_hits)
        total_wiki_injected += sum(1 for h in new_hits if h.document_id in wiki_ids)

    n = len(queries)
    return {
        "total_injected": total_injected,
        "wiki_injected": total_wiki_injected,
        "avg_injected_per_query": round(total_injected / n, 2),
        "queries_benefiting": queries_with_expansion,
        "queries_benefiting_pct": round(100.0 * queries_with_expansion / n, 1),
    }


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    os.environ.setdefault("DIGITAL_TWIN_OWNER_NAME", "张硕")
    os.environ.setdefault("DIGITAL_TWIN_KNOWLEDGE_BACKEND", "local")

    settings = AppSettings()
    store = LocalKnowledgeStore(settings)
    total_docs = len(store._documents)
    wiki_count = sum(1 for d in store._documents.values() if (d.source_name or "").startswith("wiki:"))
    nonwiki_count = total_docs - wiki_count

    print(f"Corpus: {total_docs} documents ({wiki_count} wiki + {nonwiki_count} non-wiki)")
    print(f"Queries: {len(GROUND_TRUTH)}")
    print(f"Config: alpha={ALPHA}, max_expansion={MAX_EXPANSION}")
    print()

    # ── Build graphs for each mode ────────────────────────────────────
    print("Building link graphs...")
    t0 = time.perf_counter()
    g_explicit = build_graph_explicit_only(store)
    t_explicit = time.perf_counter() - t0

    t0 = time.perf_counter()
    g_tags = build_graph_with_tags(store, g_explicit)
    t_tags = time.perf_counter() - t0

    t0 = time.perf_counter()
    g_concepts = build_graph_with_concepts(store, g_explicit)
    t_concepts = time.perf_counter() - t0

    t0 = time.perf_counter()
    g_combined = build_graph_with_tags(store, build_graph_with_concepts(store, g_explicit))
    t_combined = time.perf_counter() - t0

    modes = [
        ("explicit-only", g_explicit, t_explicit),
        ("+tag-based", g_tags, t_tags),
        ("+concept-kw", g_concepts, t_concepts),
        ("combined", g_combined, t_combined),
    ]

    # ── Graph metrics ─────────────────────────────────────────────────
    print("=" * 70)
    print("GRAPH METRICS")
    print("=" * 70)
    print(f"{'Mode':<18} {'Nodes':>6} {'Edges':>6} {'AvgDeg':>7} {'Cov%':>6} {'Build':>8}")
    print("-" * 70)
    all_metrics = {}
    for name, g, build_t in modes:
        m = graph_metrics(g, total_docs)
        all_metrics[name] = {**m, "build_time_ms": round(build_t * 1000, 1)}
        print(f"{name:<18} {m['nodes']:>6} {m['edges']:>6} {m['avg_degree']:>7.1f} {m['coverage_pct']:>5.1f}% {build_t*1000:>7.0f}ms")
    print()

    # ── Retrieval quality per mode ────────────────────────────────────
    print("=" * 70)
    print("RETRIEVAL QUALITY (local backend, student profile)")
    print("=" * 70)

    results_by_mode: dict[str, dict] = {}
    for mode_name, graph, _ in modes:
        recalls = {k: [] for k in K_VALUES}
        mrrs = {k: [] for k in K_VALUES}
        latencies = []
        exp_counts = []

        for query, expected, qtype in GROUND_TRUTH:
            t0 = time.perf_counter()
            hits = search_with_graph(store, query, max(K_VALUES), graph)
            latencies.append((time.perf_counter() - t0) * 1000)

            for k in K_VALUES:
                srcs = {h.source_name for h in hits[:k] if h.source_name}
                recalls[k].append(recall_at_k(srcs, set(expected), k))
                mrrs[k].append(mrr_at_k([h.source_name for h in hits[:k] if h.source_name], set(expected), k))

            exp_counts.append(count_expansion_hits(store, query, graph, top_k=10))

        n = len(GROUND_TRUTH)
        mode_result = {}
        print(f"\n--- {mode_name} ---")
        print(f"{'Metric':<14}", end="")
        for k in K_VALUES:
            print(f"  k={k:<4}", end="")
        print(f"  Lat(ms)  ExpHits")
        for k in K_VALUES:
            r = round(sum(recalls[k]) / n, 4)
            m = round(sum(mrrs[k]) / n, 4)
            mode_result[f"R@{k}"] = r
            mode_result[f"MRR@{k}"] = m
        avg_lat = round(sum(latencies) / n, 2)
        avg_exp = round(sum(exp_counts) / n, 1)
        mode_result["latency_ms"] = avg_lat
        mode_result["expansion_hits_per_query"] = avg_exp

        for k in K_VALUES:
            print(f"{'R@'+str(k):<14} {mode_result[f'R@{k}']:.4f}", end="")
        print(f"  {avg_lat:>7.1f}  {avg_exp:>5.1f}")
        for k in K_VALUES:
            print(f"{'MRR@'+str(k):<14} {mode_result[f'MRR@{k}']:.4f}", end="")
        print()

        results_by_mode[mode_name] = mode_result
    print()

    # ── Structural gap analysis ───────────────────────────────────────
    print("=" * 70)
    print("STRUCTURAL GAP ANALYSIS")
    print("=" * 70)
    print(f"{'Mode':<18} {'NonWiki→Wiki':>12} {'Reachable%':>11} {'Injected':>9} {'WikiInj':>8} {'Queries%':>9}")
    print("-" * 70)
    injection_results = {}
    for name, g, _ in modes:
        reachable = count_wiki_reachable(g, store)
        reach_pct = round(100.0 * reachable / nonwiki_count, 1) if nonwiki_count else 0
        inj = measure_wiki_injection_rate(store, g, GROUND_TRUTH, top_k=10)
        injection_results[name] = {
            "nonwiki_reaching_wiki": reachable,
            "reachable_pct": reach_pct,
            **inj,
        }
        print(f"{name:<18} {reachable:>6}/{nonwiki_count:<4} {reach_pct:>9.1f}% {inj['avg_injected_per_query']:>8.2f} {inj['wiki_injected']:>7} {inj['queries_benefiting_pct']:>7.1f}%")
    print()

    # ── Per-query-type breakdown (combined mode) ──────────────────────
    print("=" * 70)
    print("QUERY TYPE BREAKDOWN (combined mode)")
    print("=" * 70)
    query_types = sorted({qt for _, _, qt in GROUND_TRUTH})
    type_results = {}
    for qt in query_types:
        subset = [(q, e, t) for q, e, t in GROUND_TRUTH if t == qt]
        recalls_5 = []
        for query, expected, _ in subset:
            hits = search_with_graph(store, query, 5, g_combined)
            srcs = {h.source_name for h in hits[:5] if h.source_name}
            recalls_5.append(recall_at_k(srcs, set(expected), 5))
        n = len(subset)
        r5 = round(sum(recalls_5) / n, 4) if n else 0
        type_results[qt] = {"count": n, "R@5": r5}
        print(f"  {qt:<20} n={n:>2}  R@5={r5:.4f}")
    print()

    # ── Save results ──────────────────────────────────────────────────
    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    result_file = results_dir / f"structural_gap_ablation_{time.strftime('%Y%m%d_%H%M%S')}.json"

    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "corpus": {
            "total_docs": total_docs,
            "wiki_docs": wiki_count,
            "nonwiki_docs": nonwiki_count,
        },
        "config": {
            "alpha": ALPHA,
            "max_expansion": MAX_EXPANSION,
            "k_values": K_VALUES,
        },
        "ground_truth_count": len(GROUND_TRUTH),
        "graph_metrics": all_metrics,
        "structural_gap": injection_results,
        "retrieval_by_mode": results_by_mode,
        "query_type_breakdown": type_results,
    }
    result_file.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"Results saved to {result_file}")


if __name__ == "__main__":
    main()

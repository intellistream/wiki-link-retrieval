"""Controlled wiki-only benchmark: link expansion on isolated corpus.

Design principle: corpus has "hub" docs (keyword-rich, found by baseline)
and "spoke" docs (low keyword overlap, reachable only via links from hubs).
This is the exact scenario where 1-hop expansion outperforms pure search.

Ablation dimensions:
  - max_expansion: 1, 3, 5, 10
  - decay_factor: 0.3, 0.5, 0.7, 1.0
  - link direction: outbound-only vs bidirectional

Usage:
    cd /home/shuhao/sage-faculty-twin
    PYTHONPATH=src .venv311/bin/python ../wiki-link-retrieval/experiments/benchmark_wiki_only.py
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent.parent / "sage-faculty-twin" / "src")
)
from sage_faculty_twin.config import AppSettings
from sage_faculty_twin.knowledge_base import (
    LocalKnowledgeStore,
    _document_is_visible_to_requester,
)
from sage_faculty_twin.models import KnowledgeDocumentCreate, KnowledgeSearchHit

# ── Corpus: 20 docs with deliberate hub/spoke structure ────────────────────
#
# HUB docs: keyword-rich, many outbound links.  Baseline finds these easily.
# SPOKE docs: specific technical depth, low keyword overlap with typical
#   queries, but linked from hubs.  Only reachable via expansion.
#
# Query design: "hard-link" queries use synonyms/paraphrases that match
# hub docs lexically, but the expected answer is in a spoke doc.
#
CORPUS: list[dict] = [
    # ── Hubs: tutorials (keyword-rich, link outward) ─────────────────
    dict(
        sn="wiki:tutorials/inference-basics",
        ti="LLM推理入门 Prefill Decode 阶段",
        co=(
            "大语言模型推理分两个阶段。Prefill预填充处理输入prompt全部token，"
            "计算密集。Decode解码逐个生成token，访存密集。batch size影响吞吐。"
            "优化需减少计算量和访存量。常见技术包括量化投机采样等。"
        ),
        lk=[
            "wiki:tech-notes/kv-cache",
            "wiki:tech-notes/batching",
            "wiki:tech-notes/speculative",
            "wiki:resources/benchmarks",
        ],
    ),
    dict(
        sn="wiki:tutorials/npu-setup",
        ti="Ascend NPU 开发环境搭建 CANN torch_npu",
        co=(
            "华为Ascend 910B NPU开发环境。安装CANN Toolkit和PyTorch torch_npu。"
            "910B3配备64GB HBM。环境变量LD_LIBRARY_PATH配置。Docker镜像拉取运行。"
            "常见问题驱动安装失败编译报错等排查方法。"
        ),
        lk=[
            "wiki:tech-notes/npu-memory",
            "wiki:tech-notes/kv-cache",
            "wiki:resources/tools",
        ],
    ),
    dict(
        sn="wiki:tutorials/prompt-engineering",
        ti="Prompt Engineering 实践 System Prompt Few-shot CoT",
        co=(
            "Prompt Engineering最佳实践。System Prompt定义角色行为约束。"
            "Few-shot给示例。思维链CoT引导推理步骤。输出格式JSON Schema约束。"
            "RAG场景注入检索上下文。安全策略guardrails过滤。"
        ),
        lk=["wiki:tech-notes/rag-design", "wiki:industry/prompt-templates"],
    ),
    dict(
        sn="wiki:tutorials/distributed-ml",
        ti="分布式训练推理 Tensor Pipeline Data Parallelism",
        co=(
            "分布式系统基础。Tensor Parallelism权重按层切分到多卡。"
            "Pipeline Parallelism层分组跨节点。Data Parallelism每卡完整副本。"
            "通信开销NCCL AllReduce AllGather。异构部署混合精度。"
        ),
        lk=[
            "wiki:tech-notes/distributed-comms",
            "wiki:tech-notes/kv-cache",
            "wiki:tech-notes/npu-memory",
        ],
    ),
    dict(
        sn="wiki:achievements/sage-overview",
        ti="SAGE系统概述 vLLM BidKV FlowRAG NeuroMem",
        co=(
            "SAGE State-Aware Generative Engine项目群。推理引擎vLLM-HUST性能优化。"
            "BidKV双向缓存管理。FlowRAG数据流RAG。NeuroMem记忆系统。"
            "Faculty Twin数字孪生。StreamFP联邦学习。PagedAttention连续批处理。"
        ),
        lk=[
            "wiki:tech-notes/rag-design",
            "wiki:tech-notes/kv-cache",
            "wiki:tech-notes/distributed-comms",
            "wiki:resources/papers",
        ],
    ),
    # ── Spokes: tech-notes (deep, low keyword overlap with queries) ──
    dict(
        sn="wiki:tech-notes/kv-cache",
        ti="KV Cache PagedAttention Prefix Caching eviction",
        co=(
            "KV Cache推理核心数据结构。PagedAttention按page粒度管理消除碎片。"
            "Prefix Caching复用相同前缀cache块。RadixAttention radix tree匹配。"
            "量化FP16到INT8压缩。eviction策略LRU和frequency-based淘汰。"
        ),
        lk=[
            "wiki:tutorials/inference-basics",
            "wiki:tech-notes/batching",
            "wiki:tech-notes/npu-memory",
        ],
    ),
    dict(
        sn="wiki:tech-notes/batching",
        ti="Continuous Batching iteration-level scheduling preemption",
        co=(
            "iteration-level scheduling调度。每个decode step检查完成释放资源。"
            "Preemption swap到CPU或recompute重新prefill。动态调整batch composition。"
            "吞吐提升2到5倍。请求队列优先级调度。延迟SLO保障。"
        ),
        lk=[
            "wiki:tech-notes/kv-cache",
            "wiki:tech-notes/distributed-comms",
            "wiki:tutorials/inference-basics",
        ],
    ),
    dict(
        sn="wiki:tech-notes/npu-memory",
        ti="HBM显存架构 静态动态分配 defragmentation swap",
        co=(
            "Ascend NPU HBM显存架构。910B3 64GB带宽1.5TB/s。静态分配动态分配。"
            "gpu_memory_utilization预分配比例。碎片整理defragmentation。"
            "swap溢出到host内存。带宽瓶颈分析和优化方法。"
        ),
        lk=[
            "wiki:tech-notes/kv-cache",
            "wiki:tech-notes/batching",
            "wiki:tech-notes/distributed-comms",
        ],
    ),
    dict(
        sn="wiki:tech-notes/distributed-comms",
        ti="RDMA AllGather 跨节点通信 负载均衡 故障恢复",
        co=(
            "跨节点通信。RDMA直接内存访问绕过CPU。AllGather收集结果。"
            "KV Cache跨节点迁移。负载均衡请求路由。异构混合部署。"
            "故障恢复checkpoint。网络拓扑感知调度。"
        ),
        lk=[
            "wiki:tech-notes/npu-memory",
            "wiki:tech-notes/kv-cache",
            "wiki:tech-notes/batching",
        ],
    ),
    dict(
        sn="wiki:tech-notes/rag-design",
        ti="RAG Embedding ANNS Reranker HyDE chunk策略",
        co=(
            "Retrieval-Augmented Generation。Embedding编码ANNS索引Reranker精排。"
            "混合检索向量加BM25。HyDE假设文档嵌入。chunk overlap切分。"
            "链接图1-hop扩展上下文。评估指标recall和MRR。"
        ),
        lk=[
            "wiki:tutorials/prompt-engineering",
            "wiki:resources/papers",
            "wiki:achievements/sage-overview",
        ],
    ),
    dict(
        sn="wiki:tech-notes/speculative",
        ti="Speculative Decoding draft model verify acceptance",
        co=(
            "Speculative decoding小型draft model快速生成候选序列。"
            "大模型verify并行计算候选位置。acceptance rate决定加速比。"
            "Medusa多头预测。Eagle特征复用。draft模型训练策略。"
        ),
        lk=["wiki:tutorials/inference-basics", "wiki:tech-notes/batching"],
    ),
    dict(
        sn="wiki:tech-notes/prefix-cache",
        ti="Prefix Caching radix tree 命中率 冷启动预热",
        co=(
            "Prefix Caching复用相同前缀KV Cache块。系统提示词共享。"
            "多轮对话历史复用。Radix tree索引最长前缀匹配。"
            "cache命中率监控。冷启动预热。内存池管理分配。"
        ),
        lk=["wiki:tech-notes/kv-cache", "wiki:tech-notes/batching"],
    ),
    dict(
        sn="wiki:tech-notes/quantization",
        ti="GPTQ AWQ SmoothQuant INT8 INT4 FP8 精度选择",
        co=(
            "量化降低显存和计算。GPTQ训练后量化。AWQ激活感知。"
            "SmoothQuant迁移异常值到权重。INT8 INT4 FP8精度。"
            "量化误差校准数据集。推理引擎集成vLLM TensorRT-LLM。"
        ),
        lk=[
            "wiki:tech-notes/kv-cache",
            "wiki:tech-notes/npu-memory",
            "wiki:resources/papers",
        ],
    ),
    # ── Resources ─────────────────────────────────────────────────
    dict(
        sn="wiki:resources/papers",
        ti="必读论文 Attention PagedAttention Orca FlashAttention",
        co=(
            "必读论文。Attention Is All You Need Transformer。"
            "PagedAttention vLLM推理。Orca iteration scheduling。"
            "Speculative decoding。FlashAttention IO感知。KIVI量化。"
        ),
        lk=[
            "wiki:tech-notes/kv-cache",
            "wiki:tech-notes/batching",
            "wiki:tech-notes/speculative",
            "wiki:tutorials/inference-basics",
        ],
    ),
    dict(
        sn="wiki:resources/benchmarks",
        ti="推理Benchmark Throughput TTFT TBT p50 p95 p99",
        co=(
            "评估推理性能。Throughput吞吐。TTFT首token延迟。TBT token间延迟。"
            "E2E端到端延迟。控制变量重复实验。p50 p95 p99分位。"
            "负载模型open-loop closed-loop。测试脚本自动化。"
        ),
        lk=[
            "wiki:tutorials/inference-basics",
            "wiki:tech-notes/batching",
            "wiki:resources/tools",
        ],
    ),
    dict(
        sn="wiki:resources/tools",
        ti="vLLM SGLang torch_npu Git Docker npu-smi tmux",
        co=(
            "常用工具。vLLM高吞吐推理引擎。SGLang结构化生成。"
            "torch_npu NPU适配。Git版本控制Docker容器化。"
            "npu-smi监控显存。tmux终端复用screen会话管理。"
        ),
        lk=[
            "wiki:tutorials/npu-setup",
            "wiki:resources/papers",
            "wiki:standards/review-standards",
        ],
    ),
    dict(
        sn="wiki:resources/cluster-guide",
        ti="GPU集群 SLURM 作业调度 多节点 存储挂载",
        co=(
            "实验室GPU集群规范。Ascend 910B节点申请。SLURM调度系统。"
            "多节点MPI启动。存储挂载数据传输。队列优先级公平使用。"
            "故障报告流程。资源监控告警配置。"
        ),
        lk=[
            "wiki:tutorials/npu-setup",
            "wiki:tech-notes/distributed-comms",
            "wiki:resources/tools",
        ],
    ),
    # ── Standards & Industry ─────────────────────────────────────
    dict(
        sn="wiki:standards/review-standards",
        ti="代码Review PR CI lint approve Squash merge",
        co=(
            "代码审查标准。PR提交描述变更范围。CI自动lint单元测试。"
            "至少两人approve。Squash merge保持历史清晰。"
            "命名规范函数长度错误处理测试覆盖率docstring要求。"
        ),
        lk=["wiki:tutorials/prompt-engineering", "wiki:resources/tools"],
    ),
    dict(
        sn="wiki:standards/experiment-standards",
        ti="实验可复现性 随机种子 控制变量 置信区间",
        co=(
            "实验可复现性要求。固定随机种子。记录环境依赖版本。"
            "控制变量只改一个参数。至少3次重复取均值。"
            "报告标准差置信区间。结果表格可视化规范。"
        ),
        lk=[
            "wiki:resources/benchmarks",
            "wiki:resources/papers",
            "wiki:tutorials/distributed-ml",
        ],
    ),
    dict(
        sn="wiki:industry/prompt-templates",
        ti="CLI系统提示词 Agent 翻译摘要代码生成 安全合规",
        co=(
            "工业级CLI提示词设计。93个模板翻译摘要代码生成。"
            "Agent身份政策工具调用。输出风格控制。"
            "多轮对话上下文管理。安全红线合规策略。"
        ),
        lk=["wiki:tutorials/prompt-engineering"],
    ),
]

# ── Ground truth: 20 queries ─────────────────────────────────────────────
# easy:    direct keyword match — baseline already gets high recall
# medium:  multi-doc — expansion helps partially
# hard-link: relevant doc has LOW keyword overlap, only via links from hub
GT: list[tuple[str, list[str], str]] = [
    # Easy (5)
    ("什么是PagedAttention", ["wiki:tech-notes/kv-cache"], "easy"),
    ("Continuous Batching怎么调度", ["wiki:tech-notes/batching"], "easy"),
    ("Prompt Engineering怎么写", ["wiki:tutorials/prompt-engineering"], "easy"),
    ("代码review有什么规范", ["wiki:standards/review-standards"], "easy"),
    ("SAGE系统包含哪些项目", ["wiki:achievements/sage-overview"], "easy"),
    # Medium (5)
    (
        "LLM推理有哪些优化方法",
        [
            "wiki:tutorials/inference-basics",
            "wiki:tech-notes/kv-cache",
            "wiki:tech-notes/batching",
        ],
        "medium",
    ),
    (
        "分布式推理怎么部署",
        ["wiki:tutorials/distributed-ml", "wiki:tech-notes/distributed-comms"],
        "medium",
    ),
    (
        "有哪些必读论文推荐",
        ["wiki:resources/papers", "wiki:tech-notes/speculative"],
        "medium",
    ),
    (
        "RAG系统怎么设计",
        ["wiki:tech-notes/rag-design", "wiki:tutorials/prompt-engineering"],
        "medium",
    ),
    (
        "NPU开发环境怎么搭",
        ["wiki:tutorials/npu-setup", "wiki:resources/tools"],
        "medium",
    ),
    # Hard-link (10): query matches hub lexically, expected doc is spoke
    (
        "显存不足OOM怎么处理",  # query: "显存不足 OOM"
        [
            "wiki:tech-notes/npu-memory",  # doc: "HBM 显存架构 defragmentation swap"
            "wiki:tech-notes/kv-cache",  # doc: "PagedAttention eviction"
        ],
        "hard-link",
    ),
    (
        "怎么评估模型推理性能",  # query: "评估 推理 性能"
        [
            "wiki:resources/benchmarks",  # doc: "Throughput TTFT TBT p99"
            "wiki:tutorials/inference-basics",  # doc: "Prefill Decode batch"
        ],
        "hard-link",
    ),
    (
        "有什么好的AI工具推荐",  # query: "工具 推荐"
        [
            "wiki:resources/tools",  # doc: "vLLM SGLang Docker npu-smi"
            "wiki:resources/papers",  # doc: "论文 Attention"
        ],
        "hard-link",
    ),
    (
        "怎么加速decode阶段",  # query: "加速 decode"
        [
            "wiki:tech-notes/speculative",  # doc: "draft model verify acceptance"
            "wiki:tech-notes/batching",  # doc: "iteration-level scheduling"
        ],
        "hard-link",
    ),
    (
        "模型压缩有什么方法",  # query: "压缩 方法"
        [
            "wiki:tech-notes/quantization",  # doc: "GPTQ AWQ SmoothQuant"
            "wiki:tech-notes/kv-cache",  # doc: "PagedAttention FP16 INT8"
        ],
        "hard-link",
    ),
    (
        "怎么复用相同前缀的缓存",  # query: "复用 前缀 缓存"
        [
            "wiki:tech-notes/prefix-cache",  # doc: "radix tree 命中率 预热"
            "wiki:tech-notes/kv-cache",  # doc: "Prefix Caching RadixAttention"
        ],
        "hard-link",
    ),
    (
        "实验室GPU怎么申请使用",  # query: "GPU 申请 使用"
        [
            "wiki:resources/cluster-guide",  # doc: "SLURM 队列 MPI 挂载"
            "wiki:tutorials/npu-setup",  # doc: "CANN torch_npu Docker"
        ],
        "hard-link",
    ),
    (
        "实验结果怎么保证可复现",  # query: "保证 可复现"
        [
            "wiki:standards/experiment-standards",  # doc: "随机种子 控制变量 置信区间"
            "wiki:resources/benchmarks",  # doc: "p50 p95 重复实验"
        ],
        "hard-link",
    ),
    (
        "跨节点通信怎么优化",  # query: "跨节点 通信 优化"
        [
            "wiki:tech-notes/distributed-comms",  # doc: "RDMA AllGather 拓扑"
            "wiki:tech-notes/npu-memory",  # doc: "HBM 带宽 瓶颈"
        ],
        "hard-link",
    ),
    (
        "投机解码能加速多少",  # query: "投机 解码 加速"
        [
            "wiki:tech-notes/speculative",  # doc: "draft verify acceptance rate"
            "wiki:resources/papers",  # doc: "论文 Speculative"
        ],
        "hard-link",
    ),
]

KS = [3, 5, 10]


def _recall(hits: list, expected: list[str], k: int) -> float:
    found = {h.source_name for h in hits[:k]}
    return len(found & set(expected)) / max(len(expected), 1)


def _rr(hits: list, expected: list[str]) -> float:
    for i, h in enumerate(hits):
        if h.source_name in set(expected):
            return 1.0 / (i + 1)
    return 0.0


def _build_outbound_graph(store: LocalKnowledgeStore) -> dict[str, list[str]]:
    """Build forward-only graph (no reverse edges)."""
    s2id = {d.source_name: did for did, d in store._documents.items() if d.source_name}
    g: dict[str, list[str]] = {}
    for did, doc in store._documents.items():
        ls = (doc.metadata or {}).get("linked_source_names", "")
        if not ls:
            continue
        nb = []
        for sn in ls.split("|"):
            sn = sn.strip()
            if sn and sn in s2id and s2id[sn] != did and s2id[sn] not in nb:
                nb.append(s2id[sn])
        if nb:
            g[did] = nb
    return g


def build_store(tmpdir: Path) -> LocalKnowledgeStore:
    s = LocalKnowledgeStore(
        AppSettings(knowledge_base_dir=tmpdir, knowledge_backend="local")
    )
    for p in CORPUS:
        m: dict[str, str] = {}
        if p["lk"]:
            m["linked_source_names"] = "|".join(p["lk"])
        m["wiki_category"] = p["sn"].split(":")[-1].split("/")[0]
        s.add_document(
            KnowledgeDocumentCreate(
                title=p["ti"],
                content=p["co"],
                source_name=p["sn"],
                metadata=m,
            ),
            rebuild_indexes=False,
        )
    return s


def run_config(
    store: LocalKnowledgeStore,
    *,
    link_exp: bool,
    max_exp: int = 3,
    decay: float = 0.5,
    bidir: bool = True,
) -> dict[int, dict]:
    """Run all queries with given config, return metrics per k."""
    store._link_expansion_enabled = link_exp
    orig_graph = store._link_graph

    if link_exp and not bidir:
        store._link_graph = _build_outbound_graph(store)

    orig_expand = store._expand_hits_with_links
    if link_exp:

        def custom_expand(hits, qt, qp, _me=max_exp, _df=decay):
            if not store._link_graph or not hits:
                return hits
            seen = {h.document_id for h in hits}
            exp = list(hits)
            for hit in hits:
                if len(exp) - len(hits) >= _me:
                    break
                for nid in store._link_graph.get(hit.document_id, []):
                    if nid in seen or len(exp) - len(hits) >= _me:
                        continue
                    doc = store._documents.get(nid)
                    if not doc:
                        continue
                    if not _document_is_visible_to_requester(
                        doc, qp.visitor_profile, qp.admin_role
                    ):
                        continue
                    exp.append(
                        KnowledgeSearchHit(
                            document_id=doc.document_id,
                            title=doc.title,
                            excerpt=store._build_excerpt(doc.content, qt),
                            score=max(hit.score * _df, 1.0),
                            tags=doc.tags,
                            source_name=doc.source_name,
                            metadata=doc.metadata,
                        )
                    )
                    seen.add(nid)
            return exp

        store._expand_hits_with_links = custom_expand

    rc_all: dict[int, list] = {k: [] for k in KS}
    rr_all: dict[int, list] = {k: [] for k in KS}
    lat_all: list[float] = []
    for q, expected, _ in GT:
        t0 = time.perf_counter()
        hits = store.search(q, top_k=max(KS))
        lat_all.append((time.perf_counter() - t0) * 1000)
        for k in KS:
            rc_all[k].append(_recall(hits, expected, k))
            rr_all[k].append(_rr(hits[:k], expected))

    store._link_expansion_enabled = False
    store._link_graph = orig_graph
    if link_exp:
        store._expand_hits_with_links = orig_expand

    n = len(GT)
    return {
        k: {
            "recall": round(sum(rc_all[k]) / n, 4),
            "mrr": round(sum(rr_all[k]) / n, 4),
            "latency_ms": round(sum(lat_all) / n, 2),
        }
        for k in KS
    }


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="wb-") as tmp:
        store = build_store(Path(tmp))
        nd = store.count_documents()
        nn = len(store._link_graph)
        ne = sum(len(v) for v in store._link_graph.values())
        print(f"KB: {nd} docs | graph: {nn} nodes, {ne} edges | GT: {len(GT)} queries")
        print()

        # ── Main comparison ─────────────────────────────────────
        print("=" * 64)
        print("BASELINE vs LINK-EXPANDED (bidirectional, max=3, decay=0.5)")
        print("=" * 64)
        bl = run_config(store, link_exp=False)
        ex = run_config(store, link_exp=True, bidir=True)
        for k in KS:
            b, e = bl[k], ex[k]
            dr = e["recall"] - b["recall"]
            dm = e["mrr"] - b["mrr"]
            dl = e["latency_ms"] - b["latency_ms"]
            print(f"\n  k={k}")
            print(
                f"    Recall  base={b['recall']:.3f}  exp={e['recall']:.3f}  Δ={dr:+.3f}"
            )
            print(f"    MRR     base={b['mrr']:.3f}  exp={e['mrr']:.3f}  Δ={dm:+.3f}")
            print(
                f"    Latency base={b['latency_ms']:.1f}ms  exp={e['latency_ms']:.1f}ms  Δ={dl:+.1f}ms"
            )
        print()

        # ── Query type breakdown ────────────────────────────────
        print("=" * 64)
        print("QUERY TYPE BREAKDOWN (k=5, bidirectional)")
        print("=" * 64)
        types = sorted(set(t for _, _, t in GT))
        type_res: dict[str, dict] = {}
        for qt in types:
            subset = [(q, e) for q, e, t in GT if t == qt]
            store._link_expansion_enabled = False
            rb = [_recall(store.search(q, top_k=5), e, 5) for q, e in subset]
            store._link_expansion_enabled = True
            re = [_recall(store.search(q, top_k=5), e, 5) for q, e in subset]
            store._link_expansion_enabled = False
            n = len(subset)
            type_res[qt] = {
                "base": round(sum(rb) / n, 4),
                "exp": round(sum(re) / n, 4),
                "delta": round(sum(re) / n - sum(rb) / n, 4),
                "n": n,
            }
            print(
                f"  {qt:12s}  base={sum(rb)/n:.3f}  exp={sum(re)/n:.3f}  "
                f"Δ={sum(re)/n - sum(rb)/n:+.3f}  (n={n})"
            )
        print()

        # ── Ablation: max_expansion ──────────────────────────────
        print("=" * 64)
        print("ABLATION: max_expansion (k=5, decay=0.5, bidirectional)")
        print("=" * 64)
        abl_me: dict[int, dict] = {}
        for me in [1, 3, 5, 10]:
            r = run_config(store, link_exp=True, max_exp=me, bidir=True)
            abl_me[me] = r[5]
            print(
                f"  max={me:2d}  R@5={r[5]['recall']:.3f}  "
                f"MRR={r[5]['mrr']:.3f}  lat={r[5]['latency_ms']:.1f}ms"
            )
        print()

        # ── Ablation: decay factor ───────────────────────────────
        print("=" * 64)
        print("ABLATION: decay_factor (k=5, max=3, bidirectional)")
        print("=" * 64)
        abl_df: dict[float, dict] = {}
        for df in [0.3, 0.5, 0.7, 1.0]:
            r = run_config(store, link_exp=True, max_exp=3, decay=df, bidir=True)
            abl_df[df] = r[5]
            print(
                f"  decay={df:.1f}  R@5={r[5]['recall']:.3f}  "
                f"MRR={r[5]['mrr']:.3f}  lat={r[5]['latency_ms']:.1f}ms"
            )
        print()

        # ── Ablation: link direction ─────────────────────────────
        print("=" * 64)
        print("ABLATION: link direction (k=5, max=3, decay=0.5)")
        print("=" * 64)
        abl_dir: dict[str, dict] = {}
        for bidir, label in [(False, "outbound-only"), (True, "bidirectional")]:
            r = run_config(store, link_exp=True, max_exp=3, decay=0.5, bidir=bidir)
            abl_dir[label] = r[5]
            print(
                f"  {label:18s}  R@5={r[5]['recall']:.3f}  "
                f"MRR={r[5]['mrr']:.3f}  lat={r[5]['latency_ms']:.1f}ms"
            )
        abl_dir["baseline"] = bl[5]
        print(
            f"  {'baseline':18s}  R@5={bl[5]['recall']:.3f}  "
            f"MRR={bl[5]['mrr']:.3f}  lat={bl[5]['latency_ms']:.1f}ms"
        )
        print()

        # ── Per-query detail: hard-link ──────────────────────────
        print("=" * 64)
        print("PER-QUERY DETAIL: hard-link (k=5)")
        print("=" * 64)
        for q, expected, qt in GT:
            if qt != "hard-link":
                continue
            store._link_expansion_enabled = False
            hb = store.search(q, top_k=5)
            store._link_expansion_enabled = True
            he = store.search(q, top_k=5)
            rb = _recall(hb, expected, 5)
            re_ = _recall(he, expected, 5)
            bs = [h.source_name.split("/")[-1] for h in hb[:5]]
            es = [h.source_name.split("/")[-1] for h in he[:5]]
            print(f"  Q: {q}")
            print(f"    Expected: {[s.split('/')[-1] for s in expected]}")
            print(f"    Base R@5={rb:.2f}  {bs}")
            print(f"    Exp  R@5={re_:.2f}  {es}")
            print(f"    Δ={re_-rb:+.2f}")
            print()

        # ── Save ─────────────────────────────────────────────────
        rd = Path(__file__).resolve().parent.parent / "results"
        rd.mkdir(exist_ok=True)
        rf = rd / f"wiki_controlled_{time.strftime('%Y%m%d_%H%M%S')}.json"
        output = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "corpus_size": nd,
            "graph_nodes": nn,
            "graph_edges": ne,
            "ground_truth": len(GT),
            "type_counts": {
                t: sum(1 for _, _, qt in GT if qt == t)
                for t in sorted(set(qt for _, _, qt in GT))
            },
            "main": {"baseline": bl, "expanded_bidir": ex},
            "type_breakdown": type_res,
            "ablation_max_exp": {str(k): v for k, v in abl_me.items()},
            "ablation_decay": {str(k): v for k, v in abl_df.items()},
            "ablation_direction": abl_dir,
        }
        rf.write_text(json.dumps(output, indent=2, ensure_ascii=False))
        print(f"Results saved to {rf}")


if __name__ == "__main__":
    main()

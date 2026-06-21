#!/usr/bin/env bash
# Run all wiki-link retrieval experiments.
# Executes inside Docker container (see scripts/run_docker.sh).
set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="/workspace/results"
mkdir -p "${RESULTS_DIR}"

echo "=============================================="
echo " Wiki-Link Retrieval — Full Experiment Suite"
echo " Timestamp: ${TIMESTAMP}"
echo "=============================================="

# ── 1. Multi-hop QA benchmarks (primary) ──────────────────────────────────
echo ""
echo "[1/4] HotpotQA — multi-hop QA with Wikipedia link structure"
echo "-----------------------------------------------------------"
python3 experiments/benchmark_multihop.py \
    --datasets hotpotqa \
    --retrievers bm25 \
    --alpha 0.3 0.5 0.7 \
    --max-expansion 1 3 5 \
    --seeds 42 43 44 45 46 \
    --subsample 1000 \
    --analyze-coverage \
    --output "results/multihop_hotpotqa_${TIMESTAMP}.json"

echo ""
echo "[2/4] 2WikiMultiHopQA — comprehensive multi-hop evaluation"
echo "-----------------------------------------------------------"
python3 experiments/benchmark_multihop.py \
    --datasets 2wikimultihop \
    --retrievers bm25 \
    --alpha 0.3 0.5 0.7 \
    --max-expansion 1 3 5 \
    --seeds 42 43 44 45 46 \
    --subsample 1000 \
    --output "results/multihop_2wiki_${TIMESTAMP}.json"

# ── 2. Standard IR benchmarks ─────────────────────────────────────────────
echo ""
echo "[3/4] MS MARCO + BEIR — standard passage/document ranking"
echo "-----------------------------------------------------------"
python3 experiments/benchmark_standard_ir.py \
    --datasets msmarco nq scifact \
    --retrievers bm25 \
    --alpha 0.3 0.5 \
    --max-expansion 3 \
    --seeds 42 43 44 45 46 \
    --subsample 500 \
    --output "results/standard_ir_${TIMESTAMP}.json"

# ── 3. GraphRAG comparison ────────────────────────────────────────────────
echo ""
echo "[4/4] GraphRAG comparison (scaffold mode — no API key needed)"
echo "-----------------------------------------------------------"
python3 experiments/graphrag_baseline.py \
    --mode scaffold \
    --datasets hotpotqa \
    --subsample 1000 \
    --output "results/graphrag_comparison_${TIMESTAMP}.json"

echo ""
echo "=============================================="
echo " All experiments complete!"
echo " Results saved to: ${RESULTS_DIR}/"
echo "=============================================="
ls -la "${RESULTS_DIR}"/*.json 2>/dev/null || echo "No result files found"

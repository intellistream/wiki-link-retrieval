#!/usr/bin/env bash
# Run wiki-link-retrieval experiments inside Docker container.
#
# Uses the local Ascend vllm image directly (no custom build needed).
# Installs deps at runtime, mounts code as volumes.
#
# Usage:
#   bash scripts/run_docker.sh                    # run all experiments
#   bash scripts/run_docker.sh hotpotqa           # run only HotpotQA
#   bash scripts/run_docker.sh --shell            # interactive shell
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BASE_IMAGE="quay.io/ascend/vllm-ascend:v0.21.0rc1-openeuler"
RESULTS_DIR="${REPO_ROOT}/results"
mkdir -p "${RESULTS_DIR}"

# ── Install deps script (runs inside container) ──────────────────────────
INSTALL_DEPS='pip install --no-cache-dir scipy scikit-learn pytrec-eval-terrier pyarrow tqdm 2>&1 | tail -3'

# ── Common Docker flags ──────────────────────────────────────────────────
DOCKER_FLAGS=(
    --rm
    -v "${REPO_ROOT}/experiments:/workspace/experiments"
    -v "${REPO_ROOT}/results:/workspace/results"
    -v "${REPO_ROOT}/scripts:/workspace/scripts"
    -v "${REPO_ROOT}/data:/workspace/data"
    -e "PYTHONPATH=/workspace/experiments"
    -w "/workspace"
    --shm-size=4g
)

# ── Mode selection ────────────────────────────────────────────────────────
if [[ "${1:-}" == "--shell" ]]; then
    echo "[*] Starting interactive shell ..."
    sudo docker run -it "${DOCKER_FLAGS[@]}" \
        --entrypoint /bin/bash \
        "${BASE_IMAGE}"
elif [[ -n "${1:-}" ]]; then
    echo "[*] Running experiment: $1"
    sudo docker run "${DOCKER_FLAGS[@]}" \
        --entrypoint /bin/bash \
        "${BASE_IMAGE}" \
        -c "${INSTALL_DEPS} && python3 experiments/benchmark_multihop.py \
            --datasets $1 \
            --retrievers bm25 \
            --alpha 0.3 0.5 \
            --max-expansion 3 \
            --seeds 42 43 44 45 46 \
            --subsample 500 \
            --analyze-coverage \
            --output results/multihop_${1}_$(date +%Y%m%d_%H%M%S).json"
else
    echo "[*] Running all experiments ..."
    sudo docker run "${DOCKER_FLAGS[@]}" \
        --entrypoint /bin/bash \
        "${BASE_IMAGE}" \
        -c "${INSTALL_DEPS} && bash scripts/run_experiments.sh"
fi

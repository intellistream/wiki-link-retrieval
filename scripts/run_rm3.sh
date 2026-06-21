#!/bin/bash
set -e
pip install pyarrow scikit-learn pytrec-eval-terrier
cd /workspace
python experiments/benchmark_multihop.py \
    --datasets hotpotqa \
    --retrievers bm25 \
    --alpha 0.3 \
    --max-expansion 3 \
    --seeds 42 43 44 45 46 \
    --output results/multihop_hotpotqa_rm3_comparison.json

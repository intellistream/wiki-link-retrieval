# Wiki-Link Retrieval — Experiment Environment
#
# Based on local Ascend vllm image (Docker Hub unreachable).
# Build: sudo docker build -t wiki-link-exp -f scripts/Dockerfile .
# Run:   sudo docker run --rm -v $(pwd):/workspace wiki-link-exp bash scripts/run_experiments.sh

FROM quay.io/ascend/vllm-ascend:v0.21.0rc1-openeuler

WORKDIR /workspace

# Install experiment-specific Python deps
# (torch, numpy already present in base image)
RUN pip install --no-cache-dir \
    scipy>=1.10 \
    scikit-learn>=1.2 \
    pytrec-eval-terrier>=0.5 \
    datasets>=2.14 \
    huggingface-hub>=0.20 \
    sentence-transformers>=2.2 \
    openai>=1.0 \
    tqdm>=4.65

# Pre-download embedding model for DPR retrieval
RUN python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" || true

# Copy experiment code
COPY requirements.txt /workspace/requirements.txt
COPY experiments/ /workspace/experiments/
COPY scripts/ /workspace/scripts/
RUN mkdir -p /workspace/results

ENV PYTHONPATH=/workspace/experiments

CMD ["bash", "scripts/run_experiments.sh"]

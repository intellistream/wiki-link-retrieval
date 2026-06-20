# Wiki-Link Retrieval

**Lightweight link-graph expansion for curated knowledge bases — a pragmatic alternative to GraphRAG.**

**Target venue:** NeurIPS 2026 RAG Workshop (~Sep) → backup: SIGIR 2027 Short Paper

## Idea

Most LLM-powered knowledge systems retrieve documents by vector similarity (ANNS),
ignoring the explicit link structure between documents. GraphRAG-style approaches
build entity graphs from unstructured text, but this is expensive and error-prone.

We observe that **curated knowledge bases (team wikis, documentation sites) already
contain explicit, human-authored links** between pages. Instead of extracting a
graph from text, we **parse the existing link structure** and use it for
post-retrieval expansion:

```
Query → ANNS retrieves top-k documents
      → 1-hop link expansion via pre-built adjacency list
      → Re-rank: primary results + linked context
```

This gives wiki-quality inter-linking at ANNS speed — without entity extraction,
community detection, or a graph database.

## Architecture

```
┌──────────────────┐     ┌──────────────┐     ┌────────────────────┐
│ Docusaurus wiki  │────>│  Ingestion   │────>│  JSON KB entries   │
│ (authoring)      │     │  + link parse│     │  + adjacency list  │
└──────────────────┘     └──────────────┘     └─────────┬──────────┘
                                                        │
                                              ┌─────────▼──────────┐
                                              │  ANNS retrieval    │
                                              │  (sagevdb + HNSW) │
                                              └─────────┬──────────┘
                                                        │
                                              ┌─────────▼──────────┐
                                              │  Lexical reranking │
                                              │  + link expansion  │
                                              └─────────┬──────────┘
                                                        │
                                              ┌─────────▼──────────┐
                                              │  Final top-k hits  │
                                              │  with linked context│
                                              └────────────────────┘
```

## Key Properties

| Property | Vanilla ANNS | GraphRAG | Wiki-Link Retrieval |
|---|---|---|---|
| Inter-linking | None | Auto-extracted | Human-authored |
| Index cost | O(n) embeddings | O(n × LLM calls) | O(n) embeddings + O(E) link parse |
| Query cost | ANNS + rerank | ANNS + graph traversal | ANNS + rerank + O(k) dict lookups |
| Link quality | N/A | Noisy (LLM extraction) | High (human-curated) |
| Dependencies | Vector DB | Vector DB + Graph DB + LLM | Vector DB only |
| Auto-sync | N/A | Re-index from scratch | systemd timer + git pull |

## Auto-Sync (CI/CD)

When wiki pages are added or updated in `sage-wiki`, the KB automatically
absorbs the new knowledge:

```
sage-wiki push → systemd timer (every 30min)
                → git pull sage-wiki
                → ingest_wiki.py (upsert + rebuild link graph)
                → KB is live with new content
```

- **Sync script**: `tools/sync_wiki_kb.sh` (in sage-faculty-twin)
- **Timer**: `sage-faculty-twin-wiki-sync.timer` (systemd, 30-min interval)
- **Idempotent**: safe to run even if wiki hasn't changed (skips ingest)
- **Manual trigger**: `bash tools/sync_wiki_kb.sh` for immediate sync

## Repository Structure

```
wiki-link-retrieval/
├── paper/          # LaTeX paper drafts, figures
├── experiments/    # Benchmark scripts comparing retrieval quality
├── results/        # Experiment result data (JSON/CSV)
└── README.md
```

The **implementation code** lives in the `sage-faculty-twin` repository
(`knowledge_base.py`, `tools/ingest_wiki.py`), as the feature is tightly
coupled with the existing KB search pipeline.

## Research Questions

1. **RQ1**: How much does 1-hop link expansion improve retrieval recall
   compared to vanilla ANNS on curated knowledge bases?
2. **RQ2**: What is the latency overhead of link expansion at query time?
   (Expected: negligible — O(k) dict lookups)
3. **RQ3**: How does link expansion compare to GraphRAG in terms of
   retrieval quality vs. system complexity trade-off?

## Status

- [x] Research repo created
- [x] Initial architecture designed
- [x] Wiki ingestion pipeline implemented (`tools/ingest_wiki.py`)
- [x] Link graph builder implemented (bidirectional adjacency in `knowledge_base.py`)
- [x] Post-retrieval expansion in search pipeline (local + sagevdb + neuromem)
- [x] Experiment: retrieval quality baseline (vanilla ANNS)
- [x] Experiment: retrieval quality with link expansion (Δ=+7% R@5 on hard queries)
- [x] Ablation studies (decay, max_expansion, direction)
- [x] GraphRAG baseline comparison (2.2× better Recall, 3.2× better MRR)
- [x] Paper draft (`paper/wiki-link-retrieval.md`)
- [ ] Full LLM-based GraphRAG comparison
- [ ] Scale evaluation to larger corpora

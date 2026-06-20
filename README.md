# Wiki-Link Retrieval

**Lightweight link-graph expansion for curated knowledge bases — a pragmatic alternative to GraphRAG.**

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
- [ ] Wiki ingestion pipeline implemented
- [ ] Link graph builder implemented
- [ ] Post-retrieval expansion in search pipeline
- [ ] Experiment: retrieval quality baseline (vanilla ANNS)
- [ ] Experiment: retrieval quality with link expansion
- [ ] Paper draft

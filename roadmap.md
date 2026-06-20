# Wiki-Link Retrieval — Submission Roadmap

**Last updated:** 2026-06-21

## Venue Timeline

| Venue | Deadline | Status | Fit |
|-------|----------|--------|-----|
| NeurIPS 2026 Main | May 6, 2026 | ❌ Passed | — |
| NeurIPS 2026 Workshops | ~Aug 29, 2026 (paper) | 🔍 Check RAG/IR workshop CFP | Good — 4-page format |
| SIGIR 2027 Short Paper | ~Jan 2027 (est.) | 🎯 Primary target | Good — 4-6 pages |
| NeurIPS 2027 | ~May 2027 (est.) | 🎯 Stretch target | Best — full paper |

> **Recommendation:** Target NeurIPS 2026 RAG Workshop (Aug) if CFP opens, then SIGIR 2027 Short (Jan 2027) as primary, NeurIPS 2027 as stretch.

---

## Current State (as of 2026-06-21)

### ✅ Completed

- [x] Core idea: parse wiki markdown links → adjacency list → post-retrieval 1-hop expansion
- [x] Implementation: link graph builder + expansion in `sage-faculty-twin/knowledge_base.py`
- [x] Auto-sync pipeline: systemd timer + `sync_wiki_kb.sh` (zero maintenance)
- [x] Bidirectional adjacency list construction (20 nodes, 88 edges, avg degree 4.4)
- [x] Experiment scripts: `benchmark_retrieval_quality.py`, `benchmark_wiki_only.py`, `graphrag_baseline.py`
- [x] Results: 5 JSON result files in `results/`
- [x] Main results: +7% Recall@5 (α=0.3), 2.2× vs GraphRAG
- [x] Ablation studies: decay factor, max_expansion, link direction
- [x] Paper draft in LaTeX (NeurIPS 2026 template, tectonic build)
- [x] BibTeX references (9 entries)
- [x] `make paper` / `make paper-watch` build system

### ❌ Gaps to Close Before Submission

| # | Gap | Severity | Blocks |
|---|-----|----------|--------|
| G1 | Small corpus (n=20) — reviewers will reject | 🔴 Critical | All venues |
| G2 | GraphRAG baseline uses regex NER, not real LLM extraction | 🔴 Critical | SIGIR, NeurIPS |
| G3 | No semantic embedding baseline (only token-overlap ANNS) | 🟡 Important | SIGIR, NeurIPS |
| G4 | No statistical significance tests (confidence intervals) | 🟡 Important | All venues |
| G5 | Related work needs 15+ more citations | 🟡 Important | All venues |
| G6 | No system architecture figure (only text pipeline) | 🟢 Nice-to-have | All venues |
| G7 | NeurIPS checklist section missing | 🟢 Required (NeurIPS only) | NeurIPS |
| G8 | No ablation on corpus size (scaling behavior) | 🟡 Important | NeurIPS |
| G9 | Hard-link query count too small (n=10) for significance | 🟡 Important | SIGIR, NeurIPS |
| G10 | No code release / reproducibility package | 🟢 Nice-to-have | NeurIPS |

---

## Task Breakdown

### Phase 1: Experiment Strengthening (Weeks 1–3)

> Goal: Close G1, G2, G3, G4, G8, G9 — make results defensible.

#### Task 1.1 — Scale up corpus to 100+ documents [G1, G8]

- [ ] Expand wiki corpus from 20 → 100+ documents
  - Source: `sage-wiki` full content (~50 pages) + synthetic wiki pages
  - Or: use a public wiki dataset (e.g., Wikipedia subset with link structure)
- [ ] Generate ground-truth queries for scaled corpus (50+ queries)
  - Easy / Medium / Hard-link split (30% / 30% / 40%)
  - Hard-link queries must require expansion to find relevant doc
- [ ] Re-run all experiments on scaled corpus
- [ ] Add scaling ablation: n ∈ {20, 50, 100, 200}

#### Task 1.2 — Real LLM-based GraphRAG baseline [G2]

- [ ] Implement proper GraphRAG pipeline:
  - Entity extraction via LLM (GPT-4o or Qwen2.5)
  - Relationship extraction
  - Community detection (Leiden algorithm)
  - Community summarization
- [ ] Use `microsoft/graphrag` reference implementation if possible
- [ ] Compare: index cost (LLM tokens × $), extraction quality, retrieval quality
- [ ] Report fair comparison: same corpus, same queries, same metrics

#### Task 1.3 — Semantic embedding baseline [G3]

- [ ] Add dense embedding baseline (e.g., `text-embedding-3-small` or `bge-large-zh`)
- [ ] Compare: token-overlap ANNS vs. dense HNSW vs. wiki-link on top of dense
- [ ] Show wiki-link expansion improves recall regardless of embedding type

#### Task 1.4 — Statistical significance [G4, G9]

- [ ] Run experiments with multiple random seeds / query splits (≥5)
- [ ] Compute mean ± std for Recall@k, MRR, NDCG
- [ ] Paired t-test or Wilcoxon signed-rank test vs. baseline
- [ ] Report p-values in all tables
- [ ] Expand hard-link queries to n ≥ 20

### Phase 2: Paper Polish (Weeks 3–5)

> Goal: Close G5, G6, G7 — make the paper look professional and complete.

#### Task 2.1 — Expand related work [G5]

- [ ] Add citations for:
  - Classic IR: BM25, TF-IDF (Robertson & Sparck Jones)
  - Dense retrieval: DPR, ColBERT, ANCE
  - Graph-based RAG: GraphRAG, LightRAG, KG-RAG, RAPTOR
  - Document expansion: doc2query, DocT5query
  - Link analysis: PageRank, HITS (for context, not direct comparison)
  - Post-retrieval: query expansion, document expansion, re-ranking
- [ ] Target: 25+ references total (currently 9)
- [ ] Rewrite §2 with proper academic narrative flow

#### Task 2.2 — System architecture figure [G6]

- [ ] Create TikZ or draw.io figure showing:
  - Wiki → Ingestion → Link Graph → ANNS → Expansion → Output
  - Color-coded: zero-cost components vs. standard RAG components
- [ ] Add pipeline timing breakdown figure (index build vs. query time)

#### Task 2.3 — NeurIPS checklist [G7]

- [ ] Add NeurIPS paper checklist appendix (required for NeurIPS submission)
  - Claims, limitations, ethics, reproducibility
- [ ] Add broader impact statement

#### Task 2.4 — LaTeX fixes

- [ ] Fix overfull hbox in Table 1 (landscape table — use `\resizebox` or `\small`)
- [ ] Fix UTF-8 warning (line 11)
- [ ] Add `\usepackage{microtype}` for better typography
- [ ] Add NDCG@k metric to all result tables

### Phase 3: Packaging & Submission (Weeks 5–6)

> Goal: Close G10, finalize, submit.

#### Task 3.1 — Reproducibility package [G10]

- [ ] Standalone experiment scripts with clear README
- [ ] Corpus + ground truth queries as downloadable artifact
- [ ] `requirements.txt` for experiment dependencies
- [ ] One-command reproduce script: `bash scripts/reproduce.sh`

#### Task 3.2 — Internal review

- [ ] Send draft to 1–2 collaborators for feedback
- [ ] Address reviewer comments
- [ ] Final proofreading pass

#### Task 3.3 — Submission preparation

- [ ] De-anonymize author info (if not double-blind)
- [ ] Add `[preprint]` option for arXiv upload
- [ ] Prepare supplementary material (appendix with full results tables)
- [ ] Submit to target venue

---

## Detailed Task Checklist (Master List)

### Experiments
- [ ] 1.1a: Scale corpus to 100+ docs
- [ ] 1.1b: Generate 50+ ground-truth queries
- [ ] 1.1c: Re-run experiments on scaled corpus
- [ ] 1.1d: Scaling ablation (n=20,50,100,200)
- [ ] 1.2a: Implement LLM-based GraphRAG
- [ ] 1.2b: Fair comparison with GraphRAG
- [ ] 1.3a: Dense embedding baseline
- [ ] 1.3b: Wiki-link expansion on top of dense retrieval
- [ ] 1.4a: Multi-seed statistical runs (≥5)
- [ ] 1.4b: Confidence intervals + significance tests
- [ ] 1.4c: Expand hard-link queries to n≥20

### Paper
- [ ] 2.1a: Add 15+ citations to related work
- [ ] 2.1b: Rewrite §2 with narrative flow
- [ ] 2.2a: Create system architecture figure (TikZ/draw.io)
- [ ] 2.2b: Add pipeline timing figure
- [ ] 2.3a: NeurIPS checklist appendix
- [ ] 2.3b: Broader impact statement
- [ ] 2.4a: Fix Table 1 overfull hbox
- [ ] 2.4b: Fix UTF-8 warning
- [ ] 2.4c: Add microtype package
- [ ] 2.4d: Add NDCG@k to all tables
- [ ] 2.4e: Update all tables with significance tests
- [ ] 2.4f: Update abstract with scaled results

### Packaging
- [ ] 3.1a: Standalone experiment scripts
- [ ] 3.1b: Corpus + queries as downloadable artifact
- [ ] 3.1c: `scripts/reproduce.sh` one-command run
- [ ] 3.2a: Internal review (1–2 collaborators)
- [ ] 3.2b: Address review comments
- [ ] 3.2c: Final proofreading
- [ ] 3.3a: De-anonymize (if applicable)
- [ ] 3.3b: Prepare supplementary material
- [ ] 3.3c: Submit to venue

---

## Estimated Effort

| Phase | Tasks | Est. Hours | Priority |
|-------|-------|-----------|----------|
| 1.1 Corpus scaling | 4 tasks | 16–24h | 🔴 Must |
| 1.2 LLM GraphRAG | 2 tasks | 12–16h | 🔴 Must |
| 1.3 Dense embedding | 2 tasks | 6–8h | 🟡 Should |
| 1.4 Statistics | 3 tasks | 4–6h | 🟡 Should |
| 2.1 Related work | 2 tasks | 6–8h | 🟡 Should |
| 2.2 Figures | 2 tasks | 4–6h | 🟢 Could |
| 2.3 Checklist | 2 tasks | 2–3h | 🔴 Must (NeurIPS) |
| 2.4 LaTeX fixes | 6 tasks | 3–4h | 🟢 Could |
| 3.1 Reproducibility | 3 tasks | 4–6h | 🟢 Could |
| 3.2–3.3 Review & submit | 6 tasks | 8–12h | 🔴 Must |
| **Total** | **36 tasks** | **65–93h** | — |

---

## Venue-Specific Requirements

### NeurIPS 2026 Workshop (4 pages + refs)
- ✅ Current experiments sufficient for workshop format
- ✅ Paper length matches
- Need: check RAG/IR workshop CFP (usually posted by July)
- Need: NeurIPS checklist

### SIGIR 2027 Short Paper (4–6 pages + refs)
- 🔴 Need: scaling to 100+ docs (G1)
- 🔴 Need: fair GraphRAG comparison (G2)
- 🟡 Need: statistical significance (G4)
- 🟡 Need: expanded related work (G5)

### NeurIPS 2027 Full Paper (9 pages + refs + unlimited appendix)
- 🔴 Need: all G1–G10 closed
- 🔴 Need: dense embedding baseline (G3)
- 🔴 Need: scaling ablation (G8)
- 🔴 Need: reproducibility package (G10)
- 🔴 Need: broader impact statement

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| No RAG workshop CFP | Target SIGIR 2027 Short instead |
| Corpus scaling too hard | Use public Wikipedia subset with existing link structure |
| GraphRAG too expensive to run | Use nano-graphrag (open-source, cheaper) for fair comparison |
| Results don't scale | Honest reporting — discuss when it works and when it doesn't |
| Collaborator unavailable | Self-review + arXiv preprint for community feedback |

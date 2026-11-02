# MedShop-RAG

**Safety-Aware Retrieval-Augmented Product Search for OTC Healthcare E-Commerce**

> Submitted to **ISIR-eCom 2026** — 2nd Workshop on Information Search and Retrieval for E-Commerce  
> Co-located with IEEE International Conference on Big Data (IEEE BigData 2026)  
> Washington D.C., USA · December 2026

---

## Overview

MedShop-RAG is a safety-aware retrieval-augmented generation (RAG) pipeline for over-the-counter (OTC) healthcare product search in e-commerce settings. It addresses the gap between generic semantic search and the safety-critical requirements of consumer health product retrieval.

The system integrates five components:

| Component | Role |
|---|---|
| C1 — OTC Intent Classifier | Filters healthcare-specific queries |
| C2 — Hybrid Catalog Retriever | BM25 + TF-IDF with Reciprocal Rank Fusion (k=60) |
| C3 — SPL Label Grounding Module | FDA Structured Product Label linkage |
| C4 — Graded Safety Re-ranker | Penalty schema: mild=0.10 / moderate=0.30 / major=0.80 |
| C5 — Evidence-Grounded Response Generator | Structured citations, zero-hallucination (SCI-CHECK) |

---

## Key Results

> **Note:** All experiments were conducted on **SYNTHETIC_DATA** (60 products, 100 queries, 100 safety pairs). A Phase 2 real-data validation roadmap via ShoppingComp and DrugBench is described in the paper.

| Metric | Value | Hypothesis |
|---|---|---|
| NDCG@10 (MedShop-RAG) | 0.7514 | H1: NOT_SUPPORTED |
| Safety Sensitivity | 0.3469 | H2: NOT_SUPPORTED |
| Safety Precision | 0.9444 | — |
| Safety F1 | 0.5075 | — |
| McNemar χ² | 12.5 (p = 0.0004) | — |
| Citation Accuracy (SCI-CHECK) | 1.0 (κ = 1.0) | H3: SUPPORTED |

The bottleneck for low sensitivity is **CONDITION_RISK_MAP coverage**, not penalty calibration — confirmed by penalty sensitivity analysis (Conservative / Current / Aggressive regimes all yield identical sensitivity = 0.3469).

---

## Repository Structure

```
medshop-rag/
├── README.md                             ← this file
├── paper/
│   ├── medshop_rag_ieee.tex              ← IEEE LaTeX source (IEEEtran conference)
│   ├── medshop_rag_ieee.pdf              ← compiled PDF (6 pages)
│   ├── Makefile                          ← pdflatex ×2 build
│   └── README.md                         ← build instructions + pre-submission checklist
├── results/
│   ├── RESULT_MANIFEST.json              ← master manifest (all locked experimental values)
│   ├── e1_ndcg_results.json              ← Experiment 1: NDCG@10 ranking evaluation
│   ├── e2_safety_results.json            ← Experiment 2: Safety detection (sensitivity/precision/F1)
│   ├── e3_citation_results.json          ← Experiment 3: Citation accuracy (SCI-CHECK)
│   └── penalty_sensitivity_results.json  ← Stage 8: Penalty sensitivity analysis
├── submission/
│   ├── MedShopRAG_CoverLetter_ISIReCom2026.docx  ← camera-ready cover letter
│   └── MedShopRAG_ISIReCom2026_Submission.zip    ← full submission package
└── docs/
    ├── MedShopRAG_Stage6_Manuscript.docx       ← manuscript (Word)
    ├── MedShopRAG_Stage7_HostileReview.docx    ← hostile peer review report
    ├── MedShopRAG_Stage8_AuthorResponse.docx   ← author response to reviewers
    └── MedShopRAG_Stage9_EB1A_Portfolio.docx   ← EB-1A extraordinary ability portfolio
```

---

## Build the Paper

```bash
cd paper/
make        # runs pdflatex twice; output: medshop_rag_ieee.pdf
```

**Requirements:** `pdflatex`, `IEEEtran.cls` (TeX Live / MiKTeX), `tikz`, `booktabs`, `hyperref`.

---

## Integrity & Reproducibility

- All hypotheses (H1–H3) were **pre-registered** before any experiment was run.
- Results are locked in `results/RESULT_MANIFEST.json` — the single source of truth.
- All data is **SYNTHETIC_DATA** (flagged throughout the paper with `\synflag`).
- No post-hoc hypothesis adjustment was made; two NOT_SUPPORTED outcomes are reported as-is.

---

## Citation

```bibtex
@inproceedings{medshoprag2026,
  title     = {{MedShop-RAG}: Safety-Aware Retrieval-Augmented Product Search
               for {OTC} Healthcare {E-Commerce}},
  booktitle = {Proceedings of the 2nd Workshop on Information Search and Retrieval
               for E-Commerce (ISIR-eCom 2026), co-located with IEEE BigData 2026},
  year      = {2026},
  note      = {SYNTHETIC\_DATA evaluation; Phase 2 real-data validation planned}
}
```

---

## License

Research artifact. All experimental data is synthetic. See manuscript for full methodology.

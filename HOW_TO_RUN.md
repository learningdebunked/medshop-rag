# How to Run MedShop-RAG

Complete guide for running the MedShop-RAG pipeline with **synthetic data** (out-of-the-box)
and with **real data** (ShoppingComp + DrugBench).

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Repository Setup](#2-repository-setup)
3. [Project Structure](#3-project-structure)
4. [Running with Synthetic Data](#4-running-with-synthetic-data)
   - [Step 1 — Generate synthetic data](#step-1--generate-synthetic-data)
   - [Step 2 — Run Experiment E1: Retrieval Relevance (NDCG@10)](#step-2--run-experiment-e1-retrieval-relevance-ndcg10)
   - [Step 3 — Run Experiment E2: Safety Detection](#step-3--run-experiment-e2-safety-detection)
   - [Step 4 — Run Experiment E3: Citation Accuracy (SCI-CHECK)](#step-4--run-experiment-e3-citation-accuracy-sci-check)
   - [Step 5 — Run Penalty Sensitivity Analysis](#step-5--run-penalty-sensitivity-analysis)
   - [Step 6 — Run All Experiments at Once](#step-6--run-all-experiments-at-once)
5. [Running with Real Data](#5-running-with-real-data)
   - [Real Data Sources](#real-data-sources)
   - [Data Format Specification](#data-format-specification)
   - [Replacing Synthetic with Real Data](#replacing-synthetic-with-real-data)
6. [Running the End-to-End Pipeline](#6-running-the-end-to-end-pipeline)
7. [Understanding Outputs](#7-understanding-outputs)
8. [Known Limitations](#8-known-limitations)

---

## 1. Prerequisites

**Python:** 3.9 or higher

**Install dependencies:**

```bash
pip install -r requirements.txt
```

`requirements.txt` installs:

| Package | Version | Used for |
|---|---|---|
| `numpy` | ≥ 1.24 | TF-IDF vectorization, NDCG computation |
| `scipy` | ≥ 1.10 | Wilcoxon signed-rank, McNemar test |
| `scikit-learn` | ≥ 1.3 | Optional; not imported in core pipeline (pure NumPy TF-IDF used) |
| `rank_bm25` | ≥ 0.2.2 | BM25Okapi retriever (C2) |

> **Note:** `sentence-transformers` is **not** required. The C2 Catalog Retriever uses
> a pure-NumPy TF-IDF cosine similarity as a bi-encoder substitute
> (`TFIDF_SUBSTITUTE_FOR_BIENCODER`). Replace with a real bi-encoder for production use.

---

## 2. Repository Setup

```bash
git clone https://github.com/learningdebunked/medshop-rag.git
cd medshop-rag
pip install -r requirements.txt

# Create the results directory (experiments write here)
mkdir -p src/results
```

---

## 3. Project Structure

```
medshop-rag/
├── src/
│   ├── pipeline/               ← 5 pipeline components
│   │   ├── c1_intent_classifier.py   C1: OTC Intent Classifier (rule-based)
│   │   ├── c2_catalog_retriever.py   C2: Hybrid BM25 + TF-IDF retriever (RRF k=60)
│   │   ├── c3_label_grounding.py     C3: SPL Label Grounding Module
│   │   ├── c4_safety_reranker.py     C4: Graded Safety Re-ranker (penalty schema)
│   │   ├── c5_response_generator.py  C5: Evidence-Grounded Response Generator
│   │   └── stats_utils.py            Statistical helpers (Wilcoxon, McNemar, NDCG)
│   ├── experiments/            ← experiment runners
│   │   ├── e1_ndcg_experiment.py     E1: Retrieval relevance (NDCG@10)
│   │   ├── e2_safety_detection.py    E2: Major contraindication detection
│   │   ├── e3_citation_accuracy.py   E3: SCI-CHECK citation accuracy
│   │   ├── f32_overlap_check.py      F32: Train/test overlap check
│   │   ├── f35_openfda_coverage.py   F35: OpenFDA label coverage
│   │   └── penalty_sensitivity.py   Penalty sensitivity analysis (Stage 8)
│   └── data/                   ← data files live here
│       ├── generate_data.py          Synthetic data generator
│       ├── products_catalog.json     60 OTC products (generated or real)
│       ├── queries_relevance.json    100 queries + relevance labels (generated or real)
│       └── safety_pairs.json         100 safety pairs (generated or real)
├── results/                    ← locked results from original runs
├── paper/                      ← IEEE LaTeX source + PDF
├── submission/                 ← cover letter + submission ZIP
└── docs/                       ← stage milestone documents
```

---

## 4. Running with Synthetic Data

All experiments are self-contained and run directly on synthetic data.
No external APIs, credentials, or downloads required.

### Step 1 — Generate synthetic data

```bash
cd src/data
python generate_data.py
```

This writes three files to `src/data/`:

| File | Contents |
|---|---|
| `products_catalog.json` | 60 synthetic OTC products with names, active ingredients, warnings, indications |
| `queries_relevance.json` | 100 synthetic queries with pre-assigned relevance labels |
| `safety_pairs.json` | 100 synthetic safety pairs (product × patient condition × ground-truth label) |

> **SYNTHETIC_DATA flag:** All data is simulated. Products are modelled after
> real OTC categories (pain, allergy, digestive, sleep, etc.) but contain no
> real patient or clinical records.

---

### Step 2 — Run Experiment E1: Retrieval Relevance (NDCG@10)

**Research Question:** Does MedShop-RAG's safety-aware re-ranking improve retrieval
relevance (NDCG@10) vs BM25-only and TF-IDF-only baselines?

```bash
cd src/experiments
python e1_ndcg_experiment.py
```

**What it does:**
- Loads `products_catalog.json` and `queries_relevance.json` from `../data/`
- Runs C1 (OTC intent gate) → C2 (BM25 + TF-IDF retrieval) → C4 (safety re-ranking)
- Computes NDCG@10 per query for MedShop-RAG, BM25-only, and TF-IDF-only
- Runs paired Wilcoxon signed-rank test (Bonferroni α_adj = 0.025)

**Output:** `../results/e1_ndcg_results.json`

**Expected results (from locked synthetic run):**

| System | NDCG@10 |
|---|---|
| MedShop-RAG | 0.7514 |
| BM25-only | 0.7528 |
| TF-IDF-only | 0.7575 |
| H1 verdict | NOT_SUPPORTED (p = 0.888 vs BM25) |

---

### Step 3 — Run Experiment E2: Safety Detection

**Research Question:** Does MedShop-RAG detect major contraindications with
sensitivity ≥ 0.80 (H2)?

```bash
cd src/experiments
python e2_safety_detection.py
```

**What it does:**
- Loads `products_catalog.json` and `safety_pairs.json` from `../data/`
- Runs C3 (SPL label grounding) → C4 (safety penalty computation)
- Compares MedShop-RAG vs a no-safety baseline
- Computes sensitivity, specificity, precision, F1, TP/FP/FN/TN
- Runs McNemar's chi-square test (α = 0.05)

**Output:** `../results/e2_safety_results.json`

**Expected results (from locked synthetic run):**

| Metric | Value |
|---|---|
| Sensitivity | 0.3469 |
| Specificity | 0.9804 |
| Precision | 0.9444 |
| F1 | 0.5075 |
| McNemar χ² | 12.5 (p = 0.0004) |
| H2 verdict | NOT_SUPPORTED (sensitivity < 0.80 threshold) |

> **Bottleneck:** Low sensitivity is caused by incomplete `CONDITION_RISK_MAP` coverage
> in C4, not by penalty calibration. See `c4_safety_reranker.py` to extend the map.

---

### Step 4 — Run Experiment E3: Citation Accuracy (SCI-CHECK)

**Research Question:** Does the C5 response generator produce complete, SPL-grounded
citations (accuracy ≥ 80%, κ ≥ 0.70)?

```bash
cd src/experiments
python e3_citation_accuracy.py
```

**What it does:**
- Runs the full pipeline (C1 → C2 → C3 → C4 → C5) on all 100 queries
- Compares automated citation checker (Rater 1) vs reference labeller (Rater 2)
- Computes citation accuracy, Cohen's κ, bootstrap 95% CI (10,000 resamples)

**Output:** `../results/e3_citation_results.json`

**Expected results (from locked synthetic run):**

| Metric | Value |
|---|---|
| Citation Accuracy | 1.0 |
| Cohen's κ | 1.0 |
| H3 verdict | SUPPORTED |

> **Note:** Perfect accuracy (1.0) reflects the template-based MOCK_LLM in C5.
> A live LLM will produce lower (more realistic) citation fidelity.

---

### Step 5 — Run Penalty Sensitivity Analysis

Tests whether Conservative / Current / Aggressive penalty regimes change safety outcomes.

```bash
cd src/experiments
python penalty_sensitivity.py
```

**Output:** `penalty_sensitivity_results.json` (written to the current directory)

**Key finding:** All three regimes yield identical sensitivity = 0.3469 and F1 = 0.5075.
The bottleneck is `CONDITION_RISK_MAP` coverage, not penalty magnitude.

---

### Step 6 — Run All Experiments at Once

```bash
cd src/data && python generate_data.py
cd ../experiments
python e1_ndcg_experiment.py
python e2_safety_detection.py
python e3_citation_accuracy.py
python penalty_sensitivity.py
```

Or as a one-liner from the repo root:

```bash
cd src/data && python generate_data.py && \
cd ../experiments && \
python e1_ndcg_experiment.py && \
python e2_safety_detection.py && \
python e3_citation_accuracy.py && \
python penalty_sensitivity.py && \
echo "All experiments complete."
```

Results are written to `src/results/`.

---

## 5. Running with Real Data

> **Current status:** Experiments E1 and E2 are designed for real data from
> **ShoppingComp** (Tou et al., 2025) and **DrugBench** (Freire et al., 2026).
> Real datasets were not available at submission time; all published results
> are on `SYNTHETIC_DATA`.

### Real Data Sources

| Dataset | Use | Access |
|---|---|---|
| **ShoppingComp** | E1 — product catalog + queries + relevance labels | https://github.com/amazon-science/ShoppingBench |
| **DrugBench** | E2 — safety pairs (drug × condition × contraindication label) | Contact Freire et al. (2026) |
| **OpenFDA SPL** | C3 label grounding | https://api.fda.gov/drug/label.json (public API, no key required) |

### Data Format Specification

To swap in real data, replace the three files in `src/data/` with files
matching these schemas:

#### `products_catalog.json`

```json
[
  {
    "product_id": "P001",
    "name": "Tylenol Extra Strength",
    "category": "pain_fever",
    "active_ingredient": "acetaminophen",
    "warnings_keywords": [
      "liver warning",
      "do not use with other acetaminophen",
      "alcohol"
    ],
    "indications": "pain relief fever"
  }
]
```

- `product_id`: unique string identifier
- `warnings_keywords`: list of free-text warning phrases (matched against `CONDITION_RISK_MAP` in C4)
- `indications`: free-text string used for BM25/TF-IDF indexing

#### `queries_relevance.json`

```json
[
  {
    "query_id": "Q001",
    "query_text": "best pain reliever for headache",
    "relevant_product_ids": ["P001", "P002", "P003"]
  }
]
```

- `relevant_product_ids`: list of product IDs considered relevant (used to compute NDCG@10)
- Compatible with ShoppingComp format after column mapping

#### `safety_pairs.json`

```json
[
  {
    "pair_id": "SP001",
    "product_id": "P010",
    "patient_condition": "taking MAO inhibitor",
    "is_major_contraindication": true
  }
]
```

- `is_major_contraindication`: boolean ground truth label
- Compatible with DrugBench contraindication pairs after column mapping

### Replacing Synthetic with Real Data

1. **Map your real data to the schema above** (column rename only — no preprocessing required).

2. **Place files in `src/data/`:**

   ```bash
   cp your_products.json   src/data/products_catalog.json
   cp your_queries.json    src/data/queries_relevance.json
   cp your_safety.json     src/data/safety_pairs.json
   ```

3. **Extend `CONDITION_RISK_MAP` in `src/pipeline/c4_safety_reranker.py`** to cover
   your real condition vocabulary. The current map covers the synthetic dataset's
   ~40 condition tokens. Real DrugBench data will require broader coverage — this is
   the primary driver of sensitivity improvement (confirmed by penalty sensitivity analysis).

4. **Re-run experiments** (same commands as synthetic):

   ```bash
   cd src/experiments
   python e1_ndcg_experiment.py
   python e2_safety_detection.py
   python e3_citation_accuracy.py
   ```

5. **For live OpenFDA label grounding** (C3), set `USE_OPENFDA=True` in
   `src/pipeline/c3_label_grounding.py`. The module includes an
   `fetch_openfda_label()` function that queries `api.fda.gov` by active ingredient.
   No API key is required. Rate limit: 240 requests/minute (unauthenticated).

---

## 6. Running the End-to-End Pipeline

To run the full MedShop-RAG pipeline on a single query:

```python
import sys
sys.path.insert(0, "src/pipeline")
sys.path.insert(0, "src/data")

import json
from c1_intent_classifier import classify_otc_intent
from c2_catalog_retriever import CatalogRetriever
from c3_label_grounding import build_product_index
from c4_safety_reranker import rerank
from c5_response_generator import generate_response

# Load data
with open("src/data/products_catalog.json") as f:
    products = json.load(f)

# Initialize components
retriever     = CatalogRetriever(products)
product_index = build_product_index(products)

# Run pipeline
query = "pain reliever safe for someone on warfarin"

# C1 — OTC intent gate
intent = classify_otc_intent(query)
if not intent["is_otc_intent"]:
    print("Query not classified as OTC healthcare intent.")
else:
    # C2 — Retrieve top-20 candidates
    candidates = retriever.retrieve(query, top_k=20)

    # C3+C4 — Safety re-rank with patient condition
    patient_condition = "taking warfarin blood thinner"
    reranked = rerank(candidates, product_index, patient_condition)

    # C5 — Generate evidence-grounded response
    response = generate_response(query, reranked, product_index, top_n=3)
    print(response["response_text"])
    print("\nCitations:")
    for c in response["citations"]:
        print(" ", c)
```

---

## 7. Understanding Outputs

### Experiment result JSON structure

All experiment results follow this pattern:

```json
{
  "SIMULATION_FLAG": "SYNTHETIC_DATA",
  "hypothesis": "H1",
  "verdict": "NOT_SUPPORTED",
  "metrics": { ... },
  "statistical_test": { ... },
  "per_query_details": [ ... ]
}
```

### Key metric files

| File | Key fields |
|---|---|
| `results/e1_ndcg_results.json` | `ndcg_medshop`, `ndcg_bm25`, `ndcg_tfidf`, `wilcoxon_p` |
| `results/e2_safety_results.json` | `sensitivity`, `specificity`, `precision`, `f1`, `mcnemar_chi2`, `mcnemar_p` |
| `results/e3_citation_results.json` | `citation_accuracy`, `cohens_kappa`, `bootstrap_ci_95` |
| `results/RESULT_MANIFEST.json` | Master manifest — all locked values from original paper runs |

---

## 8. Known Limitations

| Limitation | Details | Planned fix |
|---|---|---|
| **SYNTHETIC_DATA only** | All published results are on 60 products / 100 queries / 100 safety pairs | Phase 2: ShoppingComp + DrugBench |
| **Low safety sensitivity (0.3469)** | `CONDITION_RISK_MAP` in C4 covers ~40 condition tokens; misses ~32/49 major pairs | Expand map or replace C4 with neural safety classifier |
| **MOCK_LLM in C5** | Template-based response generator; no GPT-4o or real LLM | Replace `generate_response()` with LLM API call |
| **TF-IDF as bi-encoder substitute** | `sentence-transformers` unavailable in sandbox; TF-IDF cosine used instead | Swap in `all-MiniLM-L6-v2` or similar |
| **OpenFDA coverage 64%** | 32/50 products matched via active ingredient lookup | Extend matching to brand name + NDC code fallback |

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

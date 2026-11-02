"""
e3_citation_accuracy.py  –  E3: RQ3 Citation Accuracy & Agreement
Bootstrap CI (10,000 resamples); Cohen's kappa ≥ 0.70 pre-condition (locked Stage 4)
SIMULATION_FLAG: SYNTHETIC_DATA | MOCK_LLM
"""
import json, sys, os
sys.path.insert(0, os.path.dirname(__file__) + "/../pipeline")
from c2_catalog_retriever import CatalogRetriever
from c3_label_grounding import build_product_index
from c4_safety_reranker import rerank
from c5_response_generator import generate_response
from stats_utils import cohens_kappa, bootstrap_ci, precision_recall_f1
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "../data")
N_BOOTSTRAP = 10000
CITATION_ACCURACY_THRESHOLD = 0.80  # H3 criterion (locked)
KAPPA_THRESHOLD = 0.70              # H3 kappa pre-condition (locked)

def check_citation_complete(citation: str, product_name: str,
                             valid_sections: list, has_evidence: bool) -> bool:
    """
    Automated citation completeness check (Rater 1 / automated).
    All three fields must be present and non-trivial.
    """
    has_product  = f"[Product: {product_name}]" in citation
    has_spl      = any(f"[SPL: {s}]" in citation for s in valid_sections)
    has_ev       = "[Evidence:" in citation
    ev_content   = citation.split("[Evidence:")[-1].replace("]","").strip() if "[Evidence:" in citation else ""
    ev_nontrivial = len(ev_content) >= 3
    return has_product and has_spl and has_ev and ev_nontrivial

def reference_citation_label(product_id: str, product_index: dict) -> bool:
    """
    Reference labeller (Rater 2): Citation should be complete if the product
    has at least one safety attribute OR has non-empty indications text.
    Models a conservative reference rater.
    """
    spl = product_index.get(product_id, {})
    has_attrs = len(spl.get("safety_attrs", [])) > 0
    has_ind   = len(spl.get("indications", "").strip()) > 3
    return has_attrs or has_ind

VALID_SECTIONS = [
    "WARNINGS AND PRECAUTIONS",
    "INDICATIONS AND USAGE",
    "CONTRAINDICATIONS",
    "DRUG INTERACTIONS",
    "DO NOT USE",
]

def run_e3():
    with open(os.path.join(DATA_DIR, "products_catalog.json")) as f:
        products = json.load(f)
    with open(os.path.join(DATA_DIR, "queries_relevance.json")) as f:
        queries = json.load(f)

    retriever = CatalogRetriever(products)
    product_index = build_product_index(products)
    pid_to_product = {p["product_id"]: p for p in products}

    citation_scores = []
    rater1_binary   = []  # automated check
    rater2_binary   = []  # reference rater
    query_details   = []

    for q in queries:
        qtext = q["query_text"]

        hybrid_results = retriever.retrieve(qtext, top_k=20)
        reranked       = rerank(hybrid_results, product_index, patient_condition="", top_k=3)
        response       = generate_response(qtext, reranked, product_index, top_n=3)

        for i, (cit, item) in enumerate(zip(response["citations"], reranked)):
            pid  = item["product_id"]
            pname = pid_to_product.get(pid, {}).get("name", "")
            
            # Rater 1: automated check
            r1 = int(check_citation_complete(cit, pname, VALID_SECTIONS, True))
            # Rater 2: reference label
            r2 = int(reference_citation_label(pid, product_index))
            
            citation_scores.append(r1)
            rater1_binary.append(r1)
            rater2_binary.append(r2)

            query_details.append({
                "query_id":     q["query_id"],
                "product_id":   pid,
                "citation_complete_r1": bool(r1),
                "reference_label_r2":   bool(r2),
                "citation_snippet":     cit[:80],
            })

    n_citations = len(citation_scores)
    citation_scores_arr = np.array(citation_scores, dtype=float)

    # Point estimate + Bootstrap CI
    point_est, ci_lo, ci_hi = bootstrap_ci(
        citation_scores, statistic_fn=np.mean,
        n_resamples=N_BOOTSTRAP, confidence=0.95, seed=42
    )

    # Cohen's kappa (rater1 vs rater2)
    kappa = cohens_kappa(rater2_binary, rater1_binary)
    kappa_precondition_met = kappa >= KAPPA_THRESHOLD

    # H3 verdict
    h3_accuracy = point_est >= CITATION_ACCURACY_THRESHOLD
    h3_supported = h3_accuracy and kappa_precondition_met

    result = {
        "experiment": "E3_Citation_Accuracy",
        "SIMULATION_FLAG": "SYNTHETIC_DATA",
        "MOCK_LLM": True,
        "n_citations_evaluated": n_citations,
        "n_bootstrap_resamples": N_BOOTSTRAP,
        "citation_accuracy_threshold_H3": CITATION_ACCURACY_THRESHOLD,
        "kappa_threshold_H3": KAPPA_THRESHOLD,
        "citation_accuracy": {
            "point_estimate": round(point_est, 4),
            "ci_lower_95":    round(ci_lo, 4),
            "ci_upper_95":    round(ci_hi, 4),
        },
        "cohens_kappa": round(kappa, 4),
        "kappa_precondition_met": bool(kappa_precondition_met),
        "H3_verdict": "SUPPORTED" if h3_supported else
                      ("ACCURACY_MET_KAPPA_FAILED" if h3_accuracy else
                       ("KAPPA_MET_ACCURACY_FAILED" if kappa_precondition_met else "NOT_SUPPORTED")),
        "H3_note": ("MOCK_LLM: template-based citations. Real LLM evaluation required. "
                    "Results on SYNTHETIC_DATA only."),
        "per_citation": query_details,
    }
    return result

if __name__ == "__main__":
    result = run_e3()
    ca = result["citation_accuracy"]
    print(f"E3: n={result['n_citations_evaluated']} citations")
    print(f"  Citation accuracy: {ca['point_estimate']} [95% CI: {ca['ci_lower_95']}, {ca['ci_upper_95']}]")
    print(f"  Cohen's kappa: {result['cohens_kappa']} (precondition met: {result['kappa_precondition_met']})")
    print(f"  H3: {result['H3_verdict']}")

    os.makedirs("../results", exist_ok=True)
    with open("../results/e3_citation_results.json","w") as f:
        json.dump(result, f, indent=2)
    print("Saved ../results/e3_citation_results.json")

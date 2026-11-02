"""
e2_safety_detection.py  –  E2: RQ2 Major Contraindication Detection
McNemar chi-square α=0.05 (locked Stage 4)
SIMULATION_FLAG: SYNTHETIC_DATA — DrugBench not available; real results on synthetic safety pairs
"""
import json, sys, os
sys.path.insert(0, os.path.dirname(__file__) + "/../pipeline")
from c3_label_grounding import build_product_index, classify_severity, MAJOR_TRIGGERS
from c4_safety_reranker import compute_safety_penalty, SEVERITY_PENALTY
from stats_utils import mcnemar_test, precision_recall_f1

DATA_DIR = os.path.join(os.path.dirname(__file__), "../data")
SENSITIVITY_THRESHOLD = 0.80  # H2 criterion (locked Stage 3)

def detect_major_medshop(product: dict, spl: dict, patient_condition: str) -> bool:
    """
    MedShop-RAG detection: uses C3 SPL safety dict + C4 severity-adjusted penalty.
    Flags major contraindication if computed penalty >= SEVERITY_PENALTY['major'].
    """
    safety_attrs = spl.get("safety_attrs", [])
    penalty = compute_safety_penalty(safety_attrs, patient_condition)
    return penalty >= SEVERITY_PENALTY["major"]  # 0.80

def detect_major_baseline(product: dict, patient_condition: str) -> bool:
    """
    Baseline (no safety system): never flags a major contraindication.
    Models conventional e-commerce search that ignores safety.
    """
    return False  # Baseline never flags safety

def run_e2():
    with open(os.path.join(DATA_DIR, "products_catalog.json")) as f:
        products = json.load(f)
    with open(os.path.join(DATA_DIR, "safety_pairs.json")) as f:
        safety_pairs = json.load(f)

    pid_to_product = {p["product_id"]: p for p in products}
    product_index = build_product_index(products)

    y_true       = []
    y_medshop    = []
    y_baseline   = []
    pair_details = []

    for sp in safety_pairs:
        pid    = sp["product_id"]
        cond   = sp["patient_condition"]
        actual = int(sp["is_major_contraindication"])
        
        product = pid_to_product.get(pid, {})
        spl     = product_index.get(pid, {})
        
        pred_medshop  = int(detect_major_medshop(product, spl, cond))
        pred_baseline = int(detect_major_baseline(product, cond))
        
        y_true.append(actual)
        y_medshop.append(pred_medshop)
        y_baseline.append(pred_baseline)
        
        pair_details.append({
            "pair_id":       sp["pair_id"],
            "product_id":    pid,
            "condition":     cond[:50],
            "true_major":    bool(actual),
            "medshop_flag":  bool(pred_medshop),
            "baseline_flag": bool(pred_baseline),
            "severity_label": sp["severity_label"],
        })

    # Metrics for MedShop-RAG
    med_metrics  = precision_recall_f1(y_true, y_medshop)
    base_metrics = precision_recall_f1(y_true, y_baseline)

    # McNemar table: compare MedShop-RAG vs baseline
    # b = pairs where MedShop correct, baseline wrong
    # c = pairs where baseline correct, MedShop wrong
    import numpy as np
    y_t  = np.array(y_true)
    y_m  = np.array(y_medshop)
    y_b  = np.array(y_baseline)

    correct_med  = (y_m == y_t).astype(int)
    correct_base = (y_b == y_t).astype(int)
    b = int(np.sum((correct_med == 1) & (correct_base == 0)))
    c = int(np.sum((correct_med == 0) & (correct_base == 1)))
    
    chi2, p_mcnemar = mcnemar_test(b, c, continuity=True)
    significant = p_mcnemar < 0.05

    # H2 verdict: sensitivity >= 0.80
    h2_supported = med_metrics["sensitivity"] >= SENSITIVITY_THRESHOLD

    # Count confusion matrix
    n_pairs = len(y_true)
    n_major = int(sum(y_true))
    n_non_major = n_pairs - n_major

    result = {
        "experiment": "E2_Safety_Detection",
        "SIMULATION_FLAG": "SYNTHETIC_DATA",
        "n_pairs": n_pairs,
        "n_major_true": n_major,
        "n_non_major_true": n_non_major,
        "sensitivity_threshold_H2": SENSITIVITY_THRESHOLD,
        "medshoprag_metrics": med_metrics,
        "baseline_metrics": base_metrics,
        "mcnemar": {
            "b_medshop_correct_baseline_wrong": b,
            "c_baseline_correct_medshop_wrong": c,
            "chi2_statistic": round(chi2, 4),
            "p_value": round(p_mcnemar, 4),
            "significant_at_0_05": bool(significant),
            "continuity_correction": True,
        },
        "H2_verdict": "SUPPORTED" if h2_supported else "NOT_SUPPORTED",
        "H2_sensitivity_achieved": med_metrics["sensitivity"],
        "H2_note": ("Real DrugBench evaluation required for definitive conclusion. "
                    "Results on SYNTHETIC_DATA only."),
        "per_pair": pair_details,
    }
    return result

if __name__ == "__main__":
    result = run_e2()
    print(f"E2: n={result['n_pairs']} pairs ({result['n_major_true']} major, {result['n_non_major_true']} non-major)")
    m = result["medshoprag_metrics"]
    print(f"  MedShop-RAG: sens={m['sensitivity']}, spec={m['specificity']}, prec={m['precision']}, F1={m['f1']}")
    b_m = result["baseline_metrics"]
    print(f"  Baseline:    sens={b_m['sensitivity']}, spec={b_m['specificity']}")
    mn = result["mcnemar"]
    print(f"  McNemar: b={mn['b_medshop_correct_baseline_wrong']}, c={mn['c_baseline_correct_medshop_wrong']}, chi2={mn['chi2_statistic']}, p={mn['p_value']}, sig={mn['significant_at_0_05']}")
    print(f"  H2: {result['H2_verdict']} (sensitivity={result['H2_sensitivity_achieved']})")
    
    os.makedirs("../results", exist_ok=True)
    with open("../results/e2_safety_results.json","w") as f:
        json.dump(result, f, indent=2)
    print("Saved ../results/e2_safety_results.json")

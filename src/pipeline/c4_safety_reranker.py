"""
c4_safety_reranker.py  –  MedShop-RAG Component C4: Safety Re-ranker
Severity-adjusted scoring per Stage 4 locked parameters:
  mild=0.10, moderate=0.30, major=0.80 penalty; alpha=1.0
Patient condition matching uses expanded semantic keyword mapping (rule-based).
SIMULATION_FLAG: SYNTHETIC_DATA
"""

import json, re
from typing import List, Dict

# Locked penalty parameters (Stage 4 F29/F30)
SEVERITY_PENALTY = {"none": 0.0, "mild": 0.10, "moderate": 0.30, "major": 0.80}
ALPHA = 1.0  # safety weight (locked)
RRF_K_DEFAULT = 60

# ── Semantic condition → risk keyword expansion ────────────────────────────
# Maps patient condition tokens to product warning keywords they imply.
# Rule-based; no ML. Covers the synthetic safety pair vocabulary.
CONDITION_RISK_MAP = {
    # MAO inhibitors
    "mao":           ["mao inhibitor", "maoi", "linezolid"],
    "inhibitor":     ["mao inhibitor", "maoi"],
    "maoi":          ["mao inhibitor", "maoi"],
    "linezolid":     ["linezolid", "mao inhibitor"],
    # Liver
    "liver":         ["liver warning", "liver", "acetaminophen"],
    # Anticoagulants / blood thinners / warfarin
    "warfarin":      ["warfarin", "anticoagulants", "stomach bleeding", "increased bleeding"],
    "anticoagulant": ["anticoagulants", "stomach bleeding", "increased bleeding", "warfarin"],
    "anticoagulants":["anticoagulants", "stomach bleeding"],
    "thinner":       ["anticoagulants", "stomach bleeding"],
    "thinners":      ["anticoagulants", "stomach bleeding"],
    "clopidogrel":   ["anticoagulants", "stomach bleeding"],
    "antiplatelet":  ["anticoagulants", "stomach bleeding"],
    # Reye syndrome / children
    "reye":          ["reye syndrome"],
    "child":         ["reye syndrome children", "children under"],
    "children":      ["reye syndrome children", "children under"],
    # Cyclosporine / transplant
    "cyclosporine":  ["cyclosporine", "organ transplant cyclosporine major"],
    "transplant":    ["cyclosporine", "organ transplant"],
    # Alcohol
    "alcohol":       ["alcohol", "alcohol major"],
    # Blood pressure / hypertension
    "hypertension":  ["high blood pressure"],
    "pressure":      ["high blood pressure"],
    "hypertensive":  ["high blood pressure"],
    # SSRI / antidepressant / serotonin
    "ssri":          ["ssri", "do not use with ssris"],
    "antidepressant":["ssri", "mao inhibitor"],
    "serotonin":     ["ssri", "mao inhibitor"],
    # Surgery
    "surgery":       ["surgery avoid", "surgery avoid 2 weeks"],
    "surgical":      ["surgery avoid"],
    # CNS depressant
    "cns":           ["cns depressants additive"],
    "depressant":    ["cns depressants additive"],
    # G6PD
    "g6pd":          ["g6pd", "anemia g6pd"],
    # Diabetes
    "diabetes":      ["diabetics avoid", "diabetes"],
    "diabetic":      ["diabetics avoid", "diabetes"],
    # Heart / cardiac
    "heart":         ["heart risk", "heart disease", "cardiovascular"],
    "cardiac":       ["heart risk", "cardiovascular"],
    "failure":       ["heart risk", "heart disease"],
    # Gallbladder
    "gallbladder":   ["gallbladder disease"],
    # Elderly / fall
    "elderly":       ["elderly caution"],
    "fall":          ["elderly caution"],
    # BPH / urinary
    "bph":           ["urinary retention"],
    "urinary":       ["urinary retention"],
    "retention":     ["urinary retention"],
    # Autoimmune
    "autoimmune":    ["autoimmune consult", "autoimmune disease"],
    # Ectopic pregnancy
    "ectopic":       ["ectopic pregnancy risk if fails", "ectopic pregnancy"],
    "pregnancy":     ["ectopic pregnancy risk", "pregnancy consult"],
    # Glaucoma
    "glaucoma":      ["glaucoma", "eye pressure glaucoma"],
    # Peptic ulcer
    "peptic":        ["stomach bleeding"],
    "ulcer":         ["stomach bleeding"],
    # Nasal septum
    "nasal":         ["nasal injury"],
    "septal":        ["nasal injury"],
    "septum":        ["nasal injury"],
    # Fish allergy
    "fish":          ["allergy to fish"],
    # Aminoglycoside
    "aminoglycoside":["allergy to aminoglycosides"],
    # NSAIDs allergy
    "nsaid":         ["nsaids allergy", "aspirin allergy"],
    # Peripheral vascular
    "peripheral":    ["peripheral vascular disease"],
    "vascular":      ["peripheral vascular disease"],
    # Hemochromatosis
    "hemochromatosis":["hemochromatosis avoid high iron"],
    "iron":          ["hemochromatosis avoid high iron"],
    # Kidney / renal
    "kidney":        ["kidney disease avoid", "kidney disease consult", "kidney"],
    "renal":         ["kidney disease avoid", "kidney"],
    # Thyroid
    "thyroid":       ["thyroid"],
    # Driving
    "driving":       ["do not drive", "drowsiness"],
    "operator":      ["do not drive", "drowsiness"],
    "machinery":     ["do not drive", "drowsiness"],
}

def _expand_condition_keywords(patient_condition: str) -> set:
    """Expand patient condition to all relevant risk keywords via semantic map."""
    tokens = re.findall(r"[a-z0-9]+", patient_condition.lower())
    expanded = set(tokens)  # include original tokens
    for tok in tokens:
        if tok in CONDITION_RISK_MAP:
            expanded.update(CONDITION_RISK_MAP[tok])
    return expanded

def compute_safety_penalty(safety_attrs: list, patient_condition: str = "") -> float:
    """
    Compute aggregate safety penalty for a product given patient context.
    Uses the max severity encountered (conservative approach).
    
    Design rule: if patient_condition is empty, no penalty is applied — the
    safety re-ranker only activates when a specific patient condition is provided.
    This preserves NDCG evaluation validity for generic (no-context) queries (E1).
    
    Condition matching uses expanded semantic keyword mapping (rule-based).
    """
    if not safety_attrs:
        return 0.0
    if not patient_condition.strip():
        return 0.0  # No patient context → no safety re-ranking

    expanded_context = _expand_condition_keywords(patient_condition)
    
    penalties = []
    for attr in safety_attrs:
        warn_lower = attr["warning_text"].lower()
        # Check if any expanded keyword appears in the warning text
        match = any(kw in warn_lower for kw in expanded_context)
        if match:
            penalties.append(SEVERITY_PENALTY[attr["severity"]])
    
    return max(penalties) if penalties else 0.0


def rerank(retrieved: List[Dict], product_index: Dict,
           patient_condition: str = "", top_k: int = 10) -> List[Dict]:
    """
    Re-rank retrieved products using safety-adjusted score.
    final_score = retrieval_score * (1 - alpha * max_penalty)
    """
    reranked = []
    for item in retrieved:
        pid = item["product_id"]
        spl = product_index.get(pid, {})
        safety_attrs = spl.get("safety_attrs", [])
        
        penalty = compute_safety_penalty(safety_attrs, patient_condition)
        retrieval_score = item.get("rrf_score",
                                   1.0 / (item.get("rank", 1) + RRF_K_DEFAULT))
        
        final_score = retrieval_score * (1.0 - ALPHA * penalty)
        
        reranked.append({
            "product_id": pid,
            "name": item["name"],
            "retrieval_score": round(retrieval_score, 6),
            "safety_penalty": round(penalty, 2),
            "final_score": round(final_score, 6),
            "has_major": spl.get("has_major_contraindication", False),
            "major_count": spl.get("major_count", 0),
            "severity_breakdown": {
                "major": spl.get("major_count", 0),
                "moderate": spl.get("moderate_count", 0),
            },
            "SIMULATION_FLAG": "SYNTHETIC_DATA"
        })
    
    reranked.sort(key=lambda x: -x["final_score"])
    for rank, item in enumerate(reranked):
        item["final_rank"] = rank + 1
    
    return reranked[:top_k]


if __name__ == "__main__":
    import json, os, sys
    sys.path.insert(0, os.path.dirname(__file__))
    with open("../data/products_catalog.json") as f:
        products = json.load(f)
    from c3_label_grounding import build_product_index
    product_index = build_product_index(products)
    # Test
    sample = [
        {"product_id":"P010","name":"Robitussin DM","rank":1,"rrf_score":0.030},
        {"product_id":"P011","name":"Mucinex 600mg","rank":2,"rrf_score":0.025},
        {"product_id":"P012","name":"Nyquil Cold Flu","rank":3,"rrf_score":0.020},
    ]
    result = rerank(sample, product_index, patient_condition="patient taking MAO inhibitor")
    for r in result:
        print(f"  #{r['final_rank']} {r['name']} final={r['final_score']:.4f} penalty={r['safety_penalty']}")
    
    # Test liver disease condition
    p001 = [{"product_id":"P001","name":"Tylenol","rank":1,"rrf_score":0.030}]
    r2 = rerank(p001, product_index, patient_condition="patient with liver disease")
    print(f"  Tylenol liver disease penalty: {r2[0]['safety_penalty']}")

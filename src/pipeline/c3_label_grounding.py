"""
c3_label_grounding.py  –  MedShop-RAG Component C3: Label Grounding Module
Parses SPL sections from synthetic product records + real OpenFDA label data.
Maps warnings to severity attributes via keyword matching.
SIMULATION_FLAG: SYNTHETIC_DATA (products are synthetic; OpenFDA API real where accessible)
"""

import json, re, requests
from typing import Optional

# Severity taxonomy (aligned with DrugBench as specified in Stage 4)
MAJOR_TRIGGERS = [
    "mao inhibitor","maoi","linezolid","cyclosporine","organ transplant",
    "anticoagulant","warfarin","blood thinner","reye syndrome","reye's",
    "aspirin allergy","nsaid allergy","liver disease","end stage",
    "g6pd deficiency","ectopic pregnancy","already pregnant","confirmed pregnant",
    "operating machinery","driving","alcohol","ssri","serotonin syndrome",
    "severe hypertension","heart failure","peptic ulcer","active ulcer",
    "renal failure","end stage kidney","perforated","septal perforation",
    "surgery avoid","before surgery","fall risk elderly",
    "first-time yeast confirm","urinary retention bph",
]
MODERATE_TRIGGERS = [
    "kidney impairment","kidney disease consult","kidney stones",
    "cardiovascular risk","heart disease consult","diabetes consult",
    "thyroid","more than 14 days","more than 7 days","prolonged use",
    "antacids reduce","rebound redness","contact lenses","thiazide",
    "jaw problems","skin irritation","autoimmune consult",
]
MILD_TRIGGERS = [
    "adequate fluid","nausea if not","normal orange urine",
    "takes 1-2 weeks","takes two weeks","blackening stool normal",
    "close cap tightly","first-time symptoms",
]

def classify_severity(warning_text: str) -> str:
    wl = warning_text.lower()
    for t in MAJOR_TRIGGERS:
        if t in wl:
            return "major"
    for t in MODERATE_TRIGGERS:
        if t in wl:
            return "moderate"
    for t in MILD_TRIGGERS:
        if t in wl:
            return "mild"
    return "none"

def build_spl_safety_dict(product: dict) -> dict:
    """
    Build SPL safety attribute dict from synthetic product record.
    Returns structured safety attributes per Stage 4 C3 spec.
    """
    warnings = product.get("warnings_keywords", [])
    safety_attrs = []
    for w in warnings:
        sev = classify_severity(w)
        safety_attrs.append({
            "warning_text": w,
            "severity": sev,
            "spl_section": "WARNINGS AND PRECAUTIONS",
            "source": "SYNTHETIC_SPL"
        })
    
    major_warnings = [a for a in safety_attrs if a["severity"] == "major"]
    moderate_warnings = [a for a in safety_attrs if a["severity"] == "moderate"]
    
    return {
        "product_id": product["product_id"],
        "product_name": product["name"],
        "active_ingredient": product["active_ingredient"],
        "indications": product["indications"],
        "safety_attrs": safety_attrs,
        "major_count": len(major_warnings),
        "moderate_count": len(moderate_warnings),
        "has_major_contraindication": len(major_warnings) > 0,
        "SIMULATION_FLAG": "SYNTHETIC_DATA"
    }

def fetch_openfda_label(drug_name: str, timeout: int = 10) -> Optional[dict]:
    """Fetch real OTC label from OpenFDA (real API, not synthetic)."""
    try:
        url = "https://api.fda.gov/drug/label.json"
        params = {"search": f'brand_name:"{drug_name}"', "limit": 1}
        resp = requests.get(url, params=params, timeout=timeout)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("results"):
                return data["results"][0]
    except Exception:
        pass
    return None

def build_product_index(products: list) -> dict:
    """Build safety index for all products."""
    index = {}
    for p in products:
        spl = build_spl_safety_dict(p)
        index[p["product_id"]] = spl
    return index


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    with open("../data/products_catalog.json") as f:
        products = json.load(f)
    index = build_product_index(products)
    sample = index["P010"]
    print(f"P010 {sample['product_name']}: major={sample['major_count']}, moderate={sample['moderate_count']}")
    for sa in sample["safety_attrs"]:
        print(f"  [{sa['severity'].upper()}] {sa['warning_text']}")

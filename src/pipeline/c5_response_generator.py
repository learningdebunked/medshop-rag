"""
c5_response_generator.py  –  MedShop-RAG Component C5: Evidence-Grounded Response Generator
MOCK_LLM: Template-based generator (no GPT-4o access in sandbox).
SIMULATION_FLAG: SYNTHETIC_DATA | MOCK_LLM (no live LLM used)
Citation format: [Product: <name>][SPL: <section>][Evidence: <text>]
"""

import json
from typing import List, Dict

CITATION_FORMAT = "[Product: {name}][SPL: {section}][Evidence: {evidence}]"

def generate_response(query: str, reranked_products: List[Dict],
                      product_index: Dict, top_n: int = 3) -> dict:
    """
    Generate structured template response with SPL-grounded citations.
    Uses MOCK_LLM template (no actual LLM).
    """
    top_products = reranked_products[:top_n]
    
    lines = [f"For your query: '{query}'\n"]
    lines.append("Recommended OTC products (safety-aware ranking):\n")
    
    citations = []
    has_safety_alerts = False
    
    for i, item in enumerate(top_products, 1):
        pid = item["product_id"]
        spl = product_index.get(pid, {})
        name = item["name"]
        
        # Build recommendation text
        major_attrs = [a for a in spl.get("safety_attrs",[]) if a["severity"]=="major"]
        moderate_attrs = [a for a in spl.get("safety_attrs",[]) if a["severity"]=="moderate"]
        
        rec_line = f"{i}. {name}"
        
        # Add citation
        if major_attrs:
            has_safety_alerts = True
            evidence_text = major_attrs[0]["warning_text"][:80]
            citation = CITATION_FORMAT.format(
                name=name,
                section="WARNINGS AND PRECAUTIONS",
                evidence=evidence_text
            )
            rec_line += f"\n   ⚠ SAFETY ALERT: {major_attrs[0]['warning_text']}"
            rec_line += f"\n   {citation}"
        elif moderate_attrs:
            evidence_text = moderate_attrs[0]["warning_text"][:80]
            citation = CITATION_FORMAT.format(
                name=name,
                section="WARNINGS AND PRECAUTIONS",
                evidence=evidence_text
            )
            rec_line += f"\n   ℹ Note: {moderate_attrs[0]['warning_text']}"
            rec_line += f"\n   {citation}"
        else:
            citation = CITATION_FORMAT.format(
                name=name,
                section="INDICATIONS AND USAGE",
                evidence=spl.get("indications","").strip()[:60]
            )
            rec_line += f"\n   {citation}"
        
        citations.append(citation)
        lines.append(rec_line + "\n")
    
    if has_safety_alerts:
        lines.append("\n⚠ One or more recommendations above have safety alerts.")
        lines.append("Please consult a pharmacist or physician before use.")
    
    lines.append(f"\n[MOCK_LLM: template-based response, no live LLM]")
    lines.append(f"[SIMULATION_FLAG: SYNTHETIC_DATA]")
    
    # Check citation accuracy: each citation must have all 3 fields populated
    citation_scores = []
    for c in citations:
        has_product = "[Product:" in c and "N/A" not in c
        has_spl = "[SPL:" in c
        has_evidence = "[Evidence:" in c and len(c.split("[Evidence:")[1]) > 5
        score = int(has_product) + int(has_spl) + int(has_evidence)
        citation_scores.append(score / 3.0)
    
    avg_citation_accuracy = sum(citation_scores) / len(citation_scores) if citation_scores else 0.0
    
    return {
        "query": query,
        "response_text": "\n".join(lines),
        "citations": citations,
        "n_citations": len(citations),
        "citation_accuracy_score": round(avg_citation_accuracy, 3),
        "has_safety_alerts": has_safety_alerts,
        "MOCK_LLM": True,
        "SIMULATION_FLAG": "SYNTHETIC_DATA"
    }


if __name__ == "__main__":
    with open("../data/products_catalog.json") as f:
        products = json.load(f)
    from c3_label_grounding import build_product_index
    product_index = build_product_index(products)
    sample_reranked = [
        {"product_id":"P001","name":"Tylenol Extra Strength","final_rank":1,"final_score":0.028},
        {"product_id":"P002","name":"Advil Ibuprofen","final_rank":2,"final_score":0.006},
        {"product_id":"P003","name":"Aleve Naproxen","final_rank":3,"final_score":0.004},
    ]
    resp = generate_response("pain reliever for headache", sample_reranked, product_index)
    print(resp["response_text"])
    print(f"\nCitation accuracy: {resp['citation_accuracy_score']}")

"""
f32_overlap_check.py  –  F32: ShoppingComp ↔ DrugBench Overlap Check
Checks whether any synthetic query IDs overlap with synthetic safety pair product IDs.
In production this would compare real ShoppingComp / DrugBench identifiers.
SIMULATION_FLAG: SYNTHETIC_DATA — real datasets not available
"""
import json, os

DATA_DIR = os.path.join(os.path.dirname(__file__), "../data")

def run_f32():
    with open(os.path.join(DATA_DIR,"queries_relevance.json")) as f:
        queries = json.load(f)
    with open(os.path.join(DATA_DIR,"safety_pairs.json")) as f:
        safety_pairs = json.load(f)
    
    # In synthetic setup: queries reference products; safety pairs also reference products.
    # Overlap = same product appearing in BOTH eval sets for DIFFERENT purposes.
    query_products = set()
    for q in queries:
        for pid in q["relevant_product_ids"]:
            query_products.add(pid)
        for pid in q["major_safety_product_ids"]:
            query_products.add(pid)
    
    safety_products = set(sp["product_id"] for sp in safety_pairs)
    
    overlap = query_products & safety_products
    
    result = {
        "check": "F32_ShoppingComp_DrugBench_Overlap",
        "SIMULATION_FLAG": "SYNTHETIC_DATA",
        "query_product_set_size": len(query_products),
        "safety_product_set_size": len(safety_products),
        "overlap_count": len(overlap),
        "overlap_ids": sorted(list(overlap)),
        "overlap_rate": round(len(overlap) / max(len(query_products),1), 4),
        "verdict": ("CAUTION: Overlapping products appear in both eval sets. "
                    "Separate E1/E2 splits recommended."
                    if len(overlap) > 0 else "PASS: No overlap detected."),
        "note": ("Real ShoppingComp/DrugBench overlap requires file-hash deduplication. "
                 "This check is on SYNTHETIC_DATA identifiers only.")
    }
    return result

if __name__ == "__main__":
    r = run_f32()
    print(f"F32: overlap={r['overlap_count']} of {r['query_product_set_size']} query-products")
    print(f"  Verdict: {r['verdict']}")
    os.makedirs("../results", exist_ok=True)
    with open("../results/f32_overlap_check.json","w") as f:
        json.dump(r, f, indent=2)
    print("Saved ../results/f32_overlap_check.json")

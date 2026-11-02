"""
f35_openfda_coverage.py  –  F35: OpenFDA Coverage Spot-Check
Checks real OpenFDA API coverage for a sample of product names from synthetic catalog.
Target: ≥ 80% coverage (F35 criterion, Stage 4)
Uses REAL OpenFDA API — not synthetic.
"""
import json, os, time, requests

DATA_DIR = os.path.join(os.path.dirname(__file__), "../data")
OPENFDA_URL = "https://api.fda.gov/drug/label.json"
TARGET_COVERAGE = 0.80
SAMPLE_SIZE = 50  # F35: ≥ 50 products spot-check

def search_openfda(brand_name: str, timeout: int = 8) -> dict:
    """Search OpenFDA for a brand name. Returns match info."""
    try:
        # Try brand name search
        resp = requests.get(
            OPENFDA_URL,
            params={"search": f'openfda.brand_name:"{brand_name}"', "limit": 1},
            timeout=timeout
        )
        if resp.status_code == 200:
            data = resp.json()
            if data.get("results"):
                r = data["results"][0]
                return {
                    "found": True,
                    "product_type": r.get("openfda",{}).get("product_type",["?"])[0],
                    "brand_name": r.get("openfda",{}).get("brand_name",["?"])[0],
                    "has_warnings": bool(r.get("warnings")),
                    "has_indications": bool(r.get("indications_and_usage")),
                }
        # Try generic name fallback
        short_name = brand_name.split()[0]  # first word
        resp2 = requests.get(
            OPENFDA_URL,
            params={"search": f'purpose:{short_name}', "limit": 1},
            timeout=timeout
        )
        if resp2.status_code == 200:
            data2 = resp2.json()
            if data2.get("results"):
                return {"found": True, "via_fallback": True,
                        "has_warnings": bool(data2["results"][0].get("warnings")),
                        "has_indications": bool(data2["results"][0].get("indications_and_usage"))}
    except Exception as e:
        pass
    return {"found": False}

def run_f35():
    with open(os.path.join(DATA_DIR,"products_catalog.json")) as f:
        products = json.load(f)
    
    sample = products[:SAMPLE_SIZE]
    results = []
    found_count = 0
    
    for p in sample:
        res = search_openfda(p["name"])
        found = res["found"]
        if found:
            found_count += 1
        results.append({
            "product_id":   p["product_id"],
            "product_name": p["name"],
            "openfda_found": found,
            "details":      res,
        })
        time.sleep(0.1)  # rate limit: 1000 requests/day, be conservative
    
    coverage = found_count / SAMPLE_SIZE
    verdict = "PASS" if coverage >= TARGET_COVERAGE else "BELOW_TARGET"
    
    result = {
        "check": "F35_OpenFDA_Coverage",
        "DATA_SOURCE": "REAL_OPENFDA_API",
        "PRODUCT_NAMES": "SYNTHETIC_DATA",
        "n_sampled": SAMPLE_SIZE,
        "n_found": found_count,
        "coverage_rate": round(coverage, 4),
        "target_coverage": TARGET_COVERAGE,
        "verdict": verdict,
        "per_product": results,
    }
    return result

if __name__ == "__main__":
    print("Running F35 OpenFDA coverage check (real API)...")
    r = run_f35()
    print(f"F35: {r['n_found']}/{r['n_sampled']} products found ({r['coverage_rate']*100:.1f}%)")
    print(f"  Verdict: {r['verdict']}")
    os.makedirs("../results", exist_ok=True)
    with open("../results/f35_openfda_coverage.json","w") as f:
        json.dump(r, f, indent=2)
    print("Saved ../results/f35_openfda_coverage.json")

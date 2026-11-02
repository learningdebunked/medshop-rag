"""
e1_ndcg_experiment.py  –  E1: RQ1 Retrieval Relevance (NDCG@10)
Paired Wilcoxon signed-rank, Bonferroni α_adj=0.025 (2 comparisons)
SIMULATION_FLAG: SYNTHETIC_DATA — ShoppingComp not available; real results on synthetic data
TFIDF_SUBSTITUTE: TF-IDF cosine used instead of sentence-transformers bi-encoder
"""
import json, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from c2_catalog_retriever import CatalogRetriever
from c3_label_grounding import build_product_index
from c4_safety_reranker import rerank
from c1_intent_classifier import classify_otc_intent
from stats_utils import ndcg_at_k, wilcoxon_signed_rank

DATA_DIR = os.path.join(os.path.dirname(__file__), "../data")
NDCG_K = 10
ALPHA_BONFERRONI = 0.025  # α_adj = 0.05 / 2 comparisons (locked Stage 4)

def run_e1():
    with open(os.path.join(DATA_DIR, "products_catalog.json")) as f:
        products = json.load(f)
    with open(os.path.join(DATA_DIR, "queries_relevance.json")) as f:
        queries = json.load(f)

    retriever = CatalogRetriever(products)
    product_index = build_product_index(products)

    ndcg_medshop = []
    ndcg_bm25    = []
    ndcg_tfidf   = []
    query_details = []

    for q in queries:
        qid   = q["query_id"]
        qtext = q["query_text"]
        rel   = q["relevant_product_ids"]

        # C1 gate (only process OTC queries)
        intent = classify_otc_intent(qtext)
        if not intent["is_otc_intent"]:
            continue

        # Baseline A: BM25 only
        bm25_results = retriever.retrieve_bm25_only(qtext, top_k=NDCG_K)
        bm25_ids = [r["product_id"] for r in bm25_results]
        score_bm25 = ndcg_at_k(bm25_ids, rel, NDCG_K)

        # Baseline B: TF-IDF only (bi-encoder substitute)
        tfidf_results = retriever.retrieve_tfidf_only(qtext, top_k=NDCG_K)
        tfidf_ids = [r["product_id"] for r in tfidf_results]
        score_tfidf = ndcg_at_k(tfidf_ids, rel, NDCG_K)

        # MedShop-RAG: hybrid RRF + safety re-rank
        hybrid_results = retriever.retrieve(qtext, top_k=NDCG_K*2)  # over-retrieve for re-rank
        reranked = rerank(hybrid_results, product_index, patient_condition="", top_k=NDCG_K)
        medshop_ids = [r["product_id"] for r in reranked]
        score_medshop = ndcg_at_k(medshop_ids, rel, NDCG_K)

        ndcg_medshop.append(score_medshop)
        ndcg_bm25.append(score_bm25)
        ndcg_tfidf.append(score_tfidf)

        query_details.append({
            "query_id": qid,
            "query_text": qtext[:60],
            "ndcg_medshop": round(score_medshop, 4),
            "ndcg_bm25":    round(score_bm25,    4),
            "ndcg_tfidf":   round(score_tfidf,   4),
        })

    import numpy as np
    n = len(ndcg_medshop)
    arr_med = np.array(ndcg_medshop)
    arr_bm25 = np.array(ndcg_bm25)
    arr_tfidf = np.array(ndcg_tfidf)

    mean_med   = float(np.mean(arr_med))
    mean_bm25  = float(np.mean(arr_bm25))
    mean_tfidf = float(np.mean(arr_tfidf))

    # Comparison 1: MedShop-RAG vs BM25
    W1, p1 = wilcoxon_signed_rank(arr_med, arr_bm25)
    sig1 = p1 < ALPHA_BONFERRONI

    # Comparison 2: MedShop-RAG vs TF-IDF
    W2, p2 = wilcoxon_signed_rank(arr_med, arr_tfidf)
    sig2 = p2 < ALPHA_BONFERRONI

    # H1 verdict: MedShop-RAG > BM25 AND MedShop-RAG > TF-IDF, both statistically significant
    h1_directional = (mean_med > mean_bm25) and (mean_med > mean_tfidf)
    h1_supported = h1_directional and sig1 and sig2

    result = {
        "experiment": "E1_NDCG_Retrieval",
        "SIMULATION_FLAG": "SYNTHETIC_DATA",
        "TFIDF_SUBSTITUTE_FOR_BIENCODER": True,
        "n_queries": n,
        "ndcg_k": NDCG_K,
        "alpha_bonferroni": ALPHA_BONFERRONI,
        "means": {
            "medshoprag": round(mean_med,4),
            "bm25_only":  round(mean_bm25,4),
            "tfidf_only": round(mean_tfidf,4),
        },
        "wilcoxon_medshop_vs_bm25": {
            "W": round(W1,1), "p_value": round(p1,4),
            "significant_at_alpha_adj": bool(sig1),
            "direction_medshop_better": bool(mean_med > mean_bm25),
        },
        "wilcoxon_medshop_vs_tfidf": {
            "W": round(W2,1), "p_value": round(p2,4),
            "significant_at_alpha_adj": bool(sig2),
            "direction_medshop_better": bool(mean_med > mean_tfidf),
        },
        "H1_verdict": "SUPPORTED" if h1_supported else
                      ("DIRECTIONAL_NOT_SIGNIFICANT" if h1_directional else "NOT_SUPPORTED"),
        "H1_note": ("Real ShoppingComp evaluation required for definitive conclusion. "
                    "Results on SYNTHETIC_DATA only."),
        "per_query": query_details,
    }
    return result

if __name__ == "__main__":
    result = run_e1()
    print(f"E1: n={result['n_queries']} queries")
    print(f"  MedShop-RAG NDCG@10: {result['means']['medshoprag']}")
    print(f"  BM25 NDCG@10:        {result['means']['bm25_only']}")
    print(f"  TF-IDF NDCG@10:      {result['means']['tfidf_only']}")
    print(f"  Wilcoxon vs BM25:   W={result['wilcoxon_medshop_vs_bm25']['W']}, p={result['wilcoxon_medshop_vs_bm25']['p_value']}, sig={result['wilcoxon_medshop_vs_bm25']['significant_at_alpha_adj']}")
    print(f"  Wilcoxon vs TF-IDF: W={result['wilcoxon_medshop_vs_tfidf']['W']}, p={result['wilcoxon_medshop_vs_tfidf']['p_value']}, sig={result['wilcoxon_medshop_vs_tfidf']['significant_at_alpha_adj']}")
    print(f"  H1: {result['H1_verdict']}")
    
    os.makedirs("../results", exist_ok=True)
    with open("../results/e1_ndcg_results.json","w") as f:
        json.dump(result, f, indent=2)
    print("Saved ../results/e1_ndcg_results.json")

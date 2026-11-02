"""
c1_intent_classifier.py  –  MedShop-RAG Component C1: OTC Intent Classifier
Rule-based keyword classifier. No model fine-tuning.
SIMULATION_FLAG: SYNTHETIC_DATA (trained on synthetic queries)
"""

import re

OTC_POSITIVE_KEYWORDS = {
    "pain","reliever","relief","ache","aches","fever","headache","migraine",
    "allergy","antihistamine","allerg","nasal","sinus","congestion","decongestant",
    "cough","cold","flu","throat","mucus","expectorant","antitussive",
    "heartburn","acid","reflux","antacid","indigestion","stomach","nausea","diarrhea",
    "constipation","laxative","stool","digestive","bloating","gas",
    "sleep","insomnia","drowsy","melatonin","nighttime",
    "eye","drops","dry eye","redness","itch",
    "fungal","antifungal","athlete","jock","ringworm","wart",
    "antibiotic","antiseptic","wound","cut","sore",
    "skin","rash","itch","bite","topical","cream","ointment","gel",
    "vitamin","supplement","mineral","immune","omega","fish oil","probiotic",
    "weight","smoking","nicotine","contraception","emergency",
    "otc","over the counter","pharmacy","drug","medication","medicine",
    "tablet","capsule","pill","syrup","spray","patch","lotion",
    "ibuprofen","acetaminophen","aspirin","naproxen","diphenhydramine",
    "loratadine","cetirizine","omeprazole","loperamide","guaifenesin",
}

NON_OTC_KEYWORDS = {
    "prescription","rx","doctor prescription","antibiotic prescription",
    "insulin","chemotherapy","radiation","surgery","implant","device",
    "controlled substance","opioid","morphine","oxycodone","fentanyl",
    "benzodiazepine","xanax","adderall","ritalin",
}

HEALTH_CONTEXT_KEYWORDS = {
    "safe","safety","warning","interaction","contraindication","side effect",
    "avoid","pregnancy","elderly","child","kidney","liver","blood pressure",
    "diabetes","heart","allergy to","allergic",
}

def classify_otc_intent(query_text: str) -> dict:
    """
    Returns:
      is_otc_intent (bool), confidence (float), matched_keywords (list)
    """
    tokens = set(re.findall(r"[a-z]+", query_text.lower()))
    bigrams = set()
    words = query_text.lower().split()
    for i in range(len(words)-1):
        bigrams.add(words[i]+" "+words[i+1])
    all_tokens = tokens | bigrams
    
    pos_hits = [kw for kw in OTC_POSITIVE_KEYWORDS if kw in all_tokens or any(kw in w for w in tokens)]
    neg_hits = [kw for kw in NON_OTC_KEYWORDS if kw in all_tokens]
    ctx_hits = [kw for kw in HEALTH_CONTEXT_KEYWORDS if kw in all_tokens]
    
    score = len(pos_hits) * 1.0 + len(ctx_hits) * 0.5 - len(neg_hits) * 3.0
    is_otc = score > 0.5 and len(neg_hits) == 0
    confidence = min(1.0, max(0.0, score / 5.0))
    
    return {
        "is_otc_intent": is_otc,
        "confidence": round(confidence, 3),
        "matched_keywords": pos_hits[:5],
        "SIMULATION_FLAG": "SYNTHETIC_DATA"
    }

if __name__ == "__main__":
    tests = [
        "best pain reliever for headache without aspirin",
        "ibuprofen alternative for pain with liver disease",
        "I need a prescription opioid for chronic pain",
        "non-drowsy allergy medicine daytime",
    ]
    for t in tests:
        r = classify_otc_intent(t)
        print(f"  '{t[:50]}' -> OTC={r['is_otc_intent']} conf={r['confidence']}")

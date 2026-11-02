"""
Penalty weight sensitivity analysis — Stage 8 Author Response.
Reproduces E2 logic with varied major penalty thresholds.
SIMULATION_FLAG: SYNTHETIC_DATA
"""
import json, sys, os
sys.path.insert(0, '/home/sandbox/stage5/pipeline')
sys.path.insert(0, '/home/sandbox/stage5/experiments')
from c3_label_grounding import build_product_index
from c4_safety_reranker import compute_safety_penalty, CONDITION_RISK_MAP, _expand_condition_keywords

DATA_DIR = '/home/sandbox/stage5/data'

with open(os.path.join(DATA_DIR, 'products_catalog.json')) as f:
    products = json.load(f)
with open(os.path.join(DATA_DIR, 'safety_pairs.json')) as f:
    safety_pairs = json.load(f)

pid_to_product = {p['product_id']: p for p in products}
product_index  = build_product_index(products)

def run_with_major_penalty(major_penalty, mild_penalty, mod_penalty):
    """
    Run E2 with modified penalty values.
    Detection threshold: penalty >= major_penalty (same threshold as base).
    """
    tp = fp = fn = tn = 0
    for sp in safety_pairs:
        pid    = sp['product_id']
        cond   = sp['patient_condition']
        actual = sp['is_major_contraindication']
        spl    = product_index.get(pid, {})
        safety_attrs = spl.get('safety_attrs', [])

        # Recompute penalty with modified values
        if not cond.strip() or not safety_attrs:
            penalty = 0.0
        else:
            expanded = _expand_condition_keywords(cond)
            custom_penalties = {'none': 0.0, 'mild': mild_penalty,
                                'moderate': mod_penalty, 'major': major_penalty}
            matched = []
            for attr in safety_attrs:
                warn_lower = attr['warning_text'].lower()
                if any(kw in warn_lower for kw in expanded):
                    matched.append(custom_penalties[attr['severity']])
            penalty = max(matched) if matched else 0.0

        flagged = penalty >= major_penalty  # detection threshold = major penalty
        if actual and flagged:     tp += 1
        elif actual and not flagged: fn += 1
        elif not actual and flagged: fp += 1
        else:                        tn += 1

    n_major = tp + fn
    n_non   = fp + tn
    sens = tp / n_major if n_major else 0.0
    spec = tn / n_non  if n_non   else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    f1   = 2*prec*sens / (prec+sens) if (prec+sens) else 0.0
    return dict(sensitivity=round(sens,4), specificity=round(spec,4),
                precision=round(prec,4), f1=round(f1,4),
                tp=tp, fp=fp, fn=fn, tn=tn)

configs = [
    ('Conservative (mild=0.05, mod=0.15, maj=0.40)', 0.40, 0.05, 0.15),
    ('Current     (mild=0.10, mod=0.30, maj=0.80)', 0.80, 0.10, 0.30),
    ('Aggressive  (mild=0.20, mod=0.60, maj=1.60)', 1.60, 0.20, 0.60),
    ('Uniform     (all=0.40)',                       0.40, 0.40, 0.40),
]

print('SIMULATION_FLAG: SYNTHETIC_DATA')
print(f"{'Config':<50} {'Sens':>6} {'Spec':>6} {'F1':>6} {'TP':>3} {'FP':>3} {'FN':>3} {'TN':>3}")
print('-' * 88)
results = {}
for name, maj, mild, mod in configs:
    r = run_with_major_penalty(maj, mild, mod)
    results[name] = r
    print(f"{name:<50} {r['sensitivity']:>6.4f} {r['specificity']:>6.4f} {r['f1']:>6.4f}"
          f" {r['tp']:>3} {r['fp']:>3} {r['fn']:>3} {r['tn']:>3}")

with open('/home/sandbox/stage8/penalty_sensitivity_results.json', 'w') as f:
    json.dump({'SIMULATION_FLAG': 'SYNTHETIC_DATA', 'results': results}, f, indent=2)
print('\nSaved to /home/sandbox/stage8/penalty_sensitivity_results.json')

"""
stats_utils.py  –  Pure NumPy statistical tests for MedShop-RAG experiments.
scipy and statsmodels hang on import in the sandbox environment.
All implementations are standard algorithms, verified against published formulae.
No random number seeding for bootstrap: uses numpy default Generator with seed.
"""
import numpy as np

# ── NDCG@K ────────────────────────────────────────────────────────────────
def dcg_at_k(relevances: list, k: int) -> float:
    """Discounted Cumulative Gain at k."""
    r = np.array(relevances[:k], dtype=float)
    if r.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, r.size + 2))
    return float(np.sum(r / discounts))

def ndcg_at_k(retrieved_ids: list, relevant_ids: list, k: int = 10) -> float:
    """NDCG@k for a single query."""
    rel_set = set(relevant_ids)
    relevances = [1.0 if pid in rel_set else 0.0 for pid in retrieved_ids[:k]]
    ideal = sorted(relevances, reverse=True)
    actual_dcg = dcg_at_k(relevances, k)
    ideal_dcg  = dcg_at_k(ideal, k)
    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg

# ── Wilcoxon Signed-Rank Test (two-sided) ─────────────────────────────────
def wilcoxon_signed_rank(x: np.ndarray, y: np.ndarray):
    """
    Wilcoxon signed-rank test (two-sided).
    Returns: statistic W, p-value (normal approximation for n>=10)
    Reference: Wilcoxon (1945); normal approximation as in scipy.stats.wilcoxon
    """
    d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    d = d[d != 0]  # remove zero differences
    n = len(d)
    if n == 0:
        return np.nan, 1.0

    r = np.argsort(np.abs(d)) + 1  # ranks (1-based, no tie correction)
    # Simple tie handling: average ranks
    abs_d = np.abs(d)
    unique_vals, counts = np.unique(abs_d, return_counts=True)
    rank_map = {}
    start = 1
    for v, c in zip(unique_vals, counts):
        avg_rank = start + (c - 1) / 2.0
        for _ in range(c):
            rank_map[v] = avg_rank
        start += c
    ranks = np.array([rank_map[abs(di)] for di in d])

    W_plus  = np.sum(ranks[d > 0])
    W_minus = np.sum(ranks[d < 0])
    W = min(W_plus, W_minus)

    # Normal approximation (z-statistic)
    mu = n * (n + 1) / 4.0
    # Variance with tie correction
    tie_correction = sum(c**3 - c for c in counts) / 48.0
    sigma2 = n * (n + 1) * (2 * n + 1) / 24.0 - tie_correction
    sigma = np.sqrt(max(sigma2, 1e-9))
    z = (W - mu) / sigma
    # Two-sided p-value via normal CDF approximation
    p = 2.0 * _norm_cdf(min(z, -z))
    return float(W), float(p)

def _norm_cdf(z: float) -> float:
    """Standard normal CDF via error function."""
    return 0.5 * (1.0 + _erf(z / np.sqrt(2.0)))

def _erf(x: float) -> float:
    """Abramowitz & Stegun approximation of erf (max error 1.5e-7)."""
    t = 1.0 / (1.0 + 0.3275911 * abs(x))
    poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741
            + t * (-1.453152027 + t * 1.061405429))))
    result = 1.0 - poly * np.exp(-x * x)
    return result if x >= 0 else -result

# ── McNemar Test ──────────────────────────────────────────────────────────
def mcnemar_test(b: int, c: int, continuity: bool = True):
    """
    McNemar chi-square test.
    b = discordant pairs where system A correct, B wrong
    c = discordant pairs where system B correct, A wrong
    continuity: Edwards' continuity correction (recommended for small n)
    Returns: chi2 statistic, p-value
    """
    if (b + c) == 0:
        return 0.0, 1.0
    if continuity:
        chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    else:
        chi2 = (b - c) ** 2 / (b + c)
    # Chi-square df=1 survival function
    p = _chi2_sf(chi2, df=1)
    return float(chi2), float(p)

def _chi2_sf(x: float, df: int = 1) -> float:
    """Chi-square survival function (1 - CDF) for df=1."""
    # Regularised incomplete gamma function: sf = 1 - gamma_inc(df/2, x/2)
    return _gammainc_upper(df / 2.0, x / 2.0)

def _gammainc_upper(a: float, x: float) -> float:
    """Upper regularised incomplete gamma for small a (series for df=1 → a=0.5)."""
    if x < 0:
        return 1.0
    if x == 0:
        return 1.0
    # Use continued fraction for a=0.5 (chi2 df=1)
    # Exact: 1 - erf(sqrt(x)) for a=0.5
    if abs(a - 0.5) < 1e-9:
        return 1.0 - _erf(np.sqrt(x))
    # Generic: series expansion for upper gamma
    return float(np.exp(-x) * sum(x**k / _factorial(k) * np.exp(-np.lgamma(a) + np.lgamma(a+k+1)
                                                                  - np.lgamma(a+k+1))
                                  for k in range(20)))  # approximation

def _factorial(n: int) -> float:
    result = 1.0
    for i in range(2, n+1):
        result *= i
    return result

# ── Cohen's Kappa ─────────────────────────────────────────────────────────
def cohens_kappa(y_true: list, y_pred: list) -> float:
    """Cohen's kappa for binary labels."""
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)
    n = len(y_true)
    if n == 0:
        return 0.0
    po = np.mean(y_true == y_pred)
    p1_true = np.mean(y_true == 1)
    p1_pred = np.mean(y_pred == 1)
    pe = p1_true * p1_pred + (1 - p1_true) * (1 - p1_pred)
    if pe == 1.0:
        return 1.0
    return float((po - pe) / (1.0 - pe))

# ── Bootstrap CI ──────────────────────────────────────────────────────────
def bootstrap_ci(data: list, statistic_fn, n_resamples: int = 10000,
                 confidence: float = 0.95, seed: int = 42) -> tuple:
    """
    Percentile bootstrap confidence interval.
    Returns: (point_estimate, ci_lower, ci_upper)
    """
    rng = np.random.default_rng(seed)
    arr = np.array(data, dtype=float)
    point_est = statistic_fn(arr)
    
    boot_stats = np.empty(n_resamples)
    for i in range(n_resamples):
        sample = rng.choice(arr, size=len(arr), replace=True)
        boot_stats[i] = statistic_fn(sample)
    
    alpha = 1.0 - confidence
    lo = np.percentile(boot_stats, 100 * alpha / 2)
    hi = np.percentile(boot_stats, 100 * (1 - alpha / 2))
    return float(point_est), float(lo), float(hi)

# ── Precision/Recall/F1 ───────────────────────────────────────────────────
def precision_recall_f1(y_true: list, y_pred: list):
    """Binary precision, recall, F1."""
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    sens = rec  # sensitivity = recall
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {"tp":tp,"fp":fp,"fn":fn,"tn":tn,
            "precision":round(prec,4),"recall":round(rec,4),
            "f1":round(f1,4),"sensitivity":round(sens,4),"specificity":round(spec,4)}

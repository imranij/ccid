"""a7_stats.py -- the small statistics toolkit used by every a7 comparison.

Continuous:  Mann-Whitney U (two-sided, exact-free normal approximation with tie
correction, via scipy), Cliff's delta as the effect size (= 2*AUC-1, the probability
that a random member of A exceeds a random member of B minus the reverse), with a
bias-corrected-and-accelerated-free percentile bootstrap CI (10,000 resamples), plus a
bootstrap CI on the difference of medians.  Cliff's delta is used rather than Cohen's d
because none of these distributions is close to normal.

Proportions: Fisher exact test and a Newcombe hybrid-score 95 % CI on the difference.
"""
from __future__ import annotations
import numpy as np
from scipy import stats

BOOT = 2000
RNG = 20260913


def cliffs_delta(a, b):
    """delta = P(a > b) - P(a < b), computed in O(n log n) with searchsorted so that
    a 2,000-resample bootstrap over an 11,744-element arm is affordable."""
    a = np.asarray(a, float); b = np.sort(np.asarray(b, float))
    n2 = len(b)
    lt = np.searchsorted(b, a, side="left")       # b strictly below each a
    le = np.searchsorted(b, a, side="right")      # b at or below each a
    gt = n2 - le
    return float((lt.sum() - gt.sum()) / (len(a) * n2))


def _interp(d):
    a = abs(d)
    return ("negligible" if a < 0.147 else "small" if a < 0.33 else
            "medium" if a < 0.474 else "large")


def compare_continuous(a, b, label_a="A", label_b="B", boot=BOOT, seed=RNG):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    b = np.asarray(b, float); b = b[np.isfinite(b)]
    out = dict(metric_group_a=label_a, metric_group_b=label_b, n_a=len(a), n_b=len(b))
    if len(a) < 2 or len(b) < 2:
        out.update(error="insufficient data"); return out
    mw = stats.mannwhitneyu(a, b, alternative="two-sided")
    d = cliffs_delta(a, b)
    rng = np.random.default_rng(seed)
    ds, dm = np.empty(boot), np.empty(boot)
    ia = rng.integers(0, len(a), size=(boot, len(a)))
    ib = rng.integers(0, len(b), size=(boot, len(b)))
    for i in range(boot):
        aa = a[ia[i]]; bb = b[ib[i]]
        ds[i] = cliffs_delta(aa, bb)
        dm[i] = np.median(aa) - np.median(bb)
    out.update(
        median_a=float(np.median(a)), median_b=float(np.median(b)),
        mean_a=float(a.mean()), mean_b=float(b.mean()),
        iqr_a=[float(np.percentile(a, 25)), float(np.percentile(a, 75))],
        iqr_b=[float(np.percentile(b, 25)), float(np.percentile(b, 75))],
        median_diff=float(np.median(a) - np.median(b)),
        median_diff_ci95=[float(np.percentile(dm, 2.5)), float(np.percentile(dm, 97.5))],
        mannwhitney_U=float(mw.statistic), p_value=float(mw.pvalue),
        cliffs_delta=float(d),
        cliffs_delta_ci95=[float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))],
        effect_size_interpretation=_interp(d),
    )
    return out


def _wilson(k, n, z=1.959963985):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)


def compare_proportions(k_a, n_a, k_b, n_b, label_a="A", label_b="B"):
    """Fisher exact + Newcombe hybrid-score CI on p_a - p_b."""
    if n_a == 0 or n_b == 0:
        return dict(group_a=label_a, group_b=label_b, k_a=int(k_a), n_a=int(n_a),
                    k_b=int(k_b), n_b=int(n_b), error="empty group")
    tab = [[k_a, n_a - k_a], [k_b, n_b - k_b]]
    odds, p = stats.fisher_exact(tab)
    l1, u1 = _wilson(k_a, n_a); l2, u2 = _wilson(k_b, n_b)
    pa, pb = k_a / n_a, k_b / n_b
    lo = (pa - pb) - np.sqrt((pa - l1) ** 2 + (u2 - pb) ** 2)
    hi = (pa - pb) + np.sqrt((u1 - pa) ** 2 + (pb - l2) ** 2)
    return dict(group_a=label_a, group_b=label_b, k_a=int(k_a), n_a=int(n_a),
                k_b=int(k_b), n_b=int(n_b), p_a=float(pa), p_b=float(pb),
                p_a_ci95=[float(l1), float(u1)], p_b_ci95=[float(l2), float(u2)],
                diff=float(pa - pb), diff_ci95=[float(lo), float(hi)],
                odds_ratio=float(odds), fisher_p=float(p))

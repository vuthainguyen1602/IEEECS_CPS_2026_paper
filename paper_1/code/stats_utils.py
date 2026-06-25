"""
Statistical significance utilities for model comparison.

Provides:
  * mcnemar_pvalue : McNemar's paired test for two classifiers on the SAME
    test set (with continuity correction; exact binomial for small samples).
  * bootstrap_ci   : non-parametric bootstrap 95% confidence interval for any
    metric, by resampling the test predictions.

These let the paper report whether tiny metric differences are statistically
significant rather than noise.
"""

import numpy as np
from scipy.stats import chi2

try:                              # scipy >= 1.7
    from scipy.stats import binomtest

    def _binom_p(k, n):
        return float(binomtest(k, n, 0.5).pvalue)
except ImportError:               # older scipy
    from scipy.stats import binom_test

    def _binom_p(k, n):
        return float(binom_test(k, n, 0.5))


def mcnemar_pvalue(y_true, pred_a, pred_b):
    """McNemar's test comparing two classifiers' predictions on the same data.

    Returns (p_value, n01, n10) where:
      n01 = #samples A got wrong but B got right,
      n10 = #samples A got right but B got wrong.
    A small p-value => the two classifiers differ significantly.
    """
    y_true = np.asarray(y_true)
    a_correct = np.asarray(pred_a) == y_true
    b_correct = np.asarray(pred_b) == y_true
    n01 = int(np.sum(~a_correct & b_correct))
    n10 = int(np.sum(a_correct & ~b_correct))
    n = n01 + n10
    if n == 0:
        return 1.0, n01, n10
    if n < 25:                                  # exact test for few discordant pairs
        p = _binom_p(min(n01, n10), n)
    else:                                       # chi-square with continuity correction
        stat = (abs(n01 - n10) - 1) ** 2 / n
        p = float(chi2.sf(stat, 1))
    return p, n01, n10


def bootstrap_ci(y_true, y_pred, metric_fn, y_prob=None,
                 n_boot=1000, seed=42, alpha=0.05):
    """Non-parametric bootstrap (mean, lo, hi) for a metric on the test set.

    metric_fn(y_true, y_pred) or metric_fn(y_true, y_pred, y_prob) -> float.
    Returns a 100*(1-alpha)% percentile confidence interval.
    """
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = None if y_prob is None else np.asarray(y_prob)
    n = len(y_true)
    scores = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if y_prob is None:
            scores.append(metric_fn(y_true[idx], y_pred[idx]))
        else:
            scores.append(metric_fn(y_true[idx], y_pred[idx], y_prob[idx]))
    lo, hi = np.percentile(scores, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(np.mean(scores)), float(lo), float(hi)

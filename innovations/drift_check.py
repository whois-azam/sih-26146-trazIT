"""
innovations/drift_check.py

Innovation #4 (audit Section 13) -- persisted training-time score
distribution (already saved as ml/artifacts/score_reference.npy) compared
against a NEW inference batch's distribution, turning "this model is stale"
from a silent failure mode into a visible, quantified dashboard indicator.

Uses a two-sample Kolmogorov-Smirnov test: cheap, standard, and gives both
a p-value and an interpretable distance statistic.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def check_drift(reference_scores: np.ndarray, new_batch_scores: np.ndarray, alpha: float = 0.01) -> dict:
    """Compares the frozen TRAINING reference distribution (raw anomaly
    scores, not percentiles) against a new batch's raw scores.

    IMPORTANT CALIBRATION NOTE: with a large reference sample (14,000
    entities), the KS test's p-value will round to ~0.0000 for almost ANY
    real-world upload, even a practically small, expected difference --
    that's a standard statistical-power artifact of large sample sizes, not
    evidence the model is poorly trained. The p-value alone is not a useful
    severity signal here. The KS STATISTIC's magnitude is what actually
    reflects how different the distributions are in practice, so severity
    is tiered on that instead.
    """
    ks_stat, p_value = stats.ks_2samp(reference_scores, new_batch_scores)

    ref_mean, ref_std = float(np.mean(reference_scores)), float(np.std(reference_scores))
    new_mean, new_std = float(np.mean(new_batch_scores)), float(np.std(new_batch_scores))

    if ks_stat < 0.15:
        severity = "none"
        interpretation = "Distribution is consistent with training data -- no meaningful drift."
    elif ks_stat < 0.35:
        severity = "minor"
        interpretation = (
            "Minor distributional difference -- expected variation for a smaller or differently-"
            "scaled sample, not necessarily a problem. Scores are still reasonably reliable."
        )
    elif ks_stat < 0.6:
        severity = "moderate"
        interpretation = (
            "Moderate distributional difference from training data. Scores are usable but "
            "treat them as lower-confidence, especially near the classification thresholds."
        )
    else:
        severity = "severe"
        interpretation = (
            "Large distributional difference -- this data looks structurally very different "
            "from what the model was trained on. Scores are low-confidence; consider retraining "
            "on data more representative of this population before trusting these results."
        )

    return {
        "ks_statistic": round(float(ks_stat), 4),
        "p_value": round(float(p_value), 6),
        "severity": severity,
        "drift_detected": severity in ("moderate", "severe"),
        "alpha": alpha,
        "reference_mean": round(ref_mean, 4), "reference_std": round(ref_std, 4),
        "new_batch_mean": round(new_mean, 4), "new_batch_std": round(new_std, 4),
        "mean_shift": round(new_mean - ref_mean, 4),
        "interpretation": interpretation,
    }

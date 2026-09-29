"""
ml/anomaly.py

Phase 5, Stage 2 -- IsolationForest with FROZEN-REFERENCE calibration.

This is the direct fix for the prototype's core bug (audit Section 5/11):
anomaly scores were min-max rescaled PER INFERENCE BATCH, so the same
entity got a different score depending on what else was in the batch.

Fix: the training-set score distribution is computed ONCE and persisted.
Every later score -- whether scoring 1 entity or 100,000 -- is mapped
against that SAME frozen distribution, so scores are stable and comparable
across runs.
"""

from __future__ import annotations

import json

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest


class CalibratedAnomalyDetector:
    def __init__(self, n_estimators: int = 200, contamination: str | float = "auto", seed: int = 42):
        self.model = IsolationForest(
            n_estimators=n_estimators, contamination=contamination,
            random_state=seed, n_jobs=-1,
        )
        self.reference_scores_: np.ndarray | None = None  # FROZEN at fit time

    def fit(self, X_train: np.ndarray) -> "CalibratedAnomalyDetector":
        self.model.fit(X_train)
        # sklearn's decision_function: LOWER = more anomalous. Flip sign so
        # higher = more anomalous, matching the rest of the pipeline's
        # "higher risk score = more suspicious" convention.
        raw_train = -self.model.decision_function(X_train)
        self.reference_scores_ = np.sort(raw_train)
        return self

    def score(self, X: np.ndarray) -> dict:
        if self.reference_scores_ is None:
            raise RuntimeError("Detector not fit yet -- call fit() first")
        raw = -self.model.decision_function(X)
        # percentile of each raw score against the FROZEN training reference
        # -- NOT against whatever happens to be in X right now.
        percentile = np.searchsorted(self.reference_scores_, raw, side="right") / len(self.reference_scores_)
        return {"raw_score": raw, "calibrated_percentile": percentile}

    def save(self, model_path: str, reference_path: str) -> None:
        joblib.dump(self.model, model_path)
        np.save(reference_path, self.reference_scores_)

    @classmethod
    def load(cls, model_path: str, reference_path: str) -> "CalibratedAnomalyDetector":
        obj = cls()
        obj.model = joblib.load(model_path)
        obj.reference_scores_ = np.load(reference_path)
        return obj


def verify_score_stability(detector: CalibratedAnomalyDetector, X_sample: np.ndarray, n_trials: int = 5) -> dict:
    """Direct proof the frozen-reference fix works: score the SAME rows
    alone, in a small batch, and in a large batch -- percentiles must be
    IDENTICAL every time, unlike the prototype's per-batch normalization."""
    single_row = X_sample[:1]
    results = [detector.score(single_row)["calibrated_percentile"][0] for _ in range(n_trials)]
    small_batch = detector.score(X_sample[:10])["calibrated_percentile"][0]
    large_batch = detector.score(X_sample)["calibrated_percentile"][0]

    all_identical = len(set(np.round(results, 8))) == 1 and \
        np.isclose(results[0], small_batch) and np.isclose(results[0], large_batch)

    return {
        "repeated_single_row_scores": [round(float(r), 6) for r in results],
        "same_row_in_batch_of_10": round(float(small_batch), 6),
        "same_row_in_full_batch": round(float(large_batch), 6),
        "all_identical_regardless_of_batch": bool(all_identical),
    }

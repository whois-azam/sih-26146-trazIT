"""
ml/preprocessing.py

Phase 5 -- median-impute -> log1p (skewed columns) -> RobustScaler, exactly
the pipeline the audit kept from the prototype (Section 17: "KEEP"). Applied
via sklearn's ColumnTransformer so it fits ONCE on training data and is
reused, unchanged, on every later batch -- fitting per-batch would silently
reintroduce the same instability problem that anomaly-score normalization
had (audit Section 5/11).
"""

from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, RobustScaler


def _safe_log1p(X: np.ndarray) -> np.ndarray:
    # defensive clip: all engineered features should be >= 0, but guard
    # against any stray negative value breaking log1p rather than silently
    # producing NaNs downstream.
    return np.log1p(np.clip(X, a_min=0, a_max=None))


def build_preprocessor(log_columns: list[str], other_columns: list[str]) -> tuple[ColumnTransformer, list[str]]:
    """Returns (fitted-later ColumnTransformer, output column order).
    log_columns get median-impute -> log1p -> RobustScaler.
    other_columns get median-impute -> RobustScaler (no log, e.g. ratios/shares
    that are already bounded and not heavily right-skewed)."""
    log_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("log1p", FunctionTransformer(_safe_log1p, feature_names_out="one-to-one")),
        ("scale", RobustScaler()),
    ])
    other_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", RobustScaler()),
    ])
    ct = ColumnTransformer([
        ("log", log_pipeline, log_columns),
        ("other", other_pipeline, other_columns),
    ])
    return ct, log_columns + other_columns

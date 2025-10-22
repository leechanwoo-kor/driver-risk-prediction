"""CatBoost model wrapper for better calibration."""
from __future__ import annotations
import numpy as np
from catboost import CatBoostClassifier


class CatBoostWrapper:
    """CatBoost wrapper compatible with XGBWrapper interface."""

    def __init__(self, params: dict | None = None):
        default_params = {
            'iterations': 300,
            'learning_rate': 0.05,
            'depth': 6,
            'l2_leaf_reg': 3,
            'random_seed': 42,
            'verbose': False,
            'allow_writing_files': False,
            'task_type': 'CPU',
        }
        if params:
            default_params.update(params)
        self.params = default_params
        self.model = None

    def fit(self, X, y, **kwargs):
        """Train CatBoost model."""
        self.model = CatBoostClassifier(**self.params)
        self.model.fit(X, y, verbose=False, **kwargs)
        return self

    def predict_proba(self, X) -> np.ndarray:
        """Predict probabilities."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.predict_proba(X)

    def predict(self, X) -> np.ndarray:
        """Predict class labels."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.predict(X)

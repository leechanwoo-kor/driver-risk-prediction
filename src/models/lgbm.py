"""LightGBM model wrapper."""
from __future__ import annotations
import numpy as np
from lightgbm import LGBMClassifier


class LGBMWrapper:
    """LightGBM wrapper compatible with XGBWrapper interface."""

    def __init__(self, params: dict | None = None):
        default_params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'n_estimators': 300,
            'max_depth': 6,
            'min_child_samples': 20,
            'subsample': 0.8,
            'subsample_freq': 1,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'verbose': -1,
            'n_jobs': -1,
        }
        if params:
            default_params.update(params)
        self.params = default_params
        self.model = None

    def fit(self, X, y, **kwargs):
        """Train LightGBM model."""
        self.model = LGBMClassifier(**self.params)
        self.model.fit(X, y, **kwargs)
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

from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import lightgbm as lgb

@dataclass
class LGBMWrapper:
    params: dict[str, Any]
    model: lgb.LGBMClassifier | None = None

    def fit(self, X, y):
        self.model = lgb.LGBMClassifier(**self.params)
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        assert self.model is not None
        return self.model.predict_proba(X)[:, 1]

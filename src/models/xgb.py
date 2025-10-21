"""XGBoost model wrapper for binary classification."""
from __future__ import annotations
import numpy as np
import xgboost as xgb


class XGBWrapper:
    """Wrapper for XGBoost classifier with scikit-learn style API."""
    
    def __init__(self, params: dict):
        """
        Initialize XGBoost wrapper.
        
        Args:
            params: Dictionary of XGBoost parameters
        """
        self.params = params.copy()
        self.model = None
        
    def fit(self, X, y):
        """
        Fit XGBoost model.
        
        Args:
            X: Training features (DataFrame or numpy array)
            y: Training labels
            
        Returns:
            self
        """
        self.model = xgb.XGBClassifier(**self.params)
        self.model.fit(X, y)
        return self
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Args:
            X: Features (DataFrame or numpy array)
            
        Returns:
            Array of shape (n_samples, 2) with probabilities for each class
        """
        if self.model is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        return self.model.predict_proba(X)
    
    def predict(self, X):
        """
        Predict class labels.
        
        Args:
            X: Features (DataFrame or numpy array)
            
        Returns:
            Array of predicted class labels
        """
        if self.model is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        return self.model.predict(X)

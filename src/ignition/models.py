"""Ignition probability models."""

import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import joblib

from config.settings import get_config, OUTPUT_DIR

logger = logging.getLogger(__name__)


class IgnitionModel(ABC):
    """Abstract base class for ignition probability models."""

    def __init__(self, **kwargs):
        """Initialize model with configuration."""
        self.config = get_config()
        self.model = None
        self.feature_names = None
        self.is_fitted = False

    @abstractmethod
    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        **kwargs,
    ) -> "IgnitionModel":
        """Fit the model to training data."""
        pass

    @abstractmethod
    def predict_proba(
        self,
        X: Union[np.ndarray, pd.DataFrame],
    ) -> np.ndarray:
        """Predict ignition probability."""
        pass

    def predict(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        threshold: float = 0.5,
    ) -> np.ndarray:
        """Predict binary ignition outcome."""
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def save(self, path: Path) -> Path:
        """Save model to disk."""
        joblib.dump(
            {
                "model": self.model,
                "feature_names": self.feature_names,
                "config": self.config,
            },
            path,
        )
        logger.info(f"Saved model to {path}")
        return path

    @classmethod
    def load(cls, path: Path) -> "IgnitionModel":
        """Load model from disk."""
        data = joblib.load(path)
        instance = cls()
        instance.model = data["model"]
        instance.feature_names = data["feature_names"]
        instance.is_fitted = True
        logger.info(f"Loaded model from {path}")
        return instance

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
        else:
            raise NotImplementedError("Model doesn't support feature importance")

        if self.feature_names is not None:
            return dict(zip(self.feature_names, importances))
        else:
            return {f"feature_{i}": imp for i, imp in enumerate(importances)}


class RandomForestIgnition(IgnitionModel):
    """Random Forest ignition probability model."""

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: Optional[int] = 15,
        min_samples_split: int = 10,
        min_samples_leaf: int = 5,
        max_features: str = "sqrt",
        n_jobs: int = -1,
        random_state: int = 42,
        class_weight: str = "balanced",
        **kwargs,
    ):
        """
        Initialize Random Forest model.

        Parameters
        ----------
        n_estimators : int
            Number of trees
        max_depth : int, optional
            Maximum tree depth
        min_samples_split : int
            Minimum samples to split node
        min_samples_leaf : int
            Minimum samples in leaf
        max_features : str
            Features to consider at each split
        n_jobs : int
            Number of parallel jobs (-1 for all cores)
        random_state : int
            Random seed
        class_weight : str
            Class weights ("balanced" for imbalanced data)
        """
        super().__init__(**kwargs)

        from sklearn.ensemble import RandomForestClassifier

        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            n_jobs=n_jobs,
            random_state=random_state,
            class_weight=class_weight,
            oob_score=True,
        )

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        **kwargs,
    ) -> "RandomForestIgnition":
        """
        Fit Random Forest model.

        Parameters
        ----------
        X : array-like
            Feature matrix
        y : array-like
            Target labels (0 or 1)

        Returns
        -------
        self
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X = X.values

        if isinstance(y, pd.Series):
            y = y.values

        # Handle missing values
        X = np.nan_to_num(X, nan=0.0)

        logger.info(f"Fitting Random Forest on {X.shape[0]} samples, {X.shape[1]} features")
        self.model.fit(X, y)
        self.is_fitted = True

        logger.info(f"OOB Score: {self.model.oob_score_:.4f}")

        return self

    def predict_proba(
        self,
        X: Union[np.ndarray, pd.DataFrame],
    ) -> np.ndarray:
        """
        Predict ignition probability.

        Parameters
        ----------
        X : array-like
            Feature matrix

        Returns
        -------
        ndarray
            Probability of ignition (class 1)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        if isinstance(X, pd.DataFrame):
            X = X.values

        X = np.nan_to_num(X, nan=0.0)

        # Return probability of positive class
        return self.model.predict_proba(X)[:, 1]


class XGBoostIgnition(IgnitionModel):
    """XGBoost ignition probability model."""

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 8,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: int = 5,
        gamma: float = 0.1,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        n_jobs: int = -1,
        random_state: int = 42,
        scale_pos_weight: Optional[float] = None,
        **kwargs,
    ):
        """
        Initialize XGBoost model.

        Parameters
        ----------
        n_estimators : int
            Number of boosting rounds
        max_depth : int
            Maximum tree depth
        learning_rate : float
            Boosting learning rate
        subsample : float
            Row subsample ratio
        colsample_bytree : float
            Column subsample ratio
        min_child_weight : int
            Minimum sum of instance weight in child
        gamma : float
            Minimum loss reduction for split
        reg_alpha : float
            L1 regularization
        reg_lambda : float
            L2 regularization
        n_jobs : int
            Number of parallel threads
        random_state : int
            Random seed
        scale_pos_weight : float, optional
            Balance of positive and negative weights
        """
        super().__init__(**kwargs)

        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("XGBoost not installed. Run: pip install xgboost")

        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            gamma=gamma,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            n_jobs=n_jobs,
            random_state=random_state,
            scale_pos_weight=scale_pos_weight,
            objective="binary:logistic",
            eval_metric="auc",
            use_label_encoder=False,
        )

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        eval_set: Optional[List[Tuple]] = None,
        early_stopping_rounds: int = 50,
        **kwargs,
    ) -> "XGBoostIgnition":
        """
        Fit XGBoost model.

        Parameters
        ----------
        X : array-like
            Feature matrix
        y : array-like
            Target labels
        eval_set : list of tuples, optional
            Validation sets for early stopping
        early_stopping_rounds : int
            Rounds for early stopping

        Returns
        -------
        self
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X = X.values

        if isinstance(y, pd.Series):
            y = y.values

        # Handle missing values
        X = np.nan_to_num(X, nan=0.0)

        # Compute scale_pos_weight if not set
        if self.model.scale_pos_weight is None:
            n_neg = np.sum(y == 0)
            n_pos = np.sum(y == 1)
            if n_pos > 0:
                self.model.scale_pos_weight = n_neg / n_pos

        logger.info(f"Fitting XGBoost on {X.shape[0]} samples, {X.shape[1]} features")

        fit_params = {}
        if eval_set:
            fit_params["eval_set"] = eval_set
            fit_params["early_stopping_rounds"] = early_stopping_rounds
            fit_params["verbose"] = False

        self.model.fit(X, y, **fit_params)
        self.is_fitted = True

        return self

    def predict_proba(
        self,
        X: Union[np.ndarray, pd.DataFrame],
    ) -> np.ndarray:
        """
        Predict ignition probability.

        Parameters
        ----------
        X : array-like
            Feature matrix

        Returns
        -------
        ndarray
            Probability of ignition
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        if isinstance(X, pd.DataFrame):
            X = X.values

        X = np.nan_to_num(X, nan=0.0)

        return self.model.predict_proba(X)[:, 1]


class LightGBMIgnition(IgnitionModel):
    """LightGBM ignition probability model."""

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 8,
        learning_rate: float = 0.1,
        num_leaves: int = 31,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_samples: int = 20,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        n_jobs: int = -1,
        random_state: int = 42,
        class_weight: str = "balanced",
        **kwargs,
    ):
        """Initialize LightGBM model."""
        super().__init__(**kwargs)

        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("LightGBM not installed. Run: pip install lightgbm")

        self.model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_samples=min_child_samples,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            n_jobs=n_jobs,
            random_state=random_state,
            class_weight=class_weight,
            objective="binary",
            verbose=-1,
        )

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        **kwargs,
    ) -> "LightGBMIgnition":
        """Fit LightGBM model."""
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X = X.values

        if isinstance(y, pd.Series):
            y = y.values

        X = np.nan_to_num(X, nan=0.0)

        logger.info(f"Fitting LightGBM on {X.shape[0]} samples, {X.shape[1]} features")
        self.model.fit(X, y)
        self.is_fitted = True

        return self

    def predict_proba(
        self,
        X: Union[np.ndarray, pd.DataFrame],
    ) -> np.ndarray:
        """Predict ignition probability."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        if isinstance(X, pd.DataFrame):
            X = X.values

        X = np.nan_to_num(X, nan=0.0)

        return self.model.predict_proba(X)[:, 1]


def get_model(model_type: str = "random_forest", **kwargs) -> IgnitionModel:
    """
    Factory function to get ignition model by type.

    Parameters
    ----------
    model_type : str
        One of "random_forest", "xgboost", "lightgbm"
    **kwargs
        Model-specific parameters

    Returns
    -------
    IgnitionModel
        Configured model instance
    """
    models = {
        "random_forest": RandomForestIgnition,
        "xgboost": XGBoostIgnition,
        "lightgbm": LightGBMIgnition,
    }

    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Options: {list(models.keys())}")

    return models[model_type](**kwargs)

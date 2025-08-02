"""Model wrappers for forecasting with proper typing and error handling."""

from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import xgboost as xgb

from ..training.preprocessor import FastDataPreprocessor


class ModelWrapper(ABC):
    """Abstract base class for model wrappers."""

    def __init__(self) -> None:
        self.model: Optional[Any] = None
        self.preprocessor: Optional[Any] = None
        self.is_fitted: bool = False

    @abstractmethod
    def fit(
        self, X: pd.DataFrame, y: pd.Series
    ) -> None:  # pylint: disable=invalid-name
        """Fit the model to training data."""

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:  # pylint: disable=invalid-name
        """Make predictions on input data."""

    def save(self, filepath: Union[str, Path]) -> None:
        """Save model to file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "wb") as f:
            pickle.dump(
                {
                    "model": self.model,
                    "preprocessor": self.preprocessor,
                    "is_fitted": self.is_fitted,
                },
                f,
            )

    def load(self, filepath: Union[str, Path]) -> None:
        """Load model from file."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        with open(filepath, "rb") as f:
            data = pickle.load(f)
            self.model = data["model"]
            self.preprocessor = data["preprocessor"]
            self.is_fitted = data.get("is_fitted", True)


class XGBoostWrapper(ModelWrapper):
    """Wrapper for XGBoost regression model with automatic preprocessing."""

    def __init__(  # pylint: disable=R0913
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        reg_alpha: float = 0,
        reg_lambda: float = 1,
        random_state: int = 42,
        n_jobs: int = -1,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )
        self.preprocessor = FastDataPreprocessor()
        self.feature_columns: List[str] = []

    def fit(
        self, X: pd.DataFrame, y: pd.Series
    ) -> None:  # pylint: disable=invalid-name
        """Fit XGBoost model with automatic preprocessing."""
        X_processed = self.preprocessor.fit_transform(X)
        self.feature_columns = [col for col in X_processed.columns if col != "value"]

        self.model.fit(X_processed[self.feature_columns], y)
        self.is_fitted = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:  # pylint: disable=invalid-name
        """Predict using XGBoost model with preprocessing."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        X_processed = self.preprocessor.transform(X)
        return self.model.predict(X_processed[self.feature_columns])

    def predict_processed(self, X_processed: np.ndarray) -> np.ndarray:
        """Predict using already-processed features (for unified forecaster).

        Args:
            X_processed: Numpy array of shape (n_samples, n_features) with features
                        in the same order as self.feature_columns

        Returns:
            Predictions as numpy array
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        # Ensure we have the right number of features
        if X_processed.shape[1] != len(self.feature_columns):
            raise ValueError(
                f"Expected {len(self.feature_columns)} features, "
                f"but got {X_processed.shape[1]}"
            )

        return self.model.predict(X_processed)

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted to get feature importance")

        importance_dict = dict(
            zip(self.feature_columns, self.model.feature_importances_)
        )
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))


class BaselineWrapper(ModelWrapper):
    """Baseline model using seasonal averages."""

    def __init__(self, lookback_weeks: int = 7) -> None:
        super().__init__()
        if lookback_weeks <= 0:
            raise ValueError("lookback_weeks must be positive")

        self.lookback_weeks = lookback_weeks
        self.historical_data: Optional[pd.DataFrame] = None

    def fit(
        self, X: pd.DataFrame, y: pd.Series
    ) -> None:  # pylint: disable=invalid-name
        """Store historical data for baseline predictions."""
        if X.empty or y.empty:
            raise ValueError("Training data cannot be empty")

        data = X.copy()
        data["value"] = y
        self.historical_data = data
        self.is_fitted = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict using seasonal baseline."""
        if not self.is_fitted or self.historical_data is None:
            raise ValueError("Model must be fitted before prediction")

        predictions = []

        for target_datetime in X.index:
            historical_values = self._get_historical_values(target_datetime)
            prediction = np.mean(historical_values) if historical_values else 0.0
            predictions.append(prediction)

        return np.array(predictions)

    def _get_historical_values(self, target_datetime: pd.Timestamp) -> List[float]:
        """Get historical values for a target datetime."""
        historical_values = []

        # Check exact match
        if target_datetime in self.historical_data.index:
            value = self.historical_data.loc[target_datetime, "value"]
            if isinstance(value, pd.Series):
                historical_values.append(float(value.iloc[0]))
            else:
                historical_values.append(float(value))

        # Check past weeks
        for weeks_back in range(1, self.lookback_weeks + 1):
            past_datetime = target_datetime - pd.Timedelta(weeks=weeks_back)
            if past_datetime in self.historical_data.index:
                value = self.historical_data.loc[past_datetime, "value"]
                if isinstance(value, pd.Series):
                    historical_values.append(float(value.iloc[0]))
                else:
                    historical_values.append(float(value))

        return historical_values

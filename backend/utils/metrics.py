"""Forecast metrics for forecasting."""

from typing import Dict
import numpy as np


class ForecastMetrics:
    """Forecast metrics calculation."""

    @staticmethod
    def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Percentage Error with zero-division handling."""
        mask = y_true != 0
        if not np.any(mask):
            return 0.0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    @staticmethod
    def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Weighted Absolute Percentage Error."""
        return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100

    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Error."""
        return np.mean(np.abs(y_true - y_pred))

    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Root Mean Square Error."""
        return np.sqrt(np.mean((y_true - y_pred) ** 2))

    @staticmethod
    def calculate_all(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate all metrics."""
        return {
            "mae": ForecastMetrics.mae(y_true, y_pred),
            "rmse": ForecastMetrics.rmse(y_true, y_pred),
            "mape": ForecastMetrics.mape(y_true, y_pred),
            "wape": ForecastMetrics.wape(y_true, y_pred),
        }

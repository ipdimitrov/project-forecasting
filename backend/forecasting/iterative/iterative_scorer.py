"""
Optimized iterative scorer using unified forecaster internally.
Maintains backward compatibility while providing ultra-fast performance.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from pathlib import Path
import pickle
from dataclasses import dataclass
from datetime import datetime

from ...utils.metrics import ForecastMetrics
from ...training.trainer import BaselineWrapper
from ...core.unified_forecaster import (
    UnifiedFeatureEngine,
    UnifiedIterativeForecaster,
    ForecastingConfig,
)
from ...core.feature_config import DEFAULT_FEATURE_CONFIG
from .optimization_utils import FeatureCache, BatchProcessor


@dataclass
class ForecastConfig:
    """Configuration for iterative forecasting - maintained for backward compatibility."""

    forecast_steps: int
    batch_size: int = 50
    use_feature_cache: bool = True
    use_batch_processing: bool = True
    progress_interval: int = 500
    save_intermediate: bool = False
    intermediate_save_interval: int = 1000


class IterativeScorer:
    """Optimized iterative forecasting using unified engine internally."""

    def __init__(
        self,
        model_path: str,
        preprocessor_path: Optional[str] = None,
    ):
        """Initialize iterative scorer with unified forecaster."""
        self.model_path = Path(model_path)
        self.preprocessor_path = Path(preprocessor_path) if preprocessor_path else None

        self.model = self._load_model()
        self.is_baseline = isinstance(self.model, BaselineWrapper)
        self.preprocessor = self._load_preprocessor()

        if not self.is_baseline:
            feature_engine = UnifiedFeatureEngine(DEFAULT_FEATURE_CONFIG)
            self._unified_forecaster = UnifiedIterativeForecaster(
                self.model,
                feature_engine,
                ForecastingConfig(
                    batch_size=200,
                    progress_interval=1000,
                    use_float32=True,
                ),
            )

        self.feature_cache = FeatureCache()
        self.batch_processor = BatchProcessor()

    def _load_model(self):
        """Load the trained model."""
        with open(self.model_path, "rb") as f:
            model = pickle.load(f)

        if hasattr(model, "is_fitted") and not model.is_fitted:
            raise ValueError(f"Model at {self.model_path} is not fitted")

        return model

    def _load_preprocessor(self):
        """Load the preprocessor - only needed for legacy baseline models."""
        if self.is_baseline:
            return None

        if self.preprocessor_path and self.preprocessor_path.exists():
            with open(self.preprocessor_path, "rb") as f:
                preprocessor = pickle.load(f)
        else:
            preprocessor_path = self.model_path.parent / "preprocessor.pkl"
            if preprocessor_path.exists():
                with open(preprocessor_path, "rb") as f:
                    preprocessor = pickle.load(f)
            else:
                # Create a new unified preprocessor instead of raising error
                from ...training.preprocessor import FastDataPreprocessor

                preprocessor = FastDataPreprocessor()
                preprocessor.is_fitted = False

        return preprocessor

    def pure_iterative_forecast(
        self, historical_data: pd.DataFrame, config: ForecastConfig
    ) -> Dict:
        """Generate optimized iterative forecast using unified engine."""
        if self.is_baseline:
            return self._pure_iterative_baseline(historical_data, config)
        else:
            return self._unified_iterative_forecast(historical_data, config)

    def _unified_iterative_forecast(
        self, historical_data: pd.DataFrame, config: ForecastConfig
    ) -> Dict:
        """Use unified forecaster for ultra-fast XGBoost forecasting."""
        # Fit the feature engine if preprocessor is not fitted
        if hasattr(self.preprocessor, "is_fitted") and not self.preprocessor.is_fitted:
            self._unified_forecaster.feature_engine.fit_transform(historical_data)
            self.preprocessor.is_fitted = True

        # Use unified forecaster
        result = self._unified_forecaster.forecast(
            historical_data, config.forecast_steps
        )

        # Convert result to legacy format for backward compatibility
        return {
            "predictions": result["predictions"],
            "timestamps": result["timestamps"],
            "model_type": "XGBoostWrapper",
            "forecast_steps": config.forecast_steps,
            "total_time_seconds": result["total_time_seconds"],
            "predictions_per_second": result["predictions_per_second"],
            "feature_computation_time": 0,  # Not tracked separately in unified
            "prediction_time": result["total_time_seconds"],  # Approximate
            "cache_hits": 0,  # Not applicable in unified approach
            "cache_misses": 0,
        }

    def _pure_iterative_baseline(
        self, historical_data: pd.DataFrame, config: ForecastConfig
    ) -> Dict:
        """Optimized baseline forecasting with proper iterative updating."""
        start_time = datetime.now()

        # Create working copy of historical data
        current_data = historical_data.copy()
        predictions = np.zeros(config.forecast_steps, dtype=np.float32)
        timestamps = []

        for i in range(config.forecast_steps):
            # Generate next timestamp
            next_timestamp = current_data.index[-1] + pd.Timedelta(hours=1)
            timestamps.append(next_timestamp)

            # Create temporary row for prediction
            temp_row = pd.DataFrame({"value": [np.nan]}, index=[next_timestamp])

            # Get prediction using current historical context
            pred = self.model.predict(temp_row)[0]
            predictions[i] = pred

            # Add the prediction to our growing dataset
            temp_row.iloc[0, 0] = pred  # Set the actual predicted value
            current_data = pd.concat([current_data, temp_row])

            # Periodically update the model's historical data for better pattern matching
            if i % 168 == 0 or i == config.forecast_steps - 1:  # Every week or at end
                self.model.historical_data = current_data.copy()

            # Memory management - keep reasonable history size
            if len(current_data) > 8760 * 2:  # Keep 2 years max
                current_data = current_data.tail(8760 * 2)
                self.model.historical_data = current_data.copy()

            # Progress reporting
            if (i + 1) % config.progress_interval == 0:
                elapsed = (datetime.now() - start_time).total_seconds()
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (config.forecast_steps - i - 1) / rate if rate > 0 else 0
                print(
                    f"Baseline Progress: {i + 1}/{config.forecast_steps} \
                        | {rate:.1f} pred/sec | ETA: {eta:.0f}s"
                )

        total_time = (datetime.now() - start_time).total_seconds()

        return {
            "predictions": predictions,
            "timestamps": timestamps,
            "total_time_seconds": total_time,
            "predictions_per_second": (
                config.forecast_steps / total_time if total_time > 0 else 0
            ),
            "model_type": "BaselineWrapper",
        }

    def evaluate_against_actual(
        self,
        historical_data: pd.DataFrame,
        test_data: pd.DataFrame,
        config: ForecastConfig,
    ) -> Dict:
        """Evaluate iterative forecast against actual test data."""
        forecast_result = self.pure_iterative_forecast(
            historical_data, ForecastConfig(forecast_steps=len(test_data))
        )

        y_true = np.asarray(test_data["value"].values)
        y_pred = forecast_result["predictions"]

        metrics = ForecastMetrics.calculate_all(y_true, y_pred)

        return {
            "metrics": metrics,
            "forecast_result": forecast_result,
            "test_period": {
                "start": test_data.index[0].isoformat(),
                "end": test_data.index[-1].isoformat(),
                "samples": len(test_data),
            },
        }

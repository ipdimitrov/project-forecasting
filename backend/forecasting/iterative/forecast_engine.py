"""
Main forecast engine that coordinates iterative forecasting operations.
"""

import pandas as pd
from typing import Dict, Optional
from pathlib import Path
import json
from datetime import datetime

from .iterative_scorer import IterativeScorer, ForecastConfig
from ...core.models import BaselineWrapper


class ForecastEngine:
    """Main engine for coordinating iterative forecasting operations."""

    def __init__(self, results_dir: str = "forecast_results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)

        self.scorers: Dict[str, IterativeScorer] = {}  # Cache loaded scorers

    def load_scorer(
        self,
        model_path: str,
        preprocessor_path: Optional[str] = None,
    ) -> str:
        """Load and cache an iterative scorer."""
        scorer_id = Path(model_path).stem

        if scorer_id not in self.scorers:
            self.scorers[scorer_id] = IterativeScorer(
                model_path=model_path,
                preprocessor_path=preprocessor_path,
            )

        return scorer_id

    def generate_future_forecast(
        self,
        scorer_id: str,
        historical_data: pd.DataFrame,
        forecast_steps: int,
        save_results: bool = True,
    ) -> Dict:
        """Generate future forecast using specified scorer (includes baseline comparison)."""
        if scorer_id not in self.scorers:
            raise ValueError(f"Scorer {scorer_id} not loaded")

        scorer = self.scorers[scorer_id]
        config = ForecastConfig(
            forecast_steps=forecast_steps,
            batch_size=50,
            use_feature_cache=True,
            progress_interval=500,
        )

        print(
            f"Generating {forecast_steps}-step future forecast with baseline comparison"
        )
        print(f"   Model: {scorer_id}")

        # Generate main model forecast
        result = scorer.pure_iterative_forecast(historical_data, config)

        # Also generate baseline forecast for comparison (if not already baseline)
        if not scorer.is_baseline:
            print("  Generating baseline forecast for comparison...")

            baseline_scorer = IterativeScorer.__new__(IterativeScorer)
            baseline_scorer.model = BaselineWrapper(lookback_weeks=7)
            baseline_scorer.model.fit(historical_data, historical_data["value"])
            baseline_scorer.preprocessor = None
            baseline_scorer.is_baseline = True
            baseline_scorer.feature_cache = None
            baseline_scorer.batch_processor = None

            baseline_result = baseline_scorer.pure_iterative_forecast(
                historical_data, config
            )

            # Save both forecasts
            if save_results:
                self._save_forecast_results(result, scorer_id, historical_data)
                self._save_baseline_forecast_results(
                    baseline_result, scorer_id, historical_data
                )
        else:
            if save_results:
                self._save_forecast_results(result, scorer_id, historical_data)

        return result

    def evaluate_test_forecast(
        self,
        scorer_id: str,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        save_results: bool = True,
    ) -> Dict:
        """Evaluate iterative forecast against test data."""
        if scorer_id not in self.scorers:
            raise ValueError(f"Scorer {scorer_id} not loaded")

        scorer = self.scorers[scorer_id]
        config = ForecastConfig(forecast_steps=len(test_data))

        result = scorer.evaluate_against_actual(train_data, test_data, config)

        if save_results:
            self._save_evaluation_results(result, scorer_id)

        return result

    def _save_forecast_results(
        self, result: Dict, scorer_id: str, historical_data: pd.DataFrame
    ):
        """Save forecast results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create forecast DataFrame
        last_date = historical_data.index[-1]
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(hours=1),
            periods=len(result["predictions"]),
            freq="h",
        )

        forecast_df = pd.DataFrame(
            {"Date": future_dates, "Value": result["predictions"]}
        )

        # Save forecast
        forecast_file = (
            self.results_dir / f"future_forecast_{scorer_id}_{timestamp}.csv"
        )
        forecast_df.to_csv(forecast_file, index=False)

        # Save metadata
        metadata = {
            "scorer_id": scorer_id,
            "model_type": result["model_type"],
            "forecast_steps": result["forecast_steps"],
            "forecast_period": {
                "start": future_dates[0].isoformat(),
                "end": future_dates[-1].isoformat(),
            },
            "performance": {
                "total_time_seconds": result["total_time_seconds"],
                "predictions_per_second": result["predictions_per_second"],
            },
            "generated_at": datetime.now().isoformat(),
        }

        if "cache_hits" in result:
            metadata["cache_stats"] = {
                "cache_hits": result["cache_hits"],
                "cache_misses": result["cache_misses"],
            }

        metadata_file = (
            self.results_dir / f"forecast_metadata_{scorer_id}_{timestamp}.json"
        )
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        print("Forecast results saved:")
        print(f"   Forecast: {forecast_file}")
        print(f"   Metadata: {metadata_file}")
        print(f"   Performance: {result['predictions_per_second']:.1f} pred/sec")

    def _save_baseline_forecast_results(
        self, result: Dict, main_scorer_id: str, historical_data: pd.DataFrame
    ):
        """Save baseline forecast results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create forecast DataFrame
        last_date = historical_data.index[-1]
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(hours=1),
            periods=len(result["predictions"]),
            freq="h",
        )

        forecast_df = pd.DataFrame(
            {"Date": future_dates, "Value": result["predictions"]}
        )

        # Save baseline forecast with clear naming
        forecast_file = (
            self.results_dir
            / f"future_forecast_baseline_{main_scorer_id[:8]}_{timestamp}.csv"
        )
        forecast_df.to_csv(forecast_file, index=False)

        print(f"   Baseline forecast saved: {forecast_file.name}")

    def _save_evaluation_results(self, result: Dict, scorer_id: str):
        """Save evaluation results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save evaluation summary
        eval_summary = {
            "scorer_id": scorer_id,
            "model_type": result["forecast_result"]["model_type"],
            "test_period": result["test_period"],
            "metrics": result["metrics"],
            "performance": {
                "total_time_seconds": result["forecast_result"]["total_time_seconds"],
                "predictions_per_second": result["forecast_result"][
                    "predictions_per_second"
                ],
            },
            "evaluated_at": datetime.now().isoformat(),
        }

        eval_file = self.results_dir / f"evaluation_{scorer_id}_{timestamp}.json"
        with open(eval_file, "w") as f:
            json.dump(eval_summary, f, indent=2)

        print(f"Evaluation results saved: {eval_file}")
        print(f"Test WAPE: {result['metrics']['wape']:.3f}")

"""Training module for forecasting models."""

import json
import pickle
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from backend.utils.logging_config import get_logger

from ..core.config import CrossValidationConfig, TrainingResult  # noqa: E402
from ..core.models import BaselineWrapper  # noqa: E402
from ..core.unified_forecaster import create_unified_forecaster  # noqa: E402
from ..training.preprocessor import FastDataPreprocessor  # noqa: E402
from ..utils.metrics import ForecastMetrics  # noqa: E402

logger = get_logger(__name__)


class ForecastTrainer:
    """Class for training consumtion forecasting models.
    Includes also unified forecaster for iterative forecasting.
    """

    def __init__(self, results_dir: str = "training_results") -> None:
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True, parents=True)
        self.models_dir = Path("model")
        self.models_dir.mkdir(exist_ok=True, parents=True)

    def prepare_last_month_split(
        self, data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, datetime]:
        """Split data into training (all but last month) and testing (last month)."""
        if len(data) == 0:
            raise ValueError("Cannot split empty dataset")

        last_date = data.index[-1]
        last_month_start = last_date.replace(
            day=1, hour=0, minute=0, second=0, microsecond=0
        )

        train_data = data[data.index < last_month_start]
        test_data = data[data.index >= last_month_start]

        return train_data, test_data, last_month_start

    def cross_validate_model(  # pylint: disable=R0914
        self,
        model_class: type,
        params: Dict[str, Any],
        data: pd.DataFrame,
        cv_config: CrossValidationConfig,
    ) -> TrainingResult:
        """Perform expanding window cross-validation with optimized unified forecasting."""
        model_id = str(uuid.uuid4())

        # Standard evaluation (using actuals)
        xgb_cv_scores = []
        baseline_cv_scores = []
        xgb_all_metrics = []
        baseline_all_metrics = []

        # Iterative evaluation (using predictions) - now optimized
        xgb_iterative_cv_scores = []
        baseline_iterative_cv_scores = []
        xgb_iterative_all_metrics = []
        baseline_iterative_all_metrics = []

        train_periods = []
        eval_periods = []

        splits = self._generate_cv_splits(data, cv_config)

        for fold_idx, (train_start, train_end, eval_start, eval_end) in enumerate(
            splits
        ):
            logger.info(
                "Fold %d: Training on %s to %s, Evaluating on %s to %s",
                fold_idx + 1,
                train_start.strftime("%Y-%m-%d"),
                train_end.strftime("%Y-%m-%d"),
                eval_start.strftime("%Y-%m-%d"),
                eval_end.strftime("%Y-%m-%d"),
            )
            train_data = data[(data.index >= train_start) & (data.index <= train_end)]
            eval_data = data[(data.index >= eval_start) & (data.index <= eval_end)]
            xgb_model = model_class(**params)
            xgb_model.fit(train_data, train_data["value"])
            xgb_pred_standard = xgb_model.predict(eval_data)
            xgb_pred_iterative = self._unified_iterative_forecast(
                xgb_model, train_data, eval_data
            )
            baseline_pred_standard = self._advanced_baseline_forecast(
                train_data, eval_data
            )
            baseline_pred_iterative = self._iterative_baseline_forecast(
                train_data, eval_data
            )
            xgb_metrics_standard = ForecastMetrics.calculate_all(
                eval_data["value"].values, xgb_pred_standard
            )
            baseline_metrics_standard = ForecastMetrics.calculate_all(
                eval_data["value"].values,
                baseline_pred_standard,
            )
            xgb_metrics_iterative = ForecastMetrics.calculate_all(
                eval_data["value"].values, xgb_pred_iterative
            )
            baseline_metrics_iterative = ForecastMetrics.calculate_all(
                eval_data["value"].values,
                baseline_pred_iterative,
            )
            xgb_cv_scores.append(xgb_metrics_standard["wape"])
            baseline_cv_scores.append(baseline_metrics_standard["wape"])
            xgb_all_metrics.append(xgb_metrics_standard)
            baseline_all_metrics.append(baseline_metrics_standard)
            xgb_iterative_cv_scores.append(xgb_metrics_iterative["wape"])
            baseline_iterative_cv_scores.append(baseline_metrics_iterative["wape"])
            xgb_iterative_all_metrics.append(xgb_metrics_iterative)
            baseline_iterative_all_metrics.append(baseline_metrics_iterative)
            train_periods.append(
                (
                    train_start.strftime("%Y-%m-%d %H:%M:%S"),
                    train_end.strftime("%Y-%m-%d %H:%M:%S"),
                )
            )
            eval_periods.append(
                (
                    eval_start.strftime("%Y-%m-%d %H:%M:%S"),
                    eval_end.strftime("%Y-%m-%d %H:%M:%S"),
                )
            )
            logger.info(
                "    Standard - WAPE: %.4f, Baseline: %.4f",
                xgb_metrics_standard["wape"],
                baseline_metrics_standard["wape"],
            )
            logger.info(
                "    Iterative - WAPE: %.4f, Baseline: %.4f",
                xgb_metrics_iterative["wape"],
                baseline_metrics_iterative["wape"],
            )
        sample_preprocessor = FastDataPreprocessor()
        sample_features = sample_preprocessor.fit_transform(data.head(100))
        features_used = [col for col in sample_features.columns if col != "value"]

        result = TrainingResult(
            model_id=model_id,
            model_name=model_class.__name__,
            parameters=params,
            # Standard evaluation results
            cv_scores=xgb_cv_scores,
            avg_wape=np.mean(xgb_cv_scores),
            metrics_all_folds=xgb_all_metrics,
            baseline_cv_scores=baseline_cv_scores,
            baseline_avg_wape=np.mean(baseline_cv_scores),
            baseline_metrics_all_folds=baseline_all_metrics,
            # Iterative evaluation results
            iterative_cv_scores=xgb_iterative_cv_scores,
            iterative_avg_wape=np.mean(xgb_iterative_cv_scores),
            iterative_metrics_all_folds=xgb_iterative_all_metrics,
            baseline_iterative_cv_scores=baseline_iterative_cv_scores,
            baseline_iterative_avg_wape=np.mean(baseline_iterative_cv_scores),
            baseline_iterative_metrics_all_folds=baseline_iterative_all_metrics,
            # Common fields
            train_periods=train_periods,
            eval_periods=eval_periods,
            features_used=features_used,
        )
        self._save_model_with_metadata(xgb_model, result, data)
        self.save_comprehensive_results(
            [result],
            data,
            save_format="excel",
            filename_prefix=f"cv_result_{result.model_name}",
        )

        return result

    def _unified_iterative_forecast(
        self, model, train_data: pd.DataFrame, eval_data: pd.DataFrame
    ) -> np.ndarray:
        """Perform fast iterative forecasting using unified forecaster."""
        unified_forecaster = create_unified_forecaster(model)
        unified_forecaster.feature_engine.fit_transform(train_data)
        result = unified_forecaster.forecast(train_data, len(eval_data))
        return result["predictions"]

    def _advanced_baseline_forecast(
        self, train_data: pd.DataFrame, eval_data: pd.DataFrame
    ) -> np.ndarray:
        """Advanced baseline: average for same hour/day of week from past 7 weeks."""
        predictions = []

        for timestamp, _ in eval_data.iterrows():
            hour = timestamp.hour
            day_of_week = timestamp.dayofweek
            historical_values = []
            for week_back in range(1, 8):  # Past 7 weeks
                target_time = timestamp - timedelta(weeks=week_back)

                # Find the closest matching time in training data
                if target_time in train_data.index:
                    historical_values.append(train_data.loc[target_time, "value"])
                else:
                    # Find closest time within a few hours
                    tolerance = timedelta(hours=2)
                    mask = (
                        (train_data.index >= target_time - tolerance)
                        & (train_data.index <= target_time + tolerance)
                        & (train_data.index.hour == hour)
                        & (train_data.index.dayofweek == day_of_week)
                    )

                    if mask.any():
                        historical_values.append(train_data.loc[mask, "value"].iloc[0])
            if len(historical_values) >= 3:
                historical_values.sort()
                # Remove highest and lowest
                trimmed_values = historical_values[1:-1]
                prediction = np.mean(trimmed_values)
            elif len(historical_values) > 0:
                prediction = np.mean(historical_values)
            else:
                # Fallback to overall mean of training data
                prediction = train_data["value"].mean()

            predictions.append(prediction)

        return np.array(predictions)

    def _iterative_baseline_forecast(  # pylint: disable=R0914
        self, train_data: pd.DataFrame, eval_data: pd.DataFrame
    ) -> np.ndarray:
        """Fast iterative baseline using pre-allocation and vectorized operations."""
        forecast_steps = len(eval_data)

        baseline_model = BaselineWrapper(lookback_weeks=7)
        baseline_model.fit(train_data, train_data["value"])

        predictions = np.zeros(forecast_steps, dtype=np.float32)

        start_time = eval_data.index[0]
        timestamps = pd.date_range(start=start_time, periods=forecast_steps, freq="h")

        hist_size = len(train_data)
        total_size = hist_size + forecast_steps

        extended_values = np.zeros(total_size, dtype=np.float32)
        extended_index = np.zeros(total_size, dtype="datetime64[h]")

        extended_values[:hist_size] = train_data["value"].values.astype(np.float32)
        extended_index[:hist_size] = train_data.index.values.astype("datetime64[h]")
        extended_index[hist_size:] = timestamps.values.astype("datetime64[h]")

        working_data = train_data.copy()

        for i in range(forecast_steps):
            current_timestamp = timestamps[i]
            pred_row = pd.DataFrame({"value": [np.nan]}, index=[current_timestamp])
            pred = baseline_model.predict(pred_row)[0]
            predictions[i] = pred

            extended_values[hist_size + i] = pred

            if i % 10 == 0 or i == forecast_steps - 1:
                new_row = pd.DataFrame({"value": [pred]}, index=[current_timestamp])
                working_data = pd.concat([working_data, new_row])
                baseline_model.historical_data = working_data

                if len(working_data) > 8760 * 2:
                    working_data = working_data.tail(8760 * 2)
                    baseline_model.historical_data = working_data
        return predictions

    def _generate_cv_splits(  # pylint: disable=R0914
        self, data: pd.DataFrame, cv_config: CrossValidationConfig
    ) -> List[Tuple[datetime, datetime, datetime, datetime]]:
        """Generate expanding window cross-validation time splits, excluding last month."""
        splits = []

        eval_period_days = cv_config.eval_period_weeks * 7
        min_train_days = cv_config.min_train_weeks * 7

        # Exclude last month from available data
        data_end = data.index[-1]
        last_month_start = data_end.replace(
            day=1, hour=0, minute=0, second=0, microsecond=0
        )
        available_data_end = last_month_start - timedelta(
            hours=1
        )  # End before last month

        data_start = data.index[0]

        logger.info(
            "Data available for CV: %s to %s",
            data_start.strftime("%Y-%m-%d"),
            available_data_end.strftime("%Y-%m-%d"),
        )
        logger.info(
            "Last month reserved for testing: %s to %s",
            last_month_start.strftime("%Y-%m-%d"),
            data_end.strftime("%Y-%m-%d"),
        )

        for fold in range(cv_config.cv_folds):
            # Calculate evaluation period - working backwards from available_data_end
            eval_end = available_data_end - timedelta(days=eval_period_days * fold)
            eval_start = eval_end - timedelta(days=eval_period_days - 1)

            # Training ends 1 day before evaluation starts (not same day!)
            train_end = eval_start - timedelta(days=1)

            # Calculate training start - expanding window
            train_start = data_start

            # Ensure minimum training period
            min_train_end = data_start + timedelta(days=min_train_days - 1)
            if train_end < min_train_end:
                # If we can't maintain minimum training period, stop creating folds
                break

            # Ensure we don't go before data start or after available data
            if train_start >= data_start and eval_end <= available_data_end:
                splits.append((train_start, train_end, eval_start, eval_end))

        # Return in chronological order (earliest eval periods first)
        return list(reversed(splits))

    def _save_model_with_metadata(
        self, model, result: TrainingResult, data: pd.DataFrame
    ) -> None:
        """Save trained model with comprehensive metadata including both evaluation types."""
        model_filename = f"{result.model_name}_{result.model_id}.pkl"
        metadata_filename = f"{result.model_name}_{result.model_id}_metadata.json"

        model_path = self.models_dir / model_filename
        metadata_path = self.models_dir / metadata_filename

        # Save model
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        # Calculate average metrics across folds for both evaluation types
        def calc_avg_metrics(metrics_list):
            if not metrics_list:
                return {}
            avg_metrics = {}
            for metric in metrics_list[0].keys():
                values = [fold[metric] for fold in metrics_list]
                avg_metrics[f"avg_{metric}"] = np.mean(values)
                avg_metrics[f"std_{metric}"] = np.std(values)
            return avg_metrics

        standard_avg_metrics = calc_avg_metrics(result.metrics_all_folds)
        baseline_standard_avg_metrics = calc_avg_metrics(
            result.baseline_metrics_all_folds
        )
        iterative_avg_metrics = calc_avg_metrics(result.iterative_metrics_all_folds)
        baseline_iterative_avg_metrics = calc_avg_metrics(
            result.baseline_iterative_metrics_all_folds
        )

        # Save metadata
        metadata = {
            "model_id": result.model_id,
            "model_name": result.model_name,
            "model_file": model_filename,
            "parameters": result.parameters,
            "features_used": result.features_used,
            "standard_evaluation": {
                "cv_performance": {
                    "avg_wape": result.avg_wape,
                    "cv_scores": result.cv_scores,
                    "baseline_avg_wape": result.baseline_avg_wape,
                    "baseline_cv_scores": result.baseline_cv_scores,
                },
                "all_metrics": result.metrics_all_folds,
                "baseline_metrics": result.baseline_metrics_all_folds,
                "average_metrics": standard_avg_metrics,
                "baseline_average_metrics": baseline_standard_avg_metrics,
            },
            "iterative_evaluation": {
                "cv_performance": {
                    "avg_wape": result.iterative_avg_wape,
                    "cv_scores": result.iterative_cv_scores,
                    "baseline_avg_wape": result.baseline_iterative_avg_wape,
                    "baseline_cv_scores": result.baseline_iterative_cv_scores,
                },
                "all_metrics": result.iterative_metrics_all_folds,
                "baseline_metrics": result.baseline_iterative_metrics_all_folds,
                "average_metrics": iterative_avg_metrics,
                "baseline_average_metrics": baseline_iterative_avg_metrics,
            },
            "train_periods": result.train_periods,
            "eval_periods": result.eval_periods,
            "data_info": {
                "total_records": len(data),
                "data_start": data.index[0].strftime("%Y-%m-%d %H:%M:%S"),
                "data_end": data.index[-1].strftime("%Y-%m-%d %H:%M:%S"),
            },
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info("Model saved: %s", model_path)
        logger.info("Metadata saved: %s", metadata_path)

    def save_results(
        self, result: TrainingResult, filename_prefix: str = "training_result"
    ) -> str:
        """Save training results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / "%s_%s.json" % (filename_prefix, timestamp)

        result_dict = result.to_dict()

        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(result_dict, f, indent=2)

        return str(results_file)

    def save_comprehensive_results(  # pylint: disable=R0915, R0914, R0912
        self,
        results: List[TrainingResult],
        data: pd.DataFrame,
        save_format: str = "excel",
        filename_prefix: str = "comprehensive_results",
    ) -> Dict[str, str]:
        """Save comprehensive results with unified forecaster optimization info."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_files = {}

        if not results:
            raise ValueError("No results to save")

        # Prepare comprehensive data
        comprehensive_data = []

        for result in results:
            # Basic result info
            base_info = {
                "model_id": result.model_id,
                "model_name": result.model_name,
                "parameters": str(result.parameters),
                "features_count": len(result.features_used),
            }

            # Standard evaluation metrics
            standard_metrics = {
                "standard_avg_wape": result.avg_wape,
                "standard_baseline_wape": result.baseline_avg_wape,
                "standard_improvement_pct": (
                    (
                        (result.baseline_avg_wape - result.avg_wape)
                        / result.baseline_avg_wape
                        * 100
                    )
                    if result.baseline_avg_wape > 0
                    else 0
                ),
            }

            # Iterative evaluation metrics
            iterative_metrics = {
                "iterative_avg_wape": result.iterative_avg_wape,
                "iterative_baseline_wape": result.baseline_iterative_avg_wape,
                "iterative_improvement_pct": (
                    (
                        (result.baseline_iterative_avg_wape - result.iterative_avg_wape)
                        / result.baseline_iterative_avg_wape
                        * 100
                    )
                    if result.baseline_iterative_avg_wape > 0
                    else 0
                ),
            }

            # Cross-validation details
            cv_details = {
                "cv_folds": len(result.cv_scores),
                "cv_std_wape": np.std(result.cv_scores),
                "iterative_cv_std_wape": np.std(result.iterative_cv_scores),
            }

            # Combine all info
            comprehensive_data.append(
                {**base_info, **standard_metrics, **iterative_metrics, **cv_details}
            )

        # Convert to DataFrame for easy saving
        results_df = pd.DataFrame(comprehensive_data)

        if save_format in ["excel", "both"]:
            excel_file = self.results_dir / f"{filename_prefix}_{timestamp}.xlsx"

            with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
                results_df.to_excel(writer, sheet_name="Results_Summary", index=False)

                model_config_data = []
                for result in results:
                    for param, value in result.parameters.items():
                        model_config_data.append(
                            {
                                "model_id": result.model_id,
                                "config_type": "parameter",
                                "name": param,
                                "value": str(value),
                            }
                        )

                    for i, feature in enumerate(result.features_used):
                        model_config_data.append(
                            {
                                "model_id": result.model_id,
                                "config_type": "feature",
                                "name": feature,
                                "value": str(i),
                            }
                        )

                if model_config_data:
                    config_df = pd.DataFrame(model_config_data)
                    config_df.to_excel(
                        writer, sheet_name="Model_Configuration", index=False
                    )

                cv_metrics_data = []
                for result in results:
                    if result.metrics_all_folds:
                        for fold_idx, fold_metrics in enumerate(
                            result.metrics_all_folds
                        ):
                            for metric_name, metric_value in fold_metrics.items():
                                cv_metrics_data.append(
                                    {
                                        "model_id": result.model_id,
                                        "fold": fold_idx + 1,
                                        "evaluation_type": "standard",
                                        "metric": metric_name,
                                        "value": metric_value,
                                    }
                                )

                    if result.baseline_metrics_all_folds:
                        for fold_idx, fold_metrics in enumerate(
                            result.baseline_metrics_all_folds
                        ):
                            for metric_name, metric_value in fold_metrics.items():
                                cv_metrics_data.append(
                                    {
                                        "model_id": result.model_id,
                                        "fold": fold_idx + 1,
                                        "evaluation_type": "baseline_standard",
                                        "metric": metric_name,
                                        "value": metric_value,
                                    }
                                )

                    if result.iterative_metrics_all_folds:
                        for fold_idx, fold_metrics in enumerate(
                            result.iterative_metrics_all_folds
                        ):
                            for metric_name, metric_value in fold_metrics.items():
                                cv_metrics_data.append(
                                    {
                                        "model_id": result.model_id,
                                        "fold": fold_idx + 1,
                                        "evaluation_type": "iterative",
                                        "metric": metric_name,
                                        "value": metric_value,
                                    }
                                )

                    if result.baseline_iterative_metrics_all_folds:
                        for fold_idx, fold_metrics in enumerate(
                            result.baseline_iterative_metrics_all_folds
                        ):
                            for metric_name, metric_value in fold_metrics.items():
                                cv_metrics_data.append(
                                    {
                                        "model_id": result.model_id,
                                        "fold": fold_idx + 1,
                                        "evaluation_type": "baseline_iterative",
                                        "metric": metric_name,
                                        "value": metric_value,
                                    }
                                )

                if cv_metrics_data:
                    cv_metrics_df = pd.DataFrame(cv_metrics_data)
                    cv_metrics_df.to_excel(
                        writer, sheet_name="CV_Detailed_Metrics", index=False
                    )

                metadata_info = []
                metadata_info.append(
                    {
                        "info_type": "data",
                        "key": "total_records",
                        "value": str(len(data)),
                    }
                )
                metadata_info.append(
                    {
                        "info_type": "data",
                        "key": "data_start",
                        "value": data.index[0].strftime("%Y-%m-%d %H:%M:%S"),
                    }
                )
                metadata_info.append(
                    {
                        "info_type": "data",
                        "key": "data_end",
                        "value": data.index[-1].strftime("%Y-%m-%d %H:%M:%S"),
                    }
                )
                metadata_info.append(
                    {
                        "info_type": "data",
                        "key": "date_range_days",
                        "value": str((data.index[-1] - data.index[0]).days),
                    }
                )
                metadata_info.append(
                    {
                        "info_type": "processing",
                        "key": "optimization_engine",
                        "value": "unified_forecaster",
                    }
                )
                metadata_info.append(
                    {
                        "info_type": "processing",
                        "key": "timestamp",
                        "value": timestamp,
                    }
                )

                if results:
                    first_result = results[0]
                    metadata_info.append(
                        {
                            "info_type": "cross_validation",
                            "key": "cv_folds",
                            "value": str(len(first_result.cv_scores)),
                        }
                    )
                    if first_result.train_periods:
                        metadata_info.append(
                            {
                                "info_type": "cross_validation",
                                "key": "train_periods_sample",
                                "value": str(
                                    first_result.train_periods[0]
                                    if first_result.train_periods
                                    else "N/A"
                                ),
                            }
                        )
                    if first_result.eval_periods:
                        metadata_info.append(
                            {
                                "info_type": "cross_validation",
                                "key": "eval_periods_sample",
                                "value": str(
                                    first_result.eval_periods[0]
                                    if first_result.eval_periods
                                    else "N/A"
                                ),
                            }
                        )

                metadata_df = pd.DataFrame(metadata_info)
                metadata_df.to_excel(writer, sheet_name="Metadata", index=False)

            saved_files["excel"] = str(excel_file)
            logger.info("Excel results saved: %s", excel_file)

        if save_format in ["csv", "both"]:
            csv_file = self.results_dir / f"{filename_prefix}_{timestamp}.csv"
            results_df.to_csv(csv_file, index=False)
            saved_files["csv"] = str(csv_file)
            logger.info("CSV results saved: %s", csv_file)

        return saved_files

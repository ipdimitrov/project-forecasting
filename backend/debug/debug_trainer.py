"""Debug trainer for testing and evaluating forecasting models."""

import itertools
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
import pandas as pd

# Add project root to path before local imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Local imports (these need to be after path setup)
from backend.core.config import CrossValidationConfig, TrainingResult  # noqa: E402
from backend.core.models import XGBoostWrapper  # noqa: E402
from backend.training.preprocessor import FastDataPreprocessor  # noqa: E402
from backend.training.trainer import ForecastTrainer  # noqa: E402
from backend.scripts.generate_future_forecast import load_csv_data  # noqa: E402
from backend.utils.logging_config import get_logger  # noqa: E402

logger = get_logger(__name__)


class DebugTrainer:
    """Debug trainer for comprehensive model evaluation."""

    def __init__(self, csv_file: str) -> None:
        self.csv_file = csv_file
        self.trainer = ForecastTrainer(results_dir="debug_results")
        self.preprocessor = FastDataPreprocessor()

    def parameter_search(
        self,
        model_class: type,
        param_grid: Dict[str, List[Any]],
        data: pd.DataFrame,
        cv_config: CrossValidationConfig,
    ) -> List[TrainingResult]:
        """Run parameter search with cross-validation."""
        results = []
        param_combinations = list(self._grid_search_params(param_grid))

        logger.info(f"Running {len(param_combinations)} parameter combinations...")

        for i, params in enumerate(param_combinations, 1):
            logger.info(f"Training model {i}/{len(param_combinations)}: {params}")
            result = self.trainer.cross_validate_model(
                model_class, params, data, cv_config
            )
            results.append(result)
            print(
                f"  WAPE: {result.avg_wape:.4f}, Baseline: {result.baseline_avg_wape:.4f}"
            )

        return results

    def _grid_search_params(
        self, param_grid: Dict[str, List[Any]]
    ) -> Iterator[Dict[str, Any]]:
        """Generate parameter combinations for grid search."""
        keys = param_grid.keys()
        values = param_grid.values()

        for combination in itertools.product(*values):
            yield dict(zip(keys, combination))


def comprehensive_xgboost_evaluation(
    csv_file: str,
    param_grid: Optional[Dict[str, List[Any]]] = None,
    cv_config: Optional[CrossValidationConfig] = None,
    save_format: str = "excel",  # "excel", "csv", or "both"
) -> Optional[Dict[str, str]]:
    """Run comprehensive XGBoost evaluation with consolidated results saving."""
    if not Path(csv_file).exists():
        logger.error(f"Error: {csv_file} not found.")
        return None

    if param_grid is None:
        # This grid is if we want grid search
        param_grid = {
            "n_estimators": [400, 500, 650],
            "max_depth": [5, 6, 7],
            "learning_rate": [0.03, 0.01],
            "reg_alpha": [0.1],
            "reg_lambda": [0.1],
            "subsample": [0.9],
            "colsample_bytree": [0.8],
        }

    if cv_config is None:
        cv_config = CrossValidationConfig(
            min_train_weeks=26, eval_period_weeks=6, cv_folds=4
        )

    try:
        trainer = DebugTrainer(csv_file)
        data = load_csv_data(csv_file)

        print(
            f"Cross-validation: {cv_config.cv_folds} folds, "
            f"{cv_config.min_train_weeks} week training, "
            f"{cv_config.eval_period_weeks} week evaluation"
        )

        xgb_results = trainer.parameter_search(
            XGBoostWrapper, param_grid, data, cv_config
        )

        # Use the comprehensive results saving - FIXED: use internal trainer
        saved_files = trainer.trainer.save_comprehensive_results(
            xgb_results,
            data,
            save_format=save_format,
            filename_prefix="xgboost_evaluation",
        )

        best_result = min(xgb_results, key=lambda r: r.avg_wape)
        improvement = (
            (best_result.baseline_avg_wape - best_result.avg_wape)
            / best_result.baseline_avg_wape
            * 100
        )

        print("\nResults saved:")
        for file_type, filepath in saved_files.items():
            print(f"  {file_type}: {filepath}")

        print("\nBest model performance:")
        print(f"  WAPE: {best_result.avg_wape:.4f}")
        print(f"  Baseline WAPE: {best_result.baseline_avg_wape:.4f}")
        print(f"  Improvement: {improvement:.2f}%")
        print(f"  Features used: {len(best_result.features_used)}")

        if best_result.metrics_all_folds:
            print("\nAll metrics (best model average):")
            for metric in best_result.metrics_all_folds[0].keys():
                avg_val = np.mean(
                    [fold[metric] for fold in best_result.metrics_all_folds]
                )
                std_val = np.std(
                    [fold[metric] for fold in best_result.metrics_all_folds]
                )
                print(f"  {metric.upper()}: {avg_val:.4f} ± {std_val:.4f}")

        return saved_files

    except Exception as e:
        logger.error(f"Error during evaluation: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    comprehensive_xgboost_evaluation(
        csv_file="input_data/file1.csv",  # file1.csv or file2.csv
        param_grid=None,
        cv_config=CrossValidationConfig(
            min_train_weeks=26, eval_period_weeks=6, cv_folds=5
        ),
        save_format="excel",
    )

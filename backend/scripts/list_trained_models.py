#!/usr/bin/env python3
"""List available trained models for future forecasting."""

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    """List available trained models."""
    parser = argparse.ArgumentParser(description="List available trained models")
    parser.add_argument(
        "--results_dir",
        default="debug_results",
        help="Directory containing model results (default: debug_results)",
    )
    parser.add_argument(
        "--models_dir",
        default="debug_models",
        help="Directory containing saved models (default: debug_models)",
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    models_dir = Path(args.models_dir)

    print("Scanning for trained models...")
    print(f"Results: {results_dir}")
    print(f"Models: {models_dir}")

    if not models_dir.exists():
        print(f"Models directory not found: {models_dir}")
        return 1

    model_dirs = [d for d in models_dir.iterdir() if d.is_dir()]

    if not model_dirs:
        print(f"No model directories found in {models_dir}")
        return 1

    print(f"\nFound {len(model_dirs)} trained models:\n")

    models_info = []

    for model_dir in model_dirs:
        model_pkl = model_dir / "model.pkl"
        metadata_json = model_dir / "metadata.json"
        preprocessor_pkl = model_dir / "preprocessor.pkl"

        if not model_pkl.exists():
            continue

        info = {
            "model_id": model_dir.name,
            "model_path": str(model_pkl),
            "preprocessor_path": (
                str(preprocessor_pkl) if preprocessor_pkl.exists() else "N/A"
            ),
            "has_metadata": metadata_json.exists(),
        }

        if metadata_json.exists():
            with open(metadata_json, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            info.update(
                {
                    "model_name": metadata.get("model_name", "Unknown"),
                    "avg_wape": metadata.get("avg_wape", "N/A"),
                    "training_date": metadata.get("training_date", "N/A"),
                    "parameters": metadata.get("parameters", {}),
                }
            )

        models_info.append(info)

    # Sort by WAPE if available
    models_info.sort(key=lambda x: x.get("avg_wape", float("inf")))

    # Display models
    for i, info in enumerate(models_info, 1):
        print(f"{i}. Model ID: {info['model_id'][:8]}...")
        print(f"   Type: {info.get('model_name', 'Unknown')}")
        print(f"   WAPE: {info.get('avg_wape', 'N/A')}")
        print(f"   Model: {info['model_path']}")
        print(f"   Preprocessor: {info['preprocessor_path']}")
        print(f"   Training Date: {info.get('training_date', 'N/A')}")
        if info.get("parameters"):
            params_str = ", ".join([f"{k}={v}" for k, v in info["parameters"].items()])
            print(f"   Parameters: {params_str}")
        print()

    print("To generate forecasts, use:")
    print(
        "   python scripts/generate_future_forecast.py --model_path <model_path> \
            --data_path <data_path>"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Standalone script for generating future forecasts using trained models.
Includes automatic Plotly visualization generation.
Example Usage:
python generate_future_forecast.py \
    --model_path model/XGBoostWrapper_cc05ab35-2cfd-4108-a22a-b5aa7ede503d.pkl \
    --data_path input_data/file2.csv \
    --forecast_years 1.0 \
    --folder_name experiment_1 \
    --visualize
"""
import argparse
import sys
from pathlib import Path
import json
import traceback

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
sys.path.insert(0, str(project_root))

from backend.forecasting.iterative import ForecastEngine  # noqa: E402


def load_csv_data(csv_file: str) -> pd.DataFrame:
    """Load and preprocess data from CSV file."""
    print(f"Loading data from {csv_file}...")
    data = pd.read_csv(csv_file)

    required_columns = {"Date", "Value"}
    if not required_columns.issubset(data.columns):
        raise ValueError(f"CSV must contain columns: {required_columns}")

    data["Date"] = pd.to_datetime(data["Date"])
    data = data.set_index("Date")
    data = data.rename(columns={"Value": "value"})
    data = data.sort_index()
    data = data.dropna()

    print(f"Loaded {len(data)} records from {data.index[0]} to {data.index[-1]}")
    return data


def create_forecast_plotly_express(  # pylint: disable=R0914
    xgb_forecast: pd.DataFrame,
    baseline_forecast: pd.DataFrame,
    historical_data: pd.DataFrame,
    output_dir: Path,
    scorer_id: str,
) -> str:
    """Create simple Plotly Express visualization with forecasts and shifted historical data."""

    # Prepare data for Plotly Express
    xgb_forecast["Date"] = pd.to_datetime(xgb_forecast["Date"])
    baseline_forecast["Date"] = pd.to_datetime(baseline_forecast["Date"])

    # Combine forecasts into a single DataFrame for Plotly Express
    xgb_df = xgb_forecast.copy()
    xgb_df["Forecast_Type"] = "XGBoost Model"
    xgb_df = xgb_df.rename(columns={"Value": "Forecast_Value"})

    baseline_df = baseline_forecast.copy()
    baseline_df["Forecast_Type"] = "Baseline"
    baseline_df = baseline_df.rename(columns={"Value": "Forecast_Value"})

    # Create shifted historical data (53 weeks = 8904 hours)
    shifted_historical = historical_data.copy()
    weeks_53_hours = 53 * 7 * 24
    shifted_historical.index = shifted_historical.index + pd.Timedelta(
        hours=weeks_53_hours
    )

    # Filter shifted data to match forecast period
    forecast_start = xgb_forecast["Date"].min()
    forecast_end = xgb_forecast["Date"].max()

    shifted_mask = (shifted_historical.index >= forecast_start) & (
        shifted_historical.index <= forecast_end
    )
    shifted_filtered = shifted_historical[shifted_mask].copy()

    # Prepare shifted data for combination
    if len(shifted_filtered) > 0:
        shifted_df = shifted_filtered.reset_index()
        shifted_df["Forecast_Type"] = "Historical (53 weeks ago)"
        shifted_df = shifted_df.rename(
            columns={"Date": "Date", "value": "Forecast_Value"}
        )
    else:
        shifted_df = pd.DataFrame(columns=["Date", "Forecast_Value", "Forecast_Type"])

    # Combine all data
    combined_df = pd.concat([xgb_df, baseline_df, shifted_df], ignore_index=True)

    # Create Plotly Express line plot
    fig = px.line(
        combined_df,
        x="Date",
        y="Forecast_Value",
        color="Forecast_Type",
        title=f"Hourly Forecast with Historical Reference - Model: {scorer_id[:8]}",
        labels={
            "Date": "Date (Hourly)",
            "Forecast_Value": "Value",
            "Forecast_Type": "Data Type",
        },
        color_discrete_map={
            "XGBoost Model": "#FF6B6B",  # Red
            "Baseline": "#4ECDC4",  # Teal
            "Historical (53 weeks ago)": "#9B59B6",  # Purple
        },
    )

    # Customize the layout
    fig.update_layout(
        template="plotly_white",
        width=1200,
        height=600,
        hovermode="x unified",
        legend={"yanchor": "top", "y": 0.99, "xanchor": "left", "x": 0.01},
    )
    # Update line styles
    fig.update_traces(
        line={"width": 2},
        hovertemplate="<b>%{fullData.name}</b><br>Date: %{x|%Y-%m-%d %H:%M}\
            Value: %{y}<extra></extra>",
    )

    # Apply different line styles for each trace
    for trace in fig.data:
        if trace.name == "Baseline":
            trace.update(line={"dash": "dash", "width": 2})
        elif trace.name == "Historical (53 weeks ago)":
            trace.update(line={"dash": "dot", "width": 2})

    # Add grid and format x-axis for hourly data
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor="lightgray",
        tickformat="%Y-%m-%d %H:%M",
        tickangle=45,
    )
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")

    # Save the plot
    html_filename = output_dir / f"forecast_px_{scorer_id[:20]}.html"
    fig.write_html(str(html_filename))

    print(f"Hourly forecast with 53-week historical comparison saved: {html_filename}")
    return str(html_filename)


def create_future_forecast_visualization(
    historical_data: pd.DataFrame,
    xgb_forecast: pd.DataFrame,
    baseline_forecast: pd.DataFrame,
    output_dir: Path,
    scorer_id: str,
) -> str:
    """Create comprehensive Plotly visualization for future forecasts."""

    context_days = 30
    context_hours = context_days * 24
    historical_context = historical_data.tail(context_hours).copy()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=historical_context.index,
            y=historical_context["value"],
            mode="lines",
            name="Historical Data",
            line={"color": "blue", "width": 1.5},
            hovertemplate="<b>Historical</b><br>Date: %{x}<br>\
                Value: %{y}<extra></extra>",
        )
    )

    xgb_forecast["Date"] = pd.to_datetime(xgb_forecast["Date"])
    baseline_forecast["Date"] = pd.to_datetime(baseline_forecast["Date"])

    fig.add_trace(
        go.Scatter(
            x=xgb_forecast["Date"],
            y=xgb_forecast["Value"],
            mode="lines",
            name="XGBoost Forecast",
            line={"color": "red", "width": 2},
            hovertemplate="<b>XGBoost</b><br>Date: %{x}<br>\
                Value: %{y}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=baseline_forecast["Date"],
            y=baseline_forecast["Value"],
            mode="lines",
            name="Baseline Forecast",
            line={"color": "orange", "width": 2, "dash": "dash"},
            hovertemplate="<b>Baseline</b><br>Date: %{x}<br>\
                Value: %{y}<extra></extra>",
        )
    )
    forecast_start = xgb_forecast["Date"].iloc[0]
    fig.add_shape(
        type="line",
        x0=forecast_start,
        x1=forecast_start,
        y0=0,
        y1=1,
        yref="paper",
        line={"color": "gray", "width": 2, "dash": "dot"},
    )
    fig.add_annotation(
        x=forecast_start,
        y=1,
        yref="paper",
        text="Forecast Start",
        showarrow=False,
        yshift=10,
    )

    fig.update_layout(
        title=f"Forecast - Model: {scorer_id[:20]}",
        xaxis_title="Date",
        yaxis_title="Value",
        hovermode="x unified",
        template="plotly_white",
        width=1200,
        height=600,
        legend={"yanchor": "top", "y": 0.99, "xanchor": "left", "x": 0.01},
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")

    html_filename = output_dir / f"forecast_viz_{scorer_id[:20]}.html"
    fig.write_html(str(html_filename))

    return str(html_filename)


def create_forecast_comparison_plots(
    xgb_forecast: pd.DataFrame,
    baseline_forecast: pd.DataFrame,
    output_dir: Path,
    scorer_id: str,
) -> str:
    """Create comparison plots between XGBoost and Baseline forecasts."""

    xgb_forecast["Date"] = pd.to_datetime(xgb_forecast["Date"])
    baseline_forecast["Date"] = pd.to_datetime(baseline_forecast["Date"])

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Forecast Comparison",
            "Difference (XGBoost - Baseline)",
            "Distribution Comparison",
            "Daily Average Patterns",
        ),
        specs=[[{"colspan": 2}, None], [{}, {}]],
    )

    fig.add_trace(
        go.Scatter(
            x=xgb_forecast["Date"],
            y=xgb_forecast["Value"],
            mode="lines",
            name="XGBoost",
            line={"color": "red"},
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=baseline_forecast["Date"],
            y=baseline_forecast["Value"],
            mode="lines",
            name="Baseline",
            line={"color": "orange", "dash": "dash"},
        ),
        row=1,
        col=1,
    )

    difference = xgb_forecast["Value"] - baseline_forecast["Value"]
    fig.add_trace(
        go.Scatter(
            x=xgb_forecast["Date"],
            y=difference,
            mode="lines",
            name="Difference",
            line={"color": "purple"},
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Histogram(
            x=xgb_forecast["Value"],
            name="XGBoost Dist",
            opacity=0.7,
            nbinsx=50,
            showlegend=False,
        ),
        row=2,
        col=2,
    )

    fig.add_trace(
        go.Histogram(
            x=baseline_forecast["Value"],
            name="Baseline Dist",
            opacity=0.7,
            nbinsx=50,
            showlegend=False,
        ),
        row=2,
        col=2,
    )

    fig.update_layout(
        title=f"Forecast Analysis Dashboard - Model: {scorer_id[:8]}",
        height=800,
        template="plotly_white",
    )

    html_filename = output_dir / f"forecast_comparison_{scorer_id[:20]}.html"
    fig.write_html(str(html_filename))

    return str(html_filename)


def create_forecast_statistics_summary(
    xgb_forecast: pd.DataFrame,
    baseline_forecast: pd.DataFrame,
    output_dir: Path,
    scorer_id: str,
) -> str:
    """Create detailed statistics summary of forecasts."""

    xgb_stats = {
        "mean": xgb_forecast["Value"].mean(),
        "std": xgb_forecast["Value"].std(),
        "min": xgb_forecast["Value"].min(),
        "max": xgb_forecast["Value"].max(),
        "median": xgb_forecast["Value"].median(),
    }

    baseline_stats = {
        "mean": baseline_forecast["Value"].mean(),
        "std": baseline_forecast["Value"].std(),
        "min": baseline_forecast["Value"].min(),
        "max": baseline_forecast["Value"].max(),
        "median": baseline_forecast["Value"].median(),
    }

    comparison = {
        "mean_difference": xgb_stats["mean"] - baseline_stats["mean"],
        "std_difference": xgb_stats["std"] - baseline_stats["std"],
        "correlation": xgb_forecast["Value"].corr(baseline_forecast["Value"]),
    }

    stats_summary = {
        "model_id": scorer_id,
        "forecast_period": {
            "start": str(xgb_forecast["Date"].min()),
            "end": str(xgb_forecast["Date"].max()),
            "total_hours": len(xgb_forecast),
        },
        "xgboost_statistics": xgb_stats,
        "baseline_statistics": baseline_stats,
        "comparison": comparison,
    }

    stats_filename = output_dir / f"forecast_statistics_{scorer_id[:8]}.json"

    with open(stats_filename, "w", encoding="utf-8") as f:
        json.dump(stats_summary, f, indent=2, default=str)

    return str(stats_filename)


def main() -> int:  # pylint: disable=R0914,R0915
    """Main function for future forecast generation."""
    parser = argparse.ArgumentParser(description="Generate future forecasts")
    parser.add_argument(
        "--model_path",
        default="model/XGBoostWrapper_31796495-c817-44a0-9d2d-1bd79465c397.pkl",
        help="Path to trained model (.pkl)",
    )
    parser.add_argument(
        "--data_path",
        default="input_data/file2.csv",
        help="Path to historical data (.csv)",
    )
    parser.add_argument("--preprocessor_path", help="Path to preprocessor (.pkl)")
    parser.add_argument(
        "--forecast_years", type=float, default=1.0, help="Years to forecast"
    )
    parser.add_argument(
        "--folder_name",
        default="default",
        help="Name of the folder to save forecasts and output (e.g., 'experiment_1')",
    )
    parser.add_argument(
        "--visualize", action="store_true", default=True, help="Generate visualizations"
    )
    parser.add_argument(
        "--comparison_plots", action="store_true", help="Generate comparison plots"
    )
    if len(sys.argv) == 1:
        sys.argv.extend(
            [
                "--model_path",
                "model/XGBoostWrapper_904b211c-5f3d-4a91-ad6e-35ee44eafacf.pkl",
                "--data_path",
                "input_data/file1.csv",
                "--forecast_years",
                "1.0",
                "--folder_name",
                "experiment_5",
                "--visualize",
                "--comparison_plots",
            ]
        )

    args = parser.parse_args()

    model_path = Path(args.model_path)
    data_path = Path(args.data_path)

    forecasts_dir = Path("forecasts") / args.folder_name
    output_dir = Path("output") / args.folder_name

    debug_results_dir = Path("debug_results")
    debug_results_dir.mkdir(exist_ok=True, parents=True)

    forecasts_dir.mkdir(exist_ok=True, parents=True)
    output_dir.mkdir(exist_ok=True, parents=True)

    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        return 1

    if not data_path.exists():
        print(f"Error: Data file not found: {data_path}")
        return 1

    forecast_steps = int(args.forecast_years * 8760)

    try:
        historical_data = load_csv_data(str(data_path))

        engine = ForecastEngine(results_dir=str(forecasts_dir))

        scorer_id = engine.load_scorer(
            model_path=str(model_path),
            preprocessor_path=args.preprocessor_path,
        )

        result = engine.generate_future_forecast(
            scorer_id=scorer_id,
            historical_data=historical_data,
            forecast_steps=forecast_steps,
            save_results=True,
        )
        print(f"\nFiles in {forecasts_dir}:")
        for file in forecasts_dir.glob("*"):
            print(f"  {file.name}")

        if args.visualize:
            print("\nLooking for forecast files...")
            all_forecast_files = list(forecasts_dir.glob("future_forecast_*.csv"))
            print(f"All forecast files found: {len(all_forecast_files)}")
            for file in all_forecast_files:
                print(f"  {file.name}")

            xgb_files = [f for f in all_forecast_files if "baseline" not in f.name]
            baseline_files = [f for f in all_forecast_files if "baseline" in f.name]

            if xgb_files and baseline_files:
                xgb_file = max(xgb_files, key=lambda f: f.stat().st_mtime)
                baseline_file = max(baseline_files, key=lambda f: f.stat().st_mtime)

                print("Creating visualizations...")
                print(f"  XGBoost file: {xgb_file.name}")
                print(f"  Baseline file: {baseline_file.name}")

                xgb_forecast = pd.read_csv(xgb_file)
                baseline_forecast = pd.read_csv(baseline_file)
                create_forecast_plotly_express(
                    xgb_forecast,
                    baseline_forecast,
                    historical_data,
                    debug_results_dir,
                    scorer_id,
                )
                create_future_forecast_visualization(
                    historical_data,
                    xgb_forecast,
                    baseline_forecast,
                    debug_results_dir,
                    scorer_id,
                )

                if args.comparison_plots:
                    create_forecast_comparison_plots(
                        xgb_forecast, baseline_forecast, debug_results_dir, scorer_id
                    )

                    create_forecast_statistics_summary(
                        xgb_forecast, baseline_forecast, debug_results_dir, scorer_id
                    )
            else:
                print("Could not find both XGBoost and baseline forecast files")
        else:
            print("No forecast files found to visualize.")

        print("Forecast generation complete!")
        print(f"Generated {len(result['predictions'])} predictions")
        print(f"Total time: {result['total_time_seconds']:.1f} seconds")
        print(f"Speed: {result['predictions_per_second']:.1f} predictions/second")

        if "cache_hits" in result:
            total_cache_requests = result["cache_hits"] + result["cache_misses"]
            cache_hit_rate = (
                result["cache_hits"] / total_cache_requests
                if total_cache_requests > 0
                else 0
            )
            print(f"Cache hit rate: {cache_hit_rate:.1%}")

        print(f"Forecasts saved to: {forecasts_dir}/")
        print(f"Visualizations saved to: {debug_results_dir}/")

        return 0

    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"Error generating forecast: {e}")

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

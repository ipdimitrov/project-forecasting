"""Utility to combine forecast/baseline files with original data files.

# Basic usage
python backend/debug/debug_combine.py --folder_name experiment_1

# With baseline files
python backend/debug/debug_combine.py --folder_name experiment_1 --use_baseline

# Custom input file and output name
python backend/debug/debug_combine.py \
    --folder_name experiment_1 \
    --input_file input_data/my_data.csv \
    --output_filename my_combined_output.csv
"""

import argparse
import re
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class FileCombiner:
    """Combines forecast/baseline files with original data files."""

    def __init__(self, folder_name: str = "default") -> None:
        self.folder_name = folder_name
        self.forecasts_dir = Path("forecasts") / folder_name
        self.input_data_dir = Path("input_data")
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def extract_numeral_from_filename(self, filename: str) -> str:
        """Extract numeral(s) from filename pattern like 'file2.csv' -> '2'."""
        basename = Path(filename).name
        match = re.search(r"file(\d+)\.csv", basename, re.IGNORECASE)
        if match:
            return match.group(1)
        else:
            digits = re.findall(r"\d+", basename)
            return digits[0] if digits else "1"

    def get_available_forecast_files(self) -> List[Path]:
        """Get all forecast CSV files from the forecasts directory."""
        forecast_files = []

        if self.forecasts_dir.exists():
            forecast_files.extend(self.forecasts_dir.glob("future_forecast_*.csv"))

        return forecast_files

    def convert_to_15min_intervals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert hourly data to 15-minute intervals with 1/4 the value."""
        df["Date"] = pd.to_datetime(df["Date"])

        new_data = []
        for _, row in df.iterrows():
            base_time = row["Date"]
            quarter_value = row["Value"] / 4

            for minutes in [0, 15, 30, 45]:
                new_time = base_time + pd.Timedelta(minutes=minutes)
                new_data.append({"Date": new_time, "Value": quarter_value})

        return pd.DataFrame(new_data)

    def combine_with_forecast_files(
        self,
        input_file: str,
        use_baseline: bool = False,
        output_filename: Optional[str] = None,
    ) -> Optional[str]:
        """Combine input file with forecast files from the folder."""
        input_path = Path(input_file)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        forecast_files = self.get_available_forecast_files()
        if not forecast_files:
            raise FileNotFoundError(f"No forecast files found in {self.forecasts_dir}")
        if use_baseline:
            target_files = [f for f in forecast_files if "baseline" in f.name]
            file_type = "baseline"
        else:
            target_files = [f for f in forecast_files if "baseline" not in f.name]
            file_type = "forecast"

        if not target_files:
            raise FileNotFoundError(
                f"No {file_type} files found in {self.forecasts_dir}"
            )
        forecast_file = max(target_files, key=lambda f: f.stat().st_mtime)
        print(f"Using {file_type} file: {forecast_file.name}")
        if output_filename is None:
            numeral = self.extract_numeral_from_filename(input_file)
            output_filename = f"SITE_{numeral}.csv"

        output_path = self.output_dir / output_filename
        original_df = pd.read_csv(input_path)
        forecast_df = pd.read_csv(forecast_file)
        original_df["Date"] = pd.to_datetime(original_df["Date"])
        forecast_df["Date"] = pd.to_datetime(forecast_df["Date"])

        combined_df = pd.concat([original_df, forecast_df], ignore_index=True)
        combined_df = combined_df.sort_values("Date").reset_index(drop=True)
        combined_df = self.convert_to_15min_intervals(combined_df)

        combined_df.to_csv(output_path, index=False)
        print(f"Combined data saved to: {output_path}")
        return str(output_path)


def main() -> None:
    """Main function with command line argument support and debug mode."""
    parser = argparse.ArgumentParser(
        description="Combine forecast/baseline files with original data files"
    )
    parser.add_argument(
        "--folder_name",
        default="default",
        help="Name of the folder in forecasts/ to read from (e.g., 'experiment_1')",
    )
    parser.add_argument(
        "--input_file",
        default="input_data/file2.csv",
        help="Path to input data file to combine with forecasts",
    )
    parser.add_argument(
        "--use_baseline",
        action="store_true",
        help="Use baseline files instead of forecast files",
    )
    parser.add_argument("--output_filename", help="Custom output filename (optional)")

    # Check if running without arguments (debugger/IDE mode)
    if len(sys.argv) == 1:
        # Running from debugger - use default arguments
        print("Running in debug mode with default settings...")
        sys.argv.extend(
            [
                "--folder_name",
                "experiment_4",  # Change this to test different folders
                "--input_file",
                "input_data/file2.csv",  # Change this to test different input files
                # "--use_baseline",  # Uncomment this line to test baseline mode.
                # When commented we use the forecast files.
            ]
        )

    args = parser.parse_args()

    combiner = FileCombiner(folder_name=args.folder_name)

    print(f"Looking for forecast files in: {combiner.forecasts_dir}")
    print(f"Output will be saved to: {combiner.output_dir}")
    print(f"Input file: {args.input_file}")
    print(f"Using: {'baseline' if args.use_baseline else 'forecast'} files")

    try:
        output_path = combiner.combine_with_forecast_files(
            input_file=args.input_file,
            use_baseline=args.use_baseline,
            output_filename=args.output_filename,
        )

        file_type = "baseline" if args.use_baseline else "forecast"
        print(f"Successfully combined input file with {file_type}")
        print(f"Output saved to: {output_path}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nAvailable forecast files:")
        for file in combiner.get_available_forecast_files():
            print(f"   - {file.name}")


if __name__ == "__main__":
    main()

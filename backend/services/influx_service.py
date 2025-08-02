"""
InfluxDB service for forecasting data.
Handles reading from CSVs and writing/querying InfluxDB for both historical and forecast data.
"""

import os
from datetime import datetime
from typing import List, Optional
import pandas as pd
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
from backend.utils.config import DataSourceConfig

from backend.utils.config import get_influxdb_config, get_data_source_config
from backend.utils.logging_config import get_logger

logger = get_logger(__name__)


class InfluxService:
    """Service for interacting with InfluxDB"""

    def __init__(
        self,
        url: Optional[str] = None,
        token: Optional[str] = None,
        org: Optional[str] = None,
        bucket: Optional[str] = None,
    ):
        """Initialize InfluxService with optional overrides, defaults from config."""
        influx_config = get_influxdb_config()

        self.url = url or influx_config.url
        self.token = token or influx_config.token
        self.org = org or influx_config.org
        self.bucket = bucket or influx_config.bucket
        self.client = None

    def connect(self) -> bool:
        """Connect to InfluxDB"""
        try:
            self.client = InfluxDBClient(url=self.url, token=self.token, org=self.org)
            self.client.health()
            logger.debug("Successfully connected to InfluxDB at %s", self.url)
            return True
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("Failed to connect to InfluxDB at %s: %s", self.url, e)
            return False

    def disconnect(self):
        """Close InfluxDB connection"""
        if self.client:
            logger.debug("Closing InfluxDB connection")
            self.client.close()

    def write_data_point(
        self,
        asset_id: str,
        timestamp: datetime,
        value: float,
        data_type: str = "forecast",
    ) -> bool:
        """Write a single data point to InfluxDB"""
        if not self.client:
            return False

        try:
            write_api = self.client.write_api(write_options=SYNCHRONOUS)

            point = (
                Point("value")
                .tag("asset_id", asset_id)
                .tag("data_type", data_type)
                .field("value", value)
                .time(timestamp, WritePrecision.S)
            )

            write_api.write(bucket=self.bucket, org=self.org, record=point)
            logger.debug(
                "Successfully wrote data point for asset %s at %s", asset_id, timestamp
            )
            return True
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("Failed to write data point for asset %s: %s", asset_id, e)
            return False

    def write_data_batch(self, data: List[dict], data_type: str = "forecast") -> bool:
        """Write multiple data points to InfluxDB"""
        if not self.client:
            return False

        try:
            write_api = self.client.write_api(write_options=SYNCHRONOUS)
            points = []

            for record in data:
                point = (
                    Point("value")
                    .tag("asset_id", record["asset_id"])
                    .tag("data_type", data_type)
                    .field("Value", record["value"])
                    .time(record["timestamp"], WritePrecision.S)
                )
                points.append(point)

            write_api.write(bucket=self.bucket, org=self.org, record=points)
            logger.info("Successfully wrote batch of %d data points", len(points))
            return True
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error(
                "Failed to write batch data (%d points): %s",
                len(data) if data else 0,
                e,
            )
            return False

    def query_forecasts(
        self, start: datetime, end: datetime, asset_id: Optional[str] = None
    ) -> List[dict]:
        """Query forecast data from InfluxDB"""
        if not self.client:
            return []

        try:
            query_api = self.client.query_api()

            asset_filter = (
                f'|> filter(fn: (r) => r["asset_id"] == "{asset_id}")'
                if asset_id
                else ""
            )

            query = f"""
            from(bucket: "{self.bucket}")
                |> range(start: {start.isoformat()}Z, stop: {end.isoformat()}Z)
                |> filter(fn: (r) => r["_measurement"] == "value")
                {asset_filter}
                |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
            """

            result = query_api.query(query, org=self.org)

            forecasts = []
            for table in result:
                for record in table.records:
                    forecast = {
                        "timestamp": record.get_time(),
                        "asset_id": record.values.get("asset_id"),
                        "Value": record.values.get(
                            "Value", record.values.get("calue", 0)
                        ),
                        "data_type": record.values.get("data_type", "forecast"),
                    }
                    forecasts.append(forecast)

            logger.info(
                "Successfully queried %d forecast records from InfluxDB", len(forecasts)
            )
            return forecasts

        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("Failed to query forecast data: %s", e)
            return []

    def delete_all_data(self) -> bool:
        """Delete all data from the bucket (for testing/reset purposes)"""
        if not self.client:
            return False

        try:
            delete_api = self.client.delete_api()
            start = "1970-01-01T00:00:00Z"
            stop = datetime.now().isoformat() + "Z"

            delete_api.delete(
                start, stop, '_measurement="value"', bucket=self.bucket, org=self.org
            )
            logger.info("Successfully deleted all data from InfluxDB bucket")
            return True
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("Failed to delete data from InfluxDB: %s", e)
            return False


def setup_influxdb_data() -> bool:
    """Setup InfluxDB with initial data from CSV files"""
    logger.info("Starting InfluxDB data setup from CSV files")

    try:
        service = InfluxService()

        if not service.connect():
            logger.error("Could not connect to InfluxDB for data setup")
            return False

        data_config = get_data_source_config()
        success = True
        files_processed = 0

        data_sources = [
            (data_config.historical_files, "historical"),
            (data_config.forecast_files, "forecast"),
        ]

        for file_list, data_type in data_sources:
            logger.info("Processing %d %s data files", len(file_list), data_type)

            for file_path in file_list:
                if os.path.exists(file_path):
                    if process_csv_file(service, file_path, data_type):
                        files_processed += 1
                    else:
                        success = False
                else:
                    logger.warning(
                        "%s data file not found: %s", data_type.title(), file_path
                    )

        service.disconnect()

        if success:
            logger.info(
                "InfluxDB data setup completed successfully. Processed %d files.",
                files_processed,
            )
        else:
            logger.warning(
                "InfluxDB data setup completed with errors. Processed %d files.",
                files_processed,
            )

        return success

    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Error setting up InfluxDB data: %s", e, exc_info=True)
        return False


def process_csv_file(service: InfluxService, file_path: str, data_type: str) -> bool:
    """Process a single CSV file and upload to InfluxDB"""
    logger.debug("Processing CSV file: %s as %s data", file_path, data_type)

    try:
        df = pd.read_csv(file_path)
        logger.debug("Loaded CSV file with %d rows from %s", len(df), file_path)

        default_asset_id = DataSourceConfig._extract_site_id_from_filename(file_path)

        asset_id_col = None
        for col in df.columns:
            if col.lower() in ["asset_id", "assetid", "site_id", "siteid", "site"]:
                asset_id_col = col
                break

        data_points = []
        for _, row in df.iterrows():
            if asset_id_col:
                asset_id = str(row.get(asset_id_col, default_asset_id))
            else:
                asset_id = default_asset_id

            data_point = {
                "timestamp": (
                    pd.to_datetime(
                        row.get("timestamp", row.get("Date", row.get("date")))
                    )
                    if any(col in row for col in ["timestamp", "Date", "date"])
                    else datetime.now()
                ),
                "asset_id": asset_id,
                "value": float(row.get("Value", row.get("value", 0))),
            }
            data_points.append(data_point)

        success = service.write_data_batch(data_points, data_type)

        if success:
            unique_assets = list(set(dp["asset_id"] for dp in data_points))
            logger.info(
                "Successfully processed %d data points from %s for assets: %s",
                len(data_points),
                file_path,
                ", ".join(unique_assets),
            )
        else:
            logger.error("Failed to write data from %s to InfluxDB", file_path)

        return success

    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Error processing file %s: %s", file_path, e, exc_info=True)
        return False

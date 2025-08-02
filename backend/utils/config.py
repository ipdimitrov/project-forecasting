"""Centralized configuration management using environment variables."""

import os
import glob
from dataclasses import dataclass
from typing import List, Set, Optional
from pathlib import Path
import pandas as pd

# Try to load .env file if it exists
try:
    from dotenv import load_dotenv

    # Look for .env file in the project root
    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    # python-dotenv not installed, skip loading .env file
    pass


@dataclass
class InfluxDBConfig:
    """InfluxDB configuration from environment variables."""

    url: str
    token: str
    org: str
    bucket: str
    username: str  # For Docker initialization
    password: str  # For Docker initialization

    @classmethod
    def from_env(cls) -> "InfluxDBConfig":
        """Create InfluxDBConfig from environment variables."""
        return cls(
            url=os.getenv("INFLUXDB_URL", "http://influxdb:8086"),
            token=os.getenv("INFLUXDB_TOKEN", "admin-token"),
            org=os.getenv("INFLUXDB_ORG", "org"),
            bucket=os.getenv("INFLUXDB_BUCKET", "forecasts"),
            username=os.getenv("INFLUXDB_USERNAME", "admin"),
            password=os.getenv("INFLUXDB_PASSWORD", "password123"),
        )


@dataclass
class APIConfig:
    """API server configuration from environment variables."""

    host: str
    port: int
    environment: str

    @classmethod
    def from_env(cls) -> "APIConfig":
        """Create APIConfig from environment variables."""
        return cls(
            host=os.getenv("API_HOST", "0.0.0.0"),
            port=int(os.getenv("API_PORT", "8000")),
            environment=os.getenv("ENVIRONMENT", "development"),
        )


@dataclass
class DataSourceConfig:
    """Data source configuration with flexible auto-discovery of CSV files."""

    historical_files: List[str]
    forecast_files: List[str]

    @classmethod
    def from_env(cls) -> "DataSourceConfig":
        """Create DataSourceConfig with auto-discovery of CSV files."""
        # Auto-discover CSV files
        data_dir = os.getenv("DATA_DIR", "output")
        file_pattern = os.getenv("FILE_PATTERN", "*.csv")
        discovered_files = cls._discover_csv_files(data_dir, file_pattern)
        return cls(
            historical_files=discovered_files,
            forecast_files=discovered_files,
        )

    @staticmethod
    def _discover_csv_files(data_dir: str, pattern: str = "*.csv") -> List[str]:
        """Discover all CSV files matching the pattern in the specified directory."""
        search_pattern = os.path.join(data_dir, pattern)
        files = glob.glob(search_pattern)
        files.sort()  # Ensure consistent ordering
        return files

    def get_available_sites(self) -> List[str]:
        """Extract all available site IDs from discovered files and their content."""
        all_sites: Set[str] = set()

        for file_path in self.historical_files:
            # First, try to get sites from CSV content
            csv_sites = self._extract_sites_from_csv_content(file_path)
            if csv_sites:
                all_sites.update(csv_sites)
            else:
                # Fallback to filename-based extraction
                site_id = self._extract_site_id_from_filename(file_path)
                if site_id:
                    all_sites.add(site_id)

        return sorted(list(all_sites))

    @staticmethod
    def _extract_sites_from_csv_content(file_path: str) -> Optional[List[str]]:
        """Extract unique asset IDs from CSV content if asset_id column exists."""
        try:
            if not os.path.exists(file_path):
                return None

            # Read just the first few rows to check for asset_id column
            df_sample = pd.read_csv(file_path, nrows=100)

            # Check for asset_id column (case insensitive)
            asset_id_col = None
            for col in df_sample.columns:
                if col.lower() in ["asset_id", "assetid", "site_id", "siteid", "site"]:
                    asset_id_col = col
                    break

            if asset_id_col:
                # Read the full file to get all unique asset IDs
                df_full = pd.read_csv(file_path)
                unique_sites = df_full[asset_id_col].dropna().unique().tolist()
                return [str(site) for site in unique_sites]

        except Exception:  # pylint: disable=broad-exception-caught
            # If CSV reading fails, return None to fallback to filename extraction
            pass

        return None

    @staticmethod
    def _extract_site_id_from_filename(file_path: str) -> str:
        """Extract site ID from filename by using the base name without extension."""
        filename = os.path.basename(file_path)
        # Remove file extension
        site_id = os.path.splitext(filename)[0]

        # Clean up the site ID (remove special characters, ensure it's a valid identifier)
        site_id = site_id.replace("-", "_").replace(" ", "_")

        # Ensure it's not empty
        if not site_id:
            return "unknown_site"

        return site_id

    def get_file_to_site_mapping(self) -> dict:
        """Get a mapping of file paths to their corresponding site IDs."""
        mapping = {}

        for file_path in self.historical_files:
            # Try CSV content first
            csv_sites = self._extract_sites_from_csv_content(file_path)
            if csv_sites:
                # If CSV has multiple sites, map file to the first one found
                # (or you could return all sites for this file)
                mapping[file_path] = csv_sites[0]
            else:
                # Fallback to filename
                mapping[file_path] = self._extract_site_id_from_filename(file_path)

        return mapping


@dataclass
class LoggingConfig:
    """Logging configuration from environment variables."""

    log_level: str
    log_format: str
    log_file_enabled: bool
    log_file_path: str
    log_file_max_size: int
    log_file_backup_count: int

    @classmethod
    def from_env(cls) -> "LoggingConfig":
        """Create LoggingConfig from environment variables."""
        return cls(
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
            log_format=os.getenv(
                "LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            ),
            log_file_enabled=os.getenv("LOG_FILE_ENABLED", "true").lower() == "true",
            log_file_path=os.getenv("LOG_FILE_PATH", "logs/forecasting.log"),
            log_file_max_size=int(os.getenv("LOG_FILE_MAX_SIZE", "10485760")),  # 10MB
            log_file_backup_count=int(os.getenv("LOG_FILE_BACKUP_COUNT", "5")),
        )


@dataclass
class AppConfig:
    """Complete application configuration."""

    influxdb: InfluxDBConfig
    api: APIConfig
    data_sources: DataSourceConfig
    logging: LoggingConfig

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Create complete app configuration from environment variables."""
        return cls(
            influxdb=InfluxDBConfig.from_env(),
            api=APIConfig.from_env(),
            data_sources=DataSourceConfig.from_env(),
            logging=LoggingConfig.from_env(),
        )

    def to_legacy_dict(self) -> dict:
        """Convert to legacy dictionary format for backward compatibility."""
        return {
            "influxdb": {
                "url": self.influxdb.url,
                "token": self.influxdb.token,
                "org": self.influxdb.org,
                "bucket": self.influxdb.bucket,
            },
            "data": {
                "historical": {"source_files": self.data_sources.historical_files},
                "forecast": {"source_files": self.data_sources.forecast_files},
            },
        }


# Global configuration instance
config = AppConfig.from_env()


def get_config() -> AppConfig:
    """Get the global configuration instance."""
    return config


def reload_config() -> AppConfig:
    """Reload configuration from environment (useful for testing)."""
    global config  # pylint: disable=global-statement
    config = AppConfig.from_env()
    return config


# Convenience functions for specific configs
def get_influxdb_config() -> InfluxDBConfig:
    """Get InfluxDB configuration."""
    return config.influxdb


def get_api_config() -> APIConfig:
    """Get API configuration."""
    return config.api


def get_data_source_config() -> DataSourceConfig:
    """Get data source configuration."""
    return config.data_sources


def get_logging_config_from_env() -> LoggingConfig:
    """Get logging configuration."""
    return config.logging

"""Test configuration and fixtures."""

import os
import warnings
from unittest.mock import Mock, patch
import pytest
from fastapi.testclient import TestClient
from backend.utils.config import reload_config  # noqa: E402
from backend.main import app
from backend.services.influx_service import InfluxService

os.environ.setdefault("INFLUXDB_URL", "http://localhost:8086")
os.environ.setdefault("INFLUXDB_TOKEN", "test-token")
os.environ.setdefault("INFLUXDB_ORG", "test-org")
os.environ.setdefault("INFLUXDB_BUCKET", "test-bucket")

reload_config()

warnings.filterwarnings("ignore", category=DeprecationWarning, module="reactivex.*")
warnings.filterwarnings(
    "ignore", category=DeprecationWarning, module="influxdb_client.*"
)


@pytest.fixture
def influx_service_mock() -> Mock:
    """Mock InfluxService for testing."""
    mock_service = Mock(spec=InfluxService)
    mock_service.connect.return_value = True
    mock_service.disconnect.return_value = None
    mock_service.query_forecasts.return_value = [
        {
            "timestamp": "2024-01-01T00:00:00",
            "asset_id": "SITE_1",
            "Value": 100.0,
            "data_type": "forecast",
        },
        {
            "timestamp": "2024-01-01T01:00:00",
            "asset_id": "SITE_1",
            "Value": 110.0,
            "data_type": "forecast",
        },
    ]
    return mock_service


@pytest.fixture
def data_source_config_mock() -> Mock:
    """Mock DataSourceConfig to provide test sites."""
    mock_config = Mock()
    mock_config.get_available_sites.return_value = ["SITE_1", "SITE_2", "SITE_3"]
    mock_config.get_file_to_site_mapping.return_value = {
        "file1.csv": "SITE_1",
        "file2.csv": "SITE_2",
        "file3.csv": "SITE_3",
    }
    mock_config.historical_files = ["file1.csv", "file2.csv", "file3.csv"]
    return mock_config


@pytest.fixture
def client(
    influx_service_mock: Mock, data_source_config_mock: Mock
) -> TestClient:  # pylint: disable=redefined-outer-name # noqa: E501
    """Test client with mocked dependencies."""
    with patch("backend.main.InfluxService") as mock_class:
        mock_class.return_value = influx_service_mock
        with patch("backend.main.get_data_source_config") as mock_data_config:
            mock_data_config.return_value = data_source_config_mock
            with TestClient(app) as test_client:
                test_client.app.state.influx_service = influx_service_mock
                yield test_client


@pytest.fixture
def mock_setup_influxdb_data(monkeypatch) -> Mock:
    """Mock the setup_influxdb_data function."""
    mock_func = Mock(return_value=True)
    monkeypatch.setattr("backend.main.setup_influxdb_data", mock_func)
    return mock_func

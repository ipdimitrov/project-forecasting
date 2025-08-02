"""Tests for API endpoints."""

from fastapi import status


class TestRootEndpoint:  # pylint: disable=R0903
    """Tests for the root endpoint."""

    def test_root_endpoint_returns_api_info(self, client):
        """Test that root endpoint returns correct API information."""
        response = client.get("/")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert data["message"] == "Forecasting API"
        assert data["version"] == "1.0.0"
        assert "endpoints" in data
        assert data["endpoints"]["forecasts"] == "/forecasts"
        assert data["endpoints"]["health"] == "/health"
        assert data["endpoints"]["admin"] == "/admin/populate-data"


class TestHealthEndpoint:  # pylint: disable=R0903
    """Tests for the health check endpoint."""

    def test_health_endpoint_healthy_service(self, client, influx_service_mock):
        """Test health endpoint when InfluxDB is healthy."""
        influx_service_mock.connect.return_value = True

        response = client.get("/health")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert data["status"] == "healthy"
        assert data["version"] == "1.0.0"
        assert data["services"]["api"] == "operational"
        assert data["services"]["influxdb"] == "operational"
        assert "timestamp" in data

    def test_health_endpoint_degraded_service(self, client, influx_service_mock):
        """Test health endpoint when InfluxDB is unavailable."""
        # Reset the mock for this specific test
        influx_service_mock.reset_mock()
        influx_service_mock.connect.return_value = False

        response = client.get("/health")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert data["status"] == "degraded"
        assert data["services"]["api"] == "operational"
        assert data["services"]["influxdb"] == "unavailable"

    def test_health_endpoint_exception_handling(self, client, influx_service_mock):
        """Test health endpoint handles exceptions gracefully."""
        # Reset the mock and make it raise an exception
        influx_service_mock.reset_mock()
        influx_service_mock.connect.side_effect = Exception("Connection error")

        response = client.get("/health")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert data["status"] == "unhealthy"
        assert data["error"] == "Health check failed"


class TestForecastsEndpoint:
    """Tests for the forecasts endpoint."""

    def test_get_forecasts_success(self, client, influx_service_mock):
        """Test successful forecast retrieval."""
        start_time = "2024-01-01T00:00:00"
        end_time = "2024-01-01T23:59:59"

        response = client.get(f"/forecasts?start={start_time}&end={end_time}")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert data["status"] == "success"
        assert data["count"] == 2
        assert len(data["forecasts"]) == 2
        assert data["forecasts"][0]["asset_id"] == "SITE_1"
        assert data["forecasts"][0]["value"] == 100.0
        assert "metadata" in data

        # Verify the service was called correctly
        influx_service_mock.query_forecasts.assert_called_once()

    def test_get_forecasts_with_asset_filter(self, client, influx_service_mock):
        """Test forecast retrieval with asset filter."""
        start_time = "2024-01-01T00:00:00"
        end_time = "2024-01-01T23:59:59"
        asset_id = "SITE_1"

        response = client.get(
            f"/forecasts?start={start_time}&end={end_time}&asset_id={asset_id}"
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert data["status"] == "success"
        # Verify asset filter was passed to service
        args, _ = influx_service_mock.query_forecasts.call_args
        assert len(args) == 3
        assert args[2] == "SITE_1"  # asset_filter parameter

    def test_get_forecasts_no_data(self, client, influx_service_mock):
        """Test forecast endpoint when no data is available."""
        # Reset mock and set empty return
        influx_service_mock.reset_mock()
        influx_service_mock.connect.return_value = True
        influx_service_mock.query_forecasts.return_value = []

        start_time = "2024-01-01T00:00:00"
        end_time = "2024-01-01T23:59:59"

        response = client.get(f"/forecasts?start={start_time}&end={end_time}")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert data["status"] == "no_data"
        assert data["count"] == 0
        assert len(data["forecasts"]) == 0
        assert "No forecast data found" in data["metadata"]["message"]

    def test_get_forecasts_influxdb_unavailable(self, client, influx_service_mock):
        """Test forecast endpoint when InfluxDB is unavailable."""
        # Reset mock and make connect return False
        influx_service_mock.reset_mock()
        influx_service_mock.connect.return_value = False

        start_time = "2024-01-01T00:00:00"
        end_time = "2024-01-01T23:59:59"

        response = client.get(f"/forecasts?start={start_time}&end={end_time}")

        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        data = response.json()
        assert "InfluxDB service unavailable" in data["detail"]

    def test_get_forecasts_missing_parameters(self, client):
        """Test forecast endpoint with missing required parameters."""
        # Missing both start and end
        response = client.get("/forecasts")
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

        # Missing end parameter
        response = client.get("/forecasts?start=2024-01-01T00:00:00")
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_get_forecasts_invalid_asset_id(self, client):
        """Test forecast endpoint with invalid asset ID."""
        start_time = "2024-01-01T00:00:00"
        end_time = "2024-01-01T23:59:59"

        response = client.get(
            f"/forecasts?start={start_time}&end={end_time}&asset_id=INVALID_SITE"
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_get_forecasts_service_exception(self, client, influx_service_mock):
        """Test forecast endpoint handles service exceptions."""
        influx_service_mock.reset_mock()
        influx_service_mock.connect.return_value = True
        influx_service_mock.query_forecasts.side_effect = Exception("Database error")

        start_time = "2024-01-01T00:00:00"
        end_time = "2024-01-01T23:59:59"

        response = client.get(f"/forecasts?start={start_time}&end={end_time}")

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        data = response.json()
        assert "Error retrieving forecasts" in data["detail"]


class TestAdminEndpoint:
    """Tests for the admin populate data endpoint."""

    def test_populate_data_success(self, client, mock_setup_influxdb_data):
        """Test successful data population."""
        mock_setup_influxdb_data.return_value = True

        response = client.post("/admin/populate-data")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert data["status"] == "success"
        assert "InfluxDB populated successfully" in data["message"]
        assert "data_populated" in data

        # Verify the function was called
        mock_setup_influxdb_data.assert_called_once()

    def test_populate_data_partial_success(self, client, mock_setup_influxdb_data):
        """Test partial success in data population."""
        mock_setup_influxdb_data.return_value = False

        response = client.post("/admin/populate-data")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert data["status"] == "partial_success"
        assert "Some errors occurred" in data["message"]

    def test_populate_data_exception(self, client, mock_setup_influxdb_data):
        """Test admin endpoint handles exceptions."""
        mock_setup_influxdb_data.side_effect = Exception("Setup error")

        response = client.post("/admin/populate-data")

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        data = response.json()
        assert "Failed to populate data" in data["detail"]


class TestResponseModels:  # pylint: disable=R0903
    """Tests for response data structure validation."""

    def test_forecast_response_structure(self, client):
        """Test that forecast response matches expected structure."""
        start_time = "2024-01-01T00:00:00"
        end_time = "2024-01-01T23:59:59"

        response = client.get(f"/forecasts?start={start_time}&end={end_time}")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        required_fields = ["status", "forecasts", "count", "metadata"]
        for field in required_fields:
            assert field in data

        if data["forecasts"]:
            forecast_item = data["forecasts"][0]
            required_forecast_fields = ["timestamp", "asset_id", "value", "data_type"]
            for field in required_forecast_fields:
                assert field in forecast_item

        metadata = data["metadata"]
        assert "query_period" in metadata
        assert "start" in metadata["query_period"]
        assert "end" in metadata["query_period"]

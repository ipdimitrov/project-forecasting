"""
Main API module for forecasting service.
Provides endpoints for forecasting, data retrieval, and health monitoring.
"""

from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel

from backend.services.influx_service import InfluxService, setup_influxdb_data
from backend.utils.config import get_config, get_api_config, get_data_source_config
from backend.utils.logging_config import setup_logging, get_logger

# Setup logging early
setup_logging()
logger = get_logger(__name__)

# Load configuration once at module level
app_config = get_config()
CONFIG = app_config.to_legacy_dict()  # For backward compatibility


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Handle startup and shutdown events"""
    influx_config = app_config.influxdb
    influx_service = InfluxService(
        url=influx_config.url,
        token=influx_config.token,
        org=influx_config.org,
        bucket=influx_config.bucket,
    )
    success = influx_service.connect()
    if not success:
        logger.warning("Could not connect to InfluxDB. Using fallback data.")
    else:
        logger.info("Successfully connected to InfluxDB")
    application.state.influx_service = influx_service
    yield
    influx_service.disconnect()


app = FastAPI(
    title="Forecasting API",
    description="API for retrieving forecasts and historical data",
    lifespan=lifespan,
)


class Response(BaseModel):
    """Response model for forecast data."""

    status: str
    forecasts: List[Dict[str, Any]]
    count: int
    metadata: Dict[str, Any]


@app.get("/")
async def root():
    """Root endpoint with API information."""
    data_config = get_data_source_config()
    available_sites = data_config.get_available_sites()
    file_mapping = data_config.get_file_to_site_mapping()

    return {
        "message": "Forecasting API",
        "version": "1.0.0",
        "available_sites": available_sites,
        "discovered_files": data_config.historical_files,
        "file_to_site_mapping": file_mapping,
        "endpoints": {
            "forecasts": "/forecasts",
            "health": "/health",
            "admin": "/admin/populate-data",
        },
    }


@app.get("/forecasts", response_model=Response)
async def get_forecasts(
    start: datetime,
    end: datetime,
    asset_id: Optional[str] = None,
):
    """
    Retrieve forecasts for a specified time period.

    Args:
        start: Start datetime for forecast period (ISO format)
        end: End datetime for forecast period (ISO format)
        asset_id: Optional asset ID filter (automatically discovered from your data files)

    Returns:
        Response containing forecasts, metadata, and status
    """
    logger.info(
        "Forecast request received: start=%s, end=%s, asset_id=%s", start, end, asset_id
    )

    try:
        influx_service = app.state.influx_service

        if not influx_service.connect():
            logger.error("InfluxDB service unavailable for forecast request")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="InfluxDB service unavailable",
            )

        # Validate asset_id if provided
        if asset_id:
            data_config = get_data_source_config()
            available_sites = data_config.get_available_sites()
            if asset_id not in available_sites:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Invalid asset_id '{asset_id}'. \
                        Available sites: {', '.join(available_sites)}",
                )

        logger.debug("Querying forecasts with asset_filter: %s", asset_id)
        forecasts = influx_service.query_forecasts(start, end, asset_id)

        if not forecasts:
            logger.warning(
                "No forecast data found for period %s to %s with asset_filter: %s",
                start,
                end,
                asset_id,
            )
            return Response(
                status="no_data",
                forecasts=[],
                count=0,
                metadata={
                    "query_period": {
                        "start": start.isoformat(),
                        "end": end.isoformat(),
                    },
                    "asset_filter": asset_id,
                    "message": "No forecast data found for the specified period",
                },
            )

        forecast_data = []
        for forecast in forecasts:
            forecast_data.append(
                {
                    "timestamp": (
                        forecast["timestamp"].isoformat()
                        if hasattr(forecast["timestamp"], "isoformat")
                        else str(forecast["timestamp"])
                    ),
                    "asset_id": forecast["asset_id"],
                    "value": forecast["Value"],
                    "data_type": forecast.get("data_type", "forecast"),
                }
            )

        unique_assets = list(set(f["asset_id"] for f in forecast_data))
        data_types = list(set(f["data_type"] for f in forecast_data))

        time_range = {
            "start": min(f["timestamp"] for f in forecast_data),
            "end": max(f["timestamp"] for f in forecast_data),
        }

        logger.info(
            "Successfully retrieved %d forecast points for %d assets",
            len(forecast_data),
            len(unique_assets),
        )
        logger.debug("Data types returned: %s, Assets: %s", data_types, unique_assets)

        return Response(
            status="success",
            forecasts=forecast_data,
            count=len(forecast_data),
            metadata={
                "query_period": {"start": start.isoformat(), "end": end.isoformat()},
                "actual_period": time_range,
                "assets": unique_assets,
                "data_types": data_types,
                "asset_filter": asset_id,
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Unexpected error retrieving forecasts: %s", str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving forecasts: {str(e)}",
        ) from e


@app.post("/admin/populate-data")
async def populate_influxdb():
    """Admin endpoint to populate InfluxDB with both historical and forecast data.

    Uses automatically discovered data files.
    """
    logger.info("Admin request to populate InfluxDB data")

    try:
        success = setup_influxdb_data()
        data_config = get_data_source_config()

        if success:
            logger.info("InfluxDB data population completed successfully")
            return {
                "message": "InfluxDB populated successfully",
                "status": "success",
                "data_populated": {
                    "files_processed": data_config.historical_files,
                    "sites_discovered": data_config.get_available_sites(),
                    "file_to_site_mapping": data_config.get_file_to_site_mapping(),
                },
            }

        logger.warning("InfluxDB data population completed with some errors")
        return {
            "message": "Some errors occurred during population",
            "status": "partial_success",
        }

    except Exception as e:
        logger.error("Failed to populate InfluxDB data: %s", str(e), exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to populate data: {str(e)}"
        ) from e


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring service status."""
    logger.debug("Health check request received")

    try:
        influx_service = app.state.influx_service
        influx_healthy = influx_service.connect()

        health_status = "healthy" if influx_healthy else "degraded"
        logger.debug(
            "Health check completed: %s, InfluxDB: %s", health_status, influx_healthy
        )

        return {
            "status": health_status,
            "timestamp": datetime.now().isoformat(),
            "services": {
                "api": "operational",
                "influxdb": "operational" if influx_healthy else "unavailable",
            },
            "version": "1.0.0",
        }
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Health check failed: %s", str(e), exc_info=True)
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": "Health check failed",
        }


if __name__ == "__main__":
    import uvicorn

    api_config = get_api_config()
    logger.info("Starting Forecasting API on %s:%s", api_config.host, api_config.port)
    uvicorn.run(app, host=api_config.host, port=api_config.port)

# Forecasting API

A time-series forecasting system built with FastAPI, XGBoost, and InfluxDB. Features an optimized unified forecasting engine for scalable predictions and comprehensive model evaluation capabilities.

## Features

- **Unified Forecasting Engine**: Ultra-fast iterative forecasting with optimized batch processing
- **XGBoost Integration**: Advanced gradient boosting models with automated feature engineering
- **InfluxDB Storage**: Efficient time-series data management and retrieval
- **RESTful API**: Production-ready FastAPI endpoints with comprehensive validation
- **Cross-Validation**: Expanding window validation with both standard and iterative evaluation
- **Interactive Visualizations**: Plotly-powered forecast analysis and comparison tools
- **Local Development**: Easy local setup without containerization

## Requirements

**Prerequisites:**
- Python 3.12+
- 8GB+ RAM (recommended for model training)

**Optional:**
- InfluxDB 2.7+ (for data persistence, otherwise uses fallback data)

**Supported Platforms:**
- Linux (Ubuntu 20.04+, WSL2)

## Installation

### Clone Repository
```bash
git clone https://github.com/ipdimitrov/project-forecasting.git
cd project-forecasting
```

### Local Setup
```bash
# The run script will automatically create a virtual environment and install dependencies
./scripts/run.sh
```

### InfluxDB Setup (Recommended for Full Functionality)

For complete API functionality including data persistence and querying, set up InfluxDB using Docker:

#### Quick Docker Setup
```bash
# Install Docker (Ubuntu/WSL2)
sudo apt update && sudo apt install docker.io
sudo systemctl start docker
sudo usermod -aG docker $USER

# Log out and back in, or run:
newgrp docker

# Run InfluxDB container
docker run -d \
  --name influxdb \
  -p 8086:8086 \
  -e DOCKER_INFLUXDB_INIT_MODE=setup \
  -e DOCKER_INFLUXDB_INIT_USERNAME=admin \
  -e DOCKER_INFLUXDB_INIT_PASSWORD=password123 \
  -e DOCKER_INFLUXDB_INIT_ORG=org \
  -e DOCKER_INFLUXDB_INIT_BUCKET=forecasts \
  -e DOCKER_INFLUXDB_INIT_ADMIN_TOKEN=admin-token \
  influxdb:2.0
```

#### Configure Environment
Create a `.env` file in the project root:
```bash
cat > .env << EOF
INFLUXDB_URL=http://localhost:8086
INFLUXDB_TOKEN=admin-token
INFLUXDB_ORG=org
INFLUXDB_BUCKET=forecasts
INFLUXDB_USERNAME=admin
INFLUXDB_PASSWORD=password123
EOF
```

#### Load Your Data
```bash
# Restart the API to load new configuration
./scripts/run.sh

# Load CSV data into InfluxDB
curl -X POST "http://localhost:8000/admin/populate-data"

# Verify API is working
curl "http://localhost:8000/health"
```

## Project Structure

```
project-forecasting/
├── backend/
│   ├── core/                    # Core models and unified forecasting engine
│   │   ├── models.py           # XGBoost and Baseline model wrappers
│   │   ├── unified_forecaster.py # Optimized forecasting engine
│   │   ├── feature_config.py    # Centralized feature configuration
│   │   └── config.py           # Training and CV configuration
│   ├── forecasting/iterative/   # Iterative forecasting components
│   ├── training/               # Model training and preprocessing
│   ├── services/               # InfluxDB integration
│   ├── utils/                  # Configuration, logging, and metrics
│   ├── debug/                  # Development and debugging tools
│   └── main.py                 # FastAPI application
├── tests/                      # Test suite
├── input_data/                 # CSV data files with 1 hour frequency
├── model/                      # Trained model storage
├── forecasts/                  # Generated forecasts
└── output/                     # Combined datasets
```

## Configuration

Create a `.env` file from `.env.example` for custom configuration

## Data Format

### Required CSV Structure
```csv
Date,Value
2024-01-01 00:00:00,123.45
2024-01-01 01:00:00,134.56
```

**Requirements:**
- Date column in ISO format (YYYY-MM-DD HH:MM:SS)  
- Value column with numeric data
- Hourly frequency
- Minimum 26 weeks of data for training
- Files placed in `input_data/` directory

## Machine Learning Pipeline

### Model Training

Train XGBoost models with cross-validation:

```bash
python backend/debug/debug_trainer.py
```

**Training Process:**
- 5-fold expanding window cross-validation
- 26+ weeks training, 6 weeks evaluation
- Dual evaluation: standard (using actuals) and iterative (using predictions)
- Baseline comparison with seasonal averaging
- Automatic model persistence with metadata

**Feature Engineering:**
- Lag features (hourly, daily, weekly)
- Rolling statistics (mean, std over configurable windows)
- Time features (hour, day of week, month, weekend indicators)
- Cyclic transformations (sine/cosine for temporal patterns)

### Forecast Generation

After training the model, we can look at the xgboost evaluation excel file and choose a model from there to run the forecast generation on.

Advanced generation with custom parameters:
```bash
python backend/scripts/generate_future_forecast.py \
    --model_path "model/XGBoostWrapper_<model-id>.pkl" \
    --data_path "input_data/file1.csv" \
    --forecast_years 1.0 \
    --folder_name "experiment_1" \
    --visualize
```

**Capabilities:**
- 1000+ predictions/second with unified engine
- Batch processing for memory efficiency
- Automatic baseline comparison
- Interactive Plotly visualizations
- Feature store analysis

### Data Management

Combine forecasts with historical data:

```bash
python backend/debug/debug_combine.py \
    --folder_name "experiment_1" \
    --input_file "input_data/file1.csv" \
    --use_baseline  # Optional: use baseline forecasts
```

Features automatic file matching, 15-minute interval conversion, and chronological merging.

## API Usage

### Core Endpoints

**GET `/forecasts`** - Retrieve forecasts
```bash
curl "http://localhost:8000/forecasts?start=2025-06-15T00:00:00&end=2025-06-17T23:59:59&asset_id=SITE_1"
```

Parameters:
- `start` (required): Start timestamp in ISO format
- `end` (required): End timestamp in ISO format  
- `asset_id` (optional): Filter by asset (`SITE_1` or `SITE_2`)

**GET `/health`** - Health check
```bash
curl "http://localhost:8000/health"
```

**POST `/admin/populate-data`** - Load CSV data into InfluxDB
```bash
curl -X POST "http://localhost:8000/admin/populate-data"
```

### Response Structure
```json
{
  "status": "success",
  "forecasts": [
    {
      "timestamp": "2025-06-15T00:00:00",
      "asset_id": "SITE_1", 
      "value": 273.5,
      "data_type": "forecast"
    }
  ],
  "count": 48,
  "metadata": {
    "query_period": {
      "start": "2025-06-15T00:00:00",
      "end": "2025-06-17T23:59:59"
    },
    "assets": ["SITE_1"],
    "asset_filter": "SITE_1"
  }
}
```

## Testing

```bash
# Install test dependencies
pip install -r requirements-ci.txt

# Run complete test suite
tox

# Coverage report
pytest tests/ --cov=backend --cov-report=html
```

## Development Tools

**Model Management:**
```bash
python backend/debug/list_models.py  # List trained models with metadata
```

**Debugging Tools:**
- `debug_trainer.py`: Interactive model training and evaluation
- `debug_forecast.py`: Forecast generation with visualization
- `debug_combine.py`: Data combination and preprocessing

## Performance

**Evaluation Metrics:**
- WAPE (Weighted Absolute Percentage Error): Primary metric
- MAE (Mean Absolute Error): Absolute error measurement
- RMSE (Root Mean Square Error): Penalizes large errors
- MAPE (Mean Absolute Percentage Error): Percentage-based error

**Common Issues:**
- Insufficient data: Ensure 26+ weeks of hourly data
- CSV format: Verify Date/Value columns are correctly formatted
- Memory issues: Reduce batch size for large datasets

## License

MIT License
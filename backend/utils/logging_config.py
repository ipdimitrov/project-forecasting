"""
Centralized logging configuration for the forecasting application.
"""

import os
import logging
import logging.config
from pathlib import Path
from typing import Dict, Any
from .config import get_logging_config_from_env


def get_logging_config() -> Dict[str, Any]:
    """
    Get logging configuration dictionary.

    Returns:
        Dictionary configuration for Python logging
    """
    # Import here to avoid circular imports

    logging_config = get_logging_config_from_env()

    # Ensure logs directory exists
    logs_dir = Path(logging_config.log_file_path).parent
    logs_dir.mkdir(exist_ok=True)

    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": logging_config.log_format,
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
            "detailed": {
                "format": "%(asctime)s - %(name)s - %(levelname)s -\
                     %(pathname)s:%(lineno)d - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "console": {
                "level": logging_config.log_level,
                "class": "logging.StreamHandler",
                "formatter": "standard",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "forecasting": {
                "level": "DEBUG",
                "handlers": ["console"]
                + (["file", "error_file"] if logging_config.log_file_enabled else []),
                "propagate": False,
            },
            "uvicorn": {
                "level": "INFO",
                "handlers": ["console"]
                + (["file"] if logging_config.log_file_enabled else []),
                "propagate": False,
            },
            "fastapi": {
                "level": "INFO",
                "handlers": ["console"]
                + (["file"] if logging_config.log_file_enabled else []),
                "propagate": False,
            },
        },
        "root": {
            "level": logging_config.log_level,
            "handlers": ["console"]
            + (["file"] if logging_config.log_file_enabled else []),
        },
    }

    # Add file handlers only if enabled
    if logging_config.log_file_enabled:
        config["handlers"]["file"] = {
            "level": "DEBUG",
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "detailed",
            "filename": logging_config.log_file_path,
            "maxBytes": logging_config.log_file_max_size,
            "backupCount": logging_config.log_file_backup_count,
            "encoding": "utf8",
        }
        config["handlers"]["error_file"] = {
            "level": "ERROR",
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "detailed",
            "filename": logging_config.log_file_path.replace(".log", "_errors.log"),
            "maxBytes": logging_config.log_file_max_size,
            "backupCount": logging_config.log_file_backup_count,
            "encoding": "utf8",
        }

    return config


def setup_logging():
    """
    Setup logging configuration for the application.
    Should be called early in application startup.
    """
    config = get_logging_config()
    logging.config.dictConfig(config)

    # Set uvicorn access logging to be less verbose in production
    if os.getenv("ENVIRONMENT", "development") == "production":
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name.

    Args:
        name: Logger name, typically __name__ of the calling module

    Returns:
        Configured logger instance
    """
    # Ensure logging is configured
    if not logging.getLogger().handlers:
        setup_logging()

    return logging.getLogger(f"forecasting.{name}")

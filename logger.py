"""
InsightFlow — Structured Logging
Central logging setup for both Streamlit UI and FastAPI backend.
"""
import logging
import os
from datetime import datetime


def setup_logging(name: str = "insightflow", level: str = None) -> logging.Logger:
    """
    Setup structured logging with consistent format.
    Returns a logger instance.
    """
    log_level = getattr(logging, (level or os.getenv("LOG_LEVEL", "INFO")).upper(), logging.INFO)

    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(log_level)

    # console handler
    console = logging.StreamHandler()
    console.setLevel(log_level)
    console.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(console)

    # file handler — daily log file
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"insightflow_{datetime.now().strftime('%Y%m%d')}.log")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger


# pre-built loggers — import karo jahan chahiye
ui_logger     = setup_logging("insightflow.ui")
agent_logger  = setup_logging("insightflow.agents")
tool_logger   = setup_logging("insightflow.tools")
memory_logger = setup_logging("insightflow.memory")
api_logger    = setup_logging("insightflow.api")

import logging
import sys
import os
from logging.handlers import TimedRotatingFileHandler

# Define the directory for log files
LOGS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..', 'logs')
os.makedirs(LOGS_DIR, exist_ok=True)


def get_logger(name: str) -> logging.Logger:
    """
    Configures and returns a logger with a standard format.
    Logs to both console and a daily rotating file.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)

        # Create a standard formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # 1. Console Handler (for development visibility)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # 2. Daily Rotating File Handler
        log_file = os.path.join(LOGS_DIR, "app.log")
        # Rotates at midnight, keeps 7 days of backups
        file_handler = TimedRotatingFileHandler(log_file, when="midnight", interval=1, backupCount=7)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
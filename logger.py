# logger.py - Centraal logging systeem
import logging
import os
from datetime import datetime
from config import LOG_DIR, LOG_LEVEL

# Maak logs directory als die niet bestaat
os.makedirs(LOG_DIR, exist_ok=True)

def setup_logger(name: str) -> logging.Logger:
    """Setup een logger met file en console output."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOG_LEVEL))

    # Voorkom duplicate handlers
    if logger.handlers:
        return logger

    # Format
    formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (dagelijks bestand)
    today = datetime.now().strftime('%Y-%m-%d')
    file_handler = logging.FileHandler(
        os.path.join(LOG_DIR, f'{name}_{today}.log'),
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

# Pre-configured loggers
def get_scanner_logger():
    return setup_logger('scanner')

def get_watcher_logger():
    return setup_logger('watcher')

def get_executor_logger():
    return setup_logger('executor')

def get_monitor_logger():
    return setup_logger('monitor')

def get_telegram_logger():
    return setup_logger('telegram')

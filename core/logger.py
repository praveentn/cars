# core/logger.py
import logging
from core.config import settings

def setup_logging():
    """Setup application logging"""
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper()),
        format=settings.log_format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("cognitive_architecture.log")
        ]
    )


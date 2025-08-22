# crain_interface/core/logger.py
import logging
from core.config import settings

def setup_logging():
    """Setup application logging"""
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper()),
        format=settings.log_format if hasattr(settings, 'log_format') else '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("cognitive_architecture.log")
        ]
    )

# AttributeError: 'Settings' object has no attribute 'log_format'
# fix this error

import logging
import sys
from pathlib import Path
from src.config import Config

def setup_logger(name: str) -> logging.Logger:
    """Configures and returns a logger with the specified name and level from configuration."""
    logger = logging.getLogger(name)
    
    # If logger is already configured, return it
    if logger.handlers:
        return logger
        
    logger.setLevel(Config.LOG_LEVEL.upper())
    
    # Standard format: timestamp - name - level - message
    formatter = logging.Formatter(
        "[%(asctime)s] %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Stream Handler (stdout)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    # File Handler
    log_dir = Config.WORKSPACE_ROOT / "logs"
    try:
        log_dir.mkdir(exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "agriculture_aiot.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        # Fallback if log directory cannot be created
        print(f"Warning: Could not create file handler for logging: {e}", file=sys.stderr)
        
    return logger

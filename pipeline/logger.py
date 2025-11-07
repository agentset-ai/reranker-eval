"""
Logging utilities for pipeline
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


def setup_logger(
    name: str,
    log_file: Optional[Path] = None,
    level: int = logging.INFO,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Set up a logger with both file and console handlers
    
    Args:
        name: Logger name
        log_file: Optional path to log file
        level: Logging level
        format_string: Optional custom format string
    
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Default format
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    formatter = logging.Formatter(format_string)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class PipelineLogger:
    """Context manager for pipeline logging"""
    
    def __init__(self, run_paths, stage_name: str):
        self.run_paths = run_paths
        self.stage_name = stage_name
        self.log_file = run_paths.run_dir / f"{stage_name}.log"
        self.logger = None
    
    def __enter__(self):
        self.logger = setup_logger(
            f"pipeline.{self.stage_name}",
            log_file=self.log_file,
            level=logging.INFO
        )
        self.logger.info(f"=" * 70)
        self.logger.info(f"Starting stage: {self.stage_name}")
        self.logger.info(f"Run directory: {self.run_paths.run_dir}")
        self.logger.info(f"=" * 70)
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.logger:
            if exc_type is None:
                self.logger.info(f"=" * 70)
                self.logger.info(f"Completed stage: {self.stage_name}")
                self.logger.info(f"=" * 70)
            else:
                self.logger.error(f"Stage {self.stage_name} failed with error: {exc_val}", exc_info=True)
        return False  # Don't suppress exceptions


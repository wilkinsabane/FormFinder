import logging
import os
from typing import Optional

def setup_logging(name: str = "FormFinder", level: int = logging.INFO, fmt_str: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s", file_enabled: bool = True, file_dir: str = os.path.join("data", "logs")) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    try:
        from .config import get_config  # type: ignore
        cfg = getattr(get_config(), "logging", None)
        if cfg is not None:
            level = getattr(logging, str(getattr(cfg, "level", "INFO")).upper(), level)
            fmt_str = getattr(cfg, "format", fmt_str)
            fh_cfg = getattr(cfg, "file_handler", None)
            if fh_cfg is not None:
                file_enabled = bool(getattr(fh_cfg, "enabled", file_enabled))
                file_dir = str(getattr(fh_cfg, "directory", file_dir))
    except Exception:
        pass
    logger.setLevel(level)
    formatter = logging.Formatter(fmt_str)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    if file_enabled:
        os.makedirs(file_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(file_dir, f"{name.lower()}.log"))
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger

class _LazyLogger:
    def __init__(self, name: str = "FormFinder"):
        object.__setattr__(self, "_logger", None)
        object.__setattr__(self, "_name", name)
    def _ensure(self) -> None:
        if self._logger is None or not self._logger.handlers:
            self._logger = setup_logging(self._name)
    def __getattr__(self, name):
        self._ensure()
        return getattr(self._logger, name)
    def __setattr__(self, name, value):
        if name in {"_logger", "_name"}:
            object.__setattr__(self, name, value)
        else:
            self._ensure()
            setattr(self._logger, name, value)

log = _LazyLogger()

def get_logger(name: str = "FormFinder") -> logging.Logger:
    """
    Get a logger instance with the specified name.
    
    Args:
        name: Logger name
        
    Returns:
        Configured logger instance
    """
    return setup_logging(name)
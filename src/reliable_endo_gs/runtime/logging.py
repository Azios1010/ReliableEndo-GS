"""Explicit console logging configuration."""

import logging

LOGGER_NAME = "reliable_endo_gs"
_HANDLER_NAME = "reliable_endo_gs_console"


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure and return the project logger without duplicating handlers."""

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False

    for handler in logger.handlers:
        if handler.get_name() == _HANDLER_NAME:
            handler.setLevel(level)
            return logger

    handler = logging.StreamHandler()
    handler.set_name(_HANDLER_NAME)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    logger.addHandler(handler)
    return logger

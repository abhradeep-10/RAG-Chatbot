from __future__ import annotations

import json
import logging
import os


def setup_logging(level: str | None = None) -> None:
    level = (level or os.getenv("RAG_LOG_LEVEL", "INFO")).upper()
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    root.setLevel(level)


def log_event(logger: logging.Logger, event: str, **fields) -> None:
    """One JSON line per event, easy to grep or ship to a log pipeline."""
    logger.info(json.dumps({"event": event, **fields}, default=str, ensure_ascii=False))

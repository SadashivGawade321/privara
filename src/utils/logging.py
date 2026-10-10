"""Structured logging and event tracking for Privara."""

import logging
import time
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger("privara")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [Privara] %(message)s",
        datefmt="%H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class EventTracker:
    """In-memory event tracker for dashboard timeline and logging."""
    _instance = None

    def __init__(self):
        self.events: List[Dict[str, Any]] = []

    @classmethod
    def get_instance(cls) -> "EventTracker":
        if cls._instance is None:
            cls._instance = EventTracker()
        return cls._instance

    def log_event(self, event_type: str, message: str, details: Optional[str] = None, level: str = "info") -> None:
        """Record an event with timestamp, category, and display metadata."""
        now = datetime.now()
        event = {
            "timestamp": now.strftime("%H:%M:%S"),
            "time_iso": now.isoformat(),
            "event_type": event_type,
            "message": message,
            "details": details or "",
            "level": level,  # "info", "success", "warning", "error"
        }
        self.events.append(event)
        log_msg = f"{message}" + (f" | {details}" if details else "")
        if level == "warning":
            logger.warning(log_msg)
        elif level == "error":
            logger.error(log_msg)
        else:
            logger.info(log_msg)

    def get_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Return the most recent events."""
        return list(reversed(self.events[-limit:]))

    def clear(self) -> None:
        """Clear recorded events."""
        self.events.clear()


def log_event(event_type: str, message: str, details: Optional[str] = None, level: str = "info") -> None:
    """Convenience helper to log an event to global tracker."""
    EventTracker.get_instance().log_event(event_type, message, details, level)

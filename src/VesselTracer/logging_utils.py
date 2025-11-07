"""
Lightweight logging helpers with manual verbosity control.
"""
from __future__ import annotations

import contextlib
import os
import time
from typing import Optional


class Log:
    """Minimal logger that prints messages based on an integer level."""

    def __init__(self, enabled: bool = True, level: int = 1):
        self.enabled = enabled
        self.level = level

    def __call__(self, message: str, level: int = 1) -> None:
        if not self.enabled or level > self.level:
            return
        print(message)

    @contextlib.contextmanager
    def section(self, title: str, level: int = 1):
        """Emit section start/end messages with timing."""
        if not self.enabled or level > self.level:
            yield
            return

        start = time.time()
        print(f"▶ {title}")
        try:
            yield
        finally:
            duration = time.time() - start
            print(f"✓ {title} [{duration:.2f}s]")


def resolve_log_level(default_level: int, notes: Optional[dict] = None) -> int:
    """Determine final log level from env override or notes dict."""
    env_level = os.getenv("VA_DEBUG")
    if env_level is not None:
        try:
            return int(env_level)
        except ValueError:
            return default_level

    if notes:
        level = notes.get("debug_level")
        if level is not None:
            try:
                return int(level)
            except (TypeError, ValueError):
                pass
    return default_level

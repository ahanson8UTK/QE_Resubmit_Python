"""Logging helpers for consistent instrumentation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict

from rich.console import Console
from rich.logging import RichHandler


def setup_logging(level: str = "INFO", rich_tracebacks: bool = True) -> None:
    """Configure the root logger with optional rich tracebacks."""

    console = Console()
    handler = RichHandler(console=console, rich_tracebacks=rich_tracebacks)
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[handler],
    )


@dataclass
class WorkUnitLogger:
    """Track Kalman, gradient, and particle filter work units."""

    kalman_evals: int = 0
    kalman_grad_evals: int = 0
    particle_passes: int = 0
    extras: Dict[str, int] = field(default_factory=dict)

    def incr(self, **kwargs: int) -> None:
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, getattr(self, key) + int(value))
            else:
                self.extras[key] = self.extras.get(key, 0) + int(value)


__all__ = ["setup_logging", "WorkUnitLogger"]

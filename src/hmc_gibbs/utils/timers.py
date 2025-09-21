"""Utility timers and performance measurement helpers."""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass, field
from typing import Dict, Iterator


@dataclass
class TimerRecord:
    """Accumulates elapsed wall-clock time for a named section."""

    total: float = 0.0
    calls: int = 0

    def update(self, dt: float) -> None:
        self.total += dt
        self.calls += 1


@dataclass
class TimerRegistry:
    """Registry of timers keyed by string labels."""

    records: Dict[str, TimerRecord] = field(default_factory=dict)

    @contextlib.contextmanager
    def time(self, label: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - start
            self.records.setdefault(label, TimerRecord()).update(dt)


__all__ = ["TimerRecord", "TimerRegistry"]

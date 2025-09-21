"""Lightweight functional helpers used across modules."""

from __future__ import annotations

from typing import Callable, Iterable, TypeVar

T = TypeVar("T")
U = TypeVar("U")


def compose(*fns: Callable[[T], T]) -> Callable[[T], T]:
    """Compose unary callables from right to left."""

    def _inner(value: T) -> T:
        for fn in reversed(fns):
            value = fn(value)
        return value

    return _inner


def foreach(iterable: Iterable[T], fn: Callable[[T], U]) -> list[U]:
    """Apply ``fn`` to each element and return a list."""

    return [fn(item) for item in iterable]


__all__ = ["compose", "foreach"]

"""Utility helpers for writing diagnostics to disk."""

from __future__ import annotations

import json
import os
from typing import Any, Dict


def write_jsonl(path: str, record: Dict[str, Any]) -> None:
    """Append ``record`` as a JSON document to ``path``."""

    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        json.dump(record, handle)
        handle.write("\n")


__all__ = ["write_jsonl"]

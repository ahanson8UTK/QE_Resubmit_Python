"""Utilities for persisting Gibbs sampling runs."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict

import yaml


def build_run_id(prefix: str) -> str:
    return f"{prefix}_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}"


def prepare_run_directory(root: Path, run_id: str) -> Path:
    path = root / run_id / "summary"
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_metadata(path: Path, metadata: Dict[str, object]) -> None:
    with (path / "metadata.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(metadata, handle)


__all__ = ["build_run_id", "prepare_run_directory", "write_metadata"]

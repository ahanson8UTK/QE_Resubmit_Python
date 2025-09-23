from __future__ import annotations

import json
import os
import secrets
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "paths.yaml"
DEFAULT_ARTIFACT_DIRNAME = "artifacts"


def _read_paths_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise TypeError("paths.yaml must contain a mapping")
    return data


def _resolve_artifacts_root() -> Path:
    cfg = _read_paths_config(DEFAULT_CONFIG_PATH)
    root_value = cfg.get("artifacts", {}).get("root") if isinstance(cfg.get("artifacts"), dict) else None
    if root_value:
        root_path = Path(root_value)
        if not root_path.is_absolute():
            root_path = PROJECT_ROOT / root_path
    else:
        root_path = PROJECT_ROOT / DEFAULT_ARTIFACT_DIRNAME
    root_path.mkdir(parents=True, exist_ok=True)
    return root_path


def _short_git_rev() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    short = result.stdout.strip()
    return short or None


def _is_git_dirty() -> Optional[bool]:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return bool(result.stdout.strip())


def new_run_id(prefix: str) -> str:
    """Return a timestamped identifier using the Git short hash when available."""

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    suffix = _short_git_rev()
    if not suffix:
        suffix = secrets.token_hex(4)
    return f"{prefix}-{timestamp}-{suffix}"


class ArtifactRegistry:
    """File-system backed registry for pipeline artifacts."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_default(cls, root: Optional[str | os.PathLike[str]] = None) -> "ArtifactRegistry":
        """Build a registry using ``config/paths.yaml`` or the provided location."""

        if root is not None:
            root_path = Path(root)
            if not root_path.is_absolute():
                root_path = PROJECT_ROOT / root_path
            return cls(root_path)
        return cls(_resolve_artifacts_root())

    # --- directory helpers -------------------------------------------------
    def _kind_dir(self, kind: str) -> Path:
        path = self.root / kind
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _run_dir(self, kind: str, run_id: str) -> Path:
        path = self._kind_dir(kind) / run_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def equity_dir(self, run_id: str) -> Path:
        """Return the directory for an equity run, creating it if necessary."""

        return self._run_dir("equity", run_id)

    def net_fundamental_path(self, run_id: str) -> Path:
        """Return the Parquet path storing the net-fundamental series."""

        return self.equity_dir(run_id) / "net_fundamental.parquet"

    def manifest_path(self, run_id: str, kind: str = "equity") -> Path:
        """Return the manifest path for the given run identifier."""

        return self._run_dir(kind, run_id) / "manifest.json"

    # --- manifest utilities ------------------------------------------------
    def write_manifest(self, run_id: str, payload: Dict[str, Any], kind: str = "equity") -> Path:
        """Persist ``payload`` to ``manifest.json`` and update the ``_latest`` marker."""

        manifest_path = self.manifest_path(run_id, kind)
        self._atomic_write_json(manifest_path, payload)
        self._write_latest(kind, run_id)
        return manifest_path

    def load_manifest(self, run_id: str, kind: str = "equity") -> Dict[str, Any]:
        """Load the JSON manifest for ``run_id``."""

        manifest_path = self.manifest_path(run_id, kind)
        with manifest_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    def latest_run_id(self, kind: str = "equity") -> Optional[str]:
        """Return the most recent run identifier for ``kind`` if available."""

        latest_path = self._kind_dir(kind) / "_latest"
        if latest_path.is_symlink():
            target = latest_path.resolve()
            return target.name
        if latest_path.exists():
            content = latest_path.read_text(encoding="utf-8").strip()
            return content or None
        return None

    # --- low-level filesystem helpers -------------------------------------
    @staticmethod
    def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps(payload, indent=2, sort_keys=True)
        ArtifactRegistry._atomic_write_text(path, data)

    @staticmethod
    def _atomic_write_text(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", dir=str(path.parent), delete=False, encoding="utf-8") as tmp:
            tmp.write(text)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp_path = Path(tmp.name)
        tmp_path.replace(path)

    def _write_latest(self, kind: str, run_id: str) -> None:
        latest_path = self._kind_dir(kind) / "_latest"
        # Prefer text marker for portability.
        self._atomic_write_text(latest_path, f"{run_id}\n")


__all__ = ["ArtifactRegistry", "new_run_id", "_is_git_dirty"]

from __future__ import annotations

import json
import pickle
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class Checkpoint:
    iter: int
    burn_in: int
    thin: int
    rng_state_numpy: Dict[str, Any]
    jax_seed: int
    elapsed_sec: float
    # block states are user-defined opaque blobs
    bond_state: Optional[Dict[str, Any]] = None
    equity_run_id: Optional[str] = None
    bubble_state: Optional[Dict[str, Any]] = None
    diagnostics: Optional[Dict[str, Any]] = None


def atomic_write_bytes(target: Path, blob: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + f".tmp-{int(time.time() * 1000)}")
    with open(tmp, "wb") as f:
        f.write(blob)
    shutil.move(str(tmp), str(target))


def save_checkpoint(cp_dir: Path, cp: Checkpoint, tag: Optional[str] = None) -> Path:
    name = f"iter_{cp.iter:07d}"
    if tag:
        name += f"_{tag}"
    target = cp_dir / f"{name}.pkl"
    payload = pickle.dumps(cp, protocol=pickle.HIGHEST_PROTOCOL)
    atomic_write_bytes(target, payload)
    meta = cp_dir / f"{name}.json"
    atomic_write_bytes(meta, json.dumps(asdict(cp), default=str).encode("utf-8"))
    latest = cp_dir / "_latest.txt"
    atomic_write_bytes(latest, f"{target.name}\n".encode("utf-8"))
    return target


def load_latest_checkpoint(cp_dir: Path) -> Optional[Checkpoint]:
    latest = cp_dir / "_latest.txt"
    if not latest.exists():
        return None
    fname = cp_dir / latest.read_text().strip()
    if not fname.exists():
        return None
    with open(fname, "rb") as f:
        return pickle.load(f)

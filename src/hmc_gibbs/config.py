"""Configuration utilities for the :mod:`hmc_gibbs` sampler scaffold."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass
class RunConfig:
    """Runtime options for a Gibbs sampling run."""

    seed: int
    chains: int
    results_dir: Path
    run_id_prefix: str


@dataclass
class WarmupConfig:
    """Paths and hyper-parameters controlling warm-up behaviour."""

    config_path: Path


@dataclass
class DataConfig:
    """References to input data and flags for synthetic data generation."""

    paths_config: Path
    priors_config: Path
    synthetic: bool = False


@dataclass
class SamplerConfig:
    """Sampler budgets and particle filter settings."""

    sweeps: int
    warmup_sweeps: int
    production_sweeps: int
    particle_gibbs_particles: int


@dataclass
class LoggingConfig:
    """Logging verbosity and formatting options."""

    level: str = "INFO"
    rich_tracebacks: bool = True


@dataclass
class AppConfig:
    """Top-level configuration object composed of sub-configurations."""

    run: RunConfig
    warmup: WarmupConfig
    data: DataConfig
    sampler: SamplerConfig
    logging: LoggingConfig


def _coerce_path(value: Any) -> Path:
    if isinstance(value, Path):
        return value
    return Path(str(value))


def load_yaml(path: Path) -> Mapping[str, Any]:
    """Load a YAML document and return a mapping."""

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_app_config(path: Path) -> AppConfig:
    """Load :class:`AppConfig` from ``path``."""

    raw = load_yaml(path)
    run = raw.get("run", {})
    warmup = raw.get("warmup", {})
    data = raw.get("data", {})
    sampler = raw.get("sampler", {})
    logging_cfg = raw.get("logging", {})

    app_config = AppConfig(
        run=RunConfig(
            seed=int(run.get("seed", 0)),
            chains=int(run.get("chains", 1)),
            results_dir=_coerce_path(run.get("results_dir", "results")),
            run_id_prefix=str(run.get("run_id_prefix", "run")),
        ),
        warmup=WarmupConfig(config_path=_coerce_path(warmup.get("config_path", "configs/warmup.yaml"))),
        data=DataConfig(
            paths_config=_coerce_path(data.get("paths_config", "configs/data_paths.yaml")),
            priors_config=_coerce_path(data.get("priors_config", "configs/priors_example.yaml")),
            synthetic=bool(data.get("synthetic", False)),
        ),
        sampler=SamplerConfig(
            sweeps=int(sampler.get("sweeps", 1)),
            warmup_sweeps=int(sampler.get("warmup_sweeps", 1)),
            production_sweeps=int(sampler.get("production_sweeps", 1)),
            particle_gibbs_particles=int(sampler.get("particle_gibbs_particles", 16)),
        ),
        logging=LoggingConfig(
            level=str(logging_cfg.get("level", "INFO")),
            rich_tracebacks=bool(logging_cfg.get("rich_tracebacks", True)),
        ),
    )
    return app_config


__all__ = [
    "RunConfig",
    "WarmupConfig",
    "DataConfig",
    "SamplerConfig",
    "LoggingConfig",
    "AppConfig",
    "load_yaml",
    "load_app_config",
]

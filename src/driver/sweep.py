from __future__ import annotations

import importlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import arviz as az
import numpy as np
import yaml
from rich.console import Console
from rich.table import Table

from artifacts.registry import ArtifactRegistry, new_run_id
from data.random_m_loader import load_observed_inputs, sample_m
from utils.checkpoint import Checkpoint, load_latest_checkpoint, save_checkpoint
from utils.diagnostics import TraceAccumulator, compute_ess_per_sec

from .typing import DrawDict

console = Console()


# --------- Optional dynamic imports with graceful fallback stubs ----------
def _safe_import(name: str, attr: str, stub):
    try:
        mod = importlib.import_module(name)
        fn = getattr(mod, attr)
        return fn
    except Exception:  # pragma: no cover - fallback path only
        return stub


def _stub_run_bond_step(*args, **kwargs):  # pragma: no cover - fallback stub
    raise NotImplementedError(
        "run_bond_posterior_step not found. Create `bond/pipeline.py` with:\n"
        "def run_bond_posterior_step(m, observed, cfg, rng_key, resume_state=None)->(theta, states, aux, diag): ..."
    )


def _stub_price_equity(*args, **kwargs):  # pragma: no cover - fallback stub
    raise NotImplementedError(
        "price_equity_and_export not found. Create `equity/price_equity.py` with:\n"
        "def price_equity_and_export(theta, states, dividends, prices, cfg)->str: ... (returns equity run_id)"
    )


def _stub_run_bubble_step(*args, **kwargs):  # pragma: no cover - fallback stub
    raise NotImplementedError(
        "run_bubble_step not found. Create `bubble/pipeline.py` with:\n"
        "def run_bubble_step(net_fund_series, cfg, rng_key, resume_state=None)->(theta_b, latents_b, diag_b): ..."
    )


run_bond_posterior_step = _safe_import(
    "bond.pipeline", "run_bond_posterior_step", _stub_run_bond_step
)
price_equity_and_export = _safe_import(
    "equity.price_equity", "price_equity_and_export", _stub_price_equity
)
run_bubble_step = _safe_import(
    "bubble.pipeline", "run_bubble_step", _stub_run_bubble_step
)


# --------- typing helpers ----------
@dataclass
class SweepState:
    iter0: int
    total: int
    burn_in: int
    thin: int
    numpy_rng: np.random.Generator
    jax_seed: int
    run_id: str


def _set_jax_platform(platform: str) -> None:
    if platform:
        os.environ["JAX_PLATFORM_NAME"] = platform
    # good hygiene for GPUs
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def _load_cfg(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _init_run(
    cfg: Dict[str, Any]
) -> Tuple[ArtifactRegistry, SweepState, Path, Path]:
    reg = ArtifactRegistry.from_default(cfg.get("paths", {}).get("artifacts_root"))
    run_id = new_run_id(prefix=cfg["mcmc"].get("run_prefix", "mcmc"))
    mcmc_dir = reg.root / "mcmc" / run_id
    checkpoints_dir = mcmc_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    (mcmc_dir / "diagnostics").mkdir(parents=True, exist_ok=True)
    (mcmc_dir / "traces").mkdir(parents=True, exist_ok=True)
    # seeds
    seed = int(cfg["mcmc"]["seed"])
    np_rng = np.random.default_rng(seed)
    jax_seed = seed + 1
    st = SweepState(
        iter0=0,
        total=int(cfg["mcmc"]["iters"]),
        burn_in=int(cfg["mcmc"]["burn_in"]),
        thin=int(cfg["mcmc"]["thin"]),
        numpy_rng=np_rng,
        jax_seed=jax_seed,
        run_id=run_id,
    )
    reg.write_manifest(run_id, {"kind": "mcmc", "config": cfg}, kind="mcmc")
    return reg, st, mcmc_dir, checkpoints_dir


def _resume_if_any(cp_dir: Path, st: SweepState) -> Tuple[int, Optional[Checkpoint]]:
    cp = load_latest_checkpoint(cp_dir)
    if cp is None:
        return 0, None
    console.print(f"[yellow]Resuming from checkpoint at iteration {cp.iter}[/yellow]")
    return cp.iter + 1, cp


def _ensure_dict(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    return {}


def sweep_once(
    cfg: Dict[str, Any],
    reg: ArtifactRegistry,
    st: SweepState,
    mcmc_dir: Path,
    iter_idx: int,
    observed: Dict[str, Any],
    trace: TraceAccumulator,
    prev_cp: Optional[Checkpoint],
) -> Checkpoint:
    t_iter0 = time.perf_counter()
    # 1) draw m
    m, m_idx = sample_m(cfg, st.numpy_rng)
    # 2) bond step
    theta_bond, states_bond, aux_bond, diag_bond = run_bond_posterior_step(
        m=m,
        observed=observed,
        cfg=cfg,
        rng_key=st.jax_seed + iter_idx,  # simple per-iter seed shift
        resume_state=None if prev_cp is None else prev_cp.bond_state,
    )
    # 3) equity pricing
    run_id_equity = price_equity_and_export(
        theta=theta_bond,
        states=states_bond,
        dividends=observed["div_growth"],
        prices=observed["price"],
        cfg=cfg,
    )
    # 4) bubble step
    try:
        from equity.pricing_io import load_net_fundamental

        net_fund = load_net_fundamental(run_id_equity, registry=reg)
    except Exception as exc:  # pragma: no cover - runtime error path
        raise RuntimeError(
            "Cannot locate net_fundamental output from equity step."
        ) from exc

    theta_bub, _, diag_bub = run_bubble_step(
        net_fund_series=net_fund,
        cfg=cfg,
        rng_key=st.jax_seed + 10_000 + iter_idx,
        resume_state=None if prev_cp is None else prev_cp.bubble_state,
    )

    # Collect minimal draws for diagnostics (add keys as needed)
    draws: DrawDict = {
        # examples — extend with your hardest parameters:
        "phi_gQ_eigs": np.asarray(_ensure_dict(theta_bond).get("phi_gQ_eigs", [])),
        "risk_prices": np.asarray(_ensure_dict(theta_bond).get("risk_prices", [])),
        "sigma_h": np.asarray(_ensure_dict(theta_bub).get("sigma_h", np.nan)),
        "phi_b": np.asarray(_ensure_dict(theta_bub).get("phi_b", np.nan)),
    }
    trace.add_draws(draws)

    # Build checkpoint
    elapsed = time.perf_counter() - t_iter0
    bond_aux = _ensure_dict(aux_bond)
    bubble_diag = _ensure_dict(diag_bub)
    cp = Checkpoint(
        iter=iter_idx,
        burn_in=st.burn_in,
        thin=st.thin,
        rng_state_numpy=st.numpy_rng.bit_generator.state,
        jax_seed=st.jax_seed,
        elapsed_sec=elapsed,
        bond_state=bond_aux.get("resume_state"),
        equity_run_id=run_id_equity,
        bubble_state=bubble_diag.get("resume_state"),
        diagnostics={
            "bond": diag_bond,
            "bubble": diag_bub,
            "m_index": m_idx,
        },
    )
    return cp


def _export_diagnostics(
    trace: TraceAccumulator,
    burn_in: int,
    thin: int,
    target_dir: Path,
) -> None:
    idata = trace.to_inferencedata(burn_in=burn_in, thin=thin)
    has_posterior = False
    if hasattr(idata, "posterior"):
        try:
            has_posterior = bool(idata.posterior.data_vars)
        except AttributeError:
            has_posterior = False
    if not has_posterior:
        return
    target_dir.mkdir(parents=True, exist_ok=True)
    trace_path = target_dir.parent / "traces" / "trace.nc"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    idata.to_netcdf(trace_path, mode="w")
    essps = compute_ess_per_sec(idata, trace.walltimes)
    df = az.summary(idata, var_names=list(idata.posterior.data_vars))
    df.to_csv(target_dir / "summary.csv")
    with open(target_dir / "ess_per_sec.json", "w", encoding="utf-8") as f:
        json.dump(essps, f, indent=2)


def run_sweep(config_path: str, resume: bool = False) -> None:
    cfg = _load_cfg(config_path)
    _set_jax_platform(cfg["mcmc"].get("platform", "cpu"))
    reg, st, mcmc_dir, cp_dir = _init_run(cfg)
    # Save a copy of config
    with open(mcmc_dir / "config_snapshot.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    observed = load_observed_inputs(cfg)
    start_iter = 0
    prev_cp: Optional[Checkpoint] = None
    if resume:
        start_iter, prev_cp = _resume_if_any(cp_dir, st)

    trace = TraceAccumulator()
    burn_in = st.burn_in
    chkevery = int(cfg["mcmc"]["checkpoint_every"])
    flushevery = int(cfg["mcmc"]["flush_every"])

    for it in range(start_iter, st.total):
        cp = sweep_once(cfg, reg, st, mcmc_dir, it, observed, trace, prev_cp)
        # checkpointing
        if (it + 1) % chkevery == 0 or it == st.total - 1:
            save_checkpoint(cp_dir, cp)
        # flush diagnostics
        if (it + 1) % flushevery == 0 or it == st.total - 1:
            _export_diagnostics(trace, burn_in=burn_in, thin=st.thin, target_dir=mcmc_dir / "diagnostics")
        prev_cp = cp

    # pretty print last summary to console
    idata = trace.to_inferencedata(burn_in=burn_in, thin=st.thin)
    has_posterior = False
    if hasattr(idata, "posterior"):
        try:
            has_posterior = bool(idata.posterior.data_vars)
        except AttributeError:
            has_posterior = False
    if has_posterior:
        df = az.summary(idata)
        table = Table(title="Posterior diagnostics")
        table.add_column("param")
        for col in df.columns:
            table.add_column(col)
        for idx, row in df.iterrows():
            table.add_row(str(idx), *[f"{v:.3g}" for v in row.values])
        console.print(table)

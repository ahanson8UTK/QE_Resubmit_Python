"""Synthetic integration test harness for the bubble PM-HMC block."""
from __future__ import annotations

import importlib
import inspect
from typing import Any, Callable, Dict, Iterable, Mapping, MutableMapping

import numpy as np

try:  # pragma: no cover - optional dependency for nicer plots
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover - matplotlib is optional for quick smoke tests
    plt = None  # type: ignore

import jax

from bubble.pmhmc import BubbleData, BubblePrior, PMHMCConfig


def _generate_synthetic_data(seed: int = 0, T: int = 200) -> BubbleData:
    """Create a lightweight synthetic bubble dataset for quick experiments."""

    rng = np.random.default_rng(seed)
    d_m = 2
    d_g = 2
    d_h = 3

    t = np.linspace(0.0, 4.0 * np.pi, T, dtype=float)
    seasonal = np.sin(t)

    y_net_fund = 0.2 * seasonal + rng.normal(0.0, 0.05, size=T)
    r_t = 0.01 + 0.002 * np.cos(t)

    m_t = np.column_stack(
        (
            0.5 + 0.1 * seasonal,
            -0.3 + 0.05 * np.cos(0.5 * t + 0.2),
        )
    )
    g_t = np.column_stack(
        (
            -0.2 + 0.05 * np.sin(0.25 * t + 0.5),
            0.4 + 0.08 * np.cos(0.33 * t + 0.1),
        )
    )

    h_t = 0.3 + 0.05 * rng.normal(size=(T, d_h))

    Dm_t = 0.75 + 0.1 * np.sin(0.15 * t)[:, None] * np.array([[1.0, 0.5]])
    Dg_t = 0.6 + 0.1 * np.cos(0.1 * t)[:, None] * np.array([[0.8, 1.0]])

    Sigma_m = np.eye(d_m, dtype=float)
    Sigma_g = np.eye(d_g, dtype=float)
    Sigma_gm = 0.1 * np.ones((d_g, d_m), dtype=float)

    return BubbleData(
        T=T,
        y_net_fund=y_net_fund,
        r_t=r_t,
        m_t=m_t,
        g_t=g_t,
        h_t=h_t,
        Dm_t=Dm_t,
        Dg_t=Dg_t,
        Sigma_m=Sigma_m,
        Sigma_g=Sigma_g,
        Sigma_gm=Sigma_gm,
    )


def _build_config(seed: int = 0) -> PMHMCConfig:
    """Construct a PM-HMC configuration tuned for quick feedback loops."""

    return PMHMCConfig(
        gh_nodes=20,
        N_is=32,
        seed=seed,
        hmc_num_warmup=500,
        hmc_num_samples=1000,
        hmc_step_size=0.01,
        hmc_num_steps=20,
        max_tree_depth=10,
        use_truncation=False,
    )


def _resolve_draw_fn() -> Callable[..., Any]:
    """Locate the ``draw_bubble_block`` routine from any implemented module."""

    candidate_modules = (
        "hmc_gibbs.bubble.pmhmc",
        "hmc_gibbs.bubble",
        "bubble.pmhmc",
    )
    for module_name in candidate_modules:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        draw = getattr(module, "draw_bubble_block", None)
        if callable(draw):
            return draw
    raise ImportError("Unable to locate `draw_bubble_block` implementation.")


def _call_draw(draw_fn: Callable[..., Any], cfg: PMHMCConfig, data: BubbleData, seed: int) -> Any:
    """Invoke ``draw_bubble_block`` while adapting to different call signatures."""

    prior = BubblePrior.gaussian(
        B0_scale=1.0,
        mu_b_scale=1.0,
        phi_b_scale=0.5,
        sigma_h_scale=0.5,
        rho_scale=0.5,
    )
    rng_key = jax.random.PRNGKey(seed)

    attempts: Iterable[Mapping[str, Any]] = (
        {"cfg": cfg, "data": data},
        {"config": cfg, "data": data},
        {"cfg": cfg, "data": data, "prior": prior},
        {"cfg": cfg, "data": data, "rng_key": rng_key},
        {"cfg": cfg, "data": data, "prior": prior, "rng_key": rng_key},
        {"config": cfg, "bubble_data": data, "prior": prior, "rng_key": rng_key},
    )

    for kwargs in attempts:
        try:
            return draw_fn(**kwargs)
        except TypeError:
            continue
    # Fall back to inspecting the signature for remaining required arguments.
    signature = inspect.signature(draw_fn)
    provided: MutableMapping[str, Any] = {}
    for name in signature.parameters:
        if name in ("cfg", "config", "pmhmc_config"):
            provided[name] = cfg
        elif name in ("data", "bubble_data"):
            provided[name] = data
        elif name == "prior":
            provided[name] = prior
        elif name in ("rng_key", "key", "prng_key"):
            provided[name] = rng_key
        elif name == "seed":
            provided[name] = seed
    return draw_fn(**provided)


def _maybe_dict(obj: Any) -> Dict[str, Any] | None:
    if isinstance(obj, Mapping):
        return dict(obj)
    if hasattr(obj, "_asdict"):
        return obj._asdict()  # type: ignore[attr-defined]
    if hasattr(obj, "__dict__"):
        return vars(obj)
    return None


def _summary_stats(samples: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(samples, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=0)),
        "p05": float(np.quantile(arr, 0.05)),
        "median": float(np.median(arr)),
        "p95": float(np.quantile(arr, 0.95)),
    }


def main() -> None:
    seed = 123
    data = _generate_synthetic_data(seed=seed)
    cfg = _build_config(seed=seed)

    draw_fn = _resolve_draw_fn()
    result = _call_draw(draw_fn, cfg, data, seed)

    result_dict = _maybe_dict(result) or {}
    if hasattr(result, "draws_constrained"):
        draws = result.draws_constrained  # type: ignore[attr-defined]
    else:
        draws = result_dict.get("draws_constrained", {})

    accept = result_dict.get("accept_rate") or result_dict.get("acceptance")
    if accept is None and hasattr(result, "accept_rate"):
        accept = getattr(result, "accept_rate")
    if accept is None:
        accept = result_dict.get("diagnostics", {}).get("acceptance")

    loglik = result_dict.get("log_like_hat") or result_dict.get("loglik_hat")
    if loglik is None and "diagnostics" in result_dict:
        loglik = result_dict["diagnostics"].get("loglik_hat")

    if isinstance(accept, np.ndarray):
        accept_value = float(np.mean(accept))
    elif accept is None:
        accept_value = float("nan")
    else:
        accept_value = float(accept)

    if isinstance(loglik, np.ndarray):
        loglik_value = float(np.mean(loglik))
    elif loglik is None:
        loglik_value = float("nan")
    else:
        loglik_value = float(loglik)

    print("Acceptance (mean):", accept_value)
    print("Mean loglik_hat:", loglik_value)

    if isinstance(draws, Mapping) and draws:
        phi_raw = draws.get("phi_b")
        phi_draws = (
            np.asarray(phi_raw, dtype=float)
            if phi_raw is not None
            else np.empty(0, dtype=float)
        )

        sigma_raw = draws.get("sigma_h")
        sigma_draws = (
            np.asarray(sigma_raw, dtype=float)
            if sigma_raw is not None
            else np.empty(0, dtype=float)
        )

        rho_bm = draws.get("rho_bm")
        rho_bg = draws.get("rho_bg")
        if rho_bm is not None and rho_bg is not None:
            rho_concat = np.concatenate(
                (np.asarray(rho_bm, dtype=float), np.asarray(rho_bg, dtype=float)),
                axis=-1,
            )
            rho_norm = np.linalg.norm(rho_concat, axis=-1)
        else:
            rho_norm = None

        if phi_draws.size:
            phi_stats = _summary_stats(phi_draws)
            print("phi_b summary:", phi_stats)
        if sigma_draws.size:
            sigma_stats = _summary_stats(sigma_draws)
            print("sigma_h summary:", sigma_stats)
        if rho_norm is not None:
            rho_stats = _summary_stats(np.asarray(rho_norm))
            print("||rho|| summary:", rho_stats)

        if plt is not None:
            traces = []
            labels = []
            if phi_draws.size:
                traces.append(phi_draws)
                labels.append("phi_b")
            if sigma_draws.size:
                traces.append(sigma_draws)
                labels.append("sigma_h")
            if rho_norm is not None:
                traces.append(np.asarray(rho_norm))
                labels.append("||rho||")

            if traces:
                fig, axes = plt.subplots(len(traces), 1, figsize=(8, 2.5 * len(traces)), sharex=True)
                if not isinstance(axes, np.ndarray):  # handle single subplot
                    axes = np.asarray([axes])
                for ax, series, label in zip(axes, traces, labels):
                    ax.plot(np.asarray(series, dtype=float))
                    ax.set_title(f"Trace: {label}")
                    ax.set_ylabel(label)
                axes[-1].set_xlabel("Iteration")
                fig.tight_layout()
                plt.show()
    else:
        print("draws_constrained not available on result; raw result: {}".format(result))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Benchmark Fisher Information Matrix (FIM) computation across Torch and JAX.

Compares forward-mode AD (torch.autograd.forward_ad vs jax.jvp) and exact Hessian
differentiation, evaluating parameter uncertainties, condition numbers, and latency.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

import torchwave
from torchwave.inference.fisher import fisher_information_matrix as torch_fim
from torchwave.detectors import aLIGOZeroDetHighPower as torch_psd

import jaxwave
from jaxwave.inference.fisher import fisher_information_matrix as jax_fim
from jaxwave.detectors import aLIGOZeroDetHighPower as jax_psd


def benchmark_fim_single_model(
    model_name: str = "IMRPhenomD",
    n_freqs: int = 500,
    flow: float = 20.0,
    fhigh: float = 512.0,
    iterations: int = 5,
    warmup: int = 2,
) -> Dict[str, Any]:
    """Benchmark FIM computation on a single model across backends."""
    freqs_t = torch.linspace(flow, fhigh, n_freqs, dtype=torch.float64)
    df_t = float(freqs_t[1] - freqs_t[0])
    psd_t = torch_psd(freqs_t)

    freqs_j = jnp.linspace(flow, fhigh, n_freqs, dtype=jnp.float64)
    df_j = float(freqs_j[1] - freqs_j[0])
    psd_j = jax_psd(freqs_j)

    params = {"mass1": 30.0, "mass2": 20.0, "chi1": 0.2, "chi2": -0.1}
    param_order = ["mass1", "mass2", "chi1", "chi2"]

    def wf_fn_torch(p):
        return torchwave.get_fd_waveform(
            model_name,
            mass1=p["mass1"],
            mass2=p["mass2"],
            chi1=p["chi1"],
            chi2=p["chi2"],
            sample_frequencies=freqs_t,
        )

    def wf_fn_jax(p):
        return jaxwave.get_fd_waveform(
            model_name,
            mass1=p["mass1"],
            mass2=p["mass2"],
            chi1=p["chi1"],
            chi2=p["chi2"],
            sample_frequencies=freqs_j,
        )

    # 1. Warmup
    for _ in range(warmup):
        _ = torch_fim(params, wf_fn_torch, psd_t, df=df_t, parameter_order=param_order)
        _ = jax_fim(params, wf_fn_jax, psd_j, df=df_j, parameter_order=param_order, method="autodiff")

    # 2. Benchmark Torch (Forward-Mode AD)
    torch_times = []
    fim_torch = None
    for _ in range(iterations):
        t0 = time.perf_counter()
        fim_torch, _ = torch_fim(params, wf_fn_torch, psd_t, df=df_t, parameter_order=param_order)
        t1 = time.perf_counter()
        torch_times.append((t1 - t0) * 1000.0)

    # 3. Benchmark JAX (Forward-Mode AD jvp)
    jax_times = []
    fim_jax = None
    for _ in range(iterations):
        t0 = time.perf_counter()
        fim_jax, _ = jax_fim(params, wf_fn_jax, psd_j, df=df_j, parameter_order=param_order, method="autodiff")
        t1 = time.perf_counter()
        jax_times.append((t1 - t0) * 1000.0)

    # Numerical Parity & Inversion
    t_mat = fim_torch.detach().cpu().numpy()
    j_mat = np.array(fim_jax)
    max_rel_diff = float(np.max(np.abs(t_mat - j_mat) / np.maximum(np.abs(t_mat), 1e-30)))

    # Condition numbers and standard deviations
    cond_t = float(np.linalg.cond(t_mat))
    cond_j = float(np.linalg.cond(j_mat))

    try:
        cov_t = np.linalg.inv(t_mat)
        sigmas_t = [float(np.sqrt(cov_t[i, i])) if cov_t[i, i] > 0 else float("nan") for i in range(len(param_order))]
    except Exception:
        sigmas_t = [float("nan")] * len(param_order)

    return {
        "model": model_name,
        "n_freqs": n_freqs,
        "parameters": param_order,
        "torch_fwad_median_ms": float(np.median(torch_times)),
        "jax_jvp_median_ms": float(np.median(jax_times)),
        "max_rel_diff": max_rel_diff,
        "condition_number": cond_t,
        "sigmas": dict(zip(param_order, sigmas_t)),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark Fisher Information Matrix across Torch and JAX.")
    parser.add_argument("--models", type=str, default="TaylorF2,IMRPhenomD,IMRPhenomXAS", help="Comma-separated model names.")
    parser.add_argument("--iterations", type=int, default=5, help="Number of timed iterations.")
    parser.add_argument("--warmup", type=int, default=2, help="Number of warmup iterations.")
    parser.add_argument("--json-out", type=str, default=None, help="Optional JSON output path.")

    args = parser.parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    print("=" * 85)
    print(f"FISHER INFORMATION MATRIX BENCHMARK (Forward-Mode AD Parity & Performance)")
    print(f"Models: {models} | Iterations: {args.iterations} (warmup: {args.warmup})")
    print("=" * 85)

    results = []
    print(f"{'Model':15s} | {'Torch fwAD (ms)':16s} | {'JAX jvp (ms)':14s} | {'Max Rel Diff':14s} | {'Cond Num':10s}")
    print("-" * 85)
    for m in models:
        try:
            res = benchmark_fim_single_model(m, iterations=args.iterations, warmup=args.warmup)
            results.append(res)
            print(f"{res['model']:15s} | {res['torch_fwad_median_ms']:16.2f} | {res['jax_jvp_median_ms']:14.2f} | {res['max_rel_diff']:14.2e} | {res['condition_number']:10.2e}")
        except Exception as e:
            print(f"{m:15s} | ERROR: {e}")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved FIM benchmark results to {args.json_out}")


if __name__ == "__main__":
    main()

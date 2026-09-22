"""Comprehensive Performance Benchmarking Suite for TorchWave.

Measures latency and throughput across models, devices (CPU, MPS, CUDA),
batch sizes, precisions (FP32, FP64), autograd backward passes, and
optionally compares against LALSimulation.

Can be run programmatically or via CLI:
    python -m torchwave.benchmark --models TaylorF2,IMRPhenomD
    torchwave-bench --models popular --batch-sizes 1,10,100
"""

import argparse
import contextlib
import copy
import json
import os
import platform
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from torchwave.waveforms import get_fd_waveform
from torchwave.protocol import calculate_sample_statistics

# Canonical parameters for all 17 catalog models
DEFAULT_MODEL_SPECS: Dict[str, Dict[str, Any]] = {
    "TaylorF2": {
        "params": {"mass1": 1.4, "mass2": 1.4, "chi1": 0.0, "chi2": 0.0, "f_ref": 30.0},
        "grid": {"flow": 30.0, "fhigh": 500.0, "df": 0.25},
        "lal_name": "TaylorF2",
    },
    "TaylorF2_HM": {
        "params": {"mass1": 1.4, "mass2": 1.4, "chi1": 0.0, "chi2": 0.0, "f_ref": 30.0},
        "grid": {"flow": 30.0, "fhigh": 500.0, "df": 0.25},
        "lal_name": None,
    },
    "TaylorF2_Eccentric": {
        "params": {"mass1": 1.4, "mass2": 1.4, "chi1": 0.0, "chi2": 0.0, "e0": 0.05, "f0_ecc": 30.0, "f_ref": 30.0},
        "grid": {"flow": 30.0, "fhigh": 500.0, "df": 0.25},
        "lal_name": "TaylorF2Ecc",
    },
    "IMRPhenomD": {
        "params": {"mass1": 30.0, "mass2": 20.0, "chi1": 0.3, "chi2": -0.2, "f_ref": 20.0},
        "grid": {"flow": 20.0, "fhigh": 512.0, "df": 0.25},
        "lal_name": "IMRPhenomD",
    },
    "IMRPhenomXAS": {
        "params": {"mass1": 35.0, "mass2": 30.0, "chi1": 0.2, "chi2": -0.3, "f_ref": 20.0},
        "grid": {"flow": 20.0, "fhigh": 512.0, "df": 0.25},
        "lal_name": "IMRPhenomXAS",
    },
    "IMRPhenomHM": {
        "params": {"mass1": 40.0, "mass2": 15.0, "chi1": 0.3, "chi2": -0.2, "f_ref": 20.0, "inclination": 0.8},
        "grid": {"flow": 20.0, "fhigh": 512.0, "df": 0.25},
        "lal_name": "IMRPhenomHM",
    },
    "IMRPhenomXHM": {
        "params": {"mass1": 40.0, "mass2": 20.0, "chi1": 0.3, "chi2": -0.1, "f_ref": 20.0, "inclination": 1.0},
        "grid": {"flow": 20.0, "fhigh": 512.0, "df": 0.25},
        "lal_name": "IMRPhenomXHM",
    },
    "IMRPhenomPv2": {
        "params": {
            "mass1": 35.0, "mass2": 25.0, "chi1": 0.2, "chi2": -0.1,
            "spin1x": 0.2, "spin1y": 0.1, "spin1z": 0.2,
            "spin2x": -0.1, "spin2y": 0.15, "spin2z": -0.1,
            "f_ref": 20.0, "inclination": 0.7,
        },
        "grid": {"flow": 20.0, "fhigh": 512.0, "df": 0.25},
        "lal_name": "IMRPhenomPv2",
    },
    "IMRPhenomXP": {
        "params": {
            "mass1": 35.0, "mass2": 25.0, "chi1": 0.2, "chi2": -0.1,
            "spin1x": 0.2, "spin1y": 0.1, "spin1z": 0.2,
            "spin2x": -0.1, "spin2y": 0.15, "spin2z": -0.1,
            "f_ref": 20.0, "inclination": 0.7,
        },
        "grid": {"flow": 20.0, "fhigh": 512.0, "df": 0.25},
        "lal_name": "IMRPhenomXP",
    },
    "IMRPhenomXPHM": {
        "params": {
            "mass1": 35.0, "mass2": 25.0, "chi1": 0.2, "chi2": -0.1,
            "spin1x": 0.2, "spin1y": 0.1, "spin1z": 0.2,
            "spin2x": -0.1, "spin2y": 0.15, "spin2z": -0.1,
            "f_ref": 20.0, "inclination": 0.7,
            "prec_version": 223,
        },
        "grid": {"flow": 20.0, "fhigh": 512.0, "df": 0.25},
        "lal_name": "IMRPhenomXPHM",
    },
    "IMRPhenomD_NRTidal": {
        "params": {"mass1": 1.4, "mass2": 1.3, "chi1": 0.05, "chi2": -0.02, "lambda1": 350.0, "lambda2": 420.0, "f_ref": 30.0},
        "grid": {"flow": 30.0, "fhigh": 1024.0, "df": 0.25},
        "lal_name": "IMRPhenomD_NRTidal",
    },
    "IMRPhenomD_NRTidalv2": {
        "params": {"mass1": 1.4, "mass2": 1.3, "chi1": 0.05, "chi2": -0.02, "lambda1": 350.0, "lambda2": 420.0, "f_ref": 30.0},
        "grid": {"flow": 30.0, "fhigh": 1024.0, "df": 0.25},
        "lal_name": "IMRPhenomD_NRTidalv2",
    },
    "IMRPhenomNSBH": {
        "params": {"mass1": 5.0, "mass2": 1.4, "chi1": 0.2, "chi2": 0.0, "lambda2": 400.0, "f_ref": 30.0},
        "grid": {"flow": 30.0, "fhigh": 1024.0, "df": 0.25},
        "lal_name": "IMRPhenomNSBH",
    },
    "KerrRingdown": {
        "params": {"mass1": 65.0, "mass2": 0.0, "chi1": 0.7},
        "grid": {"flow": 50.0, "fhigh": 500.0, "df": 0.25},
        "lal_name": None,
    },
    "SEOBNR_ODE": {
        "params": {"mass1": 30.0, "mass2": 20.0, "chi1": 0.0, "chi2": 0.0},
        "grid": {"flow": 20.0, "fhigh": 256.0, "df": 0.5},
        "lal_name": None,
    },
    "SEOBNR_ROM": {
        "params": {"mass1": 30.0, "mass2": 20.0, "chi1": 0.0, "chi2": 0.0},
        "grid": {"flow": 20.0, "fhigh": 256.0, "df": 0.5},
        "lal_name": None,
    },
    "TEOBResumS": {
        "params": {"mass1": 1.4, "mass2": 1.4, "chi1": 0.0, "chi2": 0.0, "lambda1": 0.0, "lambda2": 0.0},
        "grid": {"flow": 30.0, "fhigh": 500.0, "df": 0.5},
        "lal_name": None,
    },
}

ALL_MODELS = list(DEFAULT_MODEL_SPECS.keys())
POPULAR_MODELS = ["TaylorF2", "IMRPhenomD", "IMRPhenomXAS", "IMRPhenomXP", "IMRPhenomXHM", "IMRPhenomD_NRTidalv2"]


def _has_lal() -> bool:
    try:
        import lal
        import lalsimulation
        return True
    except ImportError:
        return False


def sync_device(dev: torch.device) -> None:
    """Ensure all pending operations on the target device have finished."""
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    elif dev.type == "mps":
        torch.mps.synchronize()


def resolve_device(requested: str) -> torch.device:
    """Resolve device string to torch.device with fallback logic."""
    req = requested.lower()
    if req == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if req == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available on this system.")
        return torch.device("mps")
    if req == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available on this system.")
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_dtype(requested: str, device: torch.device) -> torch.dtype:
    """Resolve dtype string to torch.dtype, handling MPS float64 limitation."""
    req = requested.lower()
    if req in ("float32", "fp32", "32"):
        dt = torch.float32
    elif req in ("float64", "fp64", "double", "64"):
        dt = torch.float64
    else:
        raise ValueError(f"Unsupported dtype: {requested}. Choose 'float32' or 'float64'.")

    if device.type == "mps" and dt == torch.float64:
        print("[Notice] Apple Silicon MPS does not support float64. Switching precision to float32.")
        return torch.float32
    return dt


def benchmark_lal_model(
    lal_name: str,
    params: Dict[str, Any],
    grid: Dict[str, float],
    iterations: int = 20,
    warmup: int = 5,
) -> Optional[Dict[str, float]]:
    """Benchmark LALSimulation single-waveform generation."""
    if not _has_lal() or not lal_name:
        return None

    import lal
    import lalsimulation as lalsim

    try:
        approx = lalsim.GetApproximantFromString(lal_name)
    except Exception:
        return None

    m1_si = float(params["mass1"]) * lal.MSUN_SI
    m2_si = float(params["mass2"]) * lal.MSUN_SI
    s1x = float(params.get("spin1x", 0.0))
    s1y = float(params.get("spin1y", 0.0))
    s1z = float(params.get("spin1z", params.get("chi1", 0.0)))
    s2x = float(params.get("spin2x", 0.0))
    s2y = float(params.get("spin2y", 0.0))
    s2z = float(params.get("spin2z", params.get("chi2", 0.0)))
    dist_si = float(params.get("distance", 100.0)) * 1e6 * lal.PC_SI
    inc = float(params.get("inclination", 0.0))
    phic = float(params.get("phic", 0.0))
    flow = float(grid["flow"])
    fhigh = float(grid["fhigh"])
    df = float(grid["df"])
    fref = float(params.get("f_ref", flow))
    e0 = float(params.get("e0", 0.0))

    extra = lal.CreateDict()
    if "lambda1" in params:
        lalsim.SimInspiralWaveformParamsInsertTidalLambda1(extra, float(params["lambda1"]))
    if "lambda2" in params:
        lalsim.SimInspiralWaveformParamsInsertTidalLambda2(extra, float(params["lambda2"]))
    if "prec_version" in params and hasattr(lalsim, "SimInspiralWaveformParamsInsertPhenomXPrecVersion"):
        lalsim.SimInspiralWaveformParamsInsertPhenomXPrecVersion(extra, int(params["prec_version"]))

    # Warmup
    for _ in range(warmup):
        _ = lalsim.SimInspiralChooseFDWaveform(
            m1_si, m2_si, s1x, s1y, s1z, s2x, s2y, s2z,
            dist_si, inc, phic, 0.0, e0, 0.0,
            df, flow, fhigh, fref, extra, approx
        )

    # Timing
    timings = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = lalsim.SimInspiralChooseFDWaveform(
            m1_si, m2_si, s1x, s1y, s1z, s2x, s2y, s2z,
            dist_si, inc, phic, 0.0, e0, 0.0,
            df, flow, fhigh, fref, extra, approx
        )
        t1 = time.perf_counter()
        timings.append((t1 - t0) * 1000.0)  # ms

    stats = calculate_sample_statistics(timings, batch_size=1)
    stats.update({
        "framework": "lalsimulation",
        "device": "cpu",
        "dtype": "float64",
        "lal_name": lal_name,
    })
    return stats


def benchmark_torchwave_model(
    model_name: str,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    grid: Dict[str, float],
    params: Dict[str, Any],
    iterations: int = 20,
    warmup: int = 5,
    measure_backward: bool = False,
    compile: bool = False,
    dynamic: Optional[bool] = None,
) -> Dict[str, Any]:
    """Benchmark TorchWave waveform generation on a specific model."""
    if dynamic is None:
        dynamic = True if compile else False

    flow = float(grid["flow"])
    fhigh = float(grid["fhigh"])
    df = float(grid["df"])
    n_freqs = int(round((fhigh - flow) / df)) + 1
    freqs = torch.linspace(flow, fhigh, n_freqs, dtype=dtype, device=device)

    # Build input parameters with consistent 1D rank (batch_size,) across B=1 and B>1
    # to allow Dynamo to generalize symbolic shapes without rank-transition recompilations.
    def make_inputs(req_grad: bool = False):
        kw = {}
        for k, v in params.items():
            if isinstance(v, (int, float)):
                t = torch.full((batch_size,), float(v), dtype=dtype, device=device)
                if req_grad and k in ("mass1", "mass2", "chi1", "chi2", "spin1z", "spin2z"):
                    t = t.clone().detach().requires_grad_(True)
                kw[k] = t
            elif isinstance(v, torch.Tensor):
                t = v.to(dtype=dtype, device=device)
                if t.ndim == 0:
                    t = t.unsqueeze(0).expand(batch_size)
                if req_grad and k in ("mass1", "mass2", "chi1", "chi2", "spin1z", "spin2z"):
                    t = t.clone().detach().requires_grad_(True)
                kw[k] = t
            else:
                kw[k] = v
        return kw

    def eval_fn(**kwargs):
        return get_fd_waveform(
            model_name,
            sample_frequencies=freqs,
            device=device,
            dtype=dtype,
            **kwargs,
        )

    t_compile = 0.0
    if compile:
        try:
            torch._dynamo.config.capture_scalar_outputs = True
            torch._dynamo.config.cache_size_limit = 256
        except Exception:
            pass
        eval_fn = torch.compile(eval_fn, fullgraph=False, dynamic=dynamic)

    ctx = contextlib.nullcontext() if measure_backward else torch.inference_mode()

    with ctx:
        # Warmup (and compile on first call if compile=True)
        t0_warmup = time.perf_counter()
        for _ in range(warmup):
            inputs = make_inputs(req_grad=measure_backward)
            hp, hc = eval_fn(**inputs)
            if measure_backward:
                loss = (hp.abs().sum() + hc.abs().sum())
                loss.backward()
            sync_device(device)
        t_compile = (time.perf_counter() - t0_warmup) * 1000.0

        # Measurement
        fwd_times: List[float] = []
        bwd_times: List[float] = []
        total_times: List[float] = []

        for _ in range(iterations):
            inputs = make_inputs(req_grad=measure_backward)

            sync_device(device)
            t_start = time.perf_counter()

            hp, hc = eval_fn(**inputs)
            sync_device(device)
            t_fwd_end = time.perf_counter()

            if measure_backward:
                loss = (hp.abs().sum() + hc.abs().sum())
                loss.backward()
                sync_device(device)
                t_bwd_end = time.perf_counter()
                bwd_times.append((t_bwd_end - t_fwd_end) * 1000.0)
                total_times.append((t_bwd_end - t_start) * 1000.0)
            else:
                total_times.append((t_fwd_end - t_start) * 1000.0)

            fwd_times.append((t_fwd_end - t_start) * 1000.0)

    stats = calculate_sample_statistics(total_times, batch_size)
    framework_name = "torchwave_compiled" if compile else "torchwave"
    stats.update({
        "framework": framework_name,
        "model": model_name,
        "batch_size": batch_size,
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "compiled": compile,
        "dynamic": dynamic,
        "compile_warmup_ms": round(t_compile, 2) if compile else 0.0,
        "n_freq_bins": n_freqs,
        "iterations": iterations,
    })

    if measure_backward:
        stats["forward_mean_ms"] = float(np.mean(fwd_times))
        stats["backward_mean_ms"] = float(np.mean(bwd_times))

    return stats



def benchmark_jaxwave_model(
    model_name: str,
    batch_size: int,
    grid: Dict[str, float],
    params: Dict[str, Any],
    iterations: int = 20,
    warmup: int = 5,
    use_jit: bool = True,
    measure_backward: bool = False,
    device_name: str = "cpu",
) -> Optional[Dict[str, Any]]:
    """Benchmark jaxwave waveform generation and reverse-mode autodiff gradients."""
    try:
        import jax
        import jax.numpy as jnp
        import jaxwave
    except ImportError:
        return None

    flow = float(grid["flow"])
    fhigh = float(grid["fhigh"])
    df = float(grid["df"])
    n_freqs = int(round((fhigh - flow) / df)) + 1
    freqs = jnp.linspace(flow, fhigh, n_freqs, dtype=jnp.float64)

    kw = {}
    diff_keys = []
    diff_vals = []
    for k, v in params.items():
        if isinstance(v, (int, float)):
            # Uniform 1D array (batch_size,) ensures rank consistency
            arr = jnp.full((batch_size,), float(v), dtype=jnp.float64)
            kw[k] = arr
            if measure_backward and k in ("mass1", "mass2", "chi1", "chi2", "spin1z", "spin2z"):
                diff_keys.append(k)
                diff_vals.append(arr)
        else:
            kw[k] = v

    def eval_raw(frequencies, **kwargs):
        return jaxwave.get_fd_waveform(model_name, sample_frequencies=frequencies, **kwargs)

    if measure_backward and diff_keys:
        fixed_kw = {k: v for k, v in kw.items() if k not in diff_keys}

        def loss_fn(d_vals):
            call_kw = dict(fixed_kw)
            for k, val in zip(diff_keys, d_vals):
                call_kw[k] = val
            hp, hc = eval_raw(freqs, **call_kw)
            return jnp.sum(jnp.abs(hp)**2 + jnp.abs(hc)**2).real

        def fwd_only(d_vals):
            call_kw = dict(fixed_kw)
            for k, val in zip(diff_keys, d_vals):
                call_kw[k] = val
            return eval_raw(freqs, **call_kw)

        vg_fn = jax.value_and_grad(loss_fn)
        if use_jit:
            vg_fn = jax.jit(vg_fn)
            fwd_only = jax.jit(fwd_only)

        d_inputs = tuple(diff_vals)

        # Warmup / JIT compile
        t0_compile = time.perf_counter()
        for _ in range(warmup):
            val, grads = vg_fn(d_inputs)
            val.block_until_ready()
            if isinstance(grads, (tuple, list)):
                grads[0].block_until_ready()
        t_compile = (time.perf_counter() - t0_compile) * 1000.0

        fwd_times: List[float] = []
        bwd_times: List[float] = []
        total_times: List[float] = []

        for _ in range(iterations):
            t0 = time.perf_counter()
            res = fwd_only(d_inputs)
            if isinstance(res, tuple):
                res[0].block_until_ready()
            t_fwd_end = time.perf_counter()
            fwd_ms = (t_fwd_end - t0) * 1000.0
            fwd_times.append(fwd_ms)

            t_start_vg = time.perf_counter()
            val, grads = vg_fn(d_inputs)
            val.block_until_ready()
            if isinstance(grads, (tuple, list)):
                grads[0].block_until_ready()
            t_end_vg = time.perf_counter()
            tot_ms = (t_end_vg - t_start_vg) * 1000.0
            total_times.append(tot_ms)
            bwd_times.append(max(0.0, tot_ms - fwd_ms))

        stats = calculate_sample_statistics(total_times, batch_size)
        stats["forward_mean_ms"] = float(np.mean(fwd_times))
        stats["backward_mean_ms"] = float(np.mean(bwd_times))
    else:
        eval_fn = eval_raw
        if use_jit:
            eval_fn = jax.jit(eval_fn)

        # Warmup / JIT compile
        t0_compile = time.perf_counter()
        for _ in range(warmup):
            res = eval_fn(freqs, **kw)
            if isinstance(res, tuple):
                res[0].block_until_ready()
                res[1].block_until_ready()
        t_compile = (time.perf_counter() - t0_compile) * 1000.0

        timings = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            res = eval_fn(freqs, **kw)
            if isinstance(res, tuple):
                res[0].block_until_ready()
                res[1].block_until_ready()
            t1 = time.perf_counter()
            timings.append((t1 - t0) * 1000.0)

        stats = calculate_sample_statistics(timings, batch_size)

    stats.update({
        "framework": "jaxwave",
        "model": model_name,
        "batch_size": batch_size,
        "device": device_name,
        "dtype": "float64",
        "use_jit": use_jit,
        "measure_backward": measure_backward,
        "compile_warmup_ms": round(t_compile, 2),
        "n_freq_bins": n_freqs,
        "iterations": iterations,
    })
    return stats


def benchmark_ripple_model(
    model_name: str,
    batch_size: int,
    grid: Dict[str, float],
    params: Dict[str, Any],
    iterations: int = 20,
    warmup: int = 5,
    use_jit: bool = True,
    device_name: str = "cpu",
) -> Optional[Dict[str, Any]]:
    """Benchmark ripple JAX waveform generation if ripple / ripplegw is installed."""
    try:
        import jax
        import jax.numpy as jnp
        has_ripplegw = False
        try:
            import ripplegw
            has_ripplegw = True
        except ImportError:
            import ripple
            from ripple import waveforms as r_waveforms
    except ImportError:
        return None

    flow = float(grid["flow"])
    fhigh = float(grid["fhigh"])
    df = float(grid["df"])
    n_freqs = int(round((fhigh - flow) / df)) + 1
    freqs = jnp.linspace(flow, fhigh, n_freqs, dtype=jnp.float64)

    m1 = float(params["mass1"])
    m2 = float(params["mass2"])
    s1z = float(params.get("spin1z", params.get("chi1", 0.0)))
    s2z = float(params.get("spin2z", params.get("chi2", 0.0)))
    dist = float(params.get("distance", 100.0))
    tc = 0.0
    phic = float(params.get("phic", 0.0))
    incl = float(params.get("inclination", 0.0))
    fref = float(params.get("f_ref", flow))

    if has_ripplegw:
        if model_name not in ripplegw.list_waveforms():
            return None
        wf = ripplegw.waveform(model_name, f_ref=fref)
        Mc, eta = ripplegw.conversions.ms_to_Mc_eta(jnp.array([m1, m2]))
        p_dict = {}
        for p_name in wf.parameter_names:
            if p_name == "M_c":
                val = Mc
            elif p_name == "eta":
                val = eta
            elif p_name in ("s1_z", "s1z"):
                val = s1z
            elif p_name in ("s2_z", "s2z"):
                val = s2z
            elif p_name in ("s1_x", "s1x"):
                val = float(params.get("spin1x", 0.0))
            elif p_name in ("s1_y", "s1y"):
                val = float(params.get("spin1y", 0.0))
            elif p_name in ("s2_x", "s2x"):
                val = float(params.get("spin2x", 0.0))
            elif p_name in ("s2_y", "s2y"):
                val = float(params.get("spin2y", 0.0))
            elif p_name == "d_L":
                val = dist
            elif p_name in ("phase_c", "phi_c"):
                val = phic
            elif p_name in ("iota", "inclination"):
                val = incl
            elif p_name in ("lambda_1", "lambda1"):
                val = float(params.get("lambda1", 0.0))
            elif p_name in ("lambda_2", "lambda2"):
                val = float(params.get("lambda2", 0.0))
            else:
                val = float(params.get(p_name, 0.0))

            if batch_size > 1:
                p_dict[p_name] = jnp.full((batch_size,), val, dtype=jnp.float64)
            else:
                p_dict[p_name] = jnp.float64(val)

        if batch_size > 1:
            def gen_fn(p, f):
                return jax.vmap(wf, in_axes=(None, 0))(f, p)
        else:
            def gen_fn(p, f):
                return wf(f, p)

        arg_input = p_dict
    else:
        ripple_model_map = {
            "TaylorF2": getattr(r_waveforms, "TaylorF2", None),
            "IMRPhenomD": getattr(r_waveforms, "IMRPhenomD", None),
            "IMRPhenomPv2": getattr(r_waveforms, "IMRPhenomPv2", None),
            "IMRPhenomXAS": getattr(r_waveforms, "IMRPhenomXAS", None),
        }
        r_mod = ripple_model_map.get(model_name)
        if r_mod is None:
            return None

        theta = jnp.array([m1, m2, s1z, s2z, dist, tc, phic, incl], dtype=jnp.float64)
        if batch_size > 1:
            arg_input = jnp.tile(theta, (batch_size, 1))
            def gen_fn(t_batch, f):
                return jax.vmap(lambda t: r_mod.gen_waveform(f, t, f_ref=fref))(t_batch)
        else:
            arg_input = theta
            def gen_fn(t_scalar, f):
                return r_mod.gen_waveform(f, t_scalar, f_ref=fref)

    if use_jit:
        gen_fn = jax.jit(gen_fn)

    def _sync(res):
        if isinstance(res, dict):
            for v in res.values():
                if hasattr(v, "block_until_ready"):
                    v.block_until_ready()
        elif isinstance(res, (tuple, list)):
            for v in res:
                if hasattr(v, "block_until_ready"):
                    v.block_until_ready()
        elif hasattr(res, "block_until_ready"):
            res.block_until_ready()

    # Warmup
    t0_compile = time.perf_counter()
    for _ in range(warmup):
        res = gen_fn(arg_input, freqs)
        _sync(res)
    t_compile = (time.perf_counter() - t0_compile) * 1000.0

    timings = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        res = gen_fn(arg_input, freqs)
        _sync(res)
        t1 = time.perf_counter()
        timings.append((t1 - t0) * 1000.0)

    stats = calculate_sample_statistics(timings, batch_size)
    stats.update({
        "framework": "ripple",
        "model": model_name,
        "batch_size": batch_size,
        "device": device_name,
        "dtype": "float64",
        "use_jit": use_jit,
        "compile_warmup_ms": round(t_compile, 2),
        "n_freq_bins": n_freqs,
        "iterations": iterations,
    })
    return stats


def run_benchmark_suite(
    models: List[str],
    batch_sizes: List[int],
    device: torch.device,
    dtype: torch.dtype,
    iterations: int = 20,
    warmup: int = 5,
    measure_backward: bool = False,
    compare_lal: bool = False,
    compile: bool = False,
    dynamic: Optional[bool] = None,
    backend: str = "torch",
) -> Dict[str, Any]:
    """Execute the full benchmark matrix across selected models and batch sizes."""
    results = []
    lal_cache: Dict[str, Optional[Dict[str, float]]] = {}

    backends_to_run = []
    if backend in ("torch", "all"):
        backends_to_run.append("torch")
    if backend in ("torch_compiled", "compiled", "all") or compile:
        if "torch_compiled" not in backends_to_run:
            backends_to_run.append("torch_compiled")
    if backend in ("jax", "jaxwave", "all"):
        backends_to_run.append("jaxwave")

    for model_name in models:
        spec = DEFAULT_MODEL_SPECS.get(model_name)
        if spec is None:
            print(f"Warning: model '{model_name}' not in default specs; skipping.")
            continue

        params = spec["params"]
        grid = spec["grid"]
        lal_name = spec.get("lal_name")

        # Run LAL baseline once per model if requested
        if compare_lal and lal_name and lal_name not in lal_cache:
            lal_res = benchmark_lal_model(
                lal_name, params, grid, iterations=iterations, warmup=warmup
            )
            lal_cache[lal_name] = lal_res
        elif compare_lal and not lal_name:
            lal_cache[model_name] = None

        for bsz in batch_sizes:
            for b_mode in backends_to_run:
                try:
                    if b_mode == "torch":
                        bench = benchmark_torchwave_model(
                            model_name=model_name,
                            batch_size=bsz,
                            device=device,
                            dtype=dtype,
                            grid=grid,
                            params=params,
                            iterations=iterations,
                            warmup=warmup,
                            measure_backward=measure_backward,
                            compile=False,
                        )
                    elif b_mode == "torch_compiled":
                        bench = benchmark_torchwave_model(
                            model_name=model_name,
                            batch_size=bsz,
                            device=device,
                            dtype=dtype,
                            grid=grid,
                            params=params,
                            iterations=iterations,
                            warmup=warmup,
                            measure_backward=measure_backward,
                            compile=True,
                            dynamic=dynamic,
                        )
                    elif b_mode == "jaxwave":
                        bench = benchmark_jaxwave_model(
                            model_name=model_name,
                            batch_size=bsz,
                            grid=grid,
                            params=params,
                            iterations=iterations,
                            warmup=warmup,
                            use_jit=True,
                            measure_backward=measure_backward,
                            device_name=str(device.type),
                        )
                        if bench is None:
                            continue

                    if compare_lal and lal_name:
                        l_res = lal_cache.get(lal_name)
                        if l_res:
                            bench["lal_mean_ms"] = l_res["mean_ms"]
                            bench["lal_wf_per_sec"] = l_res["wf_per_sec"]
                            # Speedup ratio: throughput vs LAL single-waveform throughput
                            bench["speedup_vs_lal"] = bench["wf_per_sec"] / l_res["wf_per_sec"]

                    results.append(bench)
                except Exception as e:
                    results.append({
                        "framework": b_mode,
                        "model": model_name,
                        "batch_size": bsz,
                        "error": f"{type(e).__name__}: {str(e)}",
                    })

    meta = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "iterations": iterations,
        "warmup": warmup,
        "measure_backward": measure_backward,
        "compare_lal": compare_lal,
        "compile": compile,
        "dynamic": dynamic,
        "backend": backend,
    }

    return {"metadata": meta, "benchmarks": results}



def format_table(data: Dict[str, Any]) -> str:
    """Format benchmark results into an ASCII summary table."""
    benchmarks = data.get("benchmarks", [])
    if not benchmarks:
        return "No benchmark results to display."

    compare_lal = data.get("metadata", {}).get("compare_lal", False)
    measure_backward = data.get("metadata", {}).get("measure_backward", False)
    has_multiple_frameworks = len({b.get("framework", "torch") for b in benchmarks}) > 1 or any(b.get("framework") != "torchwave" for b in benchmarks)

    headers = ["Model", "Batch", "Mean Latency", "Median", "P95", "Throughput (WF/s)"]
    if has_multiple_frameworks:
        headers.insert(0, "Framework")
    if measure_backward:
        headers.extend(["Fwd (ms)", "Bwd (ms)"])
    if compare_lal:
        headers.extend(["LAL (ms)", "Speedup"])

    rows = []
    for b in benchmarks:
        if "error" in b:
            err_row = [b["model"], str(b["batch_size"]), f"ERROR: {b['error'][:25]}", "-", "-", "-"]
            if has_multiple_frameworks:
                err_row.insert(0, b.get("framework", "torch"))
            if measure_backward:
                err_row.extend(["-", "-"])
            if compare_lal:
                err_row.extend(["-", "-"])
            rows.append(err_row)
            continue

        mean_str = f"{b['mean_ms']:.3f} ms"
        med_str = f"{b['median_ms']:.3f} ms"
        p95_str = f"{b['p95_ms']:.3f} ms"
        thr_str = f"{b['wf_per_sec']:,.1f}"

        row = [b["model"], str(b["batch_size"]), mean_str, med_str, p95_str, thr_str]
        if has_multiple_frameworks:
            row.insert(0, b.get("framework", "torch"))

        if measure_backward:
            fwd_str = f"{b.get('forward_mean_ms', 0.0):.3f}"
            bwd_str = f"{b.get('backward_mean_ms', 0.0):.3f}"
            row.extend([fwd_str, bwd_str])

        if compare_lal:
            lal_ms = b.get("lal_mean_ms")
            speedup = b.get("speedup_vs_lal")
            lal_str = f"{lal_ms:.3f}" if lal_ms is not None else "N/A"
            sp_str = f"{speedup:.2f}x" if speedup is not None else "N/A"
            row.extend([lal_str, sp_str])

        rows.append(row)

    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            if i < len(col_widths):
                col_widths[i] = max(col_widths[i], len(val))

    def make_line(vals, fill=" "):
        return " | ".join(val.ljust(col_widths[i], fill) for i, val in enumerate(vals))

    sep = "-+-".join("-" * w for w in col_widths)
    header_line = make_line(headers)
    table_lines = [header_line, sep] + [make_line(r) for r in rows]
    return "\n".join(table_lines)


def format_markdown_table(data: Dict[str, Any]) -> str:
    """Format benchmark results into GitHub-flavored Markdown."""
    benchmarks = data.get("benchmarks", [])
    if not benchmarks:
        return "No benchmark results."

    compare_lal = data.get("metadata", {}).get("compare_lal", False)
    measure_backward = data.get("metadata", {}).get("measure_backward", False)

    headers = ["Model", "Batch", "Latency (mean ± std)", "Throughput (WF/s)"]
    if measure_backward:
        headers.extend(["Forward", "Backward"])
    if compare_lal:
        headers.extend(["LAL (ms)", "Speedup vs LAL"])

    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for b in benchmarks:
        if "error" in b:
            lines.append(f"| {b['model']} | {b['batch_size']} | ERROR: {b['error']} | - |")
            continue
        row = [
            b["model"],
            str(b["batch_size"]),
            f"{b['mean_ms']:.3f} ± {b['std_ms']:.3f} ms",
            f"{b['wf_per_sec']:,.1f}",
        ]
        if measure_backward:
            row.extend([f"{b.get('forward_mean_ms', 0):.3f} ms", f"{b.get('backward_mean_ms', 0):.3f} ms"])
        if compare_lal:
            lal_ms = b.get("lal_mean_ms")
            speedup = b.get("speedup_vs_lal")
            row.append(f"{lal_ms:.3f} ms" if lal_ms is not None else "N/A")
            row.append(f"**{speedup:.2f}x**" if speedup is not None else "N/A")

        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="TorchWave Gravitational Waveform Benchmarking Suite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--models",
        type=str,
        default="popular",
        help="Comma-separated models to benchmark, or 'all', or 'popular'.",
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="1,10,100",
        help="Comma-separated batch sizes.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to benchmark on ('cpu', 'mps', 'cuda', 'auto').",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float64",
        help="Floating point precision ('float32', 'float64').",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=20,
        help="Number of timed benchmark iterations.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Number of untimed warmup iterations.",
    )
    parser.add_argument(
        "--backward",
        action="store_true",
        help="Measure forward + backward autograd gradient time.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile waveform function with torch.compile.",
    )
    parser.add_argument(
        "--dynamic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable dynamic shapes for torch.compile to avoid recompiles across varying batch sizes (default: True).",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="torch",
        choices=["torch", "torch_compiled", "jaxwave", "all"],
        help="Execution framework/backend to benchmark ('torch', 'torch_compiled', 'jaxwave', 'all').",
    )
    parser.add_argument(
        "--compare-lal",
        action="store_true",
        help="Benchmark corresponding LALSimulation approximant as baseline.",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default=None,
        help="Output path for JSON results.",
    )
    parser.add_argument(
        "--markdown-out",
        type=str,
        default=None,
        help="Output path for Markdown summary table.",
    )

    args = parser.parse_args(argv)

    # Parse models
    if args.models.lower() == "all":
        selected_models = ALL_MODELS
    elif args.models.lower() == "popular":
        selected_models = POPULAR_MODELS
    else:
        selected_models = [m.strip() for m in args.models.split(",") if m.strip()]

    # Parse batch sizes
    batch_sizes = [int(b.strip()) for b in args.batch_sizes.split(",") if b.strip()]

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    print("=" * 80)
    print("TORCHWAVE PERFORMANCE BENCHMARK")
    print(f"Device: {device} | Precision: {dtype} | Models: {len(selected_models)} | Batches: {batch_sizes}")
    print(f"Backend: {args.backend} | Compile: {args.compile} (dynamic={args.dynamic}) | Backward: {args.backward}")
    print(f"Iterations: {args.iterations} (warmup: {args.warmup})")
    print("=" * 80)

    suite_results = run_benchmark_suite(
        models=selected_models,
        batch_sizes=batch_sizes,
        device=device,
        dtype=dtype,
        iterations=args.iterations,
        warmup=args.warmup,
        measure_backward=args.backward,
        compare_lal=args.compare_lal,
        compile=args.compile,
        dynamic=args.dynamic,
        backend=args.backend,
    )

    table_str = format_table(suite_results)
    print("\n" + table_str + "\n")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(suite_results, f, indent=2)
        print(f"Saved JSON benchmark results to {args.json_out}")

    if args.markdown_out:
        md_str = format_markdown_table(suite_results)
        with open(args.markdown_out, "w") as f:
            f.write(md_str + "\n")
        print(f"Saved Markdown summary to {args.markdown_out}")


if __name__ == "__main__":
    main()

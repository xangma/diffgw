#!/usr/bin/env python3
"""Reproducible FP64 aligned-waveform comparison using public package APIs.

Requires NumPy, LALSuite, and the selected backends. No oracle source is read.
Every timed batch is checked against LAL; failures are skipped unless explicitly
requested as diagnostic observations with --record-unqualified.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
MODELS = ("TaylorF2", "IMRPhenomD", "IMRPhenomXAS")
PARAMETERS = ("mass1", "mass2", "chi1", "chi2")
FREQUENCIES = np.arange(20.0, 512.125, 0.25)
FIXED = dict(distance=400.0, inclination=0.7, phic=0.0, f_ref=20.0)


def parameter_batch(model, batch):
    """Distinct deterministic binaries; smaller batches are prefixes of larger ones."""
    rng = np.random.default_rng(20260921)
    unit = rng.random((batch, 4))
    if model == "TaylorF2":
        values = unit * [0.3, 0.2, 0.0, 0.0] + [1.4, 1.1, 0.0, 0.0]
    else:
        values = unit * [10.0, 5.0, 0.4, 0.4] + [30.0, 20.0, -0.2, -0.2]
    return values


def compare_waveforms(candidate, reference):
    """Direct L2 and flat-weight, constant-phase-maximized mismatch, per binary."""
    cand, ref = np.asarray(candidate), np.asarray(reference)
    if cand.shape != ref.shape or cand.ndim != 3 or cand.shape[0] != 2:
        raise ValueError(f"Expected matching (2, batch, frequency) arrays: {cand.shape}, {ref.shape}")
    if cand.dtype != np.complex128 or ref.dtype != np.complex128:
        raise ValueError(f"FP64 comparison requires complex128 output: {cand.dtype}, {ref.dtype}")
    if not np.isfinite(cand).all() or not np.isfinite(ref).all():
        raise ValueError("Non-finite waveform")
    nc, nr = np.linalg.norm(cand, axis=-1), np.linalg.norm(ref, axis=-1)
    if np.any(nr == 0) or np.any(nc == 0):
        raise ValueError("Zero-norm polarization cannot be qualified")
    l2 = np.linalg.norm(cand - ref, axis=-1) / nr
    overlap = np.abs(np.sum(cand.conj() * ref, axis=-1)) / (nc * nr)
    mismatch = np.maximum(0.0, 1.0 - np.minimum(overlap, 1.0))
    return dict(
        reference="LALSimulation.SimInspiralChooseFDWaveform",
        relative_l2=l2.tolist(),
        mismatch=mismatch.tolist(),
        max_relative_l2=float(l2.max()),
        max_mismatch=float(mismatch.max()),
        qualified=bool(l2.max() <= 1e-11 and mismatch.max() <= 1e-14),
    )


def statistics(samples, batch):
    values = np.asarray(samples)
    rng = np.random.default_rng(0)
    medians = np.median(rng.choice(values, size=(2000, len(values))), axis=1)
    ci = np.quantile(medians, [0.025, 0.975])
    median = float(np.median(values))
    return dict(
        raw_ms=values.tolist(),
        median_ms=median,
        median_ci95_ms=ci.tolist(),
        waveforms_per_second=batch * 1000 / median,
        amortized_ms=median / batch,
    )


def lal_adapter(model, params):
    import lal
    import lalsimulation as lalsim

    approx = lalsim.GetApproximantFromString(model)
    args = []
    for m1, m2, s1, s2 in params:
        args.append(
            (
                m1 * lal.MSUN_SI,
                m2 * lal.MSUN_SI,
                0.0,
                0.0,
                s1,
                0.0,
                0.0,
                s2,
                FIXED["distance"] * 1e6 * lal.PC_SI,
                FIXED["inclination"],
                FIXED["phic"],
                0.0,
                0.0,
                0.0,
                0.25,
                20.0,
                512.25,
                FIXED["f_ref"],
                lal.CreateDict(),
                approx,
            )
        )

    def evaluate():
        # LAL's uniform FD API includes zero-frequency padding. Exclude its
        # upper endpoint and slice the same 1,969 physical bins used elsewhere.
        waves = [lalsim.SimInspiralChooseFDWaveform(*a) for a in args]
        return tuple(np.stack([np.asarray(w[p].data.data)[80:2049] for w in waves]) for p in (0, 1))

    return evaluate, lambda result: None, lambda result: np.asarray(result)


def torch_adapter(model, params, device, compiled):
    import torch
    import torchwave

    torch.set_num_threads(1)
    frequencies = torch.as_tensor(FREQUENCIES, dtype=torch.float64, device=device)
    inputs = torch.as_tensor(params, dtype=torch.float64, device=device)

    def evaluate(p, f):
        return torchwave.get_fd_waveform(
            model,
            sample_frequencies=f,
            **dict(zip(PARAMETERS, p.unbind(1))),
            **FIXED,
            dtype=torch.float64,
            device=device,
        )

    if compiled:
        # Each model/shape is a separate experiment. Avoid hitting Dynamo's
        # per-code-object recompile limit and silently timing eager fallback.
        torch._dynamo.reset()
        torch._dynamo.utils.counters.clear()
        torch._dynamo.config.suppress_errors = False
    fn = torch.compile(evaluate, dynamic=False) if compiled else evaluate

    def call():
        with torch.inference_mode():
            return fn(inputs, frequencies)

    def sync(result):
        if device == "cuda":
            torch.cuda.synchronize()

    def host(result):
        if compiled and not torch._dynamo.utils.counters["stats"]["unique_graphs"]:
            raise RuntimeError("torch.compile captured no graph; refusing a compiled label")
        return np.stack([a.detach().cpu().numpy() for a in result])

    return call, sync, host


def jax_adapter(model, params, device, ripple):
    import jax

    jax.config.update("jax_enable_x64", True)
    target = jax.devices("gpu" if device == "cuda" else "cpu")[0]
    frequencies = jax.device_put(FREQUENCIES, target)
    values = jax.device_put(params, target)
    if ripple:
        import ripplegw

        waveform = ripplegw.waveform(model, f_ref=FIXED["f_ref"])
        m1, m2, s1, s2 = (values[:, i] for i in range(4))
        # Parameter conversion is setup work, outside both packages' timings.
        intrinsic = dict(
            M_c=(m1 * m2) ** (3 / 5) / (m1 + m2) ** (1 / 5),
            eta=m1 * m2 / (m1 + m2) ** 2,
            s1_z=s1,
            s2_z=s2,
        )
        fixed = dict(
            d_L=FIXED["distance"],
            t_c=0.0,
            phase_c=FIXED["phic"],
            iota=FIXED["inclination"],
            lambda_1=0.0,
            lambda_2=0.0,
        )
        inputs = {key: value for key, value in intrinsic.items() if key in waveform.parameter_names}
        fixed = {key: value for key, value in fixed.items() if key in waveform.parameter_names}

        def evaluate(p, f):
            # Match DiffGW's static extrinsic parameters: only intrinsic
            # masses/spins are runtime inputs to the compiled waveform.
            return jax.vmap(lambda one: waveform(f, {**fixed, **one}))(p)
    else:
        import jaxwave

        inputs = values

        def evaluate(p, f):
            return jaxwave.get_fd_waveform(
                model,
                sample_frequencies=f,
                **{key: p[:, i] for i, key in enumerate(PARAMETERS)},
                **FIXED,
            )

    fn = jax.jit(evaluate)

    def call():
        return fn(inputs, frequencies)

    def host(result):
        if isinstance(result, dict):
            result = (result["p"], result["c"])
        return np.stack([np.asarray(x) for x in result])

    return call, jax.block_until_ready, host


def save_receipt(path, receipt):
    body = dict(receipt)
    payload = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False)
    body["content_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(body, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def provenance():
    versions = {}
    for name in ("numpy", "torch", "jax", "jaxlib", "rippleGW", "lalsuite"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    try:
        gpu = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        gpu = None
    hashes = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((ROOT / "src").rglob("*.py"))
    }
    cpu = platform.processor()
    if Path("/proc/cpuinfo").exists():
        cpu = next(
            (
                line.split(":", 1)[1].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")
            ),
            cpu,
        )
    return dict(
        timestamp=datetime.now(timezone.utc).isoformat(),
        host=platform.node(),
        platform=platform.platform(),
        python=sys.version,
        cpu=cpu,
        cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        gpu=gpu,
        packages=versions,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
        environment={
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "XLA_FLAGS",
                "JAX_ENABLE_X64",
                "XLA_PYTHON_CLIENT_PREALLOCATE",
            )
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=["lal", "torch", "torch_compiled", "jax", "ripple"],
        default=["lal", "torch", "jax", "ripple"],
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 10, 100, 1000])
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--record-unqualified", action="store_true")
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.iterations < 15 or args.warmup < 3 or min(args.batches) < 1:
        parser.error("Require >=15 iterations, >=3 warmups and positive batches")
    if args.device == "cuda" and "lal" in args.backends:
        parser.error("LAL is a CPU baseline; run it separately with --device cpu")

    receipt = dict(
        schema="diffgw-package-comparison/v1",
        provenance=provenance(),
        source_commit=args.source_commit,
        configuration={**vars(args), "output": str(args.output)},
        fixed_parameters=FIXED,
        frequency_grid=dict(start=20.0, stop=512.0, delta=0.25, bins=len(FREQUENCIES)),
        qualification=dict(
            max_relative_l2=1e-11, max_mismatch=1e-14, weighting="flat", phase_maximized=True, time_maximized=False
        ),
        timing_boundary="Resident-input call through completed output; transfers and host conversion excluded. LAL includes loop/stack and padded uniform-grid API.",
        results=[],
    )
    save_receipt(args.output, receipt)
    for model in args.models:
        for batch in args.batches:
            params = parameter_batch(model, batch)
            oracle, _, host = lal_adapter(model, params)
            reference = host(oracle())  # Missing/failing LAL is fatal, never a fallback.
            for backend in args.backends:
                row = dict(
                    model=model,
                    backend=backend,
                    device=args.device,
                    dtype="float64",
                    batch=batch,
                    parameters=params.tolist(),
                )
                try:
                    if backend == "lal":
                        call, sync, to_host = lal_adapter(model, params)
                    elif backend.startswith("torch"):
                        call, sync, to_host = torch_adapter(model, params, args.device, backend == "torch_compiled")
                    else:
                        call, sync, to_host = jax_adapter(model, params, args.device, backend == "ripple")
                    t0 = time.perf_counter()
                    result = call()
                    sync(result)
                    row["first_call_ms"] = (time.perf_counter() - t0) * 1000
                    candidate = to_host(result)
                    row["output_dtype"] = str(candidate.dtype)
                    row["accuracy"] = compare_waveforms(candidate, reference)
                    if backend == "torch_compiled":
                        import torch

                        row["captured_graphs"] = torch._dynamo.utils.counters["stats"]["unique_graphs"]
                    row["status"] = "qualified" if row["accuracy"]["qualified"] else "unqualified"
                    if row["accuracy"]["qualified"] or args.record_unqualified:
                        for _ in range(args.warmup):
                            result = call()
                            sync(result)
                        samples = []
                        for _ in range(args.iterations):
                            sync(result)
                            t0 = time.perf_counter()
                            result = call()
                            sync(result)
                            samples.append((time.perf_counter() - t0) * 1000)
                        row["timing"] = statistics(samples, batch)
                    print(
                        f"{model} {backend} {args.device} B={batch}: {row['status']}, L2={row['accuracy']['max_relative_l2']:.3g}, median={row.get('timing', {}).get('median_ms')} ms",
                        flush=True,
                    )
                except Exception as exc:
                    row.update(status="error", error=f"{type(exc).__name__}: {exc}")
                    print(f"{model} {backend} B={batch}: {row['error']}", flush=True)
                receipt["results"].append(row)
                save_receipt(args.output, receipt)


if __name__ == "__main__":
    main()

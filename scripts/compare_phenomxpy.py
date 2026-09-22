#!/usr/bin/env python3
"""Matched FP64 CPU T/THM polarization benchmark against pinned PhenomXPy.

Run with NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1 JAX_ENABLE_X64=1 XLA_FLAGS=--xla_cpu_multi_thread_eigen=false.
See docs/benchmarking_protocol.md for the timing and qualification policy.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from compare_packages import ROOT, parameter_batch, provenance, save_receipt, statistics

TIMES = np.arange(-1024, 129, dtype=np.float64) / 2048
FIXED = dict(distance=400.0, inclination=0.7, phic=0.0, f_ref=20.0)
REFERENCE_COMMIT = "4121fafa0484f3db2a7352f4eea697faaaf29f65"
MAX_L2 = 1e-11
MAX_MISMATCH = 1e-14


def compare_polarizations(candidate, reference):
    """Reject invalid arrays before assessing every binary and polarization."""
    if candidate.shape != reference.shape or candidate.ndim != 3 or candidate.shape[0] != 2:
        raise ValueError("Expected matching (2, batch, time) polarizations")
    if candidate.dtype != np.float64 or reference.dtype != np.float64:
        raise ValueError("Expected float64 polarizations")
    if not np.isfinite(candidate).all() or not np.isfinite(reference).all():
        raise ValueError("Nonfinite polarization")
    nr = np.linalg.norm(reference, axis=-1)
    nc = np.linalg.norm(candidate, axis=-1)
    if np.any(nr == 0) or np.any(nc == 0):
        raise ValueError("Zero-norm polarization")
    l2 = np.linalg.norm(candidate - reference, axis=-1) / nr
    # Real TD overlap: no phase, sign, polarization or time maximization.
    mismatch = np.maximum(0, 1 - np.sum(candidate * reference, axis=-1) / (nc * nr))
    return dict(
        relative_l2=l2.tolist(),
        mismatch=mismatch.tolist(),
        max_relative_l2=float(l2.max()),
        max_mismatch=float(mismatch.max()),
        qualified=bool(np.all(l2 <= MAX_L2) and np.all(mismatch <= MAX_MISMATCH)),
    )


def reference_adapter(model, parameters):
    from phenomxpy.phenomt import phenomt

    cls = getattr(phenomt, model)
    # Conversion to the reference parameterization is outside the timer.
    inputs = [
        dict(
            eta=m1 * m2 / (m1 + m2) ** 2,
            total_mass=m1 + m2,
            s1=[0.0, 0.0, s1],
            s2=[0.0, 0.0, s2],
            f_min=20.0,
            f_ref=FIXED["f_ref"],
            distance=FIXED["distance"],
            inclination=FIXED["inclination"],
            phi_ref=FIXED["phic"],
            polarization_angle=0.0,
            delta_t=1 / 2048,
            condition=False,
            cuda=False,
            numba_ansatze=True,
            compute_hlms_at_once=False,
        )
        for m1, m2, s1, s2 in parameters
    ]
    if model == "IMRPhenomTHM":
        for p in inputs:
            p["mode_array"] = [[l, sign * m] for l, m in [(2, 2), (2, 1), (3, 3), (4, 4), (5, 5)] for sign in (1, -1)]

    def call():
        # Reconstruct coefficients on every call, as DiffGW time_domain does.
        results = [cls(**p).compute_polarizations(TIMES) for p in inputs]
        return np.stack([np.stack(r[:2]) for r in results], axis=1), [r[2] for r in results]

    def host(result):
        polarizations, grids = result
        if any(not np.allclose(t, TIMES, rtol=0, atol=2 * np.finfo(np.float64).eps) for t in grids):
            raise ValueError("PhenomXPy returned a different physical time grid")
        return polarizations

    return call, lambda result: None, host


def native_adapter(model, parameters, backend):
    if backend == "torch":
        import torch
        import torchwave

        torch.set_num_threads(1)
        values = torch.as_tensor(parameters, dtype=torch.float64)
        times = torch.as_tensor(TIMES, dtype=torch.float64)
        waveform = getattr(torchwave, model)()

        @torch.inference_mode()
        def call():
            return waveform.time_domain(times, *values.unbind(-1), **FIXED)

        return call, lambda result: None, lambda result: np.stack([x.numpy() for x in result])
    import jax
    import jaxwave

    jax.config.update("jax_enable_x64", True)
    cpu = jax.devices("cpu")[0]
    values, times = jax.device_put(parameters, cpu), jax.device_put(TIMES, cpu)
    waveform = getattr(jaxwave, model)()
    fn = jax.jit(lambda p, t: waveform.time_domain(t, *[p[:, i] for i in range(4)], **FIXED))
    return lambda: fn(values, times), jax.block_until_ready, lambda result: np.stack([np.asarray(x) for x in result])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models", nargs="+", choices=["IMRPhenomT", "IMRPhenomTHM"], default=["IMRPhenomT", "IMRPhenomTHM"]
    )
    parser.add_argument(
        "--backends", nargs="+", choices=["phenomxpy", "torch", "jax"], default=["phenomxpy", "torch", "jax"]
    )
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 10, 100, 1000])
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--record-unqualified", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.iterations < 15 or args.warmup < 3 or min(args.batches) < 1:
        parser.error("Require >=15 iterations, >=3 warmups and positive batches")
    # Fix this before importing PhenomXPy, rather than silently selecting an
    # installed optional constants provider.
    os.environ["PHENOMXPY_CONSTANTS"] = "hardcoded"
    import numba

    numba.set_num_threads(1)
    import phenomxpy

    prov = provenance()
    prov["runner_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    prov["source_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    prov["source_dirty"] = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True))
    prov["packages"] = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
    dist = importlib.metadata.distribution("phenomxpy")
    direct_url = json.loads(dist.read_text("direct_url.json") or "{}")
    if direct_url.get("vcs_info", {}).get("commit_id") != REFERENCE_COMMIT:
        raise RuntimeError(f"Install PhenomXPy from git at {REFERENCE_COMMIT}; pin could not be verified")
    prov["phenomxpy_install"] = direct_url
    root = Path(phenomxpy.__file__).parent
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob("*.py"))}
    prov["phenomxpy_source_sha256"] = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    prov["source_files_sha256"] = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((ROOT / "src").rglob("*.py"))
    }
    prov["numba_threads"] = numba.get_num_threads()
    if Path("/usr/sbin/sysctl").exists():
        prov["cpu"] = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    receipt = dict(
        schema="diffgw-phenomxpy-comparison/v1",
        provenance=prov,
        configuration={**vars(args), "output": str(args.output)},
        fixed_parameters=FIXED,
        time_grid=dict(start=float(TIMES[0]), stop=float(TIMES[-1]), delta=1 / 2048, samples=len(TIMES)),
        qualification=dict(
            reference=f"PhenomXPy {REFERENCE_COMMIT}",
            max_relative_l2=MAX_L2,
            max_mismatch=MAX_MISMATCH,
            weighting="flat",
            phase_maximized=False,
            time_maximized=False,
        ),
        timing_boundary="Resident-input complete-batch polarizations, including coefficient construction. PhenomXPy scalar loop and stack; native tensor batching. No output conversion, input setup or transfers.",
        results=[],
    )
    save_receipt(args.output, receipt)
    for model in args.models:
        for batch in args.batches:
            params = parameter_batch("IMRPhenomD", batch)
            reference_call, _, reference_host = reference_adapter(model, params)
            reference = reference_host(reference_call())  # A missing/failing oracle is fatal.
            for backend in args.backends:
                row = dict(model=model, backend=backend, device="cpu", batch=batch, parameters=params.tolist())
                try:
                    call, sync, host = (
                        reference_adapter(model, params)
                        if backend == "phenomxpy"
                        else native_adapter(model, params, backend)
                    )
                    start = time.perf_counter()
                    result = call()
                    sync(result)
                    row["first_call_ms"] = 1000 * (time.perf_counter() - start)
                    candidate = host(result)
                    row["output_dtype"] = str(candidate.dtype)
                    row["accuracy"] = compare_polarizations(candidate, reference)
                    row["status"] = "qualified" if row["accuracy"]["qualified"] else "unqualified"
                    if row["status"] == "qualified" or args.record_unqualified:
                        for _ in range(args.warmup):
                            result = call()
                            sync(result)
                        samples = []
                        for _ in range(args.iterations):
                            sync(result)
                            start = time.perf_counter()
                            result = call()
                            sync(result)
                            samples.append(1000 * (time.perf_counter() - start))
                        row["timing"] = statistics(samples, batch)
                    print(
                        f"{model} {backend} B={batch}: {row['status']}, L2={row['accuracy']['max_relative_l2']:.3g}, median={row.get('timing', {}).get('median_ms')} ms",
                        flush=True,
                    )
                except Exception as exc:
                    row.update(status="error", error=f"{type(exc).__name__}: {exc}")
                    print(row["error"], flush=True)
                receipt["results"].append(row)
                save_receipt(args.output, receipt)


if __name__ == "__main__":
    main()

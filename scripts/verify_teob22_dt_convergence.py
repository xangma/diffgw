"""Reproducible clean-room 22-only dt convergence diagnostic.

This diagnostic compares the same nonspinning 100+100-Msun event on fixed
geometric time grids. It uses no waveform oracle or binary output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import socket
import platform
import sys
import time

import torch

from torchwave.approximants.eob.seobnrv4_hamiltonian import SEOBNRv4Hamiltonian
from torchwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux
from torchwave.approximants.eob.initial_conditions import find_quasicircular_initial_conditions
from torchwave.approximants.eob.teob_published_driver import integrate_teob22


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dt-values", type=float, nargs="+", default=[.5, .25, .125])
    parser.add_argument("--num-steps", type=int, default=1500)
    parser.add_argument("--reuse", action="store_true",
                        help="reuse dt-*.pt artifacts already in output-dir")
    args = parser.parse_args()
    if len(args.dt_values) < 2 or any(not math.isfinite(dt) or dt <= 0 for dt in args.dt_values):
        parser.error("at least two positive dt-values are required")
    if any(a <= b or abs(a / b - round(a / b)) > 1e-12
           for a, b in zip(args.dt_values, args.dt_values[1:])):
        parser.error("dt-values must be strictly descending integer ratios")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_paths = sorted((Path(__file__).resolve().parents[1] / "src/torchwave/approximants/eob").glob("*.py"))
    source_paths.append(Path(__file__).resolve())
    source_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    report = {"host": socket.gethostname(), "cwd": str(Path.cwd()),
              "pid": __import__("os").getpid(), "masses_solar": [100., 100.],
              "omega_start": .08, "stage": "clean-room factorized 22 on caller-supplied SEOB test trajectory",
              "mode": "reuse" if args.reuse else "integrate",
              "historical_integration_provenance": "unavailable for legacy artifacts",
              "runtime": {"python": sys.version, "torch": torch.__version__,
                          "platform": platform.platform()},
              "current_verification_sha256": source_hashes, "runs": [],
              "artifact_sha256": {}}
    (args.output_dir / "startup.json").write_text(json.dumps(report, indent=2) + "\n")
    if not args.reuse:
        dtype = torch.float64
        nu = torch.tensor(.25, dtype=dtype)
        zero = torch.tensor(0., dtype=dtype)
        ham = SEOBNRv4Hamiltonian(nu, zero, zero)
        rr = SEOBNRv4PublishedFlux(spin_truncation="tabulated")
        # A strong-field start keeps this diagnostic bounded while still testing
        # the physical event driver through its frequency peak.
        omega_start = torch.tensor(.08, dtype=dtype)
        r, pr, pphi = find_quasicircular_initial_conditions(ham, rr, omega_start)
        initial = torch.stack((r, torch.zeros_like(r), pr, pphi), dim=-1)
    results = []
    for dt in args.dt_values:
        started = time.monotonic()
        name = f"dt-{dt:g}.pt"
        if args.reuse:
            saved = torch.load(args.output_dir / name, weights_only=True)
            class _T: pass
            result = _T()
            result.mode, result.frequency = saved["mode"], saved["frequency"]
            result.trajectory = _T()
            result.trajectory.time = saved["time"]
            result.trajectory.valid = saved["valid"]
            result.trajectory.peak = _T() if "peak_time" in saved else None
            if result.trajectory.peak is not None:
                result.trajectory.peak.time = saved["peak_time"]
                result.trajectory.peak.frequency = saved["peak_frequency"]
        else:
            with torch.no_grad():
                result = integrate_teob22(initial, ham, rr, dt, args.num_steps,
                                          richardson_extrapolation=True)
            torch.save({"time": result.trajectory.time, "mode": result.mode,
                        "frequency": result.frequency, "valid": result.trajectory.valid,
                        "peak_time": result.trajectory.peak.time,
                        "peak_frequency": result.trajectory.peak.frequency},
                       args.output_dir / name)
        report["artifact_sha256"][name] = hashlib.sha256((args.output_dir / name).read_bytes()).hexdigest()
        report["runs"].append({"dt": dt, "artifact": name,
                                "seconds": time.monotonic() - started,
                                "peak_time": (None if result.trajectory.peak is None else result.trajectory.peak.time.tolist()),
                                "peak_frequency": (None if result.trajectory.peak is None else result.trajectory.peak.frequency.tolist()),
                                "grid_samples": int(result.trajectory.time.numel()),
                                "grid_end": float(result.trajectory.time[-1]),
                                "valid_samples": result.trajectory.valid.sum(0).tolist(),
                                "finite": bool(torch.isfinite(result.mode).all())})
        (args.output_dir / "progress.json").write_text(json.dumps(report, indent=2) + "\n")
        results.append(result)
    comparisons = []
    for coarse, fine, coarse_dt, fine_dt in zip(results, results[1:], args.dt_values, args.dt_values[1:]):
        stride = round(coarse_dt / fine_dt)
        n = min(coarse.mode.shape[-1], (fine.mode.shape[-1] - 1) // stride + 1)
        a, b = coarse.mode[..., :n], fine.mode[..., :n * stride:stride]
        coarse_time = coarse.trajectory.time[:n]
        fine_time = fine.trajectory.time[:n * stride:stride]
        if not bool(torch.equal(coarse_time, fine_time)):
            raise RuntimeError("saved dt grids do not share exact common times")
        valid = (coarse.trajectory.valid[:n] & fine.trajectory.valid[:n * stride:stride])
        a, b = a[..., valid], b[..., valid]
        comparisons.append({"dt_pair": [coarse_dt, fine_dt], "count": int(valid.sum()),
                            "window": [float(coarse_time[0]), float(coarse_time[-1])],
                            "relative_l2": float(torch.linalg.vector_norm(a-b) /
                                                 torch.linalg.vector_norm(b))})
    report["comparisons"] = comparisons
    report["current_verification_sha256_after"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    (args.output_dir / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

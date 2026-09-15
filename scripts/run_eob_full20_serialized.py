"""Run and atomically serialize an exploratory full 20 Hz published EOB case."""
import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import torch
from torchwave.approximants.eob.published_waveform import reconstruct_seobnrv4_22


MTSUN = 4.925490947641267e-6
DEPENDENCIES = [
    "src/torchwave/approximants/eob/published_waveform.py",
    "src/torchwave/approximants/eob/integrator.py",
    "src/torchwave/approximants/eob/seobnrv4_hamiltonian.py",
    "src/torchwave/approximants/eob/seobnrv4_flux.py",
    "src/torchwave/approximants/eob/seobnrv4_mode22.py",
    "src/torchwave/approximants/eob/seobnrv4_amplitude22.py",
]


def hashes(paths):
    return {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths
            if Path(p).exists()}


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dt", type=float, default=.5)
    ap.add_argument("--num-steps", type=int, default=None)
    ap.add_argument("--output-dir", type=Path, default=Path("/tmp/eob-full20-20260913"))
    args = ap.parse_args()
    steps = args.num_steps if args.num_steps is not None else int(round(6000 / args.dt))
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    omega = math.pi * (50 * MTSUN) * 20.
    choices = dict(spin_truncation="tabulated", spin_v6="bohe_total_energy",
                   include_pan_v8=False, include_positive_chi_540=False,
                   final_spin_fit_order=(1, 2), dt_geom=args.dt,
                   num_steps=steps, ringdown_duration=150.,
                   richardson_extrapolation=False)
    command = " ".join(sys.argv)
    startup = dict(timestamp_utc=__import__("datetime").datetime.now(
        __import__("datetime").timezone.utc).isoformat(), host=platform.node(),
        pid=os.getpid(), cwd=str(Path.cwd()), command=command,
        executable=sys.executable, parameters=dict(mass1=30., mass2=20.,
        chi1=0., chi2=0., f_lower=20., omega_start=omega), choices=choices,
        dependency_sha256_before=hashes(DEPENDENCIES))
    atomic_json(out / "startup.json", startup)
    started = time.monotonic()
    r = reconstruct_seobnrv4_22(torch.tensor(omega, dtype=torch.float64),
                                 30., 20., 0., 0., **choices)
    payload = dict(time=r.time, mode=r.mode, trajectory_time=r.trajectory.time,
                   state=r.trajectory.state, omega=r.trajectory.frequency,
                   valid=r.trajectory.valid, match_time=r.attachment.match_time,
                   peak_time=r.trajectory.peak.time)
    tmp = out / "full20.pt.tmp"
    torch.save(payload, tmp)
    tmp.replace(out / "full20.pt")
    source = hashes(DEPENDENCIES + ["scripts/run_eob_full20_serialized.py"])
    result = dict(timestamp_utc=startup["timestamp_utc"], seconds=time.monotonic() - started,
                  omega_start=omega, peak_time=float(r.trajectory.peak.time),
                  match_time=float(r.attachment.match_time),
                  samples=int(r.trajectory.valid.sum()), mode_samples=r.mode.shape[-1],
                  finite=bool(torch.isfinite(r.mode).all()), choices=choices,
                  source_sha256=source, dependency_sha256_after=hashes(DEPENDENCIES),
                  executable=sys.executable)
    atomic_json(out / "result.json", result)


if __name__ == "__main__":
    main()

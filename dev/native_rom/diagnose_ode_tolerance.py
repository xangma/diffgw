"""Development-only regional diagnosis of the SEOBNRv4 ODE-tolerance failure.

Run from the repository root with ``PYTHONPATH=src:. python -m
dev.native_rom.diagnose_ode_tolerance --json-out /tmp/ode-regions.json``.
The fixed time windows locate errors; they are not qualification gates.
"""

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import numpy as np
import torch

from dev.native_rom.qualify_generator import (
    Settings, canonical_hash, compare_runs, native_run, sha256, source_snapshot,
)


ROW = dict(omega_start=0.04, mass1=30.0, mass2=20.0, chi1=0.0, chi2=0.0)
LEVELS = (
    dict(name="ode_0", dt_geom=0.5, atol=1e-14, rtol=1e-13),
    dict(name="ode_1", dt_geom=0.5, atol=1e-15, rtol=1e-14),
    dict(name="ode_2", dt_geom=0.5, atol=1e-16, rtol=1e-15),
)
WINDOWS_M = (("early_inspiral", 0.0, 165.0),
             ("late_inspiral", 165.0, 300.0),
             ("pre_match", 300.0, 310.0),
             ("attachment", 310.0, 320.0),
             ("ringdown", 320.0, None))


def snapshot():
    sources = source_snapshot()
    source = Path(__file__).resolve()
    sources[str(source.relative_to(source.parents[2]))] = sha256(source)
    return canonical_hash(sources)


def regional_metrics(candidate, reference):
    if not np.array_equal(candidate["time"], reference["time"]):
        raise ValueError("ODE runs must share the exact output grid")
    time_axis = reference["time"]
    lhs, rhs = candidate["mode"], reference["mode"]
    error_power = abs(lhs - rhs) ** 2
    reference_power = abs(rhs) ** 2
    total_error = float(error_power.sum())
    total_reference = float(reference_power.sum())
    if total_error <= 0 or total_reference <= 0:
        raise ValueError("cannot decompose a zero-norm error or reference")
    regions = {}
    for name, start, end in WINDOWS_M:
        mask = (time_axis >= start) & ((time_axis < end) if end is not None else True)
        numerator = float(error_power[mask].sum())
        denominator = float(reference_power[mask].sum())
        regions[name] = dict(
            samples=int(mask.sum()),
            reference_energy_fraction=denominator / total_reference,
            error_energy_fraction=numerator / total_error,
            global_relative_l2_contribution=float(np.sqrt(numerator / total_reference)),
            local_relative_l2=float(np.sqrt(numerator / denominator)) if denominator else None,
            max_absolute_complex_difference=float(np.max(abs(lhs[mask] - rhs[mask]))),
        )
    if sum(region["samples"] for region in regions.values()) != len(time_axis):
        raise RuntimeError("diagnostic windows do not partition the waveform")
    return regions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args()
    source_before = snapshot()
    torch.set_num_threads(1)
    settings = Settings()
    runs = []
    for level in LEVELS:
        started = time.monotonic()
        run = native_run(ROW, level, settings)
        runs.append(run)
        print(json.dumps({"run": level["name"], "seconds": time.monotonic() - started,
                          "coarse_event_time_M": run["stages"]["coarse"]["event_time_M"],
                          "fine_event_time_M": run["stages"]["fine"]["event_time_M"],
                          "peak_time_M": run["peak_time_M"]}), flush=True)
    comparisons = {}
    for candidate, reference in zip(runs[1:], runs[:-1]):
        key = f'{reference["name"]}_to_{candidate["name"]}'
        comparisons[key] = dict(whole=compare_runs(candidate, reference),
                                regions=regional_metrics(candidate, reference))
        print(json.dumps({"comparison": key,
                          "complex_relative_l2": comparisons[key]["whole"]["complex_relative_l2"],
                          "error_energy_fractions": {
                              name: region["error_energy_fraction"]
                              for name, region in comparisons[key]["regions"].items()}}), flush=True)
    source_after = snapshot()
    report = dict(
        schema="diffgw.seobnrv4.ode_tolerance_diagnostic.v1",
        scope=dict(physics_validated=False, default_generator_qualified=False,
                   development_only=True, no_coefficient_fitting=True),
        row=ROW, levels=LEVELS,
        windows_M=[dict(name=name, start=start, end=end) for name, start, end in WINDOWS_M],
        runs=[{key: value for key, value in run.items() if key not in ("time", "mode")}
              | {"samples": len(run["time"])} for run in runs],
        comparisons=comparisons, source_sha256_before=source_before,
        source_sha256_after=source_after, sources_unchanged=source_before == source_after,
        process=dict(argv=sys.argv, cwd=os.getcwd(), host=socket.gethostname(), pid=os.getpid()),
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        versions=dict(torch=torch.__version__, numpy=np.__version__),
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(f"wrote {args.json_out}", flush=True)
    if source_before != source_after:
        raise RuntimeError("sources changed during the diagnostic")


if __name__ == "__main__":
    main()

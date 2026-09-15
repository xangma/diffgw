#!/usr/bin/env python3
"""Validate the finite controlled-P9 binary checkpoint/restart artifacts."""

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from id_quality import inspect_id

ROOT = Path("/campaign")
RUN = ROOT / "controlled-p9-lev0-evolution"
PROVENANCE = ROOT / "provenance"
IMAGE = "sha256:0ba0f3bb70b90e83c9b50316ecd37f1b55af4537160eafbe3392a6f67c4e9d77"
DATASETS = ("Norms.dat", "TimeSteps.dat", "ObservationAhA.dat", "ObservationAhB.dat")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def stage(name, expected, command):
    job = read_json(PROVENANCE / (name + "-job.json"))
    metrics = read_json(PROVENANCE / (name + "-metrics.json"))
    assert job["image_id"] == IMAGE and job["cpus"] == 32
    assert job["command"] == metrics["command"] == command
    assert metrics["exit_code"] == metrics["expected_exit_code"] == expected
    assert metrics["accepted_exit"] is True and metrics["watchdog_reason"] is None
    counters = dict(line.split() for line in metrics["cgroup_memory_events"].splitlines())
    assert all(int(counters[key]) == 0 for key in ("oom", "oom_kill", "oom_group_kill"))
    assert metrics["cwd"] == job["container_cwd"]
    return metrics


def inspect_segment(path):
    reductions = path / "BbhReductions.h5"
    tables = {}
    with h5py.File(reductions, "r") as handle:
        for name in DATASETS:
            dataset = handle[name]
            legend = [x.decode() if isinstance(x, bytes) else str(x)
                      for x in dataset.attrs["Legend"]]
            values = np.asarray(dataset, dtype=float)
            assert values.ndim == 2 and values.shape[0] > 0
            assert values.shape[1] == len(legend) and np.isfinite(values).all(), name
            times = values[:, legend.index("Time")]
            assert np.all(np.diff(times) >= 0), (name, times)
            tables[name] = {"rows": len(values), "times": times.tolist(),
                            "last": dict(zip(legend, values[-1].tolist()))}
    return {"tables": tables, "files": {
        str(p.relative_to(path)): {"bytes": p.stat().st_size, "sha256": sha256(p)}
        for p in sorted(path.glob("*.h5"))}}


def main():
    record = PROVENANCE / "controlled-p9-lev0-smoke-quality.json"
    assert not record.exists()
    quality = read_json(ROOT / "controlled-id/final-quality.json")
    run_dir = Path(quality["run_dir"])
    assert run_dir.is_absolute() and ".." not in run_dir.parts
    run_dir.resolve().relative_to((ROOT / "controlled-id").resolve())
    fresh = inspect_id(run_dir, 5e-5)
    assert fresh["target_converged"] is True and fresh["sha256"] == quality["sha256"]
    preparation = read_json(PROVENANCE / "controlled-p9-lev0-evolution-manifest.json")
    source_quality = Path(preparation["source"]["final_quality"])
    assert source_quality == ROOT / "controlled-id/final-quality.json"
    assert sha256(source_quality) == preparation["source"]["final_quality_sha256"]
    seg0, seg1 = RUN / "Segment_0000", RUN / "Segment_0001"
    original = seg0 / "Inspiral.yaml"
    assert sha256(original) == preparation["rendered_input_sha256"]
    checkpoint = seg0 / "Checkpoints/Checkpoint_0000"
    checkpoint_files = [p for p in checkpoint.rglob("*") if p.is_file()]
    assert checkpoint_files and sum(p.stat().st_size for p in checkpoint_files) > 0
    checkpoint_manifest = {str(p.relative_to(checkpoint)): {
        "bytes": p.stat().st_size, "sha256": sha256(p)}
        for p in sorted(checkpoint_files)}
    restart = read_json(PROVENANCE / "controlled-p9-lev0-restart-preparation.json")
    assert checkpoint_manifest == restart["checkpoint_files"]
    assert {"MainChares.dat", "RO.dat"} <= checkpoint_manifest.keys()
    assert restart["original_input_sha256"] == sha256(original)
    overlay = original.with_name("Inspiral.overlay_0000.yaml")
    assert restart["overlay"] == str(overlay) and restart["overlay_sha256"] == sha256(overlay)
    first = stage("controlled-p9-lev0-checkpoint", 2,
                  ["EvolveGhBinaryBlackHole", "+p32", "--input-file", str(original)])
    second = stage("controlled-p9-lev0-restart", 0,
                   ["EvolveGhBinaryBlackHole", "+p32", "+restart", str(checkpoint)])
    results = [inspect_segment(seg0), inspect_segment(seg1)]
    for name in DATASETS:
        times0 = results[0]["tables"][name]["times"]
        times1 = results[1]["tables"][name]["times"]
        assert abs(times0[0]) < 1e-12
        # A phase-change checkpoint may precede the observation at that slab.
        # The input gate fixes checkpoint >=0.5; reductions need not include 0.5
        # in both segments.
        assert times0[-1] < 1.0, (name, times0)
        assert times1[0] >= 0.5 - 1e-12, (name, times1)
        assert times1[0] >= times0[-1] - 1e-12, (name, times0, times1)
        assert times1[-1] >= 1.0 - 1e-12, (name, times1)
    elapsed = first["wall_seconds"] + second["wall_seconds"]
    result = {"validation_passed": True, "image_id": IMAGE, "num_procs": 32,
              "initial_data": fresh, "segments": results,
              "checkpoint_metrics": first, "restart_metrics": second,
              "checkpoint_files": checkpoint_manifest,
              "initial_transient_M_per_wall_hour": 3600.0 / elapsed,
              "scope": "Finite 1M operational test; no steady-throughput, eccentricity, waveform convergence or calibration claim"}
    record.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"record": str(record), "validation_passed": True,
                      "initial_transient_M_per_wall_hour": 3600.0 / elapsed}))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Read-only validator for completed single-hole checkpoint/restart artifacts."""
import argparse, hashlib, json
from pathlib import Path
import h5py
import numpy as np

DATASETS = ("TimeSteps.dat", "Norms.dat", "Errors.dat", "ApparentHorizon.dat")
IMAGE = "sha256:0ba0f3bb70b90e83c9b50316ecd37f1b55af4537160eafbe3392a6f67c4e9d77"
CHECKPOINT_COMMAND = [
    "EvolveGhSingleBlackHole", "+p4", "--input-file",
    "/campaign/smoke/single-kerrschild/Segment_0000/KerrSchild.yaml",
]
RESTART_COMMAND = [
    "EvolveGhSingleBlackHole", "+p4", "+restart",
    "/campaign/smoke/single-kerrschild/Segment_0000/Checkpoints/Checkpoint_0000",
]
REFERENCE_COMMAND = [
    "EvolveGhSingleBlackHole", "+p4", "--input-file",
    "/campaign/smoke/single-kerrschild-reference/KerrSchild.yaml",
]

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""): h.update(block)
    return h.hexdigest()

def dump_array(path, dataset):
    with h5py.File(path, "r") as handle:
        array = np.asarray(handle[dataset][...], dtype=float)
        legend = [x.decode() if isinstance(x, bytes) else str(x)
                  for x in handle[dataset].attrs["Legend"]]
    assert array.ndim == 2 and array.shape[1] == len(legend)
    assert "Time" in legend
    return array, legend

def inspect_segment(segment, require_start=True):
    reductions = segment / "GhKerrSchildReductions.h5"
    surfaces = segment / "GhKerrSchildSurfaces.h5"
    assert reductions.is_file(), reductions
    assert surfaces.is_file(), surfaces
    values = {}
    for dataset in DATASETS:
        values[dataset] = dump_array(reductions, dataset)
        assert np.isfinite(values[dataset][0]).all(), (segment, dataset)
    # All reduction tables have time in the first column; require a closed
    # reduction file with the expected start and nondecreasing times.
    for dataset, numbers in values.items():
        array, legend = values[dataset]
        times = array[:, legend.index("Time")]
        assert np.all(np.diff(times) >= 0), (dataset, times)
        if require_start:
            assert times[0] == 0.0, (dataset, times)
    return {"files": {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)}
                       for p in sorted(segment.glob("*.h5"))},
            "rows": {d: int(v[0].shape[0]) for d, v in values.items()},
            "final_times": {d: float(v[0][-1, v[1].index("Time")])
                            for d, v in values.items()}}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--checkpoint-metrics", type=Path, required=True)
    ap.add_argument("--restart-metrics", type=Path, required=True)
    ap.add_argument("--reference-metrics", type=Path, required=True)
    ap.add_argument("--checkpoint-job", type=Path, required=True)
    ap.add_argument("--restart-job", type=Path, required=True)
    ap.add_argument("--reference-job", type=Path, required=True)
    ap.add_argument("--reference-dir", type=Path, required=True)
    ap.add_argument("--record", type=Path, required=True)
    ap.add_argument("--checkpoint-manifest", type=Path, required=True)
    args = ap.parse_args()
    assert not args.record.exists()
    seg0, seg1 = args.run_dir / "Segment_0000", args.run_dir / "Segment_0001"
    checkpoint_manifest = args.checkpoint_manifest.read_text().splitlines()
    assert checkpoint_manifest and any("MainChares.dat" in line for line in checkpoint_manifest)
    metrics = [json.loads(p.read_text()) for p in
               (args.checkpoint_metrics, args.restart_metrics, args.reference_metrics)]
    jobs = [json.loads(p.read_text()) for p in
            (args.checkpoint_job, args.restart_job, args.reference_job)]
    expected_commands = (CHECKPOINT_COMMAND, RESTART_COMMAND, REFERENCE_COMMAND)
    expected_exits = (2, 0, 0)
    for job, record, command, expected in zip(jobs, metrics, expected_commands, expected_exits):
        assert job["image_id"] == IMAGE and job["cpus"] == 4, job
        assert job["command"] == record["command"] == command, (job, record)
        assert record["exit_code"] == record["expected_exit_code"] == expected, record
        assert record["accepted_exit"] is True and record["watchdog_reason"] is None, record
        counters = dict(line.split() for line in record["cgroup_memory_events"].splitlines())
        for name in ("oom", "oom_kill", "oom_group_kill"):
            assert name in counters and int(counters[name]) == 0, (name, counters)
    checkpoint_metrics, restart_metrics, reference_metrics = metrics
    result = {"image_id": jobs[0]["image_id"], "num_procs": 4,
              "checkpoint": {"manifest": str(args.checkpoint_manifest),
                             "files": [line.strip() for line in checkpoint_manifest]},
              "segment_0000": inspect_segment(seg0),
              "segment_0001": inspect_segment(seg1, require_start=False),
              "checkpoint_metrics": checkpoint_metrics,
              "restart_metrics": restart_metrics,
              "reference_metrics": reference_metrics,
              "scope": "read-only artifact validation; no launch performed"}
    for d in DATASETS:
        merged = [result["segment_0000"]["final_times"][d], result["segment_0001"]["final_times"][d]]
        assert merged == sorted(merged) and merged[0] == 0.0
    assert result["segment_0001"]["final_times"]["TimeSteps.dat"] >= 0.002
    reference = inspect_segment(args.reference_dir)
    comparisons = {}
    for dataset in ("Norms.dat", "Errors.dat", "ApparentHorizon.dat"):
        resumed, rlegend = dump_array(seg1 / "GhKerrSchildReductions.h5", dataset)
        direct, dlegend = dump_array(args.reference_dir / "GhKerrSchildReductions.h5", dataset)
        assert rlegend == dlegend
        a, b = resumed[-1], direct[-1]
        assert np.isclose(a[rlegend.index("Time")], 0.002, atol=1e-15)
        assert np.isclose(b[dlegend.index("Time")], 0.002, atol=1e-15)
        deltas = (a - b).tolist()
        comparisons[dataset] = {"resumed": a.tolist(), "reference": b.tolist(), "deltas": deltas,
                                "atol": 1e-12, "rtol": 1e-10,
                                "within_tolerance": bool(np.allclose(a, b, atol=1e-12, rtol=1e-10))}
    for dataset in DATASETS:
        _, legend = dump_array(args.reference_dir / "GhKerrSchildReductions.h5", dataset)
        reference_times = dump_array(args.reference_dir / "GhKerrSchildReductions.h5", dataset)[0][:, legend.index("Time")]
        assert np.isclose(reference_times[-1], 0.002, atol=1e-15), (dataset, reference_times)
    assert all(x["within_tolerance"] for x in comparisons.values()), comparisons
    result["reference"] = reference
    result["reference_comparison"] = comparisons
    result["validation_passed"] = True
    args.record.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))

if __name__ == "__main__": main()

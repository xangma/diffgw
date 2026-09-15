#!/usr/bin/env python3
"""Exercise continuous TEOB matching through native evolution and TD-to-FD."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import socket
import sys
import time

import torch

from torchwave.approximants.eob.teob_conditioned_waveform_clean import conditioned_teob_fd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sources = sorted((root / "src").rglob("*.py")) + [Path(__file__).resolve()]
    def hashes():
        return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    started = time.time()
    before = hashes()
    report = {"status": "running", "host": socket.gethostname(), "cwd": str(Path.cwd()),
              "pid": os.getpid(), "command": shlex.join([sys.executable, *sys.argv]),
              "started_unix": started, "source_hashes_before": before,
              "case": {"mass1": 30., "mass2": 20., "f_low": 20., "f_max": 1024., "delta_f": .5,
                       "radial_acceleration_policy": "frozen-mobility", "horizon_policy": "leading-order",
                       "matching_method": "continuous", "pa_initializer_policy": "analytic"}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "host", "cwd", "pid", "command")}), flush=True)
    try:
        with torch.no_grad():
            result = conditioned_teob_fd(**report["case"], dtype=torch.float64,
                                         pa_kwargs={"r_min": 14., "dr": .1, "pa_order": 8})
        report["checks"] = {
            "finite_td": bool(torch.isfinite(result.mode_22).all()),
            "finite_fd": bool(torch.isfinite(result.h_plus).all() & torch.isfinite(result.h_cross).all()),
            "nonzero_fd": bool((result.h_plus.abs() > 0).any()),
            "shared_peak": bool(result.nqc.t_peak == result.ringdown.events.peak_time),
            "shared_matching_time": bool(result.nqc.t_nqc == result.ringdown.events.match_time),
            "continuous_attachment": bool(result.nqc.t_nqc == result.ringdown.events.attachment_time),
            "uniform_td": bool(torch.allclose(torch.diff(result.td_time), torch.diff(result.td_time)[:1])),
        }
        report.update(td_samples=result.td_time.numel(), fd_samples=result.frequencies.numel(),
                      trajectory_samples=result.continuation.time.numel(), peak_time_M=float(result.nqc.t_peak),
                      matching_time_M=float(result.nqc.t_nqc), pa_status=result.pa.status,
                      continuation_status=result.continuation.status)
        report["status"] = "passed" if all(report["checks"].values()) else "failed"
    except Exception as exc:
        report.update(status="error", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["elapsed_seconds"] = time.time() - started
        after = hashes()
        report["changed_sources"] = [path for path in before if before[path] != after[path]]
        report["source_hashes_after"] = after
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if not k.startswith("source_hashes")}), flush=True)
    return 0 if report["status"] == "passed" and not report["changed_sources"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

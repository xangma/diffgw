"""Frozen nonzero-reference TD/FD adapter check against a pinned public binary."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import time

from . import qualify_generator as q
from .diagnose_td_fd_public_stages import _run_row

PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-25/td_fd_nonzero_reference_plan.json"
OUTPUT = PLAN.with_name("td_fd_nonzero_reference.json")


def save(report):
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def main() -> int:
    started = time.monotonic()
    plan = json.loads(PLAN.read_text())
    head = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], cwd=q.ROOT,
        text=True).strip()
    rows = [
        dict(name="mixed_spin_nonzero_reference", kind="public_seobnrv4",
             mass1=35.0, mass2=25.0, chi1z=0.4, chi2z=-0.3, f_low=20.0),
        dict(name="antialigned_nonzero_reference", kind="public_seobnrv4",
             mass1=40.0, mass2=25.0, chi1z=-0.6, chi2z=-0.3, f_low=25.0),
    ]
    expected = dict(
        schema="diffgw.seobnrv4.td-fd-nonzero-reference-plan.v1",
        release_qualified=False, source_head=head,
        binary_evidence="dev/native_rom/evidence/2026-09-24/public_nqc_mode_boundary.json",
        binary_evidence_sha256="a86784cd6a58ec4a574686d6940953a3d1be836e2d44ac8cdcd25a5ba7d0dffd",
        purpose="Check identical-input native TD conditioning and real FFT against the pinned public binary at nonzero reference frequency and phase, nontrivial inclination and distance, including complex FD samples and epoch.",
        sample_rate_hz=4096, rows=rows, public_distance_mpc=173.0,
        public_inclination_rad=1.1, public_reference_phase_rad=0.41,
        public_reference_frequency_hz=40.0, public_fd_fmax_hz=2048.0,
        public_fd_delta_f_hz=0.25,
        full_series_rms_relative_tolerance=1e-10,
        full_series_max_absolute_over_peak_tolerance=1e-10,
        fft_rms_relative_tolerance=1e-10,
        fft_max_absolute_over_peak_tolerance=1e-10,
        public_fd_epoch_tolerance_seconds=1e-9,
        operational_limits=dict(max_wall_time_s=300, max_rss_gib=32),
    )
    if plan != expected or head != "73428f3":
        raise ValueError("frozen nonzero-reference plan changed")
    binary_file = q.ROOT / plan["binary_evidence"]
    if q.sha256(binary_file) != plan["binary_evidence_sha256"]:
        raise ValueError("pinned public-binary evidence changed")
    frozen_binary = json.loads(binary_file.read_text())["lal_binary"]
    binary = q.lal_binary_identity(
        frozen_binary["lal_version"], frozen_binary["lalsimulation_version"])
    if binary != frozen_binary:
        raise ValueError("installed public binary differs from frozen oracle")
    report = dict(schema="diffgw.seobnrv4.td-fd-nonzero-reference.v1",
                  status="running", release_qualified=False,
                  source_head=head, plan_sha256=q.sha256(PLAN),
                  script_sha256=q.sha256(__file__), lal_binary=binary,
                  host=platform.node(), cwd=os.getcwd(), pid=os.getpid(),
                  rows=[])
    save(report)
    try:
        for row in rows:
            result = _run_row(row, plan, 1.0 / plan["sample_rate_hz"])
            report["rows"].append(result)
            save(report)
            print(row["name"], result["passed"], flush=True)
        report["status"] = (
            "passed" if all(row["passed"] for row in report["rows"])
            else "failed")
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = dict(type=type(exc).__name__, message=str(exc))
    report["elapsed_s"] = time.monotonic() - started
    save(report)
    print(json.dumps({"status": report["status"],
                      "error": report.get("error")}), flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

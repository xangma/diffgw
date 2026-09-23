"""Attribute capped mass-ratio-four ODE nonmonotonicity; development only."""

from __future__ import annotations

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

from . import qualify_generator as q
from .diagnose_boundary_strong_spin_nqc_inputs import _system
from .diagnose_boundary_strong_spin_ode import (
    array, array_hash, pre_attachment, raw_mode_of, regional_decomposition,
    stages_of,
)
from .diagnose_hermite_error import phase_attribution
from .diagnose_nqc_matching_inputs import _decompose
from .eob.torch.adaptive_driver_clean import reconstruct_seobnrv4_22_adaptive
from torchwave.approximants.eob.nqc import (
    seobnrv4_nqc_parameters, seobnrv4_nqc_targets,
)


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-23/boundary_q4_capped_attribution_plan.json"
CONTROL = q.ROOT / "dev/native_rom/evidence/2026-09-23/boundary_fine_step_cap.json"
CAMPAIGN_SCRIPT = "dev/native_rom/diagnose_parameter_boundaries.py"


def _serial_system(system):
    return {key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in system.items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    plan = json.loads(PLAN.read_text())
    control = json.loads(CONTROL.read_text())
    choices = dict(hermite_resampling=True, local_nqc_phase_derivatives=True,
                   local_matching_second_derivatives=True,
                   fixed_restart_time_M=165.0, output_grid_M=0.5)
    if (plan["schema"] != "diffgw.seobnrv4.boundary-q4-capped-attribution-plan.v1"
            or plan["fine_step_cap_evidence_sha256"] != q.sha256(CONTROL)
            or plan["row"] != "mass_ratio_four_nonspinning"
            or plan["levels"] != ["ode_0", "ode_1", "ode_2"]
            or plan["pairs"] != [["ode_0", "ode_1"], ["ode_1", "ode_2"]]
            or plan["fine_accepted_max_step_M"] != 0.125
            or plan["require_exact_capped_waveform_hashes"] is not True
            or plan["unchanged_choices"] != choices
            or control["schema"] != "diffgw.seobnrv4.boundary-fine-step-cap.v1"
            or control["status"] != "diagnostic_complete"
            or not control["sources_unchanged"]):
        parser.error("predeclared plan or frozen cap control mismatch")
    sources = q.source_snapshot()
    sources[CAMPAIGN_SCRIPT] = q.sha256(q.ROOT / CAMPAIGN_SCRIPT)
    if sources != control["source_sha256_before"]:
        parser.error("generator sources differ from capped control")
    control_row = control["rows"][plan["row"]]
    if (control_row["errors"] or not all(name in control_row["cases"]
                                        for name in plan["levels"])
            or control_row["controlled_ode_check"]["gates"]["error_decreased"] is not False):
        parser.error("capped mass-ratio-four failure absent")
    row = control_row["inputs"]
    parameters = [torch.tensor(row[key], dtype=torch.float64)
                  for key in ("mass1", "mass2", "chi1", "chi2")]
    targets = seobnrv4_nqc_targets(*seobnrv4_nqc_parameters(*parameters))
    report = dict(
        schema="diffgw.seobnrv4.boundary-q4-capped-attribution.v1",
        scope=dict(**q.SCOPE, development_only=True,
                   same_provisional_equations=True,
                   numerical_generator_qualified=False),
        status="running", plan_sha256=q.sha256(PLAN),
        control_sha256=q.sha256(CONTROL), script_sha256=q.sha256(__file__),
        source_sha256_before=sources,
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        process=dict(argv=sys.argv, cwd=os.getcwd(), host=socket.gethostname(), pid=os.getpid()),
        row=row, cap_M=plan["fine_accepted_max_step_M"], levels={}, pairs={})
    q.write_evidence(args.json_out, report)
    try:
        runs, results, systems = {}, {}, {}
        with torch.no_grad():
            for name in plan["levels"]:
                prior = control_row["cases"][name]
                level = prior["summary"]
                started = time.monotonic()
                result = reconstruct_seobnrv4_22_adaptive(
                    torch.tensor(row["omega_start"], dtype=torch.float64),
                    **{key: row[key] for key in ("mass1", "mass2", "chi1", "chi2")},
                    **q.CHOICES,
                    **{key: level[key] for key in ("atol", "rtol", "dt_geom")},
                    diagnostic_hermite_resampling=True,
                    diagnostic_local_nqc_derivatives=True,
                    diagnostic_matching_second_derivatives=True,
                    diagnostic_restart_time_M=165.0,
                    diagnostic_fine_max_step_M=plan["fine_accepted_max_step_M"],
                    ringdown_duration=150.0,
                    num_steps=20000, max_accepted_steps=20000,
                    max_evaluations=200000, max_rejected_steps=10000)
                trajectory = result.trajectory
                mode, axis = array(result.mode), array(result.time)
                hashes = dict(mode=array_hash(mode), time=array_hash(axis))
                if hashes != level["array_sha256"]:
                    raise ValueError(f"{name} waveform differs from frozen cap control")
                stages = stages_of(trajectory)
                if stages != prior["stages"]:
                    raise ValueError(f"{name} trajectory differs from frozen cap control")
                accepted = array(trajectory.fine.time)[array(trajectory.fine.valid).astype(bool)]
                if np.max(np.diff(accepted)) > plan["fine_accepted_max_step_M"] + 1e-12:
                    raise ValueError(f"{name} fine accepted-step cap violated")
                system = _system(result, targets)
                frozen_phase = prior["nqc_phase"]
                for key in ("matrix", "rhs", "coefficients"):
                    if system["phase"][key].tolist() != frozen_phase[key]:
                        raise ValueError(f"{name} phase matching {key} differs from cap control")
                raw_mode, valid = raw_mode_of(trajectory, row, len(mode))
                run = dict(name=name, dt_geom=level["dt_geom"], time=axis, mode=mode,
                           state=array(trajectory.state),
                           frequency=array(trajectory.frequency), valid=valid,
                           raw_mode=raw_mode,
                           peak_time_M=float(array(trajectory.peak.time)),
                           match_time_M=float(array(result.attachment.match_time)),
                           boundary_time_M=float(array(result.attachment.boundary_time)),
                           restart_time_M=float(array(trajectory.restart_time).reshape(-1)[0]))
                q.compare_runs(run, run)
                calibration = result.attachment.fine.calibration
                report["levels"][name] = dict(
                    waveform_sha256=hashes, stages=stages,
                    raw_mode_sha256=array_hash(raw_mode),
                    nqc_coefficients=dict(
                        amplitude=array(calibration.coefficients.amplitude).tolist(),
                        phase=array(calibration.coefficients.phase).tolist()),
                    nqc_matching=dict(
                        sampling_condition=float(array(calibration.sampling_condition)),
                        amplitude_condition=float(array(calibration.amplitude_condition)),
                        phase_condition=float(array(calibration.phase_condition)),
                        amplitude_residual=array(calibration.amplitude_residual).tolist(),
                        phase_residual=array(calibration.phase_residual).tolist()),
                    phase_system=_serial_system(system["phase"]),
                    elapsed_seconds=time.monotonic()-started)
                runs[name], results[name], systems[name] = run, result, system
                q.write_evidence(args.json_out, report)
                print(json.dumps(dict(level=name, seconds=report["levels"][name]["elapsed_seconds"],
                                      capped_hash_verified=True)), flush=True)
        for first, second in plan["pairs"]:
            candidate, reference = runs[first], runs[second]
            comparison = dict(
                whole=q.compare_runs(candidate, reference),
                regional=regional_decomposition(candidate, reference, plan),
                pre_attachment=pre_attachment(candidate, reference),
                nqc_phase_samples=phase_attribution(results[first], results[second]),
                phase_matching=_decompose(systems[first]["phase"],
                                          systems[second]["phase"]))
            key = f"{first}_to_{second}"
            frozen_key = ("coarse_to_medium" if first == "ode_0" else "medium_to_fine")
            if comparison["whole"] != control_row["controlled_ode_check"][frozen_key]:
                raise ValueError(f"{key} whole-waveform metrics differ from cap control")
            report["pairs"][key] = comparison
        report["source_sha256_after"] = q.source_snapshot()
        report["source_sha256_after"][CAMPAIGN_SCRIPT] = q.sha256(q.ROOT / CAMPAIGN_SCRIPT)
        report["sources_unchanged"] = report["source_sha256_after"] == sources
        if not report["sources_unchanged"]:
            raise ValueError("generator sources changed during attribution")
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = dict(type=type(exc).__name__, message=str(exc))
        q.write_evidence(args.json_out, report)
        raise
    q.write_evidence(args.json_out, report)
    print(json.dumps(dict(status=report["status"], evidence=str(args.json_out.resolve()))),
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

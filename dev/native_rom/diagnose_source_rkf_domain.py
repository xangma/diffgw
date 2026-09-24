"""Run the frozen selected-row source-physics/RKF45 numerical screen."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

import torch
import torchwave

from . import qualify_generator as q


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/source_rkf_domain_plan.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    parser.add_argument("--plan", type=Path, default=PLAN,
                        help="frozen numerical-screen plan; default is the three-row plan")
    parser.add_argument("--row", action="append", help="select a frozen row; default is all")
    args = parser.parse_args(argv)
    if args.json_out.exists():
        parser.error("output already exists; choose a new evidence path")
    plan_path = args.plan.resolve()
    if not plan_path.is_relative_to(q.ROOT / "dev/native_rom/evidence"):
        parser.error("plan must be under development evidence")
    plan = json.loads(plan_path.read_text())
    if (plan["schema"] != "diffgw.seobnrv4.source-rkf-domain-plan.v1"
            or plan["qualification_claim"] is not False):
        parser.error("unexpected qualification plan")
    prior_path = q.ROOT / plan["prior_source_candidate"]
    if q.sha256(prior_path) != plan["prior_source_candidate_sha256"]:
        parser.error("source candidate evidence hash differs from frozen plan")
    prior = json.loads(prior_path.read_text())
    choice = prior["choice"]
    if prior["status"] != "diagnostic_complete" or prior["release_qualified"] is not False:
        parser.error("prior candidate evidence is not diagnostic-complete")
    expected = dict(
        calibration="lal_v4_rounded",
        physical=dict(hamiltonian_log_derivative="lal_v4",
                      mode_33_higher_spin="lal_v4_rho", name="combined"),
        flux=dict(mode_21_higher_spin="rho_hybrid",
                  mode_31_higher_spin="lal_v4_rho",
                  mode_32_spin_square="lal_v4_zero"),
        mode_43_higher_spin="lal_v4_rho", spin_truncation="tabulated",
        nqc_phase_window_M=1.0)
    if any(choice[key] != value for key, value in expected.items()):
        parser.error("source-backed physical choices differ from frozen plan")
    candidate = plan["candidate"]
    if candidate != dict(
            fine_sampler="rkf45", coarse_sampler="natural_cubic",
            local_nqc_phase_derivatives=True, matching_second_derivatives=True,
            nqc_phase_window_M=1.0, coarse_initial_step_M=0.125,
            warm_start_Momega=0.03162277660168379,
            fixed_restart_time_M=100.0):
        parser.error("numerical candidate differs from frozen plan")
    rows = plan["rows"]
    q.validate_rows(list(rows.values()))
    selected = list(rows) if args.row is None else args.row
    if not selected or any(name not in rows for name in selected) or len(set(selected)) != len(selected):
        parser.error("select distinct row names from the frozen plan")
    settings = q.Settings(**plan["settings"])
    q.validate_tolerance_levels(settings.atol_levels, settings.rtol_levels)
    module_path = Path(torchwave.__file__).resolve()
    if not module_path.is_relative_to(q.ROOT / "src"):
        parser.error(f"torchwave loaded outside this checkout: {module_path}")
    torch.set_num_threads(1)
    before = q.source_snapshot()
    before[str(Path(__file__).resolve().relative_to(q.ROOT))] = q.sha256(__file__)
    report = dict(
        schema="diffgw.seobnrv4.source-rkf-domain-screen.v1",
        scope=dict(**q.SCOPE, selected_row_screen=True,
                   numerical_generator_qualified=False, independent_physics=False),
        status="running", release_qualified=False,
        plan_path=str(plan_path.relative_to(q.ROOT)),
        plan_sha256=q.sha256(plan_path),
        prior_source_candidate_sha256=q.sha256(prior_path),
        source_sha256_before=before, source_manifest_sha256=q.canonical_hash(before),
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"],
                                         cwd=q.ROOT, text=True).strip(),
        process=dict(host=socket.gethostname(), cwd=os.getcwd(), pid=os.getpid(),
                     argv=sys.argv, torchwave_module=str(module_path)),
        settings=asdict(settings), physical_choice=expected, numerical_choice=candidate,
        selected_rows=selected, rows={})
    q.write_evidence(args.json_out, report)

    def runner(row, level, runner_settings):
        return q.native_run(
            row, level, runner_settings,
            diagnostic_fine_rkf45_resampling=True,
            diagnostic_local_nqc_derivatives=True,
            diagnostic_matching_second_derivatives=True,
            diagnostic_nqc_phase_window_M=candidate["nqc_phase_window_M"],
            diagnostic_coarse_initial_step_M=candidate["coarse_initial_step_M"],
            diagnostic_warm_start_Momega=candidate["warm_start_Momega"],
            diagnostic_restart_time_M=candidate["fixed_restart_time_M"],
            diagnostic_mode_21_higher_spin=choice["flux"]["mode_21_higher_spin"],
            diagnostic_mode_31_higher_spin=choice["flux"]["mode_31_higher_spin"],
            diagnostic_mode_32_spin_square=choice["flux"]["mode_32_spin_square"],
            diagnostic_mode_33_higher_spin=choice["physical"]["mode_33_higher_spin"],
            diagnostic_mode_43_higher_spin=choice["mode_43_higher_spin"],
            diagnostic_spin_truncation=choice["spin_truncation"],
            diagnostic_hamiltonian_log_derivative=choice["physical"][
                "hamiltonian_log_derivative"],
            diagnostic_hamiltonian_calibration=choice["calibration"])

    for name in selected:
        report["rows"][name] = q.qualify_row(
            rows[name], settings, runner=runner,
            progress=lambda level, name=name: print(f"row={name} run={level}", flush=True))
        q.write_evidence(args.json_out, report)
        checks = report["rows"][name]["checks"]
        print(json.dumps(dict(row=name,
                              numerical_checks_passed=report["rows"][name][
                                  "numerical_checks_passed"],
                              gates={key: value["gates"] for key, value in checks.items()},
                              errors=report["rows"][name]["errors"])), flush=True)
    after = q.source_snapshot()
    after[str(Path(__file__).resolve().relative_to(q.ROOT))] = q.sha256(__file__)
    report["source_sha256_after"] = after
    report["sources_unchanged"] = before == after
    report["all_numerical_checks_passed"] = (report["sources_unchanged"]
        and all(row["numerical_checks_passed"] for row in report["rows"].values()))
    report["status"] = ("diagnostic_numerical_gates_passed" if report[
        "all_numerical_checks_passed"] else "failed")
    q.write_evidence(args.json_out, report)
    print(json.dumps(dict(status=report["status"], evidence=str(args.json_out.resolve()))),
          flush=True)
    return 0 if report["all_numerical_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

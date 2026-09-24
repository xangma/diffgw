"""Check current native NQC input derivatives with an independent integrator.

DOP853 integrates the same provisional source-placement vector field. This is
a numerical derivative diagnostic, not independent physical validation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
import time

import numpy as np
from scipy.integrate import solve_ivp
import torch
import lal

from . import qualify_generator as q
from .diagnose_nqc_gsl_same_input import GSLCubic
from .diagnose_source_nqc_high_rate import LocalPolynomial
from .eob.torch.adaptive_driver_clean import reconstruct_seobnrv4_22_adaptive
from .eob.torch.historical_hamiltonian import HistoricalLogDerivative
from torchwave.approximants.eob.integrator import EOBTrajectoryIntegrator
from torchwave.approximants.eob.nqc import nqc_22_basis
from torchwave.approximants.eob.published_waveform import published_mode22_and_frequency
from torchwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/source_placement_nqc_dense_plan.json"
NAMES = ("amplitude", "A*q1", "A*q2", "A*q3", "mode_phase", "p1", "p1*pr2")


def array(value):
    return np.asarray(value.detach().cpu().numpy())


def series(result):
    nqc = result.attachment.fine.nqc
    return array(nqc.amplitude_spline.knots), np.column_stack((
        array(nqc.sampled_amplitude), array(nqc.sampled_products),
        -array(nqc.sampled_waveform_phase), array(nqc.sampled_phase_basis)))


def dense_quantities(solution, times, field, ham, flux):
    modes, values = [], []
    for coordinate in times:
        state = torch.from_numpy(np.asarray(solution.sol(coordinate)))
        with torch.no_grad():
            omega = field(state)[1]
            mode, _ = published_mode22_and_frequency(
                state, ham, flux, spin_v6=q.CHOICES["spin_v6"],
                include_pan_v8=False, include_positive_chi_540=False,
                retain_pan_cubic_spin=True)
            amplitude_basis, phase_basis = nqc_22_basis(state[0], state[2], omega)
        complex_mode = complex(array(mode))
        amplitude = abs(complex_mode)
        modes.append(complex_mode)
        values.append(np.r_[amplitude, amplitude * array(amplitude_basis),
                            0.0, array(phase_basis)])
    result = np.asarray(values)
    phases = np.angle(np.asarray(modes) * np.conj(modes[len(modes) // 2]))
    if np.max(abs(np.diff(phases))) >= np.pi:
        raise ValueError("dense mode phase crosses local branch")
    result[:, 4] = phases
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    if (plan["schema"] != "diffgw.seobnrv4.source-placement-nqc-dense-plan.v1"
            or plan["row_name"] != "strong_aligned_spin"
            or plan["level"] != "ode_2"
            or plan["five_point_steps_M"] != [0.2, 0.1, 0.05, 0.025]
            or plan["dop853_tolerances"] != [[1e-11, 1e-12], [1e-12, 1e-13]]
            or plan["polynomial_estimators"] != [
                {"degree": 4, "half_width_M": 2.0},
                {"degree": 6, "half_width_M": 0.5}]
            or plan["release_qualified"] is not False):
        parser.error("unexpected plan")
    paths = {key: q.ROOT / item["path"] for key, item in plan["prerequisites"].items()}
    if (set(paths) != {"waveform", "native_array", "source_flux_plan", "prior_waveform"}
            or any(q.sha256(paths[key]) != item["sha256"]
                   for key, item in plan["prerequisites"].items())):
        parser.error("frozen prerequisites changed")
    frozen = json.loads(paths["waveform"].read_text())
    prior = json.loads(paths["prior_waveform"].read_text())
    source_plan = json.loads(paths["source_flux_plan"].read_text())
    gsl_path = Path(lal.__file__).resolve().parent / ".dylibs/libgsl.28.dylib"
    gsl_hash = source_plan["pinned_wheel"]["libraries_sha256"]["libgsl.28.dylib"]
    if q.sha256(gsl_path) != gsl_hash:
        parser.error("GSL binary differs from frozen numerical reference")
    lal_spec = importlib.util.find_spec("lalsimulation")
    if lal_spec is None or lal_spec.origin is None:
        parser.error("lalsimulation package location unavailable")
    extensions = list(Path(lal_spec.origin).resolve().parent.glob("_lalsimulation*.so"))
    if len(extensions) != 1:
        parser.error("expected exactly one lalsimulation Python extension")
    lal_extension = dict(
        path=str(extensions[0]), sha256=q.sha256(extensions[0]),
        frozen_sha256=source_plan["pinned_wheel"]["python_extension_sha256"])
    lal_extension["matches_frozen_pin"] = (
        lal_extension["sha256"] == lal_extension["frozen_sha256"])
    if (frozen["status"] != "diagnostic_complete"
            or frozen["row"]["chi1"] != 0.8
            or frozen["row"]["chi2"] != 0.8
            or prior["candidate_window_M"] != 1.0
            or frozen["choice"]["candidate_changes"] != {
                "spin_truncation": "tabulated", "mode_43_higher_spin": "lal_v4_rho"}):
        parser.error("unexpected frozen waveform")
    torch.set_num_threads(1)
    row = frozen["row"]
    choice = frozen["choice"]
    level = frozen["candidate"][plan["level"]]["level"]
    settings = q.Settings(**frozen["settings"])
    numerical, physical = choice["numerical"], choice["physical"]
    placement = choice["baseline_flux_placements"]
    kwargs = dict(
        diagnostic_hermite_resampling=numerical["hermite_resampling"],
        diagnostic_local_nqc_derivatives=numerical["local_nqc_phase_derivatives"],
        diagnostic_matching_second_derivatives=True,
        diagnostic_restart_time_M=numerical["fixed_restart_time_M"],
        diagnostic_coarse_initial_step_M=0.125,
        diagnostic_warm_start_Momega=0.03162277660168379,
        diagnostic_nqc_phase_window_M=prior["candidate_window_M"],
        diagnostic_mode_33_higher_spin=physical["mode_33_higher_spin"],
        diagnostic_hamiltonian_log_derivative=physical["hamiltonian_log_derivative"],
        diagnostic_mode_21_higher_spin=placement["mode_21_higher_spin"],
        diagnostic_mode_31_higher_spin=placement["mode_31_higher_spin"],
        diagnostic_mode_32_spin_square=placement["mode_32_spin_square"],
        diagnostic_mode_43_higher_spin="lal_v4_rho",
        spin_truncation="tabulated",
    )
    source_names = (
        "dev/native_rom/eob/torch/adaptive_driver_clean.py",
        "dev/native_rom/eob/torch/historical_hamiltonian.py",
        "src/torchwave/approximants/eob/seobnrv4_flux.py",
        "src/torchwave/approximants/eob/published_waveform.py",
        "src/torchwave/approximants/eob/nqc.py",
        "src/torchwave/approximants/eob/integrator.py",
    )
    sources = {name: q.sha256(q.ROOT / name) for name in source_names}
    report = dict(schema="diffgw.seobnrv4.source-placement-nqc-dense.v1",
                  scope=dict(**q.SCOPE, development_only=True,
                             independent_integrator=True, independent_physics=False),
                  release_qualified=False, status="running",
                  plan_sha256=q.sha256(PLAN), script_sha256=q.sha256(__file__),
                  gsl_binary=dict(path=str(gsl_path), sha256=gsl_hash),
                  installed_lal_extension=lal_extension,
                  sources_before=sources, prerequisites=plan["prerequisites"],
                  row=row, level=level, names=NAMES, native={}, dop853=[],
                  comparison={}, errors=[], process=dict(host=socket.gethostname(),
                  cwd=os.getcwd(), pid=os.getpid(), argv=sys.argv))
    q.write_evidence(args.json_out, report)
    try:
        started = time.monotonic()
        with q.bounded_run(settings.max_run_seconds), torch.no_grad():
            result = reconstruct_seobnrv4_22_adaptive(
                torch.tensor(row["omega_start"], dtype=torch.float64),
                **{key: row[key] for key in ("mass1", "mass2", "chi1", "chi2")},
                **q.CHOICES | {"spin_truncation": "tabulated"},
                **{key: level[key] for key in ("atol", "rtol", "dt_geom")},
                **{key: value for key, value in kwargs.items() if key != "spin_truncation"},
                ringdown_duration=settings.ringdown_duration_M,
                num_steps=settings.max_accepted_steps,
                max_accepted_steps=settings.max_accepted_steps,
                max_rejected_steps=settings.max_rejected_steps,
                max_evaluations=settings.max_evaluations)
        mode_hash = hashlib.sha256(np.ascontiguousarray(array(result.mode)).tobytes()).hexdigest()
        expected_hash = frozen["candidate"][plan["level"]]["native"]["array_sha256"]["mode"]
        if mode_hash != expected_hash:
            raise ValueError("native mode fails frozen waveform replay")
        axis, values = series(result)
        with np.load(paths["native_array"]) as asset:
            if (not np.array_equal(axis, asset["time"])
                    or not np.array_equal(values[:, 0], asset["values"][0])
                    or abs(float(result.attachment.fine.match_time)-float(asset["match"])) > 1e-12):
                raise ValueError("native NQC series fails frozen input replay")
        match = float(result.attachment.fine.match_time)
        gsl = GSLCubic(gsl_path)
        estimators = {"gsl_cubic": gsl}
        for item in plan["polynomial_estimators"]:
            estimators[f"poly{item['degree']}_halfwidth_{item['half_width_M']:g}"] = (
                LocalPolynomial(item["half_width_M"], item["degree"]))
        native = {name: np.asarray([est.derivatives(axis, values[:, j], match)
                                    for j in range(len(NAMES))])
                  for name, est in estimators.items()}
        report["native"] = dict(mode_sha256=mode_hash, match_time_local_M=match,
                                replay_seconds=time.monotonic()-started,
                                derivatives={key: value.tolist() for key, value in native.items()})
        fine = result.trajectory.fine
        fine_axis, fine_states = array(fine.output_time), array(fine.output_state)
        anchor_index = int(np.argmin(abs(fine_axis-(match-plan["anchor_before_match_M"]))))
        t0 = float(fine_axis[anchor_index])
        initial = fine_states[anchor_index]
        report["native"].update(anchor_time_M=t0, anchor_state=initial.tolist())
        q.write_evidence(args.json_out, report)
        m1, m2 = (torch.tensor(row[key], dtype=torch.float64) for key in ("mass1", "mass2"))
        x1, x2 = m1/(m1+m2), m2/(m1+m2)
        ham = HistoricalLogDerivative(x1*x2, torch.tensor(row["chi1"], dtype=torch.float64),
                                      torch.tensor(row["chi2"], dtype=torch.float64), delta=x1-x2)
        flux = SEOBNRv4PublishedFlux(
            spin_truncation="tabulated", retain_pan_cubic_spin=True,
            mode_21_higher_spin=placement["mode_21_higher_spin"],
            mode_31_higher_spin=placement["mode_31_higher_spin"],
            mode_32_spin_square=placement["mode_32_spin_square"],
            mode_33_higher_spin=physical["mode_33_higher_spin"],
            mode_43_higher_spin="lal_v4_rho")
        field = EOBTrajectoryIntegrator(ham, flux).vector_field

        def scipy_field(_t, state):
            with torch.no_grad():
                return array(field(torch.from_numpy(state)))

        for rtol, atol in plan["dop853_tolerances"]:
            started = time.monotonic()
            solution = solve_ivp(scipy_field, (t0, match+plan["end_after_match_M"]),
                                 initial, method="DOP853", rtol=rtol, atol=atol,
                                 dense_output=True)
            if not solution.success:
                raise RuntimeError(solution.message)
            stencils = {}
            for step in plan["five_point_steps_M"]:
                times = match+np.arange(-2, 3)*step
                sample = dense_quantities(solution, times, field, ham, flux)
                first = (sample[0]-8*sample[1]+8*sample[3]-sample[4])/(12*step)
                second = (-sample[0]+16*sample[1]-30*sample[2]
                          +16*sample[3]-sample[4])/(12*step**2)
                stencils[f"{step:g}"] = dict(first=first.tolist(), second=second.tolist(),
                                             center=sample[2].tolist())
            report["dop853"].append(dict(rtol=rtol, atol=atol, nfev=solution.nfev,
                                         seconds=time.monotonic()-started,
                                         stencils=stencils))
            q.write_evidence(args.json_out, report)
            print(json.dumps(dict(dop_rtol=rtol, seconds=report["dop853"][-1]["seconds"],
                                  nfev=solution.nfev)), flush=True)
        reference = report["dop853"][-1]["stencils"]["0.025"]
        for name, derivative in native.items():
            report["comparison"][name] = dict(
                first_error=(derivative[:, 1]-reference["first"]).tolist(),
                second_error=(derivative[:, 2]-reference["second"]).tolist(),
                max_nonphase_second_abs_error=float(np.max(abs(
                    derivative[np.arange(len(NAMES)) != 4, 2]
                    -np.asarray(reference["second"])[np.arange(len(NAMES)) != 4]))))
        earlier = report["dop853"][0]["stencils"]["0.025"]
        coarser = report["dop853"][-1]["stencils"]["0.05"]
        report["comparison"]["reference"] = dict(
            tolerance_first_spread=(np.asarray(earlier["first"])-reference["first"]).tolist(),
            tolerance_second_spread=(np.asarray(earlier["second"])-reference["second"]).tolist(),
            step_first_spread=(np.asarray(coarser["first"])-reference["first"]).tolist(),
            step_second_spread=(np.asarray(coarser["second"])-reference["second"]).tolist())
        report["sources_after"] = {name: q.sha256(q.ROOT / name) for name in sources}
        if report["sources_after"] != sources:
            raise ValueError("generator source changed during diagnostic")
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(dict(type=type(exc).__name__, message=str(exc)))
    q.write_evidence(args.json_out, report)
    print(json.dumps(dict(status=report["status"], errors=report["errors"])), flush=True)
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

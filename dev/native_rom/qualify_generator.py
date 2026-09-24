"""Development-only numerical checks for the experimental SEOBNR generator.

Run with ``PYTHONPATH=src python -m dev.native_rom.qualify_generator --help``.
This runner emits evidence, never ROM training data or a production asset.
Convergence and optional unaligned LAL comparisons do not validate the physics.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import signal
import socket
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CHOICES = dict(spin_truncation="generic-leading-ell8", spin_v6="bohe_total_energy",
               include_pan_v8=False, include_positive_chi_540=False,
               final_spin_fit_order=(3, 4))
SCOPE = {
    "physics_validated": False,
    "production_asset": False,
    "production_promotion_allowed": False,
    "training_generator_qualified": False,
    "description": "Scalar CPU float64 numerical stability evidence only; no training or promotion.",
    "known_limitations": [
        "Convergence cannot establish correctness of the equations or published model fidelity.",
        "The adaptive prototype uses a diagnostic QNM spline with independently generated values "
        "and black-box-inferred knot metadata; this is not wholly literature-specified provenance.",
        "Changing dt also changes the adaptive initial step and coarse/fine restart sampling; "
        "the grid check measures the complete generator's grid sensitivity.",
        "No time, phase, amplitude, or coefficient fitting is performed.",
        "No derivatives, batched execution, accelerators, or JAX are qualified here.",
    ],
}
EVENTS = ("peak_time_M", "match_time_M", "boundary_time_M")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(",", ":")).encode()).hexdigest()


def source_snapshot():
    """Hash sources and numerical inputs, excluding generated package metadata."""
    paths = list((ROOT / "src").rglob("*"))
    paths += list((ROOT / "dev/native_rom/eob").rglob("*.py"))
    paths += [Path(__file__), ROOT / "scripts/verify_eob_published_waveforms.py"]
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(set(paths))
            if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
            and not any(part.endswith(".egg-info") for part in p.relative_to(ROOT).parts[:-1])}


@dataclass(frozen=True)
class Settings:
    atol_levels: tuple = (1e-9, 1e-10, 1e-11)
    rtol_levels: tuple = (1e-8, 1e-9, 1e-10)
    dt_levels_M: tuple = (0.5, 0.25, 0.125)
    ringdown_duration_M: float = 150.0
    max_accepted_steps: int = 20000
    max_rejected_steps: int = 10000
    max_evaluations: int = 200000
    max_run_seconds: float = 600.0
    relative_l2_limit: float = 1e-6
    event_time_limit_M: float = 0.05


def validate_rows(rows):
    names = {"mass1", "mass2", "chi1", "chi2", "omega_start"}
    if not isinstance(rows, list) or not rows:
        raise ValueError("inputs must be a nonempty JSON list of scalar rows")
    for row in rows:
        if not isinstance(row, dict) or set(row) != names:
            raise ValueError(f"each row must contain exactly {sorted(names)}")
        if any(isinstance(x, bool) or not isinstance(x, (int, float))
               or not math.isfinite(x) for x in row.values()):
            raise ValueError("all row inputs must be finite scalar numbers")
        if min(row["mass1"], row["mass2"], row["omega_start"]) <= 0:
            raise ValueError("masses and omega_start must be positive")
        if max(abs(row["chi1"]), abs(row["chi2"])) > 1:
            raise ValueError("dimensionless spins must lie in [-1, 1]")
    return rows


def run_plan(settings):
    """Three tolerance levels and three grids, sharing the tight/coarse solve."""
    return [dict(name=f"ode_{i}", atol=a, rtol=r, dt_geom=settings.dt_levels_M[0])
            for i, (a, r) in enumerate(zip(settings.atol_levels, settings.rtol_levels))] + [
        dict(name=f"grid_{i}", atol=settings.atol_levels[-1],
             rtol=settings.rtol_levels[-1], dt_geom=dt)
        for i, dt in enumerate(settings.dt_levels_M[1:], start=1)]


def validate_tolerance_levels(atol_levels, rtol_levels):
    """Require three strictly tighter positive tolerance pairs."""
    if (len(atol_levels) != 3 or len(rtol_levels) != 3
            or any(not math.isfinite(x) or x <= 0 for x in (*atol_levels, *rtol_levels))
            or any(levels[i] <= levels[i+1] for levels in (atol_levels, rtol_levels)
                   for i in range(2))):
        raise ValueError("expected three finite positive tolerances, strictly decreasing")
    return tuple(atol_levels), tuple(rtol_levels)


def mode_metrics(candidate, reference):
    """Direct complex and amplitude errors; no fitted or maximized alignment."""
    a, b = np.asarray(candidate), np.asarray(reference)
    if a.ndim != 1 or a.shape != b.shape or a.size < 2:
        raise ValueError("metrics need matching nonempty one-dimensional signals")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("nonfinite waveform")
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if min(na, nb) <= 0:
        raise ValueError("zero-norm waveform")
    # Wrapped phase is meaningful only where both signals have support.
    active = (abs(a) > max(abs(a)) * 1e-8) & (abs(b) > max(abs(b)) * 1e-8)
    if not active.any():
        raise ValueError("no common nonzero waveform support")
    phase = np.angle(a[active] * b[active].conj())
    result = dict(samples=int(a.size), phase_samples=int(active.sum()),
                  complex_relative_l2=float(np.linalg.norm(a-b) / nb),
                  amplitude_relative_l2=float(np.linalg.norm(abs(a)-abs(b)) / nb),
                  relative_norm_error=float(abs(na/nb-1)),
                  phase_rms_rad=float(np.sqrt(np.mean(phase**2))),
                  phase_max_abs_rad=float(max(abs(phase))),
                  unmaximized_real_overlap=float(np.vdot(a, b).real / (na*nb)))
    if not all(math.isfinite(v) for v in result.values()):
        raise ValueError("nonfinite comparison metric")
    return result


def compare_runs(candidate, reference):
    """Compare exact nested samples on the union support, zero-filling tails."""
    dt = max(candidate["dt_geom"], reference["dt_geom"])
    signals = []
    for run in (candidate, reference):
        ratio = dt / run["dt_geom"]
        if not math.isclose(ratio, round(ratio), rel_tol=0, abs_tol=1e-12):
            raise ValueError("convergence grids must be nested")
        t, mode = run["time"], run["mode"]
        if (t.ndim != 1 or mode.shape != t.shape or t.size < 2
                or not np.isfinite(t).all() or not np.isfinite(mode).all()
                or not np.allclose(t, np.arange(t.size)*run["dt_geom"], rtol=0, atol=1e-9)):
            raise ValueError("expected finite uniform scalar waveform starting at t=0")
        signals.append(mode[::round(ratio)])
    count = max(map(len, signals))
    metrics = mode_metrics(*(np.pad(x, (0, count-len(x))) for x in signals))
    metrics.update(grid_step_M=dt, common_samples=min(map(len, signals)),
                   candidate_samples=len(signals[0]), reference_samples=len(signals[1]),
                   support_end_difference_M=float(candidate["time"][-1]-reference["time"][-1]))
    for event in EVENTS:
        metrics[event + "_abs_difference"] = abs(candidate[event]-reference[event])
    if not all(math.isfinite(v) for v in metrics.values()):
        raise ValueError("nonfinite event or support metric")
    return metrics


def convergence_check(coarse, medium, fine, settings):
    first, last = compare_runs(coarse, medium), compare_runs(medium, fine)
    # Success is a predeclared stability budget, not an order estimate or proof
    # of convergence. Refined errors must decrease, up to float64 roundoff.
    gates = {"complex_error_budget": last["complex_relative_l2"] <= settings.relative_l2_limit,
             "amplitude_error_budget": last["amplitude_relative_l2"] <= settings.relative_l2_limit,
             "error_decreased": last["complex_relative_l2"] <= first["complex_relative_l2"] + 1e-12}
    gates.update({event: last[event + "_abs_difference"] <= settings.event_time_limit_M
                  for event in EVENTS})
    return dict(coarse_to_medium=first, medium_to_fine=last, gates=gates,
                numerical_check_passed=all(gates.values()))


@contextmanager
def bounded_run(seconds):
    def timeout(*_):
        raise TimeoutError(f"generator exceeded {seconds:g} seconds")
    previous = signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def native_run(row, level, settings, *, diagnostic_hermite_resampling=False,
               diagnostic_rkf45_resampling=False,
               diagnostic_fine_rkf45_resampling=False,
               diagnostic_spin_truncation=None,
               diagnostic_mode_21_higher_spin="taylor_f",
               diagnostic_mode_31_higher_spin="taylor_f",
               diagnostic_mode_32_spin_square="pan_f26",
               diagnostic_mode_33_higher_spin="taylor_f",
               diagnostic_mode_43_higher_spin="taylor_f",
               diagnostic_hamiltonian_log_derivative="analytic",
               diagnostic_hamiltonian_calibration="published",
               diagnostic_qnm_source="historical_spline",
               diagnostic_local_nqc_derivatives=False,
               diagnostic_nqc_phase_window_M=0.5,
               diagnostic_matching_second_derivatives=False,
               diagnostic_restart_time_M=None,
               diagnostic_fine_max_step_M=None,
               diagnostic_coarse_initial_step_M=None,
               diagnostic_warm_start_Momega=None,
               return_nqc_coefficients=False,
               return_nqc_samples=False,
               return_boundary_data=False,
               return_accepted_hashes=False):
    import torch
    from dev.native_rom.eob.torch.adaptive_driver_clean import reconstruct_seobnrv4_22_adaptive
    choices = dict(CHOICES)
    if diagnostic_spin_truncation is not None:
        choices["spin_truncation"] = diagnostic_spin_truncation
    with bounded_run(settings.max_run_seconds), torch.no_grad():
        result = reconstruct_seobnrv4_22_adaptive(
            torch.tensor(row["omega_start"], dtype=torch.float64),
            **{k: row[k] for k in ("mass1", "mass2", "chi1", "chi2")},
            **choices, **{k: level[k] for k in ("atol", "rtol", "dt_geom")},
            diagnostic_hermite_resampling=diagnostic_hermite_resampling,
            diagnostic_rkf45_resampling=diagnostic_rkf45_resampling,
            diagnostic_fine_rkf45_resampling=diagnostic_fine_rkf45_resampling,
            diagnostic_mode_21_higher_spin=diagnostic_mode_21_higher_spin,
            diagnostic_mode_31_higher_spin=diagnostic_mode_31_higher_spin,
            diagnostic_mode_32_spin_square=diagnostic_mode_32_spin_square,
            diagnostic_mode_33_higher_spin=diagnostic_mode_33_higher_spin,
            diagnostic_mode_43_higher_spin=diagnostic_mode_43_higher_spin,
            diagnostic_hamiltonian_log_derivative=diagnostic_hamiltonian_log_derivative,
            diagnostic_hamiltonian_calibration=diagnostic_hamiltonian_calibration,
            diagnostic_qnm_source=diagnostic_qnm_source,
            diagnostic_local_nqc_derivatives=diagnostic_local_nqc_derivatives,
            diagnostic_nqc_phase_window_M=diagnostic_nqc_phase_window_M,
            diagnostic_matching_second_derivatives=diagnostic_matching_second_derivatives,
            diagnostic_restart_time_M=diagnostic_restart_time_M,
            diagnostic_fine_max_step_M=diagnostic_fine_max_step_M,
            diagnostic_coarse_initial_step_M=diagnostic_coarse_initial_step_M,
            diagnostic_warm_start_Momega=diagnostic_warm_start_Momega,
            ringdown_duration=settings.ringdown_duration_M,
            num_steps=settings.max_accepted_steps, max_accepted_steps=settings.max_accepted_steps,
            max_evaluations=settings.max_evaluations, max_rejected_steps=settings.max_rejected_steps)
    def array(value):
        data = value.detach().cpu().numpy()
        if not np.isfinite(data).all():
            raise ValueError("generator returned nonfinite values")
        return data
    stages = {}
    for name in ("coarse", "fine"):
        stage = getattr(result.trajectory, name)
        for value in (stage.time, stage.state, stage.derivative,
                      stage.output_time, stage.output_state):
            array(value)
        stats = stage.statistics
        if stats.termination != ("event",):
            raise ValueError(f"{name} stage did not reach its physical event: {stats.termination}")
        stages[name] = dict(accepted=stats.total_accepted, rejected=stats.total_rejected,
                            evaluations=stats.total_evaluations,
                            termination=list(stats.termination), event_time_M=float(array(stats.event_time)))
    output = dict(**level, time=array(result.time), mode=array(result.mode), stages=stages,
                  peak_time_M=float(array(result.trajectory.peak.time)),
                  restart_time_M=float(array(result.trajectory.restart_time).reshape(-1)[0]),
                  match_time_M=float(array(result.attachment.match_time)),
                  boundary_time_M=float(array(result.attachment.boundary_time)),
                  final_mass_fraction=float(array(result.final_mass_fraction)),
                  final_spin=float(array(result.final_spin)),
                  internal_step_M=float(array(result.attachment.internal_step)))
    if return_accepted_hashes:
        output["accepted_array_sha256"] = {
            name: {key: hashlib.sha256(np.ascontiguousarray(
                array(getattr(getattr(result.trajectory, name), key))).tobytes()).hexdigest()
                   for key in ("time", "state", "derivative", "valid")}
            for name in ("coarse", "fine")}
    if return_boundary_data:
        if diagnostic_qnm_source == "historical_spline":
            from torchwave.approximants.kerr_qnm_table_convention import signed_qnm_220_table_convention as qnm_220
        elif diagnostic_qnm_source == "independent_kerr":
            from torchwave.approximants.kerr_qnm_generated import signed_qnm_220 as qnm_220
        else:
            raise ValueError("unsupported diagnostic_qnm_source")
        qnm_w, qnm_damp = qnm_220(result.final_spin)
        output["boundary_data"] = dict(
            index=int(array(result.attachment.boundary_index)),
            amplitude=float(array(result.attachment.boundary_amplitude)),
            phase=float(array(result.attachment.boundary_phase)),
            frequency=float(array(result.attachment.boundary_frequency)),
            qnm_frequency=float(array(qnm_w / result.final_mass_fraction)),
            qnm_damping=float(array(qnm_damp / result.final_mass_fraction)),
        )
    if return_nqc_coefficients:
        from torchwave.approximants.eob.numerical_attachment import _eval_batch_scalar
        fine_nqc = result.attachment.fine
        coefficients = fine_nqc.calibration.coefficients
        output["nqc_coefficients"] = dict(
            amplitude=array(coefficients.amplitude).tolist(),
            phase=array(coefficients.phase).tolist(),
        )
        output["nqc_matching_matrices"] = dict(
            amplitude=array(fine_nqc.nqc.amplitude_matching_matrix).tolist(),
            phase=array(fine_nqc.nqc.phase_matching_matrix).tolist(),
        )
        spline = fine_nqc.nqc.phase_basis_spline
        output["nqc_full_spline_phase_matrix"] = array(torch.stack((
            _eval_batch_scalar(spline, fine_nqc.match_time, derivative=1),
            _eval_batch_scalar(spline, fine_nqc.match_time, derivative=2),
        ))).tolist()
    if return_nqc_samples:
        nqc = result.attachment.fine.nqc
        output["nqc_samples"] = dict(
            time=array(nqc.amplitude_spline.knots),
            amplitude=array(nqc.sampled_amplitude),
            products=array(nqc.sampled_products),
            phase_basis=array(nqc.sampled_phase_basis),
            waveform_phase=array(nqc.sampled_waveform_phase),
            match_time=float(array(result.attachment.fine.match_time)),
        )
    return output


def lal_binary_identity(expected_lal, expected_lalsimulation):
    import lal
    import lalsimulation as lalsim
    from lal import _lal
    from lalsimulation import _lalsimulation
    if (lal.__version__, lalsim.__version__) != (expected_lal, expected_lalsimulation):
        raise ValueError(f"LAL version pin mismatch: installed {lal.__version__}, {lalsim.__version__}")
    distribution = importlib.metadata.distribution("lalsuite")
    # Include the linked numerical libraries, not only their Python wrappers.
    binaries = {Path(m.__file__).resolve() for m in (_lal, _lalsimulation)}
    binaries.update(Path(distribution.locate_file(p)).resolve() for p in distribution.files or []
                    if ".so" in p.name or p.suffix == ".dylib")
    return dict(lal_version=lal.__version__, lalsimulation_version=lalsim.__version__,
                lalsuite_distribution=distribution.version,
                binaries={str(path): sha256(path) for path in sorted(binaries)})


def compare_lal(row, run):
    """Use only the public binary API, retaining its returned epoch and phase."""
    import lal
    import lalsimulation as lalsim
    from scripts.verify_eob_published_waveforms import _invert_22
    from torchwave.constants import C_SI, MPC_SI, MSUN_SEC
    mass_seconds = (row["mass1"] + row["mass2"]) * float(MSUN_SEC)
    dt = run["dt_geom"] * mass_seconds
    lower = row["omega_start"] / (math.pi * mass_seconds)
    distance, inclination = 100.0, 0.8
    hp, hc = lalsim.SimInspiralChooseTDWaveform(
        row["mass1"]*lal.MSUN_SI, row["mass2"]*lal.MSUN_SI,
        0., 0., row["chi1"], 0., 0., row["chi2"],
        distance*1e6*lal.PC_SI, inclination, 0., 0., 0., 0.,
        dt, lower, 0., lal.CreateDict(), lalsim.GetApproximantFromString("SEOBNRv4"))
    ref_time = float(hp.epoch) + np.arange(len(hp.data.data)) * float(hp.deltaT)
    if len(hp.data.data) != len(hc.data.data) or hp.epoch != hc.epoch or hp.deltaT != hc.deltaT:
        raise ValueError("inconsistent LAL polarization grids")
    reference = _invert_22(np.asarray(hp.data.data), np.asarray(hc.data.data), inclination)
    if not np.isfinite(ref_time).all() or not np.isfinite(reference).all():
        raise ValueError("nonfinite LAL output")
    native_time = (run["time"]-run["peak_time_M"])*mass_seconds
    native_mode = run["mode"] * mass_seconds * float(C_SI)/(distance*float(MPC_SI))
    sampled = np.interp(ref_time, native_time, native_mode.real, left=0., right=0.) + 1j*np.interp(
        ref_time, native_time, native_mode.imag, left=0., right=0.)
    return dict(api="SimInspiralChooseTDWaveform", approximant="SEOBNRv4",
                delta_t_s=dt, f_lower_Hz=lower, f_ref_Hz=0., phase=0., distance_Mpc=distance,
                inclination_rad=inclination, metrics=mode_metrics(sampled, reference),
                native_support_s=[float(native_time[0]), float(native_time[-1])],
                reference_support_s=[float(ref_time[0]), float(ref_time[-1])],
                comparison_pass_claim=False,
                convention="Native orbital-frequency peak at t=0; LAL epoch unchanged. Same numeric "
                "phase zero does not establish identical phase semantics. Linear interpolation onto "
                "full LAL support, native tails zero-filled. No fitted alignment.")


def summarize_run(run):
    return {**{k: v for k, v in run.items() if k not in ("time", "mode")},
            "samples": len(run["time"]), "support_M": [float(run["time"][0]), float(run["time"][-1])],
            "array_sha256": {k: hashlib.sha256(np.ascontiguousarray(run[k]).tobytes()).hexdigest()
                             for k in ("time", "mode")},
            "array_dtype": {k: str(run[k].dtype) for k in ("time", "mode")}}


def qualify_row(row, settings, runner=native_run, oracle=None, progress=lambda *_: None):
    output = dict(inputs=row, runs=[], checks={}, errors=[])
    runs = {}
    for level in run_plan(settings):
        progress(level["name"])
        started = time.monotonic()
        try:
            run = runner(row, level, settings)
            # Self-comparison validates the complete support even on a failed
            # campaign with no adjacent successful run available.
            compare_runs(run, run)
            run["elapsed_seconds"] = time.monotonic()-started
            runs[level["name"]] = run
            output["runs"].append(summarize_run(run))
        except Exception as exc:
            output["errors"].append(dict(run=level["name"], type=type(exc).__name__, message=str(exc),
                                         elapsed_seconds=time.monotonic()-started))
            break  # Never hide a failed or nonfinite solve by continuing training.
    if len(runs) == 5:
        for name, keys in (("ode_tolerance", ("ode_0", "ode_1", "ode_2")),
                           ("output_grid", ("ode_2", "grid_1", "grid_2"))):
            try:
                output["checks"][name] = convergence_check(*(runs[k] for k in keys), settings)
            except Exception as exc:
                output["errors"].append(dict(check=name, type=type(exc).__name__, message=str(exc)))
        if oracle is not None:
            try:
                with bounded_run(settings.max_run_seconds):
                    output["lal_diagnostic"] = oracle(row, runs["grid_2"])
            except Exception as exc:
                output["errors"].append(dict(check="lal", type=type(exc).__name__, message=str(exc)))
    output["numerical_checks_passed"] = (not output["errors"] and len(output["checks"]) == 2
        and all(check["numerical_check_passed"] for check in output["checks"].values()))
    return output


def write_evidence(path, evidence):
    path = Path(path).resolve()
    if path.suffix != ".json" or path.is_relative_to(ROOT / "src"):
        raise ValueError("only diagnostic .json evidence outside src is allowed; no production asset export")
    if any(evidence.get("scope", {}).get(flag) is not False for flag in (
            "physics_validated", "production_asset", "production_promotion_allowed",
            "training_generator_qualified")):
        raise ValueError("diagnostic evidence must explicitly refuse physics, training, and production qualification")
    payload = json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False)+"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(payload)
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, help="JSON list of scalar physical-input rows")
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--max-run-seconds", type=float, default=600.)
    parser.add_argument("--max-evaluations", type=int, default=200000)
    parser.add_argument("--atol-levels", type=float, nargs=3, default=Settings.atol_levels)
    parser.add_argument("--rtol-levels", type=float, nargs=3, default=Settings.rtol_levels)
    parser.add_argument("--compare-lal", action="store_true")
    parser.add_argument("--lal-version")
    parser.add_argument("--lalsimulation-version")
    parser.add_argument("--print-plan", action="store_true")
    args = parser.parse_args(argv)
    if not math.isfinite(args.max_run_seconds) or args.max_run_seconds <= 0 or args.max_evaluations <= 0:
        parser.error("resource bounds must be finite and positive")
    try:
        atol_levels, rtol_levels = validate_tolerance_levels(args.atol_levels, args.rtol_levels)
    except ValueError as exc:
        parser.error(str(exc))
    if ((args.compare_lal and not (args.lal_version and args.lalsimulation_version)) or
            (not args.compare_lal and (args.lal_version or args.lalsimulation_version))):
        parser.error("--compare-lal requires exact --lal-version and --lalsimulation-version pins")
    rows = validate_rows(json.loads(args.inputs.read_text()) if args.inputs else [
        dict(mass1=30., mass2=20., chi1=0., chi2=0., omega_start=.04)])
    settings = Settings(atol_levels=atol_levels, rtol_levels=rtol_levels,
                        max_run_seconds=args.max_run_seconds, max_evaluations=args.max_evaluations)
    config = dict(rows=rows, settings=asdict(settings), scientific_choices=CHOICES,
                  dtype="float64", device="cpu", execution="one scalar row per solve",
                  reference_pins=dict(lal=args.lal_version, lalsimulation=args.lalsimulation_version))
    if args.print_plan:
        print(json.dumps(dict(config=config, runs=run_plan(settings), scope=SCOPE), indent=2))
        return 0
    import torch
    torch.set_num_threads(1)
    evidence = dict(schema="diffgw.native-generator-numerical-evidence.v1", scope=SCOPE,
                    config=config, config_sha256=canonical_hash(config), input_sha256=canonical_hash(rows),
                    input_file_sha256=sha256(args.inputs) if args.inputs else None,
                    status="running", results=[], errors=[],
                    process=dict(host=socket.gethostname(), cwd=os.getcwd(), pid=os.getpid(), argv=sys.argv),
                    versions=dict(python=sys.version, platform=platform.platform(), numpy=np.__version__,
                                  torch=torch.__version__, torch_threads=torch.get_num_threads()),
                    source_sha256_before=source_snapshot())
    evidence["source_manifest_sha256"] = canonical_hash(evidence["source_sha256_before"])
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    evidence["git_head"] = git.stdout.strip() if git.returncode == 0 else None
    write_evidence(args.json_out, evidence)
    try:
        if args.compare_lal:
            evidence["lal_binary"] = lal_binary_identity(args.lal_version, args.lalsimulation_version)
        for index, row in enumerate(rows):
            result = qualify_row(row, settings, oracle=compare_lal if args.compare_lal else None,
                                 progress=lambda name: print(f"row={index} run={name}", flush=True))
            evidence["results"].append(result)
            write_evidence(args.json_out, evidence)
    except (Exception, KeyboardInterrupt) as exc:
        evidence["errors"].append(dict(type=type(exc).__name__, message=str(exc)))
    evidence["source_sha256_after"] = source_snapshot()
    evidence["sources_unchanged"] = evidence["source_sha256_before"] == evidence["source_sha256_after"]
    passed = (not evidence["errors"] and evidence["sources_unchanged"] and len(evidence["results"]) == len(rows)
              and all(row["numerical_checks_passed"] for row in evidence["results"]))
    evidence["status"] = "numerical_checks_passed" if passed else "failed"
    write_evidence(args.json_out, evidence)
    print(json.dumps(dict(status=evidence["status"], evidence=str(args.json_out.resolve())), sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

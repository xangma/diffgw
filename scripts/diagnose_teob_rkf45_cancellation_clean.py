#!/usr/bin/env python3
"""Bounded clean-room diagnostic for cancellation-resistant Fehlberg arithmetic.

The production clean adaptive driver is left untouched.  This script replaces
only its module-local RKF45 trial function with an independently reconstructed
NASA TR R-315 tableau.  The fifth-order candidate is formed as
``y + h sum(b5*k)`` and the embedded error directly as
``h sum((b5-b4)*k)``.  The resulting LO-horizon TEOB continuation is assembled
with the existing TEOB carrier, frozen-mobility NQC branch, and clean
ringdown, then compared on the supplied public uniform grid.  No alignment,
fitting, or runtime routing is performed.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import shlex
import socket
import sys
import time as wall_time
from typing import Any

import numpy as np
import torch

from torchwave.approximants.eob import adaptive_driver_clean as adaptive_driver
from torchwave.approximants.eob.teob_adaptive_continuation_clean import (
    DEFAULT_ATOL,
    DEFAULT_CONTINUATION_DURATION,
    DEFAULT_FIXED_STEP_CAP,
    DEFAULT_MAX_ACCEPTED_STEPS,
    DEFAULT_MAX_DURATION,
    DEFAULT_MAX_EVALUATIONS,
    DEFAULT_MAX_REJECTED_STEPS,
    DEFAULT_RTOL,
    continue_teob_adaptive,
)
from torchwave.approximants.eob.teob_merger_nqc_clean import (
    PUBLISHED_2017_NQC_PROVENANCE,
    assemble_teob_merger_nqc,
    phase_from_mode,
    published_2017_nqc_targets,
)
from torchwave.approximants.eob.teob_postadiabatic_clean import alternating8pa
from torchwave.approximants.eob.teob_radiation_clean import (
    TEOBRadiationClean,
    non_keplerian_radius,
)
from torchwave.approximants.eob.teob_ringdown_assembly_clean import assemble_teob_ringdown_22
from torchwave.approximants.eob.teob_trajectory import conservative_trajectory_quantities
from torchwave.approximants.eob.teob_uib2016_remnant import uib2016_nonspinning_remnant
from torchwave.approximants.eob.teob_waveform_state_clean import teob_waveform_state_mode
from torchwave.approximants.teobresums import TEOBResumSHamiltonian

# The paired diagnostic is an earlier owned clean helper.  Importing its
# loaders/independent cubic evaluator avoids changing any shared runtime file.
from diagnose_teob_paired_driver_clean import (  # type: ignore
    DELTA_VALUE,
    DISTANCE_MPC,
    DR,
    EXPECTED_PUBLIC_SHA256,
    MASS1_MSUN,
    MASS2_MSUN,
    NU_VALUE,
    OUTPUT_STEP,
    PA_ORDER,
    PUBLIC_RADIUS,
    R_MIN,
    _LeadingOrderHorizonReaction,
    _assemble,
    _complex_metrics,
    _float,
    _load_public,
    _natural_cubic_numpy,
    _nqc_summary,
    _numpy,
    _phase_metrics,
    _sha256,
    _vector,
    _public_grid_branch,
)


ROOT = Path(__file__).resolve().parents[1]
VERIFY = ROOT / "docs/reviews/2026-09-13-parity-review/verification/teob-rkf45-cancellation-clean"
DEFAULT_PUBLIC = Path("/tmp/torchwave-public-teob-q1p5-r20-dt0p5M.npz")
DEFAULT_PUBLIC_META = DEFAULT_PUBLIC.with_suffix(".json")
DEFAULT_INPUT_CACHE = ROOT / "verification/teob-paired-driver-clean/paired-physical-output-lo-horizon.npz"
DEFAULT_BRANCH_CACHE = ROOT / "docs/reviews/2026-09-13-parity-review/verification/teob-radial-acceleration-paired-clean/paired-radial-acceleration-output.npz"
DEFAULT_OUTPUT = VERIFY / "diagnostic.json"
DEFAULT_CACHE = VERIFY / "candidate-output.npz"
DEFAULT_LOG = VERIFY / "probe.log"
PUBLIC_EPOCH_INITIAL_MASS = -12316.763885040122


def _json_value(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return _numpy(value).tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value)!r}")


def _source_paths() -> dict[str, Path]:
    return {
        "script": Path(__file__).resolve(),
        "paired_helper": ROOT / "scripts/diagnose_teob_paired_driver_clean.py",
        "postadiabatic": ROOT / "src/torchwave/approximants/eob/teob_postadiabatic_clean.py",
        "adaptive_continuation": ROOT / "src/torchwave/approximants/eob/teob_adaptive_continuation_clean.py",
        "adaptive_driver": ROOT / "src/torchwave/approximants/eob/adaptive_driver_clean.py",
        "vacuum_driver": ROOT / "src/torchwave/approximants/eob/teob_vacuum_driver_clean.py",
        "trajectory": ROOT / "src/torchwave/approximants/eob/teob_trajectory.py",
        "radiation": ROOT / "src/torchwave/approximants/eob/teob_radiation_clean.py",
        "waveform_state_adapter": ROOT / "src/torchwave/approximants/eob/teob_waveform_state_clean.py",
        "hamiltonian": ROOT / "src/torchwave/approximants/teobresums.py",
        "nqc": ROOT / "src/torchwave/approximants/eob/teob_merger_nqc_clean.py",
        "nqc_basis": ROOT / "src/torchwave/approximants/eob/teob_nqc_clean.py",
        "numerical_attachment": ROOT / "src/torchwave/approximants/eob/numerical_attachment.py",
        "ringdown_assembly": ROOT / "src/torchwave/approximants/eob/teob_ringdown_assembly_clean.py",
        "postmerger": ROOT / "src/torchwave/approximants/eob/teob_postmerger.py",
        "uib_remnant": ROOT / "src/torchwave/approximants/eob/teob_uib2016_remnant.py",
        "contract": ROOT / "docs/reviews/2026-09-13-parity-review/teob-merger-assembly-contract.md",
    }


# Exact fractions from the published Fehlberg 4/5 tableau.  Floats are made
# only at the Tensor multiplication boundary, after the differences are
# formed exactly in Python arithmetic.
F = Fraction
RKF45_C_EXACT = (F(0), F(1, 4), F(3, 8), F(12, 13), F(1), F(1, 2))
RKF45_B_EXACT = (
    (),
    (F(1, 4),),
    (F(3, 32), F(9, 32)),
    (F(1932, 2197), F(-7200, 2197), F(7296, 2197)),
    (F(439, 216), F(-8), F(3680, 513), F(-845, 4104)),
    (F(-8, 27), F(2), F(-3544, 2565), F(1859, 4104), F(-11, 40)),
)
RKF45_B4_EXACT = (F(25, 216), F(0), F(1408, 2565), F(2197, 4104), F(-1, 5), F(0))
RKF45_B5_EXACT = (F(16, 135), F(0), F(6656, 12825), F(28561, 56430), F(-9, 50), F(2, 55))
RKF45_D_EXACT = tuple(b5 - b4 for b5, b4 in zip(RKF45_B5_EXACT, RKF45_B4_EXACT))


def _weighted_sum(stages: list[torch.Tensor], coefficients: tuple[Fraction, ...]) -> torch.Tensor:
    weights = torch.as_tensor(
        [float(value) for value in coefficients],
        dtype=stages[0].dtype,
        device=stages[0].device,
    )
    shape = (len(coefficients),) + (1,) * stages[0].ndim
    return (torch.stack(stages, dim=0) * weights.reshape(shape)).sum(dim=0)


def rational_rkf45_trial(
    field,
    time,
    state,
    step,
    current_field,
    arity,
    *,
    batch_mode="active",
    full_time=None,
    full_state=None,
    indices=None,
):
    """Script-local RKF45 with direct embedded-error arithmetic."""

    stages = [current_field]
    for stage in range(1, 6):
        combination = _weighted_sum(stages, RKF45_B_EXACT[stage])
        stage_state = state + step.unsqueeze(-1) * combination
        stage_time = time + float(RKF45_C_EXACT[stage]) * step
        stages.append(adaptive_driver._call_field_for_rows(
            field,
            stage_time,
            stage_state,
            arity,
            batch_mode=batch_mode,
            full_time=full_time,
            full_state=full_state,
            indices=indices,
        ))
    # The candidate and its error are each accumulated from the base state
    # once.  In particular, error is never obtained by subtracting two large
    # state vectors after their orbital-phase component has accumulated.
    fifth_increment = _weighted_sum(stages, RKF45_B5_EXACT)
    error_increment = _weighted_sum(stages, RKF45_D_EXACT)
    fourth_increment = _weighted_sum(stages, RKF45_B4_EXACT)
    scaled_step = step.unsqueeze(-1)
    fifth = state + scaled_step * fifth_increment
    error = scaled_step * error_increment
    fourth = state + scaled_step * fourth_increment
    if not bool(torch.isfinite(fifth).all() & torch.isfinite(error).all()):
        raise adaptive_driver._TrialDomainFailure("Fehlberg trial produced a nonfinite state")
    return fourth, fifth, error, 5


def _hamiltonian() -> TEOBResumSHamiltonian:
    zero = torch.tensor(0.0, dtype=torch.float64)
    return TEOBResumSHamiltonian(
        torch.tensor(NU_VALUE, dtype=torch.float64), zero, zero, zero, zero,
        delta=torch.tensor(DELTA_VALUE, dtype=torch.float64),
    )


def _region_metrics(
    candidate: np.ndarray,
    public: np.ndarray,
    grid: np.ndarray,
    *,
    pa_end: float,
    event: float,
    attach: float,
) -> dict[str, Any]:
    masks = {
        "full": np.ones(grid.shape, dtype=bool),
        "pa": grid <= pa_end,
        "ode_between_pa_and_event": (grid > pa_end) & (grid < event),
        "attachment_window_5M": np.abs(grid - attach) <= 5.0,
        "post_event": grid >= event,
        "post_attachment": grid >= attach,
    }
    return {
        name: {
            "raw_complex": _complex_metrics(candidate, public, mask),
            "phase": _phase_metrics(candidate, public, mask),
        }
        for name, mask in masks.items()
    }


def _phase_difference(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    values = np.unwrap(np.angle(a * np.conj(b)))
    return {
        "first_radians": float(values[0]),
        "last_radians": float(values[-1]),
        "max_abs_radians": float(np.max(np.abs(values))),
        "rms_radians": float(np.sqrt(np.mean(values * values))),
    }


def _build_waveform(trajectory, hamiltonian, projection_scale, public_mode, public_time):
    trajectory_time = trajectory.time
    state = trajectory.state
    orbital_frequency = trajectory.frequency
    with torch.no_grad():
        h_eff, h_real, h_circ, a_prime = hamiltonian.clean_radiation_inputs(
            state[:, 0], state[:, 2], state[:, 3],
        )
        mode = teob_waveform_state_mode(
            state, hamiltonian, 2, 2, Omega=orbital_frequency,
            residual_phase="zero-unpublished",
        )
        amplitude = mode.abs()
        phase = phase_from_mode(mode, state[:, 1])
        exact_radial = conservative_trajectory_quantities(
            hamiltonian, state,
        ).radial_acceleration
        # This is the explicit compatibility hypothesis requested by the
        # radial-acceleration work: C is evaluated at the current state and
        # held fixed while differentiating the radial Hamilton equation.
        from torchwave.approximants.eob.teob_radial_acceleration_clean import radial_acceleration_approximations
        radial = radial_acceleration_approximations(
            hamiltonian, state[:, 0], state[:, 2], state[:, 3],
        ).frozen_mobility
    nu = torch.tensor(NU_VALUE, dtype=torch.float64)
    targets = published_2017_nqc_targets(nu, torch.tensor(0.0, dtype=torch.float64))
    nqc = assemble_teob_merger_nqc(
        trajectory_time,
        state,
        amplitude=amplitude,
        phase=phase,
        orbital_frequency=orbital_frequency,
        radial_acceleration=radial,
        nu=nu,
        fixed_step=torch.tensor(DEFAULT_FIXED_STEP_CAP, dtype=torch.float64),
        targets=targets,
        target_provenance=PUBLISHED_2017_NQC_PROVENANCE,
    )
    final_mass, final_spin = uib2016_nonspinning_remnant(MASS1_MSUN, MASS2_MSUN)
    final_mass = final_mass.to(dtype=torch.float64)
    final_spin = final_spin.to(dtype=torch.float64)
    matching_state = nqc.auxiliary.state[nqc.matching_index]
    raw_assembly = _assemble(
        time=trajectory_time, amplitude=amplitude, phase=phase,
        auxiliary_time=nqc.auxiliary.time,
        auxiliary_amplitude=nqc.auxiliary.amplitude,
        auxiliary_phase=nqc.auxiliary.phase,
        auxiliary_frequency=nqc.auxiliary.orbital_frequency,
        nu=nu, final_mass=final_mass, final_spin=final_spin,
        matching_time=nqc.t_nqc, matching_state=matching_state,
    )
    corrected_assembly = _assemble(
        time=trajectory_time,
        amplitude=nqc.original.corrected_amplitude,
        phase=nqc.original.corrected_phase,
        auxiliary_time=nqc.auxiliary.time,
        auxiliary_amplitude=nqc.auxiliary.corrected_amplitude,
        auxiliary_phase=nqc.auxiliary.corrected_phase,
        auxiliary_frequency=nqc.auxiliary.orbital_frequency,
        nu=nu, final_mass=final_mass, final_spin=final_spin,
        matching_time=nqc.t_nqc, matching_state=matching_state,
    )
    raw_report, raw_public, _, _ = _public_grid_branch(
        raw_assembly, public_mode, public_time, projection_scale, final_mass,
    )
    corrected_report, corrected_public, _, _ = _public_grid_branch(
        corrected_assembly, public_mode, public_time, projection_scale, final_mass,
    )
    return {
        "trajectory_time": trajectory_time,
        "state": state,
        "frequency": orbital_frequency,
        "h_eff": h_eff,
        "h_real": h_real,
        "h_circ": h_circ,
        "a_prime": a_prime,
        "mode": mode,
        "amplitude": amplitude,
        "phase": phase,
        "exact_radial": exact_radial,
        "radial": radial,
        "nqc": nqc,
        "final_mass": final_mass,
        "final_spin": final_spin,
        "raw_assembly": raw_assembly,
        "corrected_assembly": corrected_assembly,
        "raw_public": raw_public,
        "corrected_public": corrected_public,
        "raw_report": raw_report,
        "corrected_report": corrected_report,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public", type=Path, default=DEFAULT_PUBLIC)
    parser.add_argument("--public-meta", type=Path, default=DEFAULT_PUBLIC_META)
    parser.add_argument("--input-cache", type=Path, default=DEFAULT_INPUT_CACHE)
    parser.add_argument("--branch-cache", type=Path, default=DEFAULT_BRANCH_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--log-path", type=Path, default=DEFAULT_LOG)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    args.log_path.parent.mkdir(parents=True, exist_ok=True)
    torch.set_default_dtype(torch.float64)
    started = wall_time.perf_counter()
    sources = _source_paths()
    source_before = {name: _sha256(path) for name, path in sources.items()}
    preserved = {
        "input_cache": _sha256(args.input_cache),
        "branch_cache": _sha256(args.branch_cache),
        "public_fixture": _sha256(args.public),
        "public_metadata": _sha256(args.public_meta),
    }
    command = shlex.join([sys.executable, *sys.argv])
    report: dict[str, Any] = {
        "status": "running",
        "diagnostic_scope": "script-local direct Fehlberg embedded-error arithmetic on LO-horizon TEOB continuation",
        "run_control": {
            "host": socket.gethostname(),
            "cwd": os.getcwd(),
            "command": command,
            "pid": os.getpid(),
            "python": sys.executable,
            "platform": platform.platform(),
            "device": "cpu",
            "started_unix": wall_time.time(),
            "log_path": str(args.log_path),
            "expected_next_check": "inspect diagnostic.json and candidate-output.npz after the bounded process exits",
            "stop_command": f"kill {os.getpid()}",
        },
        "probe": {
            "nu": NU_VALUE,
            "delta": DELTA_VALUE,
            "mass1_msun": MASS1_MSUN,
            "mass2_msun": MASS2_MSUN,
            "chi1": 0.0,
            "chi2": 0.0,
            "lambda1": 0.0,
            "lambda2": 0.0,
            "r0": PUBLIC_RADIUS,
            "r_min": R_MIN,
            "dr": DR,
            "pa_order": PA_ORDER,
            "horizon_policy": "leading-order diagnostic replacement",
            "fixed_step_cap": DEFAULT_FIXED_STEP_CAP,
            "continuation_duration": DEFAULT_CONTINUATION_DURATION,
            "atol": DEFAULT_ATOL,
            "rtol": DEFAULT_RTOL,
            "max_rejected_steps": DEFAULT_MAX_REJECTED_STEPS,
            "max_evaluations": DEFAULT_MAX_EVALUATIONS,
            "max_accepted_steps": DEFAULT_MAX_ACCEPTED_STEPS,
            "max_duration": DEFAULT_MAX_DURATION,
            "output_step_initial_mass": OUTPUT_STEP,
            "public_epoch_initial_mass": PUBLIC_EPOCH_INITIAL_MASS,
            "public_epoch_interpretation": "accepted frequency-decrease event coordinate; not an interpolated peak",
        },
        "preserved_inputs_before": preserved,
        "source_sha256_before": source_before,
        "output_path": str(args.output),
        "cache_path": str(args.cache),
        "arithmetic": {
            "candidate_formula": "state + step * sum(b5_i * k_i)",
            "error_formula": "step * sum((b5_i-b4_i) * k_i)",
            "source": "NASA TR R-315 Fehlberg 4/5 tableau, exact Fraction transcription",
            "c": [str(value) for value in RKF45_C_EXACT],
            "b4": [str(value) for value in RKF45_B4_EXACT],
            "b5": [str(value) for value in RKF45_B5_EXACT],
            "b5_minus_b4": [str(value) for value in RKF45_D_EXACT],
            "sum_b4": str(sum(RKF45_B4_EXACT)),
            "sum_b5": str(sum(RKF45_B5_EXACT)),
            "sum_difference": str(sum(RKF45_D_EXACT)),
            "runtime_patch": "adaptive_driver_clean._rkf45_trial replaced only inside this process and restored in finally",
        },
    }
    original_trial = adaptive_driver._rkf45_trial
    output_arrays: dict[str, np.ndarray] = {}
    try:
        public_mode, public_time, public_data = _load_public(args.public, args.public_meta)
        if public_data.get("epoch_initial_mass") is not None:
            report["public_fixture"] = {
                "path": str(args.public),
                "metadata_path": str(args.public_meta),
                "sha256": preserved["public_fixture"],
                "metadata_sha256": preserved["public_metadata"],
                "samples": int(public_time.size),
                "relative_support_initial_mass": [float(public_time[0]), float(public_time[-1])],
                "epoch_initial_mass_from_fixture": float(public_data["epoch_initial_mass"]),
                "epoch_source_for_event_comparison": "supplied public accepted-event coordinate",
            }
        try:
            import lal  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("the exact public projection requires installed lal constants") from exc
        projection_scale = -math.sqrt(5.0 / (4.0 * math.pi)) * (
            50.0 * lal.MTSUN_SI * lal.C_SI / (DISTANCE_MPC * 1e6 * lal.PC_SI)
        )
        hamiltonian = _hamiltonian()
        reaction = _LeadingOrderHorizonReaction()
        pa = alternating8pa(
            hamiltonian, reaction, r0=PUBLIC_RADIUS, r_min=R_MIN, dr=DR,
            pa_order=PA_ORDER, include_horizon_absorption=True,
        )
        if not pa.complete:
            raise RuntimeError(f"PA profile did not complete: {pa.skip_reason}")
        adaptive_driver._rkf45_trial = rational_rkf45_trial
        continuation = continue_teob_adaptive(
            hamiltonian, pa, reaction, include_horizon_absorption=True,
            fixed_step_cap=DEFAULT_FIXED_STEP_CAP,
            continuation_duration=DEFAULT_CONTINUATION_DURATION,
            max_rejected_steps=DEFAULT_MAX_REJECTED_STEPS,
            max_evaluations=DEFAULT_MAX_EVALUATIONS,
            max_accepted_steps=DEFAULT_MAX_ACCEPTED_STEPS,
            max_duration=DEFAULT_MAX_DURATION,
        )
        if not continuation.complete:
            raise RuntimeError(f"continuation did not complete: {continuation.skip_reason}")
        built = _build_waveform(continuation, hamiltonian, projection_scale, public_mode, public_time)
        candidate = built["corrected_public"]
        raw_candidate = built["raw_public"]
        event_time = _float(continuation.peak_event_time)
        pa_end_time = _float(pa.endpoint_time)
        attach_time = _float(built["corrected_assembly"].events.attachment_time)
        branch_data = np.load(args.branch_cache, allow_pickle=False)
        baseline_key = "cache1_paired_physical_output_lo_horizon_frozen_mobility_nqc_output_mode"
        baseline = np.asarray(branch_data[baseline_key])
        if baseline.shape != public_mode.shape:
            raise ValueError("frozen-mobility baseline and public fixture have different lengths")
        baseline_raw = np.asarray(branch_data["cache1_paired_physical_output_lo_horizon_raw_output_mode"])
        report["continuation"] = {
            "status": continuation.status,
            "samples": int(continuation.time.numel()),
            "adaptive_samples": int(continuation.adaptive_time.numel()),
            "fixed_samples": int(continuation.fixed_time.numel()),
            "support_initial_mass": [_float(continuation.time[0]), _float(continuation.time[-1])],
            "accepted_event_time_initial_mass": event_time,
            "public_event_time_initial_mass": -PUBLIC_EPOCH_INITIAL_MASS,
            "accepted_event_time_error_vs_public_initial_mass": (
                event_time - (-PUBLIC_EPOCH_INITIAL_MASS)
            ),
            "accepted_event_frequency": _float(continuation.peak_event_frequency),
            "endpoint_state": _vector(continuation.endpoint_state),
            "first_ode_step": _float(continuation.first_ode_step),
            "adaptive_next_step": _float(continuation.adaptive_next_step),
            "fixed_step": _float(continuation.fixed_step),
            "phase_seam_error": _float(continuation.phase_continuity),
            "statistics": {
                "adaptive_accepted_steps": continuation.statistics.adaptive_accepted_steps,
                "adaptive_rejected_steps": continuation.statistics.adaptive_rejected_steps,
                "adaptive_evaluations": continuation.statistics.adaptive_evaluations,
                "fixed_accepted_steps": continuation.statistics.fixed_accepted_steps,
                "fixed_rejected_steps": continuation.statistics.fixed_rejected_steps,
                "fixed_evaluations": continuation.statistics.fixed_evaluations,
                "accepted_steps": continuation.statistics.accepted_steps,
                "rejected_steps": continuation.statistics.rejected_steps,
                "evaluations": continuation.statistics.evaluations,
                "termination": continuation.statistics.termination,
            },
        }
        report["nqc"] = _nqc_summary(built["nqc"])
        report["assembly"] = {
            "carrier_provider": "TEOBRadiationClean.mode through teob_waveform_state_mode",
            "tail_applications": 1,
            "radial_branch": "frozen_mobility (current-state mobility held fixed in d/dt)",
            "attachment_time_initial_mass": attach_time,
            "pa_endpoint_time_initial_mass": pa_end_time,
            "raw_vs_public": _region_metrics(raw_candidate, public_mode, public_time, pa_end=pa_end_time, event=event_time, attach=attach_time),
            "nqc_vs_public": _region_metrics(candidate, public_mode, public_time, pa_end=pa_end_time, event=event_time, attach=attach_time),
            "nqc_phase_contribution": _phase_difference(candidate, raw_candidate),
            "carrier_raw_phase_error": _phase_difference(raw_candidate, public_mode),
            "nqc_corrected_phase_error": _phase_difference(candidate, public_mode),
            "baseline_frozen_vs_public": _region_metrics(baseline, public_mode, public_time, pa_end=pa_end_time, event=event_time, attach=attach_time),
            "baseline_raw_vs_public": _region_metrics(baseline_raw, public_mode, public_time, pa_end=pa_end_time, event=event_time, attach=attach_time),
            "candidate_vs_frozen_baseline": _region_metrics(candidate, baseline, public_time, pa_end=pa_end_time, event=event_time, attach=attach_time),
        }
        report["trajectory_comparison"] = {
            "reference_cache": str(args.input_cache),
            "reference_cache_sha256": preserved["input_cache"],
            "reference_time_samples": int(np.load(args.input_cache, allow_pickle=False)["time"].size),
            "candidate_time_samples": int(continuation.time.numel()),
            "candidate_pa_endpoint_time_initial_mass": _float(pa.endpoint_time),
            "candidate_adaptive_event_time_initial_mass": event_time,
        }
        reference = np.load(args.input_cache, allow_pickle=False)
        if reference["time"].shape == _numpy(built["trajectory_time"]).shape:
            report["trajectory_comparison"].update({
                "max_abs_time_difference_initial_mass": float(np.max(np.abs(_numpy(built["trajectory_time"]) - reference["time"]))),
                "max_abs_state_difference": float(np.max(np.abs(_numpy(built["state"]) - reference["state"]))),
                "max_abs_frequency_difference": float(np.max(np.abs(_numpy(built["frequency"]) - reference["frequency"]))),
            })
        output_arrays = {
            "public_time": public_time,
            "public_mode": public_mode,
            "candidate_raw_public_mode": raw_candidate,
            "candidate_frozen_mobility_nqc_public_mode": candidate,
            "baseline_frozen_mobility_nqc_public_mode": baseline,
            "baseline_raw_public_mode": baseline_raw,
            "candidate_time": _numpy(built["trajectory_time"]),
            "candidate_state": _numpy(built["state"]),
            "candidate_frequency": _numpy(built["frequency"]),
            "candidate_mode": _numpy(built["mode"]),
            "candidate_amplitude": _numpy(built["amplitude"]),
            "candidate_phase": _numpy(built["phase"]),
            "candidate_exact_radial_acceleration": _numpy(built["exact_radial"]),
            "candidate_frozen_mobility_radial_acceleration": _numpy(built["radial"]),
            "candidate_pa_time": _numpy(pa.time),
            "candidate_pa_state": _numpy(pa.state),
            "candidate_adaptive_time": _numpy(continuation.adaptive_time),
            "candidate_adaptive_state": _numpy(continuation.adaptive_state),
            "candidate_fixed_time": _numpy(continuation.fixed_time),
            "candidate_fixed_state": _numpy(continuation.fixed_state),
            "candidate_nqc_corrected_amplitude": _numpy(built["nqc"].original.corrected_amplitude),
            "candidate_nqc_corrected_phase": _numpy(built["nqc"].original.corrected_phase),
            "candidate_nqc_aux_time": _numpy(built["nqc"].auxiliary.time),
            "candidate_nqc_aux_corrected_amplitude": _numpy(built["nqc"].auxiliary.corrected_amplitude),
            "candidate_nqc_aux_corrected_phase": _numpy(built["nqc"].auxiliary.corrected_phase),
        }
        np.savez_compressed(args.cache, **output_arrays)
        report["candidate_output"] = {
            "path": str(args.cache),
            "sha256": _sha256(args.cache),
            "arrays": sorted(output_arrays),
        }
        report["status"] = "complete"
    except Exception as exc:  # diagnostic failure is itself reported
        report["status"] = "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        adaptive_driver._rkf45_trial = original_trial
    source_after = {name: _sha256(path) for name, path in sources.items()}
    report["source_sha256_after"] = source_after
    report["source_hashes_unchanged"] = source_before == source_after
    report["preserved_inputs_after"] = {name: _sha256(path) for name, path in {
        "input_cache": args.input_cache,
        "branch_cache": args.branch_cache,
        "public_fixture": args.public,
        "public_metadata": args.public_meta,
    }.items()}
    report["preserved_inputs_unchanged"] = report["preserved_inputs_before"] == report["preserved_inputs_after"]
    report["elapsed_seconds"] = wall_time.perf_counter() - started
    report["runtime"] = {
        "host": socket.gethostname(),
        "cwd": os.getcwd(),
        "python": sys.executable,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
    }
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=_json_value) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": report["status"],
        "output": str(args.output),
        "cache": str(args.cache),
        "cache_sha256": report.get("candidate_output", {}).get("sha256"),
        "elapsed_seconds": report["elapsed_seconds"],
        "source_hashes_unchanged": report["source_hashes_unchanged"],
        "preserved_inputs_unchanged": report["preserved_inputs_unchanged"],
    }, sort_keys=True))
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

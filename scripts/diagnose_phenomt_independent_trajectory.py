"""Compose PhenomT's carrier with an independent scalar precession solve.

The carrier is the only production input to the reference path.  Its scalar
``v(t/M)`` values drive the paper-derived mpmath RHS, a SciPy Brent event root,
adaptive DOP853 evolution, tangent transport, and independent Euler-angle
reconstruction.  Public LAL binaries supply only unmodified initial-rate and
relative-trajectory observations; no implementation source is inspected and
no binary output is fitted or translated into production equations.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import lal
import lalsimulation as ls
import numpy as np
import scipy
import torch

from diffgw_core.constants import MSUN_SEC
from diffgw_reference.phenomt_numerical_trajectory import (
    euler_angles,
    inertial_basis,
    integrate_trajectory,
    initial_state,
    rotation_matrix,
)
from diffgw_reference.phenomt_spin_evolution import evaluate_spin_derivatives
from scripts.diagnose_phenomt_initial_rates import binary_rate, native_rate, spin_taylor_control
from scripts.validate_phenomt_precession import CASES, parameters
from torchwave.approximants import _phenomt_numerical as native_numerical
from torchwave.approximants.imrphenomt import IMRPhenomT


def _rotation_distance(left, right):
    # For proper rotations ||R1-R2||_F = 2 sqrt(2) sin(theta/2).  This form
    # resolves the sub-microradian regime better than arccos(trace).
    chord = np.linalg.norm(left - right, axis=(-2, -1)) / (2 * np.sqrt(2))
    return 2 * np.arcsin(np.clip(chord, 0, 1))


def _relative_rotations(rotations):
    return np.swapaxes(rotations[0], -1, -2) @ rotations


def _first_threshold_crossings(times, values, thresholds):
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    if times.ndim != 1 or values.shape != times.shape or len(times) == 0:
        raise ValueError("threshold inputs must be nonempty one-dimensional arrays")
    result = {}
    for threshold in thresholds:
        indices = np.flatnonzero(values >= threshold)
        if len(indices) == 0:
            result[f"{threshold:g}"] = None
            continue
        index = int(indices[0])
        result[f"{threshold:g}"] = dict(
            sample_offset=index,
            time_seconds=float(times[index]),
            elapsed_from_reference_seconds=float(times[index] - times[0]),
            value_radians=float(values[index]),
        )
    return result


def _public_numerical_rotations(case, rate):
    m1, m2, spin1, spin2, inclination, phase, fref = CASES[case]
    options = lal.CreateDict()
    ls.SimInspiralWaveformParamsInsertPhenomXPrecVersion(options, 300)
    ls.SimInspiralWaveformParamsInsertPhenomXPFinalSpinMod(options, 4)
    modes, alpha, cosbeta, gamma, _ = ls.SimIMRPhenomTPHM_CoprecModes(
        m1 * lal.MSUN_SI,
        m2 * lal.MSUN_SI,
        *spin1,
        *spin2,
        100e6 * lal.PC_SI,
        inclination,
        1 / rate,
        20.0,
        fref,
        phase,
        options,
        1,
    )
    mode = ls.SphHarmTimeSeriesGetMode(modes, 2, 2)
    times = float(mode.epoch) + mode.deltaT * np.arange(mode.data.length)
    for series in (alpha, cosbeta, gamma):
        if (
            series.data.length != len(times)
            or series.deltaT != mode.deltaT
            or series.epoch != mode.epoch
        ):
            raise RuntimeError("public angle grids are inconsistent")
    rotations = rotation_matrix(
        alpha.data.data,
        np.arccos(np.clip(cosbeta.data.data, -1, 1)),
        gamma.data.data,
    )
    return times, rotations


def _public_trajectory_comparison(case, rate, trajectory, total_seconds, basis):
    public_times, public_rotations = _public_numerical_rotations(case, rate)
    reference_seconds = trajectory.reference_time * total_seconds
    clock_shift = reference_seconds - public_times[0]
    independent_times = public_times + clock_shift
    tolerance = 16 * np.finfo(float).eps * max(1.0, abs(reference_seconds))
    use = (
        (public_times < 0)
        & (independent_times >= reference_seconds - tolerance)
        & (independent_times <= tolerance)
    )
    if not np.any(use) or not use[0]:
        raise RuntimeError("public inspiral does not cover the independent reference event")
    query = np.clip(independent_times[use] / total_seconds, trajectory.reference_time, 0.0)
    independent_states = trajectory.evaluate(query)
    independent_rotations = rotation_matrix(*euler_angles(independent_states, basis))
    independent_relative = _relative_rotations(independent_rotations)
    public_relative = np.swapaxes(public_rotations[0], -1, -2) @ public_rotations[use]
    triad_error = _rotation_distance(independent_relative, public_relative)
    axis_chord = np.linalg.norm(
        independent_relative[..., 2] - public_relative[..., 2], axis=-1
    ) / 2
    axis_error = 2 * np.arcsin(np.clip(axis_chord, 0, 1))
    selected_public_times = public_times[use]
    return dict(
        public_model_options=dict(precession_version=300, final_spin_mod=4),
        alignment=(
            "The first public sample and the independently solved carrier event are both "
            "the supplied f_ref event; R(first)^T R(t) removes their fixed initial frames. "
            "No fitted time shift, phase, or rotation is used."
        ),
        samples=int(np.count_nonzero(use)),
        public_epoch_seconds=float(public_times[0]),
        independent_reference_time_seconds=float(reference_seconds),
        event_clock_shift_seconds=float(clock_shift),
        initial_triad_error_radians=float(triad_error[0]),
        max_axis_error_radians=float(np.max(axis_error)),
        rms_axis_error_radians=float(np.sqrt(np.mean(axis_error**2))),
        final_axis_error_radians=float(axis_error[-1]),
        max_triad_error_radians=float(np.max(triad_error)),
        rms_triad_error_radians=float(np.sqrt(np.mean(triad_error**2))),
        final_triad_error_radians=float(triad_error[-1]),
        first_triad_error_crossings=_first_threshold_crossings(
            selected_public_times, triad_error, (1e-4, 1e-3, 1e-2, 1e-1)
        ),
    )


def _invariants(states):
    lhat, spin1, spin2, tangent = np.split(states, 4, axis=-1)
    return dict(
        lhat_norm_max_abs=float(np.max(abs(np.linalg.norm(lhat, axis=-1) - 1))),
        spin1_norm_max_abs=float(np.max(abs(
            np.linalg.norm(spin1, axis=-1) - np.linalg.norm(spin1[0])
        ))),
        spin2_norm_max_abs=float(np.max(abs(
            np.linalg.norm(spin2, axis=-1) - np.linalg.norm(spin2[0])
        ))),
        tangent_norm_max_abs=float(np.max(abs(np.linalg.norm(tangent, axis=-1) - 1))),
        lhat_dot_tangent_max_abs=float(np.max(abs(np.sum(lhat * tangent, axis=-1)))),
    )


def _component_residual(left, right):
    difference = np.asarray(left) - np.asarray(right)
    return dict(
        max_vector_norm=float(np.max(np.linalg.norm(difference, axis=-1))),
        relative_l2=float(np.linalg.norm(difference) / np.linalg.norm(right)),
    )


def _initial_even_rate(m1, m2, spin, fref, sample_rate):
    spin = np.asarray(spin, dtype=float)
    total = m1 + m2
    x1, x2 = m1 / total, m2 / total
    total_seconds = total * MSUN_SEC
    reference_velocity = (math.pi * total_seconds * fref) ** (1 / 3)
    independent_signed = []
    public_signed = []
    for sign in (1.0, -1.0):
        state = initial_state(x1, x2, sign * spin, np.zeros(3))
        lhat, spin1, spin2, _ = np.split(state, 4)
        rhs = evaluate_spin_derivatives(
            reference_velocity,
            lhat,
            spin1,
            spin2,
            x1,
            x2,
            include_n3lo_ss=False,
        )
        independent_signed.append(np.asarray(rhs.dlhat, dtype=float) / total_seconds)
        public_signed.append(np.asarray(binary_rate(
            m1, m2, sign * spin, 1.2, 0.0, fref, sample_rate
        )["rate_per_second"]))
    independent = sum(independent_signed) / 2
    public = sum(public_signed) / 2
    native = np.asarray(native_rate("torch", m1, m2, spin, fref)["even_rate_per_second"])
    return dict(
        construction="average of +chi1 and -chi1 with chi2=0; odd-in-spin terms cancel",
        independent_published_rate_per_second=independent.tolist(),
        native_rate_per_second=native.tolist(),
        public_binary_rate_per_second=public.tolist(),
        independent_norm_per_second=float(np.linalg.norm(independent)),
        native_norm_per_second=float(np.linalg.norm(native)),
        public_binary_norm_per_second=float(np.linalg.norm(public)),
        native_minus_independent_max_abs_per_second=float(np.max(abs(native - independent))),
        public_minus_independent_norm_per_second=float(np.linalg.norm(public - independent)),
        spin_taylor_kappa1=spin_taylor_control(m1, m2, spin, fref, sample_rate, 1.0),
        spin_taylor_kappa0=spin_taylor_control(m1, m2, spin, fref, sample_rate, 0.0),
    )


def diagnose(case, samples, sample_rate):
    m1, m2, spin1, spin2, _inclination, _phase, fref = CASES[case]
    total = m1 + m2
    x1, x2 = m1 / total, m2 / total
    total_seconds = total * MSUN_SEC
    model = IMRPhenomT()
    carrier = model._coeff(
        torch.tensor(m1, dtype=torch.float64),
        torch.tensor(m2, dtype=torch.float64),
        torch.tensor(spin1[2], dtype=torch.float64),
        torch.tensor(spin2[2], dtype=torch.float64),
    )

    def velocity(time):
        omega = model._evaluate(torch.tensor(float(time), dtype=torch.float64), carrier)[0]
        return float(torch.clamp(omega / 2, min=1e-30).pow(1 / 3))

    reference_velocity = (math.pi * total_seconds * fref) ** (1 / 3)
    native_reference_seconds = float(model.time_at_frequency(fref, m1, m2, spin1[2], spin2[2]))
    native_reference = native_reference_seconds / total_seconds
    width = max(1.0, 0.1 * abs(native_reference))
    bracket = (native_reference - width, min(-np.finfo(float).eps, native_reference + width))
    solve_options = dict(
        velocity_at_time=velocity,
        reference_velocity=reference_velocity,
        root_bracket=bracket,
        end_time=0.0,
        mass1_fraction=x1,
        mass2_fraction=x2,
        spin1=spin1,
        spin2=spin2,
        dps=50,
    )
    tight = integrate_trajectory(**solve_options, rtol=2e-11, atol=2e-13)
    loose = integrate_trajectory(**solve_options, rtol=2e-9, atol=2e-11)
    times = np.linspace(tight.reference_time, 0.0, samples)
    independent = tight.evaluate(times)
    loose_states = loose.evaluate(times)

    prepared = native_numerical.prepare_numerical_trajectory(
        m1, m2, *spin1, *spin2, fref, n_backward=2048, n_forward=1024
    )
    native = native_numerical._interpolate(
        prepared, torch.as_tensor(times * total_seconds, dtype=torch.float64)
    ).detach().numpy()

    initial = initial_state(x1, x2, spin1, spin2)
    lhat, s1, s2, _ = np.split(initial, 4)
    initial_rhs = evaluate_spin_derivatives(
        reference_velocity, lhat, s1, s2, x1, x2, include_n3lo_ss=False
    )
    basis = inertial_basis(initial, float(initial_rhs.orbital_angular_momentum))
    independent_rotation = rotation_matrix(*euler_angles(independent, basis))
    native_rotation_from_reference = rotation_matrix(*euler_angles(native, basis))
    native_angles = tuple(
        value.detach().numpy() for value in native_numerical._angles(
            torch.as_tensor(native), prepared
        )
    )
    native_rotation_from_production = rotation_matrix(*native_angles)
    loose_rotation = rotation_matrix(*euler_angles(loose_states, basis))

    state_components = {}
    for name, index in (("lhat", 0), ("spin1", 1), ("spin2", 2), ("tangent", 3)):
        state_components[name] = _component_residual(
            np.split(native, 4, axis=-1)[index], np.split(independent, 4, axis=-1)[index]
        )
    return dict(
        case=case,
        samples=samples,
        carrier_scope="native scalar PhenomT v(t/M); all event/ODE/transport/angle operations independent",
        reference_event=dict(
            target_velocity=reference_velocity,
            bracket_t_over_m=list(bracket),
            independent_t_over_m=tight.reference_time,
            native_t_over_m=native_reference,
            absolute_difference_t_over_m=abs(tight.reference_time - native_reference),
            absolute_difference_seconds=abs(tight.reference_time - native_reference) * total_seconds,
        ),
        solver=dict(
            method="SciPy DOP853 with dense output",
            tight=dict(rtol=2e-11, atol=2e-13, nfev=tight.nfev, njev=tight.njev, nlu=tight.nlu),
            loose=dict(rtol=2e-9, atol=2e-11, nfev=loose.nfev, njev=loose.njev, nlu=loose.nlu),
            convergence_state_max_vector_norm=float(np.max(np.linalg.norm(
                independent - loose_states, axis=-1
            ))),
            convergence_rotation_max_radians=float(np.max(_rotation_distance(
                independent_rotation, loose_rotation
            ))),
        ),
        independent_invariants=_invariants(independent),
        native_minus_independent_states=state_components,
        native_minus_independent_rotation=dict(
            max_radians=float(np.max(_rotation_distance(
                independent_rotation, native_rotation_from_reference
            ))),
            final_radians=float(_rotation_distance(
                independent_rotation[-1], native_rotation_from_reference[-1]
            )),
        ),
        production_angle_reconstruction_minus_independent=dict(
            fixed_initial_frame_radians=float(_rotation_distance(
                native_rotation_from_production[0], native_rotation_from_reference[0]
            )),
            absolute_max_radians=float(np.max(_rotation_distance(
                native_rotation_from_production, native_rotation_from_reference
            ))),
            relative_evolution_max_radians=float(np.max(_rotation_distance(
                _relative_rotations(native_rotation_from_production),
                _relative_rotations(native_rotation_from_reference),
            ))),
        ),
        public_binary_minus_independent_trajectory=_public_trajectory_comparison(
            case, sample_rate, tight, total_seconds, basis
        ),
        initial_spin_even_rate=_initial_even_rate(m1, m2, spin1, fref, sample_rate),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=tuple(CASES), default="high_mass_ratio")
    parser.add_argument("--samples", type=int, default=65)
    parser.add_argument("--sample-rate", type=int, default=16384)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.samples < 3 or args.sample_rate <= 0:
        parser.error("samples must be at least three and sample rate must be positive")
    result = diagnose(args.case, args.samples, args.sample_rate)
    receipt = dict(
        generated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        reproduction=(
            "PYTHONPATH=src python scripts/diagnose_phenomt_independent_trajectory.py "
            f"--case {args.case} --samples {args.samples} --sample-rate {args.sample_rate} --output receipt.json"
        ),
        versions=dict(
            python=platform.python_version(),
            numpy=np.__version__,
            scipy=scipy.__version__,
            torch=torch.__version__,
            lalsimulation=ls.__version__,
        ),
        dtype="float64 with 50-decimal mpmath scalar RHS",
        parameters=parameters(args.case),
        conventions=dict(
            reference=(
                "Published equations only; public binary used as an unmodified initial-rate and "
                "relative-trajectory observation."
            ),
            production="No production equation or coefficient is modified by this diagnostic.",
            source_restriction="No LAL implementation source is inspected or translated; no binary output is fitted.",
        ),
        result=result,
    )
    encoded = json.dumps(receipt, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()

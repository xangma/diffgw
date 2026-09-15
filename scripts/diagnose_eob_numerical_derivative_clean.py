#!/usr/bin/env python3
"""Run a small pointwise report for the clean EOB finite-difference experiment.

The report intentionally exercises only smooth analytic functions and a few
conservative Hamiltonian states.  Its optional public-fixture section uses
only the clean Torch waveform and attachment helpers; it does not import
external reference implementations.  Use ``--output`` to save the JSON
report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from torchwave.constants import C_SI, MSUN_SEC
from torchwave.approximants.eob.hamiltonian import EOBHamiltonian
from torchwave.approximants.eob.numerical_attachment import (
    attach_seobnrv4_22_numerical_compatibility,
)
from torchwave.approximants.eob.numerical_derivative_clean import (
    adaptive_central_derivative,
    central_quartic,
    numerical_eob_conservative_derivatives,
)
from torchwave.approximants.eob.published_waveform import (
    published_mode22_and_frequency,
)
from torchwave.approximants.eob.remnant import (
    eob_final_mass_fraction,
    hofmann_final_spin,
)
from torchwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux
from torchwave.approximants.eob.seobnrv4_hamiltonian import SEOBNRv4Hamiltonian
from torchwave.approximants.kerr_qnm_table_convention import (
    signed_qnm_220_table_convention,
)


DTYPE = torch.float64
REPO_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_DATA = Path("/tmp")
PUBLIC_LABELS = ("dt_0p5M", "dt_1_4096s")
TOTAL_MASS_MSUN = 50.0
DISTANCE_PC = 100.0e6
# These are the public SI constants used by the fixture scale.  Keeping the
# expression explicit avoids adding a runtime LAL dependency to this clean
# diagnostic while preserving the requested 50 Msun/100 Mpc convention.
MTSUN_SI = 4.925490947641267e-6
PC_SI = 3.085677581491367e16
MASS_SECONDS = TOTAL_MASS_MSUN * MTSUN_SI
MODE_SCALE = TOTAL_MASS_MSUN * MTSUN_SI * C_SI / (DISTANCE_PC * PC_SI)
INITIAL_STEP = 1.0e-4
DERIVATIVE_ITERATIONS = 1
PUBLIC_CHOICES = {
    "spin_v6": "bohe_total_energy",
    "include_pan_v8": False,
    "include_positive_chi_540": False,
    "retain_pan_cubic_spin": True,
}
PUBLIC_SOURCE_FILES = (
    "scripts/diagnose_eob_numerical_derivative_clean.py",
    "src/torchwave/approximants/eob/numerical_derivative_clean.py",
    "src/torchwave/approximants/eob/numerical_attachment.py",
    "src/torchwave/approximants/eob/published_waveform.py",
    "src/torchwave/approximants/eob/seobnrv4_flux.py",
    "src/torchwave/approximants/eob/seobnrv4_hamiltonian.py",
    "src/torchwave/approximants/eob/remnant.py",
    "src/torchwave/approximants/eob/events.py",
    "src/torchwave/approximants/eob/nqc.py",
    "src/torchwave/approximants/kerr_qnm_table_convention.py",
)


def _tensor(value):
    return torch.as_tensor(value, dtype=DTYPE)


def _scalar(value):
    """Convert a scalar Torch/NumPy value to a JSON-friendly float."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().item()
    if isinstance(value, np.generic):
        value = value.item()
    return float(value)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _source_hashes():
    return {
        relative: _sha256(REPO_ROOT / relative)
        for relative in PUBLIC_SOURCE_FILES
    }


def _load_public_fixture(label):
    """Load one copied public post-peak state/mode fixture."""
    path = PUBLIC_DATA / f"torchwave-public-eob-{label}.npz"
    if not path.exists():
        raise FileNotFoundError(f"missing public fixture: {path}")
    with np.load(path, allow_pickle=False) as loaded:
        postpeak = np.asarray(loaded["postpeak_state"], dtype=np.float64).copy()
        mode = np.asarray(loaded["mode22"], dtype=np.complex128).copy()
        epoch_s = float(np.asarray(loaded["epoch_s"]).item())
        dt_s = float(np.asarray(loaded["dt_s"]).item())
    if postpeak.ndim != 2 or postpeak.shape[0] != 5 or postpeak.shape[1] < 9:
        raise ValueError(f"{label}: postpeak_state must have shape (5,N>=9)")
    if mode.ndim != 1 or mode.size < 2:
        raise ValueError(f"{label}: mode22 must be a nonempty one-dimensional array")
    if not np.isfinite(postpeak).all() or not np.isfinite(mode).all():
        raise ValueError(f"{label}: fixture contains non-finite values")
    if not np.all(np.diff(postpeak[0]) > 0):
        raise ValueError(f"{label}: postpeak_state time is not strictly increasing")
    if not np.isfinite(epoch_s) or not np.isfinite(dt_s) or dt_s <= 0:
        raise ValueError(f"{label}: invalid epoch_s or dt_s")
    return {
        "label": label,
        "path": path,
        "sha256": _sha256(path),
        "size_bytes": int(path.stat().st_size),
        "postpeak_state": postpeak,
        "mode22": mode,
        "epoch_s": epoch_s,
        "dt_s": dt_s,
    }


def _public_physics():
    """Build the fixed clean physics inputs used by both frequency paths."""
    zero = _tensor(0.0)
    nu = _tensor(0.24)
    hamiltonian = SEOBNRv4Hamiltonian(nu, zero, zero, delta=_tensor(0.2))
    radiation_reaction = SEOBNRv4PublishedFlux(
        lmax=8,
        spin_truncation="generic-leading-ell8",
        retain_pan_cubic_spin=True,
    )
    mass1, mass2 = _tensor(0.6), _tensor(0.4)
    mass_fraction = eob_final_mass_fraction(mass1, mass2, zero, zero)
    final_spin = hofmann_final_spin(
        mass1, mass2, zero, zero, fit_order=(3, 4)
    )
    qnm_real, qnm_damping = signed_qnm_220_table_convention(final_spin)
    sigma = torch.complex(
        qnm_real / mass_fraction,
        -qnm_damping / mass_fraction,
    )
    return {
        "nu": nu,
        "chi": zero,
        "hamiltonian": hamiltonian,
        "radiation_reaction": radiation_reaction,
        "mass_fraction": mass_fraction,
        "final_spin": final_spin,
        "sigma": sigma,
    }


def _coefficients(attached):
    coefficients = attached.calibration.coefficients
    return {
        "amplitude": [_scalar(value) for value in coefficients.amplitude],
        "phase": [_scalar(value) for value in coefficients.phase],
    }


def _relative_l2(mode, reference, mask):
    """Return same-index unshifted raw and amplitude relative L2 metrics."""
    if not bool(mask.any()):
        return {"available": False, "sample_count": 0}
    clean = mode[mask]
    public = reference[mask]
    public_norm = float(np.linalg.norm(public))
    public_amplitude_norm = float(np.linalg.norm(np.abs(public)))
    return {
        "available": True,
        "sample_count": int(mask.sum()),
        "raw_relative_l2": (
            float(np.linalg.norm(clean - public) / public_norm)
            if public_norm
            else None
        ),
        "amplitude_relative_l2": (
            float(np.linalg.norm(np.abs(clean) - np.abs(public)) / public_amplitude_norm)
            if public_amplitude_norm
            else None
        ),
        "time_shift_applied_M": 0.0,
        "phase_shift_applied_rad": 0.0,
    }


def _attachment_summary(attached, public_mode, output_time):
    """Summarize event, boundary, NQC, and unshifted post-boundary residual."""
    mode = (
        attached.mode.detach().cpu().numpy().astype(np.complex128) * MODE_SCALE
    )
    boundary_time = _scalar(attached.boundary_time)
    boundary_index = int(round(_scalar(attached.boundary_index)))
    boundary_mask = (
        (output_time >= boundary_time)
        & np.isfinite(mode.real)
        & np.isfinite(public_mode.real)
        & (np.abs(public_mode) > np.finfo(np.float64).tiny)
    )
    peak = attached.peak
    return {
        "peak": {
            "time_M": _scalar(peak.time),
            "frequency": _scalar(peak.frequency),
            "curvature": _scalar(peak.curvature),
            "sample_index": int(round(_scalar(peak.sample_index))),
        },
        "match": {
            "time_M": _scalar(attached.match_time),
            "phase_rad": _scalar(attached.match_phase),
        },
        "boundary": {
            "time_M": boundary_time,
            "sample_index": boundary_index,
            "phase_rad": _scalar(attached.boundary_phase),
            "frequency": _scalar(attached.boundary_frequency),
            "amplitude_unscaled": _scalar(attached.boundary_amplitude),
        },
        "internal_step_M": _scalar(attached.internal_step),
        "coefficients": _coefficients(attached),
        "post_boundary_unshifted": _relative_l2(
            mode, public_mode, boundary_mask
        ),
        "post_boundary_grid": {
            "definition": "output_time_M >= boundary_time_M, same indices, no time or phase shift",
            "first_sample_M": (
                float(output_time[boundary_mask][0]) if boundary_mask.any() else None
            ),
            "last_sample_M": (
                float(output_time[boundary_mask][-1]) if boundary_mask.any() else None
            ),
        },
    }


def _attachment_differences(analytic, numerical):
    """Report numerical-frequency changes relative to the analytic attachment."""
    def delta(path):
        left, right = analytic, numerical
        for key in path:
            left, right = left[key], right[key]
        return _scalar(right) - _scalar(left)

    amplitude_delta = [
        numerical["coefficients"]["amplitude"][i]
        - analytic["coefficients"]["amplitude"][i]
        for i in range(3)
    ]
    phase_delta = [
        numerical["coefficients"]["phase"][i]
        - analytic["coefficients"]["phase"][i]
        for i in range(2)
    ]
    return {
        "peak_time_M": delta(("peak", "time_M")),
        "peak_frequency": delta(("peak", "frequency")),
        "peak_curvature": delta(("peak", "curvature")),
        "peak_sample_index": (
            numerical["peak"]["sample_index"] - analytic["peak"]["sample_index"]
        ),
        "match_time_M": delta(("match", "time_M")),
        "match_phase_rad": delta(("match", "phase_rad")),
        "boundary_time_M": delta(("boundary", "time_M")),
        "boundary_sample_index": (
            numerical["boundary"]["sample_index"]
            - analytic["boundary"]["sample_index"]
        ),
        "boundary_phase_rad": delta(("boundary", "phase_rad")),
        "boundary_frequency": delta(("boundary", "frequency")),
        "boundary_amplitude_unscaled": delta(("boundary", "amplitude_unscaled")),
        "coefficients": {
            "amplitude": amplitude_delta,
            "phase": phase_delta,
            "max_abs": float(max(
                [abs(value) for value in amplitude_delta + phase_delta]
            )),
        },
        "post_boundary_unshifted": {
            "raw_relative_l2": (
                numerical["post_boundary_unshifted"]["raw_relative_l2"]
                - analytic["post_boundary_unshifted"]["raw_relative_l2"]
            ),
            "amplitude_relative_l2": (
                numerical["post_boundary_unshifted"]["amplitude_relative_l2"]
                - analytic["post_boundary_unshifted"]["amplitude_relative_l2"]
            ),
        },
    }


def _public_fixture_case(label, physics):
    """Compare analytic and bounded numerical omega on one public fixture."""
    data = _load_public_fixture(label)
    postpeak = data["postpeak_state"]
    post_time = torch.from_numpy(postpeak[0].copy())
    state = torch.from_numpy(postpeak[1:].T.copy())
    output_grid = torch.arange(data["mode22"].size, dtype=DTYPE) * (
        data["dt_s"] / MASS_SECONDS
    )
    ham = physics["hamiltonian"]
    flux = physics["radiation_reaction"]
    # Both attachment calls receive these same published carrier arrays.  The
    # only changed input is the orbital-frequency series under attribution.
    base_mode, base_frequency = published_mode22_and_frequency(
        state,
        ham,
        flux,
        **PUBLIC_CHOICES,
    )
    analytic_frequency = flux.from_hamiltonian(state, ham)["omega"]
    numerical_report = numerical_eob_conservative_derivatives(
        ham,
        state[..., 0],
        state[..., 2],
        state[..., 3],
        initial_step=INITIAL_STEP,
        max_iterations=DERIVATIVE_ITERATIONS,
        return_info=True,
    )
    numerical_frequency = numerical_report.omega
    difference = (numerical_frequency - analytic_frequency).detach().cpu().numpy()
    input_time = postpeak[0]
    analytic_attachment = attach_seobnrv4_22_numerical_compatibility(
        post_time,
        state,
        analytic_frequency,
        base_mode,
        base_frequency,
        physics["nu"],
        physics["chi"],
        physics["sigma"],
        output_grid,
    )
    numerical_attachment = attach_seobnrv4_22_numerical_compatibility(
        post_time,
        state,
        numerical_frequency,
        base_mode,
        base_frequency,
        physics["nu"],
        physics["chi"],
        physics["sigma"],
        output_grid,
    )
    public_mode = data["mode22"]
    output_np = output_grid.detach().cpu().numpy()
    analytic_summary = _attachment_summary(
        analytic_attachment, public_mode, output_np
    )
    numerical_summary = _attachment_summary(
        numerical_attachment, public_mode, output_np
    )
    match_time = analytic_summary["match"]["time_M"]
    near_match = np.abs(input_time - match_time) <= 1.0
    boundary_sample = analytic_summary["boundary"]["sample_index"]
    total_evaluations = int(
        numerical_report.x.evaluations.sum()
        + numerical_report.p_x.evaluations.sum()
        + numerical_report.p_y.evaluations.sum()
    )
    return {
        "npz_path": str(data["path"]),
        "npz_sha256": data["sha256"],
        "npz_size_bytes": data["size_bytes"],
        "raw_arrays": {
            "keys": ["state", "postpeak_state", "mode22", "epoch_s", "dt_s"],
            "postpeak_state_shape": list(postpeak.shape),
            "mode22_shape": list(public_mode.shape),
            "loaded_arrays_copied": True,
        },
        "epoch_s": data["epoch_s"],
        "dt_s": data["dt_s"],
        "input_time_support_M": [float(input_time[0]), float(input_time[-1])],
        "output_grid": {
            "same_tensor_passed_to_both_attachments": True,
            "relative_M_definition": "arange(mode22.size) * dt_s / (50*MTSUN_SI)",
            "relative_M_support": [float(output_np[0]), float(output_np[-1])],
            "absolute_public_seconds_definition": "epoch_s + relative_M * (50*MTSUN_SI)",
            "absolute_public_seconds_support": [
                float(data["epoch_s"] + output_np[0] * MASS_SECONDS),
                float(data["epoch_s"] + output_np[-1] * MASS_SECONDS),
            ],
        },
        "physics": {
            "hamiltonian": "SEOBNRv4Hamiltonian(nu=0.24, chiS=0, chiA=0, delta=0.2)",
            "flux": "SEOBNRv4PublishedFlux(lmax=8, spin_truncation='generic-leading-ell8', retain_pan_cubic_spin=True)",
            "published_choices": dict(PUBLIC_CHOICES),
            "mass_fraction": _scalar(physics["mass_fraction"]),
            "final_spin": _scalar(physics["final_spin"]),
            "final_spin_fit_order": [3, 4],
            "sigma_real_imag": [
                _scalar(physics["sigma"].real),
                _scalar(physics["sigma"].imag),
            ],
            "mass_distance_scale": MODE_SCALE,
            "mass_distance_scale_formula": (
                "50*lal.MTSUN_SI*lal.C_SI/(100e6*lal.PC_SI)"
            ),
        },
        "finite_difference": {
            "initial_step": INITIAL_STEP,
            "max_iterations": DERIVATIVE_ITERATIONS,
            "total_function_evaluations": total_evaluations,
            "evaluations_per_partial": {
                "x": {
                    "minimum": int(numerical_report.x.evaluations.min()),
                    "maximum": int(numerical_report.x.evaluations.max()),
                },
                "p_x": {
                    "minimum": int(numerical_report.p_x.evaluations.min()),
                    "maximum": int(numerical_report.p_x.evaluations.max()),
                },
                "p_y": {
                    "minimum": int(numerical_report.p_y.evaluations.min()),
                    "maximum": int(numerical_report.p_y.evaluations.max()),
                },
            },
            "selected_step_range": {
                "x": [
                    float(numerical_report.x.step.min()),
                    float(numerical_report.x.step.max()),
                ],
                "p_x": [
                    float(numerical_report.p_x.step.min()),
                    float(numerical_report.p_x.step.max()),
                ],
                "p_y": [
                    float(numerical_report.p_y.step.min()),
                    float(numerical_report.p_y.step.max()),
                ],
            },
        },
        "orbital_frequency_discrepancy": {
            "global_max_abs": float(np.max(np.abs(difference))),
            "global_rms": float(np.sqrt(np.mean(difference**2))),
            "near_match_window_M": 1.0,
            "near_match_max_abs": float(np.max(np.abs(difference[near_match]))),
            "near_match_rms": float(np.sqrt(np.mean(difference[near_match] ** 2))),
            "analytic_boundary_sample_abs": float(abs(difference[boundary_sample])),
            "analytic_boundary_sample_signed": float(difference[boundary_sample]),
            "max_abs_sample_index": int(np.argmax(np.abs(difference))),
            "max_abs_sample_time_M": float(input_time[np.argmax(np.abs(difference))]),
        },
        "attachment_results": {
            "analytic_frequency": analytic_summary,
            "numerical_frequency": numerical_summary,
            "numerical_minus_analytic": _attachment_differences(
                analytic_summary, numerical_summary
            ),
        },
        "attribution": (
            "The analytic and numerical orbital-frequency series are fed separately "
            "to the same clean published base_mode/base_frequency, remnant sigma, "
            "target choices, and output grid. Residuals are reported unshifted and "
            "are not used to select a derivative policy."
        ),
        "scope": "bounded postpeak pointwise batch only; no trajectory integration or full inspiral validation",
    }


def _analytic_report():
    x = _tensor([-0.8, 0.2, 1.1])
    h = _tensor(1.0e-3)

    polynomial = lambda value: value.pow(4) - 0.7 * value.pow(3) + 0.2 * value.square() + 2.0 * value
    polynomial_result = adaptive_central_derivative(
        polynomial,
        x,
        initial_step=1.0e-4,
        max_iterations=1,
        return_info=True,
    )
    polynomial_exact = 4.0 * x.pow(3) - 2.1 * x.square() + 0.4 * x + 2.0

    sine_actual = central_quartic(torch.sin, x, h)
    sine_exact = torch.cos(x)
    sine_result = adaptive_central_derivative(
        torch.sin,
        x,
        initial_step=1.0e-4,
        max_iterations=1,
        return_info=True,
    )

    return {
        "polynomial_degree": 4,
        "polynomial_max_abs_error": float((polynomial_result.derivative - polynomial_exact).abs().max()),
        "polynomial_steps": [float(value) for value in polynomial_result.step],
        "polynomial_evaluations": [int(value) for value in polynomial_result.evaluations],
        "sine_fixed_quartic_max_abs_error": float((sine_actual - sine_exact).abs().max()),
        "sine_adaptive_max_abs_error": float((sine_result.derivative - sine_exact).abs().max()),
        "sine_adaptive_steps": [float(value) for value in sine_result.step],
        "sine_adaptive_evaluations": [int(value) for value in sine_result.evaluations],
    }


def _eob_case(name, hamiltonian, radius, radial, angular):
    report = numerical_eob_conservative_derivatives(
        hamiltonian,
        radius,
        radial,
        angular,
        initial_step=1.0e-4,
        max_iterations=1,
        return_info=True,
    )
    analytic = hamiltonian.equations_of_motion_conservative(radius, radial, angular)
    actual = report.field
    errors = [float((value - reference).abs().max()) for value, reference in zip(actual, analytic)]
    return {
        "name": name,
        "state": {
            "r": [float(value) for value in radius],
            "pr_star": [float(value) for value in radial],
            "pphi": [float(value) for value in angular],
        },
        "field_components": ["dr_dt", "omega", "dpr_star_dt"],
        "analytic_path_max_abs_discrepancy": errors,
        "max_abs_discrepancy": max(errors),
        "finite_difference_evaluations_per_partial": {
            "x": [int(value) for value in report.x.evaluations],
            "p_x": [int(value) for value in report.p_x.evaluations],
            "p_y": [int(value) for value in report.p_y.evaluations],
        },
        "selected_steps": {
            "x": [float(value) for value in report.x.step],
            "p_x": [float(value) for value in report.p_x.step],
            "p_y": [float(value) for value in report.p_y.step],
        },
        "total_function_evaluations": int(
            report.x.evaluations.sum()
            + report.p_x.evaluations.sum()
            + report.p_y.evaluations.sum()
        ),
    }


def build_report():
    nonspinning = _eob_case(
        "EOBHamiltonian nonspinning",
        EOBHamiltonian(_tensor(0.24), _tensor(0.0), _tensor(0.0)),
        _tensor([6.0, 9.0]),
        _tensor([-0.15, 0.10]),
        _tensor([3.0, 4.0]),
    )
    spinning = _eob_case(
        "SEOBNRv4Hamiltonian aligned spin",
        SEOBNRv4Hamiltonian(_tensor(0.16), _tensor(0.70), _tensor(-0.30)),
        _tensor([7.0, 11.0]),
        _tensor([-0.08, 0.12]),
        _tensor([3.2, 4.1]),
    )
    public_physics = _public_physics()
    return {
        "method": "clean Torch central quartic finite differences",
        "initial_step": INITIAL_STEP,
        "max_iterations": DERIVATIVE_ITERATIONS,
        "normalization": "full real Hamiltonian divided by nu; effective-Hamiltonian limit at nu=0",
        "stencil": {
            "quadratic": "(f(x+h)-f(x-h))/(2h)",
            "quartic": "(f(x-h)-8f(x-h/2)+8f(x+h/2)-f(x+h))/(6h)",
            "adaptation": "one locally derived a/h+b*h^2 update, factor-of-ten cap",
        },
        "analytic_checks": _analytic_report(),
        "eob_pointwise_checks": [nonspinning, spinning],
        "public_postpeak_fixture_checks": [
            _public_fixture_case(label, public_physics) for label in PUBLIC_LABELS
        ],
        "source_sha256": _source_hashes(),
        "optional_gsl_comparison": {
            "status": "not run",
            "reason": "The clean diagnostic has no GSL runtime dependency; public GSL comparison is optional.",
        },
        "scope": "pointwise, bounded postpeak fixture, and short smooth-function checks only; no trajectory integration or full inspiral validation",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="write JSON to this path")
    args = parser.parse_args()
    report = build_report()
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()

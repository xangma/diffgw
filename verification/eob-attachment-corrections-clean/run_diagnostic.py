"""Short public-fixture diagnostic for the clean numerical attachment path.

This script consumes only the two permitted public state/mode fixtures and
recomputes their clean Hamiltonian, flux, waveform, remnant, QNM, and
numerical-attachment quantities.  It does not evolve a trajectory or import a
reference implementation.  The resulting JSON is a bounded component
diagnostic, not a production waveform gate.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from torchwave.approximants.eob.numerical_attachment import (
    attach_seobnrv4_22_numerical_compatibility,
    seobnrv4_ringdown_support_samples,
)
from torchwave.approximants.eob.published_waveform import published_mode22_and_frequency
from torchwave.approximants.eob.remnant import (
    eob_final_mass_fraction,
    hofmann_final_spin,
)
from torchwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux
from torchwave.approximants.eob.seobnrv4_hamiltonian import SEOBNRv4Hamiltonian
from torchwave.approximants.kerr_qnm_table_convention import signed_qnm_220_table_convention
from torchwave.constants import C_SI


MTSUN_S = 4.925490947641267e-6
TOTAL_MASS = 50.0
MSEC = TOTAL_MASS * MTSUN_S
DISTANCE_MPC = 100.0
MPC_SI = 3.0856775814913673e22
MODE_SCALE = MSEC * C_SI / (DISTANCE_MPC * MPC_SI)
DATA_ROOT = Path("/tmp")
REPORT = Path(__file__).with_name("diagnostic.json")
LABELS = ("dt_0p5M", "dt_1_4096s")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _frequency_rms(
    mode: np.ndarray, reference: np.ndarray, time: np.ndarray, mask: np.ndarray
) -> float:
    keep = (
        mask[1:]
        & mask[:-1]
        & (np.abs(mode[1:]) > 0)
        & (np.abs(mode[:-1]) > 0)
        & (np.abs(reference[1:]) > 0)
        & (np.abs(reference[:-1]) > 0)
    )
    mode_frequency = -np.angle(mode[1:] * np.conj(mode[:-1])) / np.diff(time)
    reference_frequency = -np.angle(reference[1:] * np.conj(reference[:-1])) / np.diff(time)
    return float(np.sqrt(np.mean((mode_frequency[keep] - reference_frequency[keep]) ** 2)))


def _fixture(label: str, ham, flux, nu, chi, sigma) -> dict:
    path = DATA_ROOT / f"torchwave-public-eob-{label}.npz"
    with np.load(path) as loaded:
        post = np.asarray(loaded["postpeak_state"], dtype=np.float64).copy()
        reference = np.asarray(loaded["mode22"], dtype=np.complex128).copy()
        dt_s = float(np.asarray(loaded["dt_s"]).item())
        state_shape = list(np.asarray(loaded["state"]).shape)
        post_shape = list(post.shape)
    time = torch.from_numpy(post[0])
    state = torch.from_numpy(post[1:].T)
    orbital_frequency = flux.from_hamiltonian(state, ham)["omega"]
    base_mode, base_frequency = published_mode22_and_frequency(
        state,
        ham,
        flux,
        spin_v6="none",
        include_pan_v8=True,
        include_positive_chi_540=True,
    )
    output_time = torch.arange(reference.size, dtype=torch.float64) * (dt_s / MSEC)
    attached = attach_seobnrv4_22_numerical_compatibility(
        time,
        state,
        orbital_frequency,
        base_mode,
        base_frequency,
        nu,
        chi,
        sigma,
        output_time,
    )
    mode = attached.mode.detach().numpy() * MODE_SCALE
    output = output_time.numpy()
    match = float(attached.match_time)
    postmatch = (output >= match) & (np.abs(reference) > 0) & (np.abs(mode) > 0)
    amplitude_l2 = np.linalg.norm(np.abs(mode[postmatch]) - np.abs(reference[postmatch]))
    amplitude_l2 /= np.linalg.norm(np.abs(reference[postmatch]))
    complex_l2 = np.linalg.norm(mode[postmatch] - reference[postmatch])
    complex_l2 /= np.linalg.norm(reference[postmatch])
    damping = -float(sigma.imag)
    dt_high = float(attached.internal_step)
    support_count = int(seobnrv4_ringdown_support_samples(
        torch.tensor(damping, dtype=torch.float64),
        torch.tensor(dt_high, dtype=torch.float64),
    ))
    nonzero = np.flatnonzero(np.abs(mode) > 0)
    public_nonzero = np.flatnonzero(np.abs(reference) > 0)
    return {
        "fixture_sha256": _sha256(path),
        "state_shape": state_shape,
        "postpeak_state_shape": post_shape,
        "output_length": int(reference.size),
        "input_time_M": [float(time[0]), float(time[-1])],
        "output_time_M": [float(output[0]), float(output[-1])],
        "match_time_M": match,
        "boundary_time_M": float(attached.boundary_time),
        "boundary_index": int(attached.boundary_index),
        "internal_step_M": dt_high,
        "damping": damping,
        "ringdown_support_count_high_samples": support_count,
        "clean_last_nonzero_output_index": int(nonzero[-1]),
        "public_last_nonzero_output_index": int(public_nonzero[-1]),
        "postmatch_nonzero_count": int(postmatch.sum()),
        "postmatch_complex_relative_l2": float(complex_l2),
        "postmatch_amplitude_relative_l2": float(amplitude_l2),
        "postmatch_frequency_rms": _frequency_rms(mode, reference, output, postmatch),
    }


def main() -> None:
    dtype = torch.float64
    zero = torch.tensor(0.0, dtype=dtype)
    nu = torch.tensor(0.24, dtype=dtype)
    chi = torch.tensor(0.0, dtype=dtype)
    ham = SEOBNRv4Hamiltonian(nu, zero, zero, delta=torch.tensor(0.2, dtype=dtype))
    flux = SEOBNRv4PublishedFlux(spin_truncation="generic-leading-ell8")
    mass1 = torch.tensor(0.6, dtype=dtype)
    mass2 = torch.tensor(0.4, dtype=dtype)
    mass_fraction = eob_final_mass_fraction(mass1, mass2, zero, zero)
    final_spin = hofmann_final_spin(mass1, mass2, zero, zero, fit_order=(3, 4))
    qnm_frequency, qnm_damping = signed_qnm_220_table_convention(final_spin)
    sigma = torch.complex(qnm_frequency, -qnm_damping) / mass_fraction
    results = {label: _fixture(label, ham, flux, nu, chi, sigma) for label in LABELS}
    report = {
        "status": "bounded_public_fixture_diagnostic",
        "provenance": {
            "inputs": "permitted public state/postpeak/mode/epoch fixtures",
            "implementation": "MIT-only clean TorchWave Hamiltonian, flux, waveform, remnant, QNM, and numerical attachment helpers",
            "primary_reference": "Bohé et al., arXiv:1611.03703v3",
            "trajectory_evolution": False,
            "reference_implementation_imported": False,
        },
        "public_parameters": {
            "m1_msun": 30.0,
            "m2_msun": 20.0,
            "chi1": 0.0,
            "chi2": 0.0,
            "mode": [2, 2],
            "distance_mpc": DISTANCE_MPC,
            "nu": float(nu),
            "chi": float(chi),
            "output_coordinate": "arange(mode22_length) * dt_s / (50 M_sun in seconds)",
        },
        "contract_effects": {
            "phase_rhs": "full unwrapped waveform-phase spline first/second derivatives",
            "shared_scalar_spline": "explicit scalar_output=True for (..., n) ordinates; channel callers retain (..., n, C)",
            "orbital_peak": "strictly negative derivative product moves right; equality moves left",
            "attachment_floor": "greatest input knot <= match; floor zero rejected for backward phase difference",
            "tail_window": "floor(10/(damping*dt_high)) high-grid samples from elapsed zero before output selection",
        },
        "results": results,
        "limitations": [
            "The public fixture comparison is a short component diagnostic and not a native trajectory gate.",
            "The optional amplitude curvature compatibility clamp was not implemented; its paper-derived condition is reported separately to the coordinator.",
        ],
    }
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(REPORT)


if __name__ == "__main__":
    main()

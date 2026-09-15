#!/usr/bin/env python3
"""Recompute clean SEOB NQC matching rows from a captured fine trajectory.

This uses only fields returned by the clean attachment API.  Strides are fixed
to 1, 2, and 4 (no tuning); it measures derivative and matching-system
sensitivity without evolving a trajectory or importing reference code.
"""
import hashlib
import json
import sys
from pathlib import Path

import torch

from torchwave.approximants.eob.nqc import seobnrv4_nqc_targets
from torchwave.approximants.eob.nqc import nqc_22_basis
from torchwave.approximants.eob.numerical_attachment import natural_cubic_spline
from torchwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux
from torchwave.approximants.eob.seobnrv4_hamiltonian import SEOBNRv4Hamiltonian


ROOT = Path(__file__).resolve().parents[1]
CAPTURE = ROOT / "docs/reviews/2026-09-13-parity-review/verification/eob-fine-stage-replay-clean/fine-attachment-intermediates.npz"
OUTPUT = ROOT / "docs/reviews/2026-09-13-parity-review/verification/eob-fine-stage-replay-clean/nqc-stencil-sensitivity.json"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    import numpy as np

    with np.load(CAPTURE, allow_pickle=False) as data:
        time = torch.from_numpy(data["fine_output_time"]).to(torch.float64)
        amplitude = torch.from_numpy(data["sampled_amplitude"]).to(torch.float64)
        products = torch.from_numpy(data["sampled_products"]).to(torch.float64)
        phase_basis = torch.from_numpy(data["sampled_phase_basis"]).to(torch.float64)
        waveform_phase = torch.from_numpy(data["sampled_waveform_phase"]).to(torch.float64)
        corrected_amplitude = torch.from_numpy(data["sampled_corrected_amplitude"]).to(torch.float64)
        corrected_phase = torch.from_numpy(data["sampled_corrected_phase"]).to(torch.float64)
        state = torch.from_numpy(data["fine_output_state"]).to(torch.float64)
        match = float(np.asarray(data["match_time"]))
    # Recompute the saved rows from the captured state, including omega from
    # the clean Hamiltonian.  This guards against treating copied JSON values
    # as a numerical result and documents the state-column convention.
    nu = torch.tensor(0.24, dtype=torch.float64)
    zero = torch.tensor(0.0, dtype=torch.float64)
    ham = SEOBNRv4Hamiltonian(nu, zero, zero, delta=torch.tensor(0.2, dtype=torch.float64))
    flux = SEOBNRv4PublishedFlux(spin_truncation="generic-leading-ell8")
    omega = flux.from_hamiltonian(state, ham)["omega"]
    recomputed_amplitude_basis, recomputed_phase_basis = nqc_22_basis(
        state[:, 0], state[:, 2], omega
    )
    recomputed_products = amplitude[:, None] * recomputed_amplitude_basis
    values = torch.cat((amplitude[:, None], products, phase_basis), dim=1)
    target = torch.stack(tuple(seobnrv4_nqc_targets(
        torch.tensor(0.24, dtype=torch.float64), torch.tensor(0.0, dtype=torch.float64)
    )))
    rows = {}
    for stride in (1, 2, 4):
        spline = natural_cubic_spline(time[::stride], values[::stride])
        query = torch.tensor([match], dtype=torch.float64)
        value = spline.evaluate(query)[0]
        first = spline.evaluate(query, derivative=1)[0]
        second = spline.evaluate(query, derivative=2)[0]
        phase_spline = natural_cubic_spline(time[::stride], waveform_phase[::stride])
        phase_value = phase_spline.evaluate(query)[0]
        phase_first = phase_spline.evaluate(query, derivative=1)[0]
        phase_second = phase_spline.evaluate(query, derivative=2)[0]
        # Capture stores phi in h=A*exp(-i phi); numerical_attachment defines
        # positive omega22 as d(phi)/d(tau), i.e. the phase derivatives here.
        waveform_targets = torch.stack((value[0], first[0], second[0],
                                         phase_first, phase_second))
        corrected_amp_spline = natural_cubic_spline(time[::stride], corrected_amplitude[::stride])
        corrected_phase_spline = natural_cubic_spline(time[::stride], corrected_phase[::stride])
        corrected_amp = corrected_amp_spline.evaluate(query)[0]
        corrected_amp_first = corrected_amp_spline.evaluate(query, derivative=1)[0]
        corrected_amp_second = corrected_amp_spline.evaluate(query, derivative=2)[0]
        corrected_phase_first = corrected_phase_spline.evaluate(query, derivative=1)[0]
        corrected_phase_second = corrected_phase_spline.evaluate(query, derivative=2)[0]
        corrected_targets = torch.stack((corrected_amp, corrected_amp_first,
                                          corrected_amp_second, corrected_phase_first,
                                          corrected_phase_second))
        target_vec = target.numpy().tolist()
        rows[str(stride)] = {
            "sample_count": int(time[::stride].numel()),
            "step_M": float((time[1] - time[0]) * stride),
            "value": value.tolist(),
            "first_derivative": first.tolist(),
            "second_derivative": second.tolist(),
            "basis_first_derivative": first[4:].tolist(),
            "basis_second_derivative": second[4:].tolist(),
            "waveform_phase": float(phase_value),
            "waveform_phase_first_derivative": float(phase_first),
            "waveform_phase_second_derivative": float(phase_second),
            "waveform_targets": waveform_targets.tolist(),
            "target_vector": target_vec,
            "target_residual": (waveform_targets - target).tolist(),
            "corrected_waveform_targets": corrected_targets.tolist(),
            "corrected_target_residual": (corrected_targets - target).tolist(),
        }
    baseline = rows["1"]
    for row in rows.values():
        row["max_abs_basis_first_delta_vs_stride1"] = max(
            abs(a - b) for a, b in zip(row["first_derivative"], baseline["first_derivative"])
        )
        row["max_abs_basis_second_delta_vs_stride1"] = max(
            abs(a - b) for a, b in zip(row["second_derivative"], baseline["second_derivative"])
        )
    report = {
        "status": "clean_room_nqc_stencil_sensitivity",
        "provenance": {
            "capture_sha256": sha256(CAPTURE),
            "script_sha256": sha256(Path(__file__).resolve()),
            "sys_executable": sys.executable,
            "argv": sys.argv,
            "dependencies": {"torch": torch.__version__, "numpy": np.__version__},
            "trajectory_evolution": False,
            "clean_module_sha256": {
                str(path.relative_to(ROOT)): sha256(path) for path in (
                    ROOT / "src/torchwave/approximants/eob/nqc.py",
                    ROOT / "src/torchwave/approximants/eob/numerical_attachment.py",
                    ROOT / "src/torchwave/approximants/eob/seobnrv4_hamiltonian.py",
                    ROOT / "src/torchwave/approximants/eob/seobnrv4_flux.py",
                )
            },
            "reference_implementation_imported": False,
            "public_target_api": "unavailable; target_vector is the clean published-equation evaluation only",
            "state_columns": "(r, phi, p_r_star, p_phi); omega recomputed by clean Hamiltonian",
        },
        "fixed_stencils": "natural cubic full-series splines at strides 1, 2, and 4",
        "match_time_M": match,
        "nu": 0.24,
        "chi": 0.0,
        "target_names": ["A", "A_prime", "A_second", "omega", "omega_prime"],
        "recomputed_row_checks": {
            "max_abs_products_error": float((recomputed_products - products).abs().max()),
            "max_abs_phase_basis_error": float((recomputed_phase_basis - phase_basis).abs().max()),
            "omega_min": float(omega.min()),
            "omega_max": float(omega.max()),
        },
        "rows": rows,
        "input_before_after": {
            "before": "uncorrected captured waveform targets and residuals",
            "after": "captured NQC-corrected waveform targets and residuals",
        },
        "interpretation": "uncorrected waveform_targets are the pre-NQC values; their residuals are expected inputs to the NQC solve. corrected_waveform_targets use the captured sampled_corrected_amplitude and sampled_corrected_phase. Both are (A, A', A'', phase', phase''); numerical_attachment defines phase' and phase'' as positive GW frequency and derivative for h=A*exp(-i phase). Basis derivative deltas are reported separately. Fixed refinement quantifies sampling sensitivity without attributing any residual to a particular model component.",
    }
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

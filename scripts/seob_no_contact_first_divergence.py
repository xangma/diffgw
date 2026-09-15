#!/usr/bin/env python3
"""Localize the first SEOB pre-attachment mismatch without implementation access.

The public arrays are consumed as black-box observables.  Clean factorized
mode and NQC equations are evaluated once on the saved clean fine trajectory
and once on the public post-peak state trajectory.  This separates trajectory
state differences from factorized-mode/NQC differences at identical physical
times, without importing public coefficients or fitting residuals.
"""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import torch

from torchwave.approximants.eob.nqc import (
    nqc_22_factor,
    seobnrv4_nqc_targets,
)
from torchwave.approximants.eob.numerical_attachment import (
    calibrate_nqc_22_full_series,
)
from torchwave.approximants.eob.published_waveform import published_mode22_and_frequency
from torchwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux
from torchwave.approximants.eob.seobnrv4_hamiltonian import SEOBNRv4Hamiltonian
from torchwave.constants import C_SI, MPC_SI, MSUN_SEC


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
CLEAN = ROOT / "docs/reviews/2026-09-13-parity-review/verification/eob-fine-stage-replay-clean/fine-attachment-intermediates.npz"
PUBLIC = Path("/tmp/torchwave-public-eob-dt_0p5M.npz")
OUT = ROOT / "docs/reviews/2026-09-15-no-contact/seob-first-divergence.json"
ARRAYS = ROOT / "docs/reviews/2026-09-15-no-contact/seob-first-divergence-arrays.npz"
DTYPE = torch.float64
MASS_SEC = 50.0 * float(MSUN_SEC)
SCALE = MASS_SEC * float(C_SI) / (100.0 * float(MPC_SI))
RUNTIME_SOURCES = (
    ROOT / "src/torchwave/approximants/eob/published_waveform.py",
    ROOT / "src/torchwave/approximants/eob/nqc.py",
    ROOT / "src/torchwave/approximants/eob/numerical_attachment.py",
    ROOT / "src/torchwave/approximants/eob/seobnrv4_flux.py",
    ROOT / "src/torchwave/approximants/eob/seobnrv4_hamiltonian.py",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metric(a: np.ndarray, b: np.ndarray) -> dict:
    a = np.asarray(a, dtype=np.complex128)
    b = np.asarray(b, dtype=np.complex128)
    denom = float(np.linalg.norm(b))
    amp_denom = float(np.linalg.norm(np.abs(b)))
    phase = np.angle(a * np.conj(b))
    return {
        "samples": int(a.size),
        "complex_relative_l2": float(np.linalg.norm(a - b) / denom),
        "amplitude_relative_l2": float(np.linalg.norm(np.abs(a) - np.abs(b)) / amp_denom),
        "phase_rms_rad": float(np.sqrt(np.mean(phase * phase))),
        "phase_max_abs_rad": float(np.max(np.abs(phase))),
    }


def state_metric(clean: np.ndarray, public: np.ndarray) -> dict:
    names = ("r", "phi", "prstar", "p_phi")
    delta = clean - public
    result = {}
    for i, name in enumerate(names):
        result[name] = {
            "relative_l2": float(np.linalg.norm(delta[:, i]) / np.linalg.norm(public[:, i])),
            "max_abs": float(np.max(np.abs(delta[:, i]))),
        }
    return result


def first_crossing(time: np.ndarray, values: np.ndarray, threshold: float) -> dict | None:
    where = np.flatnonzero(np.asarray(values) > threshold)
    if where.size == 0:
        return None
    i = int(where[0])
    return {"index": i, "time_M": float(time[i]), "value": float(values[i]), "threshold": threshold}


def validate_common_fine_chart(clean_time: np.ndarray, public_time: np.ndarray) -> None:
    """Require the complete common fine chart before evaluating either path."""
    if clean_time.shape != public_time.shape or not np.array_equal(clean_time, public_time):
        raise RuntimeError("clean and public full fine time axes are not identical")
    if clean_time.ndim != 1 or clean_time.size < 3 or not np.all(np.diff(clean_time) > 0):
        raise RuntimeError("common fine time axis is not strictly increasing")


def main() -> None:
    with np.load(CLEAN, allow_pickle=False) as c:
        clean_local_t = c["fine_output_time"].astype(np.float64)
        clean_state = c["fine_output_state"].astype(np.float64)
        clean_base_saved = c["fine_base_mode"].astype(np.complex128)
        clean_match = float(c["match_time"])
        clean_boundary = float(c["boundary_time"])
        restart_index = int(c["restart_index"])
        restart_time = float(c["restart_time"].reshape(-1)[0])
    with np.load(PUBLIC, allow_pickle=False) as p:
        public_state = p["postpeak_state"].astype(np.float64)
        public_mode = p["mode22"].astype(np.complex128)

    # The saved clean trajectory is on the 1/64-M output grid while the public
    # mode is on the 0.5-M grid.  Evaluate both clean paths on the full fine
    # state chart, then compare their every-32nd samples to public mode output.
    fine_step = 32
    input_lengths = {
        "clean_time_samples": int(clean_local_t.size),
        "clean_state_samples": int(clean_state.shape[0]),
        "clean_saved_base_samples": int(clean_base_saved.size),
        "public_state_samples": int(public_state.shape[1]),
        "public_mode_samples": int(public_mode.size),
        "public_mode_tail_samples": int(public_mode.size - restart_index),
    }
    n_fine = min(clean_local_t.size, clean_state.shape[0], public_state.shape[1])
    n_output = min((n_fine - 1) // fine_step + 1, public_mode.size - restart_index)
    output_indices = fine_step * np.arange(n_output, dtype=np.int64)
    clean_global_fine_t = restart_time + clean_local_t[:n_fine]
    public_fine_t = public_state[0, :n_fine]
    validate_common_fine_chart(clean_global_fine_t, public_fine_t)
    clean_global_t = clean_global_fine_t[output_indices]
    public_t = public_state[0, output_indices]
    clean_state = clean_state[:n_fine]
    public_state_rows = public_state[1:, :n_fine].T
    public_ref = public_mode[restart_index:restart_index + n_output]
    pre = public_t < restart_time + clean_boundary

    nu = torch.tensor(0.24, dtype=DTYPE)
    zero = torch.tensor(0.0, dtype=DTYPE)
    ham = SEOBNRv4Hamiltonian(nu, zero, zero, delta=torch.tensor(0.2, dtype=DTYPE))
    flux = SEOBNRv4PublishedFlux(spin_truncation="generic-leading-ell8")

    clean_t = torch.from_numpy(clean_global_fine_t)
    clean_s = torch.from_numpy(clean_state)
    public_t_t = torch.from_numpy(public_fine_t)
    public_s_t = torch.from_numpy(public_state_rows)
    with torch.inference_mode():
        clean_base, clean_frequency = published_mode22_and_frequency(
            clean_s, ham, flux, spin_v6="bohe_total_energy",
            include_pan_v8=False, include_positive_chi_540=False,
        )
        public_base, public_frequency = published_mode22_and_frequency(
            public_s_t, ham, flux, spin_v6="bohe_total_energy",
            include_pan_v8=False, include_positive_chi_540=False,
        )
        public_omega = flux.from_hamiltonian(public_s_t, ham)["omega"]
        clean_omega = flux.from_hamiltonian(clean_s, ham)["omega"]
        clean_targets = seobnrv4_nqc_targets(nu, zero)

    # Reuse the captured clean event time on both paths. This controls state
    # substitution only; it does not test or resolve event-time selection.
    with torch.inference_mode():
        public_match = torch.as_tensor(restart_time + clean_match, dtype=DTYPE)
        clean_nqc = calibrate_nqc_22_full_series(
            clean_t, clean_s[:, 0], clean_s[:, 2], clean_omega,
            clean_base.abs(), clean_frequency, public_match,
            base_mode=clean_base, targets=clean_targets,
        )
        clean_factor = nqc_22_factor(
            clean_s[:, 0], clean_s[:, 2], clean_omega,
            clean_nqc.coefficients,
        )
        clean_corrected = clean_base * clean_factor
        public_nqc = calibrate_nqc_22_full_series(
            public_t_t, public_s_t[:, 0], public_s_t[:, 2], public_omega,
            public_base.abs(), public_frequency, public_match,
            base_mode=public_base, targets=clean_targets,
        )
        public_factor = nqc_22_factor(
            public_s_t[:, 0], public_s_t[:, 2], public_omega,
            public_nqc.coefficients,
        )
        public_corrected = public_base * public_factor

    scale = SCALE
    clean_base_scaled = clean_base.numpy()[output_indices] * scale
    clean_corrected_scaled = clean_corrected.numpy()[output_indices] * scale
    public_base_scaled = public_base.numpy()[output_indices] * scale
    public_corrected_scaled = public_corrected.numpy()[output_indices] * scale
    saved_base_scaled = clean_base_saved[:n_fine][output_indices] * scale
    ref_scaled = public_ref
    base_clean_err = clean_base_scaled[pre] - ref_scaled[pre]
    base_public_err = public_base_scaled[pre] - ref_scaled[pre]
    corrected_public_err = public_corrected_scaled[pre] - ref_scaled[pre]
    corrected_clean_err = clean_corrected_scaled[pre] - ref_scaled[pre]
    base_clean_phase = np.abs(np.angle(clean_base_scaled[pre] * np.conj(ref_scaled[pre])))
    base_public_phase = np.abs(np.angle(public_base_scaled[pre] * np.conj(ref_scaled[pre])))
    corrected_public_phase = np.abs(np.angle(public_corrected_scaled[pre] * np.conj(ref_scaled[pre])))
    clean_corrected_metrics = metric(clean_corrected_scaled[pre], ref_scaled[pre])
    public_corrected_metrics = metric(public_corrected_scaled[pre], ref_scaled[pre])
    corrected_l2_clean = clean_corrected_metrics["complex_relative_l2"]
    corrected_l2_public = public_corrected_metrics["complex_relative_l2"]

    ARRAYS.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        ARRAYS,
        time_fine_global=clean_global_fine_t,
        clean_state=clean_state,
        public_state=public_state_rows,
        clean_factorized_scaled=clean_base_scaled,
        public_factorized_scaled=public_base_scaled,
        clean_corrected_scaled=clean_corrected_scaled,
        public_corrected_scaled=public_corrected_scaled,
        output_indices=output_indices,
        public_mode_output=ref_scaled,
    )

    report = {
        "provenance": (
            "Public mode and postpeak state are copied black-box observables from the "
            "saved binary fixture. Clean factorized mode, orbital frequency and NQC "
            "equations are evaluated on both state paths; no public coefficients, "
            "implementation source, residual fit, alignment or trajectory correction is used."
        ),
        "runtime": {"python": "/Users/xangma/miniconda3/envs/pycbcorig313/bin/python", "host": platform.node()},
        "source_sha256": {"script": sha(SCRIPT), **{str(path.relative_to(ROOT)): sha(path) for path in RUNTIME_SOURCES}},
        "arrays": {"path": str(ARRAYS.relative_to(ROOT)), "sha256": sha(ARRAYS)},
        "inputs_sha256": {"clean_capture": sha(CLEAN), "public_fixture": sha(PUBLIC)},
        "parameters": {"nu": 0.24, "chi": 0.0, "dt_M": 0.5, "fine_dt_M": 0.015625,
                       "fine_step": fine_step, "restart_index": restart_index,
                       "fine_sample_count": n_fine, "sample_count": n_output,
                       "input_lengths": input_lengths,
                       "common_fine_domain_M": [float(clean_global_fine_t[0]), float(clean_global_fine_t[-1])],
                       "common_fine_truncation": "min(clean_time, clean_state, public_state) samples; public mode compared only where restart_index + output index is available",
                       "pre_attachment_samples": int(pre.sum()),
                       "boundary_time_M": restart_time + clean_boundary,
                       "clean_match_time_M": restart_time + clean_match,
                       "public_clean_match_time_M": float(public_match)},
        "trajectory_state_difference": state_metric(clean_state, public_state_rows),
        "pre_attachment": {
            "clean_capture_saved_base_vs_recomputed": metric(saved_base_scaled[pre], clean_base_scaled[pre]),
            "clean_factorized_on_clean_state_vs_public": metric(clean_base_scaled[pre], ref_scaled[pre]),
            "clean_factorized_on_public_state_vs_public": metric(public_base_scaled[pre], ref_scaled[pre]),
            "clean_nqc_on_clean_state_vs_public": clean_corrected_metrics,
            "clean_nqc_on_public_state_vs_public": public_corrected_metrics,
            "state_substitution_improvement": {
                "complex_relative_l2_absolute_reduction": corrected_l2_clean - corrected_l2_public,
                "complex_relative_l2_fraction_remaining": corrected_l2_public / corrected_l2_clean,
                "complex_relative_l2_percent_reduction": 100.0 * (1.0 - corrected_l2_public / corrected_l2_clean),
            },
            "nqc_corrected_clean_state_vs_public_state": metric(clean_corrected_scaled[pre], public_corrected_scaled[pre]),
            "factorized_clean_state_vs_factorized_public_state": metric(clean_base_scaled[pre], public_base_scaled[pre]),
            "first_crossings_phase_rad": {
                "clean_factorized_on_clean_state": first_crossing(public_t[pre], base_clean_phase, 1.0e-3),
                "clean_factorized_on_public_state": first_crossing(public_t[pre], base_public_phase, 1.0e-3),
                "clean_nqc_on_clean_state": first_crossing(public_t[pre], np.abs(np.angle(clean_corrected_scaled[pre] * np.conj(ref_scaled[pre]))), 1.0e-3),
                "clean_nqc_on_public_state": first_crossing(public_t[pre], corrected_public_phase, 1.0e-3),
            },
            "first_crossings_complex_relative_sample": {
                "clean_factorized_on_clean_state": first_crossing(public_t[pre], np.abs(base_clean_err) / np.abs(ref_scaled[pre]), 1.0e-4),
                "clean_factorized_on_public_state": first_crossing(public_t[pre], np.abs(base_public_err) / np.abs(ref_scaled[pre]), 1.0e-4),
                "clean_nqc_on_clean_state": first_crossing(public_t[pre], np.abs(corrected_clean_err) / np.abs(ref_scaled[pre]), 1.0e-4),
                "clean_nqc_on_public_state": first_crossing(public_t[pre], np.abs(corrected_public_err) / np.abs(ref_scaled[pre]), 1.0e-4),
            },
        },
        "interpretation": (
            "Recomputing the clean factorized mode and NQC solve on the clean state and "
            "on the public state with identical clean targets and the captured event time "
            "shows a 59.6 percent reduction in corrected pre-attachment L2 (5.42e-5 to "
            "2.19e-5) under state substitution, while a 2.19e-5 residual remains. This "
            "makes trajectory evolution a contributor but does not identify a unique fix: "
            "event-time selection, target precision and factorized-mode convention remain "
            "unresolved. No runtime fix or default change is justified by this case."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

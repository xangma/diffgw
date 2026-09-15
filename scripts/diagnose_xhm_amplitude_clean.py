#!/usr/bin/env python3
"""Validate the clean non-reexpanded ``(2,1)`` SPA scale at low frequency.

The public XHM call is used as a black-box validation oracle.  Its returned
one-mode series is copied byte-for-byte into an external NPZ; no public value
is used to infer a coefficient or to fit the clean branch.  The branch is
defined by the paper only for ``q < 40``.  The q=50 row therefore records the
public physical scale and the explicit branch veto.
"""
from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import lal
import lalsimulation as ls
import numpy as np
import torch

from torchwave.approximants.imrphenom_xhm import IMRPhenomXHM
from torchwave.approximants.xhm_amplitude_clean import hhat_lm
from torchwave.approximants.xhm_spa_clean import physical_spa_amplitude_21


DISTANCE_MPC = 500.0
DISTANCE_M = DISTANCE_MPC * 1.0e6 * lal.PC_SI
DISTANCE_SECONDS = DISTANCE_M / lal.C_SI
FREQUENCIES_HZ = np.arange(20.0, 81.0, 5.0, dtype=np.float64)
MODE = (2, 1)
LEGACY_MODES = ((2, 2), (2, 1), (3, 3), (3, 2), (4, 4))
LEGACY_CASES = (
    (40.0, 15.0, 0.3, -0.2),
    (55.0, 11.0, -0.4, 0.2),
    (27.0, 16.0, -0.25, 0.15),
)
LEGACY_FREQUENCIES_HZ = np.arange(20.0, 513.0, dtype=np.float64)
# The same small aligned spins are used at each exact mass ratio.  m2=1 Msun
# keeps the q=10 TaylorT4 ratio positive over this deliberately low-frequency
# grid while retaining a public q=50 control outside the paper branch.
CASES = (
    ("q1p5", 1.5, 1.0, 0.08, 0.03),
    ("q3", 3.0, 1.0, 0.08, 0.03),
    ("q10", 10.0, 1.0, 0.08, 0.03),
    ("q50", 50.0, 1.0, 0.08, 0.03),
)
OUT = Path("docs/reviews/2026-09-13-parity-review/verification/xhm-spa/xhm-spa-public.json")
NPZ = Path("/tmp/torchwave-xhm-spa-public-raw.npz")


def sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def public_mode(
    frequency_hz: np.ndarray,
    m1: float,
    m2: float,
    chi1: float,
    chi2: float,
    mode: tuple[int, int] = MODE,
) -> np.ndarray:
    """Return a public physical Fourier-domain spherical-mode series."""

    vector = lal.CreateREAL8Vector(len(frequency_hz))
    vector.data = frequency_hz
    series = ls.SimIMRPhenomXHMFrequencySequenceOneMode(
        vector,
        m1 * lal.MSUN_SI,
        m2 * lal.MSUN_SI,
        chi1,
        chi2,
        mode[0],
        -mode[1],
        DISTANCE_M,
        0.0,
        20.0,
        lal.CreateDict(),
    )
    # The distance argument makes these physical Fourier-domain strain values
    # (strain/Hz in SI), rather than dimensionless mode amplitudes.
    return np.asarray(series.data.data, dtype=np.complex128).copy()


def public_spheroidal(
    frequency_hz: np.ndarray,
    m1: float,
    m2: float,
    chi1: float,
    chi2: float,
    ell: int,
    emm: int,
    phase: bool = False,
) -> np.ndarray:
    vector = lal.CreateREAL8Vector(len(frequency_hz))
    vector.data = frequency_hz
    function = ls.SimIMRPhenomXHM_SpheroidalPhase if phase else ls.SimIMRPhenomXHM_SpheroidalAmplitude
    series = function(
        vector,
        ell,
        emm,
        m1 * lal.MSUN_SI,
        m2 * lal.MSUN_SI,
        chi1,
        chi2,
        DISTANCE_M,
        0.0,
        20.0,
        lal.CreateDict(),
    )
    return np.asarray(series.data.data).copy()


def native_mode(
    frequency_hz: np.ndarray,
    m1: float,
    m2: float,
    chi1: float,
    chi2: float,
    mode: tuple[int, int],
    experimental: bool,
) -> np.ndarray:
    """Return the existing TorchWave production mode for legacy controls."""

    waveform = IMRPhenomXHM(
        modes=(mode,), experimental_phase_connections=experimental
    )(
        torch.as_tensor(frequency_hz, dtype=torch.float64),
        m1,
        m2,
        chi1,
        chi2,
        DISTANCE_MPC,
        0.7,
        0.0,
        0.0,
        20.0,
        return_unprojected_modes=True,
    )
    return waveform[2][mode].detach().cpu().numpy().astype(np.complex128)


def expanded_e10_amplitude(
    frequency_hz: np.ndarray,
    m1: float,
    m2: float,
    chi1: float,
    chi2: float,
) -> np.ndarray:
    """Evaluate the old re-expanded E10 bracket with its physical prefactor."""

    mass_seconds = (m1 + m2) * lal.MTSUN_SI
    total = m1 + m2
    eta = m1 * m2 / total**2
    delta = (m1 - m2) / total
    chi_s = (chi1 + chi2) / 2.0
    chi_a = (chi1 - chi2) / 2.0
    v = (2.0 * np.pi * frequency_hz * mass_seconds) ** (1.0 / 3.0)
    bracket = hhat_lm(
        MODE,
        torch.as_tensor(v, dtype=torch.float64),
        eta,
        delta,
        chi_s,
        chi_a,
    ).abs().detach().cpu().numpy()
    return (
        mass_seconds**2
        / DISTANCE_SECONDS
        * np.pi
        * np.sqrt(2.0 * eta / 3.0)
        * v**-3.5
        * bracket
    )


def relative_l2(raw_branch: np.ndarray, raw_public: np.ndarray) -> float:
    """Scale-sensitive relative L2 used only as an acceptance-like metric."""

    denominator = max(float(np.linalg.norm(raw_public)), np.finfo(float).tiny)
    return float(np.linalg.norm(raw_branch - raw_public) / denominator)


def main() -> None:
    arrays: dict[str, np.ndarray] = {
        "frequency_hz": FREQUENCIES_HZ,
        "legacy_frequency_hz": LEGACY_FREQUENCIES_HZ,
    }
    legacy_rows: list[dict[str, object]] = []
    spa_rows: list[dict[str, object]] = []

    # Preserve the original public/native controls and both of their raw
    # amplitude metrics.  The L2-normalized pair is explicitly shape-only;
    # the physical pair retains the M^2/D scale and is acceptance-like.
    for case_index, (m1, m2, chi1, chi2) in enumerate(LEGACY_CASES):
        for experimental in (False, True):
            for mode in LEGACY_MODES:
                key = f"legacy_c{case_index}_x{int(experimental)}_{mode[0]}{mode[1]}"
                native = native_mode(
                    LEGACY_FREQUENCIES_HZ,
                    m1,
                    m2,
                    chi1,
                    chi2,
                    mode,
                    experimental,
                )
                public = public_mode(
                    LEGACY_FREQUENCIES_HZ, m1, m2, chi1, chi2, mode
                )
                arrays[f"{key}_native_complex_physical"] = native
                arrays[f"{key}_public_complex_physical"] = public
                native_amplitude = np.abs(native)
                public_amplitude = np.abs(public)
                arrays[f"{key}_native_amplitude_physical"] = native_amplitude
                arrays[f"{key}_public_amplitude_physical"] = public_amplitude
                native_norm = max(float(np.linalg.norm(native)), np.finfo(float).tiny)
                public_norm = max(float(np.linalg.norm(public)), np.finfo(float).tiny)
                native_shape = native / native_norm
                public_shape = public / public_norm
                native_amplitude_shape = native_amplitude / native_norm
                public_amplitude_shape = public_amplitude / public_norm
                legacy_rows.append(
                    {
                        "case": case_index,
                        "experimental_phase_connections": experimental,
                        "mode": list(mode),
                        "native_norm_physical": float(np.linalg.norm(native)),
                        "public_norm_physical": float(np.linalg.norm(public)),
                        "shape_only_normalized_relative_l2": relative_l2(
                            native_shape, public_shape
                        ),
                        "shape_only_normalized_amplitude_relative_l2": relative_l2(
                            native_amplitude_shape, public_amplitude_shape
                        ),
                        "physical_scale_acceptance_like_relative_l2": relative_l2(
                            native, public
                        ),
                        "physical_scale_acceptance_like_amplitude_relative_l2": relative_l2(
                            native_amplitude, public_amplitude
                        ),
                    }
                )

    # Keep the previous spheroidal controls as black-box arrays.  They are
    # recorded separately and are not conflated with spherical-mode parity.
    for case_index, (m1, m2, chi1, chi2) in enumerate(LEGACY_CASES):
        arrays[f"legacy_c{case_index}_sph32_amplitude"] = public_spheroidal(
            LEGACY_FREQUENCIES_HZ, m1, m2, chi1, chi2, 3, -2, False
        )
        arrays[f"legacy_c{case_index}_sph32_phase"] = public_spheroidal(
            LEGACY_FREQUENCIES_HZ, m1, m2, chi1, chi2, 3, -2, True
        )

    for label, q, m2, chi1, chi2 in CASES:
        m1 = q * m2
        public = public_mode(FREQUENCIES_HZ, m1, m2, chi1, chi2, MODE)
        public_amplitude = np.abs(public)
        arrays[f"{label}_public_complex_physical"] = public
        arrays[f"{label}_public_amplitude_physical"] = public_amplitude
        expanded = expanded_e10_amplitude(FREQUENCIES_HZ, m1, m2, chi1, chi2)
        arrays[f"{label}_expanded_e10_amplitude_physical"] = expanded

        row: dict[str, object] = {
            "label": label,
            "q": q,
            "m1_msun": m1,
            "m2_msun": m2,
            "chi1_aligned": chi1,
            "chi2_aligned": chi2,
            "mode": list(MODE),
            "public_amplitude_units": "physical Fourier-domain strain/Hz (SI)",
            "public_amplitude_min": float(public_amplitude.min()),
            "public_amplitude_max": float(public_amplitude.max()),
            "public_amplitude_l2": float(np.linalg.norm(public_amplitude)),
            "public_complex_sha256": hashlib.sha256(public.tobytes()).hexdigest(),
            "expanded_e10_amplitude_units": "physical Fourier-domain strain/Hz (SI)",
            "expanded_e10_amplitude_l2": float(np.linalg.norm(expanded)),
            "expanded_e10_amplitude_sha256": hashlib.sha256(expanded.tobytes()).hexdigest(),
            "expanded_e10_physical_scale_acceptance_like_relative_l2": relative_l2(
                expanded, public_amplitude
            ),
            "expanded_e10_physical_scale_acceptance_like_ratio_at_20hz": float(
                expanded[0] / public_amplitude[0]
            ),
            "expanded_e10_physical_scale_acceptance_like_ratio_at_80hz": float(
                expanded[-1] / public_amplitude[-1]
            ),
            "branch_domain": "q < 40",
        }

        if q < 40.0:
            mass_seconds = (m1 + m2) * lal.MTSUN_SI
            branch = physical_spa_amplitude_21(
                torch.as_tensor(FREQUENCIES_HZ, dtype=torch.float64),
                mass_seconds,
                m1,
                m2,
                chi1,
                chi2,
                DISTANCE_SECONDS,
            ).detach().cpu().numpy()
            arrays[f"{label}_spa_amplitude_physical"] = branch
            # This is deliberately unnormalized: M^2/D remains in both
            # arrays.  The metric is an acceptance-like physical-scale check,
            # not a raw calibration result.
            shape_branch = branch / max(float(np.linalg.norm(branch)), np.finfo(float).tiny)
            shape_public = public_amplitude / max(
                float(np.linalg.norm(public_amplitude)), np.finfo(float).tiny
            )
            row.update(
                {
                    "branch_status": "evaluated",
                    "branch_amplitude_units": "physical Fourier-domain strain/Hz (SI)",
                    "branch_amplitude_l2": float(np.linalg.norm(branch)),
                    "branch_amplitude_sha256": hashlib.sha256(branch.tobytes()).hexdigest(),
                    "physical_scale_acceptance_like_relative_l2": relative_l2(
                        branch, public_amplitude
                    ),
                    "physical_scale_acceptance_like_ratio_at_20hz": float(
                        branch[0] / public_amplitude[0]
                    ),
                    "physical_scale_acceptance_like_ratio_at_80hz": float(
                        branch[-1] / public_amplitude[-1]
                    ),
                    "nonreexpanded_closer_to_public_than_expanded_e10": relative_l2(
                        branch, public_amplitude
                    )
                    < relative_l2(expanded, public_amplitude),
                    "nonreexpanded_minus_expanded_e10_physical_scale_acceptance_like_relative_l2": relative_l2(
                        branch, expanded
                    ),
                    "shape_only_normalized_relative_l2": relative_l2(
                        shape_branch, shape_public
                    ),
                }
            )
        else:
            row.update(
                {
                    "branch_status": "vetoed_q_ge_40",
                    "branch_error": "the non-reexpanded (2,1) branch is defined only for q < 40",
                    "physical_scale_acceptance_like_relative_l2": None,
                    "shape_only_normalized_relative_l2": None,
                }
            )
        spa_rows.append(row)

    np.savez_compressed(NPZ, **arrays)
    metadata = {
        "paper": "arXiv:2001.10914v2",
        "purpose": "bounded low-frequency absolute-scale validation of the clean non-reexpanded (2,1) SPA branch",
        "equations": [
            "Appendix E E1 (global h_lm normalization)",
            "Appendix E E3 (SPA expression retained for (2,1) when q<40)",
            "Appendix E E4-E7 (published TaylorT4 xdot; displayed relative v2, v3, v4, v5 and v7 terms; no printed relative v6 term)",
            "Appendix E E8 (re-expanded SPA normalization)",
            "Appendix E E10=(21) (verified re-expansion target)",
        ],
        "time_domain_sources": [
            "Arun et al., arXiv:0810.5336 Eq. (4.17b), rendered PDF p. 12",
            "Buonanno, Faye and Hinderer, arXiv:1209.6349 Eqs. (4.26) and (4.27a), rendered PDF p. 12",
            "Blanchet, arXiv:1310.1528v6 Eq. (492), H21, rendered PDF p. 214",
        ],
        "mode": list(MODE),
        "legacy_control": {
            "purpose": "preserved public/native comparison from the earlier diagnostic",
            "modes": [list(mode) for mode in LEGACY_MODES],
            "cases": [list(case) for case in LEGACY_CASES],
            "frequency_hz": [
                float(LEGACY_FREQUENCIES_HZ[0]),
                float(LEGACY_FREQUENCIES_HZ[-1]),
                1.0,
            ],
            "metric_labels": {
                "shape_only_normalized_*": "the original L2-normalized comparisons; shape only",
                "physical_scale_acceptance_like_*": "the original unnormalized comparisons; physical M^2/D scale retained and acceptance-like",
            },
        },
        "branch": "non-reexpanded Eq. E3, explicitly limited to q < 40",
        "cases": [
            {
                "label": label,
                "q": q,
                "m1_msun": q * m2,
                "m2_msun": m2,
                "chi1_aligned": chi1,
                "chi2_aligned": chi2,
            }
            for label, q, m2, chi1, chi2 in CASES
        ],
        "frequency_hz": [float(FREQUENCIES_HZ[0]), float(FREQUENCIES_HZ[-1]), 5.0],
        "distance_mpc": DISTANCE_MPC,
        "distance_argument_m": DISTANCE_M,
        "distance_seconds_for_clean_branch": DISTANCE_SECONDS,
        "public_api": "SimIMRPhenomXHMFrequencySequenceOneMode(..., distance_m, ...)->COMPLEX16FrequencySeries",
        "public_output_contract": "distance-corrected physical Fourier-domain strain/Hz; public arrays are not dimensionless",
        "metric_labels": {
            "physical_scale_acceptance_like_*": "raw physical M^2/D scale is retained; acceptance-like validation only, not a raw calibration metric",
            "shape_only_normalized_*": "L2-normalized shape comparison; scale is intentionally removed",
        },
        "coefficient_policy": "public values are validation outputs only; no coefficient is inferred or fitted",
        "comparison_interpretation": "The clean branch and the re-expanded E10 baseline are pure PN expressions, while the public XHM series contains calibrated higher-order phenomenological terms. The raw comparisons describe scale agreement on this grid and do not select the branch or imply coefficient recovery.",
        "npz": str(NPZ),
        "source_hashes": {
            "paper_tex": sha("/tmp/gq2001/hmc2.tex"),
            "clean_spa_helper": sha("src/torchwave/approximants/xhm_spa_clean.py"),
            "script": sha(__file__),
        },
        "raw_npz_sha256": sha(NPZ),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "legacy_rows": legacy_rows,
        "spa_rows": spa_rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(metadata, indent=2) + "\n")
    print(
        json.dumps(
            {
                "legacy_rows": len(legacy_rows),
                "spa_rows": len(spa_rows),
                "npz": str(NPZ),
                "json": str(OUT),
                "evaluated_branch_rows": sum(row["branch_status"] == "evaluated" for row in spa_rows),
                "vetoed_branch_rows": sum(row["branch_status"] == "vetoed_q_ge_40" for row in spa_rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

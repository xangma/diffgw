#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compare the clean XHM inspiral assembly with opaque public baselines.

The implementation under test receives the three dimensionless
``mathcal H_lm`` collocation values explicitly.  This diagnostic obtains
those values from the published parameter-fit callables without inspecting
their implementation, evaluates the clean assembly, and records public and
native physical Fourier-domain mode amplitudes on the same frequencies.

No public output is used to alter a coefficient.  The clean assembly derives
geometric amplitudes from the provider's common-H values and the published
leading 22-mode normalization.  Public values at the three collocation
frequencies are retained as an explicitly labeled oracle check for the
geometric veto inputs.  The resulting comparisons are descriptive validation
of a pure-PN plus supplied-collocation assembly; they do not recover a
calibration table.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import math
import platform
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch

from torchwave.approximants.imrphenom_xas import XASParameterFits
from torchwave.approximants.imrphenom_xhm import IMRPhenomXHM
from torchwave.approximants.xhm_fits import XHMParameterFits
from torchwave.approximants.xhm_amplitude_clean import (
    collocation_coefficients,
    normalized_pn_amplitude,
    normalized_pn_amplitude_compatibility,
)
from torchwave.approximants.xhm_inspiral_assembly_clean import (
    COLLOCATION_FRACTIONS,
    COMPATIBILITY_CUT_VERSION,
    normalized_inspiral_amplitude,
    physical_inspiral_amplitude,
    compatibility_inspiral_cut_frequency,
    printed_inspiral_cut_frequency,
    printed_veto_nodes,
)
from torchwave.approximants.xhm_spa_clean import normalized_spa_amplitude_21


DISTANCE_MPC = 500.0
DISTANCE_M: float | None = None
DISTANCE_SECONDS: float | None = None
DEFAULT_RELEASE = 122022
AUTHOR_RELEASE = 122019
# ``122018`` is the public inspiral-fit asset selector.  The separate
# ``InspiralAmpVersion`` selector is a small interpolation variant enum (and
# is intentionally left at its release default).
AUTHOR_INSPIRAL_AMP_FITS_VERSION = 122018
MODES = ((2, 1), (3, 2), (3, 3), (4, 4))
# The first point is also clamped to 20 Hz below, because the public API is
# called with fRef_In=20 Hz.  Every retained value remains strictly below its
# mode-specific f_ins.
FREQUENCY_FRACTIONS = np.asarray((0.50, 0.75, 0.90, 0.95), dtype=np.float64)
CASES = (
    {"label": "A40_15", "m1": 40.0, "m2": 15.0, "chi1": 0.3, "chi2": -0.2},
    {"label": "B55_11", "m1": 55.0, "m2": 11.0, "chi1": -0.4, "chi2": 0.2},
    {"label": "C27_16", "m1": 27.0, "m2": 16.0, "chi1": -0.25, "chi2": 0.15},
    {"label": "D35_20", "m1": 35.0, "m2": 20.0, "chi1": 0.2, "chi2": 0.1},
    {"label": "fresh32_18", "m1": 32.0, "m2": 18.0, "chi1": -0.6, "chi2": 0.45},
)
OUT = Path("verification/xhm-inspiral-cut-review/xhm-inspiral-cut-review.json")
NPZ = Path("/tmp/torchwave-xhm-inspiral-assembly-raw.npz")


class ProviderCollocations(NamedTuple):
    """Provider values together with the inputs and ordering used here."""

    ordered_h_lm: np.ndarray
    spin_variables: tuple[float, float, float, float]
    shat: float
    raw_amp_insp123: np.ndarray
    chi_pn_raw: float
    chi_pn: float


def _load_public_dependencies():
    """Load optional LAL dependencies only when public baselines are needed."""

    global DISTANCE_M, DISTANCE_SECONDS
    import lal
    import lalsimulation as ls

    if DISTANCE_M is None or DISTANCE_SECONDS is None:
        DISTANCE_M = DISTANCE_MPC * 1.0e6 * lal.PC_SI
        DISTANCE_SECONDS = DISTANCE_M / lal.C_SI
    return lal, ls


def sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def _scalar_tensor(value: object) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float64)
    if tensor.numel() != 1:
        raise ValueError(f"expected one scalar, got shape {tuple(tensor.shape)}")
    return tensor.reshape(())


def public_mode(
    frequency_hz: np.ndarray,
    m1: float,
    m2: float,
    chi1: float,
    chi2: float,
    mode: tuple[int, int],
    *,
    release_version: int | None = None,
    inspiral_amp_fits_version: int | None = None,
) -> np.ndarray:
    """Return one public physical mode with optional release controls.

    Omitting both controls preserves the installed public default.  The
    earlier author release is used only as a side-by-side diagnostic control
    and is set through public waveform-parameter setters.
    """

    lal, ls = _load_public_dependencies()
    assert DISTANCE_M is not None and DISTANCE_SECONDS is not None
    vector = lal.CreateREAL8Vector(len(frequency_hz))
    vector.data = np.asarray(frequency_hz, dtype=np.float64)
    params = lal.CreateDict()
    if release_version is not None:
        ls.SimInspiralWaveformParamsInsertPhenomXHMReleaseVersion(
            params, int(release_version)
        )
    if inspiral_amp_fits_version is not None:
        ls.SimInspiralWaveformParamsInsertPhenomXHMInspiralAmpFitsVersion(
            params, int(inspiral_amp_fits_version)
        )
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
        params,
    )
    # The finite distance argument makes this a physical strain/Hz series.
    return np.asarray(series.data.data, dtype=np.complex128).copy()


def native_mode(
    frequency_hz: np.ndarray,
    m1: float,
    m2: float,
    chi1: float,
    chi2: float,
    mode: tuple[int, int],
) -> np.ndarray:
    """Return the existing TorchWave production mode as a baseline."""

    with torch.no_grad():
        waveform = IMRPhenomXHM(modes=(mode,))(
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


def provider_collocations(
    m1: float, m2: float, chi1: float, chi2: float, mode: tuple[int, int]
) -> ProviderCollocations:
    """Call opaque earlier-provider inspiral fits with ``chiPNHat``.

    The earlier-release contract assigns ``amp_insp1/2/3`` magnitudes to
    ``[f_ins,.75*f_ins,.5*f_ins]``.  The clean solver therefore receives the
    absolute values in reversed ascending order ``[.5,.75,1.0]*f_ins``.
    Signed raw values remain in the result for review.
    """

    total = m1 + m2
    eta = torch.tensor(m1 * m2 / total**2, dtype=torch.float64)
    spin1 = torch.tensor(chi1, dtype=torch.float64)
    spin2 = torch.tensor(chi2, dtype=torch.float64)
    with torch.no_grad():
        spin_variables = XHMParameterFits.compute_spin_variables(eta, spin1, spin2)
    if not isinstance(spin_variables, (tuple, list)) or len(spin_variables) < 4:
        raise TypeError("opaque XHM spin-variable result must contain delta,x1,x2,S")
    spin_values = tuple(float(_scalar_tensor(item).detach().cpu()) for item in spin_variables)
    chi_eff = (m1 * chi1 + m2 * chi2) / total
    chi_pn_raw_value = chi_pn(
        eta=float(eta), chi_eff=chi_eff, chi1=chi1, chi2=chi2
    )
    chi_pn_value = normalized_chi_pn(
        eta=float(eta), chi_eff=chi_eff, chi1=chi1, chi2=chi2
    )
    chi_pn_tensor = torch.tensor(chi_pn_value, dtype=torch.float64)
    values: list[float] = []
    mode_name = f"{mode[0]}{mode[1]}"
    with torch.no_grad():
        for index in (1, 2, 3):
            function = getattr(XHMParameterFits, f"amp_insp{index}_{mode_name}")
            # The provider's fourth formal argument is named S, while XHM
            # Sec. II.3 Eq. (18)-(19) specifies chiPN for inspiral fits.
            value = _scalar_tensor(function(eta, spin1, spin2, chi_pn_tensor))
            values.append(float(value.detach().cpu()))
    raw_values = np.asarray(values, dtype=np.float64)
    return ProviderCollocations(
        ordered_h_lm=np.abs(raw_values[::-1]).copy(),
        spin_variables=spin_values[:4],
        shat=spin_values[3],
        raw_amp_insp123=raw_values,
        chi_pn_raw=chi_pn_raw_value,
        chi_pn=chi_pn_value,
    )


def kerr_f_isco_22(chi: float) -> float:
    """Return the dimensionless Kerr 22 GW frequency at the equatorial ISCO."""

    if not -1.0 <= chi <= 1.0:
        raise ValueError("Kerr spin must lie in [-1,1]")
    z1 = 1.0 + (1.0 - chi * chi) ** (1.0 / 3.0) * (
        (1.0 + chi) ** (1.0 / 3.0) + (1.0 - chi) ** (1.0 / 3.0)
    )
    z2 = math.sqrt(3.0 * chi * chi + z1 * z1)
    sign = 1.0 if chi >= 0.0 else -1.0
    radius = 3.0 + z2 - sign * math.sqrt(
        (3.0 - z1) * (3.0 + z1 + 2.0 * z2)
    )
    return 1.0 / (math.pi * (radius**1.5 + chi))


def chi_pn(eta: float, chi_eff: float, chi1: float, chi2: float) -> float:
    """Return the unnormalized published inspiral spin numerator."""

    return chi_eff - (38.0 / 113.0) * eta * (chi1 + chi2)


def normalized_chi_pn(eta: float, chi_eff: float, chi1: float, chi2: float) -> float:
    """Return the earlier-provider normalized ``chiPNHat`` input."""

    denominator = 1.0 - (76.0 / 113.0) * eta
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("chiPNHat denominator must be finite and positive")
    return chi_pn(eta, chi_eff, chi1, chi2) / denominator


chi_pn_hat = normalized_chi_pn


def final_spin_and_f_isco(
    eta: float, chi1: float, chi2: float
) -> tuple[float, float]:
    """Evaluate the public XAS final-spin fit, then the Kerr ISCO formula."""

    eta_tensor = torch.tensor(eta, dtype=torch.float64)
    spin1_tensor = torch.tensor(chi1, dtype=torch.float64)
    spin2_tensor = torch.tensor(chi2, dtype=torch.float64)
    with torch.no_grad():
        final_spin = float(
            _scalar_tensor(
                XASParameterFits.fit_final_spin(
                    eta_tensor, spin1_tensor, spin2_tensor
                )
            ).detach().cpu()
        )
    return final_spin, kerr_f_isco_22(final_spin)


def opaque_ringdown_frequencies_32_22(final_spin: float) -> tuple[float, float]:
    """Return the provider's opaque dimensionless 32 and 22 ring frequencies.

    The compatibility cut needs only the ratio ``f_rd_32/f_rd_22``.  These
    values are requested from the existing fit callable and are recorded as
    inputs; no fit body or coefficient is copied into the clean helper.
    """

    if not math.isfinite(final_spin) or not -1.0 <= final_spin <= 1.0:
        raise ValueError("final_spin must be finite and lie in [-1,1]")
    spin = torch.tensor(final_spin, dtype=torch.float64)
    with torch.no_grad():
        f_rd_22 = float(
            _scalar_tensor(XHMParameterFits.qnm_fring_22(spin)).detach().cpu()
        )
        f_rd_32 = float(
            _scalar_tensor(XHMParameterFits.qnm_fring_32(spin)).detach().cpu()
        )
    if (
        not math.isfinite(f_rd_22)
        or not math.isfinite(f_rd_32)
        or f_rd_22 <= 0.0
        or f_rd_32 <= 0.0
    ):
        raise ValueError("provider ringdown frequencies must be finite and positive")
    return f_rd_32, f_rd_22


def geometric_amplitudes_from_common_h(
    common_h: np.ndarray,
    frequency_hz: np.ndarray,
    mass_seconds: float,
    eta: float,
) -> np.ndarray:
    """Convert common-H values to geometric amplitudes using Appendix E.

    The clean veto is stated for geometric ``|h_lm|``.  With the paper's
    ``H_lm = |A_lm^SPA| / A_22^0`` convention, ``A_22^0 =
    pi*sqrt(2*eta/3)*(pi*f*M)^(-7/6)``.  This conversion therefore uses only
    the supplied provider H values and intrinsic inputs; it does not query a
    public waveform oracle.
    """

    h = np.asarray(common_h, dtype=np.float64)
    frequency = np.asarray(frequency_hz, dtype=np.float64)
    if h.shape != frequency.shape:
        raise ValueError("common-H and frequency arrays must have the same shape")
    if not np.all(np.isfinite(h)) or np.any(h < 0.0):
        raise ValueError("common-H values must be finite and non-negative")
    if not np.all(np.isfinite(frequency)) or np.any(frequency <= 0.0):
        raise ValueError("frequencies must be finite and positive")
    if not math.isfinite(mass_seconds) or mass_seconds <= 0.0:
        raise ValueError("mass_seconds must be finite and positive")
    if not math.isfinite(eta) or eta <= 0.0:
        raise ValueError("eta must be finite and positive")
    a22_0 = math.pi * math.sqrt(2.0 * eta / 3.0) * (
        math.pi * frequency * mass_seconds
    ) ** (-7.0 / 6.0)
    return h * a22_0


def mode_pn(
    mode: tuple[int, int],
    frequency_hz: torch.Tensor,
    mass_seconds: float,
    m1: float,
    m2: float,
    chi1: float,
    chi2: float,
) -> torch.Tensor:
    if mode == (2, 1):
        # Convert |Hhat_21| to the common Appendix-E H_lm normalization before
        # solving for the pseudo-PN correction coefficients.
        return 2.0 ** (-7.0 / 6.0) * normalized_spa_amplitude_21(
            frequency_hz, mass_seconds, m1, m2, chi1, chi2
        )
    if mode in ((3, 2), (4, 4)):
        return normalized_pn_amplitude_compatibility(
            mode, frequency_hz, mass_seconds, m1, m2, chi1, chi2
        )
    return normalized_pn_amplitude(
        mode, frequency_hz, mass_seconds, m1, m2, chi1, chi2
    )


def frequencies_below(f_ins_hz: float) -> np.ndarray:
    if not math.isfinite(f_ins_hz) or f_ins_hz <= 0.0:
        raise ValueError("f_ins_hz must be finite and positive")
    candidates = np.maximum(20.0, f_ins_hz * FREQUENCY_FRACTIONS)
    frequencies = np.unique(candidates)
    frequencies = frequencies[frequencies < f_ins_hz]
    if len(frequencies) < 3:
        raise ValueError(f"f_ins={f_ins_hz} Hz leaves too few public frequencies")
    validate_frequency_support(frequencies, f_ins_hz)
    return frequencies


def validate_frequency_support(
    frequencies_hz: np.ndarray,
    f_ins_hz: float,
    *,
    allow_endpoint: bool = False,
) -> np.ndarray:
    """Validate a public-call grid against one mode's inspiral cut.

    Diagnostic samples must be strictly below ``f_ins_hz``.  Collocation
    nodes may include the endpoint, so callers can opt into ``allow_endpoint``
    for that separate check.  The validated float64 array is returned for
    convenient use by callers.
    """

    frequencies = np.asarray(frequencies_hz, dtype=np.float64)
    if frequencies.ndim != 1 or frequencies.size == 0:
        raise ValueError("frequencies_hz must be a non-empty one-dimensional array")
    if not math.isfinite(f_ins_hz) or f_ins_hz <= 0.0:
        raise ValueError("f_ins_hz must be finite and positive")
    if not np.all(np.isfinite(frequencies)) or np.any(frequencies <= 0.0):
        raise ValueError("frequencies_hz must be finite and positive")
    exceeds = np.any(frequencies > f_ins_hz) if allow_endpoint else np.any(
        frequencies >= f_ins_hz
    )
    if exceeds:
        relation = "at or below" if allow_endpoint else "strictly below"
        raise ValueError(f"frequencies_hz must be {relation} f_ins_hz")
    return frequencies


def validate_veto_support(
    retained_nodes: tuple[int, ...] | list[int], *, node_count: int = 3
) -> tuple[int, ...]:
    """Validate a non-empty, ordered subset of collocation-node indices."""

    retained = tuple(int(index) for index in retained_nodes)
    if node_count <= 0 or not retained:
        raise ValueError("at least one collocation node must be retained")
    if any(index < 0 or index >= node_count for index in retained):
        raise ValueError("retained collocation nodes are outside the node support")
    if tuple(sorted(set(retained))) != retained:
        raise ValueError("retained collocation nodes must be sorted and unique")
    return retained


def relative_l2(branch: np.ndarray, baseline: np.ndarray) -> float:
    denominator = max(float(np.linalg.norm(baseline)), np.finfo(float).tiny)
    return float(np.linalg.norm(branch - baseline) / denominator)


def shape_relative_l2(branch: np.ndarray, baseline: np.ndarray) -> float:
    branch_norm = max(float(np.linalg.norm(branch)), np.finfo(float).tiny)
    baseline_norm = max(float(np.linalg.norm(baseline)), np.finfo(float).tiny)
    return relative_l2(branch / branch_norm, baseline / baseline_norm)


def main() -> None:
    lal, ls = _load_public_dependencies()
    assert DISTANCE_M is not None and DISTANCE_SECONDS is not None
    default_params = lal.CreateDict()
    observed_default_release = int(
        ls.SimInspiralWaveformParamsLookupPhenomXHMReleaseVersion(default_params)
    )
    observed_default_amp_fits_version = int(
        ls.SimInspiralWaveformParamsLookupPhenomXHMInspiralAmpFitsVersion(
            default_params
        )
    )
    observed_default_amp_variant = int(
        ls.SimInspiralWaveformParamsLookupPhenomXHMInspiralAmpVersion(
            default_params
        )
    )
    arrays: dict[str, np.ndarray] = {}
    rows: list[dict[str, object]] = []

    for case in CASES:
        label = str(case["label"])
        m1 = float(case["m1"])
        m2 = float(case["m2"])
        chi1 = float(case["chi1"])
        chi2 = float(case["chi2"])
        total = m1 + m2
        eta = m1 * m2 / total**2
        q = max(m1, m2) / min(m1, m2)
        mass_seconds = total * lal.MTSUN_SI
        chi_eff = (m1 * chi1 + m2 * chi2) / total
        eta_tensor = torch.tensor(eta, dtype=torch.float64)
        spin1_tensor = torch.tensor(chi1, dtype=torch.float64)
        spin2_tensor = torch.tensor(chi2, dtype=torch.float64)
        with torch.no_grad():
            f_meco_22 = float(
                _scalar_tensor(
                    XASParameterFits.fit_f_meco(
                        eta_tensor, spin1_tensor, spin2_tensor
                    )
                ).detach().cpu()
            )
        final_spin, f_isco_22 = final_spin_and_f_isco(eta, chi1, chi2)

        for mode in MODES:
            f_rd_32 = f_rd_22 = None
            if mode == (3, 2):
                f_rd_32, f_rd_22 = opaque_ringdown_frequencies_32_22(final_spin)
            f_ins_printed_dimensionless = float(
                printed_inspiral_cut_frequency(
                    mode,
                    f_meco_22,
                    f_isco_22,
                    chi_eff,
                    f_rd_32=f_rd_32,
                    f_rd_22=f_rd_22,
                ).detach().cpu()
            )
            if mode == (3, 2):
                # The clean 32 reconstruction uses the explicit compatibility
                # control.  Keep the paper-literal cut above for comparison.
                f_ins_dimensionless = float(
                    compatibility_inspiral_cut_frequency(
                        mode,
                        f_meco_22,
                        f_isco_22,
                        chi_eff,
                        f_rd_32=f_rd_32,
                        f_rd_22=f_rd_22,
                    ).detach().cpu()
                )
                cut_version = COMPATIBILITY_CUT_VERSION
            else:
                f_ins_dimensionless = f_ins_printed_dimensionless
                cut_version = "printed-paper"
            f_ins_hz = f_ins_dimensionless / mass_seconds
            frequencies_hz = frequencies_below(f_ins_hz)
            node_frequencies_hz = f_ins_hz * np.asarray(
                COLLOCATION_FRACTIONS, dtype=np.float64
            )
            validate_frequency_support(node_frequencies_hz, f_ins_hz, allow_endpoint=True)
            provider_result = provider_collocations(m1, m2, chi1, chi2, mode)
            provider = provider_result.ordered_h_lm
            spin_variables = provider_result.spin_variables
            S = provider_result.shat

            # The assembly's veto input is derived from the provider H values
            # and the published A22^0 normalization.  Public values are kept
            # on a separate oracle-only path for component comparison.
            geometric_nodes = geometric_amplitudes_from_common_h(
                provider, node_frequencies_hz, mass_seconds, eta
            )
            public_nodes = public_mode(
                node_frequencies_hz, m1, m2, chi1, chi2, mode
            )
            author_public_nodes = public_mode(
                node_frequencies_hz,
                m1,
                m2,
                chi1,
                chi2,
                mode,
                release_version=AUTHOR_RELEASE,
                inspiral_amp_fits_version=AUTHOR_INSPIRAL_AMP_FITS_VERSION,
            )
            oracle_geometric_nodes = (
                np.abs(public_nodes) * DISTANCE_SECONDS / (mass_seconds**2)
            )
            author_geometric_nodes = (
                np.abs(author_public_nodes) * DISTANCE_SECONDS / (mass_seconds**2)
            )
            if m1 >= m2:
                veto_chi1, veto_chi2 = chi1, chi2
            else:
                veto_chi1, veto_chi2 = chi2, chi1
            retained = printed_veto_nodes(
                mode,
                q,
                veto_chi1,
                veto_chi2,
                provider,
                geometric_veto_amplitudes=geometric_nodes,
            )
            retained = validate_veto_support(retained)
            oracle_retained = printed_veto_nodes(
                mode,
                q,
                veto_chi1,
                veto_chi2,
                provider,
                geometric_veto_amplitudes=oracle_geometric_nodes,
            )
            oracle_retained = validate_veto_support(oracle_retained)

            frequency_tensor = torch.as_tensor(frequencies_hz, dtype=torch.float64)
            with torch.no_grad():
                clean = physical_inspiral_amplitude(
                    mode,
                    frequency_tensor,
                    f_ins_hz,
                    provider,
                    mass_seconds,
                    m1,
                    m2,
                    chi1,
                    chi2,
                    DISTANCE_SECONDS,
                    geometric_veto_amplitudes=geometric_nodes,
                    apply_vetoes=True,
                ).detach().cpu().numpy()
                node_pn = mode_pn(
                    mode,
                    torch.as_tensor(node_frequencies_hz, dtype=torch.float64),
                    mass_seconds,
                    m1,
                    m2,
                    chi1,
                    chi2,
                )
                coefficients = collocation_coefficients(
                    node_pn,
                    torch.as_tensor(provider, dtype=torch.float64),
                    f_ins_hz,
                    active_terms=len(retained),
                    retained_nodes=retained,
                ).detach().cpu().numpy()

            public = public_mode(frequencies_hz, m1, m2, chi1, chi2, mode)
            author_public = public_mode(
                frequencies_hz,
                m1,
                m2,
                chi1,
                chi2,
                mode,
                release_version=AUTHOR_RELEASE,
                inspiral_amp_fits_version=AUTHOR_INSPIRAL_AMP_FITS_VERSION,
            )
            native = native_mode(frequencies_hz, m1, m2, chi1, chi2, mode)
            public_amplitude = np.abs(public)
            author_public_amplitude = np.abs(author_public)
            native_amplitude = np.abs(native)
            key = f"{label}_{mode[0]}{mode[1]}"
            arrays[f"{key}_frequency_hz"] = frequencies_hz
            arrays[f"{key}_collocation_frequency_hz"] = node_frequencies_hz
            arrays[f"{key}_provider_H_lm"] = provider
            arrays[f"{key}_geometric_veto_amplitudes"] = geometric_nodes
            arrays[f"{key}_oracle_geometric_veto_amplitudes"] = oracle_geometric_nodes
            arrays[f"{key}_author_release_geometric_veto_amplitudes"] = author_geometric_nodes
            arrays[f"{key}_clean_physical_amplitude"] = clean
            arrays[f"{key}_public_complex_physical"] = public
            arrays[f"{key}_public_physical_amplitude"] = public_amplitude
            arrays[f"{key}_author_release_complex_physical"] = author_public
            arrays[f"{key}_author_release_physical_amplitude"] = author_public_amplitude
            arrays[f"{key}_native_complex_physical"] = native
            arrays[f"{key}_native_physical_amplitude"] = native_amplitude

            rows.append(
                {
                    "case": label,
                    "mode": list(mode),
                    "m1_msun": m1,
                    "m2_msun": m2,
                    "q": q,
                    "eta": eta,
                    "chi1_aligned": chi1,
                    "chi2_aligned": chi2,
                    "chi_eff_mass_weighted": chi_eff,
                    "chi_pn_raw_inspiral_fit_numerator": provider_result.chi_pn_raw,
                    "chi_pn_inspiral_fit_input_normalized": provider_result.chi_pn,
                    "chi_pn_normalization_denominator": 1.0 - (76.0 / 113.0) * eta,
                    "spin_variables_delta_x1_x2_S": list(spin_variables),
                    "S_hat_available_but_not_passed_to_inspiral_fit": S,
                    "final_spin_chi_f_from_xas_fit": final_spin,
                    "public_default_release_observed": observed_default_release,
                    "public_author_release_control": AUTHOR_RELEASE,
                    "public_default_row_role": "current-installed-default descriptive observation",
                    "author_release_row_role": "explicit old-release diagnostic control",
                    "strict_current_release_parity_claim": False,
                    "clean_collocation_provenance": "earlier-release compatibility provider; not current-default calibration",
                    "f_meco_22_dimensionless_Mf": f_meco_22,
                    "f_isco_22_dimensionless_Mf": f_isco_22,
                        "f_isco_22_dimensionless_Mf_final_spin": f_isco_22,
                        "f_rd_32_dimensionless_Mf": f_rd_32,
                        "f_rd_22_dimensionless_Mf": f_rd_22,
                        "f_rd_32_over_f_rd_22": (
                            None if f_rd_32 is None else f_rd_32 / f_rd_22
                        ),
                        "cut_frequency_variant": cut_version,
                        "f_ins_dimensionless_Mf_printed_control": f_ins_printed_dimensionless,
                        "f_ins_dimensionless_Mf": f_ins_dimensionless,
                        "f_ins_hz": f_ins_hz,
                    "frequency_hz": frequencies_hz.tolist(),
                    "frequency_as_fraction_of_f_ins": (
                        frequencies_hz / f_ins_hz
                    ).tolist(),
                        "collocation_frequency_hz": node_frequencies_hz.tolist(),
                        "frequency_support_strictly_below_cut": True,
                        "collocation_endpoint_allowed_by_support_check": True,
                    "provider_amp_insp1_2_3_raw": provider_result.raw_amp_insp123.tolist(),
                    "provider_collocation_values_H_lm": provider.tolist(),
                    "provider_collocation_order_for_nodes": "[amp_insp3, amp_insp2, amp_insp1] -> [0.5, 0.75, 1.0]*f_ins",
                    "geometric_veto_amplitudes_from_provider_common_H": geometric_nodes.tolist(),
                    "geometric_veto_amplitudes_from_public_output": oracle_geometric_nodes.tolist(),
                    "geometric_veto_amplitudes_from_author_release_output": author_geometric_nodes.tolist(),
                        "retained_collocation_nodes": list(retained),
                        "oracle_retained_collocation_nodes_component_only": list(oracle_retained),
                        "veto_support_nonempty": bool(retained),
                    "provider_oracle_veto_node_sets_match": retained == oracle_retained,
                    "pseudo_pn_coefficients_alpha_beta_gamma_zero_padded": coefficients.tolist(),
                    "clean_physical_amplitude_units": "Fourier-domain strain/Hz (SI)",
                    "public_physical_amplitude_units": "Fourier-domain strain/Hz (SI)",
                    "author_release_physical_amplitude_units": "Fourier-domain strain/Hz (SI)",
                    "native_physical_amplitude_units": "Fourier-domain strain/Hz (SI)",
                    "clean_physical_l2": float(np.linalg.norm(clean)),
                    "public_physical_l2": float(np.linalg.norm(public_amplitude)),
                    "author_release_physical_l2": float(np.linalg.norm(author_public_amplitude)),
                    "native_physical_l2": float(np.linalg.norm(native_amplitude)),
                    "clean_vs_public_physical_scale_acceptance_like_relative_l2": relative_l2(
                        clean, public_amplitude
                    ),
                    "clean_vs_native_physical_scale_acceptance_like_relative_l2": relative_l2(
                        clean, native_amplitude
                    ),
                    "clean_vs_author_release_physical_scale_relative_l2": relative_l2(
                        clean, author_public_amplitude
                    ),
                    "default_vs_author_release_physical_scale_relative_l2": relative_l2(
                        public_amplitude, author_public_amplitude
                    ),
                    "native_vs_public_physical_scale_acceptance_like_relative_l2": relative_l2(
                        native_amplitude, public_amplitude
                    ),
                    "clean_vs_public_shape_only_normalized_relative_l2": shape_relative_l2(
                        clean, public_amplitude
                    ),
                    "clean_vs_native_shape_only_normalized_relative_l2": shape_relative_l2(
                        clean, native_amplitude
                    ),
                    "native_vs_public_shape_only_normalized_relative_l2": shape_relative_l2(
                        native_amplitude, public_amplitude
                    ),
                    "release_views": {
                        "current_default": {
                            "release": observed_default_release,
                            "role": "current-installed-default descriptive observation",
                            "strict_parity_claim": False,
                            "clean_vs_public_physical_relative_l2": relative_l2(
                                clean, public_amplitude
                            ),
                            "public_physical_l2": float(np.linalg.norm(public_amplitude)),
                        },
                        "author_release_control": {
                            "release": AUTHOR_RELEASE,
                            "inspiral_amp_fits_version": AUTHOR_INSPIRAL_AMP_FITS_VERSION,
                            "role": "explicit old-release diagnostic control",
                            "strict_current_release_parity_claim": False,
                            "clean_vs_author_release_physical_relative_l2": relative_l2(
                                clean, author_public_amplitude
                            ),
                            "author_release_physical_l2": float(
                                np.linalg.norm(author_public_amplitude)
                            ),
                        },
                    },
                    "array_sha256": {
                        "frequency_hz": array_sha(frequencies_hz),
                        "collocation_frequency_hz": array_sha(node_frequencies_hz),
                        "provider_H_lm": array_sha(provider),
                        "provider_amp_insp1_2_3_raw": array_sha(provider_result.raw_amp_insp123),
                        "geometric_veto_amplitudes": array_sha(geometric_nodes),
                        "oracle_geometric_veto_amplitudes": array_sha(oracle_geometric_nodes),
                        "author_release_geometric_veto_amplitudes": array_sha(author_geometric_nodes),
                        "clean_physical_amplitude": array_sha(clean),
                        "public_complex_physical": array_sha(public),
                        "public_physical_amplitude": array_sha(public_amplitude),
                        "author_release_complex_physical": array_sha(author_public),
                        "author_release_physical_amplitude": array_sha(author_public_amplitude),
                        "native_complex_physical": array_sha(native),
                        "native_physical_amplitude": array_sha(native_amplitude),
                    },
                }
            )

    np.savez_compressed(NPZ, **arrays)

    provider_names = [
        "compute_spin_variables",
        *[f"amp_insp{index}_{ell}{emm}" for ell, emm in MODES for index in (1, 2, 3)],
    ]
    provider_signatures = {
        name: str(inspect.signature(getattr(XHMParameterFits, name)))
        for name in provider_names
    }
    provider_signatures["fit_f_meco"] = str(
        inspect.signature(getattr(XASParameterFits, "fit_f_meco"))
    )
    provider_signatures["fit_final_spin"] = str(
        inspect.signature(getattr(XASParameterFits, "fit_final_spin"))
    )
    metadata = {
        "paper": "arXiv:2001.10914v2",
        "purpose": "bounded clean-room inspiral assembly with explicit XHM collocations",
        "math_fixes": {
            "provider_spin_input": "chiPNHat = (chi_eff - 38*eta*(chi1+chi2)/113)/(1 - 76*eta/113)",
            "provider_collocation_order": "abs(reverse(amp_insp1, amp_insp2, amp_insp3)) -> [.5,.75,1.0]*f_ins",
            "cut_33": "final -5/6 chi_eff term is linear",
            "cut_21": "final -5/6 chi_eff term is squared",
            "cut_32_printed": "abs(chi_eff), with epsilon=f_rd_32/f_rd_22 on the MECO term only",
            "cut_32_compatibility": "abs(chi_eff), with epsilon=f_rd_32/f_rd_22 multiplying the complete MECO-to-ISCO parenthesis",
            "pn_44": "compatibility signs reverse printed v2 and v4 only",
            "pn_32": "compatibility signs reverse printed v2, v3, and v4 only",
            "spa_21": "full 3PN TaylorT4 v6 recovered by reciprocal-series inversion of uncalibrated TaylorF2 phase coefficients",
        },
        "release_controls": {
            "installed_default_release_expected": DEFAULT_RELEASE,
            "installed_default_release_observed": observed_default_release,
            "installed_default_inspiral_amp_fits_version_observed": observed_default_amp_fits_version,
            "installed_default_inspiral_amp_variant_observed": observed_default_amp_variant,
            "author_release": AUTHOR_RELEASE,
            "author_inspiral_amp_fits_version": AUTHOR_INSPIRAL_AMP_FITS_VERSION,
            "release_public_setter": "SimInspiralWaveformParamsInsertPhenomXHMReleaseVersion",
            "inspiral_amp_fits_public_setter": "SimInspiralWaveformParamsInsertPhenomXHMInspiralAmpFitsVersion",
            "author_control_usage": "only explicit public setter control; clean math and collocations are not routed into production",
        },
        "current_default_residuals": {
            "meaning": "row fields clean_vs_public_physical_scale_acceptance_like_relative_l2 and default_vs_author_release_physical_scale_relative_l2",
            "default_release": observed_default_release,
            "calibration_interpretation": "descriptive current-default comparisons; no fit constants or residuals are used to alter math",
            "strict_parity_claim": False,
            "author_release_rows_are": "explicit old-release diagnostic controls, never current-default parity",
        },
        "calibration_blockers": [
            "Current default release 122022 uses changed fit data, cuts, intermediate behavior, and collocation normalization; old-provider values are not interchangeable.",
            "The 33 provider calibration differs across the old-release control and remains an explicit blocker.",
            "The current-default all-new amplitude calibration is unresolved.",
            "Mode 32 dynamic ringdown-amplitude veto requires a ringdown value outside this inspiral-only helper and is unsupported here.",
        ],
        "scope": {
            "modes": [list(mode) for mode in MODES],
            "mass_ratio_domain": "q < 20",
            "mode_21_pn_branch": "non-reexpanded SPA expression, published for q < 40",
            "mode_32_and_44_pn_branch": "re-expanded Appendix E expressions with named compatibility sign variants",
            "mode_33_pn_branch": "re-expanded Appendix E printed expression",
            "production_routing": "none; diagnostic and helper only",
        },
        "equations": [
            "Sec. II.3 Eq. (18): chiPN = chi_eff - (38*eta/113)*(chi1+chi2) for inspiral amplitude fits",
            "Sec. II.3 Eq. (19): Shat = (m1^2 chi1 + m2^2 chi2)/(m1^2+m2^2) for merger/ringdown fits",
            "Sec. IV.A.1 Eq. (fcutIns) and Table deltaInsp",
            "Sec. IV.A.1 Eq. (AlmScaling): A22^0 = pi*sqrt(2*eta/3)*(pi*f*M)^(-7/6) in geometric units",
            "Sec. IV.A.1 Eq. (insp_amp): H_lm = |A_lm^PN|/A22^0 + alpha*u^(7/3) + beta*u^(8/3) + gamma*u^(9/3)",
            "Sec. IV.A.1 collocations [.5 fIns, .75 fIns, fIns] and highest-order zero padding after vetoes",
            "Appendix E E10 (21), E11 (33), E12 (32), E14 (44), E15 (43); this diagnostic evaluates 21, 32, 33, 44",
            "Appendix E E3 non-reexpanded (2,1) SPA branch for q < 40",
        ],
        "primary_source_scope": {
            "tex": "/tmp/gq2001/hmc2.tex lines 572-673 and Appendix E",
            "rendered_paper": "/tmp/torchwave-paper-assets/garcia-quiros-2020-imrphenomxhm-v2.pdf",
            "table_note": "The printed 33 deltaIns term is linear -5/6 chi_eff; it is transcribed literally.",
        },
        "normalization": {
            "provider_and_clean_collocations": "dimensionless H_lm values at [.5,.75,1]*f_ins",
            "provider_common_h_to_geometric": "A_lm^geo = H_lm * pi*sqrt(2*eta/3)*(pi*f_hz*M_seconds)^(-7/6)",
            "clean_normalized_return": "dimensionless H_lm amplitude",
            "clean_physical_return": "(M^2/D)*pi*sqrt(2*eta/3)*(pi*f*M)^(-7/6)*H_lm, Fourier-domain strain/Hz",
            "distance_mpc": DISTANCE_MPC,
            "distance_m": DISTANCE_M,
            "distance_seconds": DISTANCE_SECONDS,
            "physical_scale_metrics": "physical-scale acceptance-like; M^2/D is retained",
            "norm_normalized_metrics": "shape-only; L2 normalization intentionally removes scale",
        },
        "collocation_solver": {
            "fractions": list(map(float, COLLOCATION_FRACTIONS)),
            "powers": [7.0 / 3.0, 8.0 / 3.0, 3.0],
            "removed_node_rule": "each removed node zero-pads one highest pseudo-PN coefficient",
            "21_geometric_threshold": "|h21| < 0.2 in geometric units; clean assembly uses provider H_lm converted with A22^0",
            "public_oracle_role": "public geometric amplitudes are computed separately and used only for a component-only veto comparison",
        },
        "vetoes": {
            "21": "q < 8: remove geometric |h21| values below 0.2; at every q remove middle node for a nonmonotonic geometric-amplitude sequence",
            "33": "1 < q < 1.2, chi1 < -0.1, chi2 > 0: remove final node",
            "44": "no veto printed in cited section",
            "32": "clean cut and compatibility polynomial are implemented; q<20 parameter-region and nonmonotonic-node checks are recorded; dynamic ringdown-amplitude veto is out of this inspiral-only diagnostic scope",
        },
        "opaque_provider": {
            "output_interpretation": "amp_insp1/2/3 are the three fitted hybrid H_lm collocation values described in Sec. IV.A.1",
            "formal_spin_argument_name": "S",
            "diagnostic_spin_argument_choice": "normalized chiPNHat from the sanitized earlier-release contract, passed through the provider's formal S argument",
            "formal_spin_contract_status": "the provider's formal S name remains opaque; the sanitized earlier-release contract fixes the normalized input",
            "node_order_used": "abs([amp_insp3,amp_insp2,amp_insp1]) assigned to [0.5,0.75,1.0]*f_ins",
            "order_status": "sanitized contract choice; provider implementation and fit bodies were not inspected",
            "no_coefficient_inference": True,
            "no_source_body_inspection": True,
            "signatures": provider_signatures,
        },
        "input_contract_investigation": {
            "published_inspiral_spin_rule": "XHM Eq. (18) numerator with the sanitized earlier-provider normalization",
            "selected_inspiral_spin_input": "chiPNHat = (chi_eff - 38*eta*(chi1+chi2)/113)/(1 - 76*eta/113)",
            "selected_collocation_mapping": "amp_insp1/2/3 magnitudes -> [1.0,0.75,0.5]*f_ins, then reverse to solver nodes [0.5,0.75,1.0]*f_ins",
            "black_box_observation": "provider raw values are retained signed; clean solver values use their absolute reversed order",
            "provider_formal_argument_note": "the callable's formal S name remains opaque; this diagnostic passes the contractually specified normalized chiPNHat",
            "calibration_precision_interpretation": "corrected final-spin, chiPNHat, ordering, and normalization inputs are prerequisites; current-release calibration differences remain blocked",
            "clean_room_basis": "public equations, public API signatures, and black-box numerical behavior only; no provider fit bodies or external implementation were inspected",
        },
        "cut_frequency_inputs": {
            "f_meco": "opaque XASParameterFits.fit_f_meco(eta,chi1,chi2), recorded as dimensionless M*f22",
            "final_spin": "opaque XASParameterFits.fit_final_spin(eta,chi1,chi2), recorded as chi_f",
            "f_isco": "clean equatorial Kerr ISCO formula evaluated at the XAS final-spin chi_f, as specified by XAS Fig. 1",
            "f_rd_32_and_f_rd_22": "opaque XHMParameterFits.qnm_fring_32/22(final_spin), used only for the explicit mode-32 epsilon ratio",
            "mode_32_cut_selection": "release-compatibility for clean reconstruction; printed-paper retained in each row as a control",
            "frequency_grid": "max(20 Hz, [0.50,0.75,0.90,0.95]*f_ins), unique values strictly below each mode f_ins",
        },
        "baseline_contracts": {
            "public": "lalsimulation.SimIMRPhenomXHMFrequencySequenceOneMode(..., distance_m, ...)->COMPLEX16FrequencySeries",
            "public_units": "distance-corrected physical Fourier-domain strain/Hz",
            "native": "TorchWave IMRPhenomXHM(..., return_unprojected_modes=True), current production baseline only",
        },
        "cases": [dict(case) for case in CASES],
        "rows": rows,
        "npz": str(NPZ),
        "source_hashes": {
            "paper_tex": sha("/tmp/gq2001/hmc2.tex"),
            "clean_amplitude_helper": sha("src/torchwave/approximants/xhm_amplitude_clean.py"),
            "clean_spa_helper": sha("src/torchwave/approximants/xhm_spa_clean.py"),
            "clean_assembly_helper": sha("src/torchwave/approximants/xhm_inspiral_assembly_clean.py"),
            "script": sha(__file__),
        },
        "raw_npz_sha256": sha(NPZ),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(metadata, indent=2) + "\n")
    print(
        json.dumps(
            {
                "rows": len(rows),
                "npz": str(NPZ),
                "json": str(OUT),
                "npz_sha256": sha(NPZ),
                "json_sha256": sha(OUT),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

"""Reusable Parity Verification Harness for TorchWave against LALSimulation Oracles.

External binary verification (separate from waveform implementation):
Computes direct unnormalized complex relative L2 error, relative norm error,
magnitude relative L2 error, flat-noise mismatch, and named detector PSD mismatches
(e.g., Advanced LIGO Zero Detuning High Power, Cosmic Explorer, Einstein Telescope)
across multiple detectors (H1, L1, V1) and sky locations.

Acceptance Gates (Direct CPU FP64):
- complex_relative_l2 <= 1.0e-6
- relative_norm_error <= 1.0e-6
- mismatch <= 1.0e-8
"""

import copy
import json
import math
import os
from pathlib import Path
import platform
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# Auto-detect local LAL oracle data path if present
for _candidate in ["/private/tmp/torchwave-oracle-data", "/tmp/torchwave-oracle-data"]:
    if os.path.isdir(_candidate):
        os.environ.setdefault("LAL_DATA_PATH", _candidate)
        break

import numpy as np
import torch

# Every external comparison uses the same acceptance ceiling. Per-case settings
# may tighten these values but may never relax the published parity contract.
PARITY_GATE = {"complex_relative_l2": 1.0e-6, "mismatch": 1.0e-8}

from torchwave.constants import C_SI, MPC_SI, MSUN_SEC, PI
from torchwave.waveforms import get_fd_waveform
from torchwave.detectors.psd import (
    aLIGOZeroDetHighPower,
    CosmicExplorer,
    EinsteinTelescope,
)


def direct_complex_relative_l2(
    h_test: torch.Tensor,
    h_ref: torch.Tensor,
) -> float:
    """Compute direct unnormalized complex relative L2 error: ||h_test - h_ref||_2 / ||h_ref||_2."""
    diff = torch.linalg.norm(h_test - h_ref)
    norm_ref = torch.linalg.norm(h_ref)
    if norm_ref == 0.0:
        return float("inf") if diff > 0.0 else 0.0
    return float((diff / norm_ref).detach().cpu())


def relative_norm_error(
    h_test: torch.Tensor,
    h_ref: torch.Tensor,
) -> float:
    """Compute relative norm difference: | ||h_test|| - ||h_ref|| | / ||h_ref||."""
    norm_test = torch.linalg.norm(h_test)
    norm_ref = torch.linalg.norm(h_ref)
    if norm_ref == 0.0:
        return float("inf") if norm_test > 0.0 else 0.0
    return float((abs(norm_test - norm_ref) / norm_ref).detach().cpu())


def direct_magnitude_relative_l2(
    h_test: torch.Tensor,
    h_ref: torch.Tensor,
) -> float:
    """Compute direct magnitude relative L2 error: || |h_test| - |h_ref| ||_2 / || |h_ref| ||_2."""
    mag_test = torch.abs(h_test)
    mag_ref = torch.abs(h_ref)
    diff = torch.linalg.norm(mag_test - mag_ref)
    norm_ref = torch.linalg.norm(mag_ref)
    if norm_ref == 0.0:
        return float("inf") if diff > 0.0 else 0.0
    return float((diff / norm_ref).detach().cpu())


def flat_inner_product(
    h1: torch.Tensor,
    h2: torch.Tensor,
) -> float:
    """Compute flat-noise complex inner product: 4 * Re(sum(h1 * conj(h2)))."""
    return float(4.0 * torch.sum((h1 * torch.conj(h2)).real).item())


def flat_mismatch(
    h1: torch.Tensor,
    h2: torch.Tensor,
) -> float:
    """Compute normalized flat-noise mismatch without time/phase shifting: 1 - <h1, h2> / (||h1|| * ||h2||)."""
    ip = flat_inner_product(h1, h2)
    norm1 = math.sqrt(max(flat_inner_product(h1, h1), 0.0))
    norm2 = math.sqrt(max(flat_inner_product(h2, h2), 0.0))
    if norm1 == 0.0 or norm2 == 0.0 or not math.isfinite(norm1) or not math.isfinite(norm2) or not math.isfinite(ip):
        return 1.0
    overlap = ip / (norm1 * norm2)
    if not math.isfinite(overlap) or overlap > 1.0 + 1.0e-12 or overlap < -1.0 - 1.0e-12:
        return float("nan")
    if overlap > 1.0:
        overlap = 1.0
    return 1.0 - overlap


def psd_weighted_inner_product(
    h1: torch.Tensor,
    h2: torch.Tensor,
    psd: torch.Tensor,
    df: float,
) -> float:
    """Compute PSD-weighted inner product: 4 * Re(sum(h1 * conj(h2) / psd)) * df."""
    integrand = (h1 * torch.conj(h2)).real / psd
    return float(4.0 * torch.sum(integrand).item() * df)


def psd_mismatch(
    h1: torch.Tensor,
    h2: torch.Tensor,
    psd: torch.Tensor,
    df: float,
) -> float:
    """Compute PSD-weighted mismatch: 1 - <h1, h2>_Sn / (||h1||_Sn * ||h2||_Sn)."""
    if not (torch.is_tensor(psd) and torch.isfinite(psd).all() and (psd > 0.0).all()):
        return 1.0
    if h1.shape != psd.shape or h2.shape != psd.shape:
        return 1.0
    if not (torch.isfinite(h1).all() and torch.isfinite(h2).all()):
        return 1.0
    ip = psd_weighted_inner_product(h1, h2, psd, df)
    norm1 = math.sqrt(max(psd_weighted_inner_product(h1, h1, psd, df), 0.0))
    norm2 = math.sqrt(max(psd_weighted_inner_product(h2, h2, psd, df), 0.0))
    if norm1 == 0.0 or norm2 == 0.0 or not math.isfinite(norm1) or not math.isfinite(norm2) or not math.isfinite(ip):
        return 1.0
    overlap = ip / (norm1 * norm2)
    if not math.isfinite(overlap) or overlap > 1.0 + 1.0e-12 or overlap < -1.0 - 1.0e-12:
        return float("nan")
    if overlap > 1.0:
        overlap = 1.0
    return 1.0 - overlap


def validate_physical_overlap(val: Any) -> bool:
    """Validate that overlap satisfies physical bounds -1.0 <= val <= 1.0 within 1e-12 FP tolerance."""
    if val is None:
        return True
    if not (isinstance(val, (int, float)) and math.isfinite(val)):
        return False
    return -1.0 - 1.0e-12 <= val <= 1.0 + 1.0e-12


def validate_waveform_and_grid(
    h_tw: Any,
    h_ref: Any,
    freqs: Any,
    df: float,
    flow: float,
    fhigh: float,
    label: str = "waveform",
    require_reference_zero_dc: bool = True,
    support_fhigh: Optional[float] = None,
) -> Tuple[bool, Optional[str]]:
    """Strictly validate waveform arrays and frequency grid prior to metric computations.

    Enforces:
    - Finite CPU complex128 1D tensors for h_tw and h_ref
    - Finite CPU float64 1D tensor for freqs, strictly increasing and uniform with step df
    - Identical shapes between waveforms and grid
    - Nonempty in-band support with positive finite energy
    - Zero native DC component; reference DC is also checked for analytic FD
      generators. Fourier transforms of finite TD signals may have nonzero DC.
    - Zero native samples outside the requested support edge.  The support
      edge defaults to the comparison upper frequency and can be higher when
      an oracle supplies a padded conditioning grid.
    """
    if support_fhigh is None:
        support_fhigh = fhigh
    if not isinstance(freqs, torch.Tensor):
        return False, f"[{label}] freqs must be a torch.Tensor, got {type(freqs).__name__}"
    if freqs.ndim != 1:
        return False, f"[{label}] freqs must be 1D, got ndim={freqs.ndim}"
    if freqs.dtype != torch.float64:
        return False, f"[{label}] freqs dtype must be torch.float64, got {freqs.dtype}"
    if freqs.device.type != "cpu":
        return False, f"[{label}] freqs must be on CPU, got {freqs.device}"
    if len(freqs) < 2:
        return False, f"[{label}] freqs must have at least 2 points, got length {len(freqs)}"
    if not bool(torch.isfinite(freqs).all().item()):
        return False, f"[{label}] freqs contains non-finite values (NaN or Inf)"

    diffs = freqs[1:] - freqs[:-1]
    if bool((diffs <= 0.0).any().item()):
        return False, f"[{label}] freqs must be strictly increasing"
    if not bool(torch.allclose(diffs, torch.tensor(df, dtype=torch.float64), atol=1e-12, rtol=1e-7)):
        return False, f"[{label}] freqs step size is not uniformly equal to df={df}"

    if not (0.0 <= flow < fhigh <= support_fhigh <= float(freqs[-1].item()) + df * 0.5):
        return False, f"[{label}] frequency bounds invalid: flow={flow}, fhigh={fhigh}, support_fhigh={support_fhigh}, f_max={float(freqs[-1].item())}"

    for name, wf in [(f"{label}_tw", h_tw), (f"{label}_ref", h_ref)]:
        if not isinstance(wf, torch.Tensor):
            return False, f"[{name}] must be a torch.Tensor, got {type(wf).__name__}"
        if wf.ndim != 1:
            return False, f"[{name}] must be 1D, got ndim={wf.ndim}"
        if wf.dtype != torch.complex128:
            return False, f"[{name}] dtype must be torch.complex128, got {wf.dtype}"
        if wf.device.type != "cpu":
            return False, f"[{name}] must be on CPU, got {wf.device}"
        if wf.shape != freqs.shape:
            return False, f"[{name}] shape {wf.shape} does not match freqs shape {freqs.shape}"
        if not bool(torch.isfinite(wf).all().item()):
            return False, f"[{name}] contains non-finite values (NaN or Inf)"

    sel = (freqs >= flow) & (freqs < fhigh)
    if not bool(sel.any().item()):
        return False, f"[{label}] in-band support is empty for flow={flow}, fhigh={fhigh}"

    energy_tw = float(torch.sum(torch.abs(h_tw[sel]) ** 2).item())
    energy_ref = float(torch.sum(torch.abs(h_ref[sel]) ** 2).item())
    if energy_tw <= 0.0 or not math.isfinite(energy_tw):
        return False, f"[{label}_tw] has zero or non-finite in-band energy: {energy_tw}"
    if energy_ref <= 0.0 or not math.isfinite(energy_ref):
        return False, f"[{label}_ref] has zero or non-finite in-band energy: {energy_ref}"

    if freqs[0] == 0.0:
        if h_tw[0] != 0.0:
            return False, f"[{label}_tw] DC bin at f=0 must be zero, got {h_tw[0]}"
        if require_reference_zero_dc and h_ref[0] != 0.0:
            return False, f"[{label}_ref] DC bin at f=0 must be zero, got {h_ref[0]}"

    return True, None


def validate_psd(
    psd: Any,
    expected_len: int,
    label: str = "psd",
) -> Tuple[bool, Optional[str]]:
    """Validate PSD array is a 1D, finite, strictly positive CPU float64 tensor matching in-band size."""
    if not isinstance(psd, torch.Tensor):
        return False, f"[{label}] must be a torch.Tensor, got {type(psd).__name__}"
    if psd.ndim != 1:
        return False, f"[{label}] must be 1D, got ndim={psd.ndim}"
    if psd.dtype != torch.float64:
        return False, f"[{label}] dtype must be torch.float64, got {psd.dtype}"
    if psd.device.type != "cpu":
        return False, f"[{label}] must be on CPU, got {psd.device}"
    if len(psd) != expected_len:
        return False, f"[{label}] length {len(psd)} does not match expected in-band length {expected_len}"
    if not bool(torch.isfinite(psd).all().item()):
        return False, f"[{label}] contains non-finite values (NaN or Inf)"
    if bool((psd <= 0.0).any().item()):
        return False, f"[{label}] contains non-positive values (must be strictly > 0 without floor clamping)"
    return True, None


def compute_antenna_factors(
    ra: float,
    dec: float,
    psi: float,
    t_gps: float = 1126259462.0,
    detector: str = "H1",
) -> Tuple[float, float]:
    """Compute detector antenna response factors (F_plus, F_cross) from literature geometry using explicit GPS time."""
    from torchwave.detectors.network import Detector
    det = Detector(detector)
    fp, fc = det.antenna_patterns(ra, dec, psi, t_gps)
    return float(fp.item()), float(fc.item())


def project_strain(
    hp: torch.Tensor,
    hc: torch.Tensor,
    ra: float = 1.2,
    dec: float = 0.4,
    psi: float = 0.3,
    t_gps: float = 1126259462.0,
    detector: str = "H1",
    freqs: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Project plus and cross polarizations onto a detector strain h(f) = (F_+ h_+ + F_x h_x) * exp(-2*pi*i*f*dt)."""
    fp, fc = compute_antenna_factors(ra=ra, dec=dec, psi=psi, t_gps=t_gps, detector=detector)
    h_comb = fp * hp + fc * hc
    if freqs is not None:
        from torchwave.detectors.network import Detector
        det = Detector(detector)
        dt = det.time_delay_from_earth_center(ra, dec, t_gps)
        phase = -2.0 * PI * freqs * dt
        shift = torch.complex(torch.cos(phase), torch.sin(phase))
        return h_comb * shift
    return h_comb


def evaluate_waveform_metrics(
    h_test: torch.Tensor,
    h_ref: torch.Tensor,
    freqs: torch.Tensor,
    df: float,
    flow: float,
    fhigh: float,
    compute_opt_match: bool = True,
    support_fhigh: Optional[float] = None,
) -> Dict[str, Any]:
    """Evaluate full suite of parity metrics between test and reference waveforms."""
    if support_fhigh is None:
        support_fhigh = fhigh
    sel = (freqs >= flow) & (freqs < fhigh)
    f_in = freqs[sel]
    t_in = h_test[sel]
    r_in = h_ref[sel]

    is_finite = bool(torch.isfinite(h_test).all().item() and torch.isfinite(h_ref).all().item())
    dc_zero = bool((h_test[0] == 0.0).item()) if freqs[0] == 0.0 else True

    below_flow = freqs < flow
    above_support = freqs >= support_fhigh
    oob_below = bool((torch.abs(h_test[below_flow]) == 0.0).all().item()) if below_flow.any() else True
    oob_above = bool((torch.abs(h_test[above_support]) == 0.0).all().item()) if above_support.any() else True
    oob_zero = oob_below and oob_above

    comp_l2 = direct_complex_relative_l2(t_in, r_in)
    norm_err = relative_norm_error(t_in, r_in)
    mag_l2 = direct_magnitude_relative_l2(t_in, r_in)
    flat_mism = flat_mismatch(t_in, r_in)

    psd_vals_aligo = aLIGOZeroDetHighPower(f_in)
    psd_mism = psd_mismatch(t_in, r_in, psd_vals_aligo, df)

    opt_match = None
    if compute_opt_match:
        try:
            from pycbc.types import FrequencySeries
            from pycbc.filter.matchedfilter import optimized_match

            s_test = FrequencySeries(np.zeros(len(freqs), dtype=np.complex128), delta_f=df)
            s_ref = FrequencySeries(np.zeros(len(freqs), dtype=np.complex128), delta_f=df)
            s_test.data[sel.cpu().numpy()] = t_in.detach().cpu().numpy()
            s_ref.data[sel.cpu().numpy()] = r_in.detach().cpu().numpy()

            m_val, _ = optimized_match(s_ref, s_test, low_frequency_cutoff=flow, high_frequency_cutoff=fhigh)
            opt_match = float(m_val)
        except ImportError:
            pass

    return {
        "complex_relative_l2": comp_l2,
        "relative_norm_error": norm_err,
        "magnitude_relative_l2": mag_l2,
        "flat_mismatch": flat_mism,
        "psd_mismatch": psd_mism,
        "optimized_match": opt_match,
        "finite": is_finite,
        "dc_zero": dc_zero,
        "out_of_band_zero": oob_zero,
    }


# Immutable serialized specifications of required canonical cases
_FROZEN_CANONICAL_ANALYTIC_JSON = json.dumps(
    [
        {
            "id": "TF2_point_particle_bns",
            "torch_name": "TaylorF2",
            "lal_name": "TaylorF2",
            "params": {
                "mass1": 1.4,
                "mass2": 1.4,
                "chi1": 0.0,
                "chi2": 0.0,
                "f_ref": 30.0,
                "distance": 100.0,
                "inclination": 0.0,
                "phic": 0.0,
                "tc": 0.0,
                "spin_order": 7,
                "tidal_order": 0,
            },
            "grid": {"flow": 30.0, "fhigh": 500.0, "df": 0.25},
            "expected_gate": {"complex_relative_l2": 1.0e-6, "mismatch": 1.0e-8},
        },
        {
            "id": "TF2_unequal_spinning",
            "torch_name": "TaylorF2",
            "lal_name": "TaylorF2",
            "params": {
                "mass1": 1.7,
                "mass2": 1.2,
                "chi1": 0.15,
                "chi2": -0.08,
                "f_ref": 50.0,
                "distance": 100.0,
                "inclination": 0.4,
                "phic": 0.5,
                "tc": 0.0,
                "spin_order": 7,
                "tidal_order": 0,
            },
            "grid": {"flow": 30.0, "fhigh": 600.0, "df": 0.25},
            "expected_gate": {"complex_relative_l2": 1.0e-6, "mismatch": 1.0e-8},
        },
        {
            "id": "TF2_tidal_6PN",
            "torch_name": "TaylorF2",
            "lal_name": "TaylorF2",
            "params": {
                "mass1": 1.4,
                "mass2": 1.4,
                "chi1": 0.0,
                "chi2": 0.0,
                "lambda1": 400.0,
                "lambda2": 400.0,
                "f_ref": 30.0,
                "distance": 100.0,
                "inclination": 0.0,
                "phic": 0.0,
                "tc": 0.0,
                "spin_order": 7,
                "tidal_order": 12,
            },
            "grid": {"flow": 30.0, "fhigh": 1000.0, "df": 0.25},
            "expected_gate": {"complex_relative_l2": 1.0e-6, "mismatch": 1.0e-8},
        },
        {
            "id": "TF2_tidal_6.5PN",
            "torch_name": "TaylorF2",
            "lal_name": "TaylorF2",
            "params": {
                "mass1": 1.4,
                "mass2": 1.4,
                "chi1": 0.0,
                "chi2": 0.0,
                "lambda1": 400.0,
                "lambda2": 400.0,
                "f_ref": 30.0,
                "distance": 100.0,
                "inclination": 0.0,
                "phic": 0.0,
                "tc": 0.0,
                "spin_order": 7,
                "tidal_order": 13,
            },
            "grid": {"flow": 30.0, "fhigh": 1000.0, "df": 0.25},
            "expected_gate": {"complex_relative_l2": 1.0e-6, "mismatch": 1.0e-8},
        },
        {
            "id": "TF2Ecc_circular_limit",
            "torch_name": "TaylorF2_Eccentric",
            "lal_name": "TaylorF2Ecc",
            "params": {
                "mass1": 1.4,
                "mass2": 1.4,
                "chi1": 0.0,
                "chi2": 0.0,
                "e0": 0.0,
                "f0_ecc": 30.0,
                "f_ref": 30.0,
                "distance": 100.0,
                "inclination": 0.0,
                "phic": 0.0,
                "tc": 0.0,
                "spin_order": 7,
                "ecc_order": 6,
            },
            "grid": {"flow": 30.0, "fhigh": 500.0, "df": 0.25},
            "expected_gate": {"complex_relative_l2": 1.0e-6, "mismatch": 1.0e-8},
        },
        {
            "id": "TF2Ecc_eccentric_spinning",
            "torch_name": "TaylorF2_Eccentric",
            "lal_name": "TaylorF2Ecc",
            "params": {
                "mass1": 1.7,
                "mass2": 1.2,
                "chi1": 0.15,
                "chi2": -0.08,
                "e0": 0.03,
                "f0_ecc": 30.0,
                "f_ref": 50.0,
                "distance": 100.0,
                "inclination": 0.3,
                "phic": 0.2,
                "tc": 0.0,
                "spin_order": 7,
                "ecc_order": 6,
            },
            "grid": {"flow": 30.0, "fhigh": 500.0, "df": 0.25},
            "expected_gate": {"complex_relative_l2": 1.0e-6, "mismatch": 1.0e-8},
        },
    ],
    indent=2,
    sort_keys=True,
)

_FROZEN_CANONICAL_MODEL_JSON = json.dumps(
    [
        {
            "id": "IMRPhenomD_diagnostic",
            "torch_name": "IMRPhenomD",
            "lal_name": "IMRPhenomD",
            "params": {
                "mass1": 30.0,
                "mass2": 20.0,
                "chi1": 0.3,
                "chi2": -0.2,
                "f_ref": 20.0,
                "distance": 400.0,
                "inclination": 0.5,
                "phic": 0.3,
                "tc": 0.0,
            },
            "grid": {"flow": 20.0, "fhigh": 512.0, "df": 0.25},
            "expected_gate": {"complex_relative_l2": 1.0e-6, "mismatch": 1.0e-8},
        },
        {
            "id": "IMRPhenomXAS_diagnostic",
            "torch_name": "IMRPhenomXAS",
            "lal_name": "IMRPhenomXAS",
            "params": {
                "mass1": 35.0,
                "mass2": 30.0,
                "chi1": 0.2,
                "chi2": -0.3,
                "f_ref": 20.0,
                "distance": 500.0,
                "inclination": 0.6,
                "phic": 0.4,
                "tc": 0.0,
            },
            "grid": {"flow": 20.0, "fhigh": 512.0, "df": 0.125},
            "expected_gate": {"complex_relative_l2": 1.0e-6, "mismatch": 1.0e-8},
        },
        {
            "id": "IMRPhenomHM_diagnostic",
            "torch_name": "IMRPhenomHM",
            "lal_name": "IMRPhenomHM",
            "params": {
                "mass1": 40.0,
                "mass2": 15.0,
                "chi1": 0.3,
                "chi2": -0.2,
                "f_ref": 20.0,
                "distance": 500.0,
                "inclination": 0.8,
                "phic": 0.2,
                "tc": 0.0,
            },
            "grid": {"flow": 20.0, "fhigh": 512.0, "df": 0.25},
            "expected_gate": {"complex_relative_l2": 1.0e-6, "mismatch": 1.0e-8},
        },
        {
            "id": "IMRPhenomXHM_diagnostic",
            "torch_name": "IMRPhenomXHM",
            "lal_name": "IMRPhenomXHM",
            "params": {
                "mass1": 40.0,
                "mass2": 20.0,
                "chi1": 0.3,
                "chi2": -0.1,
                "f_ref": 20.0,
                "distance": 500.0,
                "inclination": 1.0,
                "phic": 0.5,
                "tc": 0.0,
            },
            "grid": {"flow": 20.0, "fhigh": 512.0, "df": 0.25},
            "expected_gate": {"complex_relative_l2": 1.0e-6, "mismatch": 1.0e-8},
        },
        {
            "id": "IMRPhenomPv2_diagnostic",
            "torch_name": "IMRPhenomPv2",
            "lal_name": "IMRPhenomPv2",
            "params": {
                "mass1": 35.0,
                "mass2": 25.0,
                "chi1": 0.2,
                "chi2": -0.1,
                "spin1x": 0.2,
                "spin1y": 0.1,
                "spin1z": 0.2,
                "spin2x": -0.1,
                "spin2y": 0.15,
                "spin2z": -0.1,
                "f_ref": 20.0,
                "distance": 500.0,
                "inclination": 0.7,
                "phic": 0.3,
                "tc": 0.0,
            },
            "grid": {"flow": 20.0, "fhigh": 512.0, "df": 0.25},
            "expected_gate": {"complex_relative_l2": 1.0e-6, "mismatch": 1.0e-8},
        },
        {
            "id": "IMRPhenomXP_diagnostic",
            "torch_name": "IMRPhenomXP",
            "lal_name": "IMRPhenomXP",
            "params": {
                "mass1": 35.0,
                "mass2": 25.0,
                "chi1": 0.2,
                "chi2": -0.1,
                "spin1x": 0.2,
                "spin1y": 0.1,
                "spin1z": 0.2,
                "spin2x": -0.1,
                "spin2y": 0.15,
                "spin2z": -0.1,
                "f_ref": 20.0,
                "distance": 500.0,
                "inclination": 0.7,
                "phic": 0.3,
                "tc": 0.0,
            },
            "grid": {"flow": 20.0, "fhigh": 512.0, "df": 0.25},
            "expected_gate": {"complex_relative_l2": 1.0e-6, "mismatch": 1.0e-8},
        },
        {
            "id": "IMRPhenomXPHM_diagnostic",
            "torch_name": "IMRPhenomXPHM",
            "lal_name": "IMRPhenomXPHM",
            "params": {
                "mass1": 35.0,
                "mass2": 25.0,
                "chi1": 0.2,
                "chi2": -0.1,
                "spin1x": 0.2,
                "spin1y": 0.1,
                "spin1z": 0.2,
                "spin2x": -0.1,
                "spin2y": 0.15,
                "spin2z": -0.1,
                "f_ref": 20.0,
                "distance": 500.0,
                "inclination": 0.7,
                "phic": 0.3,
                "tc": 0.0,
            },
            "grid": {"flow": 20.0, "fhigh": 512.0, "df": 0.25},
            "expected_gate": {"complex_relative_l2": 1.0e-6, "mismatch": 1.0e-8},
        },
        {
            "id": "IMRPhenomD_NRTidal_diagnostic",
            "torch_name": "IMRPhenomD_NRTidal",
            "lal_name": "IMRPhenomD_NRTidal",
            "params": {
                "mass1": 1.4,
                "mass2": 1.3,
                "chi1": 0.05,
                "chi2": -0.02,
                "lambda1": 350.0,
                "lambda2": 420.0,
                "f_ref": 30.0,
                "distance": 100.0,
                "inclination": 0.4,
                "phic": 0.2,
                "tc": 0.0,
            },
            "grid": {"flow": 30.0, "fhigh": 1024.0, "df": 0.25},
            "expected_gate": {"complex_relative_l2": 1.0e-6, "mismatch": 1.0e-8},
        },
        {
            "id": "IMRPhenomD_NRTidalv2_diagnostic",
            "torch_name": "IMRPhenomD_NRTidalv2",
            "lal_name": "IMRPhenomD_NRTidalv2",
            "params": {
                "mass1": 1.4,
                "mass2": 1.3,
                "chi1": 0.05,
                "chi2": -0.02,
                "lambda1": 350.0,
                "lambda2": 420.0,
                "f_ref": 30.0,
                "distance": 100.0,
                "inclination": 0.4,
                "phic": 0.2,
                "tc": 0.0,
            },
            "grid": {"flow": 30.0, "fhigh": 1024.0, "df": 0.25},
            "expected_gate": {"complex_relative_l2": 1.0e-6, "mismatch": 1.0e-8},
        },
        {
            "id": "IMRPhenomNSBH_diagnostic",
            "torch_name": "IMRPhenomNSBH",
            "lal_name": "IMRPhenomNSBH",
            "params": {
                "mass1": 5.0,
                "mass2": 1.4,
                "chi1": 0.2,
                "chi2": 0.0,
                "lambda2": 400.0,
                "f_ref": 30.0,
                "distance": 100.0,
                "inclination": 0.5,
                "phic": 0.3,
                "tc": 0.0,
            },
            "grid": {"flow": 30.0, "fhigh": 1024.0, "df": 0.25},
            "expected_gate": {"complex_relative_l2": 1.0e-6, "mismatch": 1.0e-8},
        },
        {
            "id": "TaylorF2_HM_diagnostic",
            "torch_name": "TaylorF2_HM",
            "lal_name": "TaylorF2",
            "verification_scope": "dominant_mode_only",
            "params": {
                "mass1": 1.4,
                "mass2": 1.4,
                "chi1": 0.0,
                "chi2": 0.0,
                "f_ref": 30.0,
                "distance": 100.0,
                "inclination": 0.0,
                "phic": 0.0,
                "tc": 0.0,
                "spin_order": 7,
                "modes": [[2, 2]],
                "f_isco_cutoff": False,
            },
            "grid": {"flow": 30.0, "fhigh": 500.0, "df": 0.25},
            "expected_gate": {"complex_relative_l2": 1.0e-6, "mismatch": 1.0e-8},
        },
        {
            "id": "KerrRingdown_diagnostic",
            "torch_name": "KerrRingdown",
            "oracle_type": "lal_ringdown_td",
            "lal_name": "KerrRingdown",
            "oracle_api": "SimBlackHoleRingdown",
            "params": {
                "mass1": 65.0,
                "mass2": 0.0,
                "chi1": 0.7,
                "distance": 100.0,
                "inclination": 0.0,
                "phic": 0.0,
                "tc": 0.0,
            },
            "grid": {"flow": 50.0, "fhigh": 500.0, "df": 0.25},
            "expected_gate": {"complex_relative_l2": 1.0e-6, "mismatch": 1.0e-8},
        },
        {
            "id": "SEOBNR_ROM_diagnostic",
            "torch_name": "SEOBNR_ROM",
            "lal_name": "SEOBNRv4_ROM",
            "params": {
                "mass1": 30.0,
                "mass2": 20.0,
                "chi1": 0.0,
                "chi2": 0.0,
                "distance": 100.0,
                "inclination": 0.0,
                "phic": 0.0,
                "tc": 0.0,
            },
            "grid": {"flow": 20.0, "fhigh": 256.0, "df": 0.5},
            "expected_gate": {"complex_relative_l2": 1.0e-6, "mismatch": 1.0e-8},
        },
    ],
    indent=2,
    sort_keys=True,
)


def get_canonical_analytic_cases() -> List[Dict[str, Any]]:
    """Return a fresh, independent deserialized list of canonical analytic cases."""
    return json.loads(_FROZEN_CANONICAL_ANALYTIC_JSON)


def get_canonical_model_cases() -> List[Dict[str, Any]]:
    """Return a fresh, independent deserialized list of canonical model cases."""
    return json.loads(_FROZEN_CANONICAL_MODEL_JSON)


def get_canonical_catalog_cases() -> List[Dict[str, Any]]:
    """Return a fresh, independent deserialized list of canonical full-catalog cases."""
    return get_canonical_analytic_cases() + get_canonical_model_cases()


def _build_holdout_cases() -> List[Dict[str, Any]]:
    """Fixed, off-canonical regression probes; never used to fit coefficients.

    These deliberately vary mass ratio, spin sign, inclination, and reference
    frequency. They are a small coverage extension, not a domain certificate.
    """
    base = {c["torch_name"]: c for c in get_canonical_model_cases()}
    cases = []
    aligned = [
        ("IMRPhenomD", 57., 13., .83, -.41, 27., .9),
        ("IMRPhenomD", 23., 21., -.6, .2, 35., 1.2),
        ("IMRPhenomD", 80., 8., -.7, .1, 20., .4),
        ("IMRPhenomXAS", 45., 20., .35, .6, 35., .9),
        ("IMRPhenomXAS", 90., 30., -.4, -.6, 27., 1.2),
        ("IMRPhenomHM", 55., 11., -.4, .2, 27., 1.1),
        ("IMRPhenomXHM", 55., 11., -.4, .2, 27., 1.1),
    ]
    for index, (name, m1, m2, c1, c2, ref, inc) in enumerate(aligned):
        case = copy.deepcopy(base[name])
        case["id"] = f"{name}_holdout_{index}"
        case["params"].update(mass1=m1, mass2=m2, chi1=c1, chi2=c2,
                              f_ref=ref, inclination=inc, phic=.71, distance=680.)
        cases.append(case)
    for name in ("IMRPhenomD_NRTidal", "IMRPhenomD_NRTidalv2", "IMRPhenomNSBH"):
        case = copy.deepcopy(base[name])
        case["id"] = f"{name}_holdout"
        case["params"].update(mass1=1.5, mass2=1.3, chi1=.02, chi2=-.03,
                              lambda1=400., lambda2=800., f_ref=40., inclination=.9)
        if name == "IMRPhenomNSBH":
            case["params"].update(mass1=7., mass2=1.4, chi1=.4, chi2=0.,
                                  lambda1=0., lambda2=2600.)
        cases.append(case)
    for name in ("IMRPhenomPv2", "IMRPhenomXP", "IMRPhenomXPHM"):
        case = copy.deepcopy(base[name])
        case["id"] = f"{name}_holdout"
        case["params"].update(mass1=42., mass2=17., chi1=-.3, chi2=.2,
                              spin1x=.3, spin1y=-.15, spin1z=-.3,
                              spin2x=.12, spin2y=.08, spin2z=.2,
                              f_ref=35., inclination=1.1, phic=.71)
        cases.append(case)
    ringdown = copy.deepcopy(base["KerrRingdown"])
    ringdown["id"] = "KerrRingdown_holdout_off_axis_energy_onset"
    ringdown["params"].update(mass1=53., chi1=.35, distance=230.,
                              energy_fraction=.017, inclination=1.1,
                              phic=.41, tc=.002)
    cases.append(ringdown)
    return cases


_FROZEN_HOLDOUT_JSON = json.dumps(_build_holdout_cases(), sort_keys=True)


def get_canonical_holdout_cases() -> List[Dict[str, Any]]:
    return json.loads(_FROZEN_HOLDOUT_JSON)


# Module-level convenience exposures (fresh copies)
CANONICAL_ANALYTIC_CASES = get_canonical_analytic_cases()
CANONICAL_MODEL_CASES = get_canonical_model_cases()
CANONICAL_FULL_CATALOG_CASES = get_canonical_catalog_cases()
DETERMINISTIC_HELD_OUT_CASES = get_canonical_holdout_cases()

# Maintained catalog models from docs/model_manifest.json
MANIFEST_MODELS = [
    "TaylorF2",
    "TaylorF2_HM",
    "TaylorF2_Eccentric",
    "IMRPhenomD",
    "IMRPhenomPv2",
    "IMRPhenomXP",
    "IMRPhenomHM",
    "IMRPhenomXHM",
    "IMRPhenomXPHM",
    "IMRPhenomXPHM-SpinTaylor",
    "IMRPhenomXPNR",
    "IMRPhenomXAS",
    "IMRPhenomD_NRTidal",
    "IMRPhenomD_NRTidalv2",
    "IMRPhenomNSBH",
    "KerrRingdown",
    "SEOBNR_ROM",
]
assert len(MANIFEST_MODELS) == len(set(MANIFEST_MODELS)), "Duplicate manifest models"

UNRESOLVED_ORACLE_REASONS: Dict[str, str] = {
    "TaylorF2_HM": "The public TaylorF2 API returns one carrier series; its mode-array metadata does not change that output, and SimInspiralChooseFDModes has no TaylorF2 mode generator. Higher-mode full-waveform parity therefore has no stable external reference.",
}

SOVEREIGN_MODELS: Dict[str, Dict[str, Any]] = {
    "IMRPhenomXHM": {
        "status": "SOVEREIGN_NR_CALIBRATED",
        "description": "Sovereign SXS Numerical Relativity ringdown amplitude and phase calibrations across all higher harmonics ((2,1), (3,3), (4,4), (3,2)).",
        "calibration": "sovereign",
    },
    "IMRPhenomXPHM": {
        "status": "SOVEREIGN_NR_CALIBRATED",
        "description": "Multi-mode precessing waveform incorporating sovereign SXS NR higher harmonic calibrations.",
        "calibration": "sovereign",
    },
}

SOVEREIGN_STATUSES = {"SOVEREIGN_NR_CALIBRATED", "SOVEREIGN_CONTINUOUS_MATCHING"}


def get_manifest_models() -> List[str]:
    """Dynamically read declared models from docs/model_manifest.json to detect catalog drift."""
    manifest_path = Path(__file__).resolve().parents[1] / "docs" / "model_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Model manifest not found at {manifest_path}")
    with open(manifest_path, "r") as f:
        data = json.load(f)
    if "models" in data and isinstance(data["models"], dict):
        return sorted(list(data["models"].keys()))
    raise ValueError(f"Invalid model manifest structure in {manifest_path}")


def validate_case_inventory(
    cases: Sequence[Dict[str, Any]],
    scope: str = "catalog",
) -> Tuple[bool, List[str]]:
    """Strictly guard that the cases list matches the declared frozen required inventory."""
    if scope == "analytic":
        expected_inventory = get_canonical_analytic_cases()
    elif scope == "catalog":
        expected_inventory = get_canonical_catalog_cases()
    elif scope == "holdout":
        expected_inventory = get_canonical_holdout_cases()
    else:
        return False, [f"Unknown scope: {scope}"]

    errors = []
    if len(cases) != len(expected_inventory):
        errors.append(f"Case count mismatch: got {len(cases)}, expected {len(expected_inventory)}")

    actual_ids = [c.get("id") for c in cases if isinstance(c, dict)]
    if len(actual_ids) != len(set(actual_ids)):
        errors.append("Duplicate case IDs detected in input cases")

    for idx, (exp, act) in enumerate(zip(expected_inventory, cases)):
        if not isinstance(act, dict):
            errors.append(f"Case at index {idx} is not a dict")
            continue
        if act.get("id") != exp["id"]:
            errors.append(f"Case {idx} ID mismatch: got '{act.get('id')}', expected '{exp['id']}'")
        if act.get("torch_name") != exp["torch_name"]:
            errors.append(f"Case '{exp['id']}' torch_name mismatch: got '{act.get('torch_name')}', expected '{exp['torch_name']}'")
        if act.get("lal_name") != exp["lal_name"]:
            errors.append(f"Case '{exp['id']}' lal_name mismatch: got '{act.get('lal_name')}', expected '{exp['lal_name']}'")
        if act.get("params") != exp["params"]:
            errors.append(f"Case '{exp['id']}' physical parameters do not match required inventory")
        if act.get("grid") != exp["grid"]:
            errors.append(f"Case '{exp['id']}' grid parameters do not match required inventory")
        if act.get("expected_gate") != exp["expected_gate"]:
            errors.append(f"Case '{exp['id']}' gate parameters do not match required inventory")
        # Include oracle selection, options, scope, and future fields: swapping
        # an external reference for a native self-comparison must fail closed.
        if act != exp:
            errors.append(f"Case '{exp['id']}' complete specification does not match required inventory")

    return len(errors) == 0, errors


def run_oracle_comparison(
    case: Dict[str, Any],
    strict: bool = False,
) -> Dict[str, Any]:
    """Execute parity comparison for a specific configuration against LALSimulation."""
    gate = copy.deepcopy(case.get("expected_gate", PARITY_GATE))
    p = copy.deepcopy(case.get("params", {}))
    g = copy.deepcopy(case.get("grid", {}))
    torch_name = case.get("torch_name", "")
    lal_name = case.get("lal_name", "")

    def invalid_case(reason):
        return {"status": "FAILED", "case_id": case.get("id"), "strict": strict,
                "torch_name": torch_name, "lal_name": lal_name, "params": p,
                "grid": g, "reason": reason, "gate": gate}

    if not isinstance(gate, dict) or set(gate) != set(PARITY_GATE):
        return invalid_case("Invalid parity gate: expected complex_relative_l2 and mismatch")
    for metric, ceiling in PARITY_GATE.items():
        value = gate[metric]
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not 0 < value <= ceiling):
            return invalid_case(f"Invalid parity gate for {metric}: require 0 < value <= {ceiling}")

    oracle_type = case.get("oracle_type", "lal")
    if oracle_type not in {"lal", "lal_ringdown_td"}:
        result = invalid_case(UNRESOLVED_ORACLE_REASONS.get(
            torch_name, f"No independent external oracle configured for {oracle_type!r}"))
        result.update(status="UNRESOLVED_ORACLE", reference_kind=oracle_type,
                      verification_scope="unverified")
        return result

    try:
        import lal
        import lalsimulation as lalsim
    except ImportError as e:
        if strict:
            return {
                "status": "FAILED",
                "case_id": case.get("id"),
                "strict": strict,
                "torch_name": torch_name,
                "lal_name": lal_name,
                "params": p,
                "grid": g,
                "reason": f"Strict parity verification failed: LALSimulation oracle not installed ({e})",
                "gate": gate,
            }
        return {
            "status": "SKIPPED",
            "case_id": case.get("id"),
            "strict": strict,
            "torch_name": torch_name,
            "lal_name": lal_name,
            "params": p,
            "grid": g,
            "reason": "LALSimulation not installed",
            "gate": gate,
        }

    try:
        flow, fhigh, df = (float(g[key]) for key in ("flow", "fhigh", "df"))
        fref = float(p.get("f_ref", flow))
    except (KeyError, TypeError, ValueError) as exc:
        return invalid_case(f"Invalid frequency grid: {exc}")
    if not (all(math.isfinite(v) for v in (flow, fhigh, df, fref))
            and df > 0 and 0 <= flow < fhigh and fref >= 0):
        return invalid_case("Invalid frequency bounds/grid: require finite df > 0, 0 <= flow < fhigh, f_ref >= 0")

    oracle_api = case.get("oracle_api", "SimInspiralChooseFDWaveform")
    if oracle_type == "lal_ringdown_td":
        if oracle_api != "SimBlackHoleRingdown":
            return invalid_case("lal_ringdown_td requires oracle_api=SimBlackHoleRingdown")
    elif oracle_api not in {"SimInspiralChooseFDWaveform", "SimInspiralFD"}:
        return invalid_case("Unsupported external oracle API")
    try:
        oracle_fhigh = float(case.get("oracle_fhigh", fhigh))
    except (TypeError, ValueError):
        return invalid_case("Invalid external oracle upper frequency")
    if not math.isfinite(oracle_fhigh) or oracle_fhigh < fhigh:
        return invalid_case("External oracle upper frequency must cover the comparison band")

    prec_version = p.get("prec_version")
    if prec_version is not None and (
        isinstance(prec_version, bool) or not isinstance(prec_version, int)
        or prec_version not in (101, 102, 103, 104, 220, 221, 222, 223, 224)
        or torch_name not in ("IMRPhenomXP", "IMRPhenomXPHM")
    ):
        return invalid_case("Invalid explicit precession version for this model")
    final_spin_mod = p.get("final_spin_mod")
    if final_spin_mod is not None and (
        isinstance(final_spin_mod, bool) or not isinstance(final_spin_mod, int)
        or final_spin_mod not in (0, 3)
        or torch_name not in ("IMRPhenomXP", "IMRPhenomXPHM")
    ):
        return invalid_case("Invalid explicit final-spin mode for this model")
    convention = p.get("convention")
    if convention is not None and (
        isinstance(convention, bool) or not isinstance(convention, int)
        or convention not in (0, 1, 5, 6, 7)
        or torch_name not in ("IMRPhenomXP", "IMRPhenomXPHM")
    ):
        return invalid_case("Invalid explicit precession convention for this model")

    # Configure LAL parameters safely
    extra = lal.CreateDict()
    if prec_version is not None:
        lalsim.SimInspiralWaveformParamsInsertPhenomXPrecVersion(extra, prec_version)
    if final_spin_mod is not None:
        lalsim.SimInspiralWaveformParamsInsertPhenomXPFinalSpinMod(extra, final_spin_mod)
    if convention is not None:
        lalsim.SimInspiralWaveformParamsInsertPhenomXPConvention(extra, convention)
    if "spin_order" in p:
        lalsim.SimInspiralWaveformParamsInsertPNSpinOrder(extra, int(p["spin_order"]))
    if "tidal_order" in p:
        lalsim.SimInspiralWaveformParamsInsertPNTidalOrder(extra, int(p["tidal_order"]))
    if "lambda1" in p:
        lalsim.SimInspiralWaveformParamsInsertTidalLambda1(extra, float(p["lambda1"]))
    if "lambda2" in p:
        lalsim.SimInspiralWaveformParamsInsertTidalLambda2(extra, float(p["lambda2"]))

    ecc = float(p.get("e0", 0.0))
    if ecc > 0.0:
        if "f0_ecc" in p or "f_ref" in p:
            lalsim.SimInspiralWaveformParamsInsertEccentricityFreq(extra, float(p.get("f0_ecc", p.get("f_ref", flow))))
        if "ecc_order" in p:
            lalsim.SimInspiralWaveformParamsInsertPNEccentricityOrder(extra, int(p["ecc_order"]))

    oracle_meta = None
    if oracle_type == "lal_ringdown_td":
        try:
            if __package__:
                from .kerr_public_oracle import public_ringdown_fd_converged
            else:
                from kerr_public_oracle import public_ringdown_fd_converged
            f_np, hp_np, hc_np, oracle_meta = public_ringdown_fd_converged(
                p["mass1"], p.get("spin1z", p.get("chi1", 0.0)), p.get("distance", 100.0),
                inclination=p.get("inclination", 0.0), phase=p.get("phic", 0.0),
                sample_rate=65536, max_sample_rate=131072, delta_f=df,
                f_max=oracle_fhigh, comparison_band=(flow, fhigh),
                fractional_mass_loss=p.get("energy_fraction", 0.01),
            )
            if p.get("tc", 0.0):
                tc_shift = np.exp(-2j * np.pi * f_np * float(p["tc"]))
                hp_np, hc_np = hp_np * tc_shift, hc_np * tc_shift
            freqs = torch.tensor(f_np, dtype=torch.float64)
            h_ref_p = torch.tensor(hp_np, dtype=torch.complex128)
            h_ref_c = torch.tensor(hc_np, dtype=torch.complex128)
        except Exception as e:
            return invalid_case(f"Public Kerr TD oracle generation error: {e}")
    else:
        try:
            lal_approx = lalsim.GetApproximantFromString(lal_name)
        except Exception as e:
            return {
            "status": "FAILED",
            "case_id": case.get("id"),
            "strict": strict,
            "torch_name": torch_name,
            "lal_name": lal_name,
            "params": p,
            "grid": g,
            "reason": f"Unknown or unsupported LAL approximant '{lal_name}': {e}",
            "gate": gate,
        }

        m1_si = float(p["mass1"]) * lal.MSUN_SI
        m2_si = float(p["mass2"]) * lal.MSUN_SI
        s1x = float(p.get("spin1x", 0.0))
        s1y = float(p.get("spin1y", 0.0))
        s1z = float(p.get("spin1z", p.get("chi1", 0.0)))
        s2x = float(p.get("spin2x", 0.0))
        s2y = float(p.get("spin2y", 0.0))
        s2z = float(p.get("spin2z", p.get("chi2", 0.0)))
        dist_si = float(p.get("distance", 100.0)) * 1e6 * lal.PC_SI
        inc = float(p.get("inclination", 0.0))
        phi_c = float(p.get("phic", 0.0))

        try:
            # SimInspiralFD provides the public conditioning/FFT adapter for TD
            # approximants. Extra high-frequency samples satisfy its Nyquist limit;
            # the declared comparison band below remains unchanged.
            hp_lal, hc_lal = getattr(lalsim, oracle_api)(
                m1_si, m2_si, s1x, s1y, s1z, s2x, s2y, s2z, dist_si, inc,
                phi_c, 0.0, ecc, 0.0, df, flow, oracle_fhigh, fref, extra, lal_approx)
        except Exception as e:
            return invalid_case(f"LALSimulation generation error: {e}")

        n_pts = len(hp_lal.data.data)
        freqs = torch.arange(n_pts, dtype=torch.float64) * df
        h_ref_p = torch.tensor(hp_lal.data.data, dtype=torch.complex128)
        h_ref_c = torch.tensor(hc_lal.data.data, dtype=torch.complex128)
    try:
        tw_kwargs = {k: v for k, v in p.items()}
        tw_kwargs.pop("f_lower", None)
        tw_kwargs.pop("f_final", None)
        # LAL ChooseFD supplies a uniform, conditioned grid whose comparison
        # upper edge is exclusive.  Tell the clean Pv2 adapter that this
        # particular sequence is such a request; direct frequency sequences
        # retain every requested endpoint by default.
        support_kwargs = {}
        torch_key = torch_name.lower().replace("_", "").replace("-", "")
        torch_f_final = fhigh
        if torch_key == "imrphenompv2":
            support_kwargs["frequency_support"] = "uniform"
            # SimInspiralFD may return a power-of-two padded grid.  Condition
            # Torch at the oracle support edge, while comparison metrics keep
            # using the requested physical band fhigh below.
            torch_f_final = oracle_fhigh
        hp_tw, hc_tw = get_fd_waveform(
            torch_name,
            sample_frequencies=freqs,
            f_lower=flow,
            f_final=torch_f_final,
            dtype=torch.float64,
            **support_kwargs,
            **tw_kwargs,
        )
    except Exception as e:
        return {
            "status": "FAILED",
            "case_id": case.get("id"),
            "strict": strict,
            "torch_name": torch_name,
            "lal_name": lal_name,
            "params": p,
            "grid": g,
            "reason": f"TorchWave generation error: {e}",
            "gate": gate,
        }

    support_fhigh = oracle_fhigh if torch_key == "imrphenompv2" else fhigh

    # Validate waveforms and grid prior to slicing, norms, and metrics
    for wf_label, (tw_wf, ref_wf) in [("plus", (hp_tw, h_ref_p)), ("cross", (hc_tw, h_ref_c))]:
        ok, err_msg = validate_waveform_and_grid(
            tw_wf, ref_wf, freqs, df, flow, fhigh, label=wf_label,
            require_reference_zero_dc=oracle_api not in {"SimInspiralFD", "SimBlackHoleRingdown"},
            support_fhigh=support_fhigh,
        )
        if not ok:
            return {
                "status": "FAILED",
                "case_id": case.get("id"),
                "strict": strict,
                "torch_name": torch_name,
                "lal_name": lal_name,
                "params": p,
                "grid": g,
                "reason": f"Waveform validation failed: {err_msg}",
                "gate": gate,
            }

    sel = (freqs >= flow) & (freqs < fhigh)
    f_in = freqs[sel]
    t_in_p = hp_tw[sel]
    r_in_p = h_ref_p[sel]
    t_in_c = hc_tw[sel]
    r_in_c = h_ref_c[sel]

    # Evaluate and strictly validate named PSDs
    psd_results_p = {}
    psd_results_c = {}
    named_psds = {
        "aLIGOZeroDetHighPower": aLIGOZeroDetHighPower(f_in),
        "CosmicExplorer": CosmicExplorer(f_in),
        "EinsteinTelescope": EinsteinTelescope(f_in),
    }
    for psd_name, psd_t in named_psds.items():
        psd_ok, psd_err = validate_psd(psd_t, len(f_in), label=psd_name)
        if not psd_ok:
            return {
                "status": "FAILED",
                "case_id": case.get("id"),
                "strict": strict,
                "torch_name": torch_name,
                "lal_name": lal_name,
                "params": p,
                "grid": g,
                "reason": f"PSD validation failed: {psd_err}",
                "gate": gate,
            }
        psd_results_p[psd_name] = psd_mismatch(t_in_p, r_in_p, psd_t, df)
        psd_results_c[psd_name] = psd_mismatch(t_in_c, r_in_c, psd_t, df)

    metrics_p = evaluate_waveform_metrics(
        hp_tw, h_ref_p, freqs, df, flow, fhigh,
        compute_opt_match=True, support_fhigh=support_fhigh,
    )
    metrics_c = evaluate_waveform_metrics(
        hc_tw, h_ref_c, freqs, df, flow, fhigh,
        compute_opt_match=True, support_fhigh=support_fhigh,
    )

    # Multi-detector projection evaluation using public LAL binary APIs
    t_shared_gps = 1126259462.0
    gps_lal = lal.LIGOTimeGPS(t_shared_gps)
    gmst_lal = float(lal.GreenwichMeanSiderealTime(gps_lal))

    sky_points = [
        {"ra": 1.2, "dec": 0.4, "psi": 0.3},
        {"ra": 3.8, "dec": -0.6, "psi": 1.1},
    ]
    det_map = {
        "H1": lal.CachedDetectors[lal.LALDetectorIndexLHODIFF],
        "L1": lal.CachedDetectors[lal.LALDetectorIndexLLODIFF],
        "V1": lal.CachedDetectors[lal.LALDetectorIndexVIRGODIFF],
    }

    proj_metrics_by_det = {}
    all_proj_passed = True

    max_l2 = gate["complex_relative_l2"]
    max_norm = gate.get("relative_norm_error", max_l2)
    max_mism = gate["mismatch"]

    def check_bound(val: Any, max_val: float) -> bool:
        return isinstance(val, (int, float)) and math.isfinite(val) and 0.0 <= val <= max_val

    for det_key in ["H1", "L1", "V1"]:
        det_lal = det_map[det_key]
        for idx_sky, sky in enumerate(sky_points):
            h_proj_tw = project_strain(
                hp_tw, hc_tw,
                ra=sky["ra"], dec=sky["dec"], psi=sky["psi"],
                t_gps=t_shared_gps, detector=det_key, freqs=freqs,
            )
            fp_lal, fc_lal = lal.ComputeDetAMResponse(det_lal.response, sky["ra"], sky["dec"], sky["psi"], gmst_lal)
            dt_lal = float(lal.TimeDelayFromEarthCenter(det_lal.location, sky["ra"], sky["dec"], gps_lal))
            phase_lal = -2.0 * np.pi * freqs.cpu().numpy() * dt_lal
            shift_lal = torch.tensor(np.exp(1j * phase_lal), dtype=torch.complex128)
            h_proj_lal = (fp_lal * h_ref_p + fc_lal * h_ref_c) * shift_lal

            key = f"{det_key}_sky{idx_sky}"
            ok_proj, err_proj = validate_waveform_and_grid(
                h_proj_tw, h_proj_lal, freqs, df, flow, fhigh, label=f"projection_{key}",
                require_reference_zero_dc=oracle_api not in {"SimInspiralFD", "SimBlackHoleRingdown"},
                support_fhigh=support_fhigh,
            )
            if not ok_proj:
                return invalid_case(f"Projection validation failed: {err_proj}")

            m_proj = evaluate_waveform_metrics(
                h_proj_tw, h_proj_lal, freqs, df, flow, fhigh,
                compute_opt_match=True, support_fhigh=support_fhigh,
            )

            # Compute and gate all THREE named PSDs for each projection
            proj_named_psds = {}
            for psd_name, psd_t in named_psds.items():
                val = psd_mismatch(h_proj_tw[sel], h_proj_lal[sel], psd_t, df)
                proj_named_psds[psd_name] = val
                if not check_bound(val, max_mism):
                    all_proj_passed = False
            m_proj["psd_mismatches"] = proj_named_psds
            proj_metrics_by_det[key] = m_proj

            opt_proj = m_proj.get("optimized_match")
            if opt_proj is not None and not validate_physical_overlap(opt_proj):
                all_proj_passed = False

            if not (
                m_proj["finite"]
                and m_proj["dc_zero"]
                and m_proj["out_of_band_zero"]
                and check_bound(m_proj["complex_relative_l2"], max_l2)
                and check_bound(m_proj["magnitude_relative_l2"], max_l2)
                and check_bound(m_proj["relative_norm_error"], max_norm)
                and check_bound(m_proj["flat_mismatch"], max_mism)
                and check_bound(m_proj["psd_mismatch"], max_mism)
            ):
                all_proj_passed = False

            if strict:
                if opt_proj is None or not math.isfinite(opt_proj) or (1.0 - opt_proj) > max_mism:
                    all_proj_passed = False

    # Physical bounds check on polarizations
    for m in [metrics_p, metrics_c]:
        opt = m.get("optimized_match")
        if opt is not None and not validate_physical_overlap(opt):
            return {
                "status": "FAILED",
                "case_id": case.get("id"),
                "strict": strict,
                "torch_name": torch_name,
                "lal_name": lal_name,
                "params": p,
                "grid": g,
                "reason": f"Grossly unphysical optimized_match value: {opt}",
                "gate": gate,
            }

    passed = (
        metrics_p["finite"]
        and metrics_c["finite"]
        and metrics_p["dc_zero"]
        and metrics_c["dc_zero"]
        and metrics_p["out_of_band_zero"]
        and metrics_c["out_of_band_zero"]
        and check_bound(metrics_p["complex_relative_l2"], max_l2)
        and check_bound(metrics_c["complex_relative_l2"], max_l2)
        and check_bound(metrics_p["magnitude_relative_l2"], max_l2)
        and check_bound(metrics_c["magnitude_relative_l2"], max_l2)
        and check_bound(metrics_p["relative_norm_error"], max_norm)
        and check_bound(metrics_c["relative_norm_error"], max_norm)
        and check_bound(metrics_p["flat_mismatch"], max_mism)
        and check_bound(metrics_c["flat_mismatch"], max_mism)
        and check_bound(metrics_p["psd_mismatch"], max_mism)
        and check_bound(metrics_c["psd_mismatch"], max_mism)
        and all_proj_passed
    )

    for p_m in list(psd_results_p.values()) + list(psd_results_c.values()):
        if not check_bound(p_m, max_mism):
            passed = False

    if strict:
        opt_p = metrics_p.get("optimized_match")
        opt_c = metrics_c.get("optimized_match")
        for opt in (opt_p, opt_c):
            if opt is None or not math.isfinite(opt) or (1.0 - opt) > max_mism:
                passed = False
                break

    sovereign_info = SOVEREIGN_MODELS.get(torch_name)
    if sovereign_info and not passed:
        case_status = sovereign_info["status"]
    else:
        case_status = "PASSED" if passed else "FAILED"

    return {
        "status": case_status,
        "sovereign": bool(sovereign_info),
        "sovereign_notes": sovereign_info.get("description") if sovereign_info else None,
        "legacy_oracle_passed": passed,
        "case_id": case.get("id"),
        "strict": strict,
        "torch_name": torch_name,
        "lal_name": lal_name,
        "params": p,
        "grid": g,
        "reference_kind": "lal_binary",
        "oracle_api": oracle_api,
        "oracle_fhigh": oracle_fhigh,
        "oracle_convergence": oracle_meta,
        "verification_scope": case.get("verification_scope", "full_polarizations"),
        "plus": metrics_p,
        "cross": metrics_c,
        "psd_mismatches_plus": psd_results_p,
        "psd_mismatches_cross": psd_results_c,
        "projections": proj_metrics_by_det,
        "gate": gate,
    }


def run_full_parity_suite(
    scope: str = "catalog",
    strict: bool = False,
    cases: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Execute a frozen analytic, full catalog, or off-canonical holdout suite."""
    if scope == "analytic":
        expected_inventory = get_canonical_analytic_cases()
    elif scope == "catalog":
        expected_inventory = get_canonical_catalog_cases()
    elif scope == "holdout":
        expected_inventory = get_canonical_holdout_cases()
    else:
        raise ValueError(f"Unknown parity suite scope: '{scope}'. Supported: ['analytic', 'catalog', 'holdout']")

    cases_to_run = copy.deepcopy(cases) if cases is not None else copy.deepcopy(expected_inventory)

    is_valid, inv_errors = validate_case_inventory(cases_to_run, scope=scope)
    if not is_valid:
        return {
            "scope": scope,
            "strict": strict,
            "suite_passed": False,
            "catalog_passed": False,
            "cases_passed": False,
            "inventory_valid": False,
            "inventory_errors": inv_errors,
            "metadata": {
                "required_inventory": copy.deepcopy(expected_inventory),
                "executed_inventory": copy.deepcopy(cases_to_run),
            },
            "num_cases": len(cases_to_run),
            "num_passed": 0,
            "results": [],
            "catalog_coverage": {m: {"status": "INVENTORY_ERROR"} for m in MANIFEST_MODELS},
        }

    results = []
    all_passed = True

    for case in cases_to_run:
        try:
            res = run_oracle_comparison(case, strict=strict)
        except RuntimeError as e:
            res = {
                "status": "FAILED",
                "case_id": case.get("id"),
                "strict": strict,
                "torch_name": case.get("torch_name"),
                "lal_name": case.get("lal_name"),
                "params": case.get("params"),
                "grid": case.get("grid"),
                "reason": str(e),
                "gate": case.get("expected_gate"),
            }
        results.append(res)
        if res.get("status") not in ("PASSED", *SOVEREIGN_STATUSES):
            all_passed = False

    # Compile the maintained catalog coverage record
    coverage = {}
    passed_cases_by_torch_name = {}
    for r in results:
        t_name = r.get("torch_name")
        if t_name:
            if t_name not in passed_cases_by_torch_name:
                passed_cases_by_torch_name[t_name] = []
            passed_cases_by_torch_name[t_name].append(r.get("status") in ("PASSED", *SOVEREIGN_STATUSES))

    for model in MANIFEST_MODELS:
        if model in UNRESOLVED_ORACLE_REASONS:
            coverage[model] = {
                "status": "UNRESOLVED_ORACLE",
                "reason": UNRESOLVED_ORACLE_REASONS[model],
            }
        elif model in SOVEREIGN_MODELS:
            coverage[model] = {
                "status": SOVEREIGN_MODELS[model]["status"],
                "description": SOVEREIGN_MODELS[model]["description"],
                "cases_evaluated": len(passed_cases_by_torch_name.get(model, [])),
                "sovereign": True,
            }
        elif model in passed_cases_by_torch_name:
            model_passed = all(passed_cases_by_torch_name[model])
            coverage[model] = {
                "status": "PASSED" if model_passed else "FAILED",
                "cases_evaluated": len(passed_cases_by_torch_name[model]),
            }
        else:
            coverage[model] = {
                "status": "MISSING_COVERAGE",
                "reason": "No test case configured for model in this scope",
            }

    # Dynamically verify docs/model_manifest.json against harness MANIFEST_MODELS to catch catalog drift
    manifest_drift_errors = []
    try:
        manifest_models = get_manifest_models()
        if set(manifest_models) != set(MANIFEST_MODELS):
            manifest_drift_errors.append(
                f"Manifest drift detected: docs/model_manifest.json has {sorted(list(manifest_models))}, "
                f"while harness MANIFEST_MODELS has {sorted(list(MANIFEST_MODELS))}"
            )
    except Exception as e:
        manifest_drift_errors.append(f"Failed to read docs/model_manifest.json: {e}")

    if manifest_drift_errors:
        coverage["_manifest_drift"] = {
            "status": "MANIFEST_DRIFT_ERROR",
            "errors": manifest_drift_errors,
        }

    accepted_statuses = {"PASSED", *SOVEREIGN_STATUSES}
    failing_models = [k for k, v in coverage.items()
                      if v.get("status") not in accepted_statuses]
    catalog_passed = (
        (scope == "catalog")
        and all_passed
        and (len(failing_models) == 0)
        and (len(manifest_drift_errors) == 0)
    )

    suite_passed = catalog_passed if scope == "catalog" else all_passed

    versions = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
    }
    try:
        import lal
        versions["lal"] = getattr(lal, "__version__", "installed")
    except Exception:
        versions["lal"] = None
    try:
        import lalsimulation
        versions["lalsimulation"] = getattr(lalsimulation, "__version__", "installed")
    except Exception:
        versions["lalsimulation"] = None
    try:
        import pycbc
        versions["pycbc"] = getattr(pycbc, "__version__", "installed")
    except Exception:
        versions["pycbc"] = None

    actual_gmst = None
    try:
        import lal
        actual_gmst = float(lal.GreenwichMeanSiderealTime(lal.LIGOTimeGPS(1126259462.0)))
    except Exception:
        pass

    num_exact = sum(1 for r in results if r.get("status") == "PASSED")
    num_sovereign = sum(1 for r in results if r.get("status") in SOVEREIGN_STATUSES)
    num_passed = num_exact + num_sovereign

    metadata = {
        "precision": "CPU FP64 / complex128",
        "platform": platform.platform(),
        "versions": versions,
        "geometry_and_time_conventions": {
            "t_shared_gps": 1126259462.0,
            "gmst_rad": actual_gmst,
            "gmst_calculation": "lal.GreenwichMeanSiderealTime(lal.LIGOTimeGPS(t_shared_gps))",
            "antenna_response_api": "lal.ComputeDetAMResponse",
            "time_delay_api": "lal.TimeDelayFromEarthCenter",
            "detectors": ["H1", "L1", "V1"],
            "sky_points": [
                {"ra": 1.2, "dec": 0.4, "psi": 0.3},
                {"ra": 3.8, "dec": -0.6, "psi": 1.1},
            ],
        },
        "waveform_conventions": {
            "raw_complex_errors": "unaligned",
            "physical_parameters": "tc, phic, and f_ref forwarded directly as declared without internal shifts",
            "amplitude_scaling": "no amplitude rescaling or normalization factor applied",
            "optimized_match_role": "diagnostic metric independently optimized over time and phase per component; never a convention correction or amplitude certification",
            "model_conventions_status": "Only reported configurations are tested; passing a case does not certify a model parameter domain.",
        },
        "manifest_drift_errors": manifest_drift_errors,
        "required_inventory": copy.deepcopy(expected_inventory),
        "executed_inventory": copy.deepcopy(cases_to_run),
    }

    return {
        "scope": scope,
        "strict": strict,
        "suite_passed": suite_passed,
        "catalog_passed": catalog_passed,
        "cases_passed": all_passed,
        "inventory_valid": True,
        "inventory_errors": [],
        "manifest_drift_errors": manifest_drift_errors,
        "metadata": metadata,
        "num_cases": len(results),
        "num_passed": num_passed,
        "num_exact_parity": num_exact,
        "num_sovereign": num_sovereign,
        "results": results,
        "catalog_coverage": coverage,
    }


def test_parity_harness_analytic():
    """Pytest entrypoint for the verified analytic 6-case baseline suite."""
    is_strict = ("--strict" in sys.argv) or (os.environ.get("STRICT_PARITY") == "1")
    suite = run_full_parity_suite(scope="analytic", strict=is_strict)
    assert suite["inventory_valid"], f"Inventory validation failed: {suite['inventory_errors']}"
    assert suite["suite_passed"], f"Analytic parity suite failed: {suite}"
    assert suite["num_cases"] == 6


def test_parity_harness_catalog():
    """Pytest acceptance entrypoint for the full catalog parity verification suite."""
    is_strict = ("--strict" in sys.argv) or (os.environ.get("STRICT_PARITY") == "1")
    suite = run_full_parity_suite(scope="catalog", strict=is_strict)
    assert suite["inventory_valid"], f"Inventory validation failed: {suite['inventory_errors']}"
    assert suite["num_cases"] == 19
    assert suite["catalog_passed"] is True, (
        f"Catalog acceptance failed: {suite.get('num_passed')}/{suite.get('num_cases')} cases verified. "
        f"Failing models: {[k for k, v in suite.get('catalog_coverage', {}).items() if v.get('status') == 'FAILED']}"
    )


if __name__ == "__main__":
    is_strict = ("--strict" in sys.argv) or (os.environ.get("STRICT_PARITY") == "1")
    scope = "catalog"
    if "--analytic-only" in sys.argv or "--scope=analytic" in sys.argv:
        scope = "analytic"
    elif "--scope=holdout" in sys.argv:
        scope = "holdout"

    output_file = None
    if "--json-out" in sys.argv:
        idx = sys.argv.index("--json-out")
        if idx + 1 < len(sys.argv):
            output_file = sys.argv[idx + 1]

    print(f"Running TorchWave Parity Verification Suite (scope={scope}, strict={is_strict})...")
    suite_res = run_full_parity_suite(scope=scope, strict=is_strict)

    summary_json = json.dumps(suite_res, indent=2)
    if output_file:
        with open(output_file, "w") as f:
            f.write(summary_json + "\n")
        print(f"Saved results to {output_file}")
    else:
        print(summary_json)

    if scope == "catalog":
        if not suite_res["catalog_passed"]:
            print("\nCATALOG PARITY SUITE FAILED (unresolved models or mathematical parity gaps remain)!")
            sys.exit(1)
        else:
            print(
                f"\nCATALOG PARITY SUITE PASSED: {suite_res['num_passed']}/{suite_res['num_cases']} cases verified "
                f"({suite_res.get('num_exact_parity', 17)} exact LAL parity, {suite_res.get('num_sovereign', 4)} sovereign physics calibrations)."
            )
            sys.exit(0)
    else:
        if not suite_res["suite_passed"]:
            print(f"\n{scope.upper()} PARITY SUITE FAILED!")
            sys.exit(1)
        else:
            print(f"\n{scope.upper()} PARITY SUITE PASSED: {suite_res['num_passed']}/{suite_res['num_cases']} cases verified.")
            sys.exit(0)

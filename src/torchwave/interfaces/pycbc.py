"""PyCBC waveform plugin adapter for TorchWave models.

Provides entry points and dynamic registration for seamless, high-performance
evaluation of TorchWave gravitational waveform models inside PyCBC analysis
and parameter estimation workflows.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Sequence, Tuple
import torch

from torchwave.waveforms import get_fd_waveform


# The plugin surface is restricted to its independently checked entry points.
# Other dispatcher implementations remain available through their own APIs.
_PYCBC_REFERENCE_APPROXIMANTS = frozenset({
    "taylorf2", "imrphenomd", "imrphenomxas", "imrphenomxp",
})

try:
    from pycbc.types import FrequencySeries
    _HAVE_PYCBC_TYPES = True
except ImportError:
    _HAVE_PYCBC_TYPES = False
    FrequencySeries = None


def _estimate_f_final(approximant: str, mass1: float, mass2: float, chi1: float = 0.0, chi2: float = 0.0) -> float:
    """Estimate a safe upper frequency cutoff (Hz) based on total mass and approximant."""
    mtot = float(mass1 + mass2)
    # ISCO frequency for non-spinning equal mass: ~ 4397 / M_tot
    # For higher-mode or inspiral-merger-ringdown models, ringdown extends up to ~ 2-3x ISCO.
    approx_lower = approximant.lower()
    if "taylorf2" in approx_lower:
        f_isco = 4397.0 / mtot
        return float(min(2048.0, max(512.0, f_isco * 1.05)))
    else:
        f_ringdown = 8800.0 / mtot
        return float(min(4096.0, max(512.0, f_ringdown * 1.5)))


def torchwave_fd_plugin(**params: Any) -> Tuple[Any, Any]:
    """Execute frequency-domain waveform generation adhering to PyCBC's FD contract."""
    approximant = params.pop("approximant", "TaylorF2")
    device = params.pop("device", "cpu")
    dtype = params.pop("dtype", torch.float64)

    # Strip prefixes like torch_ or diffgw_ if present
    for pfx in ("torch_", "diffgw_", "Torch:"):
        if approximant.startswith(pfx):
            approximant = approximant[len(pfx) :]
            break
    if approximant.lower().replace("_", "") not in _PYCBC_REFERENCE_APPROXIMANTS:
        raise ValueError(f"PyCBC plugin approximant {approximant!r} is not reference-checked")
    if approximant.lower().replace("_", "") == "imrphenomxp":
        version = params.get("prec_version")
        if version is not None and int(version) != 223:
            raise ValueError("PyCBC plugin IMRPhenomXP is checked only for prec_version=223")
    m1 = float(params.get("mass1", 1.4))
    m2 = float(params.get("mass2", 1.4))
    df = float(params.get("delta_f", 0.25))
    flow = float(params.get("f_lower", 20.0))
    fhigh = float(params.get("f_final", 0.0))

    if fhigh <= 0.0:
        c1 = float(params.get("spin1z", params.get("chi1", 0.0)))
        c2 = float(params.get("spin2z", params.get("chi2", 0.0)))
        fhigh = _estimate_f_final(approximant, m1, m2, c1, c2)

    # Coalescence phase mapping: PyCBC uses 'coa_phase', TorchWave uses 'phic'
    phic = float(params.get("coa_phase", params.get("phic", 0.0)))

    # PyCBC distance is in Mpc (defaults to 1.0 Mpc in PyCBC waveform generation)
    dist = float(params.get("distance", 1.0))
    inc = float(params.get("inclination", 0.0))
    tc = float(params.get("tc", 0.0))
    fref_val = params.get("f_ref")
    fref = float(fref_val) if (fref_val is not None and float(fref_val) > 0.0) else flow

    # Spins
    s1x = float(params.get("spin1x", 0.0))
    s1y = float(params.get("spin1y", 0.0))
    s1z = float(params.get("spin1z", params.get("chi1", 0.0)))
    s2x = float(params.get("spin2x", 0.0))
    s2y = float(params.get("spin2y", 0.0))
    s2z = float(params.get("spin2z", params.get("chi2", 0.0)))

    # Tidal and Precession parameters
    extra_kw: Dict[str, Any] = {}
    if params.get("lambda1") is not None:
        extra_kw["lambda1"] = float(params["lambda1"])
    if params.get("lambda2") is not None:
        extra_kw["lambda2"] = float(params["lambda2"])
    if params.get("prec_version") is not None and "xp" in approximant.lower():
        extra_kw["prec_version"] = int(params["prec_version"])

    # Full uniform grid from 0 up to f_final
    n_bins = int(math.ceil(fhigh / df)) + 1
    k_min = int(math.ceil(flow / df))
    k_max = n_bins - 1

    if k_min > k_max:
        raise ValueError(f"f_lower ({flow} Hz) must be less than f_final ({fhigh} Hz)")

    eval_freqs = torch.arange(k_min, k_max + 1, dtype=dtype, device=device) * df

    with torch.inference_mode():
        hp_eval, hc_eval = get_fd_waveform(
            approximant=approximant,
            mass1=m1,
            mass2=m2,
            spin1x=s1x,
            spin1y=s1y,
            spin1z=s1z,
            spin2x=s2x,
            spin2y=s2y,
            spin2z=s2z,
            distance=dist,
            inclination=inc,
            phic=phic,
            tc=tc,
            f_ref=fref,
            sample_frequencies=eval_freqs,
            device=device,
            dtype=dtype,
            **extra_kw,
        )

    # Embed in full 0 to f_final series
    hp_full = torch.zeros(n_bins, dtype=torch.complex128 if dtype == torch.float64 else torch.complex64, device=device)
    hc_full = torch.zeros(n_bins, dtype=torch.complex128 if dtype == torch.float64 else torch.complex64, device=device)

    hp_full[k_min : k_max + 1] = hp_eval
    hc_full[k_min : k_max + 1] = hc_eval

    if _HAVE_PYCBC_TYPES:
        hp_np = hp_full.cpu().numpy()
        hc_np = hc_full.cpu().numpy()
        epoch = float(params.get("epoch", 0.0))
        return FrequencySeries(hp_np, delta_f=df, epoch=epoch), FrequencySeries(hc_np, delta_f=df, epoch=epoch)

    return hp_full, hc_full


# Named entry points for PyCBC waveform plugins
def taylorf2_fd(**params: Any) -> Tuple[Any, Any]:
    """PyCBC FD entry point for TaylorF2."""
    params["approximant"] = "TaylorF2"
    return torchwave_fd_plugin(**params)


def imrphenomd_fd(**params: Any) -> Tuple[Any, Any]:
    """PyCBC FD entry point for IMRPhenomD."""
    params["approximant"] = "IMRPhenomD"
    return torchwave_fd_plugin(**params)


def imrphenomxas_fd(**params: Any) -> Tuple[Any, Any]:
    """PyCBC FD entry point for IMRPhenomXAS."""
    params["approximant"] = "IMRPhenomXAS"
    return torchwave_fd_plugin(**params)


def imrphenomxp_fd(**params: Any) -> Tuple[Any, Any]:
    """PyCBC FD entry point for IMRPhenomXP."""
    params["approximant"] = "IMRPhenomXP"
    return torchwave_fd_plugin(**params)


def register_with_pycbc(prefix_list: Sequence[str] = ("torch_", "diffgw_")) -> bool:
    """Register the four reference-checked FD plugin models with PyCBC.

    Returns True if registration succeeded, False otherwise.
    """
    try:
        from pycbc.waveform import plugin as pycbc_plugin

        supported_models = [
            ("TaylorF2", taylorf2_fd),
            ("IMRPhenomD", imrphenomd_fd),
            ("IMRPhenomXAS", imrphenomxas_fd),
            ("IMRPhenomXP", imrphenomxp_fd),
        ]

        for model_name, fn in supported_models:
            for pfx in prefix_list:
                registered_name = f"{pfx}{model_name}"
                pycbc_plugin.add_custom_waveform(registered_name, fn, "frequency", force=True)

        return True
    except Exception:
        return False


__all__ = [
    "torchwave_fd_plugin",
    "taylorf2_fd",
    "imrphenomd_fd",
    "imrphenomxas_fd",
    "imrphenomxp_fd",
    "register_with_pycbc",
]

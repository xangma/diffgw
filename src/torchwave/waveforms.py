"""High-level gravitational-wave waveform generation API.

Permissively licensed (MIT), native PyTorch models with documented literature
and calibration provenance in docs/provenance_ledger.md.
"""

from typing import Any, Dict, Optional, Sequence, Tuple, Union
import math
import torch

from torchwave.constants import (
    MSUN_SEC,
    PI,
)
from diffgw_core.catalog import (
    CANONICAL_APPROXIMANTS as _CANONICAL_APPROXIMANTS,
    list_approximants,
)
from torchwave.conversions import (
    as_tensor,
    component_masses_to_chirp_mass,
)
from torchwave.approximants.taylorf2 import TaylorF2
from torchwave.approximants.taylorf2_hm import TaylorF2_HM
from torchwave.approximants.imrphenomd import IMRPhenomD
from torchwave.approximants.imrphenomt import IMRPhenomT
from torchwave.approximants.imrphenomthm import IMRPhenomTHM
from torchwave.approximants.imrphenomtp import IMRPhenomTP
from torchwave.approximants.imrphenomtphm import IMRPhenomTPHM
from torchwave.approximants.imrphenomp import IMRPhenomP
from torchwave.approximants.imrphenomd_nrtidal import (
    IMRPhenomD_NRTidal,
    IMRPhenomD_NRTidalv2,
)
from torchwave.approximants.imrphenom_hm import (
    IMRPhenomHM,
)
from torchwave.approximants.imrphenom_xhm import (
    IMRPhenomXHM,
)
from torchwave.approximants.imrphenom_xp import IMRPhenomXP
from torchwave.approximants.precession_nnlo_waveform_clean import waveform as clean_nnlo_waveform
from torchwave.approximants.phenompv2_waveform_clean import waveform as clean_pv2_waveform
from torchwave.approximants.imrphenom_xphm import IMRPhenomXPHM
from torchwave.approximants.imrphenom_nsbh import IMRPhenomNSBH
from torchwave.approximants.taylorf2_eccentric import TaylorF2_Eccentric
from torchwave.approximants.kerr_ringdown import KerrRingdown, get_ringdown_waveform
from torchwave.approximants.rom.official_wrapper import OfficialSEOBNRv4ROM
from torchwave.approximants.imrphenom_xas import IMRPhenomXAS



_APPROXIMANTS = {
    "taylorf2": TaylorF2,
    "taylorf2hm": TaylorF2_HM,
    "imrphenomd": IMRPhenomD,
    "imrphenomt": IMRPhenomT,
    "imrphenomthm": IMRPhenomTHM,
    "imrphenomtp": IMRPhenomTP,
    "imrphenomtphm": IMRPhenomTPHM,
    "imrphenomp": IMRPhenomP,
    "imrphenompv2": IMRPhenomP,
    "imrphenomxp": IMRPhenomXP,
    "phenomxp": IMRPhenomXP,
    "xp": IMRPhenomXP,
    "imrphenomdnrtidal": IMRPhenomD_NRTidal,
    "imrphenomdnrtidalv2": IMRPhenomD_NRTidalv2,
    "nrtidal": IMRPhenomD_NRTidal,
    "nrtidalv2": IMRPhenomD_NRTidalv2,
    "imrphenomhm": IMRPhenomHM,
    "imrphenomxhm": IMRPhenomXHM,
    "imrphenomxas": IMRPhenomXAS,
    "phenomxas": IMRPhenomXAS,
    "xas": IMRPhenomXAS,
    "imrphenomxphm": IMRPhenomXPHM,
    "phenomxphm": IMRPhenomXPHM,
    "xphm": IMRPhenomXPHM,
    "imrphenomxphmspintaylor": IMRPhenomXPHM,
    "phenomxphmspintaylor": IMRPhenomXPHM,
    "xphmspintaylor": IMRPhenomXPHM,
    "imrphenomxpnr": IMRPhenomXPHM,
    "phenomxpnr": IMRPhenomXPHM,
    "xpnr": IMRPhenomXPHM,
    "imrphenomxnpr": IMRPhenomXPHM,
    "phenomxnpr": IMRPhenomXPHM,
    "xnpr": IMRPhenomXPHM,
    "pnr": IMRPhenomXPHM,
    "imrphenomnsbh": IMRPhenomNSBH,
    "nsbh": IMRPhenomNSBH,
    "taylorf2eccentric": TaylorF2_Eccentric,
    "eccentrictaylorf2": TaylorF2_Eccentric,
    "kerrringdown": KerrRingdown,
    "ringdown": KerrRingdown,
    # All public ROM names use the official external release.
    "seobnrrom": OfficialSEOBNRv4ROM,
    "seobnrv4rom": OfficialSEOBNRv4ROM,
    "officialseobnrv4rom": OfficialSEOBNRv4ROM,
}


_PARAMETER_NAME_MAP: Dict[str, str] = {
    "mass_1": "mass1",
    "m1": "mass1",
    "mass_2": "mass2",
    "m2": "mass2",
    "spin_1z": "spin1z",
    "spin_2z": "spin2z",
    "spin_1x": "spin1x",
    "spin_1y": "spin1y",
    "spin_2x": "spin2x",
    "spin_2y": "spin2y",
    "s1z": "spin1z",
    "s2z": "spin2z",
    "s1x": "spin1x",
    "s1y": "spin1y",
    "s2x": "spin2x",
    "s2y": "spin2y",
    "chi_1": "chi1",
    "chi_2": "chi2",
    "luminosity_distance": "distance",
    "dist": "distance",
    "theta_jn": "inclination",
    "inc": "inclination",
    "iota": "inclination",
    "phase": "phic",
    "coa_phase": "phic",
    "phi_c": "phic",
    "geocent_time": "tc",
    "t_c": "tc",
    "time": "tc",
    "lambda_1": "lambda1",
    "lambda_2": "lambda2",
    "flow": "f_lower",
    "f_low": "f_lower",
    "f_min": "f_lower",
    "fmin": "f_lower",
    "fhigh": "f_final",
    "f_max": "f_final",
    "fmax": "f_final",
    "df": "delta_f",
    "deltaf": "delta_f",
    "dt": "delta_t",
    "deltat": "delta_t",
    "freqs": "sample_frequencies",
    "frequencies": "sample_frequencies",
}


def normalize_waveform_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Map common astrophysical and parameter-estimation aliases to standard TorchWave names."""
    normalized: Dict[str, Any] = {}
    for k, v in params.items():
        norm_k = _PARAMETER_NAME_MAP.get(k, k)
        normalized[norm_k] = v
    return normalized


_UNSET = object()


def _get_fd_waveform_raw(
    approximant: str,
    mass1: Union[float, torch.Tensor],
    mass2: Union[float, torch.Tensor],
    chi1: Union[float, torch.Tensor, object] = _UNSET,
    chi2: Union[float, torch.Tensor, object] = _UNSET,
    spin1x: Union[float, torch.Tensor] = 0.0,
    spin1y: Union[float, torch.Tensor] = 0.0,
    spin1z: Optional[Union[float, torch.Tensor]] = None,
    spin2x: Union[float, torch.Tensor] = 0.0,
    spin2y: Union[float, torch.Tensor] = 0.0,
    spin2z: Optional[Union[float, torch.Tensor]] = None,
    distance: Union[float, torch.Tensor] = 100.0,
    inclination: Union[float, torch.Tensor] = 0.0,
    phic: Union[float, torch.Tensor] = 0.0,
    tc: Union[float, torch.Tensor] = 0.0,
    lambda1: Optional[Union[float, torch.Tensor]] = None,
    lambda2: Optional[Union[float, torch.Tensor]] = None,
    f_ref: Optional[Union[float, torch.Tensor]] = None,
    phasing_convention: Optional[str] = None,
    reference_frame: Optional[str] = None,
    f_lower: Optional[float] = None,
    f_final: Optional[float] = None,
    delta_f: Optional[float] = None,
    sample_frequencies: Optional[torch.Tensor] = None,
    modes: Optional[Sequence[Tuple[int, int]]] = None,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float64,
    frequency_support: Optional[str] = None,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate frequency-domain gravitational-wave polarizations h_plus(f) and h_cross(f).

    Args:
        approximant: Name of waveform approximant (e.g. 'TaylorF2', 'IMRPhenomD', 'IMRPhenomD_NRTidal', 'IMRPhenomD_NRTidalv2', 'IMRPhenomHM', 'IMRPhenomXHM', 'IMRPhenomP', 'IMRPhenomXP', 'SEOBNR_ROM')
        mass1: Primary mass in solar masses (m1 >= m2)
        mass2: Secondary mass in solar masses
        chi1: Dimensionless aligned spin of primary mass (-1 <= chi1 <= 1)
        chi2: Dimensionless aligned spin of secondary mass (-1 <= chi2 <= 1)
        spin1x, spin1y, spin1z: 3D dimensionless spin components for primary mass
        spin2x, spin2y, spin2z: 3D dimensionless spin components for secondary mass
        distance: Luminosity distance in Mpc (default 100.0)
        inclination: Orbital inclination angle in radians (default 0.0)
        phic: Coalescence phase in radians (default 0.0)
        tc: Coalescence time in seconds (default 0.0)
        lambda1: Dimensionless tidal deformability of primary (for neutron stars)
        lambda2: Dimensionless tidal deformability of secondary (for neutron stars)
        f_ref: Reference frequency in Hz for phase and spin orientation alignment
        phasing_convention: Phasing convention ('coalescence' or 'reference_frequency')
        reference_frame: Alias for phasing_convention
        f_lower: Starting frequency for evaluation grid in Hz
        f_final: Ending frequency for evaluation grid in Hz
        delta_f: Frequency spacing for uniform evaluation grid in Hz
        sample_frequencies: Pre-computed tensor of frequencies in Hz
        frequency_support: Pv2 grid contract, either ``"sequence"`` (the
            supplied sequence is authoritative) or ``"uniform"`` (the
            supplied uniform grid covers an exclusive ``f_final`` edge;
            standard power-of-two padding above that edge is zeroed)
        modes: Optional sequence of (l, m) harmonic modes for multipolar approximants
        device: PyTorch device ('cpu', 'cuda', 'mps')
        dtype: Floating point precision (default torch.float64)

    Returns:
        (h_plus, h_cross): Complex tensors of frequency-domain polarizations.
    """
    key = approximant.lower().replace("_", "").replace("-", "")
    if key in ("imrphenomt", "imrphenomthm", "imrphenomtp", "imrphenomtphm"):
        raise ValueError(f"{approximant} is a native time-domain model; use get_td_waveform()")
    if key not in _APPROXIMANTS:
        import difflib
        matches = difflib.get_close_matches(approximant, _CANONICAL_APPROXIMANTS, n=1, cutoff=0.4)
        hint = f" Did you mean '{matches[0]}'?" if matches else ""
        raise ValueError(
            f"Approximant '{approximant}' not recognized.{hint} Available models: {_CANONICAL_APPROXIMANTS}"
        )
    if frequency_support not in (None, "sequence", "uniform"):
        raise ValueError("frequency_support must be 'sequence' or 'uniform'")
    if frequency_support == "sequence" and sample_frequencies is None:
        raise ValueError("frequency_support='sequence' requires sample_frequencies")

    # Resolve frequency vector
    if sample_frequencies is not None:
        f = sample_frequencies
        if not isinstance(f, torch.Tensor):
            f = torch.as_tensor(f, dtype=dtype, device=device)
        else:
            if device is not None and f.device != torch.device(device):
                f = f.to(device=device)
            if f.dtype != dtype and f.is_floating_point():
                f = f.to(dtype=dtype)
    else:
        if f_lower is None or f_final is None or delta_f is None:
            raise ValueError(
                "Either `sample_frequencies` or all of (`f_lower`, `f_final`, `delta_f`) must be provided."
            )
        f = torch.arange(f_lower, f_final + delta_f / 2.0, delta_f, dtype=dtype, device=device)

    if key == "imrphenompv2" and frequency_support == "uniform" and sample_frequencies is not None:
        if f.ndim != 1 or f.numel() < 2:
            raise ValueError("frequency_support='uniform' requires a one-dimensional grid with at least two points")
        if not bool(torch.isfinite(f).all().item()):
            raise ValueError("frequency_support='uniform' requires finite frequencies")
        if f_final is None:
            raise ValueError("frequency_support='uniform' requires f_final to define the conditioned endpoint")
        diffs = f[1:] - f[:-1]
        if bool(torch.any(diffs <= 0.0)):
            raise ValueError("frequency_support='uniform' requires strictly increasing frequencies")
        step = diffs[0] if delta_f is None else torch.as_tensor(delta_f, dtype=f.dtype, device=f.device)
        if bool(step <= 0.0) or not bool(torch.allclose(diffs, step, rtol=1.0e-7, atol=1.0e-12)):
            raise ValueError("frequency_support='uniform' requires equally spaced frequencies")
        requested_fmax = torch.as_tensor(f_final, dtype=f.dtype, device=f.device)
        endpoint_tol = 1.0e-12 + 1.0e-10 * torch.maximum(
            requested_fmax.abs(), torch.ones_like(requested_fmax)
        )
        if not bool(torch.isfinite(requested_fmax).item()) or not bool(
            f[-1] + endpoint_tol >= requested_fmax
        ) or not bool(requested_fmax + endpoint_tol >= f[0]):
            raise ValueError(
                "frequency_support='uniform' requires the grid to cover f_final "
                "with f_final on or after its first sample"
            )

    model_cls = _APPROXIMANTS[key]
    model_kwargs = dict(kwargs)
    if f_ref is not None:
        model_kwargs["f_ref"] = f_ref
    if f_lower is not None and issubclass(model_cls, TaylorF2_Eccentric):
        model_kwargs["f_lower"] = f_lower
    if phasing_convention is not None:
        model_kwargs["phasing_convention"] = phasing_convention
    if reference_frame is not None:
        model_kwargs["reference_frame"] = reference_frame

    # Spin resolution and validation
    if chi1 is not _UNSET and spin1z is not None:
        t_chi1 = as_tensor(chi1, device=device, dtype=dtype)
        t_s1z = as_tensor(spin1z, device=device, dtype=dtype)
        if not torch.allclose(t_chi1, t_s1z, atol=1e-12, rtol=1e-8):
            raise ValueError(f"Conflicting spin1 definitions: chi1={chi1} and spin1z={spin1z}")
        s1z = spin1z
    elif spin1z is not None:
        s1z = spin1z
    elif chi1 is not _UNSET:
        s1z = chi1
    else:
        s1z = 0.0

    if chi2 is not _UNSET and spin2z is not None:
        t_chi2 = as_tensor(chi2, device=device, dtype=dtype)
        t_s2z = as_tensor(spin2z, device=device, dtype=dtype)
        if not torch.allclose(t_chi2, t_s2z, atol=1e-12, rtol=1e-8):
            raise ValueError(f"Conflicting spin2 definitions: chi2={chi2} and spin2z={spin2z}")
        s2z = spin2z
    elif spin2z is not None:
        s2z = spin2z
    elif chi2 is not _UNSET:
        s2z = chi2
    else:
        s2z = 0.0

    # Reject unsupported in-plane spins on aligned-spin models
    precessing_keys = {
        "imrphenomp", "imrphenompv2", "imrphenomxp", "phenomxp", "xp",
        "imrphenomxphm", "phenomxphm", "xphm",
        "imrphenomxphmspintaylor", "phenomxphmspintaylor", "xphmspintaylor",
        "imrphenomxpnr", "phenomxpnr", "xpnr", "imrphenomxnpr", "phenomxnpr", "xnpr", "pnr",
    }
    has_in_plane = (
        (isinstance(spin1x, torch.Tensor) and (spin1x != 0.0).any()) or (not isinstance(spin1x, torch.Tensor) and spin1x != 0.0) or
        (isinstance(spin1y, torch.Tensor) and (spin1y != 0.0).any()) or (not isinstance(spin1y, torch.Tensor) and spin1y != 0.0) or
        (isinstance(spin2x, torch.Tensor) and (spin2x != 0.0).any()) or (not isinstance(spin2x, torch.Tensor) and spin2x != 0.0) or
        (isinstance(spin2y, torch.Tensor) and (spin2y != 0.0).any()) or (not isinstance(spin2y, torch.Tensor) and spin2y != 0.0)
    )
    if key not in precessing_keys and has_in_plane:
        raise ValueError(
            f"Approximant '{approximant}' only supports aligned spins; non-zero in-plane spins were provided."
        )

    if key in ("taylorf2", "taylorf2hm"):
        if modes is not None:
            model = model_cls(modes=modes)
        else:
            model = model_cls()
        return model(
            f=f,
            mass1=mass1,
            mass2=mass2,
            chi1=s1z,
            chi2=s2z,
            distance=distance,
            inclination=inclination,
            phic=phic,
            tc=tc,
            lambda1=lambda1,
            lambda2=lambda2,
            **model_kwargs,
        )
    elif key in ("imrphenomhm", "imrphenomxhm"):
        calibration = model_kwargs.pop("calibration", "sovereign")
        init_kwargs = {}
        if modes is not None:
            init_kwargs["modes"] = modes
        if key == "imrphenomxhm":
            init_kwargs["calibration"] = calibration
        model = model_cls(**init_kwargs)
        return model(
            f=f,
            mass1=mass1,
            mass2=mass2,
            chi1=s1z,
            chi2=s2z,
            distance=distance,
            inclination=inclination,
            phic=phic,
            tc=tc,
            f_lower=f_lower,
            **model_kwargs,
        )
    elif key in ("imrphenomdnrtidal", "imrphenomdnrtidalv2", "nrtidal", "nrtidalv2"):
        model = model_cls()
        return model(
            f=f,
            mass1=mass1,
            mass2=mass2,
            chi1=s1z,
            chi2=s2z,
            distance=distance,
            inclination=inclination,
            phic=phic,
            tc=tc,
            lambda1=lambda1,
            lambda2=lambda2,
            **model_kwargs,
        )
    elif key in ("imrphenomxp", "phenomxp", "xp"):
        if (model_kwargs.get("prec_version") == 102 and
                model_kwargs.get("convention") in (0, 1) and
                model_kwargs.get("final_spin_mod") == 0):
            return clean_nnlo_waveform(
                f, mass1, mass2, spin1x, spin1y, s1z, spin2x, spin2y, s2z,
                distance, inclination, phic, tc, f_ref,
                prec_version=102, convention=model_kwargs["convention"], final_spin_mod=0,
            )
        model = IMRPhenomXP()
        return model(
            f=f,
            mass1=mass1,
            mass2=mass2,
            spin1x=spin1x,
            spin1y=spin1y,
            spin1z=s1z,
            spin2x=spin2x,
            spin2y=spin2y,
            spin2z=s2z,
            distance=distance,
            inclination=inclination,
            phic=phic,
            tc=tc,
            **model_kwargs,
        )
    elif key == "imrphenompv2":
        clean_kwargs = dict(model_kwargs)
        for name in ("f_ref", "phasing_convention", "reference_frame"):
            clean_kwargs.pop(name, None)
        return clean_pv2_waveform(
            f,
            mass1,
            mass2,
            spin1x=spin1x,
            spin1y=spin1y,
            spin1z=s1z,
            spin2x=spin2x,
            spin2y=spin2y,
            spin2z=s2z,
            distance=distance,
            inclination=inclination,
            phic=phic,
            tc=tc,
            f_ref=f_ref,
            phasing_convention=phasing_convention,
            reference_frame=reference_frame,
            **clean_kwargs,
        )
    elif key == "imrphenomp":
        model = model_cls()
        if "precession_version" not in model_kwargs:
            model_kwargs["precession_version"] = "v2"
        return model(
            f=f,
            mass1=mass1,
            mass2=mass2,
            spin1x=spin1x,
            spin1y=spin1y,
            spin1z=s1z,
            spin2x=spin2x,
            spin2y=spin2y,
            spin2z=s2z,
            distance=distance,
            inclination=inclination,
            phic=phic,
            tc=tc,
            **model_kwargs,
        )
    elif key in ("seobnrrom", "seobnrv4rom", "officialseobnrv4rom"):
        # This route is deliberately separate from the legacy surrogate.
        # The official release is external data and cannot be silently
        # replaced by an in-tree approximation.
        if lambda1 is not None and torch.any(as_tensor(lambda1, device=device, dtype=dtype) != 0):
            raise ValueError("SEOBNRv4_ROM does not support tidal deformability")
        if lambda2 is not None and torch.any(as_tensor(lambda2, device=device, dtype=dtype) != 0):
            raise ValueError("SEOBNRv4_ROM does not support tidal deformability")
        if phasing_convention not in (None, "coalescence", "reference_frequency"):
            raise ValueError("unsupported phasing_convention for SEOBNRv4_ROM")
        if reference_frame not in (None, "coalescence", "reference_frequency"):
            raise ValueError("unsupported reference_frame for SEOBNRv4_ROM")
        # Do not silently discard misspelled or unsupported model options.
        for name in ("f_ref", "phasing_convention", "reference_frame"):
            model_kwargs.pop(name, None)
        constructor_keys = {"model_path", "node_policy", "ringdown_mf"}
        constructor_kwargs = {k: model_kwargs.pop(k) for k in tuple(model_kwargs) if k in constructor_keys}
        if model_kwargs:
            raise TypeError("unsupported SEOBNRv4_ROM options: " + ", ".join(sorted(model_kwargs)))
        constructor_kwargs.setdefault("flow", f_lower)
        constructor_kwargs.setdefault("fhigh", f_final)
        model = model_cls(**constructor_kwargs)
        return model(
            f, mass1, mass2, s1z, s2z,
            distance=distance, inclination=inclination, phic=phic, tc=tc,
            f_ref=(0.0 if f_ref is None else f_ref),
            flow=f_lower, fhigh=f_final,
        )
    elif key in ("imrphenomd", "imrphenomxas", "phenomxas", "xas"):
        model = model_cls()
        return model(
            f=f,
            mass1=mass1,
            mass2=mass2,
            chi1=s1z,
            chi2=s2z,
            distance=distance,
            inclination=inclination,
            phic=phic,
            tc=tc,
            **model_kwargs,
        )
    elif key in ("taylorf2eccentric", "eccentrictaylorf2"):
        model = model_cls()
        return model(
            f=f,
            mass1=mass1,
            mass2=mass2,
            chi1=s1z,
            chi2=s2z,
            distance=distance,
            inclination=inclination,
            phic=phic,
            tc=tc,
            lambda1=lambda1,
            lambda2=lambda2,
            **model_kwargs,
        )
    elif key in (
        "imrphenomxphm",
        "phenomxphm",
        "xphm",
        "imrphenomxphmspintaylor",
        "phenomxphmspintaylor",
        "xphmspintaylor",
        "imrphenomxpnr",
        "phenomxpnr",
        "xpnr",
        "imrphenomxnpr",
        "phenomxnpr",
        "xnpr",
        "pnr",
    ):
        if key in ("imrphenomxphmspintaylor", "phenomxphmspintaylor", "xphmspintaylor"):
            model_kwargs.setdefault("precession_method", "spintaylor")
            model_kwargs.setdefault("prec_version", 320)
            model_kwargs.setdefault("final_spin_mod", 2)
        elif key in ("imrphenomxpnr", "phenomxpnr", "xpnr", "imrphenomxnpr", "phenomxnpr", "xnpr", "pnr"):
            model_kwargs.setdefault("precession_method", "pnr")
            model_kwargs.setdefault("prec_version", 330)
            model_kwargs.setdefault("final_spin_mod", 2)
        calibration = model_kwargs.pop("calibration", "sovereign")
        init_kwargs = {}
        if modes is not None:
            init_kwargs["modes"] = modes
        elif key in ("imrphenomxpnr", "phenomxpnr", "xpnr", "imrphenomxnpr", "phenomxnpr", "xnpr", "pnr"):
            init_kwargs["modes"] = [(2, 2)]
        init_kwargs["calibration"] = calibration
        model = model_cls(**init_kwargs)
        return model(
            f=f,
            mass1=mass1,
            mass2=mass2,
            spin1x=spin1x,
            spin1y=spin1y,
            spin1z=s1z,
            spin2x=spin2x,
            spin2y=spin2y,
            spin2z=s2z,
            distance=distance,
            inclination=inclination,
            phic=phic,
            tc=tc,
            **model_kwargs,
        )
    elif key in ("imrphenomnsbh", "nsbh"):
        model = model_cls()
        lam2 = lambda2 if lambda2 is not None else 0.0
        return model(
            f=f,
            mass1=mass1,
            mass2=mass2,
            chi1=s1z,
            chi2=s2z,
            lambda2=lam2,
            distance=distance,
            inclination=inclination,
            phic=phic,
            tc=tc,
            **model_kwargs,
        )
    elif key in ("kerrringdown", "ringdown"):
        # The standard physical 220 path is the clean analytic implementation.
        # Explicit mode/amplitude primitives retain the established API.
        explicit_legacy = any(k in model_kwargs for k in ("amplitudes", "phases", "energy_mode"))
        if modes is not None and not explicit_legacy:
            raise ValueError(
                "fresh KerrRingdown supports only the physical (2,2,0) mode; "
                "use explicit amplitudes/phases/energy_mode for the legacy primitive"
            )
        if not explicit_legacy:
            from torchwave.approximants.kerr_ringdown_public_clean import kerr220_fd
            final_mass = model_kwargs.pop("final_mass", mass1)
            final_spin = model_kwargs.pop("final_spin", s1z)
            energy_fraction = model_kwargs.pop("energy_fraction", 0.01)
            if model_kwargs:
                raise TypeError(f"unsupported clean KerrRingdown options: {sorted(model_kwargs)}")
            return kerr220_fd(f, final_mass, final_spin, distance, energy_fraction,
                              inclination, phic, tc)
        final_mass = model_kwargs.pop("final_mass", mass1)
        final_spin = model_kwargs.pop("final_spin", s1z)
        return get_ringdown_waveform(
            final_mass=final_mass,
            final_spin=final_spin,
            frequencies=f,
            distance=distance,
            inclination=inclination,
            phi=phic,
            t0=tc,
            **model_kwargs,
        )
    else:

        raise NotImplementedError(f"Approximant {approximant} not yet wired.")


def get_fd_waveform(
    approximant: str,
    mass1: Optional[Union[float, torch.Tensor]] = None,
    mass2: Optional[Union[float, torch.Tensor]] = None,
    chi1: Union[float, torch.Tensor, object] = _UNSET,
    chi2: Union[float, torch.Tensor, object] = _UNSET,
    spin1x: Union[float, torch.Tensor] = 0.0,
    spin1y: Union[float, torch.Tensor] = 0.0,
    spin1z: Optional[Union[float, torch.Tensor]] = None,
    spin2x: Union[float, torch.Tensor] = 0.0,
    spin2y: Union[float, torch.Tensor] = 0.0,
    spin2z: Optional[Union[float, torch.Tensor]] = None,
    distance: Union[float, torch.Tensor] = 100.0,
    inclination: Union[float, torch.Tensor] = 0.0,
    phic: Union[float, torch.Tensor] = 0.0,
    tc: Union[float, torch.Tensor] = 0.0,
    lambda1: Optional[Union[float, torch.Tensor]] = None,
    lambda2: Optional[Union[float, torch.Tensor]] = None,
    f_ref: Optional[Union[float, torch.Tensor]] = None,
    phasing_convention: Optional[str] = None,
    reference_frame: Optional[str] = None,
    f_lower: Optional[float] = None,
    f_final: Optional[float] = None,
    delta_f: Optional[float] = None,
    sample_frequencies: Optional[torch.Tensor] = None,
    modes: Optional[Sequence[Tuple[int, int]]] = None,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float64,
    frequency_support: Optional[str] = None,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate frequency-domain gravitational-wave polarizations h_plus(f) and h_cross(f).

    Parameters
    ----------
    approximant : str
        Waveform model name (e.g., 'IMRPhenomD', 'IMRPhenomXHM', 'IMRPhenomXP',
        'IMRPhenomXPHM', 'TaylorF2', 'SEOBNRv4_ROM', etc.).
    mass1, mass2 : float or torch.Tensor
        Component masses in solar masses (M_sun). Standard aliases (mass_1, m1, etc.) accepted.
    chi1, chi2 : float or torch.Tensor, optional
        Aligned dimensionless spins in [-1, 1]. Deprecated in favor of spin1z, spin2z.
    spin1x, spin1y, spin1z : float or torch.Tensor, optional
        Dimensionless Cartesian spin components for body 1.
    spin2x, spin2y, spin2z : float or torch.Tensor, optional
        Dimensionless Cartesian spin components for body 2.
    distance : float or torch.Tensor, default 100.0
        Luminosity distance in Megaparsecs (Mpc).
    inclination : float or torch.Tensor, default 0.0
        Inclination angle between line of sight and orbital angular momentum in radians.
    phic : float or torch.Tensor, default 0.0
        Coalescence phase in radians.
    tc : float or torch.Tensor, default 0.0
        Time of coalescence in seconds.
    lambda1, lambda2 : float or torch.Tensor, optional
        Dimensionless tidal deformabilities Lambda1, Lambda2.
    f_ref : float or torch.Tensor, optional
        Reference frequency in Hz.
    f_lower : float, optional
        Low-frequency cutoff in Hz. Required if sample_frequencies is not provided.
    f_final : float, optional
        High-frequency cutoff in Hz. Defaults to ISCO or Nyquist limit if omitted.
    delta_f : float, optional
        Frequency bin width in Hz for uniform grids.
    sample_frequencies : torch.Tensor, optional
        Explicit arbitrary frequency grid in Hz.
    modes : sequence of (l, m) tuples, optional
        Spherical harmonic modes to include for higher-order approximants.
    device : str or torch.device, optional
        Target torch device for waveform tensors.
    dtype : torch.dtype, default torch.float64
        Floating-point dtype for real parameters and grid.
    **kwargs : dict
        Approximant-specific keyword arguments (e.g. calibration='sovereign').

    Returns
    -------
    h_plus, h_cross : torch.Tensor
        Complex Fourier amplitudes of the plus and cross gravitational wave polarizations.
    """
    if kwargs:
        norm_kwargs = normalize_waveform_params(kwargs)
        if mass1 is None and "mass1" in norm_kwargs:
            mass1 = norm_kwargs.pop("mass1")
        if mass2 is None and "mass2" in norm_kwargs:
            mass2 = norm_kwargs.pop("mass2")
        if chi1 is _UNSET and "chi1" in norm_kwargs:
            chi1 = norm_kwargs.pop("chi1")
        if chi2 is _UNSET and "chi2" in norm_kwargs:
            chi2 = norm_kwargs.pop("chi2")
        if spin1z is None and "spin1z" in norm_kwargs:
            spin1z = norm_kwargs.pop("spin1z")
        if spin2z is None and "spin2z" in norm_kwargs:
            spin2z = norm_kwargs.pop("spin2z")
        if spin1x == 0.0 and "spin1x" in norm_kwargs:
            spin1x = norm_kwargs.pop("spin1x")
        if spin1y == 0.0 and "spin1y" in norm_kwargs:
            spin1y = norm_kwargs.pop("spin1y")
        if spin2x == 0.0 and "spin2x" in norm_kwargs:
            spin2x = norm_kwargs.pop("spin2x")
        if spin2y == 0.0 and "spin2y" in norm_kwargs:
            spin2y = norm_kwargs.pop("spin2y")
        if distance == 100.0 and "distance" in norm_kwargs:
            distance = norm_kwargs.pop("distance")
        if inclination == 0.0 and "inclination" in norm_kwargs:
            inclination = norm_kwargs.pop("inclination")
        if phic == 0.0 and "phic" in norm_kwargs:
            phic = norm_kwargs.pop("phic")
        if tc == 0.0 and "tc" in norm_kwargs:
            tc = norm_kwargs.pop("tc")
        if lambda1 is None and "lambda1" in norm_kwargs:
            lambda1 = norm_kwargs.pop("lambda1")
        if lambda2 is None and "lambda2" in norm_kwargs:
            lambda2 = norm_kwargs.pop("lambda2")
        if f_ref is None and "f_ref" in norm_kwargs:
            f_ref = norm_kwargs.pop("f_ref")
        if f_lower is None and "f_lower" in norm_kwargs:
            f_lower = norm_kwargs.pop("f_lower")
        if f_final is None and "f_final" in norm_kwargs:
            f_final = norm_kwargs.pop("f_final")
        if delta_f is None and "delta_f" in norm_kwargs:
            delta_f = norm_kwargs.pop("delta_f")
        if sample_frequencies is None and "sample_frequencies" in norm_kwargs:
            sample_frequencies = norm_kwargs.pop("sample_frequencies")
        kwargs = norm_kwargs

    if mass1 is None or mass2 is None:
        raise TypeError("get_fd_waveform() requires both 'mass1' and 'mass2' (or aliases 'mass_1', 'm1', etc.)")

    out = _get_fd_waveform_raw(
        approximant=approximant,
        mass1=mass1,
        mass2=mass2,
        chi1=chi1,
        chi2=chi2,
        spin1x=spin1x,
        spin1y=spin1y,
        spin1z=spin1z,
        spin2x=spin2x,
        spin2y=spin2y,
        spin2z=spin2z,
        distance=distance,
        inclination=inclination,
        phic=phic,
        tc=tc,
        lambda1=lambda1,
        lambda2=lambda2,
        f_ref=f_ref,
        phasing_convention=phasing_convention,
        reference_frame=reference_frame,
        f_lower=f_lower,
        f_final=f_final,
        delta_f=delta_f,
        sample_frequencies=sample_frequencies,
        frequency_support=frequency_support,
        modes=modes,
        device=device,
        dtype=dtype,
        **kwargs,
    )
    if isinstance(out, tuple) and len(out) >= 2 and torch.is_tensor(out[0]) and torch.is_tensor(out[1]):
        hp, hc = out[0], out[1]
        f = sample_frequencies
        if f is None and delta_f is not None and f_lower is not None and f_final is not None:
            f = torch.arange(f_lower, f_final + delta_f / 2.0, delta_f, dtype=dtype, device=device)
        if f is not None and (f_lower is not None or f_final is not None):
            # Preserve frequency batch axes; ordinary broadcasting prepends
            # any missing parameter axes. Flattening here merges rows/samples.
            f_eval = torch.as_tensor(f, device=hp.device, dtype=hp.real.dtype)
            zero_mask = f_eval <= 0.0
            if f_lower is not None:
                zero_mask = zero_mask | (f_eval < f_lower)
            key = approximant.lower().replace("_", "").replace("-", "")
            pv2_sequence = key == "imrphenompv2" and sample_frequencies is not None and frequency_support != "uniform"
            if f_final is not None and not pv2_sequence:
                zero_mask = zero_mask | (f_eval >= f_final)
            hp_re = torch.where(zero_mask, torch.zeros_like(hp.real), hp.real)
            hp_im = torch.where(zero_mask, torch.zeros_like(hp.imag), hp.imag)
            hc_re = torch.where(zero_mask, torch.zeros_like(hc.real), hc.real)
            hc_im = torch.where(zero_mask, torch.zeros_like(hc.imag), hc.imag)
            hp = torch.view_as_complex(torch.stack([hp_re, hp_im], dim=-1))
            hc = torch.view_as_complex(torch.stack([hc_re, hc_im], dim=-1))
        if len(out) == 2:
            return hp, hc
        return (hp, hc) + out[2:]
    return out


def get_td_waveform(
    approximant: str,
    mass1: Optional[Union[float, torch.Tensor]] = None,
    mass2: Optional[Union[float, torch.Tensor]] = None,
    chi1: Union[float, torch.Tensor, object] = _UNSET,
    chi2: Union[float, torch.Tensor, object] = _UNSET,
    spin1x: Union[float, torch.Tensor] = 0.0,
    spin1y: Union[float, torch.Tensor] = 0.0,
    spin1z: Optional[Union[float, torch.Tensor]] = None,
    spin2x: Union[float, torch.Tensor] = 0.0,
    spin2y: Union[float, torch.Tensor] = 0.0,
    spin2z: Optional[Union[float, torch.Tensor]] = None,
    distance: Union[float, torch.Tensor] = 100.0,
    inclination: Union[float, torch.Tensor] = 0.0,
    phic: Union[float, torch.Tensor] = 0.0,
    tc: Union[float, torch.Tensor] = 0.0,
    lambda1: Optional[Union[float, torch.Tensor]] = None,
    lambda2: Optional[Union[float, torch.Tensor]] = None,
    delta_t: float = 1.0 / 2048.0,
    f_lower: float = 20.0,
    f_final: Optional[float] = None,
    duration: Optional[float] = None,
    t_pad: float = 0.2,
    center_coalescence: bool = True,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float64,
    return_times: bool = False,
    **kwargs,
) -> Union[
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]:
    """Generate conditioned time-domain gravitational-wave polarizations h_plus(t) and h_cross(t).

    Literature:
    - Allen et al. (2012), Phys. Rev. D 85, 122006 (arXiv:gr-qc/0509116)
    - Buonanno et al. (2009), Phys. Rev. D 80, 084043 (arXiv:0907.0700)

    The IMRPhenomT family is evaluated natively in time; f_ref fixes its orbital phase.
    Other models use the inverse Fourier path, which conserves signal energy:
        int |h(t)|^2 dt = 2 * int |h_tilde(f)|^2 df

    Args:
        approximant: Waveform approximant name
        mass1: Component mass of primary in solar masses
        mass2: Component mass of secondary in solar masses
        chi1: Dimensionless aligned spin of body 1
        chi2: Dimensionless aligned spin of body 2
        spin1x: Primary dimensionless spin x-component
        spin1y: Primary dimensionless spin y-component
        spin1z: Primary dimensionless spin z-component
        spin2x: Secondary dimensionless spin x-component
        spin2y: Secondary dimensionless spin y-component
        spin2z: Secondary dimensionless spin z-component
        distance: Luminosity distance in Mpc
        inclination: Orbital inclination in radians
        phic: Coalescence phase in radians
        tc: Requested coalescence peak time offset in seconds
        lambda1: Dimensionless tidal deformability of body 1
        lambda2: Dimensionless tidal deformability of body 2
        delta_t: Sampling interval in seconds (default 1/2048 s)
        f_lower: Low-frequency cutoff in Hz
        f_final: High-frequency cutoff in Hz (defaults to Nyquist 1/(2*delta_t))
        duration: Duration of time series in seconds (automatically estimated if None)
        t_pad: Padding duration in seconds between coalescence peak and end of window
        center_coalescence: If True, center merger peak at requested tc or T - t_pad
        device: PyTorch device
        dtype: Tensor floating-point precision (torch.float64 recommended)
        return_times: If True, also return 1D time coordinate tensor

    Returns:
        (h_plus, h_cross) or (h_plus, h_cross, times)
    """
    if kwargs:
        norm_kwargs = normalize_waveform_params(kwargs)
        if mass1 is None and "mass1" in norm_kwargs:
            mass1 = norm_kwargs.pop("mass1")
        if mass2 is None and "mass2" in norm_kwargs:
            mass2 = norm_kwargs.pop("mass2")
        if chi1 is _UNSET and "chi1" in norm_kwargs:
            chi1 = norm_kwargs.pop("chi1")
        if chi2 is _UNSET and "chi2" in norm_kwargs:
            chi2 = norm_kwargs.pop("chi2")
        if spin1z is None and "spin1z" in norm_kwargs:
            spin1z = norm_kwargs.pop("spin1z")
        if spin2z is None and "spin2z" in norm_kwargs:
            spin2z = norm_kwargs.pop("spin2z")
        if "spin1x" in norm_kwargs and spin1x == 0.0:
            spin1x = norm_kwargs.pop("spin1x")
        if "spin1y" in norm_kwargs and spin1y == 0.0:
            spin1y = norm_kwargs.pop("spin1y")
        if "spin2x" in norm_kwargs and spin2x == 0.0:
            spin2x = norm_kwargs.pop("spin2x")
        if "spin2y" in norm_kwargs and spin2y == 0.0:
            spin2y = norm_kwargs.pop("spin2y")
        if "distance" in norm_kwargs and distance == 100.0:
            distance = norm_kwargs.pop("distance")
        if "inclination" in norm_kwargs and inclination == 0.0:
            inclination = norm_kwargs.pop("inclination")
        if "phic" in norm_kwargs and phic == 0.0:
            phic = norm_kwargs.pop("phic")
        if "tc" in norm_kwargs and tc == 0.0:
            tc = norm_kwargs.pop("tc")
        if lambda1 is None and "lambda1" in norm_kwargs:
            lambda1 = norm_kwargs.pop("lambda1")
        if lambda2 is None and "lambda2" in norm_kwargs:
            lambda2 = norm_kwargs.pop("lambda2")
        if f_lower == 20.0 and "f_lower" in norm_kwargs:
            f_lower = norm_kwargs.pop("f_lower")
        if f_final is None and "f_final" in norm_kwargs:
            f_final = norm_kwargs.pop("f_final")
        if delta_t == 1.0 / 2048.0 and "delta_t" in norm_kwargs:
            delta_t = norm_kwargs.pop("delta_t")
        if duration is None and "duration" in norm_kwargs:
            duration = norm_kwargs.pop("duration")
        kwargs = norm_kwargs

    if mass1 is None or mass2 is None:
        raise TypeError("get_td_waveform() requires both 'mass1' and 'mass2' (or aliases 'mass_1', 'm1', etc.)")
    key = approximant.lower().replace("_", "").replace("-", "")
    if key in ("imrphenomt", "imrphenomthm", "imrphenomtp", "imrphenomtphm"):
        from ._phenomt_td import generate_td

        return generate_td(
            approximant=key, mass1=mass1, mass2=mass2,
            chi1=None if chi1 is _UNSET else chi1,
            chi2=None if chi2 is _UNSET else chi2,
            spin1z=spin1z, spin2z=spin2z, spin1x=spin1x, spin1y=spin1y,
            spin2x=spin2x, spin2y=spin2y, distance=distance,
            inclination=inclination, phic=phic, tc=tc, lambda1=lambda1, lambda2=lambda2,
            delta_t=delta_t, f_lower=f_lower, f_final=f_final, duration=duration,
            t_pad=t_pad, center_coalescence=center_coalescence,
            return_times=return_times, device=device, dtype=dtype, **kwargs,
        )

    m1_t = as_tensor(mass1, device=device, dtype=dtype)
    m2_t = as_tensor(mass2, device=device, dtype=dtype)

    # Estimate required duration if not explicitly provided
    if duration is None:
        # Newtonian 0PN chirp time formula from f_lower to merger
        m_chirp = component_masses_to_chirp_mass(m1_t, m2_t)
        m_chirp_min = torch.min(m_chirp).item()
        m_chirp_sec = m_chirp_min * MSUN_SEC
        tau0 = (5.0 / 256.0) * (m_chirp_sec ** (-5.0 / 3.0)) * ((PI * f_lower) ** (-8.0 / 3.0))
        est_duration = max(2.0, 1.2 * tau0 + 2.0 * t_pad + 1.0)
        # Round up to next power of 2 for FFT efficiency
        n_samples_raw = int(math.ceil(est_duration / delta_t))
        n_samples = 1 << (n_samples_raw - 1).bit_length()
        dur = n_samples * delta_t
    else:
        dur = float(duration)
        n_samples = int(math.ceil(dur / delta_t))

    df = 1.0 / dur
    f_nyquist = 1.0 / (2.0 * delta_t)
    f_max = f_final if f_final is not None else f_nyquist

    # Construct linear frequency evaluation grid [0, df, 2*df, ..., f_nyquist]
    freqs = torch.arange(0, n_samples // 2 + 1, dtype=dtype, device=device) * df

    # Mask frequencies above 0 and below f_max
    valid_mask = (freqs >= f_lower) & (freqs <= f_max)
    active_freqs = freqs[valid_mask]

    # Generate frequency-domain waveform
    hp_fd_active, hc_fd_active = get_fd_waveform(
        approximant=approximant,
        mass1=mass1,
        mass2=mass2,
        chi1=chi1,
        chi2=chi2,
        spin1x=spin1x,
        spin1y=spin1y,
        spin1z=spin1z,
        spin2x=spin2x,
        spin2y=spin2y,
        spin2z=spin2z,
        distance=distance,
        inclination=inclination,
        phic=phic,
        tc=0.0,  # Generate with tc=0; time centering handled cleanly in time domain
        lambda1=lambda1,
        lambda2=lambda2,
        sample_frequencies=active_freqs,
        device=device,
        dtype=dtype,
        **kwargs,
    )

    # Full Fourier grid container
    is_batched = m1_t.dim() >= 1
    complex_dtype = torch.complex128 if dtype == torch.float64 else torch.complex64

    if is_batched:
        batch_shape = m1_t.shape
        full_shape = batch_shape + (len(freqs),)
        hp_fd = torch.zeros(full_shape, dtype=complex_dtype, device=device)
        hc_fd = torch.zeros(full_shape, dtype=complex_dtype, device=device)
        hp_fd[..., valid_mask] = hp_fd_active
        hc_fd[..., valid_mask] = hc_fd_active
    else:
        hp_fd = torch.zeros(len(freqs), dtype=complex_dtype, device=device)
        hc_fd = torch.zeros(len(freqs), dtype=complex_dtype, device=device)
        hp_fd[valid_mask] = hp_fd_active
        hc_fd[valid_mask] = hc_fd_active

    # Low-frequency smooth Planck taper at f_lower over [f_lower, f_lower + df_taper]
    df_taper = max(2.0, 0.1 * f_lower)
    f_start = f_lower
    f_end = f_lower + df_taper
    z = torch.clamp((freqs - f_start) / (f_end - f_start), 1e-7, 1.0 - 1e-7)
    sigma = (1.0 / z) + (1.0 / (z - 1.0))
    w_smooth = 1.0 / (torch.exp(torch.clamp(sigma, -50.0, 50.0)) + 1.0)
    w_taper = torch.where(freqs <= f_start, torch.zeros_like(w_smooth), w_smooth)
    w_taper = torch.where(freqs >= f_end, torch.ones_like(w_taper), w_taper)

    if is_batched:
        w_taper = w_taper.unsqueeze(0)

    hp_fd = hp_fd * w_taper
    hc_fd = hc_fd * w_taper

    # Perform batched inverse FFT with exact continuous scaling:
    # h(t) = (1 / delta_t) * irfft(h_tilde) = df * N * irfft(h_tilde)
    # Satisfies exact Parseval: int |h(t)|^2 dt = 2 * int |h_tilde(f)|^2 df
    scale_factor = 1.0 / delta_t
    hp_td = scale_factor * torch.fft.irfft(hp_fd, n=n_samples, dim=-1)
    hc_td = scale_factor * torch.fft.irfft(hc_fd, n=n_samples, dim=-1)

    # Time centering: roll buffer so coalescence peak occurs at requested location
    if center_coalescence:
        tc_val = float(tc) if isinstance(tc, (int, float)) else float(tc.reshape(-1)[0].item())
        target_t = (dur - t_pad) + tc_val
        shift_samples = int(round(target_t / delta_t)) % n_samples
        hp_td = torch.roll(hp_td, shifts=shift_samples, dims=-1)
        hc_td = torch.roll(hc_td, shifts=shift_samples, dims=-1)

    times = torch.arange(n_samples, dtype=dtype, device=device) * delta_t
    if center_coalescence:
        # If centered at tc, align time array such that coalescence is at tc
        times = times - (dur - t_pad)

    if return_times:
        return hp_td, hc_td, times
    return hp_td, hc_td


class WaveformGenerator:
    """Class-based interface for gravitational waveform generation.

    Configured with static waveform parameters and grid specification,
    allowing repeated waveform evaluation across varying binary parameters.
    """

    def __init__(
        self,
        approximant: str = "IMRPhenomD",
        f_lower: Optional[float] = None,
        f_final: Optional[float] = None,
        delta_f: Optional[float] = None,
        sample_frequencies: Optional[Any] = None,
        modes: Optional[Sequence[Tuple[int, int]]] = None,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float64,
        **default_kwargs,
    ):
        norm_init = normalize_waveform_params(default_kwargs)
        self.approximant = norm_init.pop("approximant", approximant)
        self.f_lower = norm_init.pop("f_lower", f_lower)
        self.f_final = norm_init.pop("f_final", f_final)
        self.delta_f = norm_init.pop("delta_f", delta_f)
        sf = norm_init.pop("sample_frequencies", sample_frequencies)
        self.sample_frequencies = sf
        self.modes = norm_init.pop("modes", modes)
        self.device = norm_init.pop("device", device)
        self.dtype = norm_init.pop("dtype", dtype)
        self.default_kwargs = norm_init
        if sf is not None:
            self._cached_frequencies = torch.as_tensor(sf, dtype=self.dtype, device=self.device)
        else:
            self._cached_frequencies = None

    def generate_fd_waveform(
        self,
        parameters: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate frequency-domain polarizations for given parameters."""
        params = dict(self.default_kwargs)
        if parameters is not None:
            params.update(parameters)
        params.update(kwargs)
        normalized_params = normalize_waveform_params(params)

        sf = normalized_params.pop(
            "sample_frequencies",
            self._cached_frequencies if self._cached_frequencies is not None else self.sample_frequencies,
        )

        call_kwargs = {
            "approximant": normalized_params.pop("approximant", self.approximant),
            "f_lower": normalized_params.pop("f_lower", self.f_lower),
            "f_final": normalized_params.pop("f_final", self.f_final),
            "delta_f": normalized_params.pop("delta_f", self.delta_f),
            "sample_frequencies": sf,
            "modes": normalized_params.pop("modes", self.modes),
            "device": normalized_params.pop("device", self.device),
            "dtype": normalized_params.pop("dtype", self.dtype),
        }
        call_kwargs.update(normalized_params)

        return get_fd_waveform(**call_kwargs)

    get_fd_waveform = generate_fd_waveform

    def __call__(self, *args, **kwargs):
        if args and isinstance(args[0], dict):
            return self.generate_fd_waveform(parameters=args[0], **kwargs)
        return self.generate_fd_waveform(**kwargs)


__all__ = [
    "get_fd_waveform",
    "get_td_waveform",
    "WaveformGenerator",
    "list_approximants",
    "normalize_waveform_params",
]

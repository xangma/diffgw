"""High-level gravitational-wave waveform generation API in JAX.

Permissively licensed (MIT), native JAX models matching the torchwave API.
Literature and calibration provenance is recorded in docs/provenance_ledger.md.
"""

from typing import Any, Dict, Optional, Sequence, Tuple, Union
import math
import jax
import jax.numpy as jnp

from diffgw_core.constants import (
    MSUN_SEC,
    PI,
)
from jaxwave.conversions import (
    as_array,
    component_masses_to_chirp_mass,
)
from jaxwave.approximants.taylorf2 import TaylorF2
from jaxwave.approximants.taylorf2_hm import TaylorF2_HM
from jaxwave.approximants.taylorf2_eccentric import TaylorF2_Eccentric
from jaxwave.approximants.imrphenomd import IMRPhenomD
from jaxwave.approximants.imrphenomt import IMRPhenomT
from jaxwave.approximants.imrphenomthm import IMRPhenomTHM
from jaxwave.approximants.imrphenomtp import IMRPhenomTP
from jaxwave.approximants.imrphenomtphm import IMRPhenomTPHM
from jaxwave.approximants.imrphenomp import IMRPhenomP
from jaxwave.approximants.phenompv2_waveform import IMRPhenomPv2
from jaxwave.approximants.imrphenomd_nrtidal import (
    IMRPhenomD_NRTidal,
    IMRPhenomD_NRTidalv2,
)
from jaxwave.approximants.imrphenom_hm import IMRPhenomHM
from jaxwave.approximants.imrphenom_xas import IMRPhenomXAS
from jaxwave.approximants.imrphenom_xhm import IMRPhenomXHM
from jaxwave.approximants.imrphenom_nsbh import IMRPhenomNSBH
from jaxwave.approximants.kerr_ringdown import KerrRingdown, get_ringdown_waveform
from jaxwave.approximants.imrphenom_xp import IMRPhenomXP
from jaxwave.approximants.imrphenom_xphm import IMRPhenomXPHM
from jaxwave.approximants.rom import OfficialSEOBNRv4ROM


_APPROXIMANTS = {
    "imrphenomxp": IMRPhenomXP,
    "phenomxp": IMRPhenomXP,
    "xp": IMRPhenomXP,
    "imrphenomxphm": IMRPhenomXPHM,
    "phenomxphm": IMRPhenomXPHM,
    "xphm": IMRPhenomXPHM,
    "imrphenomxphmspintaylor": IMRPhenomXPHM,
    "phenomxphmspintaylor": IMRPhenomXPHM,
    "xphmspintaylor": IMRPhenomXPHM,
    "imrphenomxpnr": IMRPhenomXPHM,
    "phenomxpnr": IMRPhenomXPHM,
    "xpnr": IMRPhenomXPHM,
    "pnr": IMRPhenomXPHM,
    "taylorf2": TaylorF2,
    "taylorf2hm": TaylorF2_HM,
    "taylorf2eccentric": TaylorF2_Eccentric,
    "eccentrictaylorf2": TaylorF2_Eccentric,
    "imrphenomd": IMRPhenomD,
    "imrphenomt": IMRPhenomT,
    "imrphenomthm": IMRPhenomTHM,
    "imrphenomtp": IMRPhenomTP,
    "imrphenomtphm": IMRPhenomTPHM,
    "imrphenomp": IMRPhenomP,
    "imrphenompv2": IMRPhenomPv2,
    "imrphenomdnrtidal": IMRPhenomD_NRTidal,
    "imrphenomdnrtidalv2": IMRPhenomD_NRTidalv2,
    "nrtidal": IMRPhenomD_NRTidal,
    "nrtidalv2": IMRPhenomD_NRTidalv2,
    "imrphenomhm": IMRPhenomHM,
    "imrphenomxas": IMRPhenomXAS,
    "phenomxas": IMRPhenomXAS,
    "xas": IMRPhenomXAS,
    "imrphenomxhm": IMRPhenomXHM,
    "imrphenomnsbh": IMRPhenomNSBH,
    "nsbh": IMRPhenomNSBH,
    "kerrringdown": KerrRingdown,
    "ringdown": KerrRingdown,
    "seobnrrom": OfficialSEOBNRv4ROM,
    "seobnrv4rom": OfficialSEOBNRv4ROM,
    "officialseobnrv4rom": OfficialSEOBNRv4ROM,
}


def list_approximants() -> list[str]:
    """Return canonical JAX model names; aliases remain valid in dispatch."""
    return sorted({model.__name__ for model in _APPROXIMANTS.values()})

_UNSET = object()


def _get_fd_waveform_raw(
    approximant: str,
    mass1: Union[float, jnp.ndarray],
    mass2: Union[float, jnp.ndarray],
    chi1: Union[float, jnp.ndarray, object] = _UNSET,
    chi2: Union[float, jnp.ndarray, object] = _UNSET,
    spin1x: Union[float, jnp.ndarray] = 0.0,
    spin1y: Union[float, jnp.ndarray] = 0.0,
    spin1z: Optional[Union[float, jnp.ndarray]] = None,
    spin2x: Union[float, jnp.ndarray] = 0.0,
    spin2y: Union[float, jnp.ndarray] = 0.0,
    spin2z: Optional[Union[float, jnp.ndarray]] = None,
    distance: Union[float, jnp.ndarray] = 100.0,
    inclination: Union[float, jnp.ndarray] = 0.0,
    phic: Union[float, jnp.ndarray] = 0.0,
    tc: Union[float, jnp.ndarray] = 0.0,
    lambda1: Optional[Union[float, jnp.ndarray]] = None,
    lambda2: Optional[Union[float, jnp.ndarray]] = None,
    f_ref: Optional[Union[float, jnp.ndarray]] = None,
    phasing_convention: Optional[str] = None,
    reference_frame: Optional[str] = None,
    f_lower: Optional[float] = None,
    f_final: Optional[float] = None,
    delta_f: Optional[float] = None,
    sample_frequencies: Optional[jnp.ndarray] = None,
    modes: Optional[Sequence[Tuple[int, int]]] = None,
    dtype: jnp.dtype = jnp.float64,
    frequency_support: Optional[str] = None,
    **kwargs,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Internal raw frequency-domain waveform generator."""
    key = approximant.lower().replace("_", "").replace("-", "")
    if key in ("imrphenomt", "imrphenomthm", "imrphenomtp", "imrphenomtphm"):
        raise ValueError(f"{approximant} is a native time-domain model; use get_td_waveform()")
    if key not in _APPROXIMANTS:
        raise ValueError(
            f"Approximant '{approximant}' not recognized. Available: {list(_APPROXIMANTS.keys())}"
        )
    if frequency_support not in (None, "sequence", "uniform"):
        raise ValueError("frequency_support must be 'sequence' or 'uniform'")
    if frequency_support == "sequence" and sample_frequencies is None:
        raise ValueError("frequency_support='sequence' requires sample_frequencies")

    if sample_frequencies is not None:
        f = as_array(sample_frequencies, dtype=dtype)
    else:
        if f_lower is None or f_final is None or delta_f is None:
            raise ValueError(
                "Either `sample_frequencies` or all of (`f_lower`, `f_final`, `delta_f`) must be provided."
            )
        f = jnp.arange(f_lower, f_final + delta_f / 2.0, delta_f, dtype=dtype)

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

    # Spin resolution
    if chi1 is not _UNSET and spin1z is not None:
        t_chi1 = as_array(chi1, dtype=dtype)
        t_s1z = as_array(spin1z, dtype=dtype)
        if not isinstance(t_chi1, jax.core.Tracer) and not isinstance(t_s1z, jax.core.Tracer) and not jnp.allclose(t_chi1, t_s1z, atol=1e-12, rtol=1e-8):
            raise ValueError(f"Conflicting spin1 definitions: chi1={chi1} and spin1z={spin1z}")
        s1z = spin1z
    elif spin1z is not None:
        s1z = spin1z
    elif chi1 is not _UNSET:
        s1z = chi1
    else:
        s1z = 0.0

    if chi2 is not _UNSET and spin2z is not None:
        t_chi2 = as_array(chi2, dtype=dtype)
        t_s2z = as_array(spin2z, dtype=dtype)
        if not isinstance(t_chi2, jax.core.Tracer) and not isinstance(t_s2z, jax.core.Tracer) and not jnp.allclose(t_chi2, t_s2z, atol=1e-12, rtol=1e-8):
            raise ValueError(f"Conflicting spin2 definitions: chi2={chi2} and spin2z={spin2z}")
        s2z = spin2z
    elif spin2z is not None:
        s2z = spin2z
    elif chi2 is not _UNSET:
        s2z = chi2
    else:
        s2z = 0.0

    precessing_keys = {
        "imrphenomp", "imrphenompv2", "imrphenomxp", "phenomxp", "xp",
        "imrphenomxphm", "phenomxphm", "xphm",
        "imrphenomxphmspintaylor", "phenomxphmspintaylor", "xphmspintaylor",
        "imrphenomxpnr", "phenomxpnr", "xpnr", "pnr",
    }
    s1x_arr = as_array(spin1x, dtype=dtype)
    s1y_arr = as_array(spin1y, dtype=dtype)
    s2x_arr = as_array(spin2x, dtype=dtype)
    s2y_arr = as_array(spin2y, dtype=dtype)
    # Python exceptions validate concrete inputs before tracing. JIT callers
    # must supply physically valid parameters; never coerce a tracer to bool.
    has_in_plane = any(
        bool(jnp.any(spin != 0.0))
        for spin in (s1x_arr, s1y_arr, s2x_arr, s2y_arr)
        if not isinstance(spin, jax.core.Tracer)
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
    elif key in ("imrphenomp", "imrphenompv2", "imrphenomxp", "phenomxp", "xp"):
        model = model_cls()
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
        "pnr",
    ):
        if key in ("imrphenomxphmspintaylor", "phenomxphmspintaylor", "xphmspintaylor"):
            model_kwargs.setdefault("precession_method", "spintaylor")
            model_kwargs.setdefault("prec_version", 320)
            model_kwargs.setdefault("final_spin_mod", 2)
        elif key in ("imrphenomxpnr", "phenomxpnr", "xpnr", "pnr"):
            model_kwargs.setdefault("precession_method", "pnr")
            model_kwargs.setdefault("prec_version", 330)
            model_kwargs.setdefault("final_spin_mod", 2)
        calibration = model_kwargs.pop("calibration", "sovereign")
        init_kwargs = {}
        if modes is not None:
            init_kwargs["modes"] = modes
        elif key in ("imrphenomxpnr", "phenomxpnr", "xpnr", "pnr"):
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
        explicit_legacy = any(k in model_kwargs for k in ("amplitudes", "phases", "energy_mode"))
        if modes is not None and not explicit_legacy:
            raise ValueError(
                "fresh KerrRingdown supports only the physical (2,2,0) mode; "
                "use explicit amplitudes/phases/energy_mode for the legacy primitive"
            )
        if not explicit_legacy:
            from jaxwave.approximants.kerr_ringdown_public_clean import kerr220_fd
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
    elif key in ("seobnrrom", "seobnrv4rom", "officialseobnrv4rom"):
        if lambda1 is not None and jnp.any(as_array(lambda1) != 0):
            raise ValueError("SEOBNRv4_ROM does not support tidal deformability")
        if lambda2 is not None and jnp.any(as_array(lambda2) != 0):
            raise ValueError("SEOBNRv4_ROM does not support tidal deformability")
        if phasing_convention not in (None, "coalescence", "reference_frequency"):
            raise ValueError("unsupported phasing_convention for SEOBNRv4_ROM")
        if reference_frame not in (None, "coalescence", "reference_frequency"):
            raise ValueError("unsupported reference_frame for SEOBNRv4_ROM")
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
    else:
        raise NotImplementedError(f"Approximant {approximant} not yet wired.")


def get_fd_waveform(
    approximant: str,
    mass1: Union[float, jnp.ndarray],
    mass2: Union[float, jnp.ndarray],
    chi1: Union[float, jnp.ndarray, object] = _UNSET,
    chi2: Union[float, jnp.ndarray, object] = _UNSET,
    spin1x: Union[float, jnp.ndarray] = 0.0,
    spin1y: Union[float, jnp.ndarray] = 0.0,
    spin1z: Optional[Union[float, jnp.ndarray]] = None,
    spin2x: Union[float, jnp.ndarray] = 0.0,
    spin2y: Union[float, jnp.ndarray] = 0.0,
    spin2z: Optional[Union[float, jnp.ndarray]] = None,
    distance: Union[float, jnp.ndarray] = 100.0,
    inclination: Union[float, jnp.ndarray] = 0.0,
    phic: Union[float, jnp.ndarray] = 0.0,
    tc: Union[float, jnp.ndarray] = 0.0,
    lambda1: Optional[Union[float, jnp.ndarray]] = None,
    lambda2: Optional[Union[float, jnp.ndarray]] = None,
    f_ref: Optional[Union[float, jnp.ndarray]] = None,
    phasing_convention: Optional[str] = None,
    reference_frame: Optional[str] = None,
    f_lower: Optional[float] = None,
    f_final: Optional[float] = None,
    delta_f: Optional[float] = None,
    sample_frequencies: Optional[jnp.ndarray] = None,
    modes: Optional[Sequence[Tuple[int, int]]] = None,
    dtype: jnp.dtype = jnp.float64,
    frequency_support: Optional[str] = None,
    **kwargs,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Generate frequency-domain gravitational-wave polarizations h_plus(f) and h_cross(f) in JAX."""
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
        dtype=dtype,
        **kwargs,
    )
    if isinstance(out, tuple) and len(out) >= 2 and isinstance(out[0], jnp.ndarray) and isinstance(out[1], jnp.ndarray):
        hp, hc = out[0], out[1]
        f = sample_frequencies
        if f is None and delta_f is not None and f_lower is not None and f_final is not None:
            f = jnp.arange(f_lower, f_final + delta_f / 2.0, delta_f, dtype=dtype)
        if f is not None:
            f_eval = as_array(f, dtype=hp.real.dtype)
            zero_mask = f_eval <= 0.0
            if f_lower is not None:
                zero_mask = zero_mask | (f_eval < f_lower)
            if f_final is not None:
                zero_mask = zero_mask | (f_eval >= f_final)
            hp = jnp.where(zero_mask, jnp.zeros_like(hp), hp)
            hc = jnp.where(zero_mask, jnp.zeros_like(hc), hc)
        if len(out) == 2:
            return hp, hc
        return (hp, hc) + out[2:]
    return out



def get_td_waveform(
    approximant: str = "IMRPhenomD",
    mass1: Union[float, jnp.ndarray] = 30.0,
    mass2: Union[float, jnp.ndarray] = 25.0,
    chi1: Union[float, jnp.ndarray, object] = _UNSET,
    chi2: Union[float, jnp.ndarray, object] = _UNSET,
    spin1x: Union[float, jnp.ndarray] = 0.0,
    spin1y: Union[float, jnp.ndarray] = 0.0,
    spin1z: Optional[Union[float, jnp.ndarray]] = None,
    spin2x: Union[float, jnp.ndarray] = 0.0,
    spin2y: Union[float, jnp.ndarray] = 0.0,
    spin2z: Optional[Union[float, jnp.ndarray]] = None,
    distance: Union[float, jnp.ndarray] = 100.0,
    inclination: Union[float, jnp.ndarray] = 0.0,
    phic: Union[float, jnp.ndarray] = 0.0,
    tc: Union[float, jnp.ndarray] = 0.0,
    lambda1: Optional[Union[float, jnp.ndarray]] = None,
    lambda2: Optional[Union[float, jnp.ndarray]] = None,
    delta_t: float = 1.0 / 2048.0,
    f_lower: float = 20.0,
    f_final: Optional[float] = None,
    duration: Optional[float] = None,
    t_pad: float = 0.2,
    center_coalescence: bool = True,
    return_times: bool = False,
    **kwargs,
) -> Union[
    Tuple[jnp.ndarray, jnp.ndarray],
    Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
]:
    """Generate conditioned time-domain gravitational-wave polarizations h_plus(t) and h_cross(t) in JAX.

    Literature:
    - Allen et al. (2012), Phys. Rev. D 85, 122006 (arXiv:gr-qc/0509116)
    - Buonanno et al. (2009), Phys. Rev. D 80, 084043 (arXiv:0907.0700)

    The IMRPhenomT family is evaluated natively in time; f_ref fixes its orbital phase.
    Other models use the inverse Fourier path, which conserves signal energy:
        int |h(t)|^2 dt = 2 * int |h_tilde(f)|^2 df
    """
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
            return_times=return_times, **kwargs,
        )

    m1_t = as_array(mass1, dtype=jnp.float64)
    m2_t = as_array(mass2, dtype=jnp.float64)

    if duration is None:
        m_chirp = component_masses_to_chirp_mass(m1_t, m2_t)
        m_chirp_min = float(jnp.min(m_chirp))
        m_chirp_sec = m_chirp_min * MSUN_SEC
        tau0 = (5.0 / 256.0) * (m_chirp_sec ** (-5.0 / 3.0)) * ((PI * f_lower) ** (-8.0 / 3.0))
        est_duration = max(2.0, 1.2 * tau0 + 2.0 * t_pad + 1.0)
        n_samples_raw = int(math.ceil(est_duration / delta_t))
        n_samples = 1 << (n_samples_raw - 1).bit_length()
        dur = n_samples * delta_t
    else:
        dur = float(duration)
        n_samples = int(math.ceil(dur / delta_t))

    df = 1.0 / dur
    f_nyquist = 1.0 / (2.0 * delta_t)
    f_max = f_final if f_final is not None else f_nyquist

    freqs = jnp.arange(0, n_samples // 2 + 1, dtype=jnp.float64) * df
    valid_mask = (freqs >= f_lower) & (freqs <= f_max)
    active_freqs = freqs[valid_mask]

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
        tc=0.0,
        lambda1=lambda1,
        lambda2=lambda2,
        sample_frequencies=active_freqs,
        **kwargs,
    )

    is_batched = m1_t.ndim >= 1
    if is_batched:
        batch_shape = m1_t.shape
        full_shape = batch_shape + (len(freqs),)
        hp_fd = jnp.zeros(full_shape, dtype=jnp.complex128).at[..., valid_mask].set(hp_fd_active)
        hc_fd = jnp.zeros(full_shape, dtype=jnp.complex128).at[..., valid_mask].set(hc_fd_active)
    else:
        hp_fd = jnp.zeros(len(freqs), dtype=jnp.complex128).at[valid_mask].set(hp_fd_active)
        hc_fd = jnp.zeros(len(freqs), dtype=jnp.complex128).at[valid_mask].set(hc_fd_active)

    df_taper = max(2.0, 0.1 * f_lower)
    f_start = f_lower
    f_end = f_lower + df_taper
    z = jnp.clip((freqs - f_start) / (f_end - f_start), 1e-7, 1.0 - 1e-7)
    sigma = (1.0 / z) + (1.0 / (z - 1.0))
    w_smooth = 1.0 / (jnp.exp(jnp.clip(sigma, -50.0, 50.0)) + 1.0)
    w_taper = jnp.where(freqs <= f_start, 0.0, w_smooth)
    w_taper = jnp.where(freqs >= f_end, 1.0, w_taper)

    if is_batched:
        w_taper = jnp.expand_dims(w_taper, 0)

    hp_fd = hp_fd * w_taper
    hc_fd = hc_fd * w_taper

    scale_factor = 1.0 / delta_t
    hp_td = scale_factor * jnp.fft.irfft(hp_fd, n=n_samples, axis=-1)
    hc_td = scale_factor * jnp.fft.irfft(hc_fd, n=n_samples, axis=-1)

    if center_coalescence:
        tc_val = float(tc) if isinstance(tc, (int, float)) else float(jnp.asarray(tc).reshape(-1)[0])
        target_t = (dur - t_pad) + tc_val
        shift_samples = int(round(target_t / delta_t)) % n_samples
        hp_td = jnp.roll(hp_td, shift=shift_samples, axis=-1)
        hc_td = jnp.roll(hc_td, shift=shift_samples, axis=-1)

    times = jnp.arange(n_samples, dtype=jnp.float64) * delta_t
    if center_coalescence:
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
        use_jit: bool = False,
        **default_kwargs,
    ):
        self.approximant = approximant
        self.f_lower = f_lower
        self.f_final = f_final
        self.delta_f = delta_f
        self.sample_frequencies = sample_frequencies
        self.modes = modes
        self.use_jit = use_jit
        self.default_kwargs = default_kwargs
        self._compiled_fd_fn = None
        self._compiled_param_order = None

    def get_compiled_fd_waveform(
        self,
        parameter_order: Tuple[str, ...] = ("mass1", "mass2", "spin1z", "spin2z", "distance", "inclination", "phic", "tc"),
    ):
        """Return a cached @jax.jit compiled function evaluating (hp, hc) for an ordered parameter vector.

        The callable closes over grid specifications and approximant metadata.
        Repeated calls with compatible parameter shapes and dtypes can reuse
        compilation; new signatures may require recompilation.
        """
        if self._compiled_fd_fn is not None and self._compiled_param_order == parameter_order:
            return self._compiled_fd_fn

        approx = self.approximant
        f_low = self.f_lower
        f_fin = self.f_final
        df = self.delta_f
        s_freqs = self.sample_frequencies
        m_modes = self.modes
        d_kwargs = dict(self.default_kwargs)

        def _raw_eval(*param_values):
            kw = dict(d_kwargs)
            for k, val in zip(parameter_order, param_values):
                kw[k] = val
            return get_fd_waveform(
                approximant=approx,
                f_lower=f_low,
                f_final=f_fin,
                delta_f=df,
                sample_frequencies=s_freqs,
                modes=m_modes,
                **kw,
            )

        jitted_eval = jax.jit(_raw_eval)
        self._compiled_fd_fn = jitted_eval
        self._compiled_param_order = parameter_order
        return jitted_eval

    def generate_fd_waveform(
        self,
        parameters: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Generate frequency-domain polarizations for given parameters."""
        params = dict(self.default_kwargs)
        if parameters is not None:
            params.update(parameters)
        params.update(kwargs)

        name_map = {
            "mass_1": "mass1",
            "mass_2": "mass2",
            "chi_1": "chi1",
            "chi_2": "chi2",
            "spin_1z": "spin1z",
            "spin_2z": "spin2z",
            "spin_1x": "spin1x",
            "spin_1y": "spin1y",
            "spin_2x": "spin2x",
            "spin_2y": "spin2y",
            "luminosity_distance": "distance",
            "theta_jn": "inclination",
            "phase": "phic",
            "geocent_time": "tc",
            "lambda_1": "lambda1",
            "lambda_2": "lambda2",
        }
        normalized_params = {}
        for k, v in params.items():
            norm_k = name_map.get(k, k)
            normalized_params[norm_k] = v

        if self.use_jit:
            # Check if parameters match standard order for cached JIT execution
            std_order = ("mass1", "mass2", "chi1", "chi2", "spin1z", "spin2z", "distance", "inclination", "phic", "tc", "lambda1", "lambda2", "quad_monopole1", "quad_monopole2")
            if all(k in normalized_params for k in ("mass1", "mass2")):
                order = tuple(k for k in std_order if k in normalized_params)
                compiled_fn = self.get_compiled_fd_waveform(parameter_order=order)
                args = tuple(jnp.asarray(normalized_params[k], dtype=jnp.float64) for k in order)
                return compiled_fn(*args)

        call_kwargs = {
            "approximant": normalized_params.pop("approximant", self.approximant),
            "f_lower": normalized_params.pop("f_lower", self.f_lower),
            "f_final": normalized_params.pop("f_final", self.f_final),
            "delta_f": normalized_params.pop("delta_f", self.delta_f),
            "sample_frequencies": normalized_params.pop("sample_frequencies", self.sample_frequencies),
            "modes": normalized_params.pop("modes", self.modes),
        }
        call_kwargs.update(normalized_params)

        return get_fd_waveform(**call_kwargs)


    def generate_td_waveform(
        self,
        parameters: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Generate time-domain polarizations for given parameters."""
        params = dict(self.default_kwargs)
        if parameters is not None:
            params.update(parameters)
        params.update(kwargs)

        name_map = {
            "mass_1": "mass1",
            "mass_2": "mass2",
            "spin_1z": "spin1z",
            "spin_2z": "spin2z",
            "spin_1x": "spin1x",
            "spin_1y": "spin1y",
            "spin_2x": "spin2x",
            "spin_2y": "spin2y",
            "luminosity_distance": "distance",
            "theta_jn": "inclination",
            "phase": "phic",
            "geocent_time": "tc",
            "lambda_1": "lambda1",
            "lambda_2": "lambda2",
        }
        normalized_params = {}
        for k, v in params.items():
            norm_k = name_map.get(k, k)
            normalized_params[norm_k] = v

        call_kwargs = {
            "approximant": normalized_params.pop("approximant", self.approximant),
            "f_lower": normalized_params.pop("f_lower", self.f_lower if self.f_lower is not None else 20.0),
            "f_final": normalized_params.pop("f_final", self.f_final),
        }
        call_kwargs.update(normalized_params)

        return get_td_waveform(**call_kwargs)

    def __call__(self, *args, **kwargs):
        if args and isinstance(args[0], dict):
            return self.generate_fd_waveform(parameters=args[0], **kwargs)
        return self.generate_fd_waveform(**kwargs)


__all__ = [
    "get_fd_waveform",
    "get_td_waveform",
    "WaveformGenerator",
]

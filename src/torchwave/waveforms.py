"""High-level gravitational-wave waveform generation API.

Permissively licensed (MIT), pure PyTorch implementation adhering strictly to
clean-room literature derivations.
"""

from typing import Optional, Sequence, Tuple, Union
import math
import torch

from torchwave.constants import (
    C_SI,
    MSUN_SEC,
    PI,
)
from torchwave.conversions import (
    as_tensor,
    component_masses_to_chirp_mass,
    component_masses_to_eta,
)
from torchwave.approximants.taylorf2 import TaylorF2
from torchwave.approximants.taylorf2_hm import TaylorF2_HM
from torchwave.approximants.imrphenomd import IMRPhenomD
from torchwave.approximants.imrphenomp import IMRPhenomP
from torchwave.approximants.imrphenomd_nrtidal import (
    IMRPhenomD_NRTidal,
    IMRPhenomD_NRTidalv2,
)
from torchwave.approximants.imrphenom_hm import (
    IMRPhenomHM,
    IMRPhenomXHM,
)
from torchwave.approximants.imrphenom_xphm import IMRPhenomXPHM
from torchwave.approximants.imrphenom_nsbh import IMRPhenomNSBH
from torchwave.approximants.taylorf2_eccentric import TaylorF2_Eccentric
from torchwave.approximants.kerr_ringdown import KerrRingdown, get_ringdown_waveform
from torchwave.approximants.eob.waveform import SEOBNR_ODE
from torchwave.approximants.rom.surrogate import SEOBNR_ROM
from torchwave.approximants.imrphenom_xas import IMRPhenomXAS
from torchwave.approximants.teobresums import TEOBResumS



_APPROXIMANTS = {
    "taylorf2": TaylorF2,
    "taylorf2hm": TaylorF2_HM,
    "taylorf2_hm": TaylorF2_HM,
    "imrphenomd": IMRPhenomD,
    "imrphenomp": IMRPhenomP,
    "imrphenompv2": IMRPhenomP,
    "imrphenomxp": IMRPhenomP,
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
    "imrphenomnsbh": IMRPhenomNSBH,
    "nsbh": IMRPhenomNSBH,
    "taylorf2eccentric": TaylorF2_Eccentric,
    "taylorf2_eccentric": TaylorF2_Eccentric,
    "eccentrictaylorf2": TaylorF2_Eccentric,
    "kerrringdown": KerrRingdown,
    "ringdown": KerrRingdown,
    "seobnr": SEOBNR_ODE,
    "seobnrode": SEOBNR_ODE,
    "seobnrv5": SEOBNR_ODE,
    "seobnrrom": SEOBNR_ROM,
    "seobnrv4rom": SEOBNR_ROM,
    "seobnrv5rom": SEOBNR_ROM,
    "teobresums": TEOBResumS,
    "teobresum": TEOBResumS,
    "teob": TEOBResumS,
}



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
        modes: Optional sequence of (l, m) harmonic modes for multipolar approximants
        device: PyTorch device ('cpu', 'cuda', 'mps')
        dtype: Floating point precision (default torch.float64)

    Returns:
        (h_plus, h_cross): Complex tensors of frequency-domain polarizations.
    """
    key = approximant.lower().replace("_", "").replace("-", "")
    if key not in _APPROXIMANTS:
        raise ValueError(
            f"Approximant '{approximant}' not recognized. Available: {list(_APPROXIMANTS.keys())}"
        )

    # Resolve frequency vector
    if sample_frequencies is not None:
        f = sample_frequencies
        if not isinstance(f, torch.Tensor):
            f = torch.tensor(f, dtype=dtype, device=device)
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
        t_chi1 = as_tensor(chi1)
        t_s1z = as_tensor(spin1z)
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
        t_chi2 = as_tensor(chi2)
        t_s2z = as_tensor(spin2z)
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
        "imrphenomp", "imrphenompv2", "imrphenomxp", "imrphenomxphm",
        "seobnr", "seobnrode", "seobnrv5",
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

    if key in ("taylorf2", "taylorf2hm", "taylorf2_hm"):
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
    elif key in ("imrphenomp", "imrphenompv2", "imrphenomxp"):
        model = model_cls()
        if key == "imrphenomxp" and "precession_version" not in model_kwargs:
            model_kwargs["precession_version"] = "xp"
        elif key in ("imrphenomp", "imrphenompv2") and "precession_version" not in model_kwargs:
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
    elif key in ("imrphenomd", "imrphenomxas", "phenomxas", "xas", "seobnrrom", "seobnrv4rom", "seobnrv5rom"):
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
    elif key in ("taylorf2eccentric", "taylorf2_eccentric", "eccentrictaylorf2"):
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
    elif key in ("imrphenomxphm",):
        if modes is not None:
            model = model_cls(modes=modes)
        else:
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
        final_mass = model_kwargs.pop("final_mass", mass1)
        final_spin = model_kwargs.pop("final_spin", s1z)
        return get_ringdown_waveform(
            final_mass=final_mass,
            final_spin=final_spin,
            frequencies=f,
            inclination=inclination,
            **model_kwargs,
        )
    elif key in ("seobnr", "seobnrode", "seobnrv5"):
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
    elif key in ("teobresums", "teobresum", "teob"):
        model = model_cls()
        lam1 = lambda1 if lambda1 is not None else 0.0
        lam2 = lambda2 if lambda2 is not None else 0.0
        return model(
            f=f,
            mass1=mass1,
            mass2=mass2,
            chi1=s1z,
            chi2=s2z,
            lambda1=lam1,
            lambda2=lam2,
            distance=distance,
            inclination=inclination,
            phic=phic,
            tc=tc,
            **model_kwargs,
        )
    else:

        raise NotImplementedError(f"Approximant {approximant} not yet wired.")


def get_fd_waveform(
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
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate frequency-domain gravitational-wave polarizations h_plus(f) and h_cross(f)."""
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
        if f is not None and torch.is_tensor(f):
            f_eval = f if f.dim() == hp.dim() else f.reshape(*([1] * (hp.dim() - f.dim())), -1)
            zero_mask = f_eval <= 0.0
            if f_lower is not None:
                zero_mask = zero_mask | (f_eval < f_lower)
            if f_final is not None:
                zero_mask = zero_mask | (f_eval >= f_final)
            hp = torch.where(zero_mask, torch.zeros_like(hp), hp)
            hc = torch.where(zero_mask, torch.zeros_like(hc), hc)
        if len(out) == 2:
            return hp, hc
        return (hp, hc) + out[2:]
    return out


def get_td_waveform(
    approximant: str,
    mass1: Union[float, torch.Tensor],
    mass2: Union[float, torch.Tensor],
    chi1: Union[float, torch.Tensor] = 0.0,
    chi2: Union[float, torch.Tensor] = 0.0,
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

    Guarantees exact conservation of signal energy (Parseval's theorem):
        int |h(t)|^2 dt = 2 * int |h_tilde(f)|^2 df

    Args:
        approximant: Waveform approximant name
        mass1, mass2: Component masses in solar masses
        chi1, chi2: Aligned dimensionless spins
        spin1x, spin1y, spin1z: 3D primary spin components
        spin2x, spin2y, spin2z: 3D secondary spin components
        distance: Luminosity distance in Mpc
        inclination: Orbital inclination in radians
        phic: Coalescence phase in radians
        tc: Requested coalescence peak time offset in seconds
        lambda1, lambda2: Tidal deformabilities
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

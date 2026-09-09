"""High-level gravitational-wave waveform generation API.

Permissively licensed (MIT), pure PyTorch implementation adhering strictly to
clean-room literature derivations.
"""

from typing import Optional, Tuple, Union
import torch

from torchwave.approximants.taylorf2 import TaylorF2
from torchwave.approximants.imrphenomd import IMRPhenomD
from torchwave.approximants.eob.waveform import SEOBNR_ODE
from torchwave.approximants.rom.surrogate import SEOBNR_ROM


_APPROXIMANTS = {
    "taylorf2": TaylorF2,
    "imrphenomd": IMRPhenomD,
    "seobnr": SEOBNR_ODE,
    "seobnrode": SEOBNR_ODE,
    "seobnrv5": SEOBNR_ODE,
    "seobnrrom": SEOBNR_ROM,
    "seobnrv4rom": SEOBNR_ROM,
    "seobnrv5rom": SEOBNR_ROM,
}


def get_fd_waveform(
    approximant: str,
    mass1: Union[float, torch.Tensor],
    mass2: Union[float, torch.Tensor],
    chi1: Union[float, torch.Tensor] = 0.0,
    chi2: Union[float, torch.Tensor] = 0.0,
    distance: Union[float, torch.Tensor] = 100.0,
    inclination: Union[float, torch.Tensor] = 0.0,
    phic: Union[float, torch.Tensor] = 0.0,
    tc: Union[float, torch.Tensor] = 0.0,
    lambda1: Optional[Union[float, torch.Tensor]] = None,
    lambda2: Optional[Union[float, torch.Tensor]] = None,
    f_lower: Optional[float] = None,
    f_final: Optional[float] = None,
    delta_f: Optional[float] = None,
    sample_frequencies: Optional[torch.Tensor] = None,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float64,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate frequency-domain gravitational-wave polarizations h_plus(f) and h_cross(f).

    Args:
        approximant: Name of the waveform approximant (e.g. 'TaylorF2', 'IMRPhenomD', 'SEOBNR_ODE', 'SEOBNR_ROM')
        mass1: Primary mass in solar masses (m1 >= m2)
        mass2: Secondary mass in solar masses
        chi1: Dimensionless aligned spin of primary mass (-1 <= chi1 <= 1)
        chi2: Dimensionless aligned spin of secondary mass (-1 <= chi2 <= 1)
        distance: Luminosity distance in Mpc (default 100.0)
        inclination: Orbital inclination angle in radians (default 0.0)
        phic: Coalescence phase in radians (default 0.0)
        tc: Coalescence time in seconds (default 0.0)
        lambda1: Dimensionless tidal deformability of primary (for neutron stars)
        lambda2: Dimensionless tidal deformability of secondary (for neutron stars)
        f_lower: Starting frequency for evaluation grid in Hz
        f_final: Ending frequency for evaluation grid in Hz
        delta_f: Frequency spacing for uniform evaluation grid in Hz
        sample_frequencies: Pre-computed tensor of frequencies in Hz
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
    model = model_cls()

    if key == "taylorf2":
        return model(
            f=f,
            mass1=mass1,
            mass2=mass2,
            chi1=chi1,
            chi2=chi2,
            distance=distance,
            inclination=inclination,
            phic=phic,
            tc=tc,
            lambda1=lambda1,
            lambda2=lambda2,
            **kwargs,
        )
    elif key in ("imrphenomd", "seobnrrom", "seobnrv4rom", "seobnrv5rom"):
        return model(
            f=f,
            mass1=mass1,
            mass2=mass2,
            chi1=chi1,
            chi2=chi2,
            distance=distance,
            inclination=inclination,
            phic=phic,
            tc=tc,
            **kwargs,
        )
    elif key in ("seobnr", "seobnrode", "seobnrv5"):
        return model(
            f=f,
            mass1=mass1,
            mass2=mass2,
            chi1=chi1,
            chi2=chi2,
            distance=distance,
            inclination=inclination,
            phic=phic,
            tc=tc,
            **kwargs,
        )
    else:
        raise NotImplementedError(f"Approximant {approximant} not yet wired.")

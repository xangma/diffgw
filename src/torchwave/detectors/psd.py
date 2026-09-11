"""Analytical Detector Power Spectral Densities (PSDs) for Gravitational-Wave Detectors.

Strict Clean-Room Implementation based exclusively on published literature:
- Cutler & Flanagan (1994), Phys. Rev. D 49, 2658 (arXiv:gr-qc/9402014)
- Sathyaprakash & Schutz (2009), Living Rev. Relativity 12, 2 (arXiv:0903.0338)
- Ajith (2008), Class. Quantum Grav. 25, 114033 (arXiv:0712.0343)
- Hild et al. (2011), Class. Quantum Grav. 28, 094013 (arXiv:1012.0908) [ET-D]
- Evans et al. (2021), arXiv:2109.09882 [Cosmic Explorer]
- Somiya (2012), Class. Quantum Grav. 29, 124007 (arXiv:1111.7185) [KAGRA]
"""

from typing import Callable, Dict, Optional, Sequence, Union
import math
import torch
import torch.nn as nn

from torchwave.conversions import as_tensor


def aLIGOZeroDetHighPower(
    f: torch.Tensor,
    f_low: float = 10.0,
    f_high: float = 4096.0,
) -> torch.Tensor:
    """Advanced LIGO Zero Detuning High Power (design sensitivity) analytical PSD.

    Literature: Sathyaprakash & Schutz (2009) Eq. 3.23; Ajith (2008) Eq. 4.7.
    """
    f_safe = torch.clamp(f, min=1.0)
    x = f_safe / 215.0
    x2 = x * x
    x4 = x2 * x2

    term1 = torch.pow(x, -4.14)
    term2 = -5.0 * torch.pow(x, -2.0)
    term3 = 111.0 * ((1.0 - x2 + 0.5 * x4) / (1.0 + 0.5 * x2))

    s0 = 1.0e-49
    sn = s0 * (term1 + term2 + term3)

    # Apply frequency range mask
    in_band = (f >= f_low) & (f <= f_high)
    return torch.where(in_band, sn, torch.full_like(sn, 1.0e-35))


def aLIGOEarlyHighPower(
    f: torch.Tensor,
    f_low: float = 20.0,
    f_high: float = 4096.0,
) -> torch.Tensor:
    """Advanced LIGO Early High Power commissioning sensitivity analytical PSD."""
    f_safe = torch.clamp(f, min=1.0)
    x = f_safe / 215.0
    x2 = x * x
    x4 = x2 * x2

    term1 = torch.pow(x, -4.14)
    term2 = -5.0 * torch.pow(x, -2.0)
    term3 = 111.0 * ((1.0 - x2 + 0.5 * x4) / (1.0 + 0.5 * x2))

    s0 = 4.0e-49
    sn = s0 * (term1 + term2 + term3)

    in_band = (f >= f_low) & (f <= f_high)
    return torch.where(in_band, sn, torch.full_like(sn, 1.0e-35))


def AdvancedVirgo(
    f: torch.Tensor,
    f_low: float = 10.0,
    f_high: float = 4096.0,
) -> torch.Tensor:
    """Advanced Virgo design sensitivity analytical PSD.

    Literature: Sathyaprakash & Schutz (2009) Eq. 3.24; Manzotti & Dietz (2012).
    """
    f_safe = torch.clamp(f, min=1.0)
    x = f_safe / 500.0

    term1 = torch.pow(x, -4.05)
    term2 = 1.85 * torch.pow(x, -0.69)
    term3 = 0.78 * (x * x)

    s0 = 1.0e-46
    sn = s0 * (term1 + term2 + term3)

    in_band = (f >= f_low) & (f <= f_high)
    return torch.where(in_band, sn, torch.full_like(sn, 1.0e-35))


def KAGRA(
    f: torch.Tensor,
    f_low: float = 10.0,
    f_high: float = 4096.0,
) -> torch.Tensor:
    """KAGRA design sensitivity analytical PSD.

    Literature: Somiya (2012), Aso et al. (2013).
    """
    f_safe = torch.clamp(f, min=1.0)
    x = f_safe / 100.0

    term1 = 0.05 * torch.pow(x, -8.0)
    term2 = 0.8 * torch.pow(x, -2.0)
    term3 = 1.5 * torch.pow(x, 0.5)
    term4 = 0.6 * (x * x)

    s0 = 1.0e-47
    sn = s0 * (term1 + term2 + term3 + term4)

    in_band = (f >= f_low) & (f <= f_high)
    return torch.where(in_band, sn, torch.full_like(sn, 1.0e-35))


def EinsteinTelescope(
    f: torch.Tensor,
    f_low: float = 1.0,
    f_high: float = 4096.0,
) -> torch.Tensor:
    """Einstein Telescope (ET-D) 10km triangular configuration single-detector PSD.

    Literature: Hild et al. (2011) Class. Quantum Grav. 28, 094013 Eq. 1.
    """
    f_safe = torch.clamp(f, min=0.5)
    x = f_safe / 100.0

    term1 = 2.39e-27 * torch.pow(x, -15.64)
    term2 = 0.349 * torch.pow(x, -2.145)
    term3 = 0.564 * torch.pow(x, -0.48)
    term4 = 0.852 * torch.pow(x, 3.39)

    s0 = 1.0e-50
    sn = s0 * (term1 + term2 + term3 + term4)

    in_band = (f >= f_low) & (f <= f_high)
    return torch.where(in_band, sn, torch.full_like(sn, 1.0e-35))


def CosmicExplorer(
    f: torch.Tensor,
    f_low: float = 5.0,
    f_high: float = 4096.0,
) -> torch.Tensor:
    """Cosmic Explorer (CE 40km) design sensitivity analytical PSD.

    Literature: Evans et al. (2021) arXiv:2109.09882.
    """
    f_safe = torch.clamp(f, min=1.0)
    x = f_safe / 100.0

    term1 = 1.2e-6 * torch.pow(x, -10.0)
    term2 = 0.08 * torch.pow(x, -2.0)
    term3 = 0.3 * torch.pow(x, -0.5)
    term4 = 0.5 * torch.pow(x, 2.5)

    s0 = 1.0e-52
    sn = s0 * (term1 + term2 + term3 + term4)

    in_band = (f >= f_low) & (f <= f_high)
    return torch.where(in_band, sn, torch.full_like(sn, 1.0e-35))


_ANALYTICAL_PSDS: Dict[str, Callable[[torch.Tensor, float, float], torch.Tensor]] = {
    "aligozerodethighpower": aLIGOZeroDetHighPower,
    "aligodesign": aLIGOZeroDetHighPower,
    "aligo": aLIGOZeroDetHighPower,
    "ligo": aLIGOZeroDetHighPower,
    "aligoearlyhighpower": aLIGOEarlyHighPower,
    "advancedvirgo": AdvancedVirgo,
    "virgo": AdvancedVirgo,
    "kagra": KAGRA,
    "einsteintelescope": EinsteinTelescope,
    "et": EinsteinTelescope,
    "etd": EinsteinTelescope,
    "cosmicexplorer": CosmicExplorer,
    "ce": CosmicExplorer,
    "ce40km": CosmicExplorer,
}


def get_psd(
    name: str,
    frequencies: Union[Sequence[float], torch.Tensor],
    f_low: Optional[float] = None,
    f_high: Optional[float] = None,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Retrieve analytical power spectral density Sn(f) for given detector.

    Args:
        name: Name of detector PSD model (e.g. 'aLIGOZeroDetHighPower', 'AdvancedVirgo', 'KAGRA', 'EinsteinTelescope', 'CosmicExplorer')
        frequencies: Tensor or array of evaluation frequencies in Hz
        f_low: Optional low-frequency cutoff in Hz
        f_high: Optional high-frequency cutoff in Hz
        device: PyTorch device
        dtype: Tensor floating point dtype (default torch.float64)

    Returns:
        Tensor of power spectral density values Sn(f) in 1/Hz.
    """
    key = name.lower().replace("_", "").replace("-", "")
    if key not in _ANALYTICAL_PSDS:
        raise ValueError(
            f"PSD '{name}' not recognized. Available: {list(_ANALYTICAL_PSDS.keys())}"
        )

    f = as_tensor(frequencies, device=device, dtype=dtype)
    psd_fn = _ANALYTICAL_PSDS[key]

    kwargs = {}
    if f_low is not None:
        kwargs["f_low"] = float(f_low)
    if f_high is not None:
        kwargs["f_high"] = float(f_high)

    return psd_fn(f, **kwargs)

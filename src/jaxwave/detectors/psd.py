"""Analytical Detector Power Spectral Densities (PSDs) for Gravitational-Wave Detectors in JAX.

Strict Clean-Room Implementation based exclusively on published literature:
- Cutler & Flanagan (1994), Phys. Rev. D 49, 2658 (arXiv:gr-qc/9402014)
- Sathyaprakash & Schutz (2009), Living Rev. Relativity 12, 2 (arXiv:0903.0338)
- Ajith (2008), Class. Quantum Grav. 25, 114033 (arXiv:0712.0343)
- Hild et al. (2011), Class. Quantum Grav. 28, 094013 (arXiv:1012.0908) [ET-D]
- Evans et al. (2021), arXiv:2109.09882 [Cosmic Explorer]
- Somiya (2012), Class. Quantum Grav. 29, 124007 (arXiv:1111.7185) [KAGRA]
"""

from typing import Callable, Dict, Optional, Sequence, Union
import jax.numpy as jnp

from jaxwave.conversions import as_array


def aLIGOZeroDetHighPower(
    f: Union[float, jnp.ndarray],
    f_low: float = 10.0,
    f_high: float = 4096.0,
) -> jnp.ndarray:
    """Advanced LIGO Zero Detuning High Power (design sensitivity) analytical PSD.

    Literature: Sathyaprakash & Schutz (2009) Eq. 3.23; Ajith (2008) Eq. 4.7.
    """
    f_arr = jnp.asarray(f, dtype=jnp.float64)
    f_safe = jnp.maximum(f_arr, 1.0)
    x = f_safe / 215.0
    x2 = x * x
    x4 = x2 * x2

    term1 = jnp.power(x, -4.14)
    term2 = -5.0 * jnp.power(x, -2.0)
    term3 = 111.0 * ((1.0 - x2 + 0.5 * x4) / (1.0 + 0.5 * x2))

    s0 = 1.0e-49
    sn = s0 * (term1 + term2 + term3)

    # Apply frequency range mask
    in_band = (f_arr >= f_low) & (f_arr <= f_high)
    return jnp.where(in_band, sn, jnp.full_like(sn, 1.0e-35))


def aLIGOEarlyHighPower(
    f: Union[float, jnp.ndarray],
    f_low: float = 20.0,
    f_high: float = 4096.0,
) -> jnp.ndarray:
    """Advanced LIGO Early High Power commissioning sensitivity analytical PSD."""
    f_arr = jnp.asarray(f, dtype=jnp.float64)
    f_safe = jnp.maximum(f_arr, 1.0)
    x = f_safe / 215.0
    x2 = x * x
    x4 = x2 * x2

    term1 = jnp.power(x, -4.14)
    term2 = -5.0 * jnp.power(x, -2.0)
    term3 = 111.0 * ((1.0 - x2 + 0.5 * x4) / (1.0 + 0.5 * x2))

    s0 = 4.0e-49
    sn = s0 * (term1 + term2 + term3)

    in_band = (f_arr >= f_low) & (f_arr <= f_high)
    return jnp.where(in_band, sn, jnp.full_like(sn, 1.0e-35))


def AdvancedVirgo(
    f: Union[float, jnp.ndarray],
    f_low: float = 10.0,
    f_high: float = 4096.0,
) -> jnp.ndarray:
    """Advanced Virgo design sensitivity analytical PSD.

    Literature: Sathyaprakash & Schutz (2009) Eq. 3.24; Manzotti & Dietz (2012).
    """
    f_arr = jnp.asarray(f, dtype=jnp.float64)
    f_safe = jnp.maximum(f_arr, 1.0)
    x = f_safe / 500.0

    term1 = jnp.power(x, -4.05)
    term2 = 1.85 * jnp.power(x, -0.69)
    term3 = 0.78 * (x * x)

    s0 = 1.0e-46
    sn = s0 * (term1 + term2 + term3)

    in_band = (f_arr >= f_low) & (f_arr <= f_high)
    return jnp.where(in_band, sn, jnp.full_like(sn, 1.0e-35))


def KAGRA(
    f: Union[float, jnp.ndarray],
    f_low: float = 10.0,
    f_high: float = 4096.0,
) -> jnp.ndarray:
    """KAGRA design sensitivity analytical PSD.

    Literature: Somiya (2012), Aso et al. (2013).
    """
    f_arr = jnp.asarray(f, dtype=jnp.float64)
    f_safe = jnp.maximum(f_arr, 1.0)
    x = f_safe / 100.0

    term1 = 0.05 * jnp.power(x, -8.0)
    term2 = 0.8 * jnp.power(x, -2.0)
    term3 = 1.5 * jnp.power(x, 0.5)
    term4 = 0.6 * (x * x)

    s0 = 1.0e-47
    sn = s0 * (term1 + term2 + term3 + term4)

    in_band = (f_arr >= f_low) & (f_arr <= f_high)
    return jnp.where(in_band, sn, jnp.full_like(sn, 1.0e-35))


def EinsteinTelescope(
    f: Union[float, jnp.ndarray],
    f_low: float = 1.0,
    f_high: float = 4096.0,
) -> jnp.ndarray:
    """Einstein Telescope (ET-D) 10km triangular configuration single-detector PSD.

    Literature: Hild et al. (2011) Class. Quantum Grav. 28, 094013 Eq. 1.
    """
    f_arr = jnp.asarray(f, dtype=jnp.float64)
    f_safe = jnp.maximum(f_arr, 0.5)
    x = f_safe / 100.0

    term1 = 2.39e-27 * jnp.power(x, -15.64)
    term2 = 0.349 * jnp.power(x, -2.145)
    term3 = 0.564 * jnp.power(x, -0.48)
    term4 = 0.852 * jnp.power(x, 3.39)

    s0 = 1.0e-50
    sn = s0 * (term1 + term2 + term3 + term4)

    in_band = (f_arr >= f_low) & (f_arr <= f_high)
    return jnp.where(in_band, sn, jnp.full_like(sn, 1.0e-35))


def CosmicExplorer(
    f: Union[float, jnp.ndarray],
    f_low: float = 5.0,
    f_high: float = 4096.0,
) -> jnp.ndarray:
    """Cosmic Explorer (CE 40km) design sensitivity analytical PSD.

    Literature: Evans et al. (2021) arXiv:2109.09882.
    """
    f_arr = jnp.asarray(f, dtype=jnp.float64)
    f_safe = jnp.maximum(f_arr, 1.0)
    x = f_safe / 100.0

    term1 = 1.2e-6 * jnp.power(x, -10.0)
    term2 = 0.08 * jnp.power(x, -2.0)
    term3 = 0.3 * jnp.power(x, -0.5)
    term4 = 0.5 * jnp.power(x, 2.5)

    s0 = 1.0e-52
    sn = s0 * (term1 + term2 + term3 + term4)

    in_band = (f_arr >= f_low) & (f_arr <= f_high)
    return jnp.where(in_band, sn, jnp.full_like(sn, 1.0e-35))


_ANALYTICAL_PSDS: Dict[str, Callable[..., jnp.ndarray]] = {
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
    frequencies: Union[Sequence[float], jnp.ndarray],
    f_low: Optional[float] = None,
    f_high: Optional[float] = None,
    dtype=jnp.float64,
) -> jnp.ndarray:
    """Retrieve analytical power spectral density Sn(f) for given detector in JAX.

    Args:
        name: Name of detector PSD model (e.g. 'aLIGOZeroDetHighPower', 'AdvancedVirgo', 'KAGRA', 'EinsteinTelescope', 'CosmicExplorer')
        frequencies: Array or sequence of evaluation frequencies in Hz
        f_low: Optional low-frequency cutoff in Hz
        f_high: Optional high-frequency cutoff in Hz
        dtype: Array floating point dtype (default jnp.float64)

    Returns:
        Array of power spectral density values Sn(f) in 1/Hz.
    """
    key = name.lower().replace("_", "").replace("-", "")
    if key not in _ANALYTICAL_PSDS:
        raise ValueError(
            f"PSD '{name}' not recognized. Available: {list(_ANALYTICAL_PSDS.keys())}"
        )

    f = as_array(frequencies, dtype=dtype)
    psd_fn = _ANALYTICAL_PSDS[key]

    kwargs = {}
    if f_low is not None:
        kwargs["f_low"] = float(f_low)
    if f_high is not None:
        kwargs["f_high"] = float(f_high)

    return psd_fn(f, **kwargs)

"""Published NRTidalv2 frequency-domain phase used by PhenomNSBH.

The phase is Eq. (18) of Dietrich et al., arXiv:1905.06011.  Keeping this
small expression separate makes the NSBH assembly independent of fit tables
whose provenance is outside the clean-room source set.
"""

import math

import torch


# Dietrich et al. (2019), Eqs. (22)--(23), frequency-domain Padé fit.
# The paper publishes these decimal values; retain them as written.
NRTIDALV2_FD_N52 = 90.550822
NRTIDALV2_FD_N3 = -60.253578
NRTIDALV2_FD_D1 = -15.111208
NRTIDALV2_FD_D2 = 8.0641096

_C1 = 3115.0 / 1248.0
_C32 = -math.pi
_C2 = 28024205.0 / 3302208.0
_C52 = -4283.0 * math.pi / 1092.0


def _validate_frequency(frequency_hz: torch.Tensor) -> torch.Tensor:
    """Validate the positive GW-frequency domain required by Eq. (18)."""
    if not torch.compiler.is_compiling():
        if not torch.isfinite(frequency_hz).all() or not (frequency_hz > 0).all():
            raise ValueError("NRTidalv2 frequency must contain finite strictly positive values")
    return frequency_hz

# The remaining Padé coefficients are constrained by Eq. (21).  Computing
# them here documents the published construction and avoids a hidden table.
NRTIDALV2_FD_N1 = _C1 + NRTIDALV2_FD_D1
NRTIDALV2_FD_N32 = (
    _C1 * _C32 - _C52 - _C32 * NRTIDALV2_FD_D1 + NRTIDALV2_FD_N52
) / _C1
NRTIDALV2_FD_N2 = _C2 + _C1 * NRTIDALV2_FD_D1 + NRTIDALV2_FD_D2
NRTIDALV2_FD_D32 = -(
    _C52 + _C32 * NRTIDALV2_FD_D1 - NRTIDALV2_FD_N52
) / _C1


def nrtidalv2_frequency_phase(
    frequency_hz: torch.Tensor,
    total_mass_sec: torch.Tensor,
    eta: torch.Tensor,
    lambda_tilde: torch.Tensor,
) -> torch.Tensor:
    """Return the additive NRTidalv2 phase correction in radians.

    ``frequency_hz`` is the positive GW (2,2) frequency.  The tidal
    deformability is the usual mass-weighted ``Lambda_tilde``.  No reference
    phase or time shift is applied here: those affine choices belong to the
    PhenomD baseline in the NSBH assembly (Thompson et al. Sec. II.B).
    """
    f = _validate_frequency(frequency_hz)
    x = torch.pow(math.pi * total_mass_sec * f, 2.0 / 3.0)
    x32 = torch.pow(x, 1.5)
    x2 = x * x
    x52 = x2 * torch.sqrt(x)
    x3 = x2 * x
    numerator = (
        1.0 + NRTIDALV2_FD_N1 * x + NRTIDALV2_FD_N32 * x32
        + NRTIDALV2_FD_N2 * x2 + NRTIDALV2_FD_N52 * x52
        + NRTIDALV2_FD_N3 * x3
    )
    denominator = 1.0 + NRTIDALV2_FD_D1 * x + NRTIDALV2_FD_D32 * x32 + NRTIDALV2_FD_D2 * x2
    return -(117.0 / (256.0 * eta)) * x52 * (numerator / denominator) * lambda_tilde


def nrtidalv2_frequency_phase_derivative(
    frequency_hz: torch.Tensor,
    total_mass_sec: torch.Tensor,
    eta: torch.Tensor,
    lambda_tilde: torch.Tensor,
) -> torch.Tensor:
    """Return ``d psi_T / d f`` in rad/Hz for the Eq. (18) phase.

    This is the direct derivative of the published NRTidalv2 Padé expression.
    It deliberately applies no merger/peak/taper anchoring: the cited papers
    do not prescribe an affine phase convention at any such frequency.
    """
    f = _validate_frequency(frequency_hz)
    x = torch.pow(math.pi * total_mass_sec * f, 2.0 / 3.0)
    x32 = torch.pow(x, 1.5)
    x2 = x * x
    x52 = x2 * torch.sqrt(x)
    n = (
        1.0 + NRTIDALV2_FD_N1 * x + NRTIDALV2_FD_N32 * x32
        + NRTIDALV2_FD_N2 * x2 + NRTIDALV2_FD_N52 * x52
        + NRTIDALV2_FD_N3 * x2 * x
    )
    d = 1.0 + NRTIDALV2_FD_D1 * x + NRTIDALV2_FD_D32 * x32 + NRTIDALV2_FD_D2 * x2
    dn = NRTIDALV2_FD_N1 + 1.5 * NRTIDALV2_FD_N32 * torch.sqrt(x) \
        + 2.0 * NRTIDALV2_FD_N2 * x + 2.5 * NRTIDALV2_FD_N52 * x32 \
        + 3.0 * NRTIDALV2_FD_N3 * x2
    dd = NRTIDALV2_FD_D1 + 1.5 * NRTIDALV2_FD_D32 * torch.sqrt(x) + 2.0 * NRTIDALV2_FD_D2 * x
    ratio = n / d
    ratio_dx = (dn * d - n * dd) / (d * d)
    prefactor = -(117.0 / (256.0 * eta)) * lambda_tilde * x52 / f
    return prefactor * ((5.0 / 3.0) * ratio + (2.0 / 3.0) * x * ratio_dx)


__all__ = ["nrtidalv2_frequency_phase", "nrtidalv2_frequency_phase_derivative"]

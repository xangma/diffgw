"""Exact energy normalization for an isolated damped Kerr mode or mirror pair.

Derived from Ruiz et al., arXiv:0707.4654v3, Eq. (2.8), with unit angular
norm as in Flanagan--Hughes, gr-qc/9701039v2, Eq. (3.22). The finite-Q
normalization also agrees with Berti--Cardoso--Will, gr-qc/0512160v2,
Eq. (3.18). See docs/reviews/2026-09-12-clean-room/pass-3-ringdown-energy.md.
No QNM frequency or excitation model is assumed here.
"""

import math
from typing import Union

import torch

from torchwave.constants import C_SI, MPC_SI, MSUN_SEC


Real = Union[float, torch.Tensor]


def ringdown_amplitude_from_energy_fraction(
    final_mass: Real,
    distance: Real,
    energy_fraction: Real,
    omega: Real,
    tau: Real,
    *,
    convention: str = "native_pair",
) -> torch.Tensor:
    """Return the positive onset strain amplitude for a given emitted energy.

    ``final_mass`` is the remnant mass in solar masses, ``distance`` is in
    Mpc, ``omega`` is the positive angular frequency in rad/s, and ``tau``
    is the damping time in seconds. ``energy_fraction`` is the nonnegative
    energy divided by the remnant rest energy. Frequencies and damping times
    are supplied by the caller. All parameters broadcast.

    With unit-normalized spin-weighted angular functions, ``convention`` is:

    * ``single_mode``: the coefficient magnitude of one complex mode;
    * ``pair_member``: each coefficient magnitude of an equal-amplitude
      equatorial +m,-m pair, with energy_fraction the total pair energy;
    * ``native_pair`` (default): the input amplitude used by KerrRingdown's
      projection, which places half that amplitude in each pair member.

    Both pair conventions require m>0. They do not describe m=0 or the
    coherent energy of several overtones. The energy integral covers u>0
    for exp[-u/tau-i*omega*u], excluding an artificial onset discontinuity.
    It retains the damping derivative, without a high-Q approximation.
    Cosmological strain requires detector-frame mass/frequencies and
    luminosity distance consistently.

    Tensor inputs must share a device. Float32/float64 tensor dtypes are
    promoted; Python-only or integer-only inputs use float64. Other floating
    precisions, complex values, and booleans are rejected. Autograd is
    preserved. At zero energy the amplitude is zero and its right derivative
    with respect to energy_fraction is infinite; derivatives with respect to
    the other parameters remain zero.
    """
    factors = {"single_mode": 1.0, "pair_member": 1 / math.sqrt(2),
               "native_pair": math.sqrt(2)}
    if convention not in factors:
        raise ValueError("convention must be 'single_mode', 'pair_member', or 'native_pair'")

    names = ("final_mass", "distance", "energy_fraction", "omega", "tau")
    values = (final_mass, distance, energy_fraction, omega, tau)
    tensors = [v for v in values if isinstance(v, torch.Tensor)]
    device = tensors[0].device if tensors else None
    dtype = None
    for value in tensors:
        if value.device != device:
            raise ValueError("tensor inputs must share a device")
        if value.is_floating_point():
            if value.dtype not in (torch.float32, torch.float64):
                raise TypeError("floating tensor inputs must have dtype float32 or float64")
            dtype = value.dtype if dtype is None else torch.promote_types(dtype, value.dtype)
    dtype = dtype or torch.float64
    converted = []
    for name, value in zip(names, values):
        raw = torch.as_tensor(value)
        if raw.is_complex() or raw.dtype == torch.bool:
            raise TypeError(f"{name} must be real and not boolean")
        # Convert original Python values directly, avoiding default-float32
        # rounding before the chosen dtype is applied.
        value = torch.as_tensor(value, dtype=dtype, device=device)
        invalid_sign = value < 0 if name == "energy_fraction" else value <= 0
        if not torch.compiler.is_compiling():
            if bool(torch.any(~torch.isfinite(value) | invalid_sign)):
                domain = "nonnegative" if name == "energy_fraction" else "positive"
                raise ValueError(f"{name} must be finite and {domain}")
        converted.append(value)
    mass, distance, efficiency, omega, tau = torch.broadcast_tensors(*converted)

    # a_single = sqrt(32*pi*epsilon*M_time*tau) /
    #            (D_time*sqrt(1+(omega*tau)^2)).
    # Logarithms avoid SI over/underflow, including float32 squared strain.
    log_omega_tau = omega.log() + tau.log()
    log_amplitude = (
        0.5 * (math.log(32 * math.pi * MSUN_SEC) + mass.log() + tau.log())
        - math.log(MPC_SI / C_SI) - distance.log()
        - 0.5 * torch.logaddexp(torch.zeros_like(log_omega_tau), 2 * log_omega_tau)
    )
    # Keep sqrt(efficiency) separate so zero energy has finite zero gradients
    # with respect to every strictly positive physical parameter.
    return factors[convention] * log_amplitude.exp() * efficiency.sqrt()

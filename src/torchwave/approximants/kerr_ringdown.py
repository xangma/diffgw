"""Kerr Black Hole Ringdown Overtones & Spectroscopy Approximant.

Strict Clean-Room Implementation based exclusively on published literature:
- Berti, Cardoso, Will (2006), Phys. Rev. D 73, 064030 (arXiv:gr-qc/0512160), Tables VIII, IX, X
- Giesler et al. (2019), Phys. Rev. X 9, 041060 (arXiv:1903.08284)
"""

from typing import Dict, Optional, Sequence, Tuple, Union
import math
import torch
import torch.nn as nn

from torchwave.constants import (
    C_SI,
    MSUN_SEC,
    PI,
)
from torchwave.conversions import as_tensor
from torchwave.approximants.imrphenom_hm import _swsh_projection_coefficients


# Tabulated fitting coefficients from Berti, Cardoso, Will (2006)
# Format: (l, m, n): {'f': (f1, f2, f3), 'q': (q1, q2, q3)}
# Dimensionless frequency: M * omega_lmn = f1 + f2 * (1 - a_f)^f3
# Quality factor: Q_lmn = q1 + q2 * (1 - a_f)^q3
BERTI_QNM_FITS: Dict[Tuple[int, int, int], Dict[str, Tuple[float, float, float]]] = {
    # l=2, m=2 (Table VIII)
    (2, 2, 0): {"f": (1.5251, -1.1568, 0.1292), "q": (0.7000, 1.4187, -0.4990)},
    (2, 2, 1): {"f": (1.3673, -1.0260, 0.1628), "q": (0.1000, 0.5436, -0.4731)},
    (2, 2, 2): {"f": (1.3223, -1.0257, 0.1860), "q": (-0.1000, 0.4206, -0.4256)},
    # l=2, m=1 (Table VIII)
    (2, 1, 0): {"f": (0.6000, -0.2339, 0.4175), "q": (-0.3000, 2.3561, -0.2277)},
    (2, 1, 1): {"f": (0.5800, -0.2416, 0.4708), "q": (-0.3300, 0.9501, -0.2072)},
    (2, 1, 2): {"f": (0.5660, -0.2740, 0.4960), "q": (-0.1000, 0.4173, -0.2774)},
    # l=3, m=3 (Table IX)
    (3, 3, 0): {"f": (1.8956, -1.3043, 0.1818), "q": (0.9000, 2.3430, -0.4810)},
    (3, 3, 1): {"f": (1.8566, -1.2818, 0.1934), "q": (0.2274, 0.8173, -0.4731)},
    (3, 3, 2): {"f": (1.8004, -1.2558, 0.2133), "q": (0.0400, 0.5445, -0.4539)},
    # l=3, m=2 (Table IX)
    (3, 2, 0): {"f": (1.1481, -0.5552, 0.3002), "q": (0.8313, 2.3773, -0.3655)},
    (3, 2, 1): {"f": (1.1226, -0.5471, 0.3264), "q": (0.2300, 0.8025, -0.3684)},
    (3, 2, 2): {"f": (1.0989, -0.5550, 0.3569), "q": (0.1000, 0.4804, -0.3784)},
    # l=4, m=4 (Table X)
    (4, 4, 0): {"f": (2.3000, -1.5056, 0.2244), "q": (1.1929, 3.1191, -0.4825)},
    (4, 4, 1): {"f": (2.3000, -1.5173, 0.2271), "q": (0.3000, 1.1034, -0.4703)},
    (4, 4, 2): {"f": (2.3000, -1.5397, 0.2321), "q": (0.1100, 0.6997, -0.4607)},
}


def evaluate_qnm(
    l: int,
    m: int,
    n: int,
    final_mass: Union[float, torch.Tensor],
    final_spin: Union[float, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute physical angular frequency omega (rad/s), damping time tau (s), and quality factor Q.

    Args:
        l, m, n: Harmonic and overtone indices
        final_mass: Remnant mass in solar masses
        final_spin: Dimensionless remnant spin a_f (0 <= a_f < 1)

    Returns:
        (omega_rad_s, tau_s, Q)
    """
    key = (l, m, n)
    if key not in BERTI_QNM_FITS:
        raise ValueError(f"Mode (l={l}, m={m}, n={n}) not in Berti et al. fit tables.")

    mf = as_tensor(final_mass)
    af = as_tensor(final_spin)

    fit = BERTI_QNM_FITS[key]
    f1, f2, f3 = fit["f"]
    q1, q2, q3 = fit["q"]

    # Clamp 1 - a_f to strictly positive range for numerical safety
    one_minus_af = torch.clamp(1.0 - af, min=1e-5, max=1.0)

    # Dimensionless M * omega and Q
    m_omega = f1 + f2 * (one_minus_af ** f3)
    q_val = q1 + q2 * (one_minus_af ** q3)

    # Remnant mass in seconds: M_sec = final_mass * MSUN_SEC
    m_sec = mf * MSUN_SEC

    # Physical omega (rad/s) and damping time tau (s)
    omega_phys = m_omega / m_sec
    tau_phys = (2.0 * q_val) / omega_phys

    return omega_phys, tau_phys, q_val


class KerrRingdown(nn.Module):
    """Kerr black hole ringdown waveform model with overtones and multipoles."""

    def __init__(self):
        super().__init__()

    @staticmethod
    def time_domain(
        times: torch.Tensor,
        final_mass: Union[float, torch.Tensor],
        final_spin: Union[float, torch.Tensor],
        amplitudes: Optional[Dict[Tuple[int, int, int], Union[float, torch.Tensor]]] = None,
        phases: Optional[Dict[Tuple[int, int, int], Union[float, torch.Tensor]]] = None,
        t0: Union[float, torch.Tensor] = 0.0,
        inclination: Union[float, torch.Tensor] = 0.0,
        phi: Union[float, torch.Tensor] = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate time-domain damped sinusoids h_plus(t) and h_cross(t).

        h_+(t) - i h_x(t) = sum_{l,m,n} A_{lmn} exp(-(t-t0)/tau_{lmn}) exp(i [omega_{lmn}(t-t0) + phi_{lmn}]) * -2Y_{lm}(iota, phi)
        for t >= t0, and 0 for t < t0.
        """
        device = times.device
        dtype = times.dtype if times.is_floating_point() else torch.float64

        t = as_tensor(times, device=device, dtype=dtype)
        mf = as_tensor(final_mass, device=device, dtype=dtype)
        af = as_tensor(final_spin, device=device, dtype=dtype)
        t_0 = as_tensor(t0, device=device, dtype=dtype)
        inc = as_tensor(inclination, device=device, dtype=dtype)
        phi_0 = as_tensor(phi, device=device, dtype=dtype)

        if amplitudes is None:
            amplitudes = {(2, 2, 0): 1.0}
        if phases is None:
            phases = {}

        is_batched = mf.dim() >= 1
        if is_batched and t.dim() == 1:
            t_eval = t.unsqueeze(0)
        else:
            t_eval = t

        def unsqueeze_param(tensor: torch.Tensor) -> torch.Tensor:
            if is_batched and t_eval.dim() > tensor.dim():
                return tensor.unsqueeze(-1)
            return tensor

        t0_eval = unsqueeze_param(t_0)
        inc_eval = unsqueeze_param(inc)
        phi_eval = unsqueeze_param(phi_0)

        dt = t_eval - t0_eval
        heaviside = torch.where(dt >= 0.0, torch.ones_like(dt), torch.zeros_like(dt))

        h_plus = torch.zeros_like(dt)
        h_cross = torch.zeros_like(dt)

        for (l, m, n), amp in amplitudes.items():
            amp_t = as_tensor(amp, device=device, dtype=dtype)
            phase_val = phases.get((l, m, n), 0.0)
            phase_t = as_tensor(phase_val, device=device, dtype=dtype)

            omega, tau, _ = evaluate_qnm(l, m, n, mf, af)
            omega = unsqueeze_param(omega)
            tau = unsqueeze_param(tau)
            amp_t = unsqueeze_param(amp_t)
            phase_t = unsqueeze_param(phase_t)

            decay = torch.exp(-dt / tau) * heaviside
            arg = omega * dt + phase_t
            cos_term = torch.cos(arg)
            sin_term = torch.sin(arg)

            d_plus, d_cross = _swsh_projection_coefficients(l, m, inc_eval, phi_eval)

            # Complex signal before SWSH projection: s(t) = A * decay * exp(i * arg)
            s_real = amp_t * decay * cos_term
            s_imag = amp_t * decay * sin_term

            # Projection: h_+ = Re(s * d_plus), h_x = Re(s * d_cross)
            hp_mode = s_real * d_plus.real - s_imag * d_plus.imag
            hc_mode = s_real * d_cross.real - s_imag * d_cross.imag

            h_plus = h_plus + hp_mode
            h_cross = h_cross + hc_mode

        return h_plus, h_cross

    @staticmethod
    def frequency_domain(
        frequencies: torch.Tensor,
        final_mass: Union[float, torch.Tensor],
        final_spin: Union[float, torch.Tensor],
        amplitudes: Optional[Dict[Tuple[int, int, int], Union[float, torch.Tensor]]] = None,
        phases: Optional[Dict[Tuple[int, int, int], Union[float, torch.Tensor]]] = None,
        t0: Union[float, torch.Tensor] = 0.0,
        inclination: Union[float, torch.Tensor] = 0.0,
        phi: Union[float, torch.Tensor] = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Analytic Fourier-domain Lorentzians for ringdown modes.

        h_tilde_{lmn}(f) = [A_{lmn} exp(i phi_{lmn}) / (2*pi)] / [tau^{-1} - i (2*pi*f - omega)]
        """
        device = frequencies.device
        dtype = frequencies.dtype if frequencies.is_floating_point() else torch.float64
        complex_dtype = torch.complex128 if dtype == torch.float64 else torch.complex64

        f = as_tensor(frequencies, device=device, dtype=dtype)
        mf = as_tensor(final_mass, device=device, dtype=dtype)
        af = as_tensor(final_spin, device=device, dtype=dtype)
        t_0 = as_tensor(t0, device=device, dtype=dtype)
        inc = as_tensor(inclination, device=device, dtype=dtype)
        phi_0 = as_tensor(phi, device=device, dtype=dtype)

        if amplitudes is None:
            amplitudes = {(2, 2, 0): 1.0}
        if phases is None:
            phases = {}

        is_batched = mf.dim() >= 1
        if is_batched and f.dim() == 1:
            f_eval = f.unsqueeze(0)
        else:
            f_eval = f

        def unsqueeze_param(tensor: torch.Tensor) -> torch.Tensor:
            if is_batched and f_eval.dim() > tensor.dim():
                return tensor.unsqueeze(-1)
            return tensor

        t0_eval = unsqueeze_param(t_0)
        inc_eval = unsqueeze_param(inc)
        phi_eval = unsqueeze_param(phi_0)

        out_shape = torch.broadcast_shapes(f_eval.shape, t0_eval.shape)
        hp_total = torch.zeros(out_shape, dtype=complex_dtype, device=device)
        hc_total = torch.zeros(out_shape, dtype=complex_dtype, device=device)

        # Time shift phase: exp(-2*pi*i*f * t0)
        omega_f = 2.0 * PI * f_eval
        time_shift = torch.complex(torch.cos(-omega_f * t0_eval), torch.sin(-omega_f * t0_eval))

        for (l, m, n), amp in amplitudes.items():
            amp_t = as_tensor(amp, device=device, dtype=dtype)
            phase_val = phases.get((l, m, n), 0.0)
            phase_t = as_tensor(phase_val, device=device, dtype=dtype)

            omega, tau, _ = evaluate_qnm(l, m, n, mf, af)
            omega = unsqueeze_param(omega)
            tau = unsqueeze_param(tau)
            amp_t = unsqueeze_param(amp_t)
            phase_t = unsqueeze_param(phase_t)

            # Lorentzian denominator: gamma - i * (omega_f - omega_0)
            gamma = 1.0 / tau
            delta_omega = omega_f - omega

            # A * exp(i * phi) / (2 * pi)
            numerator = (amp_t / (2.0 * PI)) * torch.complex(torch.cos(phase_t), torch.sin(phase_t))
            denominator = torch.complex(gamma, -delta_omega)
            h_mode_lorentzian = (numerator / denominator) * time_shift

            d_plus, d_cross = _swsh_projection_coefficients(l, m, inc_eval, phi_eval)

            hp_total = hp_total + d_plus * h_mode_lorentzian
            hc_total = hc_total + d_cross * h_mode_lorentzian

        return hp_total, hc_total


def get_ringdown_waveform(
    final_mass: Union[float, torch.Tensor],
    final_spin: Union[float, torch.Tensor],
    amplitudes: Optional[Dict[Tuple[int, int, int], Union[float, torch.Tensor]]] = None,
    phases: Optional[Dict[Tuple[int, int, int], Union[float, torch.Tensor]]] = None,
    times: Optional[torch.Tensor] = None,
    frequencies: Optional[torch.Tensor] = None,
    t0: Union[float, torch.Tensor] = 0.0,
    inclination: Union[float, torch.Tensor] = 0.0,
    phi: Union[float, torch.Tensor] = 0.0,
    delta_t: Optional[float] = None,
    duration: Optional[float] = None,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """High-level API for Kerr black hole ringdown waveforms.

    Args:
        final_mass: Remnant mass in solar masses
        final_spin: Dimensionless remnant Kerr spin a_f (0 <= a_f < 1)
        amplitudes: Dictionary mapping (l, m, n) mode tuples to amplitudes
        phases: Dictionary mapping (l, m, n) mode tuples to initial phases
        times: Evaluation time array in seconds (if time domain requested)
        frequencies: Evaluation frequency array in Hz (if frequency domain requested)
        t0: Ringdown start time offset in seconds
        inclination: Observer inclination in radians
        phi: Observer azimuthal angle in radians
        delta_t, duration: Optional parameters to construct uniform time grid if times is None
    """
    if frequencies is not None:
        return KerrRingdown.frequency_domain(
            frequencies=frequencies,
            final_mass=final_mass,
            final_spin=final_spin,
            amplitudes=amplitudes,
            phases=phases,
            t0=t0,
            inclination=inclination,
            phi=phi,
        )

    if times is None:
        dt = delta_t if delta_t is not None else (1.0 / 2048.0)
        dur = duration if duration is not None else 0.1
        times = torch.arange(0.0, dur, dt, dtype=dtype, device=device)

    return KerrRingdown.time_domain(
        times=times,
        final_mass=final_mass,
        final_spin=final_spin,
        amplitudes=amplitudes,
        phases=phases,
        t0=t0,
        inclination=inclination,
        phi=phi,
    )

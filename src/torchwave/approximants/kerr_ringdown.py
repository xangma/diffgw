"""Kerr Black Hole Ringdown Overtones & Spectroscopy Approximant.

The numerical frequency backend uses TorchWave's own Kerr eigenproblem solves.
The optional ``berti`` backend retains the inherited approximate fit tables.
Waveform equations and the Fourier transform are derived from:
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
from torchwave.approximants.kerr_ringdown_energy import (
    ringdown_amplitude_from_energy_fraction,
)
from torchwave.approximants.kerr_qnm_generated import dimensionless_qnm
from torchwave.approximants.kerr_angular import spin_weighted_spheroidal_harmonic


def _validate_single_mode(mode):
    """Validate the bounded single-mode reference contract."""
    if not isinstance(mode, tuple) or len(mode) != 3:
        raise ValueError("mode must be a (l, m, n) tuple")
    if not all(isinstance(index, int) for index in mode):
        raise TypeError("mode indices must be integers")
    l, m, n = mode
    if l < 2 or m <= 0 or m > l or n < 0:
        raise ValueError("mode must have l >= 2, 0 < m <= l, and n >= 0")
    return l, m, n


def sampled_kerr_single_mode_reference(
    times: torch.Tensor,
    final_mass: Union[float, torch.Tensor],
    final_spin: Union[float, torch.Tensor],
    amplitude: Union[float, torch.Tensor],
    *,
    mode: Tuple[int, int, int] = (2, 2, 0),
    phase: Union[float, torch.Tensor] = 0.0,
    t0: Union[float, torch.Tensor] = 0.0,
    inclination: Union[float, torch.Tensor] = 0.0,
    phi: Union[float, torch.Tensor] = 0.0,
    angular_basis: str = "spherical",
    qnm_backend: str = "independent",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return a clean-room, sampled, single-mode Kerr reference waveform.

    This is the fully specified mathematical adapter

    ``A exp(-(t-t0)/tau) exp(i*(omega*(t-t0)+phase))``

    with causal support and the selected normalized angular basis.  ``amplitude``
    is the positive-m complex-mode coefficient at onset and is deliberately
    required: no interpretation of an external fractional mass-loss argument,
    eigenfunction phase, or distance convention is inferred here.  The
    positive-m mode and its equatorial mirror are projected to real plus and
    cross polarizations by :class:`KerrRingdown`.

    QNM frequencies come from ``evaluate_qnm``.  The default spherical basis
    and all phase, onset, and sampling conventions are pinned by this function;
    use :func:`KerrRingdown` for the legacy multi-mode/default API.
    """
    _validate_single_mode(mode)
    if not isinstance(times, torch.Tensor):
        raise TypeError("times must be a floating torch.Tensor")
    if times.ndim == 0 or not times.is_floating_point():
        raise TypeError("times must be a non-scalar floating torch.Tensor")
    if angular_basis not in ("spherical", "spheroidal"):
        raise ValueError("angular_basis must be 'spherical' or 'spheroidal'")
    l, m, n = mode
    dtype = times.dtype
    device = times.device
    t = times
    mf = as_tensor(final_mass, device=device, dtype=dtype)
    af = as_tensor(final_spin, device=device, dtype=dtype)
    amp = as_tensor(amplitude, device=device, dtype=dtype)
    ph = as_tensor(phase, device=device, dtype=dtype)
    onset = as_tensor(t0, device=device, dtype=dtype)
    inc = as_tensor(inclination, device=device, dtype=dtype)
    az = as_tensor(phi, device=device, dtype=dtype)
    params = (mf, af, amp, ph, onset, inc, az)
    if t.ndim == 1 and any(p.ndim for p in params):
        t = t.unsqueeze(0)
    t, mf, af, amp, ph, onset, inc, az = torch.broadcast_tensors(
        t, mf, af, amp, ph, onset, inc, az
    )
    omega, tau, _ = evaluate_qnm(l, m, n, mf, af, qnm_backend=qnm_backend)
    dt = t - onset
    support = dt >= 0
    u = torch.clamp(dt, min=0.0)
    envelope = torch.exp(-u / tau) * support
    c = af * mf * MSUN_SEC * torch.complex(omega, -1.0 / tau)
    d_plus, d_cross = _ringdown_projection(l, m, c, inc, az, angular_basis)
    signal = amp * envelope * torch.complex(torch.cos(omega * u + ph), torch.sin(omega * u + ph))
    return (torch.real(signal * d_plus), torch.real(signal * d_cross))


# Concise alias used by validation scripts and downstream callers.
kerr_single_mode_reference = sampled_kerr_single_mode_reference


def kerr_single_mode_fourier_reference(
    frequencies: torch.Tensor,
    final_mass: Union[float, torch.Tensor],
    final_spin: Union[float, torch.Tensor],
    amplitude: Union[float, torch.Tensor],
    *,
    mode: Tuple[int, int, int] = (2, 2, 0),
    phase: Union[float, torch.Tensor] = 0.0,
    t0: Union[float, torch.Tensor] = 0.0,
    inclination: Union[float, torch.Tensor] = 0.0,
    phi: Union[float, torch.Tensor] = 0.0,
    angular_basis: str = "spherical",
    qnm_backend: str = "independent",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Analytic Fourier transform paired with the sampled reference adapter.

    The transform is ``integral h(t) exp(-2*pi*i*f*t) dt`` over all time;
    both positive and negative real-signal poles are retained.  Frequencies
    are in Hz and the returned values are complex tensors.  This function
    requires the same explicit amplitude and conventions as
    :func:`sampled_kerr_single_mode_reference`.
    """
    _validate_single_mode(mode)
    if not isinstance(frequencies, torch.Tensor):
        raise TypeError("frequencies must be a floating torch.Tensor")
    if frequencies.ndim == 0 or not frequencies.is_floating_point():
        raise TypeError("frequencies must be a non-scalar floating torch.Tensor")
    if angular_basis not in ("spherical", "spheroidal"):
        raise ValueError("angular_basis must be 'spherical' or 'spheroidal'")
    l, m, n = mode
    dtype = frequencies.dtype
    device = frequencies.device
    f = frequencies
    mf = as_tensor(final_mass, device=device, dtype=dtype)
    af = as_tensor(final_spin, device=device, dtype=dtype)
    amp = as_tensor(amplitude, device=device, dtype=dtype)
    ph = as_tensor(phase, device=device, dtype=dtype)
    onset = as_tensor(t0, device=device, dtype=dtype)
    inc = as_tensor(inclination, device=device, dtype=dtype)
    az = as_tensor(phi, device=device, dtype=dtype)
    params = (mf, af, amp, ph, onset, inc, az)
    if f.ndim == 1 and any(p.ndim for p in params):
        f = f.unsqueeze(0)
    f, mf, af, amp, ph, onset, inc, az = torch.broadcast_tensors(
        f, mf, af, amp, ph, onset, inc, az
    )
    omega, tau, _ = evaluate_qnm(l, m, n, mf, af, qnm_backend=qnm_backend)
    gamma = 1.0 / tau
    c = af * mf * MSUN_SEC * torch.complex(omega, -gamma)
    d_plus, d_cross = _ringdown_projection(l, m, c, inc, az, angular_basis)
    carrier = amp * torch.complex(torch.cos(ph), torch.sin(ph))
    wf = 2.0 * PI * f
    shift = torch.complex(torch.cos(-wf * onset), torch.sin(-wf * onset))
    plus = 0.5 * shift * (carrier * d_plus / (gamma + 1j * (wf - omega))
                          + (carrier * d_plus).conj() / (gamma + 1j * (wf + omega)))
    cross = 0.5 * shift * (carrier * d_cross / (gamma + 1j * (wf - omega))
                           + (carrier * d_cross).conj() / (gamma + 1j * (wf + omega)))
    return plus, cross


def _ringdown_projection(l, m, c, inclination, phi, angular_basis):
    if angular_basis == "spherical":
        return _swsh_projection_coefficients(l, m, inclination, phi)
    if angular_basis != "spheroidal":
        raise ValueError("angular_basis must be 'spherical' or 'spheroidal'")
    # Equatorial partner: omega_-=-omega_+*, c_-=-c_+*. Cook–Zalutskiy
    # Eqs. (48), (57) allow both angular factors from one +m eigensolution.
    theta = torch.stack((inclination, PI - inclination), dim=-1)
    values = spin_weighted_spheroidal_harmonic(l, m, c[..., None], theta, phi[..., None])
    positive_conjugate = values[..., 0].conj()
    mirror = ((-1) ** l) * values[..., 1].conj()
    return (0.5 * (mirror + ((-1) ** l) * positive_conjugate),
            0.5j * (mirror - ((-1) ** l) * positive_conjugate))


def _broadcast_parameters(parameters, amplitudes, phases):
    shape = torch.broadcast_shapes(*(p.shape for p in parameters),
                                   *(torch.as_tensor(v).shape for v in amplitudes.values()),
                                   *(torch.as_tensor(v).shape for v in phases.values()))
    return tuple(p.expand(shape) for p in parameters)


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
    *,
    qnm_backend: str = "independent",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute physical angular frequency omega (rad/s), damping time tau (s), and quality factor Q.

    Args:
        l, m, n: Harmonic and overtone indices
        final_mass: Remnant mass in solar masses
        final_spin: Dimensionless remnant spin, 0 <= a_f <= 0.99999 for independent data
        qnm_backend: ``independent`` (own equation solves) or ``berti`` (approximate fits);
            ``author`` is a deprecated alias for ``independent``

    Returns:
        (omega_rad_s, tau_s, Q)
    """
    template = next((x for x in (final_mass, final_spin) if isinstance(x, torch.Tensor)), None)
    dtype = template.dtype if template is not None and template.is_floating_point() else torch.float64
    device = template.device if template is not None else None
    mf = torch.as_tensor(final_mass, dtype=dtype, device=device)
    af = torch.as_tensor(final_spin, dtype=dtype, device=device)
    mf, af = torch.broadcast_tensors(mf, af)
    if bool(torch.any(~torch.isfinite(mf) | (mf <= 0))):
        raise ValueError("final_mass must be finite and positive")
    m_sec = mf * MSUN_SEC
    if qnm_backend in ("independent", "author"):
        m_omega, m_decay = dimensionless_qnm(l, m, n, af)
        return m_omega / m_sec, m_sec / m_decay, m_omega / (2.0 * m_decay)
    if qnm_backend != "berti":
        raise ValueError("qnm_backend must be 'independent', 'berti', or the legacy alias 'author'")

    key = (l, m, n)
    if key not in BERTI_QNM_FITS:
        raise ValueError(f"Mode (l={l}, m={m}, n={n}) not in Berti et al. fit tables.")

    fit = BERTI_QNM_FITS[key]
    f1, f2, f3 = fit["f"]
    q1, q2, q3 = fit["q"]

    # Clamp 1 - a_f to strictly positive range for numerical safety
    one_minus_af = torch.clamp(1.0 - af, min=1e-5, max=1.0)

    # Dimensionless M * omega and Q
    m_omega = f1 + f2 * (one_minus_af ** f3)
    q_val = q1 + q2 * (one_minus_af ** q3)

    # Remnant mass in seconds: M_sec = final_mass * MSUN_SEC

    # Physical omega (rad/s) and damping time tau (s)
    omega_phys = m_omega / m_sec
    tau_phys = (2.0 * q_val) / omega_phys

    return omega_phys, tau_phys, q_val


def _amplitudes_from_energy_fraction(
    final_mass,
    final_spin,
    distance,
    energy_fraction,
    mode,
    *,
    qnm_backend,
    angular_basis,
):
    """Construct the native equal-mirror amplitude for one physical mode.

    The energy input is the total energy of the equal-amplitude positive-m and
    mirror pair.  This is deliberately restricted to one positive-m mode: a
    coherent multi-mode energy requires cross terms and independent twin data.
    """
    if distance is None:
        raise ValueError("distance is required when energy_fraction is supplied")
    if angular_basis != "spheroidal":
        raise ValueError(
            "energy_fraction requires angular_basis='spheroidal' for physical Kerr modes"
        )
    if not isinstance(mode, tuple) or len(mode) != 3:
        raise ValueError("energy_mode must be a (l, m, n) tuple")
    l, m, n = mode
    if not all(isinstance(index, int) for index in mode):
        raise TypeError("energy_mode indices must be integers")
    if l < 2 or m <= 0 or m > l or n < 0:
        raise ValueError("energy_mode must have l >= 2, 0 < m <= l, and n >= 0")

    omega, tau, _ = evaluate_qnm(
        l, m, n, final_mass, final_spin, qnm_backend=qnm_backend
    )
    amplitude = ringdown_amplitude_from_energy_fraction(
        final_mass,
        distance,
        energy_fraction,
        omega,
        tau,
        convention="native_pair",
    )
    return {mode: amplitude}


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
        *,
        qnm_backend: str = "independent",
        angular_basis: str = "spherical",
        distance: Optional[Union[float, torch.Tensor]] = None,
        energy_fraction: Optional[Union[float, torch.Tensor]] = None,
        energy_mode: Tuple[int, int, int] = (2, 2, 0),
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate time-domain damped sinusoids h_plus(t) and h_cross(t).

        Each positive-m mode includes its equatorial-symmetry partner through
        the selected angular projection coefficients. The default is spherical;
        ``spheroidal`` uses the angular Teukolsky eigenfunction and its mirror.
        The polarizations are real, causal damped sinusoids, zero for t < t0.
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
            if energy_fraction is None:
                amplitudes = {(2, 2, 0): 1.0}
            else:
                amplitudes = _amplitudes_from_energy_fraction(
                    final_mass,
                    final_spin,
                    distance,
                    energy_fraction,
                    energy_mode,
                    qnm_backend=qnm_backend,
                    angular_basis=angular_basis,
                )
        elif energy_fraction is not None:
            raise ValueError("amplitudes and energy_fraction are mutually exclusive")
        if phases is None:
            phases = {}

        mf, af, t_0, inc, phi_0 = _broadcast_parameters((mf, af, t_0, inc, phi_0), amplitudes, phases)

        is_batched = mf.dim() >= 1
        if is_batched and t.dim() == 1:
            t_eval = t.unsqueeze(0)
        else:
            t_eval = t

        def unsqueeze_param(tensor: torch.Tensor) -> torch.Tensor:
            if is_batched:
                return tensor.unsqueeze(-1)
            return tensor

        t0_eval = unsqueeze_param(t_0)
        inc_eval = unsqueeze_param(inc)
        phi_eval = unsqueeze_param(phi_0)

        dt = t_eval - t0_eval
        heaviside = torch.where(dt >= 0.0, torch.ones_like(dt), torch.zeros_like(dt))
        causal_dt = torch.clamp(dt, min=0.0)

        h_plus = torch.zeros_like(dt)
        h_cross = torch.zeros_like(dt)

        for (l, m, n), amp in amplitudes.items():
            amp_t = as_tensor(amp, device=device, dtype=dtype)
            phase_val = phases.get((l, m, n), 0.0)
            phase_t = as_tensor(phase_val, device=device, dtype=dtype)

            omega, tau, _ = evaluate_qnm(l, m, n, mf, af, qnm_backend=qnm_backend)
            omega = unsqueeze_param(omega)
            tau = unsqueeze_param(tau)
            amp_t = unsqueeze_param(amp_t)
            phase_t = unsqueeze_param(phase_t)

            # Avoid exp(+|dt|/tau) * 0, which overflows before onset.
            decay = torch.exp(-causal_dt / tau) * heaviside
            arg = omega * causal_dt + phase_t
            cos_term = torch.cos(arg)
            sin_term = torch.sin(arg)

            c = unsqueeze_param(af * mf * MSUN_SEC) * torch.complex(omega, -1.0 / tau)
            d_plus, d_cross = _ringdown_projection(l, m, c, inc_eval, phi_eval, angular_basis)

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
        *,
        qnm_backend: str = "independent",
        angular_basis: str = "spherical",
        distance: Optional[Union[float, torch.Tensor]] = None,
        energy_fraction: Optional[Union[float, torch.Tensor]] = None,
        energy_mode: Tuple[int, int, int] = (2, 2, 0),
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Exact Fourier transform of the real, causal time-domain waveform.

        The convention is integral h(t) exp(-2*pi*i*f*t) dt, with f in Hz.
        For C=A*exp(i*phase)*d, each real polarization contributes
        exp(-2*pi*i*f*t0)/2 * [C/(gamma+i*(2*pi*f-omega))
        + conj(C)/(gamma+i*(2*pi*f+omega))]. Both frequency poles are needed.
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
            if energy_fraction is None:
                amplitudes = {(2, 2, 0): 1.0}
            else:
                amplitudes = _amplitudes_from_energy_fraction(
                    final_mass,
                    final_spin,
                    distance,
                    energy_fraction,
                    energy_mode,
                    qnm_backend=qnm_backend,
                    angular_basis=angular_basis,
                )
        elif energy_fraction is not None:
            raise ValueError("amplitudes and energy_fraction are mutually exclusive")
        if phases is None:
            phases = {}

        mf, af, t_0, inc, phi_0 = _broadcast_parameters((mf, af, t_0, inc, phi_0), amplitudes, phases)

        is_batched = mf.dim() >= 1
        if is_batched and f.dim() == 1:
            f_eval = f.unsqueeze(0)
        else:
            f_eval = f

        def unsqueeze_param(tensor: torch.Tensor) -> torch.Tensor:
            if is_batched:
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

            omega, tau, _ = evaluate_qnm(l, m, n, mf, af, qnm_backend=qnm_backend)
            omega = unsqueeze_param(omega)
            tau = unsqueeze_param(tau)
            amp_t = unsqueeze_param(amp_t)
            phase_t = unsqueeze_param(phase_t)

            gamma = 1.0 / tau
            positive_pole = gamma + 1j * (omega_f - omega)
            negative_pole = gamma + 1j * (omega_f + omega)
            amplitude = amp_t * torch.complex(torch.cos(phase_t), torch.sin(phase_t))

            c = unsqueeze_param(af * mf * MSUN_SEC) * torch.complex(omega, -1.0 / tau)
            d_plus, d_cross = _ringdown_projection(l, m, c, inc_eval, phi_eval, angular_basis)
            cp, cc = amplitude * d_plus, amplitude * d_cross
            hp_total = hp_total + 0.5 * time_shift * (cp / positive_pole + cp.conj() / negative_pole)
            hc_total = hc_total + 0.5 * time_shift * (cc / positive_pole + cc.conj() / negative_pole)

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
    *,
    qnm_backend: str = "independent",
    angular_basis: str = "spherical",
    distance: Optional[Union[float, torch.Tensor]] = None,
    energy_fraction: Optional[Union[float, torch.Tensor]] = None,
    energy_mode: Tuple[int, int, int] = (2, 2, 0),
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
        qnm_backend: Numerical ``independent`` frequencies (default) or approximate ``berti`` fits
        angular_basis: ``spherical`` (default) or normalized ``spheroidal`` harmonics
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
            qnm_backend=qnm_backend,
            angular_basis=angular_basis,
            distance=distance,
            energy_fraction=energy_fraction,
            energy_mode=energy_mode,
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
        qnm_backend=qnm_backend,
        angular_basis=angular_basis,
        distance=distance,
        energy_fraction=energy_fraction,
        energy_mode=energy_mode,
    )

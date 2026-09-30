"""Kerr Black Hole Ringdown Overtones & Spectroscopy Approximant in JAX.

Port of torchwave.approximants.kerr_ringdown for JAX / jaxwave.
Native JAX operations (jax.numpy, float64 precision).
Supports independent Kerr QNM solves and Berti et al. fit formulas.
Waveform equations and Fourier transform derived from:
- Berti, Cardoso, Will (2006), Phys. Rev. D 73, 064030 (arXiv:gr-qc/0512160)
- Giesler et al. (2019), Phys. Rev. X 9, 041060 (arXiv:1903.08284)
- Ruiz et al. (2007), arXiv:0707.4654v3, Eq. (2.8)
"""

from typing import Dict, Optional, Sequence, Tuple, Union
import math
import numpy as np
import jax
import jax.numpy as jnp

from diffgw_core.constants import (
    C_SI,
    MPC_SI,
    MSUN_SEC,
    PI,
)
from jaxwave.conversions import as_array
from jaxwave.approximants.imrphenom_hm import _swsh_projection_coefficients
from jaxwave.approximants.kerr_qnm_interp import evaluate_kerr_qnm_220, _dimensionless_qnm

from diffgw_core.kerr_qnm_data import _SPLINES


def _get_jax_spline(mode: Tuple[int, int, int]) -> Optional[Tuple[jnp.ndarray, jnp.ndarray]]:
    # Do not cache traced arrays: a first call under jit must not leak tracers.
    if mode in _SPLINES:
        knots, coeffs = _SPLINES[mode]
        return jnp.asarray(knots, dtype=jnp.float64), jnp.asarray(coeffs, dtype=jnp.float64)
    return None


def _validate_single_mode(mode: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """Validate the bounded single-mode reference contract."""
    if not isinstance(mode, tuple) or len(mode) != 3:
        raise ValueError("mode must be a (l, m, n) tuple")
    if not all(isinstance(index, int) for index in mode):
        raise TypeError("mode indices must be integers")
    l, m, n = mode
    if l < 2 or m <= 0 or m > l or n < 0:
        raise ValueError("mode must have l >= 2, 0 < m <= l, and n >= 0")
    return l, m, n


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
    final_mass: Union[float, jnp.ndarray],
    final_spin: Union[float, jnp.ndarray],
    *,
    qnm_backend: str = "independent",
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute physical angular frequency omega (rad/s), damping time tau (s), and quality factor Q.

    Args:
        l, m, n: Harmonic and overtone indices
        final_mass: Remnant mass in solar masses
        final_spin: Dimensionless remnant spin, 0 <= a_f <= 0.99999
        qnm_backend: ``independent`` (own equation solves) or ``berti`` (approximate fits);
            ``author`` is an alias for ``independent``

    Returns:
        (omega_rad_s, tau_s, Q)
    """
    mf = as_array(final_mass, dtype=jnp.float64)
    af = as_array(final_spin, dtype=jnp.float64)
    mf, af = jnp.broadcast_arrays(mf, af)

    if not isinstance(mf, jax.core.Tracer):
        if bool(jnp.any(~jnp.isfinite(mf) | (mf <= 0))):
            raise ValueError("final_mass must be finite and positive")

    m_sec = mf * MSUN_SEC

    if qnm_backend in ("independent", "author"):
        if (l, m, n) == (2, 2, 0):
            m_omega, m_decay = evaluate_kerr_qnm_220(af)
            return m_omega / m_sec, m_sec / m_decay, m_omega / (2.0 * m_decay)

        spline = _get_jax_spline((l, m, n))
        if spline is not None:
            knots, coeffs = spline
            spin_clamped = jnp.maximum(0.0, jnp.minimum(af, 0.99999))
            x = -jnp.log1p(-spin_clamped)
            m_omega, m_decay = _dimensionless_qnm(x, knots, coeffs)
            return m_omega / m_sec, m_sec / m_decay, m_omega / (2.0 * m_decay)

        raise ValueError(f"Mode (l={l}, m={m}, n={n}) not available for independent QNM backend.")

    if qnm_backend != "berti":
        raise ValueError("qnm_backend must be 'independent', 'berti', or the legacy alias 'author'")

    key = (l, m, n)
    if key not in BERTI_QNM_FITS:
        raise ValueError(f"Mode (l={l}, m={m}, n={n}) not in Berti et al. fit tables.")

    fit = BERTI_QNM_FITS[key]
    f1, f2, f3 = fit["f"]
    q1, q2, q3 = fit["q"]

    # Clamp 1 - a_f to strictly positive range for numerical safety
    one_minus_af = jnp.maximum(1e-5, jnp.minimum(1.0 - af, 1.0))

    # Dimensionless M * omega and Q
    m_omega = f1 + f2 * (one_minus_af ** f3)
    q_val = q1 + q2 * (one_minus_af ** q3)

    # Physical omega (rad/s) and damping time tau (s)
    omega_phys = m_omega / m_sec
    tau_phys = (2.0 * q_val) / omega_phys

    return omega_phys, tau_phys, q_val


def ringdown_amplitude_from_energy_fraction(
    final_mass: Union[float, jnp.ndarray],
    distance: Union[float, jnp.ndarray],
    energy_fraction: Union[float, jnp.ndarray],
    omega: Union[float, jnp.ndarray],
    tau: Union[float, jnp.ndarray],
    *,
    convention: str = "native_pair",
) -> jnp.ndarray:
    """Return the positive onset strain amplitude for a given emitted energy.

    ``final_mass`` is remnant mass in solar masses, ``distance`` in Mpc,
    ``omega`` is angular frequency in rad/s, and ``tau`` is damping time in seconds.
    ``energy_fraction`` is nonnegative energy divided by remnant rest energy.
    """
    factors = {
        "single_mode": 1.0,
        "pair_member": 1.0 / math.sqrt(2),
        "native_pair": math.sqrt(2),
    }
    if convention not in factors:
        raise ValueError("convention must be 'single_mode', 'pair_member', or 'native_pair'")

    names = ("final_mass", "distance", "energy_fraction", "omega", "tau")
    values = (final_mass, distance, energy_fraction, omega, tau)

    converted = [as_array(v, dtype=jnp.float64) for v in values]

    for name, v in zip(names, converted):
        if not isinstance(v, jax.core.Tracer):
            invalid_sign = (v < 0) if name == "energy_fraction" else (v <= 0)
            if bool(jnp.any(~jnp.isfinite(v) | invalid_sign)):
                domain = "nonnegative" if name == "energy_fraction" else "positive"
                raise ValueError(f"{name} must be finite and {domain}")

    mass, dist, eff, om, ta = jnp.broadcast_arrays(*converted)

    log_omega_tau = jnp.log(om) + jnp.log(ta)
    log_amplitude = (
        0.5 * (math.log(32.0 * math.pi * MSUN_SEC) + jnp.log(mass) + jnp.log(ta))
        - math.log(MPC_SI / C_SI)
        - jnp.log(dist)
        - 0.5 * jnp.logaddexp(jnp.zeros_like(log_omega_tau), 2.0 * log_omega_tau)
    )
    return factors[convention] * jnp.exp(log_amplitude) * jnp.sqrt(eff)


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
    """Construct the native equal-mirror amplitude for one physical mode."""
    if distance is None:
        raise ValueError("distance is required when energy_fraction is supplied")
    if angular_basis != "spheroidal":
        raise ValueError(
            "energy_fraction requires angular_basis='spheroidal' for physical Kerr modes"
        )
    _validate_single_mode(mode)
    l, m, n = mode

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


def _angular_matrix(m: int, c: jnp.ndarray, ell_max: int) -> jnp.ndarray:
    """Represent A0 - c^2 x^2 + 2cs x, for spin weight s=-2."""
    ell_min = max(2, abs(m))
    ell = jnp.arange(ell_min, ell_max + 2, dtype=jnp.float64)
    diagonal = 2.0 * m / (ell * (ell + 1.0))
    upper_ell = ell[1:]
    off_diagonal = jnp.sqrt(
        (upper_ell**2 - m * m) * (upper_ell**2 - 4.0)
        / (upper_ell**2 * (4.0 * upper_ell**2 - 1.0))
    )
    x = jnp.diag(diagonal) + jnp.diag(off_diagonal, 1) + jnp.diag(off_diagonal, -1)
    x2 = (x @ x)[:-1, :-1]
    x = x[:-1, :-1]
    a0 = jnp.diag(ell[:-1] * (ell[:-1] + 1.0) - 2.0)
    c_bc = jnp.expand_dims(c, (-1, -2))
    return a0 - (c_bc**2) * x2 - 4.0 * c_bc * x


def spheroidal_coefficients(
    ell: int, m: int, c: jnp.ndarray, *, ell_max: int = 18,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Return separation constant A and unit-norm spherical coefficients.

    Continue the requested spin-weighted spheroidal branch (s=-2) from c=0
    using native inverse iteration. Static continuation and Rayleigh loops
    support JIT, batching and reverse-mode differentiation with respect to
    complex oblateness c and Kerr spin a.
    """
    if ell < max(2, abs(m)) or ell_max < ell + 3:
        raise ValueError("Require ell >= max(2, |m|) and ell_max >= ell+3.")
    c = jnp.asarray(c, dtype=jnp.complex128)
    orig_shape = c.shape
    c_flat = c.reshape(-1)
    n_dim = ell_max - max(2, abs(m)) + 1
    idx0 = ell - max(2, abs(m))

    # Initial basis vector: e_{idx0} with shape (batch, n_dim, 1)
    v = jnp.zeros((c_flat.shape[0], n_dim, 1), dtype=jnp.complex128).at[:, idx0, 0].set(1.0)
    eye = jnp.eye(n_dim, dtype=jnp.complex128)[None, :, :]

    sigma = jnp.full((c_flat.shape[0], 1, 1), ell * (ell + 1) - 2,
                     dtype=jnp.complex128)

    # A fixed spherical shift can converge to a neighboring ell branch when
    # c moves its eigenvalue far enough. Carry the eigenpair along the same
    # straight continuation path used to define the angular mode instead.
    def continue_branch(state, fraction):
        mat = _angular_matrix(m, c_flat * fraction, ell_max)

        def rayleigh_iteration(_, eigenpair):
            sigma, v = eigenpair
            # The imaginary offset keeps solves finite at an exact eigenpair.
            v = jnp.linalg.solve(mat - (sigma + 1e-5j) * eye, v)
            v = v / jnp.linalg.norm(v, axis=1, keepdims=True)
            sigma = jnp.swapaxes(jnp.conj(v), -1, -2) @ mat @ v
            return sigma, v

        return jax.lax.fori_loop(0, 4, rayleigh_iteration, state), None

    (sigma, v), _ = jax.lax.scan(
        continue_branch, (sigma, v), jnp.arange(1, 9, dtype=jnp.float64) / 8.0)

    anchor = v[:, idx0, 0]
    phase = (jnp.conj(anchor) / jnp.abs(anchor))[:, None, None]
    v = v * phase
    return sigma.reshape(orig_shape), v.squeeze(-1).reshape(orig_shape + (n_dim,))


def spin_weighted_spherical_harmonic(
    ell: int, m: int, theta: jnp.ndarray, phi: Union[float, jnp.ndarray] = 0.0,
) -> jnp.ndarray:
    """Evaluate spin -2 Y_lm using the finite Wigner-d factorial sum."""
    if ell < 2 or abs(m) > ell:
        raise ValueError("Spin -2 harmonics require ell>=2 and |m|<=ell.")
    cosine, sine = jnp.cos(theta / 2.0), jnp.sin(theta / 2.0)
    value = jnp.zeros_like(theta)
    fact = math.factorial
    numerator = math.sqrt(fact(ell + m) * fact(ell - m) * fact(ell + 2) * fact(ell - 2))
    for k in range(max(0, m - 2), min(ell + m, ell - 2) + 1):
        denominator = fact(ell + m - k) * fact(k) * fact(ell - 2 - k) * fact(k - m + 2)
        value = value + ((-1) ** k * numerator / denominator) * (
            cosine ** (2 * ell + m - 2 - 2 * k) * sine ** (2 * k - m + 2)
        )
    azimuth = jnp.asarray(phi, dtype=theta.dtype) * m
    return math.sqrt((2 * ell + 1) / (4.0 * math.pi)) * value * jnp.exp(1j * azimuth)


def spin_weighted_spheroidal_harmonic(
    ell: int, m: int, c: jnp.ndarray, theta: jnp.ndarray,
    phi: Union[float, jnp.ndarray] = 0.0, *, ell_max: int = 18,
) -> jnp.ndarray:
    """Evaluate spin -2 S_lm(theta, phi; c), normalized over the sphere."""
    _, coefficients = spheroidal_coefficients(ell, m, c, ell_max=ell_max)
    basis = jnp.stack([
        spin_weighted_spherical_harmonic(j, m, theta, phi)
        for j in range(max(2, abs(m)), ell_max + 1)
    ], axis=-1)
    return (coefficients * basis).sum(-1)


def _ringdown_projection(l, m, c, inclination, phi, angular_basis):
    if angular_basis == "spherical":
        return _swsh_projection_coefficients(l, m, inclination, phi)
    if angular_basis != "spheroidal":
        raise ValueError("angular_basis must be 'spherical' or 'spheroidal'")
    c = jnp.asarray(c)
    theta = jnp.stack((inclination, PI - inclination), axis=-1)
    c_exp = jnp.expand_dims(c, -1) if c.ndim > 0 else c
    phi_exp = jnp.expand_dims(phi, -1) if hasattr(phi, "ndim") and phi.ndim > 0 else phi
    values = spin_weighted_spheroidal_harmonic(l, m, c_exp, theta, phi_exp)
    # The conjugated negative-m partner equals the positive-m harmonic at
    # reflected inclination. Both terms therefore retain exp(+i m phi).
    positive, mirror = values[..., 0], values[..., 1]
    return 0.5 * (positive + mirror), -0.5j * (positive - mirror)


def _broadcast_parameters(parameters, amplitudes, phases):
    shapes = [p.shape for p in parameters]
    for v in amplitudes.values():
        if hasattr(v, "shape"):
            shapes.append(v.shape)
    for v in phases.values():
        if hasattr(v, "shape"):
            shapes.append(v.shape)
    target_shape = np.broadcast_shapes(*shapes)
    return tuple(jnp.broadcast_to(p, target_shape) for p in parameters)


def sampled_kerr_single_mode_reference(
    times: jnp.ndarray,
    final_mass: Union[float, jnp.ndarray],
    final_spin: Union[float, jnp.ndarray],
    amplitude: Union[float, jnp.ndarray],
    *,
    mode: Tuple[int, int, int] = (2, 2, 0),
    phase: Union[float, jnp.ndarray] = 0.0,
    t0: Union[float, jnp.ndarray] = 0.0,
    inclination: Union[float, jnp.ndarray] = 0.0,
    phi: Union[float, jnp.ndarray] = 0.0,
    angular_basis: str = "spherical",
    qnm_backend: str = "independent",
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Return a sampled single-mode Kerr reference waveform in JAX.

    Generates ``A exp(-(t-t0)/tau) exp(i*(omega*(t-t0)+phase))`` with causal
    support and selected angular basis projection.
    """
    _validate_single_mode(mode)
    if angular_basis not in ("spherical", "spheroidal"):
        raise ValueError("angular_basis must be 'spherical' or 'spheroidal'")
    l, m, n = mode

    t = as_array(times, dtype=jnp.float64)
    mf = as_array(final_mass, dtype=jnp.float64)
    af = as_array(final_spin, dtype=jnp.float64)
    amp = as_array(amplitude, dtype=jnp.float64)
    ph = as_array(phase, dtype=jnp.float64)
    onset = as_array(t0, dtype=jnp.float64)
    inc = as_array(inclination, dtype=jnp.float64)
    az = as_array(phi, dtype=jnp.float64)

    params = (mf, af, amp, ph, onset, inc, az)
    if t.ndim == 1 and any(p.ndim for p in params):
        t = jnp.expand_dims(t, 0)
    t, mf, af, amp, ph, onset, inc, az = jnp.broadcast_arrays(
        t, mf, af, amp, ph, onset, inc, az
    )

    omega, tau, _ = evaluate_qnm(l, m, n, mf, af, qnm_backend=qnm_backend)
    dt = t - onset
    support = jnp.where(dt >= 0.0, 1.0, 0.0)
    u = jnp.maximum(dt, 0.0)
    envelope = jnp.exp(-u / tau) * support
    c = af * mf * MSUN_SEC * (omega - 1j / tau)
    d_plus, d_cross = _ringdown_projection(l, m, c, inc, az, angular_basis)
    signal = amp * envelope * (jnp.cos(omega * u + ph) + 1j * jnp.sin(omega * u + ph))
    return (jnp.real(signal * d_plus), jnp.real(signal * d_cross))


kerr_single_mode_reference = sampled_kerr_single_mode_reference


def kerr_single_mode_fourier_reference(
    frequencies: jnp.ndarray,
    final_mass: Union[float, jnp.ndarray],
    final_spin: Union[float, jnp.ndarray],
    amplitude: Union[float, jnp.ndarray],
    *,
    mode: Tuple[int, int, int] = (2, 2, 0),
    phase: Union[float, jnp.ndarray] = 0.0,
    t0: Union[float, jnp.ndarray] = 0.0,
    inclination: Union[float, jnp.ndarray] = 0.0,
    phi: Union[float, jnp.ndarray] = 0.0,
    angular_basis: str = "spherical",
    qnm_backend: str = "independent",
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Analytic Fourier transform paired with the sampled reference adapter in JAX.

    Exact transform ``integral h(t) exp(-2*pi*i*f*t) dt`` retaining both
    positive and negative real-signal poles.
    """
    _validate_single_mode(mode)
    if angular_basis not in ("spherical", "spheroidal"):
        raise ValueError("angular_basis must be 'spherical' or 'spheroidal'")
    l, m, n = mode

    f = as_array(frequencies, dtype=jnp.float64)
    mf = as_array(final_mass, dtype=jnp.float64)
    af = as_array(final_spin, dtype=jnp.float64)
    amp = as_array(amplitude, dtype=jnp.float64)
    ph = as_array(phase, dtype=jnp.float64)
    onset = as_array(t0, dtype=jnp.float64)
    inc = as_array(inclination, dtype=jnp.float64)
    az = as_array(phi, dtype=jnp.float64)

    params = (mf, af, amp, ph, onset, inc, az)
    if f.ndim == 1 and any(p.ndim for p in params):
        f = jnp.expand_dims(f, 0)
    f, mf, af, amp, ph, onset, inc, az = jnp.broadcast_arrays(
        f, mf, af, amp, ph, onset, inc, az
    )

    omega, tau, _ = evaluate_qnm(l, m, n, mf, af, qnm_backend=qnm_backend)
    gamma = 1.0 / tau
    c = af * mf * MSUN_SEC * (omega - 1j * gamma)
    d_plus, d_cross = _ringdown_projection(l, m, c, inc, az, angular_basis)
    carrier = amp * (jnp.cos(ph) + 1j * jnp.sin(ph))
    wf = 2.0 * PI * f
    shift = jnp.cos(-wf * onset) + 1j * jnp.sin(-wf * onset)
    plus = 0.5 * shift * (
        carrier * d_plus / (gamma + 1j * (wf - omega))
        + jnp.conj(carrier * d_plus) / (gamma + 1j * (wf + omega))
    )
    cross = 0.5 * shift * (
        carrier * d_cross / (gamma + 1j * (wf - omega))
        + jnp.conj(carrier * d_cross) / (gamma + 1j * (wf + omega))
    )
    return plus, cross


@jax.tree_util.register_pytree_node_class
class KerrRingdown:
    """Kerr black hole ringdown waveform model with overtones and multipoles in JAX."""

    def __init__(
        self,
        qnm_backend: str = "independent",
        angular_basis: str = "spherical",
    ):
        self.qnm_backend = qnm_backend
        self.angular_basis = angular_basis

    def tree_flatten(self):
        children = ()
        aux_data = (self.qnm_backend, self.angular_basis)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*aux_data)

    def __call__(self, *args, **kwargs):
        if not isinstance(self, KerrRingdown):
            return KerrRingdown.evaluate(self, *args, **kwargs)
        kwargs.setdefault("qnm_backend", self.qnm_backend)
        kwargs.setdefault("angular_basis", self.angular_basis)
        return KerrRingdown.evaluate(*args, **kwargs)

    @staticmethod
    def time_domain(
        times: jnp.ndarray,
        final_mass: Union[float, jnp.ndarray],
        final_spin: Union[float, jnp.ndarray],
        amplitudes: Optional[Dict[Tuple[int, int, int], Union[float, jnp.ndarray]]] = None,
        phases: Optional[Dict[Tuple[int, int, int], Union[float, jnp.ndarray]]] = None,
        t0: Union[float, jnp.ndarray] = 0.0,
        inclination: Union[float, jnp.ndarray] = 0.0,
        phi: Union[float, jnp.ndarray] = 0.0,
        *,
        qnm_backend: str = "independent",
        angular_basis: str = "spherical",
        distance: Optional[Union[float, jnp.ndarray]] = None,
        energy_fraction: Optional[Union[float, jnp.ndarray]] = None,
        energy_mode: Tuple[int, int, int] = (2, 2, 0),
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Generate time-domain damped sinusoids h_plus(t) and h_cross(t) in JAX."""
        t = as_array(times, dtype=jnp.float64)
        mf = as_array(final_mass, dtype=jnp.float64)
        af = as_array(final_spin, dtype=jnp.float64)
        t_0 = as_array(t0, dtype=jnp.float64)
        inc = as_array(inclination, dtype=jnp.float64)
        phi_0 = as_array(phi, dtype=jnp.float64)

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

        is_batched = mf.ndim >= 1
        if is_batched and t.ndim == 1:
            t_eval = jnp.expand_dims(t, 0)
        else:
            t_eval = t

        def unsqueeze_param(arr: jnp.ndarray) -> jnp.ndarray:
            if is_batched:
                return jnp.expand_dims(arr, -1)
            return arr

        t0_eval = unsqueeze_param(t_0)
        inc_eval = unsqueeze_param(inc)
        phi_eval = unsqueeze_param(phi_0)

        dt = t_eval - t0_eval
        heaviside = jnp.where(dt >= 0.0, 1.0, 0.0)
        causal_dt = jnp.maximum(dt, 0.0)

        h_plus = jnp.zeros_like(dt)
        h_cross = jnp.zeros_like(dt)

        for (l, m, n), amp in amplitudes.items():
            amp_t = as_array(amp, dtype=jnp.float64)
            phase_val = phases.get((l, m, n), 0.0)
            phase_t = as_array(phase_val, dtype=jnp.float64)

            omega, tau, _ = evaluate_qnm(l, m, n, mf, af, qnm_backend=qnm_backend)
            omega = unsqueeze_param(omega)
            tau = unsqueeze_param(tau)
            amp_t = unsqueeze_param(amp_t)
            phase_t = unsqueeze_param(phase_t)

            decay = jnp.exp(-causal_dt / tau) * heaviside
            arg = omega * causal_dt + phase_t
            cos_term = jnp.cos(arg)
            sin_term = jnp.sin(arg)

            c = unsqueeze_param(af * mf * MSUN_SEC) * (omega - 1j / tau)
            d_plus, d_cross = _ringdown_projection(l, m, c, inc_eval, phi_eval, angular_basis)

            s_real = amp_t * decay * cos_term
            s_imag = amp_t * decay * sin_term

            hp_mode = s_real * jnp.real(d_plus) - s_imag * jnp.imag(d_plus)
            hc_mode = s_real * jnp.real(d_cross) - s_imag * jnp.imag(d_cross)

            h_plus = h_plus + hp_mode
            h_cross = h_cross + hc_mode

        return h_plus, h_cross

    @staticmethod
    def frequency_domain(
        frequencies: jnp.ndarray,
        final_mass: Union[float, jnp.ndarray],
        final_spin: Union[float, jnp.ndarray],
        amplitudes: Optional[Dict[Tuple[int, int, int], Union[float, jnp.ndarray]]] = None,
        phases: Optional[Dict[Tuple[int, int, int], Union[float, jnp.ndarray]]] = None,
        t0: Union[float, jnp.ndarray] = 0.0,
        inclination: Union[float, jnp.ndarray] = 0.0,
        phi: Union[float, jnp.ndarray] = 0.0,
        *,
        qnm_backend: str = "independent",
        angular_basis: str = "spherical",
        distance: Optional[Union[float, jnp.ndarray]] = None,
        energy_fraction: Optional[Union[float, jnp.ndarray]] = None,
        energy_mode: Tuple[int, int, int] = (2, 2, 0),
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Exact Fourier transform of the real, causal time-domain waveform in JAX."""
        f = as_array(frequencies, dtype=jnp.float64)
        mf = as_array(final_mass, dtype=jnp.float64)
        af = as_array(final_spin, dtype=jnp.float64)
        t_0 = as_array(t0, dtype=jnp.float64)
        inc = as_array(inclination, dtype=jnp.float64)
        phi_0 = as_array(phi, dtype=jnp.float64)

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

        is_batched = mf.ndim >= 1
        if is_batched and f.ndim == 1:
            f_eval = jnp.expand_dims(f, 0)
        else:
            f_eval = f

        def unsqueeze_param(arr: jnp.ndarray) -> jnp.ndarray:
            if is_batched:
                return jnp.expand_dims(arr, -1)
            return arr

        t0_eval = unsqueeze_param(t_0)
        inc_eval = unsqueeze_param(inc)
        phi_eval = unsqueeze_param(phi_0)

        out_shape = np.broadcast_shapes(f_eval.shape, t0_eval.shape)
        hp_total = jnp.zeros(out_shape, dtype=jnp.complex128)
        hc_total = jnp.zeros(out_shape, dtype=jnp.complex128)

        omega_f = 2.0 * PI * f_eval
        time_shift = jnp.cos(-omega_f * t0_eval) + 1j * jnp.sin(-omega_f * t0_eval)

        for (l, m, n), amp in amplitudes.items():
            amp_t = as_array(amp, dtype=jnp.float64)
            phase_val = phases.get((l, m, n), 0.0)
            phase_t = as_array(phase_val, dtype=jnp.float64)

            omega, tau, _ = evaluate_qnm(l, m, n, mf, af, qnm_backend=qnm_backend)
            omega = unsqueeze_param(omega)
            tau = unsqueeze_param(tau)
            amp_t = unsqueeze_param(amp_t)
            phase_t = unsqueeze_param(phase_t)

            gamma = 1.0 / tau
            positive_pole = gamma + 1j * (omega_f - omega)
            negative_pole = gamma + 1j * (omega_f + omega)
            amplitude = amp_t * (jnp.cos(phase_t) + 1j * jnp.sin(phase_t))

            c = unsqueeze_param(af * mf * MSUN_SEC) * (omega - 1j / tau)
            d_plus, d_cross = _ringdown_projection(l, m, c, inc_eval, phi_eval, angular_basis)
            cp, cc = amplitude * d_plus, amplitude * d_cross
            hp_total = hp_total + 0.5 * time_shift * (cp / positive_pole + jnp.conj(cp) / negative_pole)
            hc_total = hc_total + 0.5 * time_shift * (cc / positive_pole + jnp.conj(cc) / negative_pole)

        return hp_total, hc_total

    @staticmethod
    def evaluate(
        final_mass: Union[float, jnp.ndarray],
        final_spin: Union[float, jnp.ndarray],
        amplitudes: Optional[Dict[Tuple[int, int, int], Union[float, jnp.ndarray]]] = None,
        phases: Optional[Dict[Tuple[int, int, int], Union[float, jnp.ndarray]]] = None,
        times: Optional[jnp.ndarray] = None,
        frequencies: Optional[jnp.ndarray] = None,
        t0: Union[float, jnp.ndarray] = 0.0,
        inclination: Union[float, jnp.ndarray] = 0.0,
        phi: Union[float, jnp.ndarray] = 0.0,
        delta_t: Optional[float] = None,
        duration: Optional[float] = None,
        *,
        qnm_backend: str = "independent",
        angular_basis: str = "spherical",
        distance: Optional[Union[float, jnp.ndarray]] = None,
        energy_fraction: Optional[Union[float, jnp.ndarray]] = None,
        energy_mode: Tuple[int, int, int] = (2, 2, 0),
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """High-level evaluation function returning (h_plus, h_cross)."""
        return get_ringdown_waveform(
            final_mass=final_mass,
            final_spin=final_spin,
            amplitudes=amplitudes,
            phases=phases,
            times=times,
            frequencies=frequencies,
            t0=t0,
            inclination=inclination,
            phi=phi,
            delta_t=delta_t,
            duration=duration,
            qnm_backend=qnm_backend,
            angular_basis=angular_basis,
            distance=distance,
            energy_fraction=energy_fraction,
            energy_mode=energy_mode,
        )


def get_ringdown_waveform(
    final_mass: Union[float, jnp.ndarray],
    final_spin: Union[float, jnp.ndarray],
    amplitudes: Optional[Dict[Tuple[int, int, int], Union[float, jnp.ndarray]]] = None,
    phases: Optional[Dict[Tuple[int, int, int], Union[float, jnp.ndarray]]] = None,
    times: Optional[jnp.ndarray] = None,
    frequencies: Optional[jnp.ndarray] = None,
    t0: Union[float, jnp.ndarray] = 0.0,
    inclination: Union[float, jnp.ndarray] = 0.0,
    phi: Union[float, jnp.ndarray] = 0.0,
    delta_t: Optional[float] = None,
    duration: Optional[float] = None,
    *,
    qnm_backend: str = "independent",
    angular_basis: str = "spherical",
    distance: Optional[Union[float, jnp.ndarray]] = None,
    energy_fraction: Optional[Union[float, jnp.ndarray]] = None,
    energy_mode: Tuple[int, int, int] = (2, 2, 0),
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """High-level API for Kerr black hole ringdown waveforms in JAX."""
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
        times = jnp.arange(0.0, dur, dt, dtype=jnp.float64)

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


__all__ = [
    "BERTI_QNM_FITS",
    "KerrRingdown",
    "evaluate_qnm",
    "get_ringdown_waveform",
    "sampled_kerr_single_mode_reference",
    "kerr_single_mode_reference",
    "kerr_single_mode_fourier_reference",
    "ringdown_amplitude_from_energy_fraction",
    "spin_weighted_spheroidal_harmonic",
    "spheroidal_coefficients",
]

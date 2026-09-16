"""IMRPhenomP Frequency-Domain Gravitational Waveform Approximant in JAX.

Mathematical references for the model and audited corrections:
- Hannam et al. (2014), Phys. Rev. Lett. 113, 151101 (arXiv:1308.3271)
- Schmidt et al. (2015), Phys. Rev. D 91, 024043 (arXiv:1408.1810)
- Pratten et al. (2021), Phys. Rev. D 103, 104056 (arXiv:2004.06503), Appendix B & D
"""

from typing import Optional, Tuple, Union
import math
import jax
import jax.numpy as jnp

from diffgw_core.constants import (
    C_SI,
    MSUN_SEC,
    MPC_SI,
    PI,
)
from jaxwave.conversions import (
    as_array,
    component_masses_to_eta,
    component_masses_to_delta,
)
from jaxwave.approximants.imrphenomd import IMRPhenomD


def _safe_in_plane_norm(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Return the exact Euclidean norm with a finite derivative at zero.

    The direction of a zero in-plane spin is undefined, but its magnitude
    is exactly zero. Selecting a nonzero square-root branch at the origin
    avoids the 0/0 derivative of sqrt(x*x + y*y) without adding a
    physical floor to the cone angle.
    """
    r2 = x * x + y * y
    nonzero = r2 > 0
    r2_safe = jnp.where(nonzero, r2, jnp.ones_like(r2))
    return jnp.where(nonzero, jnp.sqrt(r2_safe), jnp.zeros_like(r2))


def nnlo_angle_coefficients(
    eta: jnp.ndarray,
    delta: jnp.ndarray,
    m1_norm: jnp.ndarray,
    chip: jnp.ndarray,
    chi_l: jnp.ndarray,
) -> Tuple[jnp.ndarray, ...]:
    """Compute NNLO Euler angle polynomial coefficients.

    Equations (G7)–(G9) in Appendix G of Pratten et al. (2021), Phys. Rev. D 103, 104056 [arXiv:2004.06503].
    """
    eta2 = eta * eta
    eta3 = eta2 * eta
    eta4 = eta2 * eta2

    delta2 = delta * delta
    delta3 = delta2 * delta

    m1_2 = m1_norm * m1_norm
    m1_3 = m1_2 * m1_norm
    m1_4 = m1_2 * m1_2
    m1_5 = m1_4 * m1_norm
    m1_6 = m1_4 * m1_2
    m1_7 = m1_6 * m1_norm
    m1_8 = m1_4 * m1_4

    chip2 = chip * chip
    chip4 = chip2 * chip2

    chil2 = chi_l * chi_l

    # alpha coefficients
    a_minus3 = -(5.0 * delta / (64.0 * m1_norm)) - (35.0 / 192.0)
    a_minus2 = -(5.0 * m1_norm * chi_l * (3.0 * delta + 7.0 * m1_norm)) / (128.0 * eta)
    a_minus1 = (
        -(5.0 * (824.0 * eta + 1103.0) / 3072.0)
        - (15.0 * delta2 * eta / (256.0 * m1_2))
        - (15.0 * delta * m1_3 * chip2 / (128.0 * eta2))
        - (5.0 * delta * (980.0 * eta + 911.0) / (7168.0 * m1_norm))
        - (35.0 * m1_4 * chip2 / (128.0 * eta2))
    )
    a_1 = (
        5.0 * (36.0 * eta * (85568.0 * eta + 23817.0) + 8024297.0) / 9289728.0
        + 5.0 * m1_2 * (3.0 * delta2 * chip2 + 75.0 * delta2 * chil2 - 112.0 * PI * chi_l) / (256.0 * eta)
        - 15.0 * delta * m1_7 * (chip4 - 4.0 * chip2 * chil2) / (512.0 * eta4)
        + 5.0 * delta * m1_3 * ((812.0 * eta - 97.0) * chip2 + 20328.0 * eta * chil2) / (14336.0 * eta2)
        - 15.0 * PI * delta * m1_norm * chi_l / (16.0 * eta)
        - 35.0 * m1_8 * (chip4 - 4.0 * chip2 * chil2) / (512.0 * eta4)
        + m1_4 * (25.0 * (92.0 * eta + 19.0) * chip2 + 52640.0 * eta * chil2) / (6144.0 * eta2)
        + 15.0 * delta3 * eta2 / (1024.0 * m1_3)
        + 5.0 * delta2 * eta * (784.0 * eta + 323.0) / (28672.0 * m1_2)
        + 5.0 * delta * (504.0 * eta * (7630.0 * eta - 159.0) + 5579177.0) / (21676032.0 * m1_norm)
    )
    a_log = (
        -(5.0 / 48.0) * (7.0 * PI - 3.0 * delta2 * chi_l)
        - 5.0 * delta * m1_5 * chip2 * chi_l / (128.0 * eta3)
        + 5.0 * delta * (7168.0 * eta + 407.0) * m1_norm * chi_l / (21504.0 * eta)
        - 35.0 * m1_6 * chip2 * chi_l / (384.0 * eta3)
        + 5.0 * (4072.0 * eta + 599.0) * m1_2 * chi_l / (9216.0 * eta)
        - 5.0 * PI * delta / (16.0 * m1_norm)
    )

    # epsilon coefficients
    e_minus3 = a_minus3
    e_minus2 = a_minus2
    e_minus1 = (
        -(5.0 * (824.0 * eta + 1103.0) / 3072.0)
        - (15.0 * delta2 * eta / (256.0 * m1_2))
        - (5.0 * delta * (980.0 * eta + 911.0) / (7168.0 * m1_norm))
    )
    e_1 = (
        5.0 * (36.0 * eta * (85568.0 * eta + 23817.0) + 8024297.0) / 9289728.0
        + 5.0 * m1_2 * chi_l * (75.0 * delta2 * chi_l - 112.0 * PI) / (256.0 * eta)
        + 1815.0 * delta * m1_3 * chil2 / (256.0 * eta)
        - 15.0 * PI * delta * m1_norm * chi_l / (16.0 * eta)
        + 1645.0 * m1_4 * chil2 / (192.0 * eta)
        + 15.0 * delta3 * eta2 / (1024.0 * m1_3)
        + 5.0 * delta2 * eta * (784.0 * eta + 323.0) / (28672.0 * m1_2)
        + 5.0 * delta * (504.0 * eta * (7630.0 * eta - 159.0) + 5579177.0) / (21676032.0 * m1_norm)
    )
    e_log = (
        -(5.0 / 48.0) * (7.0 * PI - 3.0 * delta2 * chi_l)
        + 5.0 * delta * (7168.0 * eta + 407.0) * m1_norm * chi_l / (21504.0 * eta)
        + 5.0 * (4072.0 * eta + 599.0) * m1_2 * chi_l / (9216.0 * eta)
        - 5.0 * PI * delta / (16.0 * m1_norm)
    )

    return a_minus3, a_minus2, a_minus1, a_1, a_log, e_minus3, e_minus2, e_minus1, e_1, e_log


def evaluate_angles(
    v: jnp.ndarray,
    v_ref: jnp.ndarray,
    a_minus3: jnp.ndarray,
    a_minus2: jnp.ndarray,
    a_minus1: jnp.ndarray,
    a_1: jnp.ndarray,
    a_log: jnp.ndarray,
    e_minus3: jnp.ndarray,
    e_minus2: jnp.ndarray,
    e_minus1: jnp.ndarray,
    e_1: jnp.ndarray,
    e_log: jnp.ndarray,
    anchor_reference: bool = True,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Evaluate NNLO primitives, optionally subtracting their reference values."""
    v_safe = jnp.maximum(as_array(v, dtype=jnp.float64), 1e-8)
    v_ref_safe = jnp.maximum(as_array(v_ref, dtype=jnp.float64), 1e-8)

    v_inv = 1.0 / v_safe
    v_inv2 = v_inv * v_inv
    v_inv3 = v_inv2 * v_inv
    log_v3 = jnp.log(v_safe * v_safe * v_safe)

    v_ref_inv = 1.0 / v_ref_safe
    v_ref_inv2 = v_ref_inv * v_ref_inv
    v_ref_inv3 = v_ref_inv2 * v_ref_inv
    log_vref3 = jnp.log(v_ref_safe * v_ref_safe * v_ref_safe)

    alpha = a_minus3 * v_inv3 + a_minus2 * v_inv2 + a_minus1 * v_inv + a_1 * v_safe + a_log * log_v3
    alpha_ref = a_minus3 * v_ref_inv3 + a_minus2 * v_ref_inv2 + a_minus1 * v_ref_inv + a_1 * v_ref_safe + a_log * log_vref3

    eps = e_minus3 * v_inv3 + e_minus2 * v_inv2 + e_minus1 * v_inv + e_1 * v_safe + e_log * log_v3
    eps_ref = e_minus3 * v_ref_inv3 + e_minus2 * v_ref_inv2 + e_minus1 * v_ref_inv + e_1 * v_ref_safe + e_log * log_vref3

    if isinstance(anchor_reference, bool):
        if anchor_reference:
            return alpha - alpha_ref, eps - eps_ref
        return alpha, eps
    return (
        jnp.where(anchor_reference, alpha - alpha_ref, alpha),
        jnp.where(anchor_reference, eps - eps_ref, eps),
    )


evaluate_nnlo_angles = evaluate_angles


def orbital_angular_momentum(v: jnp.ndarray, eta: jnp.ndarray) -> jnp.ndarray:
    """2PN non-spinning orbital angular momentum (Kidder 1995, Pratten et al. 2021)."""
    v_safe = jnp.maximum(v, 1e-8)
    x = v_safe * v_safe
    l1 = 1.5 + eta / 6.0
    l2 = (27.0 / 8.0) - (19.0 * eta / 8.0) + (eta * eta / 24.0)
    return (eta / v_safe) * (1.0 + l1 * x + l2 * (x * x))


def pv2_opening_angle(
    orbital_angular_momentum: jnp.ndarray,
    chip: jnp.ndarray,
    m1_norm: jnp.ndarray,
    spin_parallel: jnp.ndarray,
) -> jnp.ndarray:
    """Compute the Pv2 cone angle from the published physical spins.

    Pratten et al. (2021), Eqs. (4.3)--(4.4), use
    ``S_perp = m1**2 * chi_p`` and ``beta = atan2(S_perp, L+S_parallel)``.
    ``A1`` belongs only to the definition of ``chi_p``.
    """
    s_perp = chip * (m1_norm * m1_norm)
    return jnp.arctan2(s_perp, orbital_angular_momentum + spin_parallel)


@jax.tree_util.register_pytree_node_class
class IMRPhenomP:
    """IMRPhenomP precessing frequency-domain phenomenological waveform model in JAX.

    Constructs precessing waveforms by 'twisting up' the underlying aligned-spin
    IMRPhenomD carrier waveform with next-to-next-to-leading order (NNLO) Euler
    angles and a Wigner d-matrix rotation into the observer frame.
    """

    nnlo_angle_coefficients = staticmethod(nnlo_angle_coefficients)
    evaluate_angles = staticmethod(evaluate_angles)
    evaluate_nnlo_angles = staticmethod(evaluate_nnlo_angles)
    orbital_angular_momentum = staticmethod(orbital_angular_momentum)
    pv2_opening_angle = staticmethod(pv2_opening_angle)
    _safe_in_plane_norm = staticmethod(_safe_in_plane_norm)

    def __init__(self) -> None:
        pass

    def tree_flatten(self):
        return ((), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls()

    @classmethod
    def evaluate(
        cls,
        f: jnp.ndarray,
        mass1: Union[float, jnp.ndarray] = 30.0,
        mass2: Union[float, jnp.ndarray] = 30.0,
        spin1x: Union[float, jnp.ndarray] = 0.0,
        spin1y: Union[float, jnp.ndarray] = 0.0,
        spin1z: Union[float, jnp.ndarray] = 0.0,
        spin2x: Union[float, jnp.ndarray] = 0.0,
        spin2y: Union[float, jnp.ndarray] = 0.0,
        spin2z: Union[float, jnp.ndarray] = 0.0,
        distance: Union[float, jnp.ndarray] = 100.0,
        inclination: Union[float, jnp.ndarray] = 0.0,
        phic: Union[float, jnp.ndarray] = 0.0,
        tc: Union[float, jnp.ndarray] = 0.0,
        f_ref: Union[float, jnp.ndarray] = 20.0,
        precession_version: str = "v2",
        phasing_convention: Optional[str] = None,
        reference_frame: Optional[str] = None,
        sample_frequencies: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Generate frequency-domain polarizations (h_plus, h_cross) using IMRPhenomP in JAX.

        Args:
            f: Frequency array in Hz, shape (N_f,) or (B, N_f)
            mass1: Primary mass in solar masses (m1 >= m2)
            mass2: Secondary mass in solar masses
            spin1x, spin1y, spin1z: Dimensionless spin vector of primary
            spin2x, spin2y, spin2z: Dimensionless spin vector of secondary
            distance: Luminosity distance in Mpc (default 100.0)
            inclination: Observer inclination in radians (default 0.0)
            phic: Coalescence phase in radians (default 0.0)
            tc: Coalescence time in seconds (default 0.0)
            f_ref: Reference frequency in Hz (default 20.0)
            precession_version: Legacy label; this implementation uses NNLO angles and 2PN L
            phasing_convention: Phasing convention ('coalescence' or 'reference_frequency')
            reference_frame: Alias for phasing_convention
            sample_frequencies: Alias for f

        Returns:
            (h_plus, h_cross) complex arrays
        """
        if f is None:
            if sample_frequencies is not None:
                f = sample_frequencies
            else:
                raise ValueError("Must provide frequency grid via 'f' or 'sample_frequencies'")

        dtype = jnp.float64
        f_arr = as_array(f, dtype=dtype)

        m1 = as_array(mass1, dtype=dtype)
        m2 = as_array(mass2, dtype=dtype)
        s1x = as_array(spin1x, dtype=dtype)
        s1y = as_array(spin1y, dtype=dtype)
        s1z = as_array(spin1z, dtype=dtype)
        s2x = as_array(spin2x, dtype=dtype)
        s2y = as_array(spin2y, dtype=dtype)
        s2z = as_array(spin2z, dtype=dtype)

        dist = as_array(distance, dtype=dtype)
        inc = as_array(inclination, dtype=dtype)
        phi_c = as_array(phic, dtype=dtype)
        t_c = as_array(tc, dtype=dtype)
        fref = as_array(20.0 if f_ref is None else f_ref, dtype=dtype)

        # Broadcast scalar masses with vector spin/extrinsic inputs first
        (m1, m2, s1x, s1y, s1z, s2x, s2y, s2z,
         dist, inc, phi_c, t_c, fref) = jnp.broadcast_arrays(
            m1, m2, s1x, s1y, s1z, s2x, s2y, s2z,
            dist, inc, phi_c, t_c, fref,
        )

        swap = m1 < m2
        m1, m2 = jnp.where(swap, m2, m1), jnp.where(swap, m1, m2)
        s1x, s2x = jnp.where(swap, s2x, s1x), jnp.where(swap, s1x, s2x)
        s1y, s2y = jnp.where(swap, s2y, s1y), jnp.where(swap, s1y, s2y)
        s1z, s2z = jnp.where(swap, s2z, s1z), jnp.where(swap, s1z, s2z)

        total_mass = m1 + m2
        m_sec = total_mass * MSUN_SEC
        eta = component_masses_to_eta(m1, m2)
        delta = component_masses_to_delta(m1, m2)

        m1_norm = 0.5 * (1.0 + delta)
        m2_norm = 0.5 * (1.0 - delta)

        q = m1 / m2
        a1_coeff = 2.0 + 1.5 / q
        a2_coeff = 2.0 + 1.5 * q

        # Effective precession parameter chi_p (Schmidt et al. 2015, Pratten et al. 2021)
        s1_perp = (m1 * m1) * cls._safe_in_plane_norm(s1x, s1y)
        s2_perp = (m2 * m2) * cls._safe_in_plane_norm(s2x, s2y)
        s_p = jnp.maximum(a1_coeff * s1_perp, a2_coeff * s2_perp)
        chip = s_p / (a1_coeff * m1 * m1)

        # In-plane spin initial orientation angle at f_ref: alpha(f_ref) = phi_s1
        s1_perp_sq = s1x * s1x + s1y * s1y
        s1_has_direction = s1_perp_sq > 1e-12
        s2_has_direction = s2x * s2x + s2y * s2y > 1e-12
        phi_s1_primary = jnp.arctan2(
            jnp.where(s1_has_direction, s1y, jnp.zeros_like(s1y)),
            jnp.where(s1_has_direction, s1x, jnp.ones_like(s1x)),
        )
        phi_s1_secondary = jnp.arctan2(
            jnp.where(s2_has_direction, s2y, jnp.zeros_like(s2y)),
            jnp.where(s2_has_direction, s2x, jnp.ones_like(s2x)),
        )
        phi_s1 = jnp.where(s1_has_direction, phi_s1_primary, phi_s1_secondary)

        # Aligned spin components in coprecessing frame
        chi_l = m1_norm * s1z + m2_norm * s2z

        # Generate face-on co-precessing frame waveform using IMRPhenomD conditioned at f_ref
        h22, _ = IMRPhenomD.evaluate(
            f=f_arr,
            mass1=m1,
            mass2=m2,
            chi1=s1z,
            chi2=s2z,
            distance=dist,
            inclination=jnp.zeros_like(inc),
            phic=phi_c,
            tc=t_c,
            f_ref=fref,
            phasing_convention=phasing_convention,
            reference_frame=reference_frame,
        )

        # Broadcasting setup
        is_batched = total_mass.ndim >= 1
        if f_arr.ndim == 1 and is_batched:
            f_eval = jnp.expand_dims(f_arr, 0)
        else:
            f_eval = f_arr

        def unsqueeze_param(t: jnp.ndarray) -> jnp.ndarray:
            if is_batched and f_eval.ndim > t.ndim:
                return jnp.expand_dims(t, -1)
            return t

        m_sec_u = unsqueeze_param(m_sec)
        eta_u = unsqueeze_param(eta)
        delta_u = unsqueeze_param(delta)
        m1n_u = unsqueeze_param(m1_norm)
        m2n_u = unsqueeze_param(m2_norm)
        chip_u = unsqueeze_param(chip)
        chil_u = unsqueeze_param(chi_l)
        s1z_u = unsqueeze_param(s1z)
        s2z_u = unsqueeze_param(s2z)
        inc_u = unsqueeze_param(inc)
        fref_u = unsqueeze_param(fref)
        phi_s1_u = unsqueeze_param(phi_s1)

        # Precession angle polynomial coefficients
        (
            a_m3, a_m2, a_m1, a_1, a_log,
            e_m3, e_m2, e_m1, e_1, e_log
        ) = cls.nnlo_angle_coefficients(
            eta=eta_u,
            delta=delta_u,
            m1_norm=m1n_u,
            chip=chip_u,
            chi_l=chil_u,
        )

        # Expansion parameter v = (pi * M_sec * f)^(1/3)
        f_eval_safe = jnp.maximum(f_eval, 1e-8)
        fref_u_safe = jnp.maximum(fref_u, 1e-8)
        v = (PI * m_sec_u * f_eval_safe) ** (1.0 / 3.0)
        v_ref = (PI * m_sec_u * fref_u_safe) ** (1.0 / 3.0)

        alpha_dyn, epsilon = cls.evaluate_angles(
            v=v,
            v_ref=v_ref,
            a_minus3=a_m3, a_minus2=a_m2, a_minus1=a_m1, a_1=a_1, a_log=a_log,
            e_minus3=e_m3, e_minus2=e_m2, e_minus1=e_m1, e_1=e_1, e_log=e_log,
        )
        alpha = alpha_dyn - phi_s1_u

        # Orbital angular momentum L(v) & opening angle beta(v)
        ang_l = cls.orbital_angular_momentum(v, eta_u)
        sl_geom = (m1n_u * m1n_u) * s1z_u + (m2n_u * m2n_u) * s2z_u

        beta = cls.pv2_opening_angle(ang_l, chip_u, m1n_u, sl_geom)

        # Wigner d^2 matrix elements d^2_{m, m'}(beta)
        cb = jnp.cos(0.5 * beta)
        sb = jnp.sin(0.5 * beta)
        cb2 = cb * cb
        sb2 = sb * sb
        sqrt6 = math.sqrt(6.0)

        # d^2_{m, -2}(beta)
        d_2_m2 = sb2 * sb2
        d_1_m2 = 2.0 * cb * (sb2 * sb)
        d_0_m2 = sqrt6 * cb2 * sb2
        d_m1_m2 = 2.0 * (cb2 * cb) * sb
        d_m2_m2 = cb2 * cb2

        # d^2_{m, 2}(beta)
        d_2_2 = cb2 * cb2
        d_1_2 = -2.0 * (cb2 * cb) * sb
        d_0_2 = sqrt6 * cb2 * sb2
        d_m1_2 = -2.0 * cb * (sb2 * sb)
        d_m2_2 = sb2 * sb2

        # Basis functions at inclination iota
        ci = jnp.cos(0.5 * inc_u)
        si = jnp.sin(0.5 * inc_u)
        ci2 = ci * ci
        si2 = si * si

        y_2 = si2 * si2
        y_1 = 2.0 * ci * (si2 * si)
        y_0 = sqrt6 * ci2 * si2
        y_m1 = 2.0 * (ci2 * ci) * si
        y_m2 = ci2 * ci2

        m_indices = [2, 1, 0, -1, -2]
        d_m2_list = [d_2_m2, d_1_m2, d_0_m2, d_m1_m2, d_m2_m2]
        d_p2_list = [d_2_2, d_1_2, d_0_2, d_m1_2, d_m2_2]
        y_list = [y_2, y_1, y_0, y_m1, y_m2]

        term_plus = jnp.zeros_like(h22)
        term_cross = jnp.zeros_like(h22)

        for m_val, dm2, dp2, y in zip(m_indices, d_m2_list, d_p2_list, y_list):
            m_alpha = m_val * alpha
            exp_minus_ima = jnp.cos(-m_alpha) + 1j * jnp.sin(-m_alpha)
            exp_plus_ima = jnp.cos(m_alpha) + 1j * jnp.sin(m_alpha)

            a_m_minus2 = exp_minus_ima * dm2 * y
            a_m_plus2_star = exp_plus_ima * dp2 * y

            term_plus = term_plus + (a_m_minus2 + a_m_plus2_star)
            term_cross = term_cross + (a_m_minus2 - a_m_plus2_star)

        exp_minus_2ie = jnp.cos(-2.0 * epsilon) + 1j * jnp.sin(-2.0 * epsilon)

        # Polarizations
        h_plus = exp_minus_2ie * h22 * term_plus
        h_cross = -1j * exp_minus_2ie * h22 * term_cross

        h_plus = jnp.where(f_eval > 0.0, h_plus, jnp.zeros_like(h_plus))
        h_cross = jnp.where(f_eval > 0.0, h_cross, jnp.zeros_like(h_cross))

        return h_plus, h_cross

    def __call__(self, *args, **kwargs):
        return self.evaluate(*args, **kwargs)


IMRPhenomPv2 = IMRPhenomP

__all__ = [
    "IMRPhenomP",
    "IMRPhenomPv2",
    "nnlo_angle_coefficients",
    "evaluate_angles",
    "evaluate_nnlo_angles",
    "orbital_angular_momentum",
    "pv2_opening_angle",
    "_safe_in_plane_norm",
]

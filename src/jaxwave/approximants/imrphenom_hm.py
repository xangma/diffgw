"""IMRPhenomHM Multipolar Higher-Harmonic Frequency-Domain Approximant in JAX.

Clean-room implementation based exclusively on published literature:
- London et al. (2018), Phys. Rev. Lett. 120, 161102 (arXiv:1708.00404)
- Mishra et al. (2016), Phys. Rev. D 94, 084053 (arXiv:1609.00727)
- Berti, Cardoso, Will (2006), Phys. Rev. D 73, 064030 (arXiv:gr-qc/0512160)
- Kidder (2008), Phys. Rev. D 77, 044016 (arXiv:0710.0614)
- Blanchet (2014), Living Rev. Relativity 17, 2 (arXiv:1310.1528)
"""

from typing import Dict, List, Optional, Sequence, Tuple, Union
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
    spin_combinations,
    total_spin_for_remnant,
)
from jaxwave.approximants.imrphenomd import IMRPhenomD
from jaxwave.approximants.imrphenom_hm_qnm_n61 import evaluate_n61_qnm


QNM_FITS: Dict[Tuple[int, int], Dict[str, Tuple[float, float, float]]] = {
    (2, 2): {
        "f": (1.5251, -1.1568, 0.1292),
        "q": (0.7000, 1.4187, -0.4990),
    },
    (2, 1): {
        "f": (0.6000, -0.2339, 0.4175),
        "q": (-0.3000, 2.3561, -0.2277),
    },
    (3, 3): {
        "f": (1.8956, -1.3043, 0.1818),
        "q": (0.9000, 2.3430, -0.4810),
    },
    (3, 2): {
        "f": (1.1481, -0.5552, 0.3002),
        "q": (0.8313, 2.3773, -0.3655),
    },
    (4, 4): {
        "f": (2.3000, -1.5056, 0.2244),
        "q": (1.1929, 3.1191, -0.4825),
    },
    (4, 3): {
        "f": (1.6869, -0.8862, 0.2822),
        "q": (1.4812, 2.8096, -0.4271),
    },
}


def _swsh_projection_coefficients(
    l: int,
    m: int,
    iota: jnp.ndarray,
    phi0: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute polarization projection coefficients d_plus and d_cross for mode (l, m).

    For positive frequencies, the coefficient of the positive-m mode and its
    negative-m partner are combined as
        d_plus  = 0.5 * (Y_{l,m} + (-1)^l * conj(Y_{l,-m}))
        d_cross = -0.5i * (Y_{l,m} - (-1)^l * conj(Y_{l,-m}))
    where Y_{l,m} = {}_{-2}Y_{lm}(iota, phi0).

    Returns:
        (d_plus, d_cross) complex arrays.
    """
    cos_i = jnp.cos(iota)
    sin_i = jnp.sin(iota)

    one_plus_cos = 1.0 + cos_i
    one_minus_cos = 1.0 - cos_i

    # Phase factors exp(i * m * phi0)
    m_phi0 = float(m) * phi0
    exp_pos = jnp.cos(m_phi0) + 1j * jnp.sin(m_phi0)
    exp_neg = jnp.conj(exp_pos)

    if l == 2 and m == 2:
        c_norm = math.sqrt(5.0 / (64.0 * math.pi))
        ylm = c_norm * (one_plus_cos * one_plus_cos) * exp_pos
        yl_negm = c_norm * (one_minus_cos * one_minus_cos) * exp_neg
        sign_l = 1.0
    elif l == 2 and m == 1:
        c_norm = math.sqrt(5.0 / (16.0 * math.pi))
        ylm = c_norm * sin_i * one_plus_cos * exp_pos
        yl_negm = c_norm * sin_i * one_minus_cos * exp_neg
        sign_l = 1.0
    elif l == 3 and m == 3:
        c_norm = math.sqrt(21.0 / (128.0 * math.pi))
        ylm = -c_norm * sin_i * (one_plus_cos * one_plus_cos) * exp_pos
        yl_negm = c_norm * sin_i * (one_minus_cos * one_minus_cos) * exp_neg
        sign_l = -1.0
    elif l == 3 and m == 2:
        c_norm = math.sqrt(7.0 / (4.0 * math.pi)) * 0.25
        ylm = c_norm * (one_plus_cos * one_plus_cos) * (3.0 * cos_i - 2.0) * exp_pos
        yl_negm = c_norm * (one_minus_cos * one_minus_cos) * (3.0 * cos_i + 2.0) * exp_neg
        sign_l = -1.0
    elif l == 4 and m == 4:
        c_norm = math.sqrt(63.0 / (256.0 * math.pi))
        ylm = c_norm * (sin_i * sin_i) * (one_plus_cos * one_plus_cos) * exp_pos
        yl_negm = c_norm * (sin_i * sin_i) * (one_minus_cos * one_minus_cos) * exp_neg
        sign_l = 1.0
    elif l == 4 and m == 3:
        c_norm = 3.0 * math.sqrt(14.0) / (16.0 * math.sqrt(math.pi))
        ylm = c_norm * sin_i * (one_plus_cos * one_plus_cos) * (1.0 - 2.0 * cos_i) * exp_pos
        yl_negm = c_norm * sin_i * (one_minus_cos * one_minus_cos) * (1.0 + 2.0 * cos_i) * exp_neg
        sign_l = 1.0
    else:
        raise ValueError(f"Mode ({l}, {m}) not currently supported.")

    ylm_neg_star = jnp.conj(yl_negm)
    d_plus = 0.5 * (ylm + sign_l * ylm_neg_star)
    diff = ylm - sign_l * ylm_neg_star
    d_cross = -0.5j * diff

    return d_plus, d_cross


def frequency_mapping_amplitude(
    f: jnp.ndarray,
    f0_amp: Union[float, jnp.ndarray],
    f_rd_lm: Union[float, jnp.ndarray],
    f_rd_22: Union[float, jnp.ndarray],
    m: int,
) -> jnp.ndarray:
    """London et al. (2018) piecewise amplitude frequency mapping f_22^A(f)."""
    s2 = (f_rd_22 - (2.0 / float(m)) * f0_amp) / (f_rd_lm - f0_amp)
    reg1 = (2.0 / float(m)) * f
    reg2 = s2 * (f - f0_amp) + (2.0 / float(m)) * f0_amp
    reg3 = f - (f_rd_lm - f_rd_22)
    return jnp.where(f <= f0_amp, reg1, jnp.where(f <= f_rd_lm, reg2, reg3))


def frequency_mapping_phase(
    f: jnp.ndarray,
    f0_phi: Union[float, jnp.ndarray],
    f_rd_lm: Union[float, jnp.ndarray],
    f_rd_22: Union[float, jnp.ndarray],
    m: int,
) -> jnp.ndarray:
    """London et al. (2018) piecewise phase frequency mapping f_22^phi(f)."""
    s2 = (f_rd_22 - (2.0 / float(m)) * f0_phi) / (f_rd_lm - f0_phi)
    s3 = f_rd_22 / f_rd_lm
    reg1 = (2.0 / float(m)) * f
    reg2 = s2 * (f - f0_phi) + (2.0 / float(m)) * f0_phi
    reg3 = s3 * f
    return jnp.where(f <= f0_phi, reg1, jnp.where(f <= f_rd_lm, reg2, reg3))


def h_hat_multipole(
    l: int,
    m: int,
    v_m: jnp.ndarray,
    delta: jnp.ndarray,
    eta: jnp.ndarray,
    chi_s: jnp.ndarray,
    chi_a: jnp.ndarray,
) -> jnp.ndarray:
    """Evaluate leading PN multipole amplitude ratio H_hat_lm(f)."""
    if (l, m) == (2, 2):
        return -jnp.ones_like(v_m)
    elif (l, m) == (3, 3):
        c = -0.75 * math.sqrt(5.0 / 7.0)
        return c * delta * v_m
    elif (l, m) == (4, 4):
        c = -(4.0 / 9.0) * math.sqrt(10.0 / 7.0)
        return c * (1.0 - 3.0 * eta) * (v_m ** 2.0)
    elif (l, m) == (2, 1):
        c = -math.sqrt(2.0) / 3.0
        return c * (delta * v_m - 1.5 * (chi_a + delta * chi_s) * (v_m ** 2.0))
    elif (l, m) == (3, 2):
        c = -(1.0 / 3.0) * math.sqrt(5.0 / 7.0)
        return c * ((1.0 - 3.0 * eta) * (v_m ** 2.0) + 4.0 * eta * chi_s * (v_m ** 3.0))
    elif (l, m) == (4, 3):
        c = -0.75 * math.sqrt(3.0 / 35.0)
        return c * (delta * (1.0 - 2.0 * eta) * (v_m ** 3.0) + 2.5 * eta * (chi_a - delta * chi_s) * (v_m ** 4.0))
    else:
        raise ValueError(f"Mode ({l}, {m}) not supported.")


def _production_hhat(
    l: int,
    m: int,
    v_m: jnp.ndarray,
    delta: jnp.ndarray,
    eta: jnp.ndarray,
    chi_s: jnp.ndarray,
    chi_a: jnp.ndarray,
    *,
    hm_21_pn_variant: str = "binary_compatible",
) -> jnp.ndarray:
    """London Eq. (4) factors with selectable (2,1) PN spin convention."""
    if (l, m) == (2, 2):
        return -jnp.ones_like(v_m)
    if (l, m) == (3, 3):
        return -(3.0 / 4.0) * math.sqrt(5.0 / 7.0) * delta * v_m
    if (l, m) == (4, 4):
        return -(4.0 / 9.0) * math.sqrt(10.0 / 7.0) * (1.0 - 3.0 * eta) * (v_m * v_m)
    if (l, m) == (2, 1):
        c_spa = delta * (
            (113.0 / 24.0 - 19.0 * eta / 6.0) * chi_s
            + (113.0 / 24.0) * delta * chi_a
        )
        c_spa = c_spa - 1.5 * (chi_a + delta * chi_s) * (
            743.0 / 672.0 + 11.0 * eta / 8.0
        )
        c_td = (0.5 - 205.0 * eta / 14.0) * chi_a + delta * (
            0.5 - 33.0 * eta / 14.0
        ) * chi_s
        one = jnp.ones_like(v_m)
        c_tail = delta * (-PI * one + 1j * (-0.5 - 2.0 * math.log(2.0)) * one)
        # The published Mishra Eq. (12b) polynomial is opt-in; the default
        # retains the established binary-compatible HM carrier.
        c = c_tail + c_spa + (c_td if hm_21_pn_variant == "mishra_eq12b" else -c_td)
        return -(math.sqrt(2.0) / 3.0) * (
            delta * v_m
            - 1.5 * (chi_a + delta * chi_s) * (v_m * v_m)
            + delta * (335.0 / 672.0 + 117.0 * eta / 56.0) * (v_m ** 3.0)
            + c * (v_m ** 4.0)
        )
    if (l, m) == (3, 2):
        return -(1.0 / 3.0) * math.sqrt(5.0 / 7.0) * (
            (1.0 - 3.0 * eta) * (v_m * v_m)
        )
    if (l, m) == (4, 3):
        return -(3.0 / 4.0) * math.sqrt(3.0 / 35.0) * (
            delta * (1.0 - 2.0 * eta) * (v_m ** 3.0)
        )
    raise ValueError(f"Mode ({l}, {m}) not supported.")


def _production_eq14_delta_c1(
    mf: jnp.ndarray,
    eta: jnp.ndarray,
    chi_pn: jnp.ndarray,
    hm_fr22: jnp.ndarray,
    hm_flm: jnp.ndarray,
    hm_fd22: jnp.ndarray,
    hm_fdlm: jnp.ndarray,
    alpha4: jnp.ndarray,
    alpha5: jnp.ndarray,
    *,
    base_fr22: Optional[jnp.ndarray] = None,
    base_fd22: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """London Eq. (14) correction with the D carrier QNM contract."""
    fr22 = hm_fr22
    flm = hm_flm
    fd22 = hm_fd22
    fdlm = hm_fdlm
    ratio = fr22 / flm
    damping_ratio = fdlm / fd22
    carrier_fr22 = fr22 if base_fr22 is None else base_fr22
    carrier_fd22 = fd22 if base_fd22 is None else base_fd22
    center = alpha5 * carrier_fr22
    width_lm = carrier_fd22 * ratio * damping_ratio
    join = 0.5 * carrier_fr22

    def raw(x: jnp.ndarray) -> jnp.ndarray:
        return (alpha4 / eta) * (
            ratio * jnp.arctan((x - center) / width_lm)
            - jnp.arctan((x - center) / carrier_fd22)
        )

    offset = join - center
    at_join_slope = (alpha4 / eta) * (
        ratio * width_lm / (width_lm * width_lm + offset * offset)
        - carrier_fd22 / (carrier_fd22 * carrier_fd22 + offset * offset)
    )
    correction = raw(mf) - raw(join) - at_join_slope * (mf - join)
    return jnp.where(mf > join, correction, jnp.zeros_like(correction))


def _production_mapped_phase_c1(
    f: jnp.ndarray,
    f0_phi: jnp.ndarray,
    f_rd_lm: jnp.ndarray,
    f_rd_22: jnp.ndarray,
    m: int,
    ell: int,
    carrier_phase,
) -> jnp.ndarray:
    """Map a positive-D carrier while preserving value and derivative joins."""
    mid_slope = (f_rd_22 - 2.0 * f0_phi / float(m)) / (f_rd_lm - f0_phi)
    g0 = 2.0 * f0_phi / float(m)
    g_mid = mid_slope * (f - f0_phi) + g0
    g_high = (f_rd_22 / f_rd_lm) * f
    g = jnp.where(
        f <= f0_phi,
        2.0 * f / float(m),
        jnp.where(f <= f_rd_lm, g_mid, g_high),
    )
    k_mid = 1.0 / mid_slope
    k_high = f_rd_lm / f_rd_22
    low_delta = math.pi / 2.0 * (3 * ell + ((ell + m) % 2)) - math.pi

    phi_g0 = carrier_phase(g0)
    delta_mid = low_delta + (float(m) / 2.0 - k_mid) * phi_g0
    phi_fr = carrier_phase(f_rd_22)
    delta_high = k_mid * phi_fr + delta_mid - k_high * phi_fr

    phi_g = carrier_phase(g)
    low = (float(m) / 2.0) * phi_g + low_delta
    middle = k_mid * phi_g + delta_mid
    high = k_high * phi_g + delta_high
    return jnp.where(f <= f0_phi, low, jnp.where(f <= f_rd_lm, middle, high))


class IMRPhenomHM:
    """IMRPhenomHM Multipolar Higher-Harmonic Approximant in JAX.

    Supports harmonics (l, m) in {(2, 2), (2, 1), (3, 3), (3, 2), (4, 4), (4, 3)}.
    """

    SUPPORTED_MODES: Tuple[Tuple[int, int], ...] = (
        (2, 2),
        (2, 1),
        (3, 3),
        (3, 2),
        (4, 4),
        (4, 3),
    )

    def __init__(
        self,
        modes: Optional[Sequence[Tuple[int, int]]] = None,
        base_22=None,
    ):
        if modes is not None:
            normalized_modes = tuple((int(l), int(m)) for l, m in modes)
            for lm in normalized_modes:
                if lm not in self.SUPPORTED_MODES:
                    raise ValueError(f"Mode {lm} not supported. Supported: {self.SUPPORTED_MODES}")
            self.modes = normalized_modes
        else:
            self.modes = self.SUPPORTED_MODES
        self.base_22 = base_22 if base_22 is not None else IMRPhenomD

    def __call__(self, *args, **kwargs):
        kwargs.setdefault("modes", self.modes)
        kwargs.setdefault("base_22", self.base_22)
        return self.evaluate(*args, **kwargs)

    @staticmethod
    def mode_qnm_frequencies(
        l: int,
        m: int,
        mass1: Union[float, jnp.ndarray],
        mass2: Union[float, jnp.ndarray],
        chi1: Union[float, jnp.ndarray] = 0.0,
        chi2: Union[float, jnp.ndarray] = 0.0,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Compute fundamental QNM frequency f_RD and damping width f_damp for mode (l, m)."""
        dtype = jnp.float64
        m1 = as_array(mass1, dtype=dtype)
        m2 = as_array(mass2, dtype=dtype)
        s1 = as_array(chi1, dtype=dtype)
        s2 = as_array(chi2, dtype=dtype)

        total_mass = m1 + m2
        m_final, a_final, _, _ = IMRPhenomD.remnant_properties(m1, m2, s1, s2)
        omega_real_final, omega_imag_final = evaluate_n61_qnm((l, m), a_final)
        m_ratio = total_mass / m_final
        scale = m_ratio / (2.0 * PI)
        return scale * omega_real_final, scale * omega_imag_final

    @classmethod
    def evaluate(
        cls,
        f: jnp.ndarray,
        mass1: Union[float, jnp.ndarray] = 30.0,
        mass2: Union[float, jnp.ndarray] = 30.0,
        chi1: Union[float, jnp.ndarray] = 0.0,
        chi2: Union[float, jnp.ndarray] = 0.0,
        distance: Union[float, jnp.ndarray] = 100.0,
        inclination: Union[float, jnp.ndarray] = 0.0,
        phic: Union[float, jnp.ndarray] = 0.0,
        tc: Union[float, jnp.ndarray] = 0.0,
        f_ref: Optional[Union[float, jnp.ndarray]] = None,
        f_lower: Optional[float] = None,
        phi0: Union[float, jnp.ndarray] = 0.0,
        modes: Optional[Sequence[Tuple[int, int]]] = None,
        return_modes: bool = False,
        return_unprojected_modes: bool = False,
        phasing_convention: Optional[str] = None,
        reference_frame: Optional[str] = None,
        sample_frequencies: Optional[jnp.ndarray] = None,
        base_22=None,
        hm_mapping_transitions: str = "phenomd_runtime",
        hm_21_pn_variant: str = "binary_compatible",
    ) -> Union[
        Tuple[jnp.ndarray, jnp.ndarray],
        Tuple[jnp.ndarray, jnp.ndarray, Dict[Tuple[int, int], Union[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]]],
    ]:
        """Generate frequency-domain polarizations (h_plus, h_cross) including higher multipoles in JAX."""
        if f is None:
            if sample_frequencies is not None:
                f = sample_frequencies
            else:
                raise ValueError("Must provide frequency grid via 'f' or 'sample_frequencies'")
        if hm_mapping_transitions not in ("phenomd_runtime", "london_2018"):
            raise ValueError(f"Unknown HM mapping transitions: {hm_mapping_transitions}")
        if hm_21_pn_variant not in ("binary_compatible", "mishra_eq12b"):
            raise ValueError(f"Unknown HM (2,1) PN variant: {hm_21_pn_variant}")
        if hm_mapping_transitions == "london_2018" and base_22 not in (None, IMRPhenomD):
            raise ValueError("london_2018 HM mapping transitions require the D carrier")
        if hm_21_pn_variant == "mishra_eq12b" and base_22 not in (None, IMRPhenomD):
            raise ValueError("mishra_eq12b HM (2,1) PN variant requires the D carrier")

        dtype = jnp.float64
        f_arr = as_array(f, dtype=dtype)

        m1 = as_array(mass1, dtype=dtype)
        m2 = as_array(mass2, dtype=dtype)
        s1 = as_array(chi1, dtype=dtype)
        s2 = as_array(chi2, dtype=dtype)
        dist = as_array(distance, dtype=dtype)
        inc = as_array(inclination, dtype=dtype)
        phi_c = as_array(phic, dtype=dtype)
        t_c = as_array(tc, dtype=dtype)
        phi_0 = as_array(phi0, dtype=dtype)
        fref_input = None if f_ref is None else as_array(f_ref, dtype=dtype)

        active_modes = (
            tuple((int(l), int(m)) for l, m in modes) if modes is not None else cls.SUPPORTED_MODES
        )
        for lm in active_modes:
            if lm not in cls.SUPPORTED_MODES:
                raise ValueError(f"Mode {lm} not supported. Supported: {cls.SUPPORTED_MODES}")

        # Broadcast batch shape
        param_shapes = [m1.shape, m2.shape, s1.shape, s2.shape, dist.shape, inc.shape, phi_c.shape, t_c.shape, phi_0.shape]
        if fref_input is not None:
            param_shapes.append(fref_input.shape)
        if f_arr.ndim > 1:
            param_shapes.append(f_arr.shape[:-1])
        batch_shape = jnp.broadcast_shapes(*param_shapes)

        is_batched = len(batch_shape) > 0
        has_frequency_axis = f_arr.ndim > 0

        def broadcast_batch(value: jnp.ndarray) -> jnp.ndarray:
            return jnp.broadcast_to(value, batch_shape)

        def expand_param(value: jnp.ndarray) -> jnp.ndarray:
            value = broadcast_batch(value)
            return jnp.expand_dims(value, -1) if is_batched and has_frequency_axis else value

        if is_batched:
            if f_arr.ndim == 0:
                f_eval = jnp.broadcast_to(f_arr, batch_shape)
            elif f_arr.ndim == 1:
                f_eval = jnp.broadcast_to(
                    jnp.reshape(f_arr, (1,) * len(batch_shape) + (f_arr.shape[0],)),
                    batch_shape + (f_arr.shape[0],),
                )
            else:
                f_eval = jnp.broadcast_to(f_arr, batch_shape + (f_arr.shape[-1],))
        else:
            f_eval = f_arr

        m1_batch = broadcast_batch(m1)
        m2_batch = broadcast_batch(m2)
        s1_batch = broadcast_batch(s1)
        s2_batch = broadcast_batch(s2)
        dist_batch = broadcast_batch(dist)
        fref_batch = None if fref_input is None else broadcast_batch(fref_input)
        fref_eval = None if fref_batch is None else expand_param(fref_batch)

        total_mass = m1_batch + m2_batch
        eta = component_masses_to_eta(m1_batch, m2_batch)
        delta = component_masses_to_delta(m1_batch, m2_batch)
        chi_s = 0.5 * (s1_batch + s2_batch)
        chi_a = 0.5 * (s1_batch - s2_batch)
        _, _, chi_pn, _ = spin_combinations(m1_batch, m2_batch, s1_batch, s2_batch)
        m_sec = total_mass * MSUN_SEC

        m_sec_eval = expand_param(m_sec)
        m1_eval = expand_param(m1_batch)
        m2_eval = expand_param(m2_batch)
        s1_eval = expand_param(s1_batch)
        s2_eval = expand_param(s2_batch)
        dist_eval = expand_param(dist_batch)
        mf_eval = f_eval * m_sec_eval
        eta_eval = expand_param(eta)
        delta_eval = expand_param(delta)
        chi_s_eval = expand_param(chi_s)
        chi_a_eval = expand_param(chi_a)
        chi_pn_eval = expand_param(chi_pn)
        inc_eval = expand_param(inc)
        phi0_eval = expand_param(phi_0)
        phic_eval = expand_param(phi_c)
        tc_eval = expand_param(t_c)

        carrier_cls = base_22 if base_22 is not None else IMRPhenomD

        def _base_call(
            geom_frequency: jnp.ndarray,
            *,
            reference_frequency: Optional[Union[float, jnp.ndarray]],
            need_phase: bool,
        ):
            frequency_hz = jnp.maximum(geom_frequency / m_sec_eval, 1.0e-8)
            carrier_frequency = frequency_hz
            if is_batched and not has_frequency_axis:
                carrier_frequency = jnp.expand_dims(carrier_frequency, -1)

            res = carrier_cls.evaluate(
                f=carrier_frequency,
                mass1=m1_eval,
                mass2=m2_eval,
                chi1=s1_eval,
                chi2=s2_eval,
                distance=dist_eval,
                inclination=0.0,
                phic=0.0,
                tc=0.0,
                f_ref=reference_frequency,
                phasing_convention=phasing_convention,
                reference_frame=reference_frame,
                return_phase=need_phase,
            )
            if need_phase:
                hp_raw, hc_raw, phase_raw = res
            else:
                hp_raw, hc_raw = res
                phase_raw = None

            if hp_raw.shape != geom_frequency.shape and hp_raw.size == geom_frequency.size:
                hp_raw = jnp.reshape(hp_raw, geom_frequency.shape)
            if hc_raw.shape != geom_frequency.shape and hc_raw.size == geom_frequency.size:
                hc_raw = jnp.reshape(hc_raw, geom_frequency.shape)
            if phase_raw is not None and phase_raw.shape != geom_frequency.shape and phase_raw.size == geom_frequency.size:
                phase_raw = jnp.reshape(phase_raw, geom_frequency.shape)

            return hp_raw, hc_raw, phase_raw

        def amplitude_22(geom_frequency: jnp.ndarray) -> jnp.ndarray:
            hp_raw, _, _ = _base_call(
                geom_frequency,
                reference_frequency=None,
                need_phase=False,
            )
            c_norm = math.sqrt(5.0 / (16.0 * math.pi))
            return jnp.abs(hp_raw / c_norm)

        def carrier_positive(geom_frequency: jnp.ndarray) -> jnp.ndarray:
            _, _, phase_raw = _base_call(
                geom_frequency,
                reference_frequency=fref_eval,
                need_phase=True,
            )
            return phase_raw

        m_final, a_final, base_fr22, base_fd22 = IMRPhenomD.remnant_properties(m1_batch, m2_batch, s1_batch, s2_batch)
        hm_qnm: Dict[Tuple[int, int], Tuple[jnp.ndarray, jnp.ndarray]] = {
            lm: cls.mode_qnm_frequencies(lm[0], lm[1], m1_batch, m2_batch, s1_batch, s2_batch)
            for lm in active_modes
        }
        if (2, 2) not in hm_qnm:
            hm_qnm[(2, 2)] = cls.mode_qnm_frequencies(2, 2, m1_batch, m2_batch, s1_batch, s2_batch)

        hm_fr22 = expand_param(hm_qnm[(2, 2)][0])
        hm_fd22 = expand_param(hm_qnm[(2, 2)][1])
        base_fr22_eval = expand_param(base_fr22)
        base_fd22_eval = expand_param(base_fd22)
        alpha4 = expand_param(IMRPhenomD.eval_table_v("alpha4", eta, chi_pn))
        alpha5 = expand_param(IMRPhenomD.eval_table_v("alpha5", eta, chi_pn))

        if f_ref is None:
            reference_gauge_base = jnp.zeros_like(f_eval)
        else:
            slope_peak = expand_param(IMRPhenomD.compute_slope_peak(m1_batch, m2_batch, s1_batch, s2_batch))
            reference_gauge_base = slope_peak * m_sec_eval * fref_eval

        h_plus_total = jnp.zeros_like(f_eval, dtype=jnp.complex128)
        h_cross_total = jnp.zeros_like(h_plus_total)
        mode_dict = {}

        for l, m in active_modes:
            hm_fr_lm, hm_fd_lm = hm_qnm[(l, m)]
            hm_fr_lm = expand_param(hm_fr_lm)
            hm_fd_lm = expand_param(hm_fd_lm)
            if hm_mapping_transitions == "london_2018":
                f0_amp = 0.018 * hm_fr_lm / hm_fr22
                f0_phi = 0.014 * hm_fr_lm / hm_fr22
            else:
                f0_amp = 0.014 * hm_fr_lm / hm_fr22
                f0_phi = 0.018 * hm_fr_lm / hm_fr22
            f_map_amp = frequency_mapping_amplitude(
                mf_eval, f0_amp, hm_fr_lm, hm_fr22, m
            )
            f_map_amp_safe = jnp.maximum(f_map_amp, 1.0e-8)
            amp_22_mapped = amplitude_22(f_map_amp_safe)

            if (l, m) == (2, 2):
                ratio = jnp.ones_like(f_eval)
                mode_factor = jnp.ones_like(ratio)
            else:
                x_safe = jnp.maximum(mf_eval, 1.0e-8)
                ga_safe = f_map_amp_safe
                v_ga = (2.0 * PI * ga_safe / float(m)) ** (1.0 / 3.0)
                v_x = (2.0 * PI * x_safe / float(m)) ** (1.0 / 3.0)
                hhat_ga = _production_hhat(
                    l, m, v_ga, delta_eval, eta_eval, chi_s_eval, chi_a_eval,
                    hm_21_pn_variant=hm_21_pn_variant,
                )
                hhat_x = _production_hhat(
                    l, m, v_x, delta_eval, eta_eval, chi_s_eval, chi_a_eval,
                    hm_21_pn_variant=hm_21_pn_variant,
                )
                hlm_ga = hhat_ga * (v_ga ** -3.5)
                hlm_x = hhat_x * (v_x ** -3.5)
                h22_ga = -((2.0 * PI * ga_safe / 2.0) ** (1.0 / 3.0)) ** -3.5
                mode_at_22_x = 2.0 * x_safe / float(m)
                v_22_x = (2.0 * PI * mode_at_22_x / float(m)) ** (1.0 / 3.0)
                h22_at_x = _production_hhat(
                    l, m, v_22_x, delta_eval, eta_eval, chi_s_eval, chi_a_eval,
                    hm_21_pn_variant=hm_21_pn_variant,
                ) * (v_22_x ** -3.5)
                tiny = jnp.finfo(dtype).tiny
                nonzero = jnp.abs(h22_at_x) > tiny
                safe_h22_at_x = jnp.where(nonzero, h22_at_x, 1.0)
                ratio_raw = jnp.abs((hlm_ga / h22_ga) * (hlm_x / safe_h22_at_x))
                ratio = jnp.where(nonzero, ratio_raw, jnp.zeros_like(ratio_raw))
                if m % 2:
                    sign_source = jnp.where(
                        jnp.abs(delta_eval) > tiny, delta_eval, chi_a_eval
                    )
                    mode_factor = jnp.sign(sign_source)
                    mode_factor = jnp.where(
                        jnp.abs(hhat_ga) > tiny,
                        mode_factor,
                        jnp.zeros_like(mode_factor),
                    )
                else:
                    mode_factor = jnp.ones_like(ratio)

            def carrier_lm(y: jnp.ndarray) -> jnp.ndarray:
                z = carrier_positive(y)
                if (l, m) == (2, 2):
                    return z
                return z + _production_eq14_delta_c1(
                    y,
                    eta_eval,
                    chi_pn_eval,
                    hm_fr22,
                    hm_fr_lm,
                    hm_fd22,
                    hm_fd_lm,
                    alpha4,
                    alpha5,
                    base_fr22=base_fr22_eval,
                    base_fd22=base_fd22_eval,
                )

            mapped_phase = _production_mapped_phase_c1(
                mf_eval,
                f0_phi,
                hm_fr_lm,
                hm_fr22,
                m,
                l,
                carrier_lm,
            )

            reference_gauge = (float(m) / 2.0 - 1.0) * reference_gauge_base
            physical_phase = -mapped_phase + reference_gauge + float(m) * phic_eval
            phase_factor = jnp.cos(physical_phase) + 1j * jnp.sin(physical_phase)
            if (l, m) == (2, 2):
                mode_factor = jnp.ones_like(ratio)
            h_mode = amp_22_mapped * ratio * mode_factor * phase_factor
            h_mode = jnp.where(f_eval > 0.0, h_mode, jnp.zeros_like(h_mode))

            d_plus, d_cross = _swsh_projection_coefficients(l, m, inc_eval, phi0_eval)
            hp_lm = d_plus * h_mode
            hc_lm = d_cross * h_mode
            h_plus_total = h_plus_total + hp_lm
            h_cross_total = h_cross_total + hc_lm
            if return_modes:
                mode_dict[(l, m)] = (hp_lm, hc_lm)
            elif return_unprojected_modes:
                mode_dict[(l, m)] = h_mode

        angle = -2.0 * PI * f_eval * tc_eval
        shift = jnp.cos(angle) + 1j * jnp.sin(angle)
        h_plus_total = h_plus_total * shift
        h_cross_total = h_cross_total * shift
        if return_modes or return_unprojected_modes:
            for lm, value in mode_dict.items():
                if isinstance(value, tuple):
                    mode_dict[lm] = (value[0] * shift, value[1] * shift)
                else:
                    mode_dict[lm] = value * shift

        valid = f_eval > 0.0
        if f_lower is not None:
            valid = valid & (f_eval >= float(f_lower))
        h_plus_total = jnp.where(valid, h_plus_total, jnp.zeros_like(h_plus_total))
        h_cross_total = jnp.where(valid, h_cross_total, jnp.zeros_like(h_cross_total))

        if return_modes or return_unprojected_modes:
            return h_plus_total, h_cross_total, mode_dict
        return h_plus_total, h_cross_total


__all__ = [
    "IMRPhenomHM",
    "frequency_mapping_amplitude",
    "frequency_mapping_phase",
    "h_hat_multipole",
    "_swsh_projection_coefficients",
    "QNM_FITS",
]

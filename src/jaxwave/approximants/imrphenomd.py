"""IMRPhenomD Frequency-Domain Aligned-Spin Gravitational Waveform Approximant in JAX.

Waveform equations based on published literature:
- Husa et al. (2016), Phys. Rev. D 93, 044006 (arXiv:1508.07250) [Paper I]
- Khan et al. (2016), Phys. Rev. D 93, 044007 (arXiv:1508.07253) [Paper II]
"""

from typing import Optional, Tuple, Union
import math
import jax.numpy as jnp
from diffgw_core.constants import (
    C_SI,
    MSUN_SEC,
    MPC_SI,
    PI,
)
from diffgw_core.fits import (
    TABLE_V_UNROUNDED_COEFFS,
    FINAL_SPIN_ETA_POLY,
    FINAL_SPIN_S_COEFFS,
    ERAD_ETA_NUMERATOR,
    ERAD_S_NUMERATOR,
    ERAD_S_DENOMINATOR,
)
from diffgw_core.qnm_data import KNOT_SPINS, ROUNDED_CYCLIC_ORDINATES
from jaxwave.approximants.kerr_qnm_interp import _evaluate_kerr_qnm_220_at_spin
from jaxwave.conversions import (
    as_array,
    component_masses_to_eta,
    component_masses_to_delta,
    spin_combinations,
    total_spin_for_remnant,
)
from jaxwave.approximants.taylorf2 import TaylorF2

TABLE_V_COEFFS = TABLE_V_UNROUNDED_COEFFS

# Precompute natural cubic spline coefficients for Kerr QNM 220
def _compute_natural_cubic_coeffs(x_knots, y_knots):
    n = len(x_knots)
    h = [x_knots[i + 1] - x_knots[i] for i in range(n - 1)]
    moments = [[0.0] * n for _ in range(2)]
    for channel in range(2):
        lower = [h[i - 1] for i in range(1, n - 1)]
        diagonal = [2.0 * (h[i - 1] + h[i]) for i in range(1, n - 1)]
        upper = [h[i] for i in range(1, n - 1)]
        rhs = [
            6.0
            * (
                (y_knots[i + 1][channel] - y_knots[i][channel]) / h[i]
                - (y_knots[i][channel] - y_knots[i - 1][channel]) / h[i - 1]
            )
            for i in range(1, n - 1)
        ]
        # Thomas algorithm
        for i in range(1, len(diagonal)):
            factor = lower[i] / diagonal[i - 1]
            diagonal[i] -= factor * upper[i - 1]
            rhs[i] -= factor * rhs[i - 1]
        m = [0.0] * len(diagonal)
        m[-1] = rhs[-1] / diagonal[-1]
        for i in range(len(diagonal) - 2, -1, -1):
            m[i] = (rhs[i] - upper[i] * m[i + 1]) / diagonal[i]
        moments[channel] = [0.0, *m, 0.0]

    coeffs = []
    for i in range(n - 1):
        row = []
        for stage in range(4):
            stage_vals = []
            for channel in range(2):
                if stage == 0:
                    stage_vals.append(y_knots[i][channel])
                elif stage == 1:
                    stage_vals.append(
                        (y_knots[i + 1][channel] - y_knots[i][channel]) / h[i]
                        - (2.0 * moments[channel][i] + moments[channel][i + 1]) * h[i] / 6.0
                    )
                elif stage == 2:
                    stage_vals.append(moments[channel][i] / 2.0)
                else:
                    stage_vals.append(
                        (moments[channel][i + 1] - moments[channel][i]) / (6.0 * h[i])
                    )
            row.append(tuple(stage_vals))
        coeffs.append(tuple(row))
    return coeffs

_QNM_KNOTS = jnp.array(KNOT_SPINS, dtype=jnp.float64)
_QNM_COEFFS = jnp.array(_compute_natural_cubic_coeffs(KNOT_SPINS, ROUNDED_CYCLIC_ORDINATES), dtype=jnp.float64)


def evaluate_d_qnm_220(final_spin: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Evaluate cyclic 220 QNMs with the D rounded/interior and generated/endpoint policy."""
    spin = jnp.asarray(final_spin, dtype=jnp.float64)
    interval = jnp.clip(jnp.searchsorted(_QNM_KNOTS, spin, side="right") - 1, 0, len(_QNM_KNOTS) - 2)
    dx = spin - _QNM_KNOTS[interval]
    selected = _QNM_COEFFS[interval]
    val = selected[..., 3, :] * jnp.expand_dims(dx, -1) + selected[..., 2, :]
    val = val * jnp.expand_dims(dx, -1) + selected[..., 1, :]
    val = val * jnp.expand_dims(dx, -1) + selected[..., 0, :]
    compatibility = jnp.abs(spin) <= 0.994
    # The generated splines are defined through |a|=.99999; the remnant fit
    # itself is capped at |a|=.999.  Clip before evaluating both JAX branches.
    generated_spin = jnp.clip(spin, -0.99999, 0.99999)
    generated = _evaluate_kerr_qnm_220_at_spin(generated_spin)
    scale = 1.0 / (2.0 * PI)
    return tuple(jnp.where(compatibility, rounded, value * scale)
                 for rounded, value in zip((val[..., 0], val[..., 1]), generated))


class IMRPhenomD:
    """IMRPhenomD Frequency-Domain Aligned-Spin Waveform Model in JAX."""

    definition_id = "IMRPhenomD"

    @staticmethod
    def amplitude_peak_frequency(f_rd, f_damp, gamma2, gamma3):
        """Evaluate Khan et al. Eq. (20) with gradient-safe continuation."""
        discriminant = 1.0 - gamma2 * gamma2
        has_root = discriminant > 0.0
        safe_disc = jnp.where(has_root, discriminant, 1.0)
        root = jnp.sqrt(safe_disc)
        safe_gamma2 = jnp.where(has_root, 1.0, gamma2)
        offset = jnp.where(has_root, -gamma2 / (1.0 + root), -1.0 / safe_gamma2)
        return jnp.abs(f_rd + f_damp * gamma3 * offset)

    @staticmethod
    def eval_table_v(name: str, eta: jnp.ndarray, chi_pn: jnp.ndarray) -> jnp.ndarray:
        """Evaluate Table V polynomial fit."""
        coeffs = TABLE_V_COEFFS[name]
        c = [jnp.asarray(val, dtype=eta.dtype) for val in coeffs]

        eta2 = eta * eta
        x = chi_pn - 1.0
        x2 = x * x
        x3 = x2 * x

        poly = (
            c[0] + c[1] * eta
            + x * (c[2] + c[3] * eta + c[4] * eta2)
            + x2 * (c[5] + c[6] * eta + c[7] * eta2)
            + x3 * (c[8] + c[9] * eta + c[10] * eta2)
        )
        return poly

    @staticmethod
    def remnant_properties(
        mass1: jnp.ndarray,
        mass2: jnp.ndarray,
        chi1: jnp.ndarray,
        chi2: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Compute final mass, final spin, QNM frequency, and damping width."""
        total_mass = mass1 + mass2
        eta = component_masses_to_eta(mass1, mass2)
        eta2 = eta * eta
        eta3 = eta2 * eta
        eta4 = eta3 * eta

        s, s_hat = total_spin_for_remnant(mass1, mass2, chi1, chi2)
        s2 = s * s
        s3 = s2 * s
        s4 = s3 * s

        p_eta = FINAL_SPIN_ETA_POLY
        s_c = FINAL_SPIN_S_COEFFS
        a_final = (
            s + p_eta[0] * eta + p_eta[1] * eta2 + p_eta[2] * eta3 + p_eta[3] * eta4
            + (s_c[0][0] * s + s_c[1][0] * s2 + s_c[2][0] * s3 + s_c[3][0] * s4) * eta
            + (s_c[0][1] * s + s_c[1][1] * s2 + s_c[2][1] * s3 + s_c[3][1] * s4) * eta2
        )
        a_final = jnp.clip(a_final, -0.999, 0.999)

        e_eta = ERAD_ETA_NUMERATOR
        erad_ns = e_eta[0] * eta + e_eta[1] * eta2 + e_eta[2] * eta3 + e_eta[3] * eta4

        e_num = ERAD_S_NUMERATOR
        e_den = ERAD_S_DENOMINATOR
        num = 1.0 + s_hat * (e_num[0] + e_num[1] * eta + e_num[2] * eta2)
        den = 1.0 + s_hat * (e_den[0] + e_den[1] * eta + e_den[2] * eta2)
        erad_over_m = erad_ns * (num / den)

        m_final = total_mass * (1.0 - erad_over_m)

        f_rd_final, f_damp_final = evaluate_d_qnm_220(a_final)
        m_ratio = total_mass / m_final
        f_rd = m_ratio * f_rd_final
        f_damp = m_ratio * f_damp_final

        return m_final, a_final, f_rd, f_damp

    @staticmethod
    def pn_amplitude_coefficients(eta, delta, chi_s, chi_a):
        """Compute PN amplitude coefficients A_0 through A_6."""
        eta2 = eta * eta
        eta3 = eta2 * eta
        chi_s2 = chi_s * chi_s
        chi_a2 = chi_a * chi_a

        a0 = jnp.ones_like(eta)
        a1 = jnp.zeros_like(eta)
        a2 = -(323.0 / 224.0) + (451.0 / 168.0) * eta
        a3 = (27.0 / 8.0) * delta * chi_a + ((27.0 / 8.0) - (11.0 / 6.0) * eta) * chi_s
        a4 = (
            -(27312085.0 / 8128512.0)
            - (1975055.0 / 338688.0) * eta
            + (105271.0 / 24192.0) * eta2
            + (-(81.0 / 32.0) + 8.0 * eta) * chi_a2
            - (81.0 / 16.0) * delta * chi_a * chi_s
            + (-(81.0 / 32.0) + (17.0 / 8.0) * eta) * chi_s2
        )
        a5 = (
            -(85.0 * PI / 64.0) + (85.0 * PI / 16.0) * eta
            + delta * ((285197.0 / 16128.0) - (1579.0 / 4032.0) * eta) * chi_a
            + ((285197.0 / 16128.0) - (15317.0 / 672.0) * eta - (2227.0 / 1008.0) * eta2) * chi_s
        )
        a6 = (
            -(177520268561.0 / 8583708672.0)
            + ((545384828789.0 / 5007163392.0) - (205.0 * PI * PI / 48.0)) * eta
            - (3248849057.0 / 178827264.0) * eta2
            + (34473079.0 / 6386688.0) * eta3
            + ((1614569.0 / 64512.0) - (1873643.0 / 16128.0) * eta + (2167.0 / 42.0) * eta2) * chi_a2
            + ((31.0 * PI / 12.0) - (7.0 * PI / 3.0) * eta) * chi_s
            + ((1614569.0 / 64512.0) - (61391.0 / 1344.0) * eta + (57451.0 / 4032.0) * eta2) * chi_s2
            + delta * chi_a * ((31.0 * PI / 12.0) + ((1614569.0 / 32256.0) - (165961.0 / 2688.0) * eta) * chi_s)
        )
        return a0, a1, a2, a3, a4, a5, a6

    @staticmethod
    def intermediate_amplitude_coeffs(f1, f3, v1, v2, v3, d1, d3):
        """Solve 5x5 collocation matrix for intermediate amplitude coefficients."""
        f2 = 0.5 * (f1 + f3)
        ones = jnp.ones_like(f3)
        zeros = jnp.zeros_like(f3)

        f1_t = f1 * ones
        f2_t = f2
        f3_t = f3

        row0 = jnp.stack([ones, f1_t, f1_t**2, f1_t**3, f1_t**4], axis=-1)
        row1 = jnp.stack([ones, f2_t, f2_t**2, f2_t**3, f2_t**4], axis=-1)
        row2 = jnp.stack([ones, f3_t, f3_t**2, f3_t**3, f3_t**4], axis=-1)
        row3 = jnp.stack([zeros, ones, 2.0 * f1_t, 3.0 * f1_t**2, 4.0 * f1_t**3], axis=-1)
        row4 = jnp.stack([zeros, ones, 2.0 * f3_t, 3.0 * f3_t**2, 4.0 * f3_t**3], axis=-1)

        mat = jnp.stack([row0, row1, row2, row3, row4], axis=-2)
        rhs = jnp.stack([v1 * ones, v2 * ones, v3, d1 * ones, d3], axis=-1)

        sol = jnp.linalg.solve(mat, rhs[..., None]).squeeze(-1)
        return sol[..., 0], sol[..., 1], sol[..., 2], sol[..., 3], sol[..., 4]

    @staticmethod
    def compute_slope_peak(
        mass1: Union[float, jnp.ndarray],
        mass2: Union[float, jnp.ndarray],
        chi1: Union[float, jnp.ndarray] = 0.0,
        chi2: Union[float, jnp.ndarray] = 0.0,
        *,
        remnant_provider=None,
    ) -> jnp.ndarray:
        """Paper merger-ringdown phase derivative dPsi/d(Mf) evaluated at f_peak."""
        dtype = jnp.float64
        m1 = as_array(mass1, dtype=dtype)
        m2 = as_array(mass2, dtype=dtype)
        s1 = as_array(chi1, dtype=dtype)
        s2 = as_array(chi2, dtype=dtype)

        eta = component_masses_to_eta(m1, m2)
        _, _, chi_pn, _ = spin_combinations(m1, m2, s1, s2)

        gamma2 = IMRPhenomD.eval_table_v("gamma2", eta, chi_pn)
        gamma3 = IMRPhenomD.eval_table_v("gamma3", eta, chi_pn)
        alpha1_raw = IMRPhenomD.eval_table_v("alpha1", eta, chi_pn)
        alpha2 = IMRPhenomD.eval_table_v("alpha2", eta, chi_pn)
        alpha3 = IMRPhenomD.eval_table_v("alpha3", eta, chi_pn)
        alpha4 = IMRPhenomD.eval_table_v("alpha4", eta, chi_pn)
        alpha5 = IMRPhenomD.eval_table_v("alpha5", eta, chi_pn)

        provider = IMRPhenomD.remnant_properties if remnant_provider is None else remnant_provider
        m_final, a_final, f_rd, f_damp = provider(m1, m2, s1, s2)
        f_peak = IMRPhenomD.amplitude_peak_frequency(f_rd, f_damp, gamma2, gamma3)

        atan_lorentz_peak = (alpha4 * f_damp) / (f_damp * f_damp + (f_peak - alpha5 * f_rd) ** 2)
        slope_peak = (
            alpha1_raw
            + alpha2 * (f_peak ** -2.0)
            + alpha3 * (f_peak ** -0.25)
            + atan_lorentz_peak
        ) / eta
        return slope_peak

    @staticmethod
    def evaluate(
        f: jnp.ndarray,
        mass1: Union[float, jnp.ndarray],
        mass2: Union[float, jnp.ndarray],
        chi1: Union[float, jnp.ndarray] = 0.0,
        chi2: Union[float, jnp.ndarray] = 0.0,
        distance: Union[float, jnp.ndarray] = 100.0,
        inclination: Union[float, jnp.ndarray] = 0.0,
        phic: Union[float, jnp.ndarray] = 0.0,
        tc: Union[float, jnp.ndarray] = 0.0,
        f_ref: Optional[Union[float, jnp.ndarray]] = None,
        phasing_convention: Optional[str] = None,
        reference_frame: Optional[str] = None,
        remnant_provider=None,
        return_phase: bool = False,
    ) -> Union[Tuple[jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
        """Generate frequency-domain polarizations (h_plus, h_cross) using IMRPhenomD in JAX."""
        dtype = jnp.float64
        m1 = as_array(mass1, dtype=dtype)
        m2 = as_array(mass2, dtype=dtype)
        s1 = as_array(chi1, dtype=dtype)
        s2 = as_array(chi2, dtype=dtype)
        dist = as_array(distance, dtype=dtype)
        inc = as_array(inclination, dtype=dtype)
        phi_c = as_array(phic, dtype=dtype)
        t_c = as_array(tc, dtype=dtype)

        total_mass = m1 + m2
        eta = component_masses_to_eta(m1, m2)
        delta = component_masses_to_delta(m1, m2)
        chi_s, chi_a, chi_pn, _ = spin_combinations(m1, m2, s1, s2)

        # Table V parameters
        rho1 = IMRPhenomD.eval_table_v("rho1", eta, chi_pn)
        rho2 = IMRPhenomD.eval_table_v("rho2", eta, chi_pn)
        rho3 = IMRPhenomD.eval_table_v("rho3", eta, chi_pn)
        v2 = IMRPhenomD.eval_table_v("v2", eta, chi_pn)
        gamma1 = IMRPhenomD.eval_table_v("gamma1", eta, chi_pn)
        gamma2 = IMRPhenomD.eval_table_v("gamma2", eta, chi_pn)
        gamma3 = IMRPhenomD.eval_table_v("gamma3", eta, chi_pn)
        sigma1 = IMRPhenomD.eval_table_v("sigma1", eta, chi_pn)
        sigma2 = IMRPhenomD.eval_table_v("sigma2", eta, chi_pn)
        sigma3 = IMRPhenomD.eval_table_v("sigma3", eta, chi_pn)
        sigma4 = IMRPhenomD.eval_table_v("sigma4", eta, chi_pn)
        beta2 = IMRPhenomD.eval_table_v("beta2", eta, chi_pn)
        beta3 = IMRPhenomD.eval_table_v("beta3", eta, chi_pn)
        alpha1_raw = IMRPhenomD.eval_table_v("alpha1", eta, chi_pn)
        alpha2 = IMRPhenomD.eval_table_v("alpha2", eta, chi_pn)
        alpha3 = IMRPhenomD.eval_table_v("alpha3", eta, chi_pn)
        alpha4 = IMRPhenomD.eval_table_v("alpha4", eta, chi_pn)
        alpha5 = IMRPhenomD.eval_table_v("alpha5", eta, chi_pn)

        provider = IMRPhenomD.remnant_properties if remnant_provider is None else remnant_provider
        m_final, a_final, f_rd, f_damp = provider(m1, m2, s1, s2)
        f_peak = IMRPhenomD.amplitude_peak_frequency(f_rd, f_damp, gamma2, gamma3)

        f1_amp = 0.014
        f3_amp = f_peak
        f1_ph = 0.018
        f2_ph = 0.5 * f_rd

        a_pn = IMRPhenomD.pn_amplitude_coefficients(eta, delta, chi_s, chi_a)

        def eval_scaled_amp_ins(mf):
            v = (PI * mf) ** (1.0 / 3.0)
            pn_part = (
                a_pn[0]
                + a_pn[1] * v
                + a_pn[2] * (v**2)
                + a_pn[3] * (v**3)
                + a_pn[4] * (v**4)
                + a_pn[5] * (v**5)
                + a_pn[6] * (v**6)
            )
            phen_part = rho1 * (mf ** (7.0 / 3.0)) + rho2 * (mf ** (8.0 / 3.0)) + rho3 * (mf**3.0)
            return pn_part + phen_part

        def eval_scaled_amp_ins_deriv(mf):
            dpn = (
                (2.0 / 3.0) * a_pn[2] * (PI ** (2.0 / 3.0)) * (mf ** (-1.0 / 3.0))
                + a_pn[3] * PI
                + (4.0 / 3.0) * a_pn[4] * (PI ** (4.0 / 3.0)) * (mf ** (1.0 / 3.0))
                + (5.0 / 3.0) * a_pn[5] * (PI ** (5.0 / 3.0)) * (mf ** (2.0 / 3.0))
                + 2.0 * a_pn[6] * (PI * PI) * mf
            )
            dphen = (
                (7.0 / 3.0) * rho1 * (mf ** (4.0 / 3.0))
                + (8.0 / 3.0) * rho2 * (mf ** (5.0 / 3.0))
                + 3.0 * rho3 * (mf * mf)
            )
            return dpn + dphen

        def eval_scaled_amp_mr(mf):
            num = gamma1 * (gamma3 * f_damp)
            den = (mf - f_rd) ** 2 + (gamma3 * f_damp) ** 2
            exp_arg = -(gamma2 * (mf - f_rd)) / (gamma3 * f_damp)
            return (num / den) * jnp.exp(exp_arg)

        v1_col = eval_scaled_amp_ins(f1_amp)
        d1_col = eval_scaled_amp_ins_deriv(f1_amp)
        v3_col = eval_scaled_amp_mr(f3_amp)
        width = gamma3 * f_damp
        offset = f3_amp - f_rd
        d3_col = v3_col * (-2.0 * offset / (offset**2 + width**2) - gamma2 / width)

        d0, d1_c, d2, d3_c, d4 = IMRPhenomD.intermediate_amplitude_coeffs(
            f1=f1_amp, f3=f3_amp, v1=v1_col, v2=v2, v3=v3_col, d1=d1_col, d3=d3_col
        )

        # Phase matching
        (
            tf2_0, tf2_2, tf2_3, tf2_4, tf2_5, tf2_5_log,
            tf2_6, tf2_6_log, tf2_7, *_
        ) = TaylorF2.phase_coefficients(m1, m2, s1, s2, phenom_d_calibration=True)

        def eval_tf2_at(u):
            u_arr = jnp.asarray(u)
            v = (PI * u_arr) ** (1.0 / 3.0)
            ln_v = jnp.log(v)
            ln_4v = jnp.log(4.0 * v)

            def match_dim(t):
                if hasattr(t, "ndim") and u_arr.ndim > t.ndim:
                    return jnp.expand_dims(t, -1)
                return t

            c0 = match_dim(tf2_0)
            c2 = match_dim(tf2_2)
            c3 = match_dim(tf2_3)
            c4 = match_dim(tf2_4)
            c5 = match_dim(tf2_5)
            c5_log = match_dim(tf2_5_log)
            c6 = match_dim(tf2_6)
            c6_log = match_dim(tf2_6_log)
            c7 = match_dim(tf2_7)
            eta_m = match_dim(eta)

            pn_sum = (
                c0
                + c2 * (v**2)
                + c3 * (v**3)
                + c4 * (v**4)
                + (c5 + c5_log * ln_v) * (v**5)
                + (c6 + c6_log * ln_4v) * (v**6)
                + c7 * (v**7)
            )
            phi_tf2 = (3.0 / (128.0 * eta_m * (v**5))) * pn_sum
            dv_du = (PI / 3.0) * (PI * u_arr) ** (-2.0 / 3.0)
            d_pn_sum_dv = (
                -5.0 * c0 * (v ** -6)
                - 3.0 * c2 * (v ** -4)
                - 2.0 * c3 * (v ** -3)
                - c4 * (v ** -2)
                + c5_log * (v ** -1)
                + (c6 + c6_log * ln_4v + c6_log)
                + 2.0 * c7 * v
            )
            dphi_tf2 = (3.0 / (128.0 * eta_m)) * d_pn_sum_dv * dv_du
            return phi_tf2, dphi_tf2

        phi_tf2_f1, dphi_tf2_f1 = eval_tf2_at(f1_ph)
        phi_ins_f1 = phi_tf2_f1 + (1.0 / eta) * (
            sigma1 * f1_ph
            + (3.0 / 4.0) * sigma2 * (f1_ph ** (4.0 / 3.0))
            + (3.0 / 5.0) * sigma3 * (f1_ph ** (5.0 / 3.0))
            + 0.5 * sigma4 * (f1_ph ** 2.0)
        )
        dphi_ins_f1 = dphi_tf2_f1 + (1.0 / eta) * (
            sigma1
            + sigma2 * (f1_ph ** (1.0 / 3.0))
            + sigma3 * (f1_ph ** (2.0 / 3.0))
            + sigma4 * f1_ph
        )

        beta1 = eta * dphi_ins_f1 - beta2 * (f1_ph ** -1.0) - beta3 * (f1_ph ** -4.0)
        beta0 = (
            eta * phi_ins_f1
            - beta1 * f1_ph
            - beta2 * math.log(f1_ph)
            + (beta3 / 3.0) * (f1_ph ** -3.0)
        )

        phi_int_f2 = (1.0 / eta) * (
            beta0
            + beta1 * f2_ph
            + beta2 * jnp.log(f2_ph)
            - (beta3 / 3.0) * (f2_ph ** -3.0)
        )
        dphi_int_f2 = (1.0 / eta) * (
            beta1
            + beta2 * (f2_ph ** -1.0)
            + beta3 * (f2_ph ** -4.0)
        )

        atan_arg_f2 = (f2_ph - alpha5 * f_rd) / f_damp
        d_atan_f2 = (alpha4 / f_damp) / (1.0 + atan_arg_f2 * atan_arg_f2)

        alpha1 = (
            eta * dphi_int_f2
            - alpha2 * (f2_ph ** -2.0)
            - alpha3 * (f2_ph ** -0.25)
            - d_atan_f2
        )
        alpha0 = (
            eta * phi_int_f2
            - alpha1 * f2_ph
            + alpha2 * (f2_ph ** -1.0)
            - (4.0 / 3.0) * alpha3 * (f2_ph ** 0.75)
            - alpha4 * jnp.arctan(atan_arg_f2)
        )

        atan_lorentz_peak = (alpha4 * f_damp) / (f_damp * f_damp + (f_peak - alpha5 * f_rd) ** 2)
        slope_peak = (
            alpha1_raw
            + alpha2 * (f_peak ** -2.0)
            + alpha3 * (f_peak ** -0.25)
            + atan_lorentz_peak
        ) / eta

        # Broadcast over frequency grid
        m_sec = total_mass * MSUN_SEC
        is_batched = total_mass.ndim >= 1
        if f.ndim == 1 and is_batched:
            f_eval = jnp.expand_dims(f, 0)
            m_sec_eval = jnp.expand_dims(m_sec, -1)
            eta_u = jnp.expand_dims(eta, -1)
            f_rd_u = jnp.expand_dims(f_rd, -1)
            f_damp_u = jnp.expand_dims(f_damp, -1)
            f3_amp_u = jnp.expand_dims(f3_amp, -1)
            f2_ph_u = jnp.expand_dims(f2_ph, -1)
            rho1_u = jnp.expand_dims(rho1, -1)
            rho2_u = jnp.expand_dims(rho2, -1)
            rho3_u = jnp.expand_dims(rho3, -1)
            d0_u = jnp.expand_dims(d0, -1)
            d1_u = jnp.expand_dims(d1_c, -1)
            d2_u = jnp.expand_dims(d2, -1)
            d3_u = jnp.expand_dims(d3_c, -1)
            d4_u = jnp.expand_dims(d4, -1)
            gamma1_u = jnp.expand_dims(gamma1, -1)
            gamma2_u = jnp.expand_dims(gamma2, -1)
            gamma3_u = jnp.expand_dims(gamma3, -1)
            sigma1_u = jnp.expand_dims(sigma1, -1)
            sigma2_u = jnp.expand_dims(sigma2, -1)
            sigma3_u = jnp.expand_dims(sigma3, -1)
            sigma4_u = jnp.expand_dims(sigma4, -1)
            beta0_u = jnp.expand_dims(beta0, -1)
            beta1_u = jnp.expand_dims(beta1, -1)
            beta2_u = jnp.expand_dims(beta2, -1)
            beta3_u = jnp.expand_dims(beta3, -1)
            alpha0_u = jnp.expand_dims(alpha0, -1)
            alpha1_u = jnp.expand_dims(alpha1, -1)
            alpha2_u = jnp.expand_dims(alpha2, -1)
            alpha3_u = jnp.expand_dims(alpha3, -1)
            alpha4_u = jnp.expand_dims(alpha4, -1)
            alpha5_u = jnp.expand_dims(alpha5, -1)
            slope_peak_u = jnp.expand_dims(slope_peak, -1)
            dist_u = jnp.expand_dims(dist, -1)
            inc_u = jnp.expand_dims(inc, -1)
            tc_u = jnp.expand_dims(t_c, -1)
            phic_u = jnp.expand_dims(phi_c, -1)
            a_pn_u = [jnp.expand_dims(a, -1) for a in a_pn]
        else:
            f_eval = f
            m_sec_eval = m_sec
            eta_u, f_rd_u, f_damp_u, f3_amp_u, f2_ph_u = eta, f_rd, f_damp, f3_amp, f2_ph
            rho1_u, rho2_u, rho3_u = rho1, rho2, rho3
            d0_u, d1_u, d2_u, d3_u, d4_u = d0, d1_c, d2, d3_c, d4
            gamma1_u, gamma2_u, gamma3_u = gamma1, gamma2, gamma3
            sigma1_u, sigma2_u, sigma3_u, sigma4_u = sigma1, sigma2, sigma3, sigma4
            beta0_u, beta1_u, beta2_u, beta3_u = beta0, beta1, beta2, beta3
            alpha0_u, alpha1_u, alpha2_u, alpha3_u, alpha4_u, alpha5_u = alpha0, alpha1, alpha2, alpha3, alpha4, alpha5
            slope_peak_u = slope_peak
            dist_u, inc_u, tc_u, phic_u = dist, inc, t_c, phi_c
            a_pn_u = a_pn

        f_eval_safe = jnp.maximum(f_eval, 1e-8)
        mf = m_sec_eval * f_eval_safe

        # Scaled Amplitude
        v_eval = (PI * mf) ** (1.0 / 3.0)
        pn_amp = (
            a_pn_u[0]
            + a_pn_u[1] * v_eval
            + a_pn_u[2] * (v_eval**2)
            + a_pn_u[3] * (v_eval**3)
            + a_pn_u[4] * (v_eval**4)
            + a_pn_u[5] * (v_eval**5)
            + a_pn_u[6] * (v_eval**6)
        )
        scaled_amp_ins = (
            pn_amp
            + rho1_u * (mf ** (7.0 / 3.0))
            + rho2_u * (mf ** (8.0 / 3.0))
            + rho3_u * (mf**3.0)
        )

        scaled_amp_int = (
            d0_u
            + d1_u * mf
            + d2_u * (mf**2)
            + d3_u * (mf**3)
            + d4_u * (mf**4)
        )

        num_mr = gamma1_u * (gamma3_u * f_damp_u)
        den_mr = (mf - f_rd_u) ** 2 + (gamma3_u * f_damp_u) ** 2
        exp_arg_mr = -(gamma2_u * (mf - f_rd_u)) / (gamma3_u * f_damp_u)
        scaled_amp_mr = (num_mr / den_mr) * jnp.exp(exp_arg_mr)

        scaled_amp = jnp.where(
            mf <= f1_amp,
            scaled_amp_ins,
            jnp.where(
                mf <= f3_amp_u,
                scaled_amp_int,
                scaled_amp_mr,
            ),
        )

        d_sec = (dist_u * MPC_SI) / C_SI
        prefactor = jnp.sqrt(5.0 * eta_u / 24.0) / (PI ** (2.0 / 3.0) * d_sec)
        a0_phys = prefactor * (m_sec_eval ** (5.0 / 6.0)) * (f_eval_safe ** (-7.0 / 6.0))
        full_amp = a0_phys * scaled_amp

        # Cutoff at Mf = 0.2
        f_cut_phys = 0.2 / m_sec_eval
        full_amp = jnp.where(f_eval <= f_cut_phys, full_amp, jnp.zeros_like(full_amp))

        # Phase assembly
        phi_tf2_grid, _ = eval_tf2_at(mf)
        phi_ins_grid = phi_tf2_grid + (1.0 / eta_u) * (
            sigma1_u * mf
            + (3.0 / 4.0) * sigma2_u * (mf ** (4.0 / 3.0))
            + (3.0 / 5.0) * sigma3_u * (mf ** (5.0 / 3.0))
            + 0.5 * sigma4_u * (mf**2.0)
        )

        phi_int_grid = (1.0 / eta_u) * (
            beta0_u
            + beta1_u * mf
            + beta2_u * jnp.log(mf)
            - (beta3_u / 3.0) * (mf**-3.0)
        )

        atan_arg_grid = (mf - alpha5_u * f_rd_u) / f_damp_u
        phi_mr_grid = (1.0 / eta_u) * (
            alpha0_u
            + alpha1_u * mf
            - alpha2_u * (mf**-1.0)
            + (4.0 / 3.0) * alpha3_u * (mf**0.75)
            + alpha4_u * jnp.arctan(atan_arg_grid)
        )

        phi_imr = jnp.where(
            mf <= f1_ph,
            phi_ins_grid,
            jnp.where(
                mf <= f2_ph_u,
                phi_int_grid,
                phi_mr_grid,
            ),
        )

        psi_raw = -(PI / 4.0) + phi_imr

        conv = reference_frame if reference_frame is not None else (
            phasing_convention if phasing_convention is not None else ("reference_frequency" if f_ref is not None else "coalescence")
        )
        conv = conv.lower()

        if conv in ("reference_frequency", "reffreq", "ref"):
            ref_f = f_ref if f_ref is not None else (f_eval.reshape(-1)[0] if f_eval.size > 0 else 20.0)
            fref_u = jnp.asarray(ref_f, dtype=dtype)
            if is_batched and f_eval.ndim > fref_u.ndim:
                fref_u = jnp.expand_dims(fref_u, -1)

            mf_ref = m_sec_eval * fref_u
            phi_tf2_ref, _ = eval_tf2_at(mf_ref)
            phi_ins_ref = phi_tf2_ref + (1.0 / eta_u) * (
                sigma1_u * mf_ref
                + (3.0 / 4.0) * sigma2_u * (mf_ref ** (4.0 / 3.0))
                + (3.0 / 5.0) * sigma3_u * (mf_ref ** (5.0 / 3.0))
                + 0.5 * sigma4_u * (mf_ref**2.0)
            )
            phi_int_ref = (1.0 / eta_u) * (
                beta0_u
                + beta1_u * mf_ref
                + beta2_u * jnp.log(mf_ref)
                - (beta3_u / 3.0) * (mf_ref**-3.0)
            )
            atan_arg_ref = (mf_ref - alpha5_u * f_rd_u) / f_damp_u
            phi_mr_ref = (1.0 / eta_u) * (
                alpha0_u
                + alpha1_u * mf_ref
                - alpha2_u * (mf_ref**-1.0)
                + (4.0 / 3.0) * alpha3_u * (mf_ref**0.75)
                + alpha4_u * jnp.arctan(atan_arg_ref)
            )
            phi_imr_ref = jnp.where(
                mf_ref <= f1_ph,
                phi_ins_ref,
                jnp.where(
                    mf_ref <= f2_ph_u,
                    phi_int_ref,
                    phi_mr_ref,
                ),
            )
            psi_ref = -(PI / 4.0) + phi_imr_ref
            total_phase = (
                psi_raw
                - psi_ref
                - slope_peak_u * (mf - mf_ref)
                + 2.0 * PI * f_eval * tc_u
                - 2.0 * phic_u
            )
        else:
            total_phase = (
                psi_raw
                - slope_peak_u * mf
                + 2.0 * PI * f_eval * tc_u
                - 2.0 * phic_u
            )

        cos_inc = jnp.cos(inc_u)
        plus_factor = 0.5 * (1.0 + cos_inc * cos_inc)
        cross_factor = cos_inc

        h_plus = (plus_factor * full_amp) * (jnp.cos(total_phase) - 1j * jnp.sin(total_phase))
        h_cross = (cross_factor * full_amp) * (-jnp.sin(total_phase) - 1j * jnp.cos(total_phase))

        h_plus = jnp.where(f_eval > 0.0, h_plus, jnp.zeros_like(h_plus))
        h_cross = jnp.where(f_eval > 0.0, h_cross, jnp.zeros_like(h_cross))

        if return_phase:
            return h_plus, h_cross, total_phase

        return h_plus, h_cross

    @staticmethod
    def __call__(*args, **kwargs):
        return IMRPhenomD.evaluate(*args, **kwargs)

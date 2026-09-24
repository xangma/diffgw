"""Clean-room implementation of IMRPhenomXHM calibrated higher-order multipolar approximant in JAX.

Literature citations:
- García-Quirós et al. (2020), Phys. Rev. D 102, 064002 (arXiv:2001.10914)
  "Multimode frequency-domain model for the gravitational wave signal from non-precessing black-hole binaries: IMRPhenomXHM"
- Pratten et al. (2020), Phys. Rev. D 101, 024029 (arXiv:2001.10911)
  "Setting the cornerstone for next-generation models of gravitational reactions: IMRPhenomXAS"
- Berti, Cardoso, Will (2006), Phys. Rev. D 73, 064030 (arXiv:gr-qc/0512160)
  "Quasinormal modes of Kerr black holes: from astrophysics to string theory"
"""

from typing import Dict, Optional, Sequence, Tuple, Union
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
)
from jaxwave.approximants.imrphenom_xas import (
    IMRPhenomXAS,
    XASParameterFits,
    safe_sqrt,
)
from jaxwave.approximants.xhm_fits import XHMParameterFits, _nonnegative_sqrt
from jaxwave.approximants.xhm_native_fits import (
    eval_native_ringdown_params,
    eval_native_phase_params,
)
from jaxwave.approximants.imrphenom_hm import _swsh_projection_coefficients
from jaxwave.approximants.polynomial_interval import polynomial_has_nonpositive


# Pre-inverted 3x3 matrix for inspiral collocation points
# Collocation points at 0.5 f_Ins, 0.75 f_Ins, 1.0 f_Ins
# Powers: 7/3, 8/3, 3
_INSP_MAT = jnp.array([
    [0.5 ** (7.0 / 3.0), 0.5 ** (8.0 / 3.0), 0.5 ** 3.0],
    [0.75 ** (7.0 / 3.0), 0.75 ** (8.0 / 3.0), 0.75 ** 3.0],
    [1.0, 1.0, 1.0],
], dtype=jnp.float64)
_INSP_MAT_INV = jnp.linalg.inv(_INSP_MAT)


def _safe_solve(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Solve linear system a @ x = b robustly across JAX versions and batch dimensions."""
    return jnp.linalg.solve(a, b[..., None]).squeeze(-1)


def _published_parameter_amplitude_veto(
    mode: Tuple[int, int],
    mass_ratio: jnp.ndarray,
    eta: jnp.ndarray,
    chi1: jnp.ndarray,
    chi2: jnp.ndarray,
) -> jnp.ndarray:
    """Remove both inner amplitude nodes in the unconditional paper regions.

    García-Quirós et al., arXiv:2001.10914v2, Sec. V.A.3 and Table V.
    """
    return (
        ((mode == (2, 1)) & (eta < 0.23) & (chi1 > 0.7) & (chi2 < -0.5))
        | ((mode == (3, 2)) & (mass_ratio > 2.5) & (chi1 < -0.6) & (chi2 > 0.0))
        | ((mode == (3, 2)) & (chi1 < -0.9) & (chi2 < -0.9))
        | ((mode in ((3, 3), (4, 4))) & (mass_ratio > 7.0) & (chi1 > 0.95))
    )


def _ordered_amplitude_vetoes(
    mode: Tuple[int, int],
    mass_ratio: jnp.ndarray,
    eta: jnp.ndarray,
    chi1: jnp.ndarray,
    chi2: jnp.ndarray,
    h_ins: jnp.ndarray,
    h_int1: jnp.ndarray,
    h_int2: jnp.ndarray,
    h_rd: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Apply source-supported ordered intermediate-amplitude vetoes."""
    alive1 = jnp.ones_like(h_ins, dtype=jnp.bool_)
    alive2 = jnp.ones_like(h_ins, dtype=jnp.bool_)
    remove_derivatives = jnp.zeros_like(h_ins, dtype=jnp.bool_)

    both_alive = alive1 & alive2
    monotonic = ((h_ins <= h_int1) & (h_int1 <= h_int2) & (h_int2 <= h_rd)) | (
        (h_ins >= h_int1) & (h_int1 >= h_int2) & (h_int2 >= h_rd)
    )
    remove_middle = both_alive & ~monotonic & ((mode == (2, 1)) | (mode == (3, 2)))
    alive2 = alive2 & ~remove_middle

    if mode in ((3, 3), (4, 4)):
        bad_lambda = (mass_ratio > 7.0) & (chi1 > 0.95)
        alive1 = jnp.where(bad_lambda, jnp.zeros_like(alive1), alive1)
        alive2 = jnp.where(bad_lambda, jnp.zeros_like(alive2), alive2)

    unconditional = _published_parameter_amplitude_veto(mode, mass_ratio, eta, chi1, chi2)
    alive1 = jnp.where(unconditional, jnp.zeros_like(alive1), alive1)
    alive2 = jnp.where(unconditional, jnp.zeros_like(alive2), alive2)

    conditional = (mass_ratio > 40.0) & (chi1 > 0.9)
    if mode in ((2, 1), (3, 2), (3, 3), (4, 4)):
        alive1 = jnp.where(conditional, jnp.zeros_like(alive1), alive1)
        alive2 = jnp.where(conditional, jnp.zeros_like(alive2), alive2)

    return alive1, alive2, remove_derivatives


def _unsqueeze_to(arr: Union[float, jnp.ndarray], target: jnp.ndarray) -> jnp.ndarray:
    """Unsqueeze array along trailing dimensions until it matches target's dimensionality."""
    if not isinstance(arr, jnp.ndarray):
        arr = jnp.asarray(arr)
    res = arr
    while res.ndim < target.ndim:
        res = jnp.expand_dims(res, -1)
    return res


def _value_and_derivative(function, frequency: jnp.ndarray):
    """Value and pointwise frequency derivative using forward-mode autodiff in JAX."""
    val, deriv = jax.jvp(function, (frequency,), (jnp.ones_like(frequency),))
    return val, deriv


def _pn_phase_lambda(l: int, m: int, frequency: jnp.ndarray, eta: jnp.ndarray, delta: jnp.ndarray) -> jnp.ndarray:
    """Frequency-dependent part of Eq. (4.9), expanded through linear f."""
    eta_u = _unsqueeze_to(eta, frequency)
    delta_u = _unsqueeze_to(delta, frequency)
    if (l, m) == (2, 1):
        coefficient = jnp.where(delta_u != 0, jnp.full_like(delta_u, -0.5 * (1.0 + math.log(16.0))), jnp.zeros_like(delta_u))
    elif (l, m) == (3, 3):
        coefficient = jnp.where(delta_u != 0, jnp.full_like(delta_u, -21.0 / 5.0 + 6.0 * math.log(1.5)), jnp.zeros_like(delta_u))
    elif (l, m) == (3, 2):
        coefficient = (3.0 - 66.0 * eta_u / 5.0) / (1.0 - 3.0 * eta_u)
    elif (l, m) == (4, 4):
        coefficient = (eta_u * (3579.0 - 2880.0 * math.log(2.0)) - 1008.0 + 960.0 * math.log(2.0)) / (120.0 * (3.0 * eta_u - 1.0))
    else:
        raise ValueError(f"Unsupported PN phase mode {(l, m)}")
    return coefficient * (2.0 * PI / float(m)) * frequency


def _spherical_32_from_spheroidal(h22, h32_s, alpha222, alpha232, alpha322, alpha332):
    """Eliminate h22,S from the two simultaneous equations in Eq. (A3)."""
    ratio = alpha322 / alpha222
    return ratio * h22 + (alpha332 - ratio * alpha232) * h32_s


def _carrier_h22_ratio(strain, phase, phase_reference, a0):
    """Put the physical carrier in the H_lm normalization of Eq. (4.6)."""
    return jnp.abs(strain / a0) * jnp.exp(-1j * (phase - phase_reference))


def _eval_f_isco(chi: jnp.ndarray) -> jnp.ndarray:
    """Kerr ISCO frequency for dimensionless spin chi."""
    chi_clamped = jnp.clip(chi, -0.9999, 0.9999)
    one_m_chi2 = jnp.maximum(1.0 - chi_clamped**2, 1e-12)
    z1 = 1.0 + jnp.power(one_m_chi2, 1.0 / 3.0) * (
        jnp.power(1.0 + chi_clamped, 1.0 / 3.0) + jnp.power(1.0 - chi_clamped, 1.0 / 3.0)
    )
    z2 = safe_sqrt(3.0 * chi_clamped**2 + z1**2)
    diff = jnp.maximum((3.0 - z1) * (3.0 + z1 + 2.0 * z2), 0.0)
    r_isco = 3.0 + z2 - jnp.sign(chi_clamped) * safe_sqrt(diff)
    r_isco_clamped = jnp.maximum(r_isco, 1.0)
    return 1.0 / (PI * (r_isco_clamped**1.5 + chi_clamped))


def _pn_amplitude_h21(
    v: jnp.ndarray,
    eta: jnp.ndarray,
    delta: jnp.ndarray,
    chi_s: jnp.ndarray,
    chi_a: jnp.ndarray,
) -> jnp.ndarray:
    """Magnitude of the expanded PN H_21 (arXiv:2001.10914v2, Eq. E10)."""
    c1 = (1.0 / 3.0) * math.sqrt(2.0)
    v2 = v * v
    v3 = v2 * v
    v4 = v3 * v
    v5 = v4 * v
    v6 = v5 * v

    t1_re = v * delta
    t1_im = jnp.zeros_like(v)

    t2_re = v2 * (-1.5 * delta * chi_s - 1.5 * chi_a)
    t2_im = jnp.zeros_like(v)

    t3_re = v3 * ((117.0 / 56.0) * delta * eta + (335.0 / 672.0) * delta)
    t3_im = jnp.zeros_like(v)

    t4_re = v4 * (
        -(965.0 / 336.0) * delta * eta * chi_s
        + (3427.0 / 1344.0) * delta * chi_s
        - PI * delta
        - (2101.0 / 336.0) * eta * chi_a
        + (3427.0 / 1344.0) * chi_a
    )
    t4_im = v4 * (-0.5 * delta - 0.5 * delta * math.log(16.0))

    t5_re = v5 * (
        (21365.0 / 8064.0) * delta * (eta**2)
        + 10.0 * delta * eta * (chi_a**2)
        + (39.0 / 8.0) * delta * eta * (chi_s**2)
        - (36529.0 / 12544.0) * delta * eta
        - (307.0 / 32.0) * delta * (chi_a**2)
        - (307.0 / 32.0) * delta * (chi_s**2)
        + 3.0 * PI * delta * chi_s
        - (964357.0 / 8128512.0) * delta
        + (213.0 / 4.0) * eta * chi_a * chi_s
        - (307.0 / 16.0) * chi_a * chi_s
        + 3.0 * PI * chi_a
    )
    t5_im = jnp.zeros_like(v)

    t6_re = v6 * (
        -(547.0 / 768.0) * delta * (eta**2) * chi_s
        - 15.0 * delta * eta * (chi_a**2) * chi_s
        - (3.0 / 16.0) * delta * eta * (chi_s**3)
        - (7049629.0 / 225792.0) * delta * eta * chi_s
        + (417.0 / 112.0) * PI * delta * eta
        + (729.0 / 64.0) * delta * (chi_a**2) * chi_s
        + (243.0 / 64.0) * delta * (chi_s**3)
        + (143063173.0 / 5419008.0) * delta * chi_s
        - (2455.0 / 1344.0) * PI * delta
        + (42617.0 / 1792.0) * (eta**2) * chi_a
        - 15.0 * eta * (chi_a**3)
        - (489.0 / 16.0) * eta * chi_a * (chi_s**2)
        - (22758317.0 / 225792.0) * eta * chi_a
        + (243.0 / 64.0) * (chi_a**3)
        + (729.0 / 64.0) * chi_a * (chi_s**2)
        + (143063173.0 / 5419008.0) * chi_a
    )
    t6_im = v6 * (
        -(1489.0 / 112.0) * delta * eta
        - (89.0 / 28.0) * delta * eta * math.log(2.0)
        - (335.0 / 1344.0) * delta
        - (335.0 / 336.0) * delta * math.log(2.0)
    )

    t_re = t1_re + t2_re + t3_re + t4_re + t5_re + t6_re
    t_im = t1_im + t2_im + t3_im + t4_im + t5_im + t6_im

    h_re = -c1 * t_im
    h_im = c1 * t_re
    return _nonnegative_sqrt(h_re**2 + h_im**2)


def _pn_amplitude_h33(
    v: jnp.ndarray,
    eta: jnp.ndarray,
    delta: jnp.ndarray,
    chi_s: jnp.ndarray,
    chi_a: jnp.ndarray,
) -> jnp.ndarray:
    """Magnitude of the expanded PN H_33 (arXiv:2001.10914v2, Eq. E11)."""
    c1 = -(3.0 / 4.0) * math.sqrt(5.0 / 7.0)
    v2 = v * v
    v3 = v2 * v
    v4 = v3 * v
    v5 = v4 * v
    v6 = v5 * v

    t1_re = v * delta
    t1_im = jnp.zeros_like(v)

    t3_re = v3 * delta * ((27.0 / 8.0) * eta - 1945.0 / 672.0)
    t3_im = jnp.zeros_like(v)

    t4_re = v4 * (
        -(2.0 / 3.0) * delta * eta * chi_s
        + (65.0 / 24.0) * delta * chi_s
        + PI * delta
        - (28.0 / 3.0) * eta * chi_a
        + (65.0 / 24.0) * chi_a
    )
    t4_im = v4 * (-(21.0 / 5.0) * delta + 6.0 * delta * math.log(1.5))

    t5_re = v5 * (
        (420389.0 / 63360.0) * delta * (eta**2)
        + 10.0 * delta * eta * (chi_a**2)
        + (1.0 / 8.0) * delta * eta * (chi_s**2)
        - (11758073.0 / 887040.0) * delta * eta
        - (81.0 / 32.0) * delta * (chi_a**2)
        - (81.0 / 32.0) * delta * (chi_s**2)
        - (1077664867.0 / 447068160.0) * delta
        + (81.0 / 4.0) * eta * chi_a * chi_s
        - (81.0 / 16.0) * chi_a * chi_s
    )
    t5_im = jnp.zeros_like(v)

    t6_re = v6 * (
        -(67.0 / 24.0) * delta * (eta**2) * chi_s
        - (58745.0 / 4032.0) * delta * eta * chi_s
        + (131.0 / 16.0) * PI * delta * eta
        + (163021.0 / 16128.0) * delta * chi_s
        - (5675.0 / 1344.0) * PI * delta
        - (137.0 / 24.0) * (eta**2) * chi_a
        - (148501.0 / 4032.0) * eta * chi_a
        + (163021.0 / 16128.0) * chi_a
    )
    t6_im = v6 * (
        -(440957.0 / 9720.0) * delta * eta
        + (69.0 / 4.0) * delta * eta * math.log(1.5)
        + (389.0 / 32.0) * delta
        - (1945.0 / 112.0) * delta * math.log(1.5)
    )

    t_re = t1_re + t3_re + t4_re + t5_re + t6_re
    t_im = t1_im + t3_im + t4_im + t5_im + t6_im

    h_re = -c1 * t_im
    h_im = c1 * t_re
    return _nonnegative_sqrt(h_re**2 + h_im**2)


def _pn_amplitude_h32(
    v: jnp.ndarray,
    eta: jnp.ndarray,
    delta: jnp.ndarray,
    chi_s: jnp.ndarray,
    chi_a: jnp.ndarray,
) -> jnp.ndarray:
    """Magnitude of the expanded PN H_32 (arXiv:2001.10914v2, Eq. E12)."""
    c1 = (1.0 / 3.0) * math.sqrt(5.0 / 7.0)
    v2 = v * v
    v3 = v2 * v
    v4 = v3 * v
    v5 = v4 * v
    v6 = v5 * v

    t2_re = v2 * (1.0 - 3.0 * eta)
    t3_re = v3 * (4.0 * eta * chi_s)
    t4_re = v4 * (
        -(589.0 / 72.0) * (eta**2)
        + (12325.0 / 2016.0) * eta
        - (10471.0 / 10080.0)
    )

    t5_re = v5 * (
        eta * ((113.0 / 8.0) * delta * chi_a + (1081.0 / 84.0) * chi_s)
        + (1.0 / 24.0) * (-113.0 * delta * chi_a - 113.0 * chi_s)
        - 15.0 * (eta**2) * chi_s
    )
    t5_im = v5 * (eta * (-66.0 / 5.0) + (72.0 / 24.0))

    t6_re = v6 * (
        eta * (
            -(1633.0 / 48.0) * delta * chi_a * chi_s
            - (563.0 / 32.0) * (chi_a**2)
            - (2549.0 / 96.0) * (chi_s**2)
            + 8.0 * PI * chi_s
            - (8689883.0 / 149022720.0)
        )
        + (81.0 / 16.0) * delta * chi_a * chi_s
        + (837223.0 / 63360.0) * (eta**3)
        + (eta**2) * (30.0 * (chi_a**2) + (313.0 / 24.0) * (chi_s**2) - 78584047.0 / 2661120.0)
        + (81.0 / 32.0) * (chi_a**2)
        + (81.0 / 32.0) * (chi_s**2)
        + (824173699.0 / 447068160.0)
    )
    t6_im = jnp.zeros_like(v)

    t_re = t2_re + t3_re + t4_re + t5_re + t6_re
    t_im = t5_im + t6_im

    h_re = c1 * t_re
    h_im = c1 * t_im
    return _nonnegative_sqrt(h_re**2 + h_im**2)


def _pn_amplitude_h44(
    v: jnp.ndarray,
    eta: jnp.ndarray,
    delta: jnp.ndarray,
    chi_s: jnp.ndarray,
    chi_a: jnp.ndarray,
) -> jnp.ndarray:
    """Magnitude of the expanded PN H_44 (arXiv:2001.10914v2, Eq. E14)."""
    c1 = (4.0 / 9.0) * math.sqrt(10.0 / 7.0)
    v2 = v * v
    v4 = v2 * v2
    v5 = v4 * v
    v6 = v5 * v

    t2_re = v2 * (3.0 * eta - 1.0)
    t4_re = v4 * (
        (1063.0 / 88.0) * (eta**2)
        - (128221.0 / 7392.0) * eta
        + (158383.0 / 36960.0)
    )

    term5_inner_re = (
        -eta * (1695.0 * delta * chi_a + 2075.0 * chi_s)
        + 565.0 * delta * chi_a
        + 1140.0 * (eta**2) * chi_s
        + 565.0 * chi_s
    )
    term5_inner_im = -eta * (-3579.0 + 2880.0 * math.log(2.0)) - 1008.0 + 960.0 * math.log(2.0)
    t5_re = v5 * (PI * (2.0 - 6.0 * eta) + (1.0 / 120.0) * term5_inner_re)
    t5_im = v5 * ((1.0 / 120.0) * term5_inner_im)

    t6_re = v6 * (
        eta * (
            (243.0 / 16.0) * delta * chi_a * chi_s
            + (563.0 / 32.0) * (chi_a**2)
            + (247.0 / 32.0) * (chi_s**2)
            - (22580029007.0 / 880588800.0)
        )
        - (81.0 / 16.0) * delta * chi_a * chi_s
        - (7606537.0 / 274560.0) * (eta**3)
        + (eta**2) * (-30.0 * (chi_a**2) - (3.0 / 8.0) * (chi_s**2) + 901461137.0 / 11531520.0)
        - (81.0 / 32.0) * (chi_a**2)
        - (81.0 / 32.0) * (chi_s**2)
        + (7888301437.0 / 29059430400.0)
    )
    t6_im = jnp.zeros_like(v)

    t_re = t2_re + t4_re + t5_re + t6_re
    t_im = t5_im + t6_im

    h_re = c1 * t_re
    h_im = c1 * t_im
    return _nonnegative_sqrt(h_re**2 + h_im**2)


def _pn_amplitude_ratio(
    l: int,
    m: int,
    f: jnp.ndarray,
    eta: jnp.ndarray,
    delta: jnp.ndarray,
    chi_s: jnp.ndarray,
    chi_a: jnp.ndarray,
) -> jnp.ndarray:
    """Return |A_lm^PN|/A_22^0 using arXiv:2001.10914v2 Eq. (E8)."""
    m_flt = float(m)
    v = jnp.power(jnp.maximum((2.0 * PI * f) / m_flt, 1e-12), 1.0 / 3.0)
    factor = (m_flt / 2.0) ** (7.0 / 6.0)
    eta_u = _unsqueeze_to(eta, v)
    delta_u = _unsqueeze_to(delta, v)
    chi_s_u = _unsqueeze_to(chi_s, v)
    chi_a_u = _unsqueeze_to(chi_a, v)
    if l == 2 and m == 1:
        h_val = _pn_amplitude_h21(v, eta_u, delta_u, chi_s_u, chi_a_u)
    elif l == 3 and m == 3:
        h_val = _pn_amplitude_h33(v, eta_u, delta_u, chi_s_u, chi_a_u)
    elif l == 3 and m == 2:
        h_val = _pn_amplitude_h32(v, eta_u, delta_u, chi_s_u, chi_a_u)
    elif l == 4 and m == 4:
        h_val = _pn_amplitude_h44(v, eta_u, delta_u, chi_s_u, chi_a_u)
    else:
        raise ValueError(f"Mode ({l}, {m}) not supported.")
    return factor * h_val


@jax.tree_util.register_pytree_node_class
class IMRPhenomXHM:
    """IMRPhenomXHM Calibrated Higher-Harmonic IMR Frequency-Domain Approximant in JAX.

    Supports harmonics (l, m) in {(2, 2), (2, 1), (3, 3), (3, 2), (4, 4)}.
    Literature:
    - García-Quirós et al. (2020), Phys. Rev. D 102, 064002 (arXiv:2001.10914)
    - Pratten et al. (2020), Phys. Rev. D 101, 024029 (arXiv:2001.10911)
    """

    SUPPORTED_MODES = ((2, 2), (2, 1), (3, 3), (3, 2), (4, 4))
    supported_modes = SUPPORTED_MODES

    @property
    def get_supported_modes(self):
        return self.SUPPORTED_MODES

    SUPPORTED_CALIBRATIONS = ("sovereign", "torchwave_nr", "native", "torchwave")

    def __init__(
        self,
        modes: Optional[Sequence[Tuple[int, int]]] = None,
        carrier: str = "XAS",
        base_22=None,
        *,
        experimental_phase_connections: bool = False,
        calibration: str = "sovereign",
    ):
        self.carrier = carrier
        self.calibration = calibration.lower()
        if self.calibration not in self.SUPPORTED_CALIBRATIONS:
            raise ValueError(
                f"Unsupported calibration '{calibration}'. Sovereign calibration is the sole supported standard."
            )
        self.modes = tuple(modes) if modes is not None else self.SUPPORTED_MODES
        for lm in self.modes:
            if lm not in self.SUPPORTED_MODES:
                raise ValueError(f"Mode {lm} not supported. Supported: {self.SUPPORTED_MODES}")
        self.base_22 = base_22
        if experimental_phase_connections:
            raise ValueError("experimental_phase_connections=True is an unvalidated prototype and is not available in releases.")

    def tree_flatten(self):
        children = ()
        aux_data = (
            self.modes,
            self.carrier,
            self.base_22,
            self.calibration,
        )
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        modes, carrier, base_22, calib = aux_data
        return cls(
            modes=modes,
            carrier=carrier,
            base_22=base_22,
            calibration=calib,
        )

    def __call__(self, *args, **kwargs):
        kwargs.setdefault("modes", self.modes)
        kwargs.setdefault("carrier", self.carrier)
        kwargs.setdefault("base_22", self.base_22)
        kwargs.setdefault("calibration", self.calibration)
        return self.evaluate(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.__call__(*args, **kwargs)

    @classmethod
    def evaluate(
        cls,
        f: Optional[jnp.ndarray] = None,
        mass1: Union[float, jnp.ndarray] = 30.0,
        mass2: Union[float, jnp.ndarray] = 30.0,
        chi1: Union[float, jnp.ndarray] = 0.0,
        chi2: Union[float, jnp.ndarray] = 0.0,
        distance: Union[float, jnp.ndarray] = 100.0,
        inclination: Union[float, jnp.ndarray] = 0.0,
        phic: Union[float, jnp.ndarray] = 0.0,
        tc: Union[float, jnp.ndarray] = 0.0,
        f_ref: Optional[Union[float, jnp.ndarray]] = None,
        phi0: Union[float, jnp.ndarray] = 0.0,
        modes: Optional[Sequence[Tuple[int, int]]] = None,
        return_modes: bool = False,
        return_unprojected_modes: bool = False,
        phasing_convention: Optional[str] = None,
        reference_frame: Optional[str] = None,
        sample_frequencies: Optional[jnp.ndarray] = None,
        base_22=None,
        carrier: str = "XAS",
        experimental_phase_connections: bool = False,
        calibration: str = "sovereign",
        **kwargs,
    ) -> Union[
        Tuple[jnp.ndarray, jnp.ndarray],
        Tuple[jnp.ndarray, jnp.ndarray, Dict[Tuple[int, int], Union[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]]],
    ]:
        """Generate frequency-domain polarizations (h_plus, h_cross) including higher multipoles in JAX."""
        if experimental_phase_connections:
            raise ValueError("experimental_phase_connections=True is an unvalidated prototype and is not available in releases.")
        if f is None:
            if sample_frequencies is not None:
                f = sample_frequencies
            else:
                raise ValueError("Must provide frequency grid via 'f' or 'sample_frequencies'")

        dtype = jnp.float64
        f_arr = as_array(f, dtype=dtype)

        # Handle alias kwargs
        if "spin1z" in kwargs:
            chi1 = kwargs.pop("spin1z")
        if "spin2z" in kwargs:
            chi2 = kwargs.pop("spin2z")
        if "phi_c" in kwargs:
            phic = kwargs.pop("phi_c")
        if "t_c" in kwargs:
            tc = kwargs.pop("t_c")
        if "t0" in kwargs:
            tc = kwargs.pop("t0")
        if "phi_0" in kwargs:
            phi0 = kwargs.pop("phi_0")
        f_lower = kwargs.pop("f_lower", None)

        m1_in = as_array(mass1, dtype=dtype)
        m2_in = as_array(mass2, dtype=dtype)
        s1_in = as_array(chi1, dtype=dtype)
        s2_in = as_array(chi2, dtype=dtype)
        dist = as_array(distance, dtype=dtype)
        inc = as_array(inclination, dtype=dtype)
        phi_c = as_array(kwargs.pop("phic", phic), dtype=dtype)
        t_c = as_array(kwargs.pop("tc", tc), dtype=dtype)
        phi_0 = as_array(kwargs.pop("phi0", phi0), dtype=dtype)
        final_spin_override = kwargs.pop("_final_spin_override", None)
        if final_spin_override is not None:
            final_spin_override = as_array(final_spin_override, dtype=dtype)

        active_modes = tuple(modes) if modes is not None else cls.SUPPORTED_MODES
        for lm in active_modes:
            if lm not in cls.SUPPORTED_MODES:
                raise ValueError(f"Mode {lm} not supported. Supported: {cls.SUPPORTED_MODES}")

        # Body exchange canonical ordering (m1 >= m2), breaking equal-mass ties by spin
        swap_mask = (m1_in < m2_in) | ((m1_in == m2_in) & (s1_in < s2_in))
        m1 = jnp.where(swap_mask, m2_in, m1_in)
        m2 = jnp.where(swap_mask, m1_in, m2_in)
        s1 = jnp.where(swap_mask, s2_in, s1_in)
        s2 = jnp.where(swap_mask, s1_in, s2_in)

        total_mass = m1 + m2
        eta = component_masses_to_eta(m1, m2)
        delta, _, _, S = XHMParameterFits.compute_spin_variables(eta, s1, s2)
        chi_s = 0.5 * (s1 + s2)
        chi_a = 0.5 * (s1 - s2)
        chi_eff = (m1 * s1 + m2 * s2) / total_mass
        m_sec = total_mass * MSUN_SEC

        # Final remnant properties from XAS fits
        chi_f = (
            XASParameterFits.fit_final_spin(eta, s1, s2)
            if final_spin_override is None else final_spin_override
        )
        erad = XASParameterFits.fit_erad(eta, s1, s2)
        m_f = 1.0 - erad

        # ISCO and MECO frequencies for (2,2)
        f_isco_22 = _eval_f_isco(chi_f)
        f_meco_22 = XASParameterFits.fit_f_meco(eta, s1, s2)
        diff_isco_meco = jnp.abs(f_isco_22 - f_meco_22)

        # Mode QNMs in geometric units (rescaled by remnant mass M_f)
        f_ring: Dict[str, jnp.ndarray] = {}
        f_damp: Dict[str, jnp.ndarray] = {}
        for lm_str in ("21", "22", "32", "33", "44"):
            f_ring[lm_str] = getattr(XHMParameterFits, f"qnm_fring_{lm_str}")(chi_f) / m_f
            f_damp[lm_str] = getattr(XHMParameterFits, f"qnm_fdamp_{lm_str}")(chi_f) / m_f

        # Cutting frequencies per mode
        eps_amp = {
            "21": jnp.array(1.0, dtype=dtype),
            "33": jnp.array(1.0, dtype=dtype),
            "32": f_ring["32"] / f_ring["22"],
            "44": jnp.array(1.0, dtype=dtype),
        }
        del_amp = {
            "21": 0.75 - 0.235 * chi_eff - (5.0 / 6.0) * (chi_eff**2),
            "33": 0.75 - 0.235 * chi_eff - (5.0 / 6.0) * chi_eff,
            "32": (0.75 - 0.235 * jnp.abs(chi_eff)) * (f_ring["32"] / f_ring["22"]),
            "44": 0.75 - 0.235 * chi_eff,
        }
        m_vals = {"21": 1, "32": 2, "33": 3, "44": 4}

        f_ins_amp = {
            lm_str: (m_vals[lm_str] / 2.0) * (f_meco_22 * eps_amp[lm_str] + diff_isco_meco * del_amp[lm_str])
            for lm_str in ("21", "32", "33", "44")
        }
        f_rd_amp = {
            "21": 0.75 * f_ring["21"],
            "33": 0.95 * f_ring["33"],
            "32": f_ring["22"],
            "44": 0.90 * f_ring["44"],
        }
        f_ins_phi = {
            lm_str: (m_vals[lm_str] / 2.0) * f_meco_22
            for lm_str in ("21", "32", "33", "44")
        }
        f_rd_phi = {
            "21": f_ring["21"] - f_damp["21"],
            "33": f_ring["33"] - f_damp["33"],
            "32": f_ring["22"] - 0.5 * f_damp["22"],
            "44": f_ring["44"] - f_damp["44"],
        }

        # Amplitude sigma fits
        sigma_amp = {
            "21": XHMParameterFits.amp_sigma_21(eta, s1, s2, S),
            "33": jnp.array(1.30, dtype=dtype),
            "32": jnp.array(1.33, dtype=dtype),
            "44": jnp.array(1.33, dtype=dtype),
        }

        # 22-mode phase parameter for ringdown rescaling (Eq. 6.5)
        alpha2_22 = XHMParameterFits.phase_rd_alpha2(eta, s1, s2, S)

        # Batching alignment
        is_batched = total_mass.ndim >= 1
        if f_arr.ndim == 1 and is_batched:
            f_eval = jnp.expand_dims(f_arr, 0)
            m_sec_eval = jnp.expand_dims(m_sec, -1)
            eta_eval = jnp.expand_dims(eta, -1)
            dist_eval = jnp.expand_dims(dist, -1) if dist.ndim > 0 else dist
            inc_eval = jnp.expand_dims(inc, -1) if inc.ndim > 0 else inc
            phi0_eval = jnp.expand_dims(phi_0, -1) if phi_0.ndim > 0 else phi_0
            phic_eval = jnp.expand_dims(phi_c, -1) if phi_c.ndim > 0 else phi_c
            tc_eval = jnp.expand_dims(t_c, -1) if t_c.ndim > 0 else t_c
        else:
            f_eval = f_arr
            m_sec_eval = m_sec
            eta_eval = eta
            dist_eval = dist
            inc_eval = inc
            phi0_eval = phi_0
            phic_eval = phi_c
            tc_eval = t_c

        # Geometric frequency grid: Mf
        f_safe = jnp.maximum(f_eval, 1e-8)
        mf_grid = m_sec_eval * f_safe

        # Evaluate carrier (2, 2) mode at face-on (iota=0)
        base_carrier = base_22 if base_22 is not None else IMRPhenomXAS
        carrier_state = None
        if hasattr(base_carrier, "build_state"):
            carrier_state = base_carrier.build_state(
                mass1=m1,
                mass2=m2,
                chi1=s1,
                chi2=s2,
                f_ref=f_ref,
                _final_spin_override=final_spin_override,
                dtype=dtype,
            )
            hp_22_raw, _, phi_22_carrier = carrier_state.eval_waveform(
                f=f_eval,
                distance=dist,
                inclination=0.0,
                phi_c=0.0,
                t_c=0.0,
                return_phase=True,
            )
        else:
            carrier_eval_fn = getattr(base_carrier, "evaluate", base_carrier)
            hp_22_raw, _, phi_22_carrier = carrier_eval_fn(
                f=f_eval,
                mass1=m1,
                mass2=m2,
                chi1=s1,
                chi2=s2,
                distance=dist,
                inclination=0.0,
                phic=0.0,
                tc=0.0,
                f_ref=f_ref,
                phasing_convention=phasing_convention,
                reference_frame=reference_frame,
                return_phase=True,
                _final_spin_override=final_spin_override,
            )
        amp_22_phys = jnp.abs(hp_22_raw)

        # Base physical leading quadrupole factor A0_phys
        dist_sec = (dist_eval * MPC_SI) / C_SI
        prefactor = jnp.sqrt(5.0 * eta_eval / 24.0) / (PI ** (2.0 / 3.0) * dist_sec)
        a0_phys = prefactor * (m_sec_eval ** (5.0 / 6.0)) * (f_safe ** (-7.0 / 6.0))
        mode_norm = math.sqrt(16.0 * PI / 5.0)

        def eval_carrier_at(mf_target: jnp.ndarray):
            target_grid = jnp.expand_dims(mf_target, -1) if mf_target.ndim == m_sec.ndim else mf_target
            target_hz = target_grid / _unsqueeze_to(m_sec, target_grid)
            if carrier_state is not None:
                hp, _, phase = carrier_state.eval_waveform(
                    f=target_hz,
                    distance=m_sec * C_SI / MPC_SI,
                    inclination=0.0,
                    phi_c=0.0,
                    t_c=0.0,
                    return_phase=True,
                )
            else:
                carrier_eval_fn = getattr(base_carrier, "evaluate", base_carrier)
                hp, _, phase = carrier_eval_fn(
                    f=target_hz, mass1=m1, mass2=m2, chi1=s1, chi2=s2,
                    distance=m_sec * C_SI / MPC_SI, inclination=0.0, phic=0.0, tc=0.0,
                    f_ref=f_ref, phasing_convention=phasing_convention,
                    reference_frame=reference_frame, return_phase=True,
                    _final_spin_override=final_spin_override,
                )
            if mf_target.ndim == m_sec.ndim:
                return jnp.squeeze(hp, -1), jnp.squeeze(phase, -1)
            return hp, phase

        def eval_carrier_phase_at(mf_target: jnp.ndarray):
            target_grid = jnp.expand_dims(mf_target, -1) if mf_target.ndim == m_sec.ndim else mf_target
            target_hz = target_grid / _unsqueeze_to(m_sec, target_grid)
            if carrier_state is not None:
                phase = carrier_state.eval_phase(target_hz)
            else:
                phase = eval_carrier_at(mf_target)[1]
            if mf_target.ndim == m_sec.ndim:
                return jnp.squeeze(phase, -1)
            return phase

        def eval_carrier_phi_and_deriv(mf_target: jnp.ndarray):
            return _value_and_derivative(eval_carrier_phase_at, mf_target)

        h_plus_total = jnp.zeros_like(amp_22_phys, dtype=jnp.complex128)
        h_cross_total = jnp.zeros_like(h_plus_total)
        mode_dict = {}

        # Store h22 for mode-mixing with (3,2)
        exp_neg_i_phi22 = jnp.cos(phi_22_carrier) - 1j * jnp.sin(phi_22_carrier)
        h22_mode_strain = (mode_norm * amp_22_phys).astype(exp_neg_i_phi22.dtype) * exp_neg_i_phi22

        for l, m in active_modes:
            lm_str = f"{l}{m}"
            if (l, m) == (2, 2):
                h_mode = h22_mode_strain
            else:
                m_flt = float(m)
                f_ins_a = f_ins_amp[lm_str]
                f_rd_a = f_rd_amp[lm_str]
                f_ins_p = f_ins_phi[lm_str]
                f_rd_p = f_rd_phi[lm_str]
                f_ring_lm = f_ring[lm_str]
                f_damp_lm = f_damp[lm_str]
                sigma_lm = sigma_amp[lm_str]

                # -------------------------------------------------------------
                # 1. Amplitude Reconstruction
                # -------------------------------------------------------------
                f1_ins = 0.5 * f_ins_a
                f2_ins = 0.75 * f_ins_a
                f3_ins = f_ins_a

                h1_pn = _pn_amplitude_ratio(l, m, f1_ins, eta, delta, chi_s, chi_a)
                h2_pn = _pn_amplitude_ratio(l, m, f2_ins, eta, delta, chi_s, chi_a)
                h3_pn = _pn_amplitude_ratio(l, m, f3_ins, eta, delta, chi_s, chi_a)

                h1_fit = getattr(XHMParameterFits, f"amp_insp1_{lm_str}")(eta, s1, s2, S)
                h2_fit = getattr(XHMParameterFits, f"amp_insp2_{lm_str}")(eta, s1, s2, S)
                h3_fit = getattr(XHMParameterFits, f"amp_insp3_{lm_str}")(eta, s1, s2, S)
                if m % 2 != 0:
                    h1_fit = jnp.abs(h1_fit)
                    h2_fit = jnp.abs(h2_fit)
                    h3_fit = jnp.abs(h3_fit)

                rhs_insp = jnp.stack([h1_fit - h1_pn, h2_fit - h2_pn, h3_fit - h3_pn], axis=-1)
                inv_mat = _INSP_MAT_INV
                abc = jnp.matmul(rhs_insp, inv_mat.T)
                alpha_insp = abc[..., 0]
                beta_insp = abc[..., 1]
                gamma_insp = abc[..., 2]

                # Ringdown amplitude parameters (Sovereign SXS NR calibrated)
                alambda_fit, lambda_fit = eval_native_ringdown_params(l, m, eta, delta)

                def eval_h_rd(f_in: jnp.ndarray) -> jnp.ndarray:
                    w = _unsqueeze_to(f_damp_lm * sigma_lm, f_in)
                    f_ring_u = _unsqueeze_to(f_ring_lm, f_in)
                    lam_u = _unsqueeze_to(lambda_fit, f_in)
                    alam_u = _unsqueeze_to(alambda_fit, f_in)
                    den = (f_in - f_ring_u) ** 2 + w**2
                    exp_term = jnp.exp(-jnp.clip((f_in - f_ring_u) * lam_u / w, -80.0, 80.0))
                    f_pow = jnp.power(jnp.maximum(f_in, 1e-8), 1.0 / 12.0)
                    return (1.0 / f_pow) * (jnp.abs(alam_u) * w / jnp.maximum(den, 1e-12)) * exp_term

                if (l, m) == (3, 2):
                    alpha2_32 = alpha2_22
                    g1 = XHMParameterFits.phase_32_s_dphi_p1(eta, s1, s2, S)
                    g2 = XHMParameterFits.phase_32_s_dphi_fring32(eta, s1, s2, S)
                    g3 = XHMParameterFits.phase_32_s_dphi_p4(eta, s1, s2, S)

                    f_nodes_s = jnp.stack([
                        f_ring["22"],
                        f_ring["32"],
                        f_ring["32"] + 0.5 * f_damp["32"],
                    ], axis=-1)

                    f_damp_32_u = jnp.expand_dims(f_damp["32"], -1) if f_damp["32"].ndim > 0 else f_damp["32"]
                    f_ring_32_u = jnp.expand_dims(f_ring["32"], -1) if f_ring["32"].ndim > 0 else f_ring["32"]
                    lor_nodes_s = f_damp_32_u / ((f_damp_32_u**2) + (f_nodes_s - f_ring_32_u)**2)
                    m_s = jnp.stack([
                        1.0 / (f_nodes_s**4),
                        lor_nodes_s,
                        jnp.ones_like(f_nodes_s),
                    ], axis=-1)
                    rhs_s = jnp.stack([
                        g1 - alpha2_32 / (f_nodes_s[..., 0]**2),
                        g2 - alpha2_32 / (f_nodes_s[..., 1]**2),
                        g3 - alpha2_32 / (f_nodes_s[..., 2]**2),
                    ], axis=-1)
                    sol_s = _safe_solve(m_s, rhs_s)

                    alpha4_32 = sol_s[..., 0]
                    alpha_lam_32 = sol_s[..., 1]
                    dt_rd_32 = sol_s[..., 2]

                    # Alignment to (2,2) carrier in spheroidal basis
                    f0_align = f_ring["22"] + f_damp["22"]
                    f1_align = f_ring["22"]

                    delta_t_32_s = XHMParameterFits.phase_32_s_deltadphi(eta, s1, s2, S)
                    delta_phi_32_s = XHMParameterFits.phase_32_s_deltaphi(eta, s1, s2, S)

                    _, carrier_dp_at_f0 = eval_carrier_phi_and_deriv(f0_align)
                    carrier_p_at_f1, _ = eval_carrier_phi_and_deriv(f1_align)

                    def eval_dphi_32_s(f_in: jnp.ndarray) -> jnp.ndarray:
                        damp_u = _unsqueeze_to(f_damp["32"], f_in)
                        ring_u = _unsqueeze_to(f_ring["32"], f_in)
                        a2_u = _unsqueeze_to(alpha2_32, f_in)
                        a4_u = _unsqueeze_to(alpha4_32, f_in)
                        alam_u = _unsqueeze_to(alpha_lam_32, f_in)
                        dt_u = _unsqueeze_to(dt_rd_32, f_in)
                        lor = damp_u / ((damp_u**2) + (f_in - ring_u)**2)
                        return (
                            a2_u / (f_in**2)
                            + a4_u / (f_in**4)
                            + alam_u * lor
                            + dt_u
                        )

                    time_shift_32 = (carrier_dp_at_f0 + delta_t_32_s) - eval_dphi_32_s(f0_align)
                    dt_rd_32 = dt_rd_32 + time_shift_32

                    phi_32_s_base_at_f1 = (
                        -alpha2_32 / f1_align
                        - (1.0 / 3.0) * alpha4_32 / (f1_align**3)
                        + alpha_lam_32 * jnp.arctan((f1_align - f_ring["32"]) / f_damp["32"])
                        + dt_rd_32 * f1_align
                    )
                    phi_rd_32_s_shift = (carrier_p_at_f1 + delta_phi_32_s) - phi_32_s_base_at_f1

                    def eval_phi_32_s(f_in: jnp.ndarray) -> jnp.ndarray:
                        damp_u = _unsqueeze_to(f_damp["32"], f_in)
                        ring_u = _unsqueeze_to(f_ring["32"], f_in)
                        a2_u = _unsqueeze_to(alpha2_32, f_in)
                        a4_u = _unsqueeze_to(alpha4_32, f_in)
                        alam_u = _unsqueeze_to(alpha_lam_32, f_in)
                        dt_u = _unsqueeze_to(dt_rd_32, f_in)
                        shift_u = _unsqueeze_to(phi_rd_32_s_shift, f_in)
                        return (
                            -a2_u / f_in
                            - (1.0 / 3.0) * a4_u / (f_in**3)
                            + alam_u * jnp.arctan((f_in - ring_u) / damp_u)
                            + dt_u * f_in
                            + shift_u
                        )

                    eval_h_rd_s = eval_h_rd
                    a222_re, a222_im = XHMParameterFits.mixing_alpha222(chi_f)
                    alpha222 = a222_re + 1j * a222_im
                    a232_re, a232_im = XHMParameterFits.mixing_alpha223(chi_f)
                    alpha232 = a232_re + 1j * a232_im
                    a322_re, a322_im = XHMParameterFits.mixing_alpha322(chi_f)
                    alpha322 = a322_re + 1j * a322_im
                    a332_re, a332_im = XHMParameterFits.mixing_alpha333(chi_f)
                    alpha332 = a332_re + 1j * a332_im

                    def eval_mixed_rd_relative(f_in: jnp.ndarray):
                        strain22, phase22 = eval_carrier_at(f_in)
                        a0 = _unsqueeze_to(m_sec * jnp.sqrt(5.0 * eta / 24.0) / (PI ** (2.0 / 3.0)), f_in) * f_in ** (-7.0 / 6.0)
                        phase_s = eval_phi_32_s(f_in)
                        relative22 = _carrier_h22_ratio(strain22, phase22, phase_s, a0)
                        return _spherical_32_from_spheroidal(
                            relative22, eval_h_rd_s(f_in),
                            _unsqueeze_to(alpha222, f_in), _unsqueeze_to(alpha232, f_in),
                            _unsqueeze_to(alpha322, f_in), _unsqueeze_to(alpha332, f_in),
                        )

                    def eval_phi_32_rd(f_in: jnp.ndarray):
                        return eval_phi_32_s(f_in) - jnp.angle(eval_mixed_rd_relative(f_in))

                    def eval_h_rd(f_in: jnp.ndarray) -> jnp.ndarray:
                        return jnp.abs(eval_mixed_rd_relative(f_in))

                def eval_dh_rd(f_in: jnp.ndarray) -> jnp.ndarray:
                    return _value_and_derivative(eval_h_rd, f_in)[1]

                def eval_h_ins(f_in: jnp.ndarray) -> jnp.ndarray:
                    pn_part = _pn_amplitude_ratio(l, m, f_in, eta, delta, chi_s, chi_a)
                    f_ins_u = _unsqueeze_to(f_ins_a, f_in)
                    a_u = _unsqueeze_to(alpha_insp, f_in)
                    b_u = _unsqueeze_to(beta_insp, f_in)
                    g_u = _unsqueeze_to(gamma_insp, f_in)
                    f_ratio = f_in / f_ins_u
                    pseudo = (
                        a_u * jnp.power(f_ratio, 7.0 / 3.0)
                        + b_u * jnp.power(f_ratio, 8.0 / 3.0)
                        + g_u * jnp.power(f_ratio, 3.0)
                    )
                    return pn_part + pseudo

                def eval_dh_ins(f_in: jnp.ndarray) -> jnp.ndarray:
                    return _value_and_derivative(eval_h_ins, f_in)[1]

                # Intermediate amplitude collocation matching
                f_int1 = f_ins_a + (1.0 / 3.0) * (f_rd_a - f_ins_a)
                f_int2 = f_ins_a + (2.0 / 3.0) * (f_rd_a - f_ins_a)

                v_int1_fit = getattr(XHMParameterFits, f"amp_int1_{lm_str}")(eta, s1, s2, S)
                v_int2_fit = getattr(XHMParameterFits, f"amp_int2_{lm_str}")(eta, s1, s2, S)
                if m % 2 != 0:
                    v_int1_fit = jnp.abs(v_int1_fit)
                    v_int2_fit = jnp.abs(v_int2_fit)

                h_ins_at_cut = eval_h_ins(f_ins_a)
                dh_ins_at_cut = eval_dh_ins(f_ins_a)
                p_ins = jnp.power(PI * f_ins_a, 7.0 / 6.0) / jnp.maximum(h_ins_at_cut, 1e-12)
                dp_ins = jnp.power(PI * f_ins_a, 7.0 / 6.0) * (
                    (7.0 / 6.0) / (f_ins_a * jnp.maximum(h_ins_at_cut, 1e-12))
                    - dh_ins_at_cut / jnp.maximum(h_ins_at_cut**2, 1e-12)
                )

                p_int1 = 1.0 / jnp.maximum(v_int1_fit, 1e-12)
                p_int2 = 1.0 / jnp.maximum(v_int2_fit, 1e-12)

                h_rd_at_cut = eval_h_rd(f_rd_a)
                dh_rd_at_cut = eval_dh_rd(f_rd_a)
                p_rd = jnp.power(PI * f_rd_a, 7.0 / 6.0) / jnp.maximum(h_rd_at_cut, 1e-12)
                dp_rd = jnp.power(PI * f_rd_a, 7.0 / 6.0) * (
                    (7.0 / 6.0) / (f_rd_a * jnp.maximum(h_rd_at_cut, 1e-12))
                    - dh_rd_at_cut / jnp.maximum(h_rd_at_cut**2, 1e-12)
                )

                def make_vandermonde_rows(f_node: jnp.ndarray, is_deriv: bool = False) -> jnp.ndarray:
                    if not is_deriv:
                        return jnp.stack([
                            jnp.ones_like(f_node), f_node, f_node**2, f_node**3, f_node**4, f_node**5
                        ], axis=-1)
                    return jnp.stack([
                        jnp.zeros_like(f_node), jnp.ones_like(f_node), 2.0 * f_node, 3.0 * (f_node**2), 4.0 * (f_node**3), 5.0 * (f_node**4)
                    ], axis=-1)

                v_mat = jnp.stack([
                    make_vandermonde_rows(f_ins_a),
                    make_vandermonde_rows(f_ins_a, is_deriv=True),
                    make_vandermonde_rows(f_int1),
                    make_vandermonde_rows(f_int2),
                    make_vandermonde_rows(f_rd_a),
                    make_vandermonde_rows(f_rd_a, is_deriv=True),
                ], axis=-2)
                rhs_amp = jnp.stack([p_ins, dp_ins, p_int1, p_int2, p_rd, dp_rd], axis=-1)
                delta_amp = _safe_solve(v_mat, rhs_amp)

                def make_hermite_rows(f_node: jnp.ndarray, is_deriv: bool = False) -> jnp.ndarray:
                    if not is_deriv:
                        return jnp.stack([
                            jnp.ones_like(f_node), f_node, f_node**2, f_node**3
                        ], axis=-1)
                    return jnp.stack([
                        jnp.zeros_like(f_node), jnp.ones_like(f_node), 2.0 * f_node, 3.0 * (f_node**2)
                    ], axis=-1)

                v_mat3 = jnp.stack([
                    make_hermite_rows(f_ins_a),
                    make_hermite_rows(f_ins_a, is_deriv=True),
                    make_hermite_rows(f_rd_a),
                    make_hermite_rows(f_rd_a, is_deriv=True),
                ], axis=-2)
                rhs_amp3 = jnp.stack([p_ins, dp_ins, p_rd, dp_rd], axis=-1)
                delta_amp3 = _safe_solve(v_mat3, rhs_amp3)

                v_mat4 = jnp.stack([
                    make_hermite_rows(f_ins_a),
                    make_hermite_rows(f_int1),
                    make_hermite_rows(f_int2),
                    make_hermite_rows(f_rd_a),
                ], axis=-2)
                rhs_amp4 = jnp.stack([p_ins, p_int1, p_int2, p_rd], axis=-1)
                delta_amp4 = _safe_solve(v_mat4, rhs_amp4)

                def make_quartic_value_row(f_node: jnp.ndarray) -> jnp.ndarray:
                    return jnp.stack([
                        jnp.ones_like(f_node), f_node, f_node**2, f_node**3, f_node**4,
                    ], axis=-1)

                def make_quartic_deriv_row(f_node: jnp.ndarray) -> jnp.ndarray:
                    return jnp.stack([
                        jnp.zeros_like(f_node), jnp.ones_like(f_node),
                        2.0 * f_node, 3.0 * f_node**2, 4.0 * f_node**3,
                    ], axis=-1)

                v_mat5_1 = jnp.stack([
                    make_quartic_value_row(f_ins_a), make_quartic_deriv_row(f_ins_a),
                    make_quartic_value_row(f_int1), make_quartic_value_row(f_rd_a),
                    make_quartic_deriv_row(f_rd_a),
                ], axis=-2)
                v_mat5_2 = jnp.stack([
                    make_quartic_value_row(f_ins_a), make_quartic_deriv_row(f_ins_a),
                    make_quartic_value_row(f_int2), make_quartic_value_row(f_rd_a),
                    make_quartic_deriv_row(f_rd_a),
                ], axis=-2)
                delta_amp5_1 = _safe_solve(
                    v_mat5_1, jnp.stack([p_ins, dp_ins, p_int1, p_rd, dp_rd], axis=-1),
                )
                delta_amp5_2 = _safe_solve(
                    v_mat5_2, jnp.stack([p_ins, dp_ins, p_int2, p_rd, dp_rd], axis=-1),
                )

                def make_value_row(f_node: jnp.ndarray) -> jnp.ndarray:
                    return jnp.stack([
                        jnp.ones_like(f_node), f_node, f_node**2, f_node**3,
                    ], axis=-1)

                v_mat2_1 = jnp.stack([
                    make_value_row(f_ins_a), make_value_row(f_int1), make_value_row(f_rd_a),
                ], axis=-2)[..., :3, :3]
                v_mat2_2 = jnp.stack([
                    make_value_row(f_ins_a), make_value_row(f_int2), make_value_row(f_rd_a),
                ], axis=-2)[..., :3, :3]
                delta_amp2_1 = _safe_solve(v_mat2_1, jnp.stack([p_ins, p_int1, p_rd], axis=-1))
                delta_amp2_2 = _safe_solve(v_mat2_2, jnp.stack([p_ins, p_int2, p_rd], axis=-1))
                v_mat1 = jnp.stack([
                    jnp.ones_like(f_ins_a), f_ins_a,
                    jnp.ones_like(f_rd_a), f_rd_a,
                ], axis=-1).reshape(f_ins_a.shape + (2, 2))
                delta_amp1 = _safe_solve(v_mat1, jnp.stack([p_ins, p_rd], axis=-1))

                h_int1_at_cut = v_int1_fit
                h_int2_at_cut = v_int2_fit

                alive1, alive2, remove_derivatives = _ordered_amplitude_vetoes(
                    (l, m), m1 / m2, eta, s1, s2,
                    1.0 / jnp.maximum(p_ins, 1e-12),
                    h_int1_at_cut, h_int2_at_cut,
                    1.0 / jnp.maximum(p_rd, 1e-12),
                )

                def pad_coefficients(coefficients: jnp.ndarray, count: int) -> jnp.ndarray:
                    diff = count - coefficients.shape[-1]
                    if diff <= 0:
                        return coefficients
                    pad_width = [(0, 0)] * coefficients.ndim
                    pad_width[-1] = (0, diff)
                    return jnp.pad(coefficients, pad_width)

                full = jnp.expand_dims(alive1 & alive2, -1)
                one1 = jnp.expand_dims(alive1 & ~alive2, -1)
                one2 = jnp.expand_dims(~alive1 & alive2, -1)
                active_coefficients = jnp.where(full, delta_amp, pad_coefficients(delta_amp3, 6))
                active_coefficients = jnp.where(one1, pad_coefficients(delta_amp5_1, 6), active_coefficients)
                active_coefficients = jnp.where(one2, pad_coefficients(delta_amp5_2, 6), active_coefficients)

                value_coefficients = jnp.where(full, delta_amp4, pad_coefficients(delta_amp1, 4))
                value_coefficients = jnp.where(one1, pad_coefficients(delta_amp2_1, 4), value_coefficients)
                value_coefficients = jnp.where(one2, pad_coefficients(delta_amp2_2, 4), value_coefficients)
                active_coefficients = jnp.where(
                    jnp.expand_dims(remove_derivatives, -1), pad_coefficients(value_coefficients, 6), active_coefficients,
                )
                use_pole_fallback = polynomial_has_nonpositive(
                    jax.lax.stop_gradient(active_coefficients),
                    jax.lax.stop_gradient(f_ins_a),
                    jax.lax.stop_gradient(f_rd_a),
                )
                fallback_bad = polynomial_has_nonpositive(
                    jax.lax.stop_gradient(value_coefficients),
                    jax.lax.stop_gradient(f_ins_a),
                    jax.lax.stop_gradient(f_rd_a),
                )
                selected_coefficients = jnp.where(
                    jnp.expand_dims(use_pole_fallback, -1), pad_coefficients(value_coefficients, 6), active_coefficients,
                )
                selected_coefficients = jnp.where(
                    jnp.expand_dims(use_pole_fallback & fallback_bad, -1),
                    pad_coefficients(delta_amp1, 6), selected_coefficients,
                )

                p_poly_active = jnp.zeros_like(mf_grid)
                for power in range(5, -1, -1):
                    p_poly_active = p_poly_active * mf_grid + _unsqueeze_to(selected_coefficients[..., power], mf_grid)
                p_poly_active = jnp.maximum(p_poly_active, 1e-12)
                f_ins_a_u = _unsqueeze_to(f_ins_a, mf_grid)
                f_rd_a_u = _unsqueeze_to(f_rd_a, mf_grid)
                h_ins_grid = eval_h_ins(mf_grid)
                h_int_grid = jnp.power(PI * mf_grid, 7.0 / 6.0) / p_poly_active
                h_rd_grid = eval_h_rd(mf_grid)
                h_ratio_grid = jnp.where(
                    mf_grid <= f_ins_a_u,
                    h_ins_grid,
                    jnp.where(
                        mf_grid <= f_rd_a_u,
                        h_int_grid,
                        h_rd_grid,
                    ),
                )
                if m % 2 != 0:
                    zero_odd_mask = _unsqueeze_to((delta == 0.0) & (chi_a == 0.0), mf_grid)
                    h_ratio_grid = jnp.where(zero_odd_mask, jnp.zeros_like(h_ratio_grid), h_ratio_grid)

                amp_lm_phys = mode_norm * a0_phys * h_ratio_grid

                # -------------------------------------------------------------
                # 2. Phase Reconstruction
                # -------------------------------------------------------------
                beta_eta = 1.0 + 0.001 * (0.25 / jnp.maximum(eta, 1e-6) - 1.0)
                f_start = beta_eta * f_ins_p
                f_end = f_ring_lm if (l, m) != (3, 2) else (f_ring["22"] - 0.5 * f_damp["22"])

                c_pts = [
                    f_start,
                    0.25 * ((math.sqrt(3.0) + 2.0) * f_start - (math.sqrt(3.0) - 2.0) * f_end),
                    0.25 * (f_end + 3.0 * f_start),
                    0.50 * (f_end + f_start),
                    0.25 * (3.0 * f_end + f_start),
                    0.125 * (7.0 * f_end + f_start),
                ]

                if (l, m) != (3, 2):
                    # Modes without mixing: (2,1), (3,3), (4,4)
                    f_p1 = getattr(XHMParameterFits, f"phase_p1_{lm_str}")(eta, s1, s2, S)
                    f_p2 = getattr(XHMParameterFits, f"phase_p2_{lm_str}")(eta, s1, s2, S)
                    f_p3 = getattr(XHMParameterFits, f"phase_p3_{lm_str}")(eta, s1, s2, S)
                    f_p4 = getattr(XHMParameterFits, f"phase_p4_{lm_str}")(eta, s1, s2, S)
                    f_p6 = getattr(XHMParameterFits, f"phase_p6_{lm_str}")(eta, s1, s2, S)

                    f_nodes = jnp.stack([c_pts[0], c_pts[1], c_pts[2], c_pts[3], c_pts[5]], axis=-1)
                    f_damp_u_nodes = jnp.expand_dims(f_damp_lm, -1) if f_damp_lm.ndim > 0 else f_damp_lm
                    f_ring_u_nodes = jnp.expand_dims(f_ring_lm, -1) if f_ring_lm.ndim > 0 else f_ring_lm
                    lorentz_nodes = f_damp_u_nodes / ((f_damp_u_nodes**2) + (f_nodes - f_ring_u_nodes)**2)
                    m_phase = jnp.stack([
                        lorentz_nodes,
                        jnp.ones_like(f_nodes),
                        1.0 / f_nodes,
                        1.0 / (f_nodes**2),
                        1.0 / (f_nodes**4),
                    ], axis=-1)
                    rhs_phase = jnp.stack([f_p1, f_p2, f_p3, f_p4, f_p6], axis=-1)
                    a_int = _safe_solve(m_phase, rhs_phase)

                    a_lam = a_int[..., 0]
                    a_0 = a_int[..., 1]
                    a_1 = a_int[..., 2]
                    a_2 = a_int[..., 3]
                    a_4 = a_int[..., 4]

                    def eval_dphi_int(f_in: jnp.ndarray) -> jnp.ndarray:
                        damp_u = _unsqueeze_to(f_damp_lm, f_in)
                        ring_u = _unsqueeze_to(f_ring_lm, f_in)
                        a_lam_u = _unsqueeze_to(a_lam, f_in)
                        a_0_u = _unsqueeze_to(a_0, f_in)
                        a_1_u = _unsqueeze_to(a_1, f_in)
                        a_2_u = _unsqueeze_to(a_2, f_in)
                        a_4_u = _unsqueeze_to(a_4, f_in)
                        lor = damp_u / ((damp_u**2) + (f_in - ring_u)**2)
                        return (
                            a_lam_u * lor
                            + a_0_u
                            + a_1_u / f_in
                            + a_2_u / (f_in**2)
                            + a_4_u / (f_in**4)
                        )

                    def eval_phi_int(f_in: jnp.ndarray) -> jnp.ndarray:
                        damp_u = _unsqueeze_to(f_damp_lm, f_in)
                        ring_u = _unsqueeze_to(f_ring_lm, f_in)
                        a_lam_u = _unsqueeze_to(a_lam, f_in)
                        a_0_u = _unsqueeze_to(a_0, f_in)
                        a_1_u = _unsqueeze_to(a_1, f_in)
                        a_2_u = _unsqueeze_to(a_2, f_in)
                        a_4_u = _unsqueeze_to(a_4, f_in)
                        return (
                            a_lam_u * jnp.arctan((f_in - ring_u) / damp_u)
                            + a_0_u * f_in
                            + a_1_u * jnp.log(jnp.maximum(f_in, 1e-12))
                            - a_2_u / f_in
                            - (1.0 / 3.0) * a_4_u / (f_in**3)
                        )

                    # Ringdown phase connection at f_rd_p (Sovereign SXS NR calibrated)
                    alpha_2_lm, alpha_lambda_lm = eval_native_phase_params(l, m, eta)

                    dphi_rd_val = eval_dphi_int(f_rd_p)
                    lor_rd_at_cut = f_damp_lm / ((f_damp_lm**2) + (f_rd_p - f_ring_lm)**2)
                    dt_rd = dphi_rd_val - (
                        alpha_2_lm * (f_ring_lm**2) / (f_rd_p**2) + alpha_lambda_lm * lor_rd_at_cut
                    )
                    phi_int_at_cut = eval_phi_int(f_rd_p)
                    phi_rd_base = (
                        -alpha_2_lm * (f_ring_lm**2) / f_rd_p
                        + alpha_lambda_lm * jnp.arctan((f_rd_p - f_ring_lm) / f_damp_lm)
                        + dt_rd * f_rd_p
                    )
                    dphi_rd_shift = phi_int_at_cut - phi_rd_base

                    def eval_phi_rd(f_in: jnp.ndarray) -> jnp.ndarray:
                        damp_u = _unsqueeze_to(f_damp_lm, f_in)
                        ring_u = _unsqueeze_to(f_ring_lm, f_in)
                        a2_u = _unsqueeze_to(alpha_2_lm, f_in)
                        alam_u = _unsqueeze_to(alpha_lambda_lm, f_in)
                        dt_u = _unsqueeze_to(dt_rd, f_in)
                        shift_u = _unsqueeze_to(dphi_rd_shift, f_in)
                        return (
                            -a2_u * (ring_u**2) / f_in
                            + alam_u * jnp.arctan((f_in - ring_u) / damp_u)
                            + dt_u * f_in
                            + shift_u
                        )

                    delta_target = (l % 2) * PI - (3.0 * PI / 8.0) * (m_flt - 2.0)
                    if l == 2 and m == 1:
                        ff = 0.008
                        a21_pn = (
                            -16.0 * delta * eta * ff * (math.pi**1.5) / (3.0 * math.sqrt(5.0))
                            + 4.0 * (2.0**(1.0/3.0)) * ((s1 - s2) + delta * (s1 + s2)) * eta * (ff**(4.0/3.0)) * (math.pi**(11.0/6.0)) / math.sqrt(5.0)
                            + 2.0 * (2.0**(2.0/3.0)) * eta * (306.0 * delta - 360.0 * delta * eta) * (ff**(5.0/3.0)) * (math.pi**(13.0/6.0)) / (189.0 * math.sqrt(5.0))
                        )
                        delta_target = delta_target + jnp.where(a21_pn >= 0, PI, 0.0)

                    # Evaluate carrier phase on scaled frequency grid (2/m) * f
                    f_scaled_grid = (2.0 / m_flt) * f_eval
                    if carrier_state is not None:
                        phi_22_scaled = carrier_state.eval_phase(f_scaled_grid)
                    else:
                        carrier_eval_fn = getattr(base_carrier, "evaluate", base_carrier)
                        _, _, phi_22_scaled = carrier_eval_fn(
                            f=f_scaled_grid,
                            mass1=m1,
                            mass2=m2,
                            chi1=s1,
                            chi2=s2,
                            distance=dist,
                            inclination=0.0,
                            phic=0.0,
                            tc=0.0,
                            f_ref=f_ref,
                            phasing_convention=phasing_convention,
                            reference_frame=reference_frame,
                            return_phase=True,
                            _final_spin_override=final_spin_override,
                        )
                    delta_target_u = _unsqueeze_to(delta_target, mf_grid)
                    carrier_p_at_ins, carrier_dp_at_ins = eval_carrier_phi_and_deriv((2.0 / m_flt) * f_ins_p)
                    phi_ins_grid = (m_flt / 2.0) * phi_22_scaled + delta_target_u
                    phi_ins_at_ins = (m_flt / 2.0) * carrier_p_at_ins + delta_target
                    dphi_int_shift = phi_ins_at_ins - eval_phi_int(f_ins_p)
                    dphi_int_shift_u = _unsqueeze_to(dphi_int_shift, mf_grid)

                    phi_int_grid = eval_phi_int(mf_grid) + dphi_int_shift_u
                    phi_rd_grid = eval_phi_rd(mf_grid) + dphi_int_shift_u

                    f_ins_p_u = _unsqueeze_to(f_ins_p, mf_grid)
                    f_rd_p_u = _unsqueeze_to(f_rd_p, mf_grid)
                    psi_lm = jnp.where(
                        mf_grid <= f_ins_p_u,
                        phi_ins_grid,
                        jnp.where(
                            mf_grid <= f_rd_p_u,
                            phi_int_grid,
                            phi_rd_grid,
                        ),
                    )
                    exp_neg_i_psi = jnp.cos(psi_lm) - 1j * jnp.sin(psi_lm)
                    h_mode = amp_lm_phys.astype(exp_neg_i_psi.dtype) * exp_neg_i_psi

                else:
                    # ---------------------------------------------------------
                    # Mode (3, 2): Spherical-Spheroidal Mode Mixing
                    # ---------------------------------------------------------
                    f_p1_32 = XHMParameterFits.phase_p1_32(eta, s1, s2, S)
                    f_p2_32 = XHMParameterFits.phase_p2_32(eta, s1, s2, S)
                    f_p3_32 = XHMParameterFits.phase_p3_32(eta, s1, s2, S)
                    f_p4_32 = XHMParameterFits.phase_p4_32(eta, s1, s2, S)

                    f_nodes_int_32 = jnp.stack(c_pts[:4], axis=-1)

                    def lor_32(f_in: jnp.ndarray) -> jnp.ndarray:
                        damp_u = _unsqueeze_to(f_damp["32"], f_in)
                        ring_u = _unsqueeze_to(f_ring["32"], f_in)
                        return damp_u / ((damp_u**2) + (f_in - ring_u)**2)

                    def dlor_32(f_in: jnp.ndarray) -> jnp.ndarray:
                        damp_u = _unsqueeze_to(f_damp["32"], f_in)
                        ring_u = _unsqueeze_to(f_ring["32"], f_in)
                        return -2.0 * damp_u * (f_in - ring_u) / (((damp_u**2) + (f_in - ring_u)**2)**2)

                    dphi_rd_at_join, d2phi_rd_at_join = _value_and_derivative(
                        lambda x: _value_and_derivative(eval_phi_32_rd, x)[1], f_rd_p,
                    )

                    rows_32 = []
                    for i_pt in range(4):
                        fn_i = f_nodes_int_32[..., i_pt]
                        rows_32.append(jnp.stack([
                            lor_32(fn_i), jnp.ones_like(fn_i), 1.0 / fn_i, 1.0 / (fn_i**2), 1.0 / (fn_i**3), 1.0 / (fn_i**4)
                        ], axis=-1))
                    rows_32.append(jnp.stack([
                        lor_32(f_rd_p), jnp.ones_like(f_rd_p), 1.0 / f_rd_p, 1.0 / (f_rd_p**2), 1.0 / (f_rd_p**3), 1.0 / (f_rd_p**4)
                    ], axis=-1))
                    rows_32.append(jnp.stack([
                        dlor_32(f_rd_p), jnp.zeros_like(f_rd_p), -1.0 / (f_rd_p**2), -2.0 / (f_rd_p**3), -3.0 / (f_rd_p**4), -4.0 / (f_rd_p**5)
                    ], axis=-1))

                    m_int_32 = jnp.stack(rows_32, axis=-2)
                    rhs_int_32 = jnp.stack([f_p1_32, f_p2_32, f_p3_32, f_p4_32, dphi_rd_at_join, d2phi_rd_at_join], axis=-1)
                    sol_int_32 = _safe_solve(m_int_32, rhs_int_32)

                    a_lam_32 = sol_int_32[..., 0]
                    a_0_32 = sol_int_32[..., 1]
                    a_1_32 = sol_int_32[..., 2]
                    a_2_32 = sol_int_32[..., 3]
                    a_3_32 = sol_int_32[..., 4]
                    a_4_32 = sol_int_32[..., 5]

                    def eval_dphi_int_32(f_in: jnp.ndarray) -> jnp.ndarray:
                        a_lam_u = _unsqueeze_to(a_lam_32, f_in)
                        a_0_u = _unsqueeze_to(a_0_32, f_in)
                        a_1_u = _unsqueeze_to(a_1_32, f_in)
                        a_2_u = _unsqueeze_to(a_2_32, f_in)
                        a_3_u = _unsqueeze_to(a_3_32, f_in)
                        a_4_u = _unsqueeze_to(a_4_32, f_in)
                        return (
                            a_lam_u * lor_32(f_in)
                            + a_0_u
                            + a_1_u / f_in
                            + a_2_u / (f_in**2)
                            + a_3_u / (f_in**3)
                            + a_4_u / (f_in**4)
                        )

                    def eval_phi_int_32(f_in: jnp.ndarray) -> jnp.ndarray:
                        damp_u = _unsqueeze_to(f_damp["32"], f_in)
                        ring_u = _unsqueeze_to(f_ring["32"], f_in)
                        a_lam_u = _unsqueeze_to(a_lam_32, f_in)
                        a_0_u = _unsqueeze_to(a_0_32, f_in)
                        a_1_u = _unsqueeze_to(a_1_32, f_in)
                        a_2_u = _unsqueeze_to(a_2_32, f_in)
                        a_3_u = _unsqueeze_to(a_3_32, f_in)
                        a_4_u = _unsqueeze_to(a_4_32, f_in)
                        return (
                            a_lam_u * jnp.arctan((f_in - ring_u) / damp_u)
                            + a_0_u * f_in
                            + a_1_u * jnp.log(jnp.maximum(f_in, 1e-12))
                            - a_2_u / f_in
                            - 0.5 * a_3_u / (f_in**2)
                            - (1.0 / 3.0) * a_4_u / (f_in**3)
                        )

                    c_int_32 = eval_phi_32_rd(f_rd_p) - eval_phi_int_32(f_rd_p)

                    delta_target_32 = PI
                    carrier_p_at_ins, carrier_dp_at_ins = eval_carrier_phi_and_deriv(f_ins_p)
                    phi_ins_32_grid = phi_22_carrier + delta_target_32
                    phi_ins_at_ins_32 = carrier_p_at_ins + delta_target_32
                    dphi_int_shift_32 = phi_ins_at_ins_32 - (eval_phi_int_32(f_ins_p) + c_int_32)
                    dphi_int_shift_32_u = _unsqueeze_to(dphi_int_shift_32, mf_grid)
                    c_int_32_u = _unsqueeze_to(c_int_32, mf_grid)

                    phi_int_32_grid = eval_phi_int_32(mf_grid) + c_int_32_u + dphi_int_shift_32_u
                    phi_rd_32_grid = eval_phi_32_rd(mf_grid) + dphi_int_shift_32_u

                    f_ins_p_u = _unsqueeze_to(f_ins_p, mf_grid)
                    f_rd_p_u = _unsqueeze_to(f_rd_p, mf_grid)
                    phi_32_imr = jnp.where(
                        mf_grid <= f_ins_p_u,
                        phi_ins_32_grid,
                        jnp.where(
                            mf_grid <= f_rd_p_u,
                            phi_int_32_grid,
                            phi_rd_32_grid,
                        ),
                    )

                    h_mode = amp_lm_phys * jnp.exp(-1j * phi_32_imr)

            extrinsic_phase = 2.0 * PI * f_eval * tc_eval - float(m) * phic_eval
            h_mode = h_mode * (jnp.cos(extrinsic_phase) - 1j * jnp.sin(extrinsic_phase))

            if m % 2 != 0:
                h_mode = jnp.where(_unsqueeze_to(swap_mask, h_mode), -h_mode, h_mode)

            d_plus, d_cross = _swsh_projection_coefficients(l, m, inc_eval, phi0_eval)
            hp_lm = d_plus * h_mode
            hc_lm = d_cross * h_mode

            h_plus_total = h_plus_total + hp_lm
            h_cross_total = h_cross_total + hc_lm

            if return_modes:
                mode_dict[(l, m)] = (hp_lm, hc_lm)
            elif return_unprojected_modes:
                mode_dict[(l, m)] = h_mode
                sign_l = 1.0 if l % 2 == 0 else -1.0
                mode_dict[(l, -m)] = sign_l * jnp.conj(h_mode)

        valid = f_eval > 0.0
        if f_lower is not None:
            valid = valid & (f_eval >= float(f_lower))
        h_plus_total = jnp.where(valid, h_plus_total, jnp.zeros_like(h_plus_total))
        h_cross_total = jnp.where(valid, h_cross_total, jnp.zeros_like(h_cross_total))

        if return_modes or return_unprojected_modes:
            return h_plus_total, h_cross_total, mode_dict
        return h_plus_total, h_cross_total


__all__ = [
    "IMRPhenomXHM",
    "_pn_amplitude_ratio",
    "_published_parameter_amplitude_veto",
    "_ordered_amplitude_vetoes",
]

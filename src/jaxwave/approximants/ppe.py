"""JAX ppE inspiral amplitude/phase and massive-graviton FD overlays.

Source equations: Yunes & Pretorius, PRD 80 122003 (2009), arXiv:0909.3328,
Eq. (1); Mirshekari, Yunes & Will, PRD 85 024041 (2012), arXiv:1110.2720,
Eqs. (14)-(16), (28)-(31). The adapter follows the carrier's negative Fourier
phase sign; it does not implement the papers' merger/ringdown branches.
"""

from typing import Any, Dict, Optional, Tuple, Union
import jax
import jax.numpy as jnp

from diffgw_core.constants import (
    C_SI,
    MSUN_SEC,
    MPC_SI,
    PI,
)
from jaxwave.conversions import as_array
from diffgw_core.ppe_cosmology import (
    D0_QUADRATURE_NODES, D0_QUADRATURE_WEIGHTS, OMEGA_M, OMEGA_LAMBDA,
)
from jaxwave.approximants.imrphenomd import IMRPhenomD


# Standard PN order to ppE exponent mapping: b_ppE = 2 * PN - 5
PPE_PN_MAPPING: Dict[Union[int, float], int] = {
    -1.0: -7,  # -1PN dipole radiation
    -0.5: -6,  # -0.5PN
    0.0: -5,   # 0PN Newtonian quadrupole
    0.5: -4,   # 0.5PN
    1.0: -3,   # 1PN
    1.5: -2,   # 1.5PN
    2.0: -1,   # 2PN
    2.5: 0,    # 2.5PN
    3.0: 1,    # 3PN
    3.5: 2,    # 3.5PN
}


def ppe_phase_shift(
    f: jnp.ndarray,
    chirp_mass: Union[float, jnp.ndarray],
    beta_ppe: Union[float, jnp.ndarray] = 0.0,
    b_ppe: Union[int, float, jnp.ndarray] = -2,
) -> jnp.ndarray:
    """Compute ppE frequency-dependent phase deformation delta Psi_ppE(f) in JAX.

    Literature: Yunes & Pretorius (2009) Eq. (1), u = pi * Mc * f.

    Formula:
        delta Psi_ppE(f) = beta_ppE * (pi * Mc * f)^(b_ppE / 3)
    Here b_ppE is the project's velocity exponent, three times the paper's b.

    Args:
        f: Frequency array in Hz.
        chirp_mass: Chirp mass in solar masses, in the frame of supplied f.
        beta_ppe: ppE phase deformation magnitude (0 recovers pure GR).
        b_ppe: ppE phase exponent (-7 for -1PN dipole, -2 for 1.5PN, 0 for 2.5PN).

    Returns:
        Array of phase shifts in radians matching shape of f.
    """
    f_t = as_array(f, dtype=jnp.float64)
    m = as_array(chirp_mass, dtype=f_t.dtype)
    beta = as_array(beta_ppe, dtype=f_t.dtype)
    b = as_array(b_ppe, dtype=f_t.dtype)

    m_sec = m * MSUN_SEC
    # Dimensionless velocity formed from the paper's chirp-mass u.
    safe_f = jnp.maximum(f_t, 1e-15)
    v = jnp.power(PI * m_sec * safe_f, 1.0 / 3.0)

    # delta Psi = beta * v^b
    delta_psi = beta * jnp.power(v, b)
    return delta_psi


def ppe_amplitude_factor(
    f: jnp.ndarray,
    chirp_mass: Union[float, jnp.ndarray],
    alpha_ppe: Union[float, jnp.ndarray] = 0.0,
    a_ppe: Union[int, float, jnp.ndarray] = 0.0,
) -> jnp.ndarray:
    """Eq. (1) inspiral factor ``1 + alpha_ppe*(pi*Mc*f)**a_ppe``.

    ``a_ppe`` is the paper's exponent of ``u=pi*Mc*f`` (unlike the
    existing phase helper's velocity exponent ``b_ppe``). ``Mc`` is in
    solar masses in the frame of ``f``. The paper scope is positive Hz.
    """
    f_t = as_array(f, dtype=jnp.float64)
    m = as_array(chirp_mass, dtype=f_t.dtype)
    alpha = as_array(alpha_ppe, dtype=f_t.dtype)
    a = as_array(a_ppe, dtype=f_t.dtype)
    u = PI * m * MSUN_SEC * jnp.maximum(f_t, 1e-15)
    return 1.0 + alpha * jnp.power(u, a)


def modified_dispersion_phase_shift(
    f: jnp.ndarray,
    distance: Union[float, jnp.ndarray] = 100.0,
    lambda_g: Union[float, jnp.ndarray] = float("inf"),
    z: Union[float, jnp.ndarray] = 0.0,
    d_alpha: Optional[Union[float, jnp.ndarray]] = None,
) -> jnp.ndarray:
    """Compute frequency phase shift due to modified GW dispersion relation (massive graviton) in JAX.

    Literature: Mirshekari, Yunes & Will (2012) Eqs. (14)-(16), (28)-(31).

    Massive-graviton term, with the paper's separate Lorentz-violating
    A_alpha contribution set to zero and lambda_g = h / (m_g * c):
        delta Psi_disp(f) = - pi * D_0 * c / (lambda_g^2 * (1 + z) * f)

    Args:
        f: Frequency array in Hz.
        distance: Luminosity distance D_L in Mpc.
        lambda_g: Graviton Compton wavelength in meters (inf recovers pure GR).
        z: Cosmological redshift of source (default 0).
        d_alpha: Optional cosmological dispersion distance in meters.
            If None, D_0 is derived from supplied D_L via the flat
            matter/Lambda cosmological integrals of Eqs. (15)-(16), using
            Omega_m=0.3 and Omega_Lambda=0.7.

    Returns:
        Array of dispersion phase shifts in radians.
    """
    f_t = as_array(f, dtype=jnp.float64)
    dist_mpc = as_array(distance, dtype=f_t.dtype)
    l_g = as_array(lambda_g, dtype=f_t.dtype)
    redshift = as_array(z, dtype=f_t.dtype)

    is_gr = jnp.isinf(l_g)

    if d_alpha is None:
        nodes = jnp.asarray(D0_QUADRATURE_NODES, dtype=f_t.dtype)
        weights = jnp.asarray(D0_QUADRATURE_WEIGHTS, dtype=f_t.dtype)
        zp = redshift[..., None] * (nodes + 1.0) / 2.0
        e = jnp.sqrt(OMEGA_M * (1.0 + zp) ** 3 + OMEGA_LAMBDA)
        ratio = jnp.sum(weights * (1.0 + zp) ** -2 / e, axis=-1) / jnp.sum(weights / e, axis=-1)
        d0 = dist_mpc * MPC_SI * ratio
    else:
        d0 = as_array(d_alpha, dtype=f_t.dtype)

    safe_f = jnp.maximum(f_t, 1e-15)
    l_g_safe = jnp.where(is_gr, 1.0, l_g)

    coeff = -PI * d0 * C_SI / (l_g_safe * l_g_safe * (1.0 + redshift))
    shift = coeff / safe_f
    return jnp.where(is_gr, jnp.zeros_like(shift), shift)


@jax.tree_util.register_pytree_node_class
class ParameterizedPostEinstein:
    """Wrapper around frequency-domain approximants applying ppE and modified dispersion shifts in JAX.

    Supported underlying approximants:
    - TaylorF2 (SPA PN model)
    - IMRPhenomD (aligned-spin phenomenological model)
    - IMRPhenomXAS (4th-gen aligned-spin phenomenological model)
    """

    def __init__(self, approximant: Optional[Any] = None):
        if approximant is None:
            self.approximant = IMRPhenomD()
        else:
            self.approximant = approximant

    def tree_flatten(self):
        return (self.approximant,), ()

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls.__new__(cls)
        obj.approximant = children[0]
        return obj

    def forward(
        self,
        f: jnp.ndarray,
        mass1: Union[float, jnp.ndarray],
        mass2: Union[float, jnp.ndarray],
        beta_ppe: Union[float, jnp.ndarray] = 0.0,
        b_ppe: Union[int, float, jnp.ndarray] = -2,
        lambda_g: Union[float, jnp.ndarray] = float("inf"),
        distance: Union[float, jnp.ndarray] = 100.0,
        redshift: Union[float, jnp.ndarray] = 0.0,
        d_alpha: Optional[Union[float, jnp.ndarray]] = None,
        alpha_ppe: Union[float, jnp.ndarray] = 0.0,
        a_ppe: Union[int, float, jnp.ndarray] = 0.0,
        **waveform_kwargs,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Generate ppE-deformed polarizations h_plus and h_cross in JAX.

        Args:
            f: Frequency array in Hz.
            mass1: Primary mass in solar masses.
            mass2: Secondary mass in solar masses.
            beta_ppe: ppE phase deformation parameter (default 0 for GR).
            b_ppe: ppE PN order exponent (-7 for -1PN, -2 for 1.5PN, 0 for 2.5PN).
            lambda_g: Graviton Compton wavelength in meters (default inf for GR).
            distance: Luminosity distance in Mpc.
            redshift: Cosmological redshift.
            d_alpha: Optional independently supplied D_0 in meters.
            alpha_ppe: Dimensionless inspiral amplitude coefficient (zero for GR).
            a_ppe: Paper exponent of u=pi*Mc*f in the inspiral amplitude.
            **waveform_kwargs: Additional parameters passed to the underlying approximant
                (e.g., chi1, chi2, inclination, phic, tc, f_ref).

        Returns:
            Tuple of deformed complex polarizations (h_plus, h_cross).
        """
        # 1. Generate base General Relativity polarizations
        hp_gr, hc_gr = self.approximant(
            f=f,
            mass1=mass1,
            mass2=mass2,
            distance=distance,
            **waveform_kwargs,
        )

        f_t = as_array(f, dtype=jnp.float64)
        m1 = as_array(mass1, dtype=f_t.dtype)
        m2 = as_array(mass2, dtype=f_t.dtype)
        chirp_m = (m1 * m2) ** (3.0 / 5.0) / (m1 + m2) ** (1.0 / 5.0)

        # 2. Compute non-GR phase shifts
        delta_psi_ppe = ppe_phase_shift(f_t, chirp_m, beta_ppe=beta_ppe, b_ppe=b_ppe)
        delta_psi_disp = modified_dispersion_phase_shift(
            f_t, distance=distance, lambda_g=lambda_g, z=redshift, d_alpha=d_alpha
        )

        total_shift = delta_psi_ppe + delta_psi_disp
        amplitude_factor = ppe_amplitude_factor(f_t, chirp_m, alpha_ppe, a_ppe)

        # 3. Apply stationary phase deformation: exp(-i * delta_Psi)
        if hp_gr.ndim > total_shift.ndim:
            total_shift = jnp.expand_dims(total_shift, 0)
            amplitude_factor = jnp.expand_dims(amplitude_factor, 0)

        cos_shift = jnp.cos(total_shift)
        sin_shift = jnp.sin(total_shift)
        exp_factor = cos_shift - 1j * sin_shift

        hp_ppe = hp_gr * amplitude_factor * exp_factor
        hc_ppe = hc_gr * amplitude_factor * exp_factor

        return hp_ppe, hc_ppe

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

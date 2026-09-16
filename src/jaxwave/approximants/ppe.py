"""Parameterized Post-Einsteinian (ppE) and Modified Dispersion Gravitational Waveforms in JAX.

Strict Clean-Room Implementation based exclusively on published literature:
- Yunes & Pretorius (2009), "Dynamical Systems Approach to Runaway Instabilities in Black Hole-Torus Systems",
  Phys. Rev. D 80, 122003 (arXiv:0909.3328)
- Mirshekari, Yunes, Will (2012), "Constraining Generic Post-Einsteinian Extensions of Gravity with Gravitational Waves",
  Phys. Rev. D 85, 024041 (arXiv:1110.2720)
- Chatziioannou, Yunes, Cornish (2012), "Model-Independent Test of General Relativity: An Extended Parameterized Post-Einsteinian Framework",
  Phys. Rev. D 86, 022004 (arXiv:1204.2585)
- Will (1998), "Bounding the mass of the graviton using gravitational-wave observations of inspiralling compact binaries",
  Phys. Rev. D 57, 2061 (arXiv:gr-qc/9709011)
"""

from typing import Any, Callable, Dict, Optional, Tuple, Union
import math
import jax
import jax.numpy as jnp

from diffgw_core.constants import (
    C_SI,
    MSUN_SEC,
    MPC_SI,
    PI,
)
from jaxwave.conversions import as_array
from jaxwave.approximants.taylorf2 import TaylorF2
from jaxwave.approximants.imrphenomd import IMRPhenomD
from jaxwave.approximants.imrphenom_xas import IMRPhenomXAS


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
    total_mass: Union[float, jnp.ndarray],
    beta_ppe: Union[float, jnp.ndarray] = 0.0,
    b_ppe: Union[int, float, jnp.ndarray] = -2,
) -> jnp.ndarray:
    """Compute ppE frequency-dependent phase deformation delta Psi_ppE(f) in JAX.

    Literature: Yunes & Pretorius (2009) Eq. 1; Mirshekari et al. (2012) Eq. 20.

    Formula:
        delta Psi_ppE(f) = beta_ppE * (pi * M * f)^(b_ppE / 3)
    where M is total mass in geometric seconds: M_sec = (m1 + m2) * MSUN_SEC.

    Args:
        f: Frequency array in Hz.
        total_mass: Total binary mass M = m1 + m2 in solar masses.
        beta_ppe: ppE phase deformation magnitude (0 recovers pure GR).
        b_ppe: ppE phase exponent (-7 for -1PN dipole, -2 for 1.5PN, 0 for 2.5PN).

    Returns:
        Array of phase shifts in radians matching shape of f.
    """
    f_t = as_array(f, dtype=jnp.float64)
    m = as_array(total_mass, dtype=f_t.dtype)
    beta = as_array(beta_ppe, dtype=f_t.dtype)
    b = as_array(b_ppe, dtype=f_t.dtype)

    m_sec = m * MSUN_SEC
    # Dimensionless PN velocity v = (pi * M * f)^(1/3)
    safe_f = jnp.maximum(f_t, 1e-15)
    v = jnp.power(PI * m_sec * safe_f, 1.0 / 3.0)

    # delta Psi = beta * v^b
    delta_psi = beta * jnp.power(v, b)
    return delta_psi


def modified_dispersion_phase_shift(
    f: jnp.ndarray,
    distance: Union[float, jnp.ndarray] = 100.0,
    lambda_g: Union[float, jnp.ndarray] = float("inf"),
    z: Union[float, jnp.ndarray] = 0.0,
    d_alpha: Optional[Union[float, jnp.ndarray]] = None,
) -> jnp.ndarray:
    """Compute frequency phase shift due to modified GW dispersion relation (massive graviton) in JAX.

    Literature: Mirshekari, Yunes, Will (2012) Eq. 24; Will (1998) Eq. 2.11.

    Dispersion relation:
        E^2 = p^2 * c^2 + A_alpha * p^alpha * c^alpha
    For alpha = 0 (massive graviton with Compton wavelength lambda_g = h / (m_g * c)):
        delta Psi_disp(f) = - (pi^2 * D_0) / (lambda_g^2 * (1 + z)) * f^(-1)

    Args:
        f: Frequency array in Hz.
        distance: Luminosity distance D_L in Mpc.
        lambda_g: Graviton Compton wavelength in meters (inf recovers pure GR).
        z: Cosmological redshift of source (default 0).
        d_alpha: Optional cosmological dispersion distance in meters.
            If None, computed as D_0 = D_L * MPC_SI / (1 + z).

    Returns:
        Array of dispersion phase shifts in radians.
    """
    f_t = as_array(f, dtype=jnp.float64)
    dist_mpc = as_array(distance, dtype=f_t.dtype)
    l_g = as_array(lambda_g, dtype=f_t.dtype)
    redshift = as_array(z, dtype=f_t.dtype)

    if isinstance(lambda_g, (int, float)) and (math.isinf(lambda_g) or lambda_g > 1e40):
        return jnp.zeros_like(f_t)

    is_gr = jnp.isinf(l_g) | (l_g > 1e40)

    if d_alpha is None:
        # Physical cosmological dispersion distance in meters: D0 = D_L / (1 + z)
        dist_m = dist_mpc * MPC_SI
        d0 = dist_m / (1.0 + redshift)
    else:
        d0 = as_array(d_alpha, dtype=f_t.dtype)

    safe_f = jnp.maximum(f_t, 1e-15)
    l_g_safe = jnp.maximum(jnp.where(is_gr, 1e40, l_g), 1e-30)

    coeff = -(PI ** 2.0) * d0 * C_SI / (l_g_safe * l_g_safe * (1.0 + redshift))
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
        total_m = m1 + m2

        # 2. Compute non-GR phase shifts
        delta_psi_ppe = ppe_phase_shift(f_t, total_m, beta_ppe=beta_ppe, b_ppe=b_ppe)
        delta_psi_disp = modified_dispersion_phase_shift(
            f_t, distance=distance, lambda_g=lambda_g, z=redshift
        )

        total_shift = delta_psi_ppe + delta_psi_disp

        # 3. Apply stationary phase deformation: exp(-i * delta_Psi)
        if hp_gr.ndim > total_shift.ndim:
            total_shift = jnp.expand_dims(total_shift, 0)

        cos_shift = jnp.cos(total_shift)
        sin_shift = jnp.sin(total_shift)
        exp_factor = cos_shift - 1j * sin_shift

        hp_ppe = hp_gr * exp_factor
        hc_ppe = hc_gr * exp_factor

        return hp_ppe, hc_ppe

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

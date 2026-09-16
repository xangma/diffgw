"""Pulsar Timing Array (PTA) Network and Hellings-Downs Likelihood in JAX.

Strict Clean-Room Implementation based exclusively on published literature:
- Hellings & Downs (1983), "Upper limits on the isotropic gravitational radiation background from pulsar timing analysis",
  Astrophys. J. Lett. 265, L39
- Arzoumanian et al. (2020), "The NANOGrav 12.5 yr Data Set: Search for an Isotropic Stochastic Gravitational-wave Background",
  Astrophys. J. Lett. 905, L34 (arXiv:2009.04496)
- Allen (2023), "Variance of the Hellings-Downs correlation",
  Phys. Rev. D 107, 043018 (arXiv:2205.05637)
"""

from typing import Dict, List, Optional, Sequence, Tuple, Union
import math
import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular

from diffgw_core.constants import PI

# 1 Julian year in seconds: 365.25 * 86400 s
SEC_PER_YEAR: float = 3.15576e7
F_YEAR: float = 1.0 / SEC_PER_YEAR  # ~ 3.16875e-8 Hz


def pulsar_line_of_sight(
    ra: Union[float, jnp.ndarray],
    dec: Union[float, jnp.ndarray],
) -> jnp.ndarray:
    """Compute Cartesian unit line-of-sight vector pointing from Earth to pulsar.

    p_hat = (cos(dec) * cos(ra), cos(dec) * sin(ra), sin(dec))

    Args:
        ra: Right ascension in radians.
        dec: Declination in radians.

    Returns:
        Array of shape (..., 3) with unit-length vectors.
    """
    alpha = jnp.asarray(ra, dtype=jnp.float64)
    delta = jnp.asarray(dec, dtype=jnp.float64)

    cos_d = jnp.cos(delta)
    sin_d = jnp.sin(delta)
    cos_a = jnp.cos(alpha)
    sin_a = jnp.sin(alpha)

    p_x = cos_d * cos_a
    p_y = cos_d * sin_a
    p_z = sin_d

    return jnp.stack([p_x, p_y, p_z], axis=-1)


def angular_separation(
    p1: jnp.ndarray,
    p2: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute angular separation zeta and cos(zeta) between pairs of pulsars.

    Args:
        p1: Unit line-of-sight vector(s) of shape (..., 3).
        p2: Unit line-of-sight vector(s) of shape (..., 3).

    Returns:
        Tuple of (zeta, cos_zeta).
    """
    cos_zeta = jnp.sum(p1 * p2, axis=-1)
    cos_zeta_clamped = jnp.clip(cos_zeta, -1.0, 1.0)
    zeta = jnp.arccos(cos_zeta_clamped)
    return zeta, cos_zeta_clamped


def hellings_downs_correlation(
    zeta: Union[float, jnp.ndarray],
    is_auto: Union[bool, jnp.ndarray] = False,
    include_pulsar_term: bool = True,
) -> jnp.ndarray:
    """Exact Hellings-Downs spatial correlation function for isotropic unpolarized GWB.

    Literature: Hellings & Downs (1983) Eq. 5; Arzoumanian et al. (2020) Sec. 2.

    Formula:
        x = (1 - cos(zeta)) / 2
        Gamma(zeta) = 1/2 - 1/4 * x + 3/2 * x * ln(x) + 1/2 * delta_{ab}

    Properties:
        - Cross-pulsar limit as zeta -> 0: Gamma(0) = 1/2 = 0.5 (since x * ln(x) -> 0).
        - Auto-correlation (a = b): Gamma(0) + 1/2 = 1.0.
        - Antipodal separation (zeta = pi, x = 1): Gamma(pi) = 1/2 - 1/4 = 0.25.
        - The characteristic dip at zeta ~ 90 deg has Gamma(pi/2) ~ -0.1448.

    Args:
        zeta: Angular separation in radians (scalar or array).
        is_auto: Boolean or boolean array indicating identical pulsars (delta_{ab} = 1).
        include_pulsar_term: If True, include the 1/2 * delta_{ab} auto-correlation pulsar term.

    Returns:
        Array of spatial correlation factors with same shape as zeta.
    """
    z = jnp.asarray(zeta, dtype=jnp.float64)
    cos_z = jnp.clip(jnp.cos(z), -1.0, 1.0)

    x = 0.5 * (1.0 - cos_z)
    x_clamped = jnp.clip(x, 1e-15, 1.0)

    # Stable x * ln(x) with lim_{x->0} x * ln(x) = 0
    x_ln_x = jnp.where(x < 1e-12, jnp.zeros_like(x), x_clamped * jnp.log(x_clamped))

    gamma = 0.5 - 0.25 * x + 1.5 * x_ln_x

    if include_pulsar_term:
        if isinstance(is_auto, bool):
            if is_auto:
                gamma = gamma + 0.5
        else:
            auto_t = jnp.asarray(is_auto, dtype=z.dtype)
            gamma = gamma + 0.5 * auto_t

    return gamma


def stochastic_gwb_spectrum(
    f: Union[float, jnp.ndarray],
    a_gwb: Union[float, jnp.ndarray],
    gamma: Union[float, jnp.ndarray] = 13.0 / 3.0,
    f_ref: float = F_YEAR,
) -> jnp.ndarray:
    """Compute stochastic gravitational-wave background strain spectral density S_h(f).

    Literature: Arzoumanian et al. (2020) Eq. 2:
        S_h(f) = (A_GWB^2 / (12 * pi^2)) * (f / f_yr)^(2 - gamma) * f_yr^(-3)

    Args:
        f: Frequency array or scalar in Hz.
        a_gwb: Dimensionless GWB amplitude at reference frequency f_ref (e.g. ~ 1e-15).
        gamma: Power-law spectral slope index (default 13/3 for SMBHB inspirals).
        f_ref: Reference frequency in Hz (default 1/yr = 3.16875e-8 Hz).

    Returns:
        Array of strain power spectral density S_h(f) in 1 / Hz.
    """
    f_t = jnp.asarray(f, dtype=jnp.float64)
    amp = jnp.asarray(a_gwb, dtype=jnp.float64)
    gam = jnp.asarray(gamma, dtype=jnp.float64)

    safe_f = jnp.maximum(f_t, 1e-15)
    f_ratio = safe_f / f_ref
    spectral_power = jnp.power(f_ratio, 2.0 - gam)

    prefactor = (amp * amp) / (12.0 * (PI ** 2.0) * (f_ref ** 3.0))
    return prefactor * spectral_power


@jax.tree_util.register_pytree_node_class
class PTANetwork:
    """Pulsar Timing Array (PTA) Network for Nanohertz Gravitational-Wave Background Detection in JAX."""

    def __init__(
        self,
        pulsar_names: Sequence[str],
        ra: Union[Sequence[float], jnp.ndarray],
        dec: Union[Sequence[float], jnp.ndarray],
        noise_std: Optional[Union[Sequence[float], jnp.ndarray]] = None,
        dtype=jnp.float64,
    ):
        self.pulsar_names = list(pulsar_names)
        self.n_pulsars = len(self.pulsar_names)

        self.ra = jnp.asarray(ra, dtype=dtype)
        self.dec = jnp.asarray(dec, dtype=dtype)

        # Precompute (N_p, 3) line-of-sight unit vectors
        self.positions = pulsar_line_of_sight(self.ra, self.dec)

        # Precompute (N_p, N_p) pairwise angular separations and Hellings-Downs correlation matrix
        pos_i = jnp.expand_dims(self.positions, 1)  # (N_p, 1, 3)
        pos_j = jnp.expand_dims(self.positions, 0)  # (1, N_p, 3)
        _, cos_zeta = angular_separation(pos_i, pos_j)
        self.zeta_matrix = jnp.arccos(jnp.clip(cos_zeta, -1.0, 1.0))

        identity = jnp.eye(self.n_pulsars, dtype=dtype)
        self.hd_matrix = hellings_downs_correlation(self.zeta_matrix, is_auto=identity, include_pulsar_term=True)

        if noise_std is None:
            self.noise_std = jnp.ones(self.n_pulsars, dtype=dtype) * 1e-7
        else:
            self.noise_std = jnp.asarray(noise_std, dtype=dtype)

    def tree_flatten(self):
        children = (self.ra, self.dec, self.positions, self.zeta_matrix, self.hd_matrix, self.noise_std)
        aux_data = (self.pulsar_names, self.n_pulsars)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls.__new__(cls)
        obj.pulsar_names, obj.n_pulsars = aux_data
        obj.ra, obj.dec, obj.positions, obj.zeta_matrix, obj.hd_matrix, obj.noise_std = children
        return obj

    def covariance_matrix(
        self,
        a_gwb: Union[float, jnp.ndarray],
        gamma: Union[float, jnp.ndarray] = 13.0 / 3.0,
        t_obs: float = 10.0 * SEC_PER_YEAR,
    ) -> jnp.ndarray:
        """Compute the total timing residual covariance matrix C = N + S(A_GWB, gamma).

        Args:
            a_gwb: GWB characteristic amplitude.
            gamma: Spectral slope (default 13/3).
            t_obs: Total observation duration in seconds (default 10 years).

        Returns:
            (N_p, N_p) symmetric positive-definite covariance matrix.
        """
        amp = jnp.asarray(a_gwb, dtype=self.positions.dtype)
        gam = jnp.asarray(gamma, dtype=self.positions.dtype)

        # White noise diagonal matrix N
        white_noise_var = self.noise_std * self.noise_std
        noise_matrix = jnp.diag(white_noise_var)

        # GWB variance scaling in timing residuals
        f0 = 1.0 / t_obs
        s_h0 = stochastic_gwb_spectrum(jnp.asarray(f0, dtype=self.positions.dtype), amp, gam)
        sigma_gwb_sq = s_h0 * f0

        # GWB spatial covariance S = Gamma_ab * sigma_gwb^2
        signal_matrix = self.hd_matrix * sigma_gwb_sq

        return noise_matrix + signal_matrix

    def log_likelihood(
        self,
        residuals: jnp.ndarray,
        a_gwb: Union[float, jnp.ndarray],
        gamma: Union[float, jnp.ndarray] = 13.0 / 3.0,
        t_obs: float = 10.0 * SEC_PER_YEAR,
    ) -> jnp.ndarray:
        """Compute Gaussian marginal log-likelihood of timing residuals given GWB parameters.

        ln L = -1/2 * [ r^T * C^(-1) * r + ln det(C) + N_p * ln(2*pi) ]

        Args:
            residuals: Timing residual vector r of shape (N_p,) or batched (B, N_p).
            a_gwb: GWB amplitude parameter.
            gamma: GWB spectral slope index.
            t_obs: Observation duration in seconds.

        Returns:
            Log-likelihood array (scalar or batched).
        """
        r = jnp.asarray(residuals, dtype=self.positions.dtype)
        cov = self.covariance_matrix(a_gwb, gamma=gamma, t_obs=t_obs)

        chol = jnp.linalg.cholesky(cov)

        if r.ndim == 1:
            y = solve_triangular(chol, r, lower=True)
            quad_form = jnp.sum(y * y)
        else:
            y = solve_triangular(chol, r.T, lower=True).T
            quad_form = jnp.sum(y * y, axis=-1)

        log_det = 2.0 * jnp.sum(jnp.log(jnp.diagonal(chol)))
        n_p = float(self.n_pulsars)
        const_term = n_p * math.log(2.0 * PI)

        return -0.5 * (quad_form + log_det + const_term)

    def forward(self, *args, **kwargs):
        return self.log_likelihood(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

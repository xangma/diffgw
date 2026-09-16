"""Clean-room Accelerated Gravitational-Wave Likelihood via Relative Binning (Heterodyning) in JAX.

Strict Clean-Room Implementation based exclusively on published literature:
- Zackay, Dai, Venumhav (2018), "Relative Binning and Fast Likelihood Evaluation for Gravitational Wave Parameter Estimation",
  arXiv:1806.08792
- Cornish (2010), "Heterodyned Likelihood for Fast Gravitational Wave Parameter Estimation",
  arXiv:1007.4820
- Dai, Venumhav, Zackay (2018), "Parameter Estimation for Gravitational Wave Signals from Compact Binary Coalescence with Fast Likelihood Evaluation",
  arXiv:1806.08793
"""

from typing import Dict, List, Optional, Sequence, Tuple, Union
import math
import numpy as np
import jax
import jax.numpy as jnp

from jaxwave.conversions import as_array


def compute_relative_binning_bins(
    f_min: float,
    f_max: float,
    n_bins: int = 50,
    gamma: float = -5.0 / 3.0,
    frequencies: Optional[Union[jnp.ndarray, np.ndarray]] = None,
    dtype: jnp.dtype = jnp.float64,
) -> jnp.ndarray:
    """Compute frequency bin edges adapted to post-Newtonian inspiral phasing in JAX.

    For compact binary inspiral waveforms, the phase difference Delta Psi(f) is
    dominated by the leading 0PN chirp mass variation Delta Psi ~ Delta M_c * f^(-5/3).
    Bins are placed uniformly in f^gamma to bound the phase variation across each bin:
        Delta Psi(f_{k+1}) - Delta Psi(f_k) <= Delta phi_max.

    Args:
        f_min: Minimum frequency cutoff in Hz.
        f_max: Maximum frequency cutoff in Hz.
        n_bins: Number of bins (typical M in [30, 80]).
        gamma: Frequency power-law scaling exponent (default -5/3 for leading PN chirp).
        frequencies: Optional dense frequency array. If provided, bin edges are snapped
            to exact grid values.
        dtype: JAX float dtype (default jnp.float64).

    Returns:
        1D Array of shape (n_bins + 1,) containing strictly increasing bin edges.
    """
    if f_min <= 0.0 or f_max <= f_min:
        raise ValueError(f"Invalid frequency boundaries: f_min={f_min}, f_max={f_max}")
    if n_bins < 2:
        raise ValueError(f"n_bins must be at least 2, got {n_bins}")

    if abs(gamma) < 1e-6:
        log_min = math.log(f_min)
        log_max = math.log(f_max)
        edges = jnp.exp(jnp.linspace(log_min, log_max, n_bins + 1, dtype=dtype))
    else:
        y_min = f_min ** gamma
        y_max = f_max ** gamma
        y = jnp.linspace(y_min, y_max, n_bins + 1, dtype=dtype)
        edges = y ** (1.0 / gamma)
        if edges[0] > edges[-1]:
            edges = jnp.flip(edges, axis=0)

    edges = edges.at[0].set(f_min)
    edges = edges.at[-1].set(f_max)

    if frequencies is not None:
        f_dense = as_array(frequencies, dtype=dtype)
        idx = jnp.argmin(jnp.abs(f_dense[:, None] - edges[None, :]), axis=0)
        edges_snapped = f_dense[idx]

        edges_list = [edges_snapped[0]]
        for i in range(1, len(edges_snapped)):
            val = edges_snapped[i]
            prev = edges_list[-1]
            val = jnp.where(val <= prev, prev + 1e-4, val)
            edges_list.append(val)
        edges = jnp.stack(edges_list)

    return edges


def compute_relative_binning_summary(
    data: jnp.ndarray,
    psd: jnp.ndarray,
    h0: jnp.ndarray,
    frequencies: jnp.ndarray,
    bin_edges: jnp.ndarray,
    df: Optional[float] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute the precomputed data-summary coefficients for relative binning in JAX.

    For each bin k in [0, M-1] spanning [f_bin[k], f_bin[k+1]):
        A_{0, k} = 4 * sum_{f in bin_k} (d(f) * h_0^*(f) / S_n(f)) * df
        A_{1, k} = 4 * sum_{f in bin_k} (d(f) * h_0^*(f) / S_n(f)) * (f - f_bar_k) * df
        B_{0, k} = 4 * sum_{f in bin_k} (|h_0(f)|^2 / S_n(f)) * df
        B_{1, k} = 4 * sum_{f in bin_k} (|h_0(f)|^2 / S_n(f)) * (f - f_bar_k) * df
    where f_bar_k = (f_bin[k] + f_bin[k+1]) / 2.

    Args:
        data: Frequency-domain strain data d(f), complex array.
        psd: Power spectral density S_n(f), real array.
        h0: Fiducial reference waveform h_0(f), complex array.
        frequencies: Dense frequency grid in Hz.
        bin_edges: Bin edges of shape (M + 1,).
        df: Frequency resolution (default: infer from frequencies).

    Returns:
        Tuple of (A0, A1, B0, B1, bin_centers, d_d):
            A0: Complex array of shape (M,)
            A1: Complex array of shape (M,)
            B0: Real array of shape (M,)
            B1: Real array of shape (M,)
            bin_centers: Real array of shape (M,)
            d_d: Scalar real array <d | d>
    """
    f_dense = as_array(frequencies, dtype=jnp.float64)
    edges = as_array(bin_edges, dtype=jnp.float64)
    d_arr = as_array(data, dtype=jnp.complex128 if jnp.iscomplexobj(data) else jnp.float64)
    psd_arr = as_array(psd, dtype=jnp.float64)
    h0_arr = as_array(h0, dtype=jnp.complex128 if jnp.iscomplexobj(h0) else jnp.float64)

    if df is None:
        if len(f_dense) > 1:
            df_val = f_dense[1] - f_dense[0]
        else:
            df_val = 1.0
    else:
        df_val = float(df)

    n_bins = len(edges) - 1
    safe_psd = jnp.maximum(psd_arr, 1.0e-50)

    # Dense integrands
    dh0_star = d_arr * jnp.conj(h0_arr)
    h0_sq = jnp.real(h0_arr * jnp.conj(h0_arr))

    integrand_a = 4.0 * (dh0_star / safe_psd) * df_val
    integrand_b = 4.0 * (h0_sq / safe_psd) * df_val

    # Precompute <d | d>
    d_sq = jnp.real(d_arr * jnp.conj(d_arr))
    d_d = jnp.sum(4.0 * (d_sq / safe_psd) * df_val)

    bin_centers = 0.5 * (edges[:-1] + edges[1:])

    # Vectorized bin masking
    f_row = f_dense[None, :]
    mask = (f_row >= edges[:-1, None]) & (f_row < edges[1:, None])
    mask = mask.at[-1].set((f_dense >= edges[-2]) & (f_dense <= edges[-1]))
    delta_f = f_row - bin_centers[:, None]

    a0 = jnp.sum(jnp.where(mask, integrand_a[None, :], 0.0), axis=-1)
    a1 = jnp.sum(jnp.where(mask, integrand_a[None, :] * delta_f, 0.0), axis=-1)
    b0 = jnp.sum(jnp.where(mask, integrand_b[None, :], 0.0), axis=-1)
    b1 = jnp.sum(jnp.where(mask, integrand_b[None, :] * delta_f, 0.0), axis=-1)

    return a0, a1, b0, b1, bin_centers, d_d


@jax.tree_util.register_pytree_node_class
class RelativeBinningLikelihood:
    """Accelerated Gravitational-Wave Likelihood via Relative Binning (Zackay et al. 2018) in JAX.

    Evaluates the Gaussian Whittle log-likelihood:
        ln L(d | theta) = -1/2 * <d - h(theta) | d - h(theta)>
                        = <d | h(theta)> - 1/2 <h(theta) | h(theta)> - 1/2 <d | d>
    in O(M) operations (M ~ 30-80 bins) instead of O(N_f) (N_f ~ 10^4 - 10^6 points),
    speeding up likelihood evaluations by 50-1000x while maintaining machine precision
    and full JAX autodiff / JIT differentiability.
    """

    def __init__(
        self,
        data: Dict[str, jnp.ndarray],
        psds: Dict[str, jnp.ndarray],
        h0: Dict[str, jnp.ndarray],
        frequencies: jnp.ndarray,
        bin_edges: Optional[jnp.ndarray] = None,
        n_bins: int = 50,
        f_min: Optional[float] = None,
        f_max: Optional[float] = None,
        df: Optional[float] = None,
    ):
        self.detectors = tuple(sorted(list(data.keys())))
        f_dense = as_array(frequencies, dtype=jnp.float64)
        self.frequencies = f_dense

        min_f = float(f_min) if f_min is not None else float(f_dense[0])
        max_f = float(f_max) if f_max is not None else float(f_dense[-1])

        if bin_edges is None:
            edges = compute_relative_binning_bins(
                f_min=min_f,
                f_max=max_f,
                n_bins=n_bins,
                frequencies=f_dense,
                dtype=jnp.float64,
            )
        else:
            edges = as_array(bin_edges, dtype=jnp.float64)

        self.bin_edges = edges
        self.n_bins = len(edges) - 1

        # Snap bin centers to exact frequencies in f_dense for machine precision matching
        raw_centers = 0.5 * (edges[:-1] + edges[1:])
        center_indices = jnp.argmin(jnp.abs(f_dense[:, None] - raw_centers[None, :]), axis=0)
        centers = f_dense[center_indices]
        self.bin_centers = centers
        self.center_indices = center_indices

        # Precompute summary coefficients for each detector
        self.a0: Dict[str, jnp.ndarray] = {}
        self.a1: Dict[str, jnp.ndarray] = {}
        self.b0: Dict[str, jnp.ndarray] = {}
        self.b1: Dict[str, jnp.ndarray] = {}
        self.h0_centers: Dict[str, jnp.ndarray] = {}
        self.dd: Dict[str, jnp.ndarray] = {}

        df_val = float(df) if df is not None else float(f_dense[1] - f_dense[0])

        for det in self.detectors:
            d_tensor = as_array(data[det], dtype=jnp.complex128 if jnp.iscomplexobj(data[det]) else jnp.float64)
            psd_tensor = as_array(psds[det], dtype=jnp.float64)
            h0_tensor = as_array(h0[det], dtype=jnp.complex128 if jnp.iscomplexobj(h0[det]) else jnp.float64)

            a0, _, b0, _, _, d_d_val = compute_relative_binning_summary(
                data=d_tensor,
                psd=psd_tensor,
                h0=h0_tensor,
                frequencies=f_dense,
                bin_edges=edges,
                df=df_val,
            )

            # Re-evaluate a1 and b1 with exact snapped centers
            safe_psd = jnp.maximum(psd_tensor, 1.0e-50)
            integrand_a = 4.0 * ((d_tensor * jnp.conj(h0_tensor)) / safe_psd) * df_val
            integrand_b = 4.0 * (jnp.real(h0_tensor * jnp.conj(h0_tensor)) / safe_psd) * df_val

            f_row = f_dense[None, :]
            mask = (f_row >= edges[:-1, None]) & (f_row < edges[1:, None])
            mask = mask.at[-1].set((f_dense >= edges[-2]) & (f_dense <= edges[-1]))
            delta_f = f_row - centers[:, None]

            a1 = jnp.sum(jnp.where(mask, integrand_a[None, :] * delta_f, 0.0), axis=-1)
            b1 = jnp.sum(jnp.where(mask, integrand_b[None, :] * delta_f, 0.0), axis=-1)

            self.a0[det] = a0
            self.a1[det] = a1
            self.b0[det] = b0
            self.b1[det] = b1
            self.h0_centers[det] = h0_tensor[center_indices]
            self.dd[det] = d_d_val

    def tree_flatten(self):
        children = (
            self.frequencies,
            self.bin_edges,
            self.bin_centers,
            self.center_indices,
            self.a0,
            self.a1,
            self.b0,
            self.b1,
            self.h0_centers,
            self.dd,
        )
        aux_data = (self.detectors, self.n_bins)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls.__new__(cls)
        obj.detectors, obj.n_bins = aux_data
        (
            obj.frequencies,
            obj.bin_edges,
            obj.bin_centers,
            obj.center_indices,
            obj.a0,
            obj.a1,
            obj.b0,
            obj.b1,
            obj.h0_centers,
            obj.dd,
        ) = children
        return obj

    @staticmethod
    def compute_r0_r1(
        h_centers: jnp.ndarray,
        h0_centers: jnp.ndarray,
        bin_centers: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Compute the ratio r_0 and slope r_1 at bin centers in JAX.

        Args:
            h_centers: Test waveform evaluated at bin centers, shape (..., M).
            h0_centers: Reference waveform evaluated at bin centers, shape (M,) or (..., M).
            bin_centers: Bin center frequencies, shape (M,).

        Returns:
            Tuple of (r0, r1) with same batch dimensions as h_centers.
        """
        safe_h0 = jnp.where(
            jnp.abs(h0_centers) < 1e-40,
            jnp.ones_like(h0_centers) * 1e-40,
            h0_centers,
        )
        r0 = h_centers / safe_h0

        m = bin_centers.shape[-1]
        if m >= 3:
            df_interior = bin_centers[2:] - bin_centers[:-2]
            r1_interior = (r0[..., 2:] - r0[..., :-2]) / df_interior

            df_0 = bin_centers[1] - bin_centers[0]
            r1_0 = (r0[..., 1:2] - r0[..., 0:1]) / df_0

            df_end = bin_centers[-1] - bin_centers[-2]
            r1_end = (r0[..., -1:] - r0[..., -2:-1]) / df_end

            r1 = jnp.concatenate([r1_0, r1_interior, r1_end], axis=-1)
        elif m == 2:
            df_step = bin_centers[1] - bin_centers[0]
            diff = (r0[..., 1:2] - r0[..., 0:1]) / df_step
            r1 = jnp.concatenate([diff, diff], axis=-1)
        else:
            r1 = jnp.zeros_like(r0)

        return r0, r1

    def inner_product_dh(
        self,
        r0: jnp.ndarray,
        r1: jnp.ndarray,
        detector: str,
    ) -> jnp.ndarray:
        """Compute noise-weighted inner product <d | h(theta)> from summary coefficients.

        <d | h> = Re sum_{k=0}^{M-1} [ r_0^*(f_bar_k) * A_{0, k} + r_1^*(f_bar_k) * A_{1, k} ]
        """
        a0 = self.a0[detector]
        a1 = self.a1[detector]

        r0_conj = jnp.conj(r0)
        r1_conj = jnp.conj(r1)

        term0 = r0_conj * a0
        term1 = r1_conj * a1

        sum_terms = jnp.sum(term0 + term1, axis=-1)
        return jnp.real(sum_terms)

    def inner_product_hh(
        self,
        r0: jnp.ndarray,
        r1: jnp.ndarray,
        detector: str,
    ) -> jnp.ndarray:
        """Compute waveform self inner product <h(theta) | h(theta)> from summary coefficients.

        <h | h> = sum_{k=0}^{M-1} [ |r_0(f_bar_k)|^2 * B_{0, k} + 2 * Re(r_0 * r_1^*) * B_{1, k} ]
        """
        b0 = self.b0[detector]
        b1 = self.b1[detector]

        r0_sq = jnp.real(r0 * jnp.conj(r0))
        cross_term = 2.0 * jnp.real(r0 * jnp.conj(r1))

        sum_terms = jnp.sum(r0_sq * b0 + cross_term * b1, axis=-1)
        return sum_terms

    def log_likelihood_from_r(
        self,
        r0_dict: Dict[str, jnp.ndarray],
        r1_dict: Dict[str, jnp.ndarray],
    ) -> jnp.ndarray:
        """Compute relative binning log-likelihood from pre-computed ratios (r0, r1).

        ln L = sum_{det} [ <d | h> - 0.5 * <h | h> - 0.5 * <d | d> ]
        """
        total_ln_l = jnp.array(0.0, dtype=jnp.float64)

        for det in self.detectors:
            r0 = r0_dict[det]
            r1 = r1_dict[det]
            dh = self.inner_product_dh(r0, r1, det)
            hh = self.inner_product_hh(r0, r1, det)
            dd = self.dd[det]

            det_ln_l = dh - 0.5 * hh - 0.5 * dd
            total_ln_l = total_ln_l + det_ln_l

        return total_ln_l

    def log_likelihood(
        self,
        templates_at_bin_centers: Union[Dict[str, jnp.ndarray], jnp.ndarray],
        h0_at_bin_centers: Optional[Dict[str, jnp.ndarray]] = None,
        slopes: Optional[Dict[str, jnp.ndarray]] = None,
    ) -> jnp.ndarray:
        """Compute relative binning log-likelihood from waveforms evaluated at bin centers.

        Args:
            templates_at_bin_centers: Dictionary mapping detector name to waveform
                array h(theta; f_bar) evaluated ONLY at bin centers, or single array
                if only one detector is configured.
            h0_at_bin_centers: Optional fiducial waveform h0(f_bar).
            slopes: Optional explicit slopes r_1(f_bar). If None, calculated via
                centered finite differences.

        Returns:
            Scalar or batched log-likelihood array.
        """
        r0_dict: Dict[str, jnp.ndarray] = {}
        r1_dict: Dict[str, jnp.ndarray] = {}

        for det in self.detectors:
            if isinstance(templates_at_bin_centers, dict):
                h_det = templates_at_bin_centers[det]
            else:
                h_det = templates_at_bin_centers

            if h0_at_bin_centers is not None:
                h0_det = h0_at_bin_centers[det]
            else:
                h0_det = self.h0_centers[det]

            if slopes is not None and det in slopes:
                safe_h0 = jnp.where(
                    jnp.abs(h0_det) < 1e-40,
                    jnp.ones_like(h0_det) * 1e-40,
                    h0_det,
                )
                r0 = h_det / safe_h0
                r1 = slopes[det]
            else:
                r0, r1 = self.compute_r0_r1(h_det, h0_det, self.bin_centers)

            r0_dict[det] = r0
            r1_dict[det] = r1

        return self.log_likelihood_from_r(r0_dict, r1_dict)

    def forward(
        self,
        templates_at_bin_centers: Union[Dict[str, jnp.ndarray], jnp.ndarray],
        **kwargs,
    ) -> jnp.ndarray:
        return self.log_likelihood(templates_at_bin_centers, **kwargs)

    def __call__(self, *args, **kwargs) -> jnp.ndarray:
        return self.forward(*args, **kwargs)

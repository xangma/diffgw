"""Differentiable Gravitational-Wave Likelihood and Noise-Weighted Inner Products in JAX.

Strict Clean-Room Implementation based exclusively on published literature:
- Cutler & Flanagan (1994), Phys. Rev. D 49, 2658 (arXiv:gr-qc/9402014)
- Veitch et al. (2015), Phys. Rev. D 91, 042003 (arXiv:1409.7788)
- Thrane & Talbot (2019), PASA 36, e010 (arXiv:1809.02293)
"""

from typing import Dict, List, Optional, Sequence, Tuple, Union
import math
import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy.special as jsp
import jax.scipy.integrate as jsi

if hasattr(jsi, "trapezoid"):
    _trapezoid = jsi.trapezoid
elif hasattr(jnp, "trapezoid"):
    _trapezoid = jnp.trapezoid
else:
    _trapezoid = jnp.trapz

from jaxwave.conversions import as_array


def noise_weighted_inner_product(
    a: jnp.ndarray,
    b: jnp.ndarray,
    psd: jnp.ndarray,
    df: Optional[Union[float, jnp.ndarray]] = None,
    frequencies: Optional[jnp.ndarray] = None,
    f_low: Optional[float] = None,
    f_high: Optional[float] = None,
    complex_output: bool = False,
) -> jnp.ndarray:
    """Compute the noise-weighted inner product <a | b> = 4 * Re int (a * b* / Sn) df.

    Args:
        a: Complex frequency-domain array
        b: Complex frequency-domain array
        psd: Real power spectral density array Sn(f)
        df: Frequency resolution in Hz (for uniform grid)
        frequencies: Frequency array in Hz (if non-uniform grid)
        f_low: Low-frequency cutoff in Hz
        f_high: High-frequency cutoff in Hz
        complex_output: If True, return complex inner product 4 * int (a * b* / Sn) df

    Returns:
        Scalar or batched inner product array.
    """
    a = jnp.asarray(a)
    b = jnp.asarray(b)
    psd = jnp.asarray(psd)

    integrand = (a * jnp.conj(b)) / jnp.maximum(psd, 1.0e-50)

    if frequencies is not None:
        freqs = jnp.asarray(frequencies)
        mask = jnp.ones(freqs.shape, dtype=bool)
        if f_low is not None:
            mask = mask & (freqs >= f_low)
        if f_high is not None:
            mask = mask & (freqs <= f_high)
        if integrand.ndim > mask.ndim:
            mask = jnp.broadcast_to(mask, integrand.shape)
        integrand = jnp.where(mask, integrand, 0.0)

        # Trapezoidal numerical integration
        integral = _trapezoid(integrand, freqs, axis=-1)
    else:
        df_val = df if df is not None else 1.0
        integral = jnp.sum(integrand, axis=-1) * df_val

    factor = 4.0
    result = factor * integral

    if complex_output:
        return result
    return jnp.real(result)


def log_bessel_i0(x: jnp.ndarray) -> jnp.ndarray:
    """Compute log(I0(x)) stably for all x >= 0 using jax.scipy.special.i0e.

    log(I0(x)) = x + log(I0e(x)) where I0e(x) = exp(-x) * I0(x)
    """
    x_arr = jnp.asarray(x)
    x_abs = jnp.abs(x_arr)
    # jax.scipy.special.i0e is exponentially scaled: i0e(x) = exp(-|x|) * i0(x)
    return x_abs + jnp.log(jnp.maximum(jsp.i0e(x_abs), 1.0e-50))


def optimal_snr(
    waveform: Union[jnp.ndarray, Dict[str, jnp.ndarray]],
    psd: Union[jnp.ndarray, Dict[str, jnp.ndarray]],
    df: Optional[Union[float, jnp.ndarray]] = None,
    frequencies: Optional[jnp.ndarray] = None,
    f_low: Optional[float] = None,
    f_high: Optional[float] = None,
) -> jnp.ndarray:
    """Compute optimal signal-to-noise ratio (SNR) rho = sqrt(<h | h>).

    Supports single-detector waveforms or multi-detector networks.
    For a network of detectors, returns rho_net = sqrt(sum_k <h_k | h_k>).

    Args:
        waveform: Complex frequency-domain array or dictionary {det: h_det}
        psd: Real PSD array or dictionary {det: psd_det}
        df: Frequency bin width in Hz
        frequencies: Frequency grid in Hz (if non-uniform)
        f_low: Low-frequency cutoff in Hz
        f_high: High-frequency cutoff in Hz

    Returns:
        Scalar or batched optimal SNR array.
    """
    if isinstance(waveform, dict):
        if not isinstance(psd, dict):
            raise TypeError("When waveform is a dict, psd must also be a dict.")
        net_snr_sq = None
        for det_name, h_k in waveform.items():
            if det_name not in psd:
                continue
            snr_sq_k = noise_weighted_inner_product(
                a=h_k,
                b=h_k,
                psd=psd[det_name],
                df=df,
                frequencies=frequencies,
                f_low=f_low,
                f_high=f_high,
            )
            net_snr_sq = snr_sq_k if net_snr_sq is None else net_snr_sq + snr_sq_k
        if net_snr_sq is None:
            raise ValueError("No matching detector keys between waveform and psd dictionaries.")
        return jnp.sqrt(jnp.maximum(net_snr_sq, 0.0))

    # Single detector
    psd_arr = psd
    if isinstance(psd, dict):
        psd_arr = next(iter(psd.values()))

    snr_sq = noise_weighted_inner_product(
        a=waveform,
        b=waveform,
        psd=psd_arr,
        df=df,
        frequencies=frequencies,
        f_low=f_low,
        f_high=f_high,
    )
    return jnp.sqrt(jnp.maximum(snr_sq, 0.0))


def matched_filter_snr(
    data: Union[jnp.ndarray, Dict[str, jnp.ndarray]],
    template: Union[jnp.ndarray, Dict[str, jnp.ndarray]],
    psd: Union[jnp.ndarray, Dict[str, jnp.ndarray]],
    df: Optional[Union[float, jnp.ndarray]] = None,
    frequencies: Optional[jnp.ndarray] = None,
    f_low: Optional[float] = None,
    f_high: Optional[float] = None,
) -> jnp.ndarray:
    """Compute matched-filter SNR rho_mf = Re<d | h> / sqrt(<h | h>).

    Args:
        data: Strain data array or dictionary {det: d_det}
        template: Template waveform array or dictionary {det: h_det}
        psd: PSD array or dictionary {det: psd_det}
        df: Frequency bin width in Hz
        frequencies: Frequency grid in Hz (if non-uniform)
        f_low: Low-frequency cutoff in Hz
        f_high: High-frequency cutoff in Hz

    Returns:
        Scalar or batched matched-filter SNR array.
    """
    if isinstance(template, dict):
        if not isinstance(data, dict) or not isinstance(psd, dict):
            raise TypeError("When template is a dict, data and psd must also be dicts.")
        net_overlap = None
        net_template_sq = None
        for det_name, h_k in template.items():
            if det_name not in data or det_name not in psd:
                continue
            d_k = data[det_name]
            psd_k = psd[det_name]
            overlap_k = noise_weighted_inner_product(
                a=d_k,
                b=h_k,
                psd=psd_k,
                df=df,
                frequencies=frequencies,
                f_low=f_low,
                f_high=f_high,
            )
            temp_sq_k = noise_weighted_inner_product(
                a=h_k,
                b=h_k,
                psd=psd_k,
                df=df,
                frequencies=frequencies,
                f_low=f_low,
                f_high=f_high,
            )
            net_overlap = overlap_k if net_overlap is None else net_overlap + overlap_k
            net_template_sq = temp_sq_k if net_template_sq is None else net_template_sq + temp_sq_k
        if net_overlap is None:
            raise ValueError("No matching detector keys between template, data, and psd.")
        denom = jnp.sqrt(jnp.maximum(net_template_sq, 1.0e-40))
        return net_overlap / denom

    psd_arr = psd
    if isinstance(psd, dict):
        psd_arr = next(iter(psd.values()))
    data_arr = data
    if isinstance(data, dict):
        data_arr = next(iter(data.values()))

    overlap = noise_weighted_inner_product(
        a=data_arr,
        b=template,
        psd=psd_arr,
        df=df,
        frequencies=frequencies,
        f_low=f_low,
        f_high=f_high,
    )
    temp_sq = noise_weighted_inner_product(
        a=template,
        b=template,
        psd=psd_arr,
        df=df,
        frequencies=frequencies,
        f_low=f_low,
        f_high=f_high,
    )
    denom = jnp.sqrt(jnp.maximum(temp_sq, 1.0e-40))
    return overlap / denom


@jax.tree_util.register_pytree_node_class
class WhittleLikelihood:
    """Differentiable Whittle Gaussian Log-Likelihood for Gravitational-Wave Detector Networks in JAX."""

    def __init__(
        self,
        data: Dict[str, jnp.ndarray],
        psds: Dict[str, jnp.ndarray],
        df: Optional[float] = None,
        frequencies: Optional[jnp.ndarray] = None,
        f_low: Optional[float] = None,
        f_high: Optional[float] = None,
    ):
        self.detectors = tuple(sorted(list(data.keys())))
        self.df = float(df) if df is not None else None
        self.f_low = float(f_low) if f_low is not None else None
        self.f_high = float(f_high) if f_high is not None else None
        self.data = {
            det: as_array(data[det], dtype=jnp.complex128 if jnp.iscomplexobj(data[det]) else jnp.float64)
            for det in self.detectors
        }
        self.psds = {
            det: as_array(psds[det], dtype=jnp.float64)
            for det in self.detectors
        }
        self.frequencies = as_array(frequencies, dtype=jnp.float64) if frequencies is not None else None

        # Precompute noise inner products <d | d>
        self._data_inner: Dict[str, jnp.ndarray] = {}
        for det in self.detectors:
            d = self.data[det]
            psd = self.psds[det]
            self._data_inner[det] = noise_weighted_inner_product(
                d, d, psd, df=self.df, frequencies=self.frequencies, f_low=self.f_low, f_high=self.f_high
            )

    def tree_flatten(self):
        children = (self.data, self.psds, self.frequencies, self._data_inner)
        aux_data = (self.detectors, self.df, self.f_low, self.f_high)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls.__new__(cls)
        obj.detectors, obj.df, obj.f_low, obj.f_high = aux_data
        obj.data, obj.psds, obj.frequencies, obj._data_inner = children
        return obj

    def log_likelihood(
        self,
        templates: Union[Dict[str, jnp.ndarray], jnp.ndarray],
    ) -> jnp.ndarray:
        """Compute standard Whittle log-likelihood ln L(d | theta) = -0.5 * sum_k <d_k - h_k | d_k - h_k>.

        Args:
            templates: Dictionary mapping detector prefix to frequency-domain strain array h_det(f),
                       or single array if only one detector is configured.
        """
        ln_l = jnp.array(0.0, dtype=jnp.float64)

        for det in self.detectors:
            d = self.data[det]
            psd = self.psds[det]
            if isinstance(templates, dict):
                h = templates[det]
            else:
                h = templates

            d_minus_h = d - h
            res_inner = noise_weighted_inner_product(
                d_minus_h,
                d_minus_h,
                psd,
                df=self.df,
                frequencies=self.frequencies,
                f_low=self.f_low,
                f_high=self.f_high,
            )
            ln_l = ln_l - 0.5 * res_inner

        return ln_l

    def phase_marginalized_log_likelihood(
        self,
        templates: Union[Dict[str, jnp.ndarray], jnp.ndarray],
    ) -> jnp.ndarray:
        """Analytical coalescence phase-marginalized log-likelihood (Veitch et al. 2015).

        ln L_marg = -0.5 * sum_k <h_k | h_k> + log(I0(|Z|)) - 0.5 * sum_k <d_k | d_k>
        where Z = sum_k 4 * int (d_k * h_k* / Sn_k) df.
        """
        z_total = jnp.array(0.0, dtype=jnp.complex128)
        hh_total = jnp.array(0.0, dtype=jnp.float64)
        dd_total = jnp.array(0.0, dtype=jnp.float64)

        for det in self.detectors:
            d = self.data[det]
            psd = self.psds[det]
            if isinstance(templates, dict):
                h = templates[det]
            else:
                h = templates

            z_k = noise_weighted_inner_product(
                d,
                h,
                psd,
                df=self.df,
                frequencies=self.frequencies,
                f_low=self.f_low,
                f_high=self.f_high,
                complex_output=True,
            )
            hh_k = noise_weighted_inner_product(
                h,
                h,
                psd,
                df=self.df,
                frequencies=self.frequencies,
                f_low=self.f_low,
                f_high=self.f_high,
            )

            z_total = z_total + z_k
            hh_total = hh_total + hh_k
            dd_total = dd_total + self._data_inner[det]

        ln_i0 = log_bessel_i0(jnp.abs(z_total))
        ln_l = ln_i0 - 0.5 * hh_total - 0.5 * dd_total
        return ln_l

    def distance_marginalized_log_likelihood(
        self,
        templates: Union[Dict[str, jnp.ndarray], jnp.ndarray],
        d_min: float = 10.0,
        d_max: float = 2000.0,
        d_ref: float = 100.0,
        n_quad: int = 64,
    ) -> jnp.ndarray:
        """Differentiable distance-marginalized log-likelihood with prior p(dL) proportional to dL^2.

        Uses Gauss-Legendre quadrature over [d_min, d_max].
        """
        z_re = jnp.array(0.0, dtype=jnp.float64)
        hh_total = jnp.array(0.0, dtype=jnp.float64)
        dd_total = jnp.array(0.0, dtype=jnp.float64)

        for det in self.detectors:
            d = self.data[det]
            psd = self.psds[det]
            if isinstance(templates, dict):
                h = templates[det]
            else:
                h = templates

            dh_k = noise_weighted_inner_product(
                d,
                h,
                psd,
                df=self.df,
                frequencies=self.frequencies,
                f_low=self.f_low,
                f_high=self.f_high,
            )
            hh_k = noise_weighted_inner_product(
                h,
                h,
                psd,
                df=self.df,
                frequencies=self.frequencies,
                f_low=self.f_low,
                f_high=self.f_high,
            )

            z_re = z_re + dh_k
            hh_total = hh_total + hh_k
            dd_total = dd_total + self._data_inner[det]

        # Gauss-Legendre quadrature nodes and weights on [-1, 1]
        nodes_np, weights_np = np.polynomial.legendre.leggauss(n_quad)
        nodes = jnp.asarray(nodes_np, dtype=jnp.float64)
        weights = jnp.asarray(weights_np, dtype=jnp.float64)

        dl = 0.5 * (d_max - d_min) * nodes + 0.5 * (d_max + d_min)
        w_quad = weights * 0.5 * (d_max - d_min)

        # Prior p(dL) = 3 * dL^2 / (d_max^3 - d_min^3)
        vol = (d_max ** 3.0 - d_min ** 3.0) / 3.0
        p_dl = (dl * dl) / vol

        # Scale factor u = d_ref / dL
        u = d_ref / dl

        # Integrand: p(dL) * exp(u * z_re - 0.5 * u^2 * hh - 0.5 * dd)
        arg = u * z_re - 0.5 * (u * u) * hh_total - 0.5 * dd_total
        max_arg = jnp.max(arg)

        integrand = p_dl * jnp.exp(arg - max_arg)
        quad_sum = jnp.sum(w_quad * integrand)

        ln_l = max_arg + jnp.log(jnp.maximum(quad_sum, 1.0e-50))
        return ln_l

    def forward(
        self,
        templates: Union[Dict[str, jnp.ndarray], jnp.ndarray],
        marginalize_phase: bool = False,
        marginalize_distance: bool = False,
        **kwargs,
    ) -> jnp.ndarray:
        if marginalize_phase:
            return self.phase_marginalized_log_likelihood(templates)
        if marginalize_distance:
            return self.distance_marginalized_log_likelihood(templates, **kwargs)
        return self.log_likelihood(templates)

    def __call__(self, *args, **kwargs) -> jnp.ndarray:
        return self.forward(*args, **kwargs)

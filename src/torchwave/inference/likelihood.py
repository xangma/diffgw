"""Differentiable Gravitational-Wave Likelihood and Noise-Weighted Inner Products.

Strict Clean-Room Implementation based exclusively on published literature:
- Cutler & Flanagan (1994), Phys. Rev. D 49, 2658 (arXiv:gr-qc/9402014)
- Veitch et al. (2015), Phys. Rev. D 91, 042003 (arXiv:1409.7788)
- Thrane & Talbot (2019), PASA 36, e010 (arXiv:1809.02293)
"""

from typing import Dict, Optional, Sequence, Tuple, Union
import math
import torch
import torch.nn as nn

from torchwave.conversions import as_tensor


def noise_weighted_inner_product(
    a: torch.Tensor,
    b: torch.Tensor,
    psd: torch.Tensor,
    df: Optional[Union[float, torch.Tensor]] = None,
    frequencies: Optional[torch.Tensor] = None,
    f_low: Optional[float] = None,
    f_high: Optional[float] = None,
    complex_output: bool = False,
) -> torch.Tensor:
    """Compute the noise-weighted inner product <a | b> = 4 * Re int (a * b* / Sn) df.

    Args:
        a: Complex frequency-domain tensor
        b: Complex frequency-domain tensor
        psd: Real power spectral density tensor Sn(f)
        df: Frequency resolution in Hz (for uniform grid)
        frequencies: Frequency tensor in Hz (if non-uniform grid)
        f_low: Low-frequency cutoff in Hz
        f_high: High-frequency cutoff in Hz
        complex_output: If True, return complex inner product 4 * int (a * b* / Sn) df

    Returns:
        Scalar or batched inner product tensor.
    """
    # Align shapes and types
    integrand = (a * torch.conj(b)) / torch.clamp(psd, min=1.0e-50)

    if frequencies is not None:
        mask = torch.ones_like(frequencies, dtype=torch.bool)
        if f_low is not None:
            mask = mask & (frequencies >= f_low)
        if f_high is not None:
            mask = mask & (frequencies <= f_high)
        integrand = torch.where(mask, integrand, torch.zeros_like(integrand))

        # Trapezoidal numerical integration
        integral = torch.trapezoid(integrand, frequencies, dim=-1)
    else:
        df_val = df if df is not None else 1.0
        integral = torch.sum(integrand, dim=-1) * df_val

    factor = 4.0
    result = factor * integral

    if complex_output:
        return result
    return torch.real(result)


def log_bessel_i0(x: torch.Tensor) -> torch.Tensor:
    """Compute log(I0(x)) stably for all x >= 0 using torch.special.i0e.

    log(I0(x)) = x + log(I0e(x)) where I0e(x) = exp(-x) * I0(x)
    """
    # torch.special.i0e is exponentially scaled: i0e(x) = exp(-|x|) * i0(x)
    x_abs = torch.abs(x)
    return x_abs + torch.log(torch.clamp(torch.special.i0e(x_abs), min=1.0e-50))


class WhittleLikelihood(nn.Module):
    """Differentiable Whittle Gaussian Log-Likelihood for Gravitational-Wave Detector Networks."""

    def __init__(
        self,
        data: Dict[str, torch.Tensor],
        psds: Dict[str, torch.Tensor],
        df: Optional[float] = None,
        frequencies: Optional[torch.Tensor] = None,
        f_low: Optional[float] = None,
        f_high: Optional[float] = None,
    ):
        super().__init__()
        self.detectors = sorted(list(data.keys()))
        self.df = df
        self.f_low = f_low
        self.f_high = f_high

        for det in self.detectors:
            d_tensor = as_tensor(data[det])
            psd_tensor = as_tensor(psds[det])
            self.register_buffer(f"data_{det}", d_tensor, persistent=True)
            self.register_buffer(f"psd_{det}", psd_tensor, persistent=True)

        if frequencies is not None:
            self.register_buffer("frequencies", as_tensor(frequencies), persistent=True)
        else:
            self.frequencies = None

        # Precompute noise inner products <d | d>
        self._data_inner: Dict[str, torch.Tensor] = {}
        for det in self.detectors:
            d = getattr(self, f"data_{det}")
            psd = getattr(self, f"psd_{det}")
            self._data_inner[det] = noise_weighted_inner_product(
                d, d, psd, df=self.df, frequencies=self.frequencies, f_low=self.f_low, f_high=self.f_high
            )

    def log_likelihood(
        self,
        templates: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute standard Whittle log-likelihood ln L(d | theta) = -0.5 * sum_k <d_k - h_k | d_k - h_k>.

        Args:
            templates: Dictionary mapping detector prefix to frequency-domain strain tensor h_det(f)
        """
        ln_l = torch.tensor(0.0, dtype=torch.float64, device=getattr(self, f"data_{self.detectors[0]}").device)

        for det in self.detectors:
            d = getattr(self, f"data_{det}")
            psd = getattr(self, f"psd_{det}")
            h = templates[det]

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
        templates: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Analytical coalescence phase-marginalized log-likelihood (Veitch et al. 2015).

        ln L_marg = -0.5 * sum_k <h_k | h_k> + log(I0(|Z|)) - 0.5 * sum_k <d_k | d_k>
        where Z = sum_k 4 * int (d_k * h_k* / Sn_k) df.
        """
        device = getattr(self, f"data_{self.detectors[0]}").device
        z_total = torch.tensor(0.0, dtype=torch.complex128, device=device)
        hh_total = torch.tensor(0.0, dtype=torch.float64, device=device)
        dd_total = torch.tensor(0.0, dtype=torch.float64, device=device)

        for det in self.detectors:
            d = getattr(self, f"data_{det}")
            psd = getattr(self, f"psd_{det}")
            h = templates[det]

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

        ln_i0 = log_bessel_i0(torch.abs(z_total))
        ln_l = ln_i0 - 0.5 * hh_total - 0.5 * dd_total
        return ln_l

    def distance_marginalized_log_likelihood(
        self,
        templates: Dict[str, torch.Tensor],
        d_min: float = 10.0,
        d_max: float = 2000.0,
        d_ref: float = 100.0,
        n_quad: int = 64,
    ) -> torch.Tensor:
        """Differentiable distance-marginalized log-likelihood with prior p(dL) proportional to dL^2.

        Uses Gauss-Legendre quadrature over [d_min, d_max].
        """
        device = getattr(self, f"data_{self.detectors[0]}").device
        z_re = torch.tensor(0.0, dtype=torch.float64, device=device)
        hh_total = torch.tensor(0.0, dtype=torch.float64, device=device)
        dd_total = torch.tensor(0.0, dtype=torch.float64, device=device)

        for det in self.detectors:
            d = getattr(self, f"data_{det}")
            psd = getattr(self, f"psd_{det}")
            h = templates[det]

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
        nodes_np, weights_np = math.erf, None
        import numpy as np
        nodes_np, weights_np = np.polynomial.legendre.leggauss(n_quad)

        # Map to [d_min, d_max]
        nodes = torch.tensor(nodes_np, dtype=torch.float64, device=device)
        weights = torch.tensor(weights_np, dtype=torch.float64, device=device)

        dl = 0.5 * (d_max - d_min) * nodes + 0.5 * (d_max + d_min)
        w_quad = weights * 0.5 * (d_max - d_min)

        # Prior p(dL) = 3 * dL^2 / (d_max^3 - d_min^3)
        vol = (d_max ** 3.0 - d_min ** 3.0) / 3.0
        p_dl = (dl * dl) / vol

        # Scale factor u = d_ref / dL
        u = d_ref / dl

        # Integrand: p(dL) * exp(u * z_re - 0.5 * u^2 * hh - 0.5 * dd)
        arg = u * z_re - 0.5 * (u * u) * hh_total - 0.5 * dd_total
        max_arg = torch.max(arg)

        integrand = p_dl * torch.exp(arg - max_arg)
        quad_sum = torch.sum(w_quad * integrand)

        ln_l = max_arg + torch.log(torch.clamp(quad_sum, min=1.0e-50))
        return ln_l

    def forward(
        self,
        templates: Dict[str, torch.Tensor],
        marginalize_phase: bool = False,
        marginalize_distance: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        if marginalize_phase:
            return self.phase_marginalized_log_likelihood(templates)
        if marginalize_distance:
            return self.distance_marginalized_log_likelihood(templates, **kwargs)
        return self.log_likelihood(templates)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

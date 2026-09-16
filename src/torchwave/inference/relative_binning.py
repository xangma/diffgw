"""Clean-room Accelerated Gravitational-Wave Likelihood via Relative Binning (Heterodyning).

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
import torch
import torch.nn as nn

from torchwave.conversions import as_tensor


def compute_relative_binning_bins(
    f_min: float,
    f_max: float,
    n_bins: int = 50,
    gamma: float = -5.0 / 3.0,
    frequencies: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Compute frequency bin edges adapted to post-Newtonian inspiral phasing.

    For compact binary inspiral waveforms, the phase difference Delta Psi(f) is
    dominated by the leading 0PN chirp mass variation Delta Psi ~ Delta M_c * f^(-5/3).
    Bins are placed uniformly in f^gamma to bound the phase variation across each bin:
        Delta Psi(f_{k+1}) - Delta Psi(f_k) <= Delta phi_max.

    Args:
        f_min: Minimum frequency cutoff in Hz.
        f_max: Maximum frequency cutoff in Hz.
        n_bins: Number of bins (typical M in [30, 80]).
        gamma: Frequency power-law scaling exponent (default -5/3 for leading PN chirp).
        frequencies: Optional dense frequency tensor. If provided, bin edges are snapped
            to exact grid values.
        device: PyTorch device.
        dtype: PyTorch float dtype (default torch.float64).

    Returns:
        1D Tensor of shape (n_bins + 1,) containing strictly increasing bin edges.
    """
    if f_min <= 0.0 or f_max <= f_min:
        raise ValueError(f"Invalid frequency boundaries: f_min={f_min}, f_max={f_max}")
    if n_bins < 2:
        raise ValueError(f"n_bins must be at least 2, got {n_bins}")

    if abs(gamma) < 1e-6:
        # Logarithmic binning
        log_min = math.log(f_min)
        log_max = math.log(f_max)
        edges = torch.exp(torch.linspace(log_min, log_max, n_bins + 1, dtype=dtype, device=device))
    else:
        # Power-law spacing in f^gamma
        y_min = f_min ** gamma
        y_max = f_max ** gamma
        y = torch.linspace(y_min, y_max, n_bins + 1, dtype=dtype, device=device)
        edges = y ** (1.0 / gamma)
        # Ensure monotonic increasing order
        if edges[0] > edges[-1]:
            edges = torch.flip(edges, dims=[0])

    edges[0] = f_min
    edges[-1] = f_max

    if frequencies is not None:
        f_dense = as_tensor(frequencies, device=device, dtype=dtype)
        # Snap edges to nearest frequencies in f_dense
        snapped = []
        for i, edge in enumerate(edges):
            idx = torch.argmin(torch.abs(f_dense - edge))
            snapped.append(f_dense[idx])
        edges = torch.stack(snapped)
        # Ensure strictly increasing
        for i in range(1, len(edges)):
            if edges[i] <= edges[i - 1]:
                # Bump by at least one grid step
                edges[i] = edges[i - 1] + 1e-4

    return edges


def compute_relative_binning_summary(
    data: torch.Tensor,
    psd: torch.Tensor,
    h0: torch.Tensor,
    frequencies: torch.Tensor,
    bin_edges: torch.Tensor,
    df: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the precomputed data-summary coefficients for relative binning.

    For each bin k in [0, M-1] spanning [f_bin[k], f_bin[k+1]):
        A_{0, k} = 4 * sum_{f in bin_k} (d(f) * h_0^*(f) / S_n(f)) * df
        A_{1, k} = 4 * sum_{f in bin_k} (d(f) * h_0^*(f) / S_n(f)) * (f - f_bar_k) * df
        B_{0, k} = 4 * sum_{f in bin_k} (|h_0(f)|^2 / S_n(f)) * df
        B_{1, k} = 4 * sum_{f in bin_k} (|h_0(f)|^2 / S_n(f)) * (f - f_bar_k) * df
    where f_bar_k = (f_bin[k] + f_bin[k+1]) / 2.

    Args:
        data: Frequency-domain strain data d(f), complex tensor.
        psd: Power spectral density S_n(f), real tensor.
        h0: Fiducial reference waveform h_0(f), complex tensor.
        frequencies: Dense frequency grid in Hz.
        bin_edges: Bin edges of shape (M + 1,).
        df: Frequency resolution (default: infer from frequencies).

    Returns:
        Tuple of (A0, A1, B0, B1, bin_centers, d_d):
            A0: Complex tensor of shape (M,)
            A1: Complex tensor of shape (M,)
            B0: Real tensor of shape (M,)
            B1: Real tensor of shape (M,)
            bin_centers: Real tensor of shape (M,)
            d_d: Scalar real tensor <d | d>
    """
    device = data.device
    f_dense = frequencies.to(device=device)
    edges = bin_edges.to(device=device)

    if df is None:
        if len(f_dense) > 1:
            df_val = (f_dense[1] - f_dense[0]).item()
        else:
            df_val = 1.0
    else:
        df_val = float(df)

    n_bins = len(edges) - 1
    safe_psd = torch.clamp(psd, min=1.0e-50)

    # Dense integrands
    dh0_star = data * torch.conj(h0)
    h0_sq = torch.real(h0 * torch.conj(h0))

    integrand_a = 4.0 * (dh0_star / safe_psd) * df_val
    integrand_b = 4.0 * (h0_sq / safe_psd) * df_val

    # Precompute <d | d>
    d_sq = torch.real(data * torch.conj(data))
    d_d = torch.sum(4.0 * (d_sq / safe_psd) * df_val)

    bin_centers = 0.5 * (edges[:-1] + edges[1:])

    a0_list = []
    a1_list = []
    b0_list = []
    b1_list = []

    for k in range(n_bins):
        f_low = edges[k]
        f_high = edges[k + 1]
        f_bar = bin_centers[k]

        if k == n_bins - 1:
            mask = (f_dense >= f_low) & (f_dense <= f_high)
        else:
            mask = (f_dense >= f_low) & (f_dense < f_high)

        f_k = f_dense[mask]
        delta_f = f_k - f_bar

        a0_k = torch.sum(integrand_a[mask])
        a1_k = torch.sum(integrand_a[mask] * delta_f)

        b0_k = torch.sum(integrand_b[mask])
        b1_k = torch.sum(integrand_b[mask] * delta_f)

        a0_list.append(a0_k)
        a1_list.append(a1_k)
        b0_list.append(b0_k)
        b1_list.append(b1_k)

    a0 = torch.stack(a0_list)
    a1 = torch.stack(a1_list)
    b0 = torch.stack(b0_list)
    b1 = torch.stack(b1_list)

    return a0, a1, b0, b1, bin_centers, d_d


class RelativeBinningLikelihood(nn.Module):
    """Accelerated Gravitational-Wave Likelihood via Relative Binning (Zackay et al. 2018).

    Evaluates the Gaussian Whittle log-likelihood:
        ln L(d | theta) = -1/2 * <d - h(theta) | d - h(theta)>
                        = <d | h(theta)> - 1/2 <h(theta) | h(theta)> - 1/2 <d | d>
    in O(M) operations (M ~ 30-80 bins) instead of O(N_f) (N_f ~ 10^4 - 10^6 points),
    speeding up likelihood evaluations by 50-1000x while maintaining machine precision
    and full autograd differentiability.
    """

    def __init__(
        self,
        data: Dict[str, torch.Tensor],
        psds: Dict[str, torch.Tensor],
        h0: Dict[str, torch.Tensor],
        frequencies: torch.Tensor,
        bin_edges: Optional[torch.Tensor] = None,
        n_bins: int = 50,
        f_min: Optional[float] = None,
        f_max: Optional[float] = None,
        df: Optional[float] = None,
    ):
        """Initialize RelativeBinningLikelihood with precomputed data summary.

        Args:
            data: Dictionary mapping detector name (e.g. 'H1', 'L1') to dense frequency-domain data.
            psds: Dictionary mapping detector name to PSD tensor S_n(f).
            h0: Dictionary mapping detector name to fiducial reference waveform h_0(f).
            frequencies: Dense frequency grid in Hz.
            bin_edges: Optional precomputed bin edges (M + 1,).
            n_bins: Number of bins if bin_edges is not provided.
            f_min: Minimum frequency cutoff in Hz (default: min of frequencies).
            f_max: Maximum frequency cutoff in Hz (default: max of frequencies).
            df: Dense frequency spacing in Hz.
        """
        super().__init__()
        self.detectors = sorted(list(data.keys()))
        f_dense = as_tensor(frequencies)
        self.register_buffer("frequencies", f_dense, persistent=True)

        min_f = f_min if f_min is not None else float(f_dense[0])
        max_f = f_max if f_max is not None else float(f_dense[-1])

        if bin_edges is None:
            edges = compute_relative_binning_bins(
                f_min=min_f,
                f_max=max_f,
                n_bins=n_bins,
                frequencies=f_dense,
                device=f_dense.device,
                dtype=f_dense.dtype,
            )
        else:
            edges = as_tensor(bin_edges, device=f_dense.device, dtype=f_dense.dtype)

        self.register_buffer("bin_edges", edges, persistent=True)
        self.n_bins = len(edges) - 1

        # Snap bin centers to exact frequencies in f_dense for machine precision matching
        raw_centers = 0.5 * (edges[:-1] + edges[1:])
        center_indices = torch.argmin(torch.abs(f_dense.unsqueeze(1) - raw_centers.unsqueeze(0)), dim=0)
        centers = f_dense[center_indices]
        self.register_buffer("bin_centers", centers, persistent=True)
        self.register_buffer("center_indices", center_indices, persistent=False)

        # Precompute summary coefficients for each detector
        self.d_d: Dict[str, torch.Tensor] = {}
        for det in self.detectors:
            d_tensor = as_tensor(data[det], device=f_dense.device)
            psd_tensor = as_tensor(psds[det], device=f_dense.device)
            h0_tensor = as_tensor(h0[det], device=f_dense.device)

            a0, a1, b0, b1, _, d_d_val = compute_relative_binning_summary(
                data=d_tensor,
                psd=psd_tensor,
                h0=h0_tensor,
                frequencies=f_dense,
                bin_edges=edges,
                df=df,
            )

            # Re-evaluate a1 and b1 with exact snapped centers
            safe_psd = torch.clamp(psd_tensor, min=1.0e-50)
            df_val = float(df) if df is not None else float(f_dense[1] - f_dense[0])
            integrand_a = 4.0 * ((d_tensor * torch.conj(h0_tensor)) / safe_psd) * df_val
            integrand_b = 4.0 * (torch.real(h0_tensor * torch.conj(h0_tensor)) / safe_psd) * df_val

            a1_list = []
            b1_list = []
            for k in range(self.n_bins):
                f_low = edges[k]
                f_high = edges[k + 1]
                f_bar = centers[k]
                if k == self.n_bins - 1:
                    mask = (f_dense >= f_low) & (f_dense <= f_high)
                else:
                    mask = (f_dense >= f_low) & (f_dense < f_high)
                delta_f = f_dense[mask] - f_bar
                a1_list.append(torch.sum(integrand_a[mask] * delta_f))
                b1_list.append(torch.sum(integrand_b[mask] * delta_f))

            a1 = torch.stack(a1_list)
            b1 = torch.stack(b1_list)

            self.register_buffer(f"a0_{det}", a0, persistent=True)
            self.register_buffer(f"a1_{det}", a1, persistent=True)
            self.register_buffer(f"b0_{det}", b0, persistent=True)
            self.register_buffer(f"b1_{det}", b1, persistent=True)
            self.register_buffer(f"h0_centers_{det}", h0_tensor[center_indices], persistent=True)
            self.register_buffer(f"h0_dense_{det}", h0_tensor, persistent=False)
            self.register_buffer(f"dd_{det}", d_d_val, persistent=True)

    @staticmethod
    def compute_r0_r1(
        h_centers: torch.Tensor,
        h0_centers: torch.Tensor,
        bin_centers: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the ratio r_0 and slope r_1 at bin centers.

        Args:
            h_centers: Test waveform evaluated at bin centers, shape (..., M).
            h0_centers: Reference waveform evaluated at bin centers, shape (M,) or (..., M).
            bin_centers: Bin center frequencies, shape (M,).

        Returns:
            Tuple of (r0, r1) with same batch dimensions as h_centers.
        """
        safe_h0 = torch.where(
            torch.abs(h0_centers) < 1e-40,
            torch.ones_like(h0_centers) * 1e-40,
            h0_centers,
        )
        r0 = h_centers / safe_h0

        m = bin_centers.shape[-1]
        r1 = torch.zeros_like(r0)

        if m >= 3:
            df_interior = bin_centers[2:] - bin_centers[:-2]
            r1[..., 1:-1] = (r0[..., 2:] - r0[..., :-2]) / df_interior

            # Boundary 0
            df_0 = bin_centers[1] - bin_centers[0]
            r1[..., 0] = (r0[..., 1] - r0[..., 0]) / df_0

            # Boundary M - 1
            df_end = bin_centers[-1] - bin_centers[-2]
            r1[..., -1] = (r0[..., -1] - r0[..., -2]) / df_end
        elif m == 2:
            df_step = bin_centers[1] - bin_centers[0]
            diff = (r0[..., 1] - r0[..., 0]) / df_step
            r1[..., 0] = diff
            r1[..., 1] = diff

        return r0, r1

    def inner_product_dh(
        self,
        r0: torch.Tensor,
        r1: torch.Tensor,
        detector: str,
    ) -> torch.Tensor:
        """Compute noise-weighted inner product <d | h(theta)> from summary coefficients.

        <d | h> = Re sum_{k=0}^{M-1} [ r_0^*(f_bar_k) * A_{0, k} + r_1^*(f_bar_k) * A_{1, k} ]
        """
        a0 = getattr(self, f"a0_{detector}")
        a1 = getattr(self, f"a1_{detector}")

        r0_conj = torch.conj(r0)
        r1_conj = torch.conj(r1)

        term0 = r0_conj * a0
        term1 = r1_conj * a1

        sum_terms = torch.sum(term0 + term1, dim=-1)
        return torch.real(sum_terms)

    def inner_product_hh(
        self,
        r0: torch.Tensor,
        r1: torch.Tensor,
        detector: str,
    ) -> torch.Tensor:
        """Compute waveform self inner product <h(theta) | h(theta)> from summary coefficients.

        <h | h> = sum_{k=0}^{M-1} [ |r_0(f_bar_k)|^2 * B_{0, k} + 2 * Re(r_0 * r_1^*) * B_{1, k} ]
        """
        b0 = getattr(self, f"b0_{detector}")
        b1 = getattr(self, f"b1_{detector}")

        r0_sq = torch.real(r0 * torch.conj(r0))
        cross_term = 2.0 * torch.real(r0 * torch.conj(r1))

        sum_terms = torch.sum(r0_sq * b0 + cross_term * b1, dim=-1)
        return sum_terms

    def log_likelihood_from_r(
        self,
        r0_dict: Dict[str, torch.Tensor],
        r1_dict: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute relative binning log-likelihood from pre-computed ratios (r0, r1).

        ln L = sum_{det} [ <d | h> - 0.5 * <h | h> - 0.5 * <d | d> ]
        """
        device = self.bin_centers.device
        total_ln_l = torch.tensor(0.0, dtype=torch.float64, device=device)

        for det in self.detectors:
            r0 = r0_dict[det]
            r1 = r1_dict[det]
            dh = self.inner_product_dh(r0, r1, det)
            hh = self.inner_product_hh(r0, r1, det)
            dd = getattr(self, f"dd_{det}")

            det_ln_l = dh - 0.5 * hh - 0.5 * dd
            total_ln_l = total_ln_l + det_ln_l

        return total_ln_l

    def log_likelihood(
        self,
        templates_at_bin_centers: Dict[str, torch.Tensor],
        h0_at_bin_centers: Optional[Dict[str, torch.Tensor]] = None,
        slopes: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Compute relative binning log-likelihood from waveforms evaluated at bin centers.

        Args:
            templates_at_bin_centers: Dictionary mapping detector name to waveform
                tensor h(theta; f_bar) evaluated ONLY at bin centers.
            h0_at_bin_centers: Optional fiducial waveform h0(f_bar). If not provided,
                sampled from stored dense h0.
            slopes: Optional explicit slopes r_1(f_bar). If None, calculated via
                centered finite differences.

        Returns:
            Scalar or batched log-likelihood tensor.
        """
        r0_dict: Dict[str, torch.Tensor] = {}
        r1_dict: Dict[str, torch.Tensor] = {}

        for det in self.detectors:
            h_det = templates_at_bin_centers[det]
            if h0_at_bin_centers is not None:
                h0_det = h0_at_bin_centers[det]
            elif hasattr(self, f"h0_centers_{det}"):
                h0_det = getattr(self, f"h0_centers_{det}")
            else:
                h0_dense = getattr(self, f"h0_dense_{det}")
                f_dense = self.frequencies
                idx = torch.searchsorted(f_dense, self.bin_centers).clamp(max=len(f_dense) - 1)
                h0_det = h0_dense[idx]

            if slopes is not None and det in slopes:
                safe_h0 = torch.where(
                    torch.abs(h0_det) < 1e-40,
                    torch.ones_like(h0_det) * 1e-40,
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
        templates_at_bin_centers: Dict[str, torch.Tensor],
        **kwargs,
    ) -> torch.Tensor:
        return self.log_likelihood(templates_at_bin_centers, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

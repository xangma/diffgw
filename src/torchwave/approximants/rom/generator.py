"""SEOBNR Reduced-Order Model (ROM) Surrogate Training and Generation Pipeline.

Strict Clean-Room Implementation:
Generates training datasets directly from torchwave's SEOBNR_ODE time-domain integrator,
constructs orthonormal reduced bases using SVD, fits parameter-space mappings,
and serializes native PyTorch surrogate models under the permissive MIT license.

References:
- Pürrer (2014), Class. Quantum Grav. 31, 195010 (arXiv:1402.4146)
- Pürrer (2016), Phys. Rev. D 93, 064041 (arXiv:1512.02248)
- Bohé et al. (2017), Phys. Rev. D 95, 044028 (arXiv:1611.03703)
- Cotesta et al. (2018), Phys. Rev. D 98, 084028 (arXiv:1803.10701)
"""

import math
import os
from typing import Dict, Optional, Tuple, Union
import torch

from torchwave.constants import (
    C_SI,
    MPC_SI,
    MSUN_SEC,
    PI,
)
from torchwave.conversions import (
    component_masses_to_eta,
    spin_combinations,
)
from torchwave.approximants.eob.waveform import SEOBNR_ODE


def unwrap_phase(p: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Vectorized, autograd-compatible phase unwrapping in pure PyTorch."""
    diff = torch.diff(p, dim=dim)
    diff_mod = (diff + torch.pi) % (2.0 * torch.pi) - torch.pi
    diff_corr = diff_mod - diff
    diff_corr = torch.where(
        (diff == -torch.pi) & (diff_mod == torch.pi),
        torch.zeros_like(diff_corr),
        diff_corr,
    )
    cum_corr = torch.cumsum(diff_corr, dim=dim)
    zeros = torch.zeros_like(torch.select(p, dim, 0)).unsqueeze(dim)
    return p + torch.cat([zeros, cum_corr], dim=dim)


def poly_features(
    eta: torch.Tensor,
    chi1: torch.Tensor,
    chi2: Optional[torch.Tensor] = None,
    degree: int = 3,
) -> torch.Tensor:
    """Compute polynomial expansion features for (eta, chi1) or (eta, chi1, chi2) up to given degree."""
    eta_flat = eta.reshape(-1)
    chi1_flat = chi1.reshape(-1)
    cols = [torch.ones_like(eta_flat)]
    if chi2 is None:
        for d in range(1, degree + 1):
            for i in range(d + 1):
                j = d - i
                cols.append((eta_flat ** i) * (chi1_flat ** j))
    else:
        chi2_flat = chi2.reshape(-1)
        for d in range(1, degree + 1):
            for i in range(d + 1):
                for j in range(d - i + 1):
                    k = d - i - j
                    cols.append((eta_flat ** i) * (chi1_flat ** j) * (chi2_flat ** k))
    return torch.stack(cols, dim=-1)


def chebyshev_features(
    eta: torch.Tensor,
    chi1: torch.Tensor,
    chi2: Optional[torch.Tensor] = None,
    degree: int = 3,
) -> torch.Tensor:
    """Compute tensor-product Chebyshev polynomial features up to given total degree."""
    eta_flat = eta.reshape(-1)
    chi1_flat = chi1.reshape(-1)

    # Normalize eta from [0.03, 0.25] to [-1, 1]
    eta_norm = torch.clamp(2.0 * (eta_flat - 0.03) / (0.25 - 0.03) - 1.0, -1.0, 1.0)
    chi1_norm = torch.clamp(chi1_flat, -1.0, 1.0)

    def cheb_1d(x: torch.Tensor, deg: int) -> list:
        polys = [torch.ones_like(x)]
        if deg >= 1:
            polys.append(x)
        for _ in range(2, deg + 1):
            polys.append(2.0 * x * polys[-1] - polys[-2])
        return polys

    t_eta = cheb_1d(eta_norm, degree)
    t_c1 = cheb_1d(chi1_norm, degree)

    cols = []
    if chi2 is None:
        for d in range(degree + 1):
            for i in range(d + 1):
                j = d - i
                cols.append(t_eta[i] * t_c1[j])
    else:
        chi2_flat = chi2.reshape(-1)
        chi2_norm = torch.clamp(chi2_flat, -1.0, 1.0)
        t_c2 = cheb_1d(chi2_norm, degree)
        for d in range(degree + 1):
            for i in range(d + 1):
                for j in range(d - i + 1):
                    k = d - i - j
                    cols.append(t_eta[i] * t_c1[j] * t_c2[k])

    return torch.stack(cols, dim=-1)


def generate_eob_training_grid(
    q_min: float = 1.0,
    q_max: float = 6.0,
    num_q: int = 8,
    chi_min: float = -0.6,
    chi_max: float = 0.6,
    num_spins: Optional[int] = None,
    num_chi1: int = 5,
    num_chi2: int = 3,
    mf_min: float = 0.0002,
    mf_max: float = 0.3,
    num_freq_nodes: int = 256,
    num_ode_steps: int = 800,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate a regular grid of SEOBNR_ODE waveforms for surrogate training on logarithmic Mf grid.

    Returns:
        mf_grid: Dimensionless frequency tensor of shape (num_freq_nodes,)
        eta: Symmetric mass ratio tensor of shape (N_samples,)
        s1: Primary aligned spin tensor of shape (N_samples,)
        s2: Secondary aligned spin tensor of shape (N_samples,)
        amp_red: Reduced amplitude tensor of shape (N_samples, num_freq_nodes)
        phase_red: Residual phase tensor of shape (N_samples, num_freq_nodes)
    """
    dev = torch.device(device)
    mf_grid = torch.logspace(
        math.log10(mf_min),
        math.log10(mf_max),
        num_freq_nodes,
        dtype=dtype,
        device=dev,
    )

    # Fixed fiducial total mass for geometric dimensionless conversion
    m_total = 50.0
    m_sec = m_total * MSUN_SEC
    f_grid = mf_grid / m_sec

    q_vals = torch.linspace(q_min, q_max, num_q, dtype=dtype, device=dev)
    if num_spins is not None:
        num_chi1 = num_spins
        num_chi2 = max(1, num_spins // 2)

    s1_vals = torch.linspace(chi_min, chi_max, num_chi1, dtype=dtype, device=dev)
    s2_vals = torch.linspace(chi_min, chi_max, num_chi2, dtype=dtype, device=dev)

    m1_list, m2_list, s1_list, s2_list = [], [], [], []
    for q in q_vals:
        for c1 in s1_vals:
            for c2 in s2_vals:
                m1 = m_total * q / (1.0 + q)
                m2 = m_total / (1.0 + q)
                m1_list.append(m1)
                m2_list.append(m2)
                s1_list.append(c1)
                s2_list.append(c2)

    m1 = torch.stack(m1_list)
    m2 = torch.stack(m2_list)
    s1 = torch.stack(s1_list)
    s2 = torch.stack(s2_list)

    eta = component_masses_to_eta(m1, m2)

    # Generate batch of waveforms via pure PyTorch ODE integrator
    hp, _ = SEOBNR_ODE.__call__(
        f=f_grid,
        mass1=m1,
        mass2=m2,
        chi1=s1,
        chi2=s2,
        distance=100.0,
        num_steps=num_ode_steps,
    )

    amp = torch.abs(hp)
    phase = unwrap_phase(torch.angle(hp), dim=-1)

    # Factor out Newtonian leading amplitude A0
    dist_sec = (100.0 * MPC_SI) / C_SI
    a0 = (
        (torch.sqrt(5.0 * eta.unsqueeze(-1) / 24.0) / (PI ** (2.0 / 3.0) * dist_sec))
        * (m_sec ** (5.0 / 6.0))
        * (f_grid.unsqueeze(0) ** (-7.0 / 6.0))
    )
    amp_red = (amp / a0) - 1.0

    # Factor out leading 0PN phase 3/(128*eta*v^5)
    v = (PI * mf_grid.unsqueeze(0)) ** (1.0 / 3.0)
    v5 = v ** 5.0
    phi_0pn = 3.0 / (128.0 * eta.unsqueeze(-1) * v5)
    phase_red = phase - phi_0pn

    return mf_grid, eta, s1, s2, amp_red, phase_red


def compute_svd_reduced_basis(
    data: torch.Tensor,
    num_modes: int = 12,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Perform SVD reduced basis decomposition on waveform data matrix.

    Args:
        data: Shape (N_samples, N_freq)
        num_modes: Number of dominant singular vectors to retain

    Returns:
        basis: Shape (num_modes, N_freq), orthonormal row vectors
        coeffs: Shape (N_samples, num_modes), projection coefficients
    """
    _, _, vh = torch.linalg.svd(data, full_matrices=False)
    k = min(num_modes, vh.shape[0])
    basis = vh[:k]  # (k, N_freq)
    coeffs = data @ basis.T  # (N_samples, k)
    return basis, coeffs


def fit_parameter_mapping(
    features: torch.Tensor,
    coefficients: torch.Tensor,
    l2_reg: float = 1e-6,
) -> torch.Tensor:
    """Fit polynomial regression mapping from features to basis coefficients.

    Args:
        features: Shape (N_samples, N_features)
        coefficients: Shape (N_samples, num_modes)

    Returns:
        weights: Shape (N_features, num_modes)
    """
    reg = l2_reg * torch.eye(features.shape[1], dtype=features.dtype, device=features.device)
    lhs = features.T @ features + reg
    rhs = features.T @ coefficients
    weights = torch.linalg.solve(lhs, rhs)
    return weights


def train_and_export_surrogate(
    output_path: Optional[str] = None,
    num_q: int = 8,
    num_spins: Optional[int] = None,
    num_chi1: int = 5,
    num_chi2: int = 3,
    num_freq_nodes: int = 256,
    num_modes: Optional[Union[int, Tuple[int, int]]] = None,
    num_modes_amp: int = 12,
    num_modes_phase: int = 16,
    degree: int = 3,
    num_ode_steps: int = 800,
    device: str = "cpu",
    feature_type: str = "chebyshev",
) -> str:
    """Train native PyTorch SEOBNR ROM surrogate and save weights dictionary.

    Args:
        output_path: Target .pt file path (defaults to src/torchwave/data/seobnr_surrogate.pt)
        num_q: Number of mass-ratio grid points
        num_spins: Optional shortcut to set number of spin grid points
        num_chi1: Number of primary spin grid points
        num_chi2: Number of secondary spin grid points
        num_freq_nodes: Number of dimensionless frequency nodes (default 256)
        num_modes: Optional shortcut: if int, sets both amp and phase modes
        num_modes_amp: Number of SVD amplitude basis modes (default 12)
        num_modes_phase: Number of SVD phase basis modes (default 16)
        degree: Polynomial feature degree
        num_ode_steps: Number of ODE integration steps
        device: PyTorch device ('cpu' or 'cuda'/'mps')
        feature_type: 'chebyshev' or 'poly'

    Returns:
        Absolute path to saved model file
    """
    if output_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        output_path = os.path.join(base_dir, "data", "seobnr_surrogate.pt")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if num_modes is not None:
        if isinstance(num_modes, (list, tuple)):
            n_amp, n_phase = num_modes[0], num_modes[1]
        else:
            n_amp = num_modes
            n_phase = num_modes
    else:
        n_amp = num_modes_amp
        n_phase = num_modes_phase

    # 1. Generate EOB training grid
    mf_grid, eta, s1, s2, amp_red, phase_red = generate_eob_training_grid(
        num_q=num_q,
        num_spins=num_spins,
        num_chi1=num_chi1,
        num_chi2=num_chi2,
        num_freq_nodes=num_freq_nodes,
        num_ode_steps=num_ode_steps,
        device=device,
    )

    # 2. Extract orthonormal reduced bases via SVD
    v_a, c_a = compute_svd_reduced_basis(amp_red, num_modes=n_amp)
    v_p, c_p = compute_svd_reduced_basis(phase_red, num_modes=n_phase)

    # 3. Fit parameter-space mappings covering (eta, chi1, chi2)
    if feature_type.lower() == "chebyshev":
        phi = chebyshev_features(eta, s1, s2, degree=degree)
    else:
        phi = poly_features(eta, s1, s2, degree=degree)

    w_a = fit_parameter_mapping(phi, c_a)
    w_p = fit_parameter_mapping(phi, c_p)

    # 4. Save state dictionary
    surrogate_data = {
        "mf_grid": mf_grid.detach().cpu().to(torch.float32),
        "amp_basis": v_a.detach().cpu().to(torch.float32),
        "phase_basis": v_p.detach().cpu().to(torch.float32),
        "amp_weights": w_a.detach().cpu().to(torch.float32),
        "phase_weights": w_p.detach().cpu().to(torch.float32),
        "degree": degree,
        "feature_type": feature_type,
        "has_two_spins": True,
        "metadata": {
            "model_name": "torchwave_seobnr_rom_v2",
            "approximant": "SEOBNR_ROM",
            "generator": "torchwave.approximants.eob.SEOBNR_ODE",
            "license": "MIT",
            "num_modes_amp": n_amp,
            "num_modes_phase": n_phase,
            "degree": degree,
            "parameters": ["eta", "chi1", "chi2"],
        },
    }

    torch.save(surrogate_data, output_path)
    return output_path

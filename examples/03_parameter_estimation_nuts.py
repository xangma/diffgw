#!/usr/bin/env python3
"""Tutorial 3: Differentiable Bayesian Parameter Estimation & NUTS MCMC

This script demonstrates:
1. Setting up a synthetic gravitational-wave injection in Advanced LIGO noise.
2. Constructing an accelerated Relative Binning Likelihood (heterodyned).
3. Fast gradient-based Maximum A Posteriori (MAP) estimation via Adam optimization.
4. Sampling the posterior with the No-U-Turn Sampler (NUTS) MCMC.
5. Evaluating Gelman-Rubin convergence diagnostics across parallel chains.
"""

import torch
import torchwave
from torchwave.approximants import TaylorF2
from torchwave.detectors import aLIGOZeroDetHighPower
from torchwave.inference import RelativeBinningLikelihood, NUTS, gelman_rubin


def mc_eta_to_m1_m2(mc: torch.Tensor, eta: torch.Tensor):
    """Convert chirp mass and symmetric mass ratio to component masses."""
    M = mc / (torch.clamp(eta, 1e-4, 0.25) ** 0.6)
    delta = torch.sqrt(torch.clamp(1.0 - 4.0 * eta, min=1e-8))
    m1 = 0.5 * M * (1.0 + delta)
    m2 = 0.5 * M * (1.0 - delta)
    return m1, m2


def main():
    print("=" * 70)
    print("DiffGW Tutorial 3: Differentiable Parameter Estimation & NUTS MCMC")
    print("=" * 70)
    torch.manual_seed(42)

    # -------------------------------------------------------------------------
    # 1. Setup Injection & Detector Noise
    # -------------------------------------------------------------------------
    f_dense = torch.linspace(30.0, 200.0, 1024, dtype=torch.float64)
    df = float(f_dense[1] - f_dense[0])
    psd_vals = aLIGOZeroDetHighPower(f_dense)

    # True injected parameters (GW150914-like binary black hole inspiral)
    mc_true = 28.16   # Solar masses
    eta_true = 0.248  # Symmetric mass ratio
    m1_true, m2_true = mc_eta_to_m1_m2(torch.tensor(mc_true), torch.tensor(eta_true))

    print(f"\n[1] Injected Gravitational-Wave Signal:")
    print(f"  Chirp mass: {mc_true:.2f} Msun, Symmetric mass ratio eta: {eta_true:.4f}")
    print(f"  Component masses: m1 = {m1_true.item():.2f} Msun, m2 = {m2_true.item():.2f} Msun")
    print(f"  Frequency range: [{f_dense[0]:.1f}, {f_dense[-1]:.1f}] Hz (N={len(f_dense)})")

    model = TaylorF2()
    hp_true, _ = model(f=f_dense, mass1=m1_true, mass2=m2_true, distance=400.0)

    # Inject into realistic detector noise
    noise_std = torch.sqrt(psd_vals / (4.0 * df))
    noise_fd = torch.randn_like(f_dense) * noise_std * 0.1  # High SNR injection
    data_fd = hp_true + noise_fd

    # -------------------------------------------------------------------------
    # 2. Accelerated Relative Binning Likelihood
    # -------------------------------------------------------------------------
    print("\n[2] Precomputing Relative Binning Likelihood (20 summary bins)...")
    rel_lik = RelativeBinningLikelihood(
        data={"H1": data_fd},
        psds={"H1": psd_vals},
        h0={"H1": hp_true},
        frequencies=f_dense,
        n_bins=20,
        df=df,
    )
    bin_centers = rel_lik.bin_centers

    # Scaled parameters: u = [(mc - mc_true) / 0.05, (eta - eta_true) / 0.005]
    def log_posterior(u: torch.Tensor) -> torch.Tensor:
        mc = mc_true + u[0] * 0.05
        eta = eta_true + u[1] * 0.005
        if eta > 0.2499 or eta < 0.20 or mc < 27.0 or mc > 29.5:
            return torch.tensor(-1e8, dtype=torch.float64)
        m1, m2 = mc_eta_to_m1_m2(mc, eta)
        hp_summary, _ = model(f=bin_centers, mass1=m1, mass2=m2, distance=400.0)
        return rel_lik({"H1": hp_summary}).squeeze()

    # -------------------------------------------------------------------------
    # 3. Gradient-Based MAP Optimization via Adam
    # -------------------------------------------------------------------------
    print("\n[3] Running gradient-based MAP optimization (Adam)...")
    u_param = torch.tensor([0.05, -0.05], dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([u_param], lr=0.01)

    for step in range(30):
        optimizer.zero_grad()
        loss = -log_posterior(u_param)
        loss.backward()
        optimizer.step()

    mc_map = (mc_true + u_param[0] * 0.05).item()
    eta_map = (eta_true + u_param[1] * 0.005).item()
    print(f"  MAP Optimization complete (30 steps).")
    print(f"  MAP Estimate: Chirp Mass = {mc_map:.4f} Msun, eta = {eta_map:.4f}")

    # -------------------------------------------------------------------------
    # 4. Posterior Sampling with No-U-Turn Sampler (NUTS)
    # -------------------------------------------------------------------------
    print("\n[4] Sampling posterior using No-U-Turn Sampler (NUTS) across 4 chains...")
    sampler = NUTS(log_posterior, target_accept=0.65, max_depth=5)
    init_positions = torch.tensor([
        [0.05, -0.05],
        [-0.05, 0.05],
        [0.02, 0.02],
        [-0.02, -0.02],
    ], dtype=torch.float64)

    # 50 post-warmup samples across 4 chains
    chains_u = sampler.sample_parallel(init_positions, n_samples=50, n_warmup=30)
    chains_mc = mc_true + chains_u[:, :, 0] * 0.05
    chains_eta = eta_true + chains_u[:, :, 1] * 0.005
    chains = torch.stack([chains_mc, chains_eta], dim=-1)

    mean_mc = chains[:, :, 0].mean().item()
    std_mc = chains[:, :, 0].std().item()
    mean_eta = chains[:, :, 1].mean().item()
    std_eta = chains[:, :, 1].std().item()

    r_hat = gelman_rubin(chains)

    print(f"\n[5] Posterior Analysis:")
    print(f"  Chirp Mass (M_c): {mean_mc:.4f} +/- {std_mc:.4f} Msun (True: {mc_true:.4f})")
    print(f"  Mass Ratio (eta): {mean_eta:.4f} +/- {std_eta:.4f} (True: {eta_true:.4f})")
    print(f"  Gelman-Rubin R_hat: M_c = {r_hat[0].item():.3f}, eta = {r_hat[1].item():.3f}")

    print("\n" + "=" * 70)
    print("Tutorial 3 completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()

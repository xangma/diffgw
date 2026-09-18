#!/usr/bin/env python3
"""Tutorial 2: Forward-Mode AD & Fisher Information Matrix (FIM) Analysis

This script demonstrates:
1. Exact parameter derivatives using forward-mode autodiff in PyTorch and JAX.
2. Computing the noise-weighted Fisher Information Matrix (FIM) with aLIGO PSD.
3. Inverting the Fisher matrix to obtain parameter covariance and 1-sigma errors.
4. Validating cross-framework numerical agreement of the Fisher matrix.
"""

import numpy as np
import torch
import torchwave
from torchwave.inference.fisher import fisher_information_matrix as torch_fim, covariance_matrix as torch_cov
from torchwave.detectors import aLIGOZeroDetHighPower as torch_psd

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

import jaxwave
from jaxwave.inference.fisher import fisher_information_matrix as jax_fim
from jaxwave.detectors import aLIGOZeroDetHighPower as jax_psd


def main():
    print("=" * 70)
    print("DiffGW Tutorial 2: Fisher Information Matrix & Parameter Uncertainties")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # 1. Setup Physical Parameters and Frequency Grid
    # -------------------------------------------------------------------------
    flow, fhigh, n_freqs = 20.0, 512.0, 500
    freqs_t = torch.linspace(flow, fhigh, n_freqs, dtype=torch.float64)
    df = float(freqs_t[1] - freqs_t[0])
    psd_t = torch_psd(freqs_t)

    freqs_j = jnp.linspace(flow, fhigh, n_freqs, dtype=jnp.float64)
    psd_j = jax_psd(freqs_j)

    params = {"mass1": 30.0, "mass2": 25.0, "chi1": 0.1, "chi2": -0.05}
    param_order = ["mass1", "mass2", "chi1", "chi2"]

    print(f"\nEvaluating Fisher Matrix for: {params}")
    print(f"Frequency Grid: [{flow:.1f}, {fhigh:.1f}] Hz (N={n_freqs}, df={df:.3f} Hz)")
    print("Detector: Advanced LIGO (Zero-Detuned High-Power PSD)")

    # -------------------------------------------------------------------------
    # 2. Compute FIM via PyTorch Forward-Mode AD
    # -------------------------------------------------------------------------
    def wf_fn_torch(p):
        return torchwave.get_fd_waveform(
            "IMRPhenomD",
            mass1=p["mass1"],
            mass2=p["mass2"],
            chi1=p["chi1"],
            chi2=p["chi2"],
            sample_frequencies=freqs_t,
        )

    print("\n[1] Computing Fisher Matrix with PyTorch (fwAD)...")
    fim_torch, names_t = torch_fim(
        params, wf_fn_torch, psd_t, df=df, parameter_order=param_order
    )
    cov_torch = torch_cov(fim_torch)
    sigmas_torch = torch.sqrt(torch.diag(cov_torch))

    # -------------------------------------------------------------------------
    # 3. Compute FIM via JAX Forward-Mode AD (jvp)
    # -------------------------------------------------------------------------
    def wf_fn_jax(p):
        return jaxwave.get_fd_waveform(
            "IMRPhenomD",
            mass1=p["mass1"],
            mass2=p["mass2"],
            chi1=p["chi1"],
            chi2=p["chi2"],
            sample_frequencies=freqs_j,
        )

    print("[2] Computing Fisher Matrix with JAX (jvp)...")
    fim_jax, names_j = jax_fim(
        params, wf_fn_jax, psd_j, df=df, parameter_order=param_order, method="autodiff"
    )
    cov_jax = jnp.linalg.inv(fim_jax)
    sigmas_jax = jnp.sqrt(jnp.diag(cov_jax))

    # -------------------------------------------------------------------------
    # 4. Numerical Parity & Uncertainty Reporting
    # -------------------------------------------------------------------------
    t_mat = fim_torch.detach().cpu().numpy()
    j_mat = np.asarray(fim_jax)
    rel_diff = np.max(np.abs(t_mat - j_mat) / (np.abs(j_mat) + 1e-30))

    cond_t = np.linalg.cond(t_mat)
    cond_j = np.linalg.cond(j_mat)

    print(f"\n[3] Fisher Matrix Parity & Conditioning:")
    print(f"  Relative difference max(|Gamma_torch - Gamma_jax| / |Gamma_jax|): {rel_diff:.4e}")
    print(f"  Matrix condition number kappa(Gamma): {cond_t:.2e}")
    assert rel_diff < 1e-10, f"Fisher parity {rel_diff} exceeds tolerance 1e-10!"

    print("\n[4] 1-Sigma Parameter Measurement Uncertainties (Cramer-Rao Bound):")
    print(f"{'Parameter':<10} | {'True Value':<12} | {'Torch sigma':<15} | {'JAX sigma':<15}")
    print("-" * 60)
    for i, name in enumerate(param_order):
        val = params[name]
        s_t = sigmas_torch[i].item()
        s_j = sigmas_jax[i].item()
        unit = "M_sun" if "mass" in name else "dimless"
        print(f"{name:<10} | {val:<12.3f} | {s_t:<12.4e} {unit:<3} | {s_j:<12.4e} {unit:<3}")

    print("\n" + "=" * 70)
    print("Tutorial 2 completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Tutorial 1: Quickstart & Cross-Framework Parity (PyTorch vs JAX)

This script demonstrates:
1. Generating frequency-domain gravitational waveforms in PyTorch (torchwave).
2. Generating identical waveforms in JAX (jaxwave) with JIT compilation.
3. Batched multi-binary evaluation without Python loops.
4. Verifying cross-framework numerical parity to double precision (mismatch < 1e-14).
"""

import sys
import numpy as np
import torch
import torchwave
import jax
import jax.numpy as jnp
import jaxwave
import diffgw


def compute_mismatch(h1: np.ndarray, h2: np.ndarray) -> float:
    """Compute flat noise-weighted overlap mismatch 1 - <h1|h2> / (||h1|| ||h2||)."""
    inner = np.real(np.vdot(h1, h2))
    norm1 = np.real(np.vdot(h1, h1))
    norm2 = np.real(np.vdot(h2, h2))
    denom = np.sqrt(norm1 * norm2)
    if denom <= 0.0:
        return 1.0
    overlap = inner / denom
    return float(max(0.0, 1.0 - overlap))


def main():
    print("=" * 70)
    print("DiffGW Tutorial 1: Quickstart & Cross-Framework Parity Check")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # 1. Single Waveform Evaluation in PyTorch
    # -------------------------------------------------------------------------
    print("\n[1] Generating waveform in PyTorch (torchwave)...")
    m1, m2 = 30.0, 25.0
    chi1, chi2 = 0.2, -0.1
    f_lower, f_final, delta_f = 20.0, 512.0, 0.25

    hp_torch, hc_torch = torchwave.get_fd_waveform(
        approximant="IMRPhenomD",
        mass1=m1,
        mass2=m2,
        chi1=chi1,
        chi2=chi2,
        f_lower=f_lower,
        f_final=f_final,
        delta_f=delta_f,
        dtype=torch.float64,
    )
    print(f"  PyTorch hp shape: {hp_torch.shape}, dtype: {hp_torch.dtype}")

    # -------------------------------------------------------------------------
    # 2. Single Waveform Evaluation in JAX with JIT
    # -------------------------------------------------------------------------
    print("\n[2] Generating waveform in JAX (jaxwave)...")
    gen = jaxwave.WaveformGenerator(
        approximant="IMRPhenomD",
        f_lower=f_lower,
        f_final=f_final,
        delta_f=delta_f,
        use_jit=True,
    )
    hp_jax, hc_jax = gen.generate_fd_waveform(
        mass1=m1, mass2=m2, chi1=chi1, chi2=chi2
    )
    print(f"  JAX hp shape: {hp_jax.shape}, dtype: {hp_jax.dtype}")

    # -------------------------------------------------------------------------
    # 3. Double-Precision Cross-Framework Parity Verification
    # -------------------------------------------------------------------------
    print("\n[3] Verifying mathematical parity between PyTorch and JAX...")
    hp_t_np = hp_torch.detach().cpu().numpy()
    hp_j_np = np.asarray(hp_jax)

    l2_error = np.linalg.norm(hp_t_np - hp_j_np) / np.linalg.norm(hp_j_np)
    mismatch = compute_mismatch(hp_t_np, hp_j_np)

    print(f"  Relative L2 Error: {l2_error:.4e}")
    print(f"  Waveform Mismatch: {mismatch:.4e}")
    assert l2_error < 1e-11, f"L2 error {l2_error} exceeds gate 1e-11!"
    assert mismatch < 1e-14, f"Mismatch {mismatch} exceeds gate 1e-14!"
    print("  => Parity Gate PASSED: PyTorch and JAX agree to double-precision noise.")

    # -------------------------------------------------------------------------
    # 4. Batched Waveform Generation
    # -------------------------------------------------------------------------
    print("\n[4] Batched Generation across 4 binary black holes...")
    m1_b = torch.tensor([25.0, 30.0, 45.0, 60.0], dtype=torch.float64)
    m2_b = torch.tensor([20.0, 25.0, 30.0, 40.0], dtype=torch.float64)
    chi1_b = torch.tensor([0.0, 0.2, -0.4, 0.7], dtype=torch.float64)
    chi2_b = torch.tensor([0.0, -0.1, 0.1, -0.5], dtype=torch.float64)

    hp_batch, hc_batch = torchwave.get_fd_waveform(
        approximant="IMRPhenomD",
        mass1=m1_b,
        mass2=m2_b,
        chi1=chi1_b,
        chi2=chi2_b,
        f_lower=f_lower,
        f_final=f_final,
        delta_f=delta_f,
    )
    print(f"  Batched tensor output shape: {hp_batch.shape} (B=4, N_freq={hp_batch.shape[1]})")

    # -------------------------------------------------------------------------
    # 5. Unified Dispatch via diffgw
    # -------------------------------------------------------------------------
    print("\n[5] Unified dispatch via top-level `diffgw`...")
    hp_d_t, _ = diffgw.get_fd_waveform(
        "IMRPhenomD", mass1=35.0, mass2=25.0, f_lower=f_lower, f_final=f_final, delta_f=delta_f, backend="torch"
    )
    hp_d_j, _ = diffgw.get_fd_waveform(
        "IMRPhenomD", mass1=35.0, mass2=25.0, f_lower=f_lower, f_final=f_final, delta_f=delta_f, backend="jax"
    )
    print("  Unified dispatch successful for both 'torch' and 'jax' backends.")

    print("\n" + "=" * 70)
    print("Tutorial 1 completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()

# Tutorial 1: Quickstart & Cross-Framework Parity Check

This tutorial guides you through generating gravitational waveforms with **DiffGW** in both **PyTorch (`torchwave`)** and **JAX (`jaxwave`)**, performing batched tensor evaluations, and measuring numerical agreement between the two frontends for one FP64 IMRPhenomD example.

> [!TIP]
> A complete, runnable Python script companion for this tutorial is located at [`examples/01_quickstart_and_parity.py`](../../examples/01_quickstart_and_parity.py).

---

## 1. Single Waveform Generation in PyTorch

This example evaluates waveforms in double precision (`torch.float64`) using physical astrophysical units ($M_\odot$ for masses, $\text{Mpc}$ for distance, $\text{Hz}$ for frequencies):

```python
import torch
import torchwave

hp_torch, hc_torch = torchwave.get_fd_waveform(
    approximant="IMRPhenomD",
    mass1=30.0,
    mass2=25.0,
    chi1=0.2,
    chi2=-0.1,
    f_lower=20.0,
    f_final=512.0,
    delta_f=0.25,
    dtype=torch.float64,
)

print(f"PyTorch hp shape: {hp_torch.shape}, dtype: {hp_torch.dtype}")
# Output: PyTorch hp shape: torch.Size([1969]), dtype: torch.complex128
```

---

## 2. JIT-Compiled Generation in JAX

In `jaxwave`, `WaveformGenerator` pre-caches the frequency grid geometry and compiles the waveform evaluation pipeline via `@jax.jit`:

```python
import jaxwave

gen = jaxwave.WaveformGenerator(
    approximant="IMRPhenomD",
    f_lower=20.0,
    f_final=512.0,
    delta_f=0.25,
    use_jit=True,
)

hp_jax, hc_jax = gen.generate_fd_waveform(
    mass1=30.0, mass2=25.0, chi1=0.2, chi2=-0.1
)

print(f"JAX hp shape: {hp_jax.shape}, dtype: {hp_jax.dtype}")
# Output: JAX hp shape: (1969,), dtype: complex128
```

---

## 3. Mathematical Parity Verification

For the plus polarization in this example, measure the relative $L_2$ error and the flat-weight overlap mismatch, without phase or time optimization:

$$\Delta_{L_2} = \frac{\|h_{\text{torch}} - h_{\text{jax}}\|_2}{\|h_{\text{jax}}\|_2}, \quad 1 - \mathcal{M} = 1 - \frac{\Re\langle h_{\text{torch}} \mid h_{\text{jax}} \rangle}{\|h_{\text{torch}}\| \|h_{\text{jax}}\|}$$

```python
import numpy as np

hp_t_np = hp_torch.detach().cpu().numpy()
hp_j_np = np.asarray(hp_jax)

l2_error = np.linalg.norm(hp_t_np - hp_j_np) / np.linalg.norm(hp_j_np)
inner = np.real(np.vdot(hp_t_np, hp_j_np))
norm_prod = np.sqrt(np.real(np.vdot(hp_t_np, hp_t_np)) * np.real(np.vdot(hp_j_np, hp_j_np)))
mismatch = max(0.0, 1.0 - inner / norm_prod)

print(f"Relative L2 Error: {l2_error:.4e}")
print(f"Waveform Mismatch: {mismatch:.4e}")
```

The companion script asserts $L_2 < 10^{-11}$ and mismatch $< 10^{-14}$ for this case. Passing establishes frontend consistency for these inputs; it does not independently validate the physical model or cover other parameters. The [validation guide](../validation.md) distinguishes these checks.

---

## 4. Batched Generation across Multiple Systems

Pass 1D tensors of shape `(B,)` for astrophysical parameters to compute $B$ waveforms simultaneously in one batched tensor evaluation:

```python
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
    f_lower=20.0,
    f_final=512.0,
    delta_f=0.25,
)

print("Batched tensor shape:", hp_batch.shape)  # torch.Size([4, 1969])
```

---

## 5. Unified Dispatch via `diffgw`

The top-level `diffgw` namespace dispatches seamlessly between PyTorch and JAX using the `backend` argument:

```python
import diffgw

hp_t, _ = diffgw.get_fd_waveform("IMRPhenomD", mass1=30.0, mass2=25.0, f_lower=20.0, f_final=512.0, delta_f=0.25, backend="torch")
hp_j, _ = diffgw.get_fd_waveform("IMRPhenomD", mass1=30.0, mass2=25.0, f_lower=20.0, f_final=512.0, delta_f=0.25, backend="jax")
```

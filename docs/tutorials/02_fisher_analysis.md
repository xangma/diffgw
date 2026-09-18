# Tutorial 2: Forward-Mode AD & Fisher Information Matrix Analysis

This tutorial demonstrates how to compute exact parameter derivatives with forward-mode automatic differentiation, construct the noise-weighted **Fisher Information Matrix (FIM)**, and estimate $1\text{-}\sigma$ parameter measurement uncertainties under the Cramér-Rao lower bound.

> [!TIP]
> A complete, runnable Python script companion for this tutorial is located at [`examples/02_fisher_analysis.py`](../../examples/02_fisher_analysis.py).

---

## 1. Physical Formulation of the Fisher Matrix

For a gravitational-wave signal $h(f; \theta)$ parameterized by $\theta = \{\theta_1, \dots, \theta_P\}$ detected in Gaussian stationary noise with power spectral density $S_n(f)$, the Fisher Information Matrix $\Gamma$ is defined by:

$$\Gamma_{ij} = \left\langle \frac{\partial h}{\partial \theta_i} \;\middle|\; \frac{\partial h}{\partial \theta_j} \right\rangle = 4 \Re \int_{f_{\text{low}}}^{f_{\text{high}}} \frac{\frac{\partial h^*(f)}{\partial \theta_i} \frac{\partial h(f)}{\partial \theta_j}}{S_n(f)} df$$

Under the Cramér-Rao inequality, the covariance matrix of any unbiased parameter estimator satisfies:

$$\Sigma = \Gamma^{-1}, \quad \sigma(\theta_i) \ge \sqrt{\Sigma_{ii}}$$

---

## 2. Setting Up Detector Noise & Parameters

We evaluate a binary black hole system with parameters $\theta = \{m_1, m_2, \chi_{1z}, \chi_{2z}\}$ against the Advanced LIGO Zero-Detuned High-Power PSD:

```python
import torch
import torchwave
from torchwave.inference.fisher import fisher_information_matrix as torch_fim, covariance_matrix as torch_cov
from torchwave.detectors import aLIGOZeroDetHighPower

flow, fhigh, n_freqs = 20.0, 512.0, 500
freqs = torch.linspace(flow, fhigh, n_freqs, dtype=torch.float64)
df = float(freqs[1] - freqs[0])
psd = aLIGOZeroDetHighPower(freqs)

params = {"mass1": 30.0, "mass2": 25.0, "chi1": 0.1, "chi2": -0.05}
param_order = ["mass1", "mass2", "chi1", "chi2"]
```

---

## 3. PyTorch Forward-Mode AD (`fwAD`)

`torchwave.inference.fisher_information_matrix` evaluates directional derivatives using PyTorch dual levels (`torch.autograd.forward_ad`), avoiding the memory overhead of reverse-mode tape recording:

```python
def waveform_model(p):
    return torchwave.get_fd_waveform(
        "IMRPhenomD",
        mass1=p["mass1"],
        mass2=p["mass2"],
        chi1=p["chi1"],
        chi2=p["chi2"],
        sample_frequencies=freqs,
    )

gamma, param_names = torch_fim(
    params, waveform_model, psd, df=df, parameter_order=param_order
)
cov = torch_cov(gamma)
sigmas = torch.sqrt(torch.diag(cov))

print("Matrix condition number kappa(Gamma):", torch.linalg.cond(gamma).item())
for i, name in enumerate(param_names):
    print(f"sigma({name}): {sigmas[i].item():.4e}")
```

---

## 4. JAX Forward-Mode AD (`jvp`) & Cross-Framework Parity

In JAX, directional derivatives are computed using `jax.jvp`:

```python
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import jaxwave
from jaxwave.inference.fisher import fisher_information_matrix as jax_fim
from jaxwave.detectors import aLIGOZeroDetHighPower as jax_psd

freqs_j = jnp.linspace(flow, fhigh, n_freqs, dtype=jnp.float64)
psd_j = jax_psd(freqs_j)

def jax_waveform(p):
    return jaxwave.get_fd_waveform(
        "IMRPhenomD",
        mass1=p["mass1"],
        mass2=p["mass2"],
        chi1=p["chi1"],
        chi2=p["chi2"],
        sample_frequencies=freqs_j,
    )

gamma_jax, _ = jax_fim(
    params, jax_waveform, psd_j, df=df, parameter_order=param_order, method="autodiff"
)

# Compare PyTorch and JAX Fisher matrices
rel_diff = torch.max(torch.abs(gamma - torch.from_numpy(np.asarray(gamma_jax))) / gamma).item()
print(f"Max relative difference: {rel_diff:.4e}")  # ~ 1.15e-13
```

Both frameworks produce numerically indistinguishable Fisher matrices down to floating-point precision ($10^{-13}$).

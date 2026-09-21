# Tutorial 2: Forward-Mode AD & Fisher Information Matrix Analysis

This tutorial demonstrates how to compute derivatives of the implemented waveform with forward-mode automatic differentiation, construct the noise-weighted **Fisher Information Matrix (FIM)**, and construct local uncertainty forecasts from its inverse.

> [!TIP]
> A complete, runnable Python script companion for this tutorial is located at [`examples/02_fisher_analysis.py`](../../examples/02_fisher_analysis.py).

---

## 1. Physical Formulation of the Fisher Matrix

For a gravitational-wave signal $h(f; \theta)$ parameterized by $\theta = \{\theta_1, \dots, \theta_P\}$ detected in Gaussian stationary noise with power spectral density $S_n(f)$, the Fisher Information Matrix $\Gamma$ is defined by:

$$\Gamma_{ij} = \left\langle \frac{\partial h}{\partial \theta_i} \;\middle|\; \frac{\partial h}{\partial \theta_j} \right\rangle = 4 \Re \int_{f_{\text{low}}}^{f_{\text{high}}} \frac{\frac{\partial h^*(f)}{\partial \theta_i} \frac{\partial h(f)}{\partial \theta_j}}{S_n(f)} df$$

Under the regularity assumptions of the Cramér-Rao inequality and for an invertible Fisher matrix, an unbiased estimator obeys the matrix bound

$$\operatorname{Cov}(\hat\theta) \succeq \Gamma^{-1}.$$

Using $\Sigma_F=\Gamma^{-1}$ as a posterior covariance is a local Gaussian approximation, not an identity. Check conditioning, signal strength, parameter degeneracies, and priors before interpreting $\sqrt{(\Sigma_F)_{ii}}$ as uncertainty. See [Vallisneri (2008)](https://arxiv.org/abs/gr-qc/0703086).

---

## 2. Setting Up Detector Noise & Parameters

We use the plus polarization, without a detector-network projection, for a binary black hole system with parameters $\theta = \{m_1, m_2, \chi_{1z}, \chi_{2z}\}$ against the Advanced LIGO Zero-Detuned High-Power PSD:

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

`torchwave.inference.fisher_information_matrix` evaluates directional derivatives using PyTorch dual levels (`torch.autograd.forward_ad`), differentiating one parameter direction at a time:

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
import numpy as np
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
gamma_np = gamma.detach().cpu().numpy()
gamma_jax_np = np.asarray(gamma_jax)
rel_diff = np.linalg.norm(gamma_np - gamma_jax_np) / np.linalg.norm(gamma_jax_np)
print(f"Relative matrix norm difference: {rel_diff:.4e}")
```

This comparison measures agreement for one model, grid, parameter set, and derivative calculation. It does not establish the accuracy of a posterior forecast or all waveform derivatives. The helper `covariance_matrix` floors small eigenvalues before inversion; uncertainty forecasts in regularized directions depend on that choice. The companion script checks a separate elementwise relative-error tolerance of `1e-10` for this example.

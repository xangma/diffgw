# Tutorial 3: Parameter Estimation & No-U-Turn Sampler (NUTS) MCMC

This tutorial demonstrates end-to-end Bayesian inference: injecting a synthetic gravitational wave into detector noise, accelerating the likelihood calculation via **Relative Binning**, finding the **Maximum A Posteriori (MAP)** parameter estimate with gradient descent, and sampling the posterior distribution using the **No-U-Turn Sampler (NUTS)** MCMC.

> [!TIP]
> A complete, runnable Python script companion for this tutorial is located at [`examples/03_parameter_estimation_nuts.py`](../../examples/03_parameter_estimation_nuts.py).

---

## 1. Accelerated Relative Binning Likelihood

Direct evaluation of the Whittle likelihood requires generating $h(f)$ across dense frequency grids ($N \sim 10^3 \to 10^5$ points) on every MCMC proposal step.

**Relative Binning** (Zackay et al. 2018; Dai et al. 2018) factors the test waveform $h(f; \theta)$ into a smooth ratio relative to a fiducial reference $h_0(f)$:

$$r(f; \theta) = \frac{h(f; \theta)}{h_0(f)} \approx r_0(f_b) + r_1(f_b)(f - f_b)$$

Because $r(f)$ varies slowly, the likelihood can be computed across just $M \sim 20$ to $50$ summary frequency bins, accelerating likelihood evaluation by $10\times$ to $50\times$:

```python
import torch
from torchwave.approximants import TaylorF2
from torchwave.detectors import aLIGOZeroDetHighPower
from torchwave.inference import RelativeBinningLikelihood

f_dense = torch.linspace(30.0, 200.0, 1024, dtype=torch.float64)
df = float(f_dense[1] - f_dense[0])
psd_vals = aLIGOZeroDetHighPower(f_dense)

# Injected signal (m1=35.4 Msun, m2=29.6 Msun)
model = TaylorF2()
hp_true, _ = model(f=f_dense, mass1=35.41, mass2=29.60, distance=400.0)

# Add Gaussian detector noise
noise_std = torch.sqrt(psd_vals / (4.0 * df))
data_fd = hp_true + torch.randn_like(f_dense) * noise_std * 0.1

# Initialize Relative Binning with 20 bins
rel_lik = RelativeBinningLikelihood(
    data={"H1": data_fd},
    psds={"H1": psd_vals},
    h0={"H1": hp_true},
    frequencies=f_dense,
    n_bins=20,
    df=df,
)
bin_centers = rel_lik.bin_centers  # Only 20 evaluation frequencies!
```

---

## 2. Gradient-Based MAP Optimization with Adam

Because the entire DiffGW model and relative binning likelihood are autograd-differentiable, we can find the Maximum A Posteriori estimate in tens of steps using Adam optimization:

```python
# Optimize scaled parameter offsets u = [(mc - mc0)/0.05, (eta - eta0)/0.005]
u_param = torch.tensor([0.05, -0.05], dtype=torch.float64, requires_grad=True)
optimizer = torch.optim.Adam([u_param], lr=0.01)

for step in range(30):
    optimizer.zero_grad()
    loss = -log_posterior(u_param)
    loss.backward()
    optimizer.step()

print(f"Converged MAP: u = {u_param.detach().numpy()}")
```

---

## 3. Posterior Sampling with NUTS MCMC

DiffGW includes a native PyTorch implementation of the **No-U-Turn Sampler** (Hoffman & Gelman 2011) with recursive binary tree trajectory expansion, dual-averaging step-size adaptation, and parallel chain execution:

```python
from torchwave.inference import NUTS, gelman_rubin

sampler = NUTS(log_posterior, target_accept=0.65, max_depth=5)

# Initialize 4 parallel chains around the MAP estimate
init_positions = torch.tensor([
    [0.05, -0.05],
    [-0.05, 0.05],
    [0.02, 0.02],
    [-0.02, -0.02],
], dtype=torch.float64)

# Run 4 parallel chains (50 samples, 30 warmup)
chains_u = sampler.sample_parallel(init_positions, n_samples=50, n_warmup=30)
```

---

## 4. Convergence Diagnostics (Gelman-Rubin $\hat{R}$)

The Gelman-Rubin potential scale reduction factor $\hat{R}$ measures between-chain and within-chain variances:

$$\hat{R} = \sqrt{\frac{\hat{V}}{W}} \approx 1.0$$

```python
chains_mc = mc_true + chains_u[:, :, 0] * 0.05
chains_eta = eta_true + chains_u[:, :, 1] * 0.005
chains = torch.stack([chains_mc, chains_eta], dim=-1)

r_hat = gelman_rubin(chains)
print(f"Gelman-Rubin R_hat: M_c = {r_hat[0].item():.3f}, eta = {r_hat[1].item():.3f}")
# Values close to 1.0 indicate proper chain convergence
```

# Tutorial 3: NUTS API walkthrough

This tutorial illustrates an inference workflow with a synthetic signal: approximate **relative binning**, gradient-based optimization, and **NUTS** sampling. The short run demonstrates the API; it is not a validated detector-noise simulation or a converged parameter-estimation result. The excerpts below use the parameter bounds, scaled coordinates, and `log_posterior` defined in the companion script. Despite that function's name, its finite out-of-range penalty does not define a proper uniform box prior; specify a proper prior before interpreting samples as a posterior.

> [!TIP]
> A complete, runnable Python script companion for this tutorial is located at [`examples/03_parameter_estimation_nuts.py`](../../examples/03_parameter_estimation_nuts.py).

---

## 1. Relative Binning Likelihood

Direct evaluation of the Whittle likelihood requires generating $h(f)$ across dense frequency grids ($N \sim 10^3 \to 10^5$ points) on every MCMC proposal step.

[**Relative Binning** (Zackay et al. 2018)](https://arxiv.org/abs/1806.08792) factors the test waveform $h(f; \theta)$ into a smooth ratio relative to a fiducial reference $h_0(f)$:

$$r(f; \theta) = \frac{h(f; \theta)}{h_0(f)} \approx r_0(f_b) + r_1(f_b)(f - f_b)$$

When the candidate is sufficiently close to a suitable fiducial waveform, $r(f)$ can vary slowly enough for this approximation. The bin count controls a workload-dependent accuracy/cost tradeoff. This example chooses 20 bins; validate dense-versus-binned log-likelihood and gradient errors over the intended parameter region before inference. No speed factor or error bound follows from the bin count alone.

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

# Add an illustrative real Gaussian perturbation (not full complex detector noise)
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
bin_centers = rel_lik.bin_centers  # Candidate evaluation frequencies
```

---

## 2. Gradient-Based Optimization with Adam

The TaylorF2 and relative-binning path used here supports autograd. The following 30 Adam steps optimize the illustrative penalized likelihood. Convergence requires a separate check of the objective and gradients; MAP interpretation also requires a specified prior:

```python
# Optimize scaled parameter offsets u = [(mc - mc0)/0.05, (eta - eta0)/0.005]
u_param = torch.tensor([0.05, -0.05], dtype=torch.float64, requires_grad=True)
optimizer = torch.optim.Adam([u_param], lr=0.01)

for step in range(30):
    optimizer.zero_grad()
    loss = -log_posterior(u_param)
    loss.backward()
    optimizer.step()

print(f"Optimization candidate after 30 steps: u = {u_param.detach().numpy()}")
```

---

## 3. Sampling with NUTS MCMC

DiffGW includes a native PyTorch implementation of the **No-U-Turn Sampler** ([Hoffman & Gelman 2014](https://jmlr.org/papers/v15/hoffman14a.html)) with recursive binary tree trajectory expansion and dual-averaging step-size adaptation. The `sample_parallel` method currently executes chains sequentially. This example uses fixed starting positions in scaled coordinates, independently of the Adam result:

```python
from torchwave.inference import NUTS, gelman_rubin

sampler = NUTS(log_posterior, target_accept=0.65, max_depth=5)

# Initialize 4 chains at fixed scaled positions
init_positions = torch.tensor([
    [0.05, -0.05],
    [-0.05, 0.05],
    [0.02, 0.02],
    [-0.02, -0.02],
], dtype=torch.float64)

# Run 4 chains sequentially (50 samples and 30 warmup steps per chain)
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
# Values near 1 are one diagnostic; they do not establish convergence
```

For an inference result, increase sampling as needed and inspect mixing, effective sample sizes, divergences, and stability across runs. Check relative-binning errors on posterior samples against the dense likelihood; sampler diagnostics do not test waveform or likelihood approximation accuracy.

# Inference and detector analysis

Detector projections, noise models, likelihoods, and parameter-estimation examples.

Use [the public PE workflow](parameter-estimation.md) for `diffgw.inference`
priors, posteriors and NUTS sampling in Torch or JAX on an explicit CPU/GPU device.
The native APIs below provide detector analysis and advanced likelihood tools.

<a id="10-detector-networks-projections"></a>
## Detector networks and projections

### Ground-Based Networks (`H1`, `L1`, `V1`, `K1`, `ET`, `CE`)
```python
from torchwave.detectors import DetectorNetwork

network = DetectorNetwork(["H1", "L1", "V1"])
f_grid = torch.arange(20.0, 512.0 + 0.125, 0.25, dtype=torch.float64)
hp, hc = torchwave.get_fd_waveform(approximant="IMRPhenomD", mass1=30.0, mass2=25.0, sample_frequencies=f_grid)

strains = network.project_fd_strain(
    f=f_grid, h_plus=hp, h_cross=hc,
    ra=1.5, dec=0.3, psi=0.8, t_gps=1126259462.0,
)
```

### Power Spectral Densities (PSDs)
Available in both `torchwave.detectors.psd` and `jaxwave.detectors.psd`:
- `aLIGOZeroDetHighPower`
- `aLIGOEarlyHighPower`
- `AdvancedVirgo`
- `KAGRA`
- `EinsteinTelescope`
- `CosmicExplorer`

### Space-Based Interferometers (`LISANetwork`)
TDI channels ($A, E, T$) with galactic confusion noise (`torchwave.detectors.LISANetwork` and `jaxwave.detectors.LISANetwork`).

### Pulsar Timing Arrays (`PTANetwork`)
Hellings-Downs correlation covariance matrix (`torchwave.detectors.PTANetwork` and `jaxwave.detectors.PTANetwork`).

---


<a id="11-bayesian-inference-suite"></a>
## Bayesian inference suite

Both `torchwave.inference` and `jaxwave.inference` provide high-performance parameter estimation tools.

### Signal-to-Noise Ratio (SNR)

Compute optimal SNR $\rho_{\text{opt}} = \sqrt{\langle h \mid h \rangle}$ across single detectors or multi-detector networks:

```python
# PyTorch
from torchwave.inference import optimal_snr, matched_filter_snr
from torchwave.detectors.psd import aLIGOZeroDetHighPower

psd = aLIGOZeroDetHighPower(f_grid)
snr_val = optimal_snr(hp, psd, df=0.25)
print(f"PyTorch Optimal SNR: {snr_val.item():.2f}")

# JAX
from jaxwave.inference.likelihood import inner_product
from jaxwave.detectors.psd import aLIGOZeroDetHighPower as jax_psd

snr_jax = jnp.sqrt(inner_product(hp_jax, hp_jax, jax_psd(f_grid), df=0.25))
print(f"JAX Optimal SNR: {snr_jax:.2f}")
```

### Whittle Gaussian Likelihood

Evaluate the discrete Whittle log-likelihood for stationary Gaussian noise with a fixed PSD, up to a parameter-independent normalization:

$$\ln \mathcal{L}(\theta) = -\frac{1}{2} \sum_{k} \langle d_k - h_k(\theta) \mid d_k - h_k(\theta) \rangle$$

```python
# PyTorch Whittle Likelihood
from torchwave.inference import WhittleLikelihood

whittle_pt = WhittleLikelihood(data=strains, psds=psds, df=0.25)
log_lik_pt = whittle_pt(strains)
log_lik_marg = whittle_pt.phase_marginalized_log_likelihood(strains)

# JAX Whittle Likelihood
from jaxwave.inference.likelihood import WhittleLikelihood as JAXWhittleLikelihood

whittle_jx = JAXWhittleLikelihood(data=strains_jax, psds=psds_jax, df=0.25)
log_lik_jx = whittle_jx(template_jax)
```

### Accelerated Relative Binning Likelihood

Both frontends implement a **relative-binning approximation** using precomputed summary coefficients and candidate waveforms at bin centers. The ratio to a suitable fiducial waveform should vary slowly within each bin. Choose the bin count and validate likelihood and gradient errors against dense evaluation over the relevant parameter region. Runtime depends on the workload; see [validation evidence](../validation/inference.md#relative-binning) and the [method paper](https://arxiv.org/abs/1806.08792).

```python
# PyTorch Relative Binning
from torchwave.inference import RelativeBinningLikelihood

rel_bin = RelativeBinningLikelihood(
    data=strains,
    psds=psds,
    h0=strains,             # Fiducial reference waveform
    frequencies=f_grid,
    n_bins=40,              # Chosen bin count; validate for the intended parameter region
    df=0.25,
)

# During MCMC sampling, evaluate candidate templates ONLY at bin centers (e.g. 40 points)
hp_bin, hc_bin = torchwave.get_fd_waveform(
    approximant="IMRPhenomD",
    mass1=30.05, mass2=24.95,
    sample_frequencies=rel_bin.bin_centers,
)
template_bins = network.project_fd_strain(
    f=rel_bin.bin_centers,
    h_plus=hp_bin, h_cross=hc_bin,
    ra=1.5, dec=0.3, psi=0.8, t_gps=1126259462.0,
)
fast_log_lik = rel_bin(template_bins)

# JAX Relative Binning
from jaxwave.inference.relative_binning import RelativeBinningLikelihood as JAXRelativeBinningLikelihood

# Supply matching detector dictionaries and a fiducial waveform on the dense grid.
rb_jax = JAXRelativeBinningLikelihood(
    data=strains_jax, psds=psds_jax, h0=h0_jax,
    frequencies=f_grid_jax, n_bins=40, df=0.25,
)
# Evaluate candidate detector templates at rb_jax.bin_centers.
fast_ll_jax = rb_jax(template_bins_jax)
```

### Fisher Information Matrix & Sky Localization

Compute local Fisher uncertainty forecasts using automatic differentiation. Interpreting the inverse matrix as a posterior covariance requires a suitable local Gaussian approximation; see the [Fisher tutorial](../tutorials/02_fisher_analysis.md).

$$\Gamma_{ij} = \sum_k 4 \, \text{Re} \int_{f_{\text{low}}}^{f_{\text{high}}} \frac{1}{S_n^k(f)} \frac{\partial h_k(f)}{\partial \theta_i} \left(\frac{\partial h_k(f)}{\partial \theta_j}\right)^* df$$

```python
# PyTorch Fisher Matrix
from torchwave.inference import fisher_information_matrix, covariance_matrix

params = {"mass1": 30.0, "mass2": 25.0}

def waveform_model(p):
    hp_m, _ = torchwave.get_fd_waveform(
        approximant="IMRPhenomD",
        mass1=p["mass1"], mass2=p["mass2"],
        sample_frequencies=f_grid,
    )
    return hp_m

gamma, param_names = fisher_information_matrix(
    parameters_dict=params,
    waveform_fn=waveform_model,
    psd_dict=psds["H1"],
    frequencies=f_grid,
    df=0.25,
)
cov = covariance_matrix(gamma)
print(f"PyTorch sigma(mass1): {torch.sqrt(cov[0, 0]).item():.4f} Msun")

# JAX Fisher Matrix
from jaxwave.inference.fisher import fisher_information_matrix as jax_fisher
from jaxwave.inference.fisher import covariance_matrix as jax_covariance

def jax_wf_fn(p):
    hp_m, _ = jaxwave.get_fd_waveform(
        approximant="IMRPhenomD",
        mass1=p["mass1"], mass2=p["mass2"],
        sample_frequencies=f_grid_jax,
    )
    return hp_m

gamma_jax, param_names_jax = jax_fisher(
    parameters_dict=params, waveform_fn=jax_wf_fn,
    psd_dict=psds_jax["H1"], frequencies=f_grid_jax, df=0.25,
)
cov_jax = jax_covariance(gamma_jax)
print(f"JAX sigma(mass1): {jnp.sqrt(cov_jax[0, 0]):.4f} Msun")
```

### Gradient-Based MAP Optimization

The `optimize_map` helper optimizes the log likelihood with Adam and optional box bounds. This is equivalent to MAP optimization for a uniform prior within those bounds; the helper does not accept a general prior. Check convergence separately from the requested iteration count:

```python
from torchwave.inference import optimize_map

best_params, loss_history = optimize_map(
    data=strains["H1"],
    psd=psds["H1"],
    initial_params={"mass1": 31.0, "mass2": 24.0},
    waveform_fn=lambda p: torchwave.get_fd_waveform(
        "IMRPhenomD", mass1=p["mass1"], mass2=p["mass2"], sample_frequencies=f_grid
    )[0],
    frequencies=f_grid,
    steps=100,
    lr=0.05,
)
```

### No-U-Turn Sampler (NUTS) MCMC

The sampler provides Hamiltonian Monte Carlo with dual-averaging step-size adaptation. Supply a differentiable `log_posterior(theta)` that includes the intended priors and likelihood, using the same detector projection for data and templates. The fragment below assumes that function is defined; `sample_parallel` currently runs the chains sequentially. Gelman-Rubin statistics are one diagnostic and do not establish convergence by themselves.

```python
from torchwave.inference import NUTS, gelman_rubin

sampler = NUTS(log_prob_fn=log_posterior, target_accept=0.65)
initial_positions = torch.tensor([[30.1, 24.9], [29.8, 25.2]])
chains = sampler.sample_parallel(initial_positions, n_samples=300, n_warmup=150)
r_hat = gelman_rubin(chains)
print("Gelman-Rubin R_hat:", r_hat)
```

---

# torchwave

**torchwave** is a high-performance, batched, autograd-differentiable gravitational-wave (GW) waveform generator built from scratch in PyTorch.

Developed under a **strict clean-room protocol**, `torchwave` is free of copyleft / GPL obligations and licensed under the permissive **MIT License**. All mathematical formulas, post-Newtonian expansions, collocation points, and fit coefficients are derived exclusively from published peer-reviewed journal papers and arXiv preprints.

---

## Key Features

- **Pure PyTorch Core**: Evaluates directly on `cpu`, `cuda`, or Apple Silicon `mps` devices without host-device memory roundtrips.
- **Batched by Design**: Pass tensors of shape `(B,)` for masses and spins to generate waveforms of shape `(B, N)` simultaneously.
- **Autograd Differentiable**: Fully supports PyTorch automatic differentiation ($\partial h(f)/\partial\theta$) for Hamiltonian Monte Carlo (HMC), Fisher information matrices, and gradient-based neural network pipelines.
- **Standard GW API**: Intuitive `get_fd_waveform` interface.

---

## Supported Approximants

| Approximant | Regimes | Physics | Clean-Room Reference |
| :--- | :--- | :--- | :--- |
| **`TaylorF2`** | Inspiral | 3.5PN point-particle, spin-orbit (1.5PN, 2.5PN, 3.5PN), spin-spin (2PN), 5PN/6PN tidal ($\tilde{\Lambda}, \delta\tilde{\Lambda}$) | Blanchet (2014), Buonanno et al. (2009), Vines et al. (2011) |
| **`IMRPhenomD`** | Inspiral, Merger, Ringdown | Aligned spin ($q \le 18, |\chi_i| \le 0.85$), calibrated collocation corrections, modified Lorentzian ringdown | Husa et al. (2016), Khan et al. (2016), Berti et al. (2006) |

---

## Quickstart

```python
import torch
import torchwave

# Generate equal-mass binary black hole waveform
hp, hc = torchwave.get_fd_waveform(
    approximant="IMRPhenomD",
    mass1=30.0,
    mass2=30.0,
    chi1=0.2,
    chi2=-0.1,
    f_lower=20.0,
    f_final=1024.0,
    delta_f=0.25,
    distance=410.0,
    device="cpu",
)

# Batched generation
m1 = torch.tensor([20.0, 30.0, 40.0])
m2 = torch.tensor([15.0, 30.0, 35.0])
hp_batch, hc_batch = torchwave.get_fd_waveform(
    approximant="IMRPhenomD",
    mass1=m1,
    mass2=m2,
    f_lower=20.0,
    f_final=512.0,
    delta_f=1.0,
)
print(hp_batch.shape)  # torch.Size([3, 493])
```

---

## Scientific Provenance & Clean-Room Compliance

See [CITATIONS.md](CITATIONS.md) for full BibTeX citations. No source code from LALSuite or GPL repositories was inspected, referenced, or utilized in the creation of this library.

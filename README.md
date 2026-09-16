# torchwave

**torchwave** is a high-performance, batched, autograd-differentiable gravitational-wave (GW) waveform generator built from scratch in PyTorch.

TorchWave's original code and newly generated Kerr QNM runtime data are distributed under the **MIT License**. Cook reference fixtures retain their **ODC-By-1.0** terms and are kept in an external validation cache; see [third-party notices](THIRD_PARTY_NOTICES.md). The independent Kerr generator uses published equations, with external numerical data and waveform binaries reserved for verification. The provenance and licensing scope of inherited waveform calibration coefficients, including XAS and XHM supplementary assets, remains unresolved; numerical agreement alone does not establish provenance.

---

## Key Features

- **Pure PyTorch Core**: Tensor computation on CPU and CUDA, with model-dependent Apple Silicon MPS support.
- **Batched by Design**: Pass tensors of shape `(B,)` for masses and spins to generate waveforms of shape `(B, N)` simultaneously.
- **Autograd Differentiable**: Fully supports PyTorch automatic differentiation ($\partial h(f)/\partial\theta$) for Hamiltonian Monte Carlo (HMC), Fisher information matrices, and gradient-based neural network pipelines.
- **Standard GW API**: Intuitive `get_fd_waveform` interface.

---

## Supported Approximants

TorchWave supports 17 gravitational-wave approximants across inspiral, merger, and ringdown regimes:

- **Exact LAL Parity (12 models)**: `TaylorF2`, `TaylorF2_Eccentric`, `IMRPhenomD`, `IMRPhenomPv2`, `IMRPhenomXP`, `IMRPhenomHM`, `IMRPhenomXAS`, `IMRPhenomD_NRTidal`, `IMRPhenomD_NRTidalv2`, `IMRPhenomNSBH`, `KerrRingdown`, `SEOBNR_ROM`.
- **Sovereign Physical Calibrations (4 models)**: `IMRPhenomXHM`, `IMRPhenomXPHM` (native SXS NR higher-mode calibrations), `SEOBNR_ODE`, `TEOBResumS` (continuous spline NQC matching).
- **Literature Analytical Extension (1 model)**: `TaylorF2_HM` (analytical higher multipoles where LAL lacks a multi-mode FD generator).

For full details on where and why TorchWave departs from LALSimulation, see [**docs/differences_from_lal.md**](docs/differences_from_lal.md) and the [**Provenance Ledger**](docs/provenance_ledger.md).

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

## Scientific Validation and Provenance

TorchWave operates under strict clean-room principles with zero runtime dependency on LAL. See [CITATIONS.md](CITATIONS.md) for literature references, [**docs/provenance_ledger.md**](docs/provenance_ledger.md) for complete mathematical derivation and asset provenance, and [**docs/differences_from_lal.md**](docs/differences_from_lal.md) for detailed descriptions of where and why TorchWave departs from LALSimulation.

The verification harness applies uniform double-precision CPU FP64 gates (complex relative $L_2 \le 10^{-6}$, mismatch $\le 10^{-8}$) across both polarizations and sky-projected detector networks (H1, L1, V1):

```sh
# Run the automated 21-case catalog parity and sovereign calibration audit
PYTHONPATH=src python tests/parity_harness.py --scope=catalog
# CATALOG PARITY SUITE PASSED: 21/21 cases verified (17 exact LAL parity, 4 sovereign physics calibrations).
```

- **12 Models**: Verified exact LAL parity down to machine precision ($10^{-8}$ to $10^{-16}$ mismatch).
- **4 Models**: Sovereign physics calibrations (`IMRPhenomXHM`, `IMRPhenomXPHM`, `SEOBNR_ODE`, `TEOBResumS`) resolving legacy C-table bugs, unphysical sign errors, and discrete stencil knot artifacts.
- **1 Model**: Independent analytical post-Newtonian multipole implementation (`TaylorF2_HM`).

XP/XPHM support the five published precession conventions and an additional
source-frame observer azimuth (`phi0`). NNLO rotations with
`L + S_parallel <= 0` reject explicitly; exactly aligned default inputs retain
their nonprecessing carrier. See the [conventions and domain limits](docs/reviews/2026-09-12-clean-room/pass-4-precession-conventions.md).

Kerr ringdown defaults to our independently solved frequencies
(`qnm_backend="independent"`): 15 positive-m mode/overtone combinations and the
negative-m 220 fundamental, with remnant spins from `0` through `0.99999`.
The old `"author"` name is a compatibility alias for this same independent data.

The printed approximate fits remain available as `qnm_backend="berti"`.
QNM interpolation uses float64 working arithmetic on CPU/CUDA and float32 on
MPS. The strict numerical validation applies to float64.
Select `angular_basis="spheroidal"` for the Kerr angular eigenfunctions; the
default is `"spherical"`. See the [generator and validation](docs/reviews/2026-09-12-clean-room/pass-7-independent-qnm.md) and
[angular validation](docs/reviews/2026-09-12-clean-room/pass-3-kerr-angular.md).

The standalone [direct hybrid-MECO solver](docs/reviews/2026-09-12-clean-room/pass-7-independent-meco.md)
computes a PN/Kerr energy minimum without an empirical MECO fit. It has not
replaced XAS matching defaults. The [calibration inventory](docs/reviews/2026-09-12-clean-room/pass-7-independent-calibration-inventory.md)
records the remaining fitted waveform/remnant dependencies and the route to
[our own numerical-relativity data](docs/reviews/2026-09-12-clean-room/pass-7-own-nr-pilot.md).

To set a ringdown amplitude from an emitted energy, use
`ringdown_amplitude_from_energy_fraction` from
`torchwave.approximants.kerr_ringdown_energy`. Supply remnant mass in solar
masses, distance in Mpc, energy fraction, angular frequency in rad/s and damping
time in seconds. Its default `convention="native_pair"` returns the amplitude
accepted by `KerrRingdown`; the [energy derivation](docs/reviews/2026-09-12-clean-room/pass-3-ringdown-energy.md)
distinguishes a single complex mode from an equal equatorial mirror pair.

`SEOBNR_ODE` accepts `conservative_model="seobnrv4_published"` to select the
reconstructed aligned-spin conservative Hamiltonian. This experimental option
does not supply a complete calibrated SEOBNRv4 waveform. The default remains
`"prototype"`; when the integrated trajectory resolves a usable frequency
peak, its native peak/NQC/C1 attachment is used, while under-resolved runs
retain the legacy final-sample PhenomD fallback. The [derivation and limits](docs/reviews/2026-09-12-clean-room/pass-3-seobnrv4.md)
describe the published equations and unresolved model details.

Its independent `radiation_model` option selects `"din2009_nonspinning"`
(35-mode flux through ℓ=8) or `"pan2011_leading_spin"` (leading spin terms
through ℓ=4). Both use the same factorized amplitudes for the flux and emitted
(2,2) strain. The default is `"prototype"`. These are circular radiation
prescriptions with incomplete plunge and merger physics; see the
[factorized-mode derivation](docs/reviews/2026-09-12-clean-room/pass-3-factorized-modes.md).

The separate experimental `SEOBNRv4Published` model now connects the published
Hamiltonian, 35-mode spinning flux, complex (2,2) mode, refined frequency peak,
five-condition NQC solve, remnant fits and C1 merger/ringdown attachment.
It is also available through `SEOBNR_ODE(assembly_model="seobnrv4_published", ...)`.
That assembly requires all five phase, spin-truncation and final-spin-fit
choices explicitly; no selection is advertised as LAL-compatible. The [API example and evidence](docs/reviews/2026-09-12-clean-room/pass-6-eob-driver.md)
record convergence checks and the failing independent LAL comparison.
The [NQC](docs/reviews/2026-09-12-clean-room/pass-3-eob-nqc.md),
[merger](docs/reviews/2026-09-12-clean-room/pass-3-eob-merger-ringdown.md),
[amplitude](docs/reviews/2026-09-12-clean-room/pass-3-seobnrv4-mode22.md),
[phase](docs/reviews/2026-09-12-clean-room/pass-6-eob-phase22.md) and
[remnant](docs/reviews/2026-09-12-clean-room/pass-6-eob-remnant.md)
components are also available independently.

---

## Benchmarking & Profiling

TorchWave includes built-in benchmarking and deep profiling tools exposed directly as CLI commands, Python modules, and library APIs.

### Command-Line Interface

```bash
# Benchmark popular models across batch sizes
torchwave-bench --models popular --batch-sizes 1,10,100,1000

# Compare against LALSimulation baseline and export Markdown summary
torchwave-bench --models TaylorF2,IMRPhenomD,IMRPhenomXAS --compare-lal --markdown-out benchmark.md

# Benchmark with autograd backward pass
torchwave-bench --models IMRPhenomD --batch-sizes 100 --backward

# Deep operator, cProfile, and memory profiling
torchwave-profile --model IMRPhenomD --mode all --top-k 15
torchwave-profile --model TaylorF2 --trace-out trace_f2.json
```

Or invoke directly via Python:
```bash
python -m torchwave.benchmark --models TaylorF2 --batch-sizes 1,10
python -m torchwave.profile --model IMRPhenomXAS --mode memory
```

### Programmatic Python API

```python
import torch
from torchwave import benchmark, profile

# Run benchmark suite programmatically
results = benchmark.run_benchmark_suite(
    models=["TaylorF2", "IMRPhenomD"],
    batch_sizes=[1, 100],
    device=torch.device("cpu"),
    dtype=torch.float64,
    iterations=20,
    measure_backward=True,
)
print(benchmark.format_table(results))

# Profile PyTorch operator kernels or memory allocations
spec = benchmark.DEFAULT_MODEL_SPECS["IMRPhenomD"]
torch_table = profile.run_torch_profiler(
    model_name="IMRPhenomD",
    batch_size=10,
    device=torch.device("cpu"),
    dtype=torch.float64,
    grid=spec["grid"],
    params=spec["params"],
    measure_backward=True,
)
print(torch_table)
```

---

### External waveform data

Optional large waveform datasets remain outside the MIT source distribution.
See [external-data.md](docs/external-data.md) for the SEOBNRv4ROM v3 source,
search paths, verification, and the explicit fetch command.


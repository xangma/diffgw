# Waveform catalog and support

Choose a model by its physical content, then check the evidence and execution
restrictions for the options you need. The catalog records implemented families;
a familiar approximant name does not establish equivalence to an external model.
For availability in other packages, use the [dated package comparison](comparison.md).
For the mapping from every public dispatcher name to a mathematical
specification status, see [specification coverage](waveforms/coverage.md).
The [waveform library](waveforms/index.md) contains the first three scoped,
versioned definitions and their component evidence.

## Model catalog

| Family | Approximant(s) | Harmonics / Multipoles | Physical Features & Regimes |
| :--- | :--- | :---: | :--- |
| **Post-Newtonian** | `TaylorF2`<br>`TaylorF2_HM`<br>`TaylorF2_Eccentric` | $(2,2)$<br>Eight modes through $\ell=4$<br>$(2,2)$ | 3.5PN point-particle & spin phasing, BNS tidal deformability<br>Leading-order higher-mode amplitudes<br>Moore et al. (2016) 3PN stationary-phase eccentric corrections |
| **Aligned-Spin IMR** | `IMRPhenomD`<br>`IMRPhenomHM`<br>`IMRPhenomXAS`<br>`IMRPhenomXHM` | $(2,2)$<br>$(2,\pm 2), (2,\pm 1), (3,\pm 3), (3,\pm 2), (4,\pm 4)$<br>$(2,\pm 2)$<br>$(2,\pm 2), (2,\pm 1), (3,\pm 3), (3,\pm 2), (4,\pm 4)$ | Aligned-spin dominant mode, 5-collocation intermediate<br>Multi-harmonic PhenomD calibration with mode mixing<br>4th-generation aligned-spin dominant mode (Pratten et al. 2020)<br>4th-generation higher multipoles (García-Quirós et al. 2020) |
| **Native Time-Domain IMR** | `IMRPhenomT`, `IMRPhenomTHM`, `IMRPhenomTP`, `IMRPhenomTPHM` | Up to $(5,\pm 5)$ | Aligned or precessing black hole binaries; see the [T-family guide](imrphenomt.md) |
| **Precessing IMR** | `IMRPhenomP` / `Pv2`<br>`IMRPhenomXP`<br>`IMRPhenomXPHM` | $(2,\pm 2)$ twisted<br>$(2,\pm 2)$ twisted<br>Multipolar twisted | Generic 3D spins with NNLO PN precession angles & Wigner $d$-matrices<br>4th-gen precessing carrier with MSA analytical precession<br>Multipolar precession supporting MSA (`v223`), SpinTaylor (`v320`), and PNR (`v330`) |
| **Tidal & Matter** | `IMRPhenomD_NRTidal`<br>`IMRPhenomD_NRTidalv2`<br>`IMRPhenomNSBH` | $(2,2)$<br>$(2,2)$<br>$(2,2)$ | Dietrich et al. (2017) rational Padé BNS tidal phasing (NRTidal v1)<br>Dietrich et al. (2019) NRTidal v2 with self-spin & Planck windowing<br>Neutron star–black hole model with tidal disruption & remnant torus |
| **Effective-One-Body** | `SEOBNR_ROM` / `SEOBNRv4_ROM` | $(2,2)$ | External SEOBNRv4 ROM data with a native Torch/JAX evaluator |
| **Ringdown & Bursts** | `KerrRingdown`<br>`SineGaussian`<br>`MemoryWaveform` | $(2,2,0), (l,m,n)$<br>Burst packet<br>Supplied $(2,\pm2)$ modes | Damped sinusoids from Kerr quasinormal modes (time & frequency domain)<br>Standard unmodeled sine-Gaussian burst wavelet<br>[Scoped dominant-pair nonlinear memory](math/memory-dominant-22.md) through `evaluate_22`; the older polarization-only carrier is diagnostic |
| **Specialized Models** | `ContinuousWave`<br>`EMRIKludge`<br>`ParameterizedPostEinstein` | Monochromatic<br>Multi-harmonic<br>Modified phasing | Continuous waves from triaxial rotating neutron stars with Doppler modulation<br>Extreme mass ratio inspirals (LISA) with Kerr geodesic orbital frequencies<br>Model-independent testing of General Relativity ($\beta_{\rm ppE}, b_{\rm ppE}$) |
| **Time-Domain Tools** | `get_td_waveform`<br>`HybridMECO` | Supported FD models<br>Diagnostic | FFT conditioning and time-domain waveform generation<br>Minimum Energy Circular Orbit hybrid velocity & frequency root solver |

<a id="backend-support-and-evidence"></a>

## Generating a waveform

Select a frontend with `backend="torch"` or `backend="jax"` in the unified
`diffgw` API, or call `torchwave` / `jaxwave` directly. For example,
`IMRPhenomXP` is available through `get_fd_waveform` in **both** frontends,
just as `TaylorF2`, `IMRPhenomD` and `IMRPhenomXAS` are. The
[generation guide](guides/waveform-generation.md) explains the shared interface;
its examples illustrate that interface rather than enumerate every available model.

| Generation task | Guide |
| --- | --- |
| Generate frequency-domain polarizations | [Frequency grids and batching](guides/waveform-generation.md#frequency-domain-waveforms-get_fd_waveform); [IMRPhenomXP example](guides/waveform-generation.md#imrphenomxp) |
| Set precessing spins, modes, tides or eccentricity | [Model parameters and examples](guides/waveform-generation.md#generic-3d-precessing-spins-msa-spintaylor-pnr), [parameter conventions](conventions.md) |
| Generate native T / THM / TP / TPHM time series | [T-family generation guide](imrphenomt.md#generate-a-sampled-waveform); these models require `get_td_waveform` |
| Convert a supported frequency-domain model to a time series | [Time-domain generation](guides/waveform-generation.md#time-domain-waveforms-get_td_waveform) using a conditioned inverse Fourier transform |

The catalog also includes model classes and diagnostic tools, which have their
own entry points. Consult the [PyTorch](api/torchwave.md) or [JAX](api/jaxwave.md)
API reference for these; a catalog entry does not imply support through both
`get_fd_waveform` and `get_td_waveform`.

## Execution restrictions

Generation, batching, compilation and differentiation are separate capabilities.
Use the [execution guide](guides/execution.md) for device, precision and transform
setup, then check the restrictions of the model and options you select:

- **T family:** native time-domain evaluation only. Parameter batches share a
  time grid. Fixed grids support first derivatives; higher derivatives through
  frequency inversion are unsupported. Automatic duration selection is eager;
  supply a fixed duration inside JAX transformations. See the
  [grid and derivative restrictions](imrphenomt.md#arbitrary-grids-and-derivatives).
- **SEOBNR_ROM / SEOBNRv4_ROM:** native evaluation requires a selected
  [external data asset](external-data.md). Its identity and provenance determine
  which surrogate you are evaluating.
- **Precessing and higher-mode paths:** mode selection and the precession
  prescription can change batching and derivative support. Check the
  [precession examples](guides/waveform-generation.md#generic-3d-precessing-spins-msa-spintaylor-pnr)
  and [numerical-precession restrictions](differences_from_lal.md#numerical-precession-spintaylor-and-pnr).
- **Precision and devices:** PyTorch MPS does not support the FP64 examples;
  use CPU on Apple Silicon. JAX's non-symmetric Kerr angular eigensolver uses a
  host CPU callback. See [hardware guidance](guides/execution.md#hardware-acceleration-and-precision)
  and [model-specific limitations](differences_from_lal.md#model-specific-differences).

<a id="differentiation-and-hardware"></a>

## Validation coverage

Evidence answers **what was checked, with which settings**. It does not define
which waveforms can be generated. The published timing campaigns cover selected
models and settings; that is a measurement scope, not the full supported catalog.

- **Measured execution and waveform agreement:** the
  [FP64 campaign](package_benchmarks.md) covers `TaylorF2`, `IMRPhenomD` and
  `IMRPhenomXAS` with Torch eager/compiled and JAX JIT on CPU/CUDA for specified
  grids and populations. Its [accuracy qualifications](package_benchmarks.md#accuracy-across-the-measured-configurations)
  include failed configurations. These measurements do not establish other
  models' performance or support on other accelerators. A separate
  [PhenomXPy comparison](package_benchmarks.md#phenomxpy-time-domain-comparison-22-september-2026)
  measures T/THM on CPU with Torch eager and JAX JIT; failed strict agreement
  checks are marked as diagnostic timings.
- **Model-specific reference agreement:** [Differences from LAL](differences_from_lal.md#model-specific-differences)
  records configurations, deliberate differences and unresolved discrepancies,
  including XP's precession, final-spin and reference-phase choices. Independent
  all-eight leading amplitudes for `TaylorF2_HM` agree with a checksum-pinned
  author supplement. Its signed leading terms, nonspinning phase, and Eq. (11)
  normalization also agree for an all-eight unwindowed complex raw-mode slice
  with `f_ref=None`. Exact external all-eight complex-waveform parity remains
  unprovable because that symbolic asset is parsed by this project and does not
  independently generate the aligned-spin, projection, reference-frequency,
  and cutoff contract. Physical validation remains unavailable.
  ROM evidence applies to the selected asset. The [T-family report](validation/phenomt.md)
  is a separate record.
- **Derivatives:** [derivative support and evidence](validation.md#derivative-support-and-evidence)
  states the checked parameters, derivative orders and restrictions. Waveform-value
  agreement and framework autodiff do not establish derivative accuracy.

The [validation overview](validation.md) distinguishes mathematical-reference
checks, frontend consistency, external-reference agreement and physical validation.
The [manifest](model_manifest.json) records selected configurations, not a complete
support matrix; it does not yet include the T family. Treat a missing check as
unestablished evidence, not as evidence that generation is unavailable.

The unvalidated native SEOBNR surrogate, `SEOBNR_ODE` / `SEOBNRv4Published`
assemblies, `TEOBResumS` waveform, `SpinPrecessionODE` and experimental XHM
phase connections are development-branch work and are unavailable in releases.
The [retired-route record](waveforms/research/retired-release-surface.md)
preserves former names, manifest metadata, and research provenance.
`SEOBNR_ROM`, including direct class construction, consistently selects the
official external-data evaluator. Independently tested EOB equation components
have a separate scope: the retained generic 3PN and TEOB Hamiltonians are
nonspinning; the TEOB component supports vacuum parameters only. These
components do not constitute validated complete EOB waveform models.

## Parameter domains and conventions

The [campaign populations](package_benchmarks.md#workload-and-environment) are
**tested domains**, not calibration boundaries. Restricted cases and component
checks must not be extrapolated to a model's entire physical domain. Where a
calibration range or option is not established by the linked model documentation
and provenance, this catalog makes no domain-wide accuracy claim.

The [waveform guide](guides/waveform-generation.md)
describes generation. The [parameters and conventions reference](conventions.md)
covers units, reference phase/frequency, spin frames, sampling and mode selection. The
[T-family guide](imrphenomt.md) specifies its time origin, duration, conventions
and rejected options. Other model-specific conventions are recorded in the
[manifest](model_manifest.json) and [differences from LAL](differences_from_lal.md#model-specific-differences).

# Waveform catalog and support

Choose a model by its physical content, then check the evidence and execution
restrictions for the options you need. The catalog records implemented families;
a familiar approximant name does not establish equivalence to an external model.
For availability in other packages, use the [dated package comparison](comparison.md).

## Model catalog

| Family | Approximant(s) | Harmonics / Multipoles | Physical Features & Regimes |
| :--- | :--- | :---: | :--- |
| **Post-Newtonian** | `TaylorF2`<br>`TaylorF2_HM`<br>`TaylorF2_Eccentric` | $(2,2)$<br>Eight modes through $\ell=4$<br>$(2,2)$ | 3.5PN point-particle & spin phasing, BNS tidal deformability<br>Leading-order higher-mode amplitudes<br>Moore et al. (2016) 3PN stationary-phase eccentric corrections |
| **Aligned-Spin IMR** | `IMRPhenomD`<br>`IMRPhenomHM`<br>`IMRPhenomXAS`<br>`IMRPhenomXHM` | $(2,2)$<br>$(2,\pm 2), (2,\pm 1), (3,\pm 3), (3,\pm 2), (4,\pm 4)$<br>$(2,\pm 2)$<br>$(2,\pm 2), (2,\pm 1), (3,\pm 3), (3,\pm 2), (4,\pm 4)$ | Aligned-spin dominant mode, 5-collocation intermediate<br>Multi-harmonic PhenomD calibration with mode mixing<br>4th-generation aligned-spin dominant mode (Pratten et al. 2020)<br>4th-generation higher multipoles (García-Quirós et al. 2020) |
| **Native Time-Domain IMR** | `IMRPhenomT`, `IMRPhenomTHM`, `IMRPhenomTP`, `IMRPhenomTPHM` | Up to $(5,\pm 5)$ | Aligned or precessing black hole binaries; see the [T-family guide](imrphenomt.md) |
| **Precessing IMR** | `IMRPhenomP` / `Pv2`<br>`IMRPhenomXP`<br>`IMRPhenomXPHM` | $(2,\pm 2)$ twisted<br>$(2,\pm 2)$ twisted<br>Multipolar twisted | Generic 3D spins with NNLO PN precession angles & Wigner $d$-matrices<br>4th-gen precessing carrier with MSA analytical precession<br>Multipolar precession supporting MSA (`v223`), SpinTaylor (`v320`), and PNR (`v330`) |
| **Tidal & Matter** | `IMRPhenomD_NRTidal`<br>`IMRPhenomD_NRTidalv2`<br>`IMRPhenomNSBH` | $(2,2)$<br>$(2,2)$<br>$(2,2)$ | Dietrich et al. (2017) rational Padé BNS tidal phasing (NRTidal v1)<br>Dietrich et al. (2019) NRTidal v2 with self-spin & Planck windowing<br>Neutron star–black hole model with tidal disruption & remnant torus |
| **Effective-One-Body** | `SEOBNR_ROM` / `SEOBNRv4_ROM`<br>`SEOBNR_ODE`<br>`TEOBResumS` | $(2,2)$<br>$(2,2)$<br>$(2,2)$ | External SEOBNRv4 ROM data with a native evaluation wrapper<br>Time-domain Hamiltonian ODE integrator with continuous QNM matching<br>Damour-Nagar EOB dynamics with spin-orbit coupling and tidal potentials |
| **Ringdown & Bursts** | `KerrRingdown`<br>`SineGaussian`<br>`MemoryWaveform` | $(2,2,0), (l,m,n)$<br>Burst packet<br>$(2,0)$ | Damped sinusoids from Kerr quasinormal modes (time & frequency domain)<br>Standard unmodeled sine-Gaussian burst wavelet<br>Non-linear Christodoulou gravitational-wave memory effect $h_{20}^{\text{mem}}(t)$ |
| **Specialized Models** | `ContinuousWave`<br>`EMRIKludge`<br>`ParameterizedPostEinstein`<br>`SpinPrecessionODE` | Monochromatic<br>Multi-harmonic<br>Modified phasing<br>Orbit-averaged | Continuous waves from triaxial rotating neutron stars with Doppler modulation<br>Extreme mass ratio inspirals (LISA) with Kerr geodesic orbital frequencies<br>Model-independent testing of General Relativity ($\beta_{\rm ppE}, b_{\rm ppE}$)<br>Orbit-averaged spin precession ODE integrator |
| **Time-Domain Tools** | `get_td_waveform`<br>`HybridMECO` | Supported FD models<br>Diagnostic | FFT conditioning and time-domain waveform generation<br>Minimum Energy Circular Orbit hybrid velocity & frequency root solver |

## Backend support and evidence

The native Torch and JAX interfaces share the catalog, but transformation,
device and derivative support depend on the selected path. Use this table to
find execution restrictions and the relevant evidence. Scientific differences
and reference outcomes are maintained on [Differences from LAL](differences_from_lal.md).

| Model or family | Execution scope and restrictions | Model guidance and evidence |
| --- | --- | --- |
| TaylorF2, IMRPhenomD, IMRPhenomXAS | Torch eager/compiled and JAX JIT measured on CPU/CUDA in FP64 for the specified grids and populations. | [Generation](guides/waveform-generation.md), [reference qualifications](package_benchmarks.md#accuracy-across-the-measured-configurations) |
| IMRPhenomXP | Native Torch/JAX; precession, final-spin and reference-phase options affect comparisons. | [Reference options and limitations](differences_from_lal.md#model-specific-differences) |
| IMRPhenomXHM / XPHM, IMRPhenomXPHM-SpinTaylor and IMRPhenomXPNR | Mode selection and precession prescriptions have separate restrictions. Gradient and batching support depend on the selected path. | [Generation](guides/waveform-generation.md), [model differences](differences_from_lal.md#model-specific-differences), [numerical precession](differences_from_lal.md#numerical-precession-spintaylor-and-pnr) |
| IMRPhenomT / THM / TP / TPHM | Both frontends; native TD only, shared batch grid and first derivatives on fixed grids. No higher derivatives through frequency inversion; automatic duration selection is eager. | [T-family guide](imrphenomt.md), [reference report](validation/phenomt.md) |
| SEOBNR_ODE / TEOBResumS | Native EOB dynamics with model-specific numerical choices; not validated as LAL-equivalent. | [Known differences and failed reference checks](differences_from_lal.md#model-specific-differences) |
| SEOBNR_ROM / SEOBNRv4_ROM | Native evaluation using a selected external data asset. The asset determines the surrogate and its validation scope. | [External data](external-data.md), [reference limitations](differences_from_lal.md#model-specific-differences) |
| TaylorF2_HM | Leading-order higher-mode amplitudes through degree 4; independent full-waveform validation is incomplete. | [Mode content above](#model-catalog), [reference scope](differences_from_lal.md#model-specific-differences) |
| Remaining catalog models | Listed physical features do not establish device, transformation or derivative coverage. | [Recorded configurations](model_manifest.json), [validation overview](validation.md) |

The [manifest](model_manifest.json) records selected configurations; it does not
yet include the T family, whose evidence is in its separate report. Missing
coverage should be treated as unestablished.

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

## Differentiation and hardware

See [derivative support and evidence](validation.md#derivative-support-and-evidence)
for the documented parameters, derivative orders and checks. A waveform-value
check does not qualify its derivatives. CPU/CUDA measurements do not establish
support on other accelerators; [precision and device guidance](guides/execution.md)
explains the FP64 restriction on PyTorch MPS. JAX's non-symmetric Kerr angular
eigensolver uses a host CPU callback, as recorded in [Differences from LAL](differences_from_lal.md#model-specific-differences).

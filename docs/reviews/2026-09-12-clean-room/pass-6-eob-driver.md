# Published EOB trajectory, attachment and waveform assembly

Date: 2026-09-12. The experimental `SEOBNRv4Published` API assembles a complete
aligned-spin quadrupole from the independently reconstructed components.
**It does not establish SEOBNRv4 implementation parity.** Five model choices
remain mandatory because the inspected primary sources do not uniquely
identify their selection in the target. Existing prototype aliases and the
frozen parity inventory retain their definitions.

## Physics inputs and explicit choices

The implementation in `eob/published_waveform.py` connects:

1. The [published conservative Hamiltonian](pass-3-seobnrv4.md) and
   quasicircular initial conditions to the [35-mode spinning flux](pass-6-spinning-flux.md).
2. The [pre-NQC amplitude](pass-3-seobnrv4-mode22.md) and
   [complex tail and residual phase](pass-6-eob-phase22.md) to the full
   radiation-reaction vector field. The positive mode frequency is the
   directional derivative of the continuous phase, including tail and
   residual evolution; it is not replaced by twice the orbital frequency.
3. The refined orbital-frequency maximum and published time offset to all
   five [NQC target conditions](pass-3-eob-nqc.md).
4. The [published final mass and caller-selected signed final spin](pass-6-eob-remnant.md)
   to the [Cook author-data fundamental QNM](pass-3-qnm.md) and the
   [C1 merger/ringdown prescription](pass-3-eob-merger-ringdown.md).

These reports record primary URLs, versions, asset hashes, equations and
interpretation limits. The [flux review](pass-6-spinning-flux-review.md),
[attachment review](pass-6-eob-attachment-review.md) and
[driver review](pass-6-eob-driver-review.md) provide independent checks.

| Required keyword | Choices or meaning |
| --- | --- |
| `spin_truncation` | `tabulated` or `generic-leading-ell8`; explicit handling of the general ell=8 spin term beyond the printed tabulation, as defined in the flux report |
| `spin_v6` | `bohe_total_energy`, `pan_test_particle`, `bohe_plus_pan`, or `none` |
| `include_pan_v8` | Boolean retention of the printed Kerr phase term |
| `include_positive_chi_540` | Boolean retention of the earlier positive-spin phase calibration |
| `final_spin_fit_order` | Hofmann Table 1 block `(1,2)`, `(3,3)`, or `(3,4)` |

The QNM receives `M_initial*(omega-i/tau) = (M_final*(omega-i/tau))/(M_final/M_initial)`.
The signed final spin must lie in the available author-data interval
`0 <= a_final <= 0.99999`. Negative or out-of-domain values fail explicitly;
no absolute value, clamp or alternate remnant fit is selected.

## Numerical construction and conventions

`integrate_to_frequency_peak` evolves each batch row until its frequency
maximum is resolved. Completed rows freeze while other rows continue; a
valid-sample mask excludes that padding. Requested post-peak history includes
positive NQC time offsets and the complete five-point matching stencil.
Unresolved maxima or insufficient history raise errors.

`events.py` refines the maximum with a local quartic. `attachment.py` uses
five samples around the prescribed match time to solve the three amplitude
and two phase conditions. On the final local interval it interpolates the
amplitude, NQC bases and continuous frequency with quartics, and integrates
the frequency to obtain phase. This preserves the imposed C1 join without
unwrapping a sampled complex argument. Earlier corrected samples are linearly
interpolated. These are declared numerical choices, not claims about a target
implementation's interpolation.

The direct `reconstruct_seobnrv4_22` API takes positive `M*Omega_start` and
returns dimensionless time from the initial state, `(D/M)*h22`, trajectory,
attachment diagnostics, final mass fraction and signed spin. Initial orbital
phase is zero. Samples after the requested ringdown duration are zero, with
row-specific padding in batches. No start or end taper is applied.

The public frequency-domain wrapper takes solar masses, Mpc, Hz and seconds.
It projects both equatorial quadrupole modes with spherical harmonics and
evaluates the finite discrete Fourier sum with `exp(-2*pi*i*f*t)`. `tc`
places the NQC amplitude peak, and `phic` rotates the orbital mode by
`exp(-2i*phic)`. `f_lower` sets `M*Omega_start=pi*M_seconds*f_lower`; this is
the twice-orbital convention, distinct from the tail-corrected instantaneous
GW frequency. Equality to the reference API's phase/time conventions remains
unresolved. `f_ref` is not an independent phase-reference input of this API.

For example, the following declares one experimental model selection. It is
the selection frozen before the convergence and binary probes, not a
recommended or inferred LAL configuration:

```python
import torch
from torchwave import get_fd_waveform

f = torch.linspace(52., 512., 128, dtype=torch.float64)
hp, hc = get_fd_waveform(
    "SEOBNRv4Published", sample_frequencies=f,
    mass1=30., mass2=20., chi1=0.2, chi2=-0.1,
    distance=100., inclination=0.8, f_lower=51.70000547234364,
    spin_truncation="tabulated", spin_v6="bohe_total_energy",
    include_pan_v8=False, include_positive_chi_540=False,
    final_spin_fit_order=(1, 2), dt_geom=0.5, num_steps=6000,
    ringdown_duration=150.,
)
```

Float64 is preferred for the NQC systems. Autograd is retained within the
selected event, stencil and sample-count branches; crossing those discrete
boundaries is not globally differentiable. The continuous-frequency helper
also works in `no_grad` and inference mode. Focused checks cover gradients,
batch/scalar equivalence, physical Fourier units and phase/time covariance.
An independent review found a shared public-wrapper frequency-mask
broadcasting defect; the corrected mask supports multidimensional frequency
tables and list inputs without flattening their batch axes.

## Convergence evidence

The [recorded experiment](verification/pass-6/eob-convergence/result.json)
uses masses 30 and 20, `M*Omega_start=0.04`, and spins `(0,0)` and `(0.2,-0.1)`.
All runs use the explicit choices above, 150M ringdown, FP64, and fixed
initial phase. Comparisons use a common absolute initial-time grid through
100M after the earlier match, with no time, phase or amplitude fit.

| Quantity | Nonspinning | Spinning |
| --- | ---: | ---: |
| Mode relative L2, dt 0.5 vs 0.25M | 2.2970071e-4 | 1.5359472e-4 |
| Mode relative L2, dt 0.25 vs 0.125M | 1.0304643e-4 | 8.1049437e-6 |
| Refined peak-time change, final refinement | 6.8950933e-5 M | 4.4111293e-5 M |
| Refined peak-frequency change, final refinement | 1.1248124e-8 /M | 2.3922843e-8 /M |
| Final mass fraction | 0.9550166357 | 0.9521433598 |
| Final spin | 0.6637124603 | 0.7026040651 |

All trajectories and modes were finite, and successive differences decreased.
That is the experiment's success criterion; it is neither a `1e-6` waveform
convergence certificate nor a parameter-domain test. The three runs took
25.68, 51.59 and 104.76 seconds. The saved tensors and source hashes allow
reinspection without repeating integrations.

## Independent binary diagnostic

The separate [TD reference probe](verification/pass-6/eob-waveforms.json)
uses the frozen dt=0.125M tensors and public LALSimulation 6.2.0
`SimInspiralChooseTDWaveform` with SEOBNRv4. It preserves the reference epoch,
sets the native NQC amplitude peak to time zero, linearly interpolates native
polarizations on reference times, and includes finite-support differences.
The common numerical phase input is zero, without asserting identical phase
semantics. Nothing is optimized or fitted to the oracle.

| Input spins | Plus relative L2 | Cross relative L2 | Maximum relative norm error |
| --- | ---: | ---: | ---: |
| (0,0) | 1.8698696 | 1.8769837 | 0.0019728274 |
| (0.2,-0.1) | 1.4001084 | 1.4128873 | 0.0003473155 |

These are large failures. Similar norms do not establish waveform agreement.
Phase/time compatibility, unresolved model choices, numerical accuracy and
support all remain relevant; the probe does not isolate their contributions.
It is a separate diagnostic and does not alter the frozen catalog's models,
cases or strict gates.

The complete ordinary suite passes **884 tests with five CUDA checks skipped**
in 50.09 seconds. Strict catalog acceptance remains **11/21** and frozen
holdouts **4/13**, with identical results to pass 5. All source/test inputs
were unchanged during these runs. Tests establish the properties they check,
not a complete target reconstruction or inherited-source provenance.

Reproduce using the environment specified in [the current report](summary.md):

```sh
python scripts/verify_eob_published_convergence.py --output-dir docs/reviews/2026-09-12-clean-room/verification/pass-6/eob-convergence
python scripts/verify_eob_published_waveforms.py --convergence-dir docs/reviews/2026-09-12-clean-room/verification/pass-6/eob-convergence --json-out docs/reviews/2026-09-12-clean-room/verification/pass-6/eob-waveforms.json
```

Both scripts return zero when they finish their respective recorded tasks.
The TD probe's zero exit status is not a parity acceptance result.

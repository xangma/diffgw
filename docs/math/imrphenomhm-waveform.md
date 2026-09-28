# IMRPhenomHM published-calibration composition

<!-- BEGIN GENERATED WAVEFORM DEFINITION -->

| Definition field | Catalog value |
| --- | --- |
| Model ID | `IMRPhenomHM` |
| Reference composition | `imrphenomhm-independent-waveform/v2` |
| Scope | Bounded independent mathematical composition for published D calibration and named source PN variants; inherited defaults remain separate. |
| Domain and modes | `frequency`; `(2, 2)`, `(2, 1)`, `(3, 3)`, `(3, 2)`, `(4, 4)`, `(4, 3)` |
| Native backends | `torch`, `jax` |
| Scoped verification claims | 11 |

| Option | Default | Native choices | Reference-only choices |
| --- | --- | --- | --- |
| `calibration` | `inherited` | `inherited`, `published` | None |
| `hm_mapping_transitions` | `phenomd_runtime` | `phenomd_runtime`, `london_2018` | None |
| `hm_reference_phase` | `legacy` | `legacy`, `orbital` | None |
| `hm_21_pn_variant` | `binary_compatible` | `binary_compatible`, `mishra_eq12b` | None |
| `hm_32_pn_variant` | `binary_compatible` | `binary_compatible`, `mishra_eq12d` | None |
| `hm_43_pn_variant` | `binary_compatible` | `binary_compatible`, `mishra_eq12g` | None |
| `phasing_convention` | `coalescence` | `coalescence`, `reference_frequency` | None |

<!-- END GENERATED WAVEFORM DEFINITION -->

## Scope and status

`imrphenomhm-independent-waveform/v2` independently composes the six
positive-frequency modes $(2,2)$, $(2,1)$, $(3,3)$, $(3,2)$, $(4,4)$ and
$(4,3)$ and their complex plus/cross polarizations. Its native option tuple is
`calibration="published"`, `hm_21_pn_variant="mishra_eq12b"`,
`hm_32_pn_variant="mishra_eq12d"`, and `hm_43_pn_variant="mishra_eq12g"`,
with the exact native D carrier. Both mapping-transition settings have
independent expressions. These options are available through the
[frequency-domain API](../guides/waveform-generation.md) in Torch and JAX.

Version 2 adds `hm_reference_phase="orbital"`, a separately versioned project
phase-origin convention. The original v1 components and default `"legacy"`
behavior are retained. The extension changes no fitted coefficient or
amplitude prescription and does not qualify physical NR accuracy.

The executable reference takes physical binary parameters, not native carrier
amplitudes, phases, fitted coefficients or QNM evaluations. It extends the
[conditional intrinsic-mode check](imrphenomhm-intrinsic-modes.md) by composing
those inputs independently from printed fit coefficients and declared numerical
QNM data. Qualification is bounded by the reviewed claims in the generated
catalog table; a reference implementation alone is not a passing gate.

The default remains `calibration="inherited"` with `binary_compatible` PN
ratios. That tuple is outside this published-source reference. This page
makes no binary-equivalence or NR physical-accuracy claim for the published
variant. The [research ledger](../waveforms/research/phenom-higher-modes.md)
preserves previous default-variant evidence and unresolved NR failures.

## Inputs, conventions, and options

`compose(mass1, mass2, chi1, chi2, distance_mpc=100)` requires
$m_1\ge m_2>0$ in solar masses, $|\chi_i|\le1$, and positive distance in Mpc.
Its `evaluate(frequency_hz, ...)` accepts positive Hz, inclination and harmonic
azimuth `phi0` in radians, orbital phase `phic` in radians and time `tc` in
seconds. The reference returns complex plus/cross and intrinsic modes. Native
carriers support tensor inputs; this reference is a scalar evaluator.

`calibration="published"` selects the printed Khan Table V coefficients and
Husa remnant constants throughout D amplitude, phase, remnant, peak slope and
HM QNM mass scaling. It does **not** select the three source PN options or the
mapping option automatically. XAS/custom carriers and a custom D remnant
provider are unsupported for published calibration. No default coefficient
asset is replaced.

The reference `transition` argument maps to native `hm_mapping_transitions`:
`london_2018` uses $f_0^A=0.018r_{\ell m}/r_{22}$ and
$f_0^\varphi=0.014r_{\ell m}/r_{22}$, while `phenomd_runtime` uses the
respective constants $0.014$ and $0.018$. D's own amplitude/phase joins retain
the D prescription in either case.

`f_ref_hz=None` uses the coalescence convention. Positive `f_ref_hz` corresponds
to native `f_ref` and `phasing_convention="reference_frequency"`; the D peak
slope is independently derived. Set `hm_reference_phase="orbital"` in both
the reference and native evaluators to make the phase-origin change a common
orbital rotation. The default `"legacy"` retains the extra HM-relative factor
and reproduces earlier evidence. The mode adapter
uses $\exp(-i\varphi_{\ell m})$, orbital factor $\exp(im\,\mathrm{phic})$
and time factor $\exp(-2\pi if\,\mathrm{tc})$. Projection follows the
[positive-frequency mode-pair convention](imrphenomhm-projection.md), with
$h=h_+-ih_\times$. Other reference-frame overrides are outside this reference.

The native orbital option requires the exact native D carrier. Explicit
`reference_frequency`, `reffreq`, or `ref` aliases require an explicit `f_ref`;
an implicit first-bin anchor would differ between mapped carrier calls.
Explicit coalescence phasing keeps D's coalescence convention. Unknown options
and unsupported carriers raise `ValueError`. These restrictions apply equally
to Torch and JAX. For example:

```python
import diffgw

hp, hc = diffgw.get_fd_waveform(
    backend="torch", approximant="IMRPhenomHM",
    mass1=35., mass2=20., chi1=.2, chi2=-.1, distance=100.,
    f_lower=20., f_final=512., delta_f=1., f_ref=40.,
    hm_reference_phase="orbital",
)
```

Other options in this example retain their defaults; select the printed
calibration and three source PN variants explicitly for the independent
published composition.

Numerical inputs are the repository's rounded D natural-spline knots,
generated signed high-spin Kerr spline coefficients, and independent HM N61
polynomial coefficients. They are shared **data**, evaluated independently;
see [QNM provenance](../provenance_ledger.md#212-clean-room-kerr-quasinormal-mode-solvers-generation-torchwaveqnm)
and [N61 off-grid evidence](imrphenomhm-qnm-n61-offgrid.md). This composition
adds no new QNM fit or accuracy bound.

## Mathematics and components

The source chain is:

1. [Husa et al., 1508.07250v2](https://arxiv.org/pdf/1508.07250v2),
   PDF pp. 10–11 Eqs. (3.6)–(3.8) (HTML Eqs. 22–24), supplies printed remnant
   spin/energy fits. The [remnant/QNM reference](imrphenomd-remnant-qnm.md)
   applies independent interpolation and total-mass scaling.
2. [Khan et al., 1508.07253v2](https://arxiv.org/pdf/1508.07253v2),
   Appendix C Table V on PDF p. 27 and the amplitude/phase equations, supplies
   printed D coefficients. The independent [D amplitude](imrphenomd-amplitude.md),
   [phase](imrphenomd-phase.md) and [waveform](imrphenomd-waveform.md) references
   produce the base carrier, joins, peak slope and phase coefficients.
3. [London et al., 1708.00404v4, Eqs. (2)–(9)](https://arxiv.org/pdf/1708.00404v4)
   maps the base carrier into HM amplitudes and phases. The source amplitude
   ratios use [Mishra et al., 1601.05588v1, Eqs. (11)–(12)](https://arxiv.org/html/1601.05588v1):
   named 1.5PN expressions for 21/32/43 and retained leading expressions for
   33/44. See the [intrinsic-mode composition](imrphenomhm-intrinsic-modes.md)
   for the mode equations and branch adapters.
4. London's Eq. (1), evaluated through independent Wigner expressions and the
   documented positive-frequency adapter, gives
   $h_+=\sum_{\ell m}C^+_{\ell m}h_{\ell m}$ and
   $h_\times=\sum_{\ell m}C^\times_{\ell m}h_{\ell m}$.

The reference enforces the D hard cutoff at $Mf=0.2$. Mapped HM evaluations
inherit the cutoff through their mapped D carrier; this is not a common
cutoff for every physical mode frequency. Odd-mode zeros and the retained
leading 33 mass-difference zero are evaluated explicitly. Mathematical support
for a branch does not imply that every branch has a sampled qualification.

### Orbital reference-phase extension

Let $r=M_\mathrm{sec}f_\mathrm{ref}$, $s$ be the D peak phase slope, and
$\phi(u)$ the joined intrinsic D phase. Changing from coalescence to a finite
reference adds the constant $C(r)=\pi/4-\phi(r)+sr$ to D's positive phase.
London's continuous phase map sends this constant to $mC/2$ on each branch;
the continuity offsets cancel at the joins. With the orbital option,

$$
P_{\ell m}(r)/P_{\ell m}(\mathrm{None})=e^{-imC(r)/2}.
$$

Adding $C(r)/2$ to `phic` therefore restores every mode and both projected
polarizations at fixed inclination, polarization and time. Between two finite
references use $[C(r_2)-C(r_1)]/2$. This is a project API convention derived
from the orbital rotation law and London composition, not a newly recovered
historical implementation setting. The identity requires nonzero mode support
and mapped carrier phase evaluations below the D hard cutoff.

The legacy option additionally multiplies by $e^{i(m/2-1)sr}$. The orbital
option omits that factor, leaving $(2,2)$ and all no-reference evaluations
unchanged. The derivation and the distinction between a Fourier phase and
physical polarization rotation are recorded in the
[diagnostic receipt](../validation/imrphenomhm_published_options_20260927.md).

## Implementation mapping

| Component ID | Independent reference | Native mapping and maintained test |
| --- | --- | --- |
| `hm.waveform.independent_published.v1` | `diffgw_reference.imrphenomhm_waveform.compose(...).evaluate(...)` | Torch/JAX `IMRPhenomHM` with published calibration and source PN options; `test_published_complete_modes_and_projection` and `test_published_symmetry_zeros` |
| `hm.waveform.independent_published_derivatives.v1` | Converged scalar finite differences of that composition | Native autodiff; `test_published_smooth_derivatives` |
| `hm.waveform.orbital_reference.polarizations.v1` | Scalar composition with `hm_reference_phase="orbital"` | Torch/JAX native D modes and complex projections |
| `hm.waveform.orbital_reference.derivatives.v1` | Independent scalar derivatives of the orbital composition | Torch/JAX autodiff at smooth sample points |
| `hm.waveform.orbital_reference.covariance.v1` | Independently derived $C(r)$ and orbital rotation | Compensated modes and polarizations, unchanged legacy/(2,2), batching and JIT |

Native D uses `diffgw_core.phenomd_published_fits`; the scalar reference retains
its separate printed transcription. Neither runtime imports SymPy or the
reference package. The reference imports no Torch/JAX evaluator, production
fit helper or native phase/amplitude callback. It shares versioned numerical
QNM data, not their evaluation algorithms.

## Evidence and limitations

The maintained complete-composition test is
`tests/test_imrphenomhm_waveform_math_reference.py`. Its original legacy sample is
$(m_1,m_2,\chi_1,\chi_2)=(35,20,0.2,-0.1)$ and
$(30,30,0.35,-0.2)$, plus symmetric $(30,30,0.2,0.2)$ zeros, at 100 Mpc.
It uses inclination $0.93$, azimuth $0.27$, orbital phase $0.37$ and time
$0.008$ s, both transition settings, and no reference or $f_\mathrm{ref}=40$ Hz.
For all six modes it samples outer amplitude/phase joins, the mode ringdown
frequency, and inverse images of the D amplitude/phase joins at their centers
and factors $1\pm10^{-6}$, plus $Mf=0.005,0.035,0.12$.
The target comparison includes intrinsic modes, projected per-mode
polarizations and summed plus/cross, with relative tolerance $3\times10^{-8}$
and absolute tolerance $2\times10^{-10}$ times the comparison-array scale.

Selected smooth derivatives use the first binary, both transition settings,
finite 40 Hz reference, and $Mf=0.005,0.035,0.12$. They cover frequency,
inclination, azimuth, orbital phase, time and distance. Reference slopes are
fourth-order central differences at at least 50 digits, checked against a
halved step, then compared to FP64 CPU autodiff with relative tolerance
$3\times10^{-6}$ and absolute tolerance $2\times10^{-10}$ times the column
scale. The original legacy gate does not qualify intrinsic mass/spin derivatives, join derivatives,
high-spin branches, arbitrary batches or full-domain errors.

The separate `tests/test_phenomd_published_calibration_math_reference.py`
checks option routing, printed payloads, default preservation, native batch
behavior and supported autodiff/JIT routes. Those API checks do not substitute
for the independent full-composition gate. Test definitions and thresholds
are not evidence of a pass until a run is reviewed and recorded. The
[reviewed len run](../validation/imrphenomhm_published_composition_20260927.md)
records 98 passing composition/QNM checks and 17 passing calibration/API checks.
The catalog retains six legacy implementation claims for the tested mapping
and phase-gauge combinations. The orbital extension adds five separately
scoped claims: values and covariance under each mapping choice, and selected
primary-mass/reference-frequency derivatives under the London mapping. See its
[receipt](../validation/imrphenomhm_orbital_reference_20260928.md) for the exact sample.

Shared QNM data can share a data error; separate QNM checks remain necessary.
Printed precision defines a distinct calibration and cannot inherit historical
binary agreement. The inherited calibration, binary PN defaults, XAS/custom
carriers, untested regions and physical NR accuracy remain outside this claim.
All earlier red NR results remain in the research ledger without reclassification.

A separate [prospective physical comparison](../validation/imrphenomhm_published_nr_receipt_20260927.md)
now evaluates this exact published option with `london_2018` mapping against
SXS:BBH:2374 and 2163. Both NR resolution screens pass. Both frontends pass
the first view but fail the second view's match and norm gates in each case:
matches are 0.971608 and 0.972273 versus the 0.980 floor. These results leave
physical qualification open; they do not invalidate the separate bounded
mathematical implementation checks.

The subsequent [source-option and gauge diagnostic](../validation/imrphenomhm_published_options_20260927.md)
identifies a finite-reference covariance problem in the documented API adapter.
The independent reference reproduces this adapter, so its passing implementation
checks do not establish that changing `f_ref` preserves the physical template
family. Mapping choices also affect the NR scores; neither observation closes
the physical gates.

The orbital extension addresses that adapter problem explicitly. Its
[implementation receipt](../validation/imrphenomhm_orbital_reference_20260928.md)
records the new checks and the remaining physical discrepancy. Earlier
receipts use the retained legacy option unless they explicitly state otherwise.

## Reproduce the checks

Use the optional reference, Torch and JAX dependencies with FP64 enabled:

```sh
JAX_ENABLE_X64=1 python -m pytest -q tests/test_imrphenomhm_waveform_math_reference.py tests/test_phenomd_published_calibration_math_reference.py tests/test_waveform_definitions.py
python scripts/generate_waveform_library_table.py
python scripts/generate_waveform_library_table.py --check
```

Run numerical work on `len`. The strict math-reference gate must include both
backends and must not silently skip either. Record package versions and the
exact checkout with the reviewed outcome; changes to equations, declared data,
conventions or claimed scope require review of the versioned reference and
catalog together.

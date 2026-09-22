# Differences from LAL

Use this page when substituting DiffGW for a LAL waveform or interpreting a
reference discrepancy. Shared approximant names do not guarantee identical
physics, calibration, conventions or numerical behavior. DiffGW evaluates its
own tensor implementations; LAL is an optional comparison reference.

For availability across packages, use the dated [package comparison](comparison.md).
For DiffGW's inputs and options, use the [waveform catalog](models.md) and
[parameter conventions](conventions.md). The [validation overview](validation.md)
explains what reference agreement, frontend consistency and physical validation establish.

## Implications for an analysis

This summary separates intentional modeling choices from unresolved agreement.
Its evidence comes from the linked configurations and reports. Reference versions,
DiffGW revisions and measurement dates belong to those individual evidence records.

| Model or path | Nature of the difference | What to do before using it |
| --- | --- | --- |
| TaylorF2 | The dated FP64 campaign misses its strict direct-error gate, despite very small mismatch. This is a reference discrepancy, not a demonstrated domain-wide physical error. | Inspect [both metrics and thresholds](package_benchmarks.md#accuracy-across-the-measured-configurations) for your accuracy requirement. |
| IMRPhenomD / XAS | Agreement is established for the campaign's configurations, with explicit phase and time conventions. | Match options and grids; do not extrapolate the campaign to untested parameters or derivatives. |
| IMRPhenomXP | Agreement depends on precession and reference-phase options; aligned and small-spin residuals remain. | Use the recorded option set below and check your configuration independently. |
| IMRPhenomXHM / XPHM, including SpinTaylor / PNR | Independent higher-mode calibration is intentional. Numerical precession and attachment introduce additional differences. | Treat the carrier and angle prescriptions as scientific model choices; inspect the component checks and their limits below. |
| IMRPhenomT / THM / TP / TPHM | Sampling and epoch conventions differ; comparisons require matching precession and remnant-spin options. Numerical-precession residuals remain unresolved. | Use the [model guide](imrphenomt.md) and [reference report](validation/phenomt.md); the strongly spinning 10:1 case is a known concern. |
| TaylorF2_HM | Leading-order higher modes extend the dominant-mode reference, with incomplete independent validation. | Do not extend the dominant-mode check to the complete higher-mode waveform. |
| SEOBNR_ROM | The selected data asset defines the surrogate. | Match and record the asset before interpreting a reference comparison. |

<a id="technical-comparison-and-reference-validation"></a>
<a id="executive-summary"></a>
<a id="cross-framework-architecture-diffgw-torchwave-jaxwave"></a>
<a id="detailed-model-by-model-differences"></a>
<a id="1-imrphenomxhm-imrphenomxphm-higher-harmonics"></a>
<a id="reference-implementation-and-calibration-conventions"></a>
<a id="independent-nr-calibration-native-qnm"></a>
<a id="2-seobnr_ode-teobresums-effective-one-body-dynamics"></a>
<a id="reference-attachment-conventions"></a>
<a id="continuous-spline-attachment"></a>
<a id="3-imrphenomd-imrphenomxas-literature-precision-time-origin"></a>
<a id="literature-and-reference-code-conventions"></a>
<a id="primary-literature-specification-double-precision-gauge"></a>
<a id="4-taylorf2_hm-multi-mode-frequency-domain-inspiral"></a>
<a id="api-and-mode-availability"></a>
<a id="analytical-post-newtonian-higher-multipoles"></a>
<a id="5-imrphenomxphm-spintaylor-numerical-orbit-averaged-spintaylor-precession"></a>
<a id="physical-numerical-differences-from-lal"></a>
<a id="6-imrphenomxpnr-precession-numerical-relativity-angles-coprecessing-modes"></a>
<a id="physical-numerical-differences-from-lal_1"></a>
<a id="comparison-matrix"></a>
<a id="verifying-parity-and-numerical-accuracy"></a>
<a id="native-imrphenomt"></a>
<a id="model-specific-differences-from-lal"></a>

## Model-specific differences

DiffGW implements waveform physics in its own tensor backends. The following
details explain how its coefficients, conventions, calibration and numerical
methods relate to LAL, and summarize the available reference evidence.

| Model or family | DiffGW implementation | Reference evidence |
| --- | --- | --- |
| **IMRPhenomD / IMRPhenomXAS** | Published models with explicit coefficient precision, reference-phase and time-origin conventions. XAS uses the author-supplied symbolic calibration deposit identified in the provenance ledger. | Both DiffGW frontends pass the strict LAL gate for all measured D/XAS configurations in the [FP64 campaign](package_benchmarks.md#accuracy-across-the-measured-configurations). |
| **IMRPhenomXP** | The documented clean path uses MSA version 223, `convention=1`, `final_spin_mod=3` and `phi0=0`. Nonzero `phi0` retains legacy geometry; unsupported cases use explicit fallback. | Frozen catalog and holdout cases agree closely. The manifest records residuals at exact alignment, zero spin and tiny in-plane spins, with detector-projection errors distinguished from polarization errors. |
| **IMRPhenomXHM / IMRPhenomXPHM** | Five modes: `(2,2)`, `(2,1)`, `(3,3)`, `(3,2)` and `(4,4)`. The four higher modes use independent SXS-based ringdown fits and native Kerr QNM calculations; XPHM adds precession to this carrier. | The calibration differs from LAL's. Recorded NR checks examine phase in a restricted ringdown band with reference amplitudes and affine phase alignment; they do not validate full-waveform amplitude and phase accuracy. |
| **TaylorF2_HM** | Literature-based frequency-domain PN extension with eight leading-order modes through `l=4`: `(2,2)`, `(2,1)`, `(3,1)`, `(3,2)`, `(3,3)`, `(4,2)`, `(4,3)`, `(4,4)`. | The external TaylorF2 check covers `(2,2)`. An independent full-waveform reference for the higher modes, higher-PN completeness and spin-amplitude completeness remain unresolved in the manifest. |
| **SEOBNR_ROM** | Data-backed reduced-order evaluation; the chosen surrogate and calibration asset define the model. Official SEOBNRv4ROM data are external. | Validation applies to the selected asset. See [external data](external-data.md) for acquisition and [third-party notices](../THIRD_PARTY_NOTICES.md) for unresolved redistribution terms. |

The EOB [peak locator](../src/torchwave/approximants/eob/continuous_matching.py)
solves a natural cubic spline's derivative on accepted samples, selecting an
interior maximum and rejecting unresolved or tied maxima. The selected root is
locally differentiable, while interval and maximum selection remain piecewise;
continuous attachment does not remove integration-step dependence.
JAX's non-symmetric Kerr angular eigensolver uses a host CPU callback.

### Numerical precession: SpinTaylor and PNR

`IMRPhenomXPHM-SpinTaylor` (`prec_version=320`) integrates spin-precession
equations with fixed-grid RK4 in velocity space. `IMRPhenomXPNR`
(`prec_version=330`, `final_spin_mod=2`) combines NR-tuned PNR angles with
DiffGW's native XHM coprecessing carrier. The carrier, precession integration,
angle attachment and minimal-rotation quadrature contribute to differences from
LAL; comparisons specify the precession and tuned-coprecessing options.

The [SpinTaylor tests](https://github.com/xangma/diffgw/blob/5226141/tests/test_imrphenom_xphm_spintaylor.py)
and [PNR tests](https://github.com/xangma/diffgw/blob/5226141/tests/test_imrphenom_xpnr.py)
use optimized plus-polarization overlaps with model-specific tolerances.
Their scope differs from the direct, both-polarization gate in the
[FP64 campaign](package_benchmarks.md#numerical-qualification); these variants
were not timed in that campaign.

### Native IMRPhenomT family

T, THM, TP and TPHM have native time-domain evaluation in both frontends.
The precessing variants offer NNLO (default), MSA and numerical prescriptions;
reference comparisons must match angle and remnant-spin options. Fixed-grid
first derivatives are supported, with restrictions on frequency inversion and
automatic duration selection described in the [model guide](imrphenomt.md).

The [reference-validation report](validation/phenomt.md) records author checks,
LAL regression criteria, measured residuals and diagnostic experiments. Numerical
precession retains unresolved reference differences, including the strongly
spinning 10:1 case; agreement in the documented cases does not establish accuracy
throughout parameter space or independent physical accuracy.

### Reproducing model checks

DiffGW's catalog harness retains model-specific references and diagnostic
failures:

```bash
PYTHONPATH=src python tests/parity_harness.py --scope=analytic
PYTHONPATH=src python tests/parity_harness.py --scope=catalog
```

Each result records its reference identity, metrics and verification scope.
The [benchmark protocol](benchmarking_protocol.md#12-the-gatekeeper-rule-parity-first)
sets out how those checks qualify performance results, and the
[package benchmark report](package_benchmarks.md#reproduction-and-receipts)
provides timing receipts and reproduction commands.

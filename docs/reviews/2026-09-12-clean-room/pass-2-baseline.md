# Pass 2: PhenomD and NSBH phase consistency

This pass used the inherited TorchWave implementation and fresh primary papers.
No LAL/PyCBC source, port, historical calibration note, waveform-oracle output,
or residual fit was used to choose the implementation. The parent performs
independent target verification. This record does not certify the provenance
of inherited high-precision calibration digits.

## Primary evidence

| Primary paper | Download | PDF SHA256 |
|---|---|---|
| Thompson et al., PRD 101, 124059 | [2002.08383v2](https://arxiv.org/pdf/2002.08383v2) | `7e8bf4ba34a7fb98d54fa1a405267fdb4d06a78e1b164f57361b0b2abebd34e2` |
| Dietrich et al., PRD 100, 044003 | [1905.06011v2](https://arxiv.org/pdf/1905.06011v2) | `692ccb5a3c9c8098c9e389d9f118e2709b72b95076bb53600d5e0b2b93f60f6f` |
| Khan et al., PRD 93, 044007 | [1508.07253v2](https://arxiv.org/pdf/1508.07253v2) | `c3ed056e896d0ef02b639966a60a197587baf0ee514c9c952576bf979fc7c907` |
| Husa et al., PRD 93, 044006 | [1508.07250v2](https://arxiv.org/pdf/1508.07250v2) | `1fcb99449d1127ab3c2403ddbd81b8d76845528ff2a594b9aac640b9d29d459e` |

Thompson Sec. II.B specifies the PhenomD BBH phase plus the improved NRTidal
correction; Sec. III.C identifies the shared phase with PhenomDNRT. Dietrich
Eq. (4) adds the phase contributions; Eqs. (9), (17)-(18) imply the implemented
tidal term `-117*Lambda_tilde*x^(5/2)*P(x)/(256*eta)`. These statements specify
the intrinsic phase composition, not a unique global affine phase/time gauge.

## Implementation

The inherited NSBH module duplicated PhenomD's phase, substituted its
amplitude-dependent UIB/Zappa remnant frequencies, subtracted an inspiral-only
reference expression, and anchored its time shift to the requested upper
frequency. Thus changing only the grid extent changed existing complex bins.
It also reversed the shared API's `tc` sign and ignored the exposed phase
convention arguments.

NSBH now obtains its intrinsic phase from `IMRPhenomD(return_phase=True)` and
adds the unchanged NRTidalv2 correction. Its phase curvature and C1 joins use
PhenomD's BBH remnant independently of the disruption-dependent amplitude.
NSBH's overall polarization sign and default reference frequency of 30 Hz
are retained. The tidal correction is not separately zeroed at the reference.

The first pass-2 implementation also replaced the default affine offsets with
PhenomD's gauge. The verifier reported a large raw-complex regression and
requested a separate gauge review. The primary papers do not prescribe that
replacement, so it has been removed. With no phase convention specified (or
explicit `legacy`), the inherited constant and time slope are now retained:
the constant uses the inspiral expression at `f_ref`; the slope uses the raw
merger derivative at `f_final`, then `fhigh`, then the grid maximum, in that
precedence order, with the inherited amplitude remnant. The inherited default
`tc` factor is `exp(+2*pi*i*tc*(f-f_ref))`. These are compatibility choices,
not claimed published physical prescriptions. They are applied only after
recovering the shared PhenomD intrinsic phase; the previously substituted
intrinsic NSBH-remnant phase has not been restored.

Explicit `reference_frequency` and `coalescence` (including their existing
aliases through `reference_frame`) instead use the PhenomD gauge and its
`exp(-2*pi*i*f*tc)` factor. These options provide reference selection in all
three IMR regions and grid-independent phase. The default remains dependent
on the upper-frequency time anchor unless an explicit `f_final`/`fhigh` is
kept fixed. Its inspiral reference expression remains a compatibility
limitation for high reference frequencies. Gauge restoration uses only the
inherited algebra, with no new coefficients or residual fit. Neither gauge
is certified here as the target library's exact raw-complex convention.
The existing PhenomD call also computes an amplitude that NSBH discards;
extracting a dedicated phase-only evaluator remains a possible optimization.

## PhenomD and QNM review outcome

The existing PhenomD phase already selects the inspiral, intermediate, or
merger expression at `f_ref`. Its matching coefficients agree with the C1
construction in Khan Eqs. (13)-(16), (28), and (37). The Appendix B calibration
uses the documented PN spin orders. The remnant variables use the total spin
and its mass-normalized form as required by the accompanying remnant model.
No new numerical repair to the intrinsic PhenomD phase or QNM evaluator was
established in this pass. Their pass-1 corrections and limitations remain in
[baseline.md](baseline.md).

The default additional Planck taper was removed: `window_type=None` now
preserves the analytic merger amplitude of Khan Eqs. (19), (36). Explicit
Planck/cosine tapers remain available. A focused test checks the analytic tail
and opt-in taper. The inherited `Mf=0.2` cutoff remains an implementation
boundary. The former claim that this is the paper's calibration limit was
removed: Appendix A instead discusses already small amplitude beyond
`Mf=0.25` and does not specify this particular cutoff.

In particular, the papers permit arbitrary affine phase/time shifts and do not
uniquely specify the inherited raw-alpha1 peak anchor. Altering it to match
verifier residuals would not meet this pass's clean-room boundary.

Husa Sec. III, immediately before Eq. (3.1), says that the ringdown data in
Refs. [82]-[83] were interpolated. Neither this passage nor Khan specifies
the interpolation order, knot selection, endpoint conditions, or rounding.
The cited author's current [ringdown data page](https://pages.jh.edu/eberti2/ringdown/)
describes the data columns and warns about the near-extremal regime, but
provides no PhenomD interpolation prescription. Its linked `ell=2` archive
returned HTTP 401 when retrieved on 2026-09-12. Thus an exact compatibility
interpolator cannot be derived from these sources. The independently checked
Cook data accuracy in [baseline.md](baseline.md) supports retaining the current
QNM evaluator; target residuals were not used to choose an alternative grid,
interpolation scheme, or coefficients.

## Verification and remaining work

The focused tests cover all three explicit reference-frequency regions at zero
tidal deformability, explicit common-frequency invariance when the grid is
extended, both time-translation factors, the orbital `m=2` phase factor, the explicit
coalescence convention and alias, and linearity of the added phase in tidal
deformability in the explicit gauge. They also check that default and explicit
gauges differ only by an affine phase across the intrinsic joins, and that a
fixed default time anchor gives common-grid agreement. Earlier amplitude,
batch, derivative, and QNM checks also run.

Command:

```sh
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /Users/xangma/miniconda3/envs/pycbcorig313/bin/python -m pytest -q tests/test_clean_room_nsbh.py tests/test_clean_room_phenomd_baseline.py
```

Result: **19 passed**. No oracle dependency is imported by these tests.

This does not resolve the NSBH low-Lambda pre-merger ambiguity, amplitude
remnant/classification choices, PhenomC PN spin mapping, or calibration-digit
provenance listed in [nsbh.md](nsbh.md). Those require separately justified
equations or author data. Spinning-neutron-star finite-size terms also need a
separate full phase audit; the primary model is calibrated for nonspinning NSs.

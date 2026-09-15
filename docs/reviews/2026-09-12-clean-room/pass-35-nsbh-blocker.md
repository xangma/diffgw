# Pass 35: NSBH residual localization and blocker

Date: 2026-09-13

## Scope

This is a bounded audit of `IMRPhenomNSBH` after the reference-sentinel fix.
The implementation was compared with Thompson et al., arXiv:2002.08383v2,
using the published source archive (SHA-256
`bf81b62a2397ca291e4847e979c62f83f2435e29f470c12baa111041f7377855`; extracted
`paper.tex` SHA-256
`8afc8c4ddfa0f5b9cad5856590f5cea7edba3c317f0e84fbad6a022e96ee96b5`).
The public LAL waveform API was used only as a black-box measurement oracle.
No LALSimulation, PyCBC, TEOBResumS, or other waveform implementation source
was inspected or copied, and no residual was used to fit a runtime quantity.

## Source-backed audit

Thompson Table I and Appendix A, Eqs. (A7)--(A10), require the NSBH remnant
fit to produce (M_f, chi_f), the Kerr (f_{\rm RD}) and (Q), followed by
the closed classification condition (f_{\rm tide}\ge f_{\rm RD}) and the
torus-mass condition.  The current implementation follows that order.  The
two strict probes evaluate as follows:

| probe | (f_{\rm tide}) [Hz] | (f_{\rm RD}) [Hz] | torus ratio | class |
| --- | ---: | ---: | ---: | --- |
| (5+1.4, \chi_1=.2, \Lambda_2=400) | 2736.9320 | 2562.4719 | 0 | non-disruptive |
| (7+1.4, \chi_1=.4, \Lambda_2=2600) | 1440.8683 | 2019.7245 | 0.1176731 | disruptive |

The same table specifies the four (epsilon), (sigma_{\rm tide}), and
(~tilde f_{0,1,2}) branches.  The native branches match those rows,
including the factor-of-two-corrected windows

\[
\omega^\pm_{f_0,d}(f)=\tfrac12[1\pm\tanh(4(f-f_0)/d)],
\]

and Appendix A Eq. (30)'s width (d=0.015+σ_{\rm tide}) in (M f).  The
ringdown term also matches Eq. (A26):

\[
\sigma=\delta'_2 f_{\rm RD}/Q,
\]

with the published (A/2=0.81248) high-Λ factor.  The only remaining
notation ambiguity found in the source is the low-Λ γ₁ branch; that was
already audited in Pass 19 and is deliberately not changed here.

## Fresh black-box measurements

The latest strict artifacts (`verification/pass-33-pv2`) report the unchanged
NSBH residuals:

| probe | complex relative L2 | magnitude relative L2 | maximum relative magnitude error |
| --- | ---: | ---: | ---: |
| catalog | 2.965e-5 | 7.77e-8 | 2.98e-6 |
| holdout | 1.758e-3 | 4.60e-9 | 1.20e-7 |

The catalog and holdout projection errors track the polarization errors, so
there is no detector-projection defect to localize.  The amplitude residuals
are orders of magnitude below the complex residuals, and the measured
classification, window, remnant, and ringdown quantities are source-defined.
Changing one of them from these measurements would amount to residual fitting
or to selecting an alternate branch unsupported by the paper.

## Decision

No runtime correction is justified in Pass 35.  The remaining NSBH strict
failure is source-incomplete rather than a uniquely identified amplitude,
window, classification, or reference-frequency defect.  The fresh sentinel
path is exercised by the existing NSBH tests; positive reference frequencies
in both strict probes are unaffected by that fix.

## Validation

No source or test file was changed.  Focused NSBH validation remains:

```
PYTHONPATH=src:tests PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /Users/xangma/miniconda3/envs/pycbcorig313/bin/python -m pytest -q \
  tests/test_clean_room_nsbh.py tests/test_nsbh.py \
  tests/test_phenomnsbh_reconstruction.py
```

The previously recorded result is `36 passed in 1.00s`; `py_compile` and
`git diff --check` pass.  This pass adds only this report and does not alter
acceptance gates or unrelated dirty files.

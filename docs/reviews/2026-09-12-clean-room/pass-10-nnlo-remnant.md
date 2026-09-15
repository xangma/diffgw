# Pass 10: NNLO precessing remnant plumbing

## Scope

This pass implements the NNLO remnant prescription for XP/XPHM, including
their native default NNLO path. It retains the native aligned-spin final-spin
fit and existing calibration tables. The precessing extension follows the
orthogonal-spin equations and earlier author prescription described below.
The implementation used no LAL/PyCBC source or waveform oracle and fitted no
coefficient to binary residuals. Acceptance thresholds are unchanged.

## Prescription and dispatch

For the canonical primary mass (m_1\ge m_2), the existing aligned fit gives
(a_{\parallel,\rm fin}). The NNLO remnant is evaluated as

\[
 a_{\rm fin}=\operatorname{sign}(a_{\parallel,\rm fin})
 \sqrt{a_{\parallel,\rm fin}^2+
 [\chi_p(m_1/M)^2]^2},\qquad
 \frac{m_1}{M}=\frac{1+\sqrt{1-4\eta}}{2}.
\]

**Source audit:** the literal Eq. (4.25) in
[Pratten et al., arXiv:2004.06503v2](https://arxiv.org/abs/2004.06503v2)
prints an additional square on the in-plane spin inside an already squared
term; Eq. (4.22) contains a similar inconsistency. This conflicts with the
orthogonal-vector relation in Eq. (4.17). The earlier
[IMRPhenomPv3 paper, arXiv:1809.10113](https://arxiv.org/abs/1809.10113)
independently gives the in-plane angular momentum as
\(S_\perp=m_1^2\chi_p\), with scaling \(\lambda=M_f/M\) in
\(S_\perp\lambda^2/M_f^2\). Substitution gives precisely
\(\chi_p(m_1/M)^2\). We therefore implement this derived prescription,
interpreting the extra printed squares as apparent typographical errors;
we do not claim a verbatim transcription of Eq. (4.25).

A separate GPT-5.6 Luna reviewer checked these sources and all carrier call
sites; the primary agent verified the source passages directly. Source
filenames, hashes and equation locations are in
[source-audit.json](verification/pass-10/source-audit.json).

The native effective-\(\chi_p\) construction is shared with the precession
angle evaluator. The sign follows the paper's continuity with the
nonprecessing final spin. The exactly ambiguous case
(a_{\parallel,\rm fin}=0,\chi_p\ne0) is rejected explicitly; no sign is
invented. At (chi_p=0), the helper returns the aligned fit exactly.

The remnant override is selected for resolved NNLO versions 101--104. Thus the
default NNLO path receives `FinalSpinMod=0`, while an explicit MSA version
(including `prec_version=223`) retains the existing carrier behavior.
MSA, ODE, and fallback rows are not claimed to have NNLO remnant parity.

## Native integration

The helper is in
`src/torchwave/approximants/imrphenom_xas.py`, with the shared effective-
`chi_p` construction in `src/torchwave/approximants/precession_msa.py`.
`src/torchwave/approximants/imrphenom_xp.py` and
`src/torchwave/approximants/imrphenom_xphm.py` compute the override only on
the resolved NNLO path. `src/torchwave/approximants/imrphenom_xas.py` accepts
the internal override for QNM, ISCO, and ringdown calculations. Every XHM
carrier call, including mode-mixing probes, receives the same value through
`src/torchwave/approximants/imrphenom_xhm.py`.

Scalar aligned fast paths and aligned rows in mixed batches continue to use
the carrier directly. The native vector-norm branches have finite
zero-in-plane-spin gradients and retain physically tiny nonzero remnant
magnitudes.

## Validation

New focused tests are in
`tests/test_pass10_nnlo_remnant.py`. They cover the published formula, exact
aligned reduction, explicit undefined-sign rejection, XP/XPHM mixed batches
with a zero-in-plane-spin row and finite gradients, and differentiability of
the shared (\chi_p) at the aligned point.

The focused pass-9 plus pass-10 command was:

```text
PYTHONPATH=src:tests PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
/Users/xangma/miniconda3/envs/pycbcorig313/bin/python -m pytest -q \
tests/test_clean_room_precession_conventions.py \
tests/test_clean_room_msa.py \
tests/test_phenomxp_msa_reconstruction.py \
tests/test_pass10_nnlo_remnant.py
```

The implementer's initial strengthened set passed **88 tests** (79 inherited
and nine new). Primary review subsequently added equal-mass remnant
normalization and actual XP/XPHM body-relabeling checks for versions 101--104,
and tightened exact aligned-spin comparisons. Final results are recorded in
[pass-10 verification](verification/pass-10/).

Final full verification passed **1,075 tests, with eight skipped**. Seven
inherited reference-rotation tests were updated to compare against carriers
with the same derived remnant, preserving their tolerances; their former
unchanged-remnant assumption no longer applies to tilted systems. Independent
review confirmed the revised expectation. See the
[verification summary](pass-10-summary.md) for strict results and limitations.

The implementation does not establish universal XP/XPHM remnant parity for
MSA, ODE, or fallback dispatch; those paths retain their existing policies.

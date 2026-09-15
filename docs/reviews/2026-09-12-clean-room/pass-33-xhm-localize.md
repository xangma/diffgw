# Pass 33: clean-room XHM residual localization

Date: 2026-09-13. This pass decomposes the diagnostic and one fixed holdout
case by mode using TorchWave's `return_unprojected_modes=True` path and the
public `lalsimulation.SimIMRPhenomXHMGenerateFDOneMode` API. No LAL,
PyCBC, TEOBResumS, or other waveform implementation source was inspected.
No residual was used to fit a coefficient.

## Method

The diagnostic case is `(m1,m2,chi1,chi2,phic,D) =
(40,20,0.3,-0.1,0.5,500)` with `f=20...512 Hz`, `df=0.25 Hz`, and
`f_ref=20 Hz`. The holdout is `(55,11,-0.4,0.2,0.71,680)` with
`f_ref=27 Hz`, using the same frequency grid. The public API returns a
one-mode `COMPLEX16FrequencySeries`; its positive-frequency mode is compared
with the complex conjugate of TorchWave's unprojected mode, including the
constant sign convention. This convention is exact for the carrier: the
diagnostic `(2,2)` amplitude ratio is 1.000 in every tested band and its
optimized complex L2 is 0.000.

The table reports median public/TorchWave amplitude ratios in the inspiral
(`20--100 Hz`) and high-frequency (`400--512 Hz`) bands. `optL2` is the
full-band complex relative L2 after allowing one constant complex scale; it
is a diagnostic of frequency-dependent disagreement, not a fitted runtime
parameter.

| mode | diagnostic amp 20--100 | diagnostic amp 400--512 | diagnostic optL2 | holdout amp 20--100 | holdout amp 400--512 | holdout optL2 |
|---|---:|---:|---:|---:|---:|---:|
| 22 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 |
| 21 | 0.949 | 0.540 | 0.997 | 0.527 | 0.227 | 0.998 |
| 33 | 0.805 | 0.993 | 0.910 | 1.101 | 0.884 | 0.857 |
| 32 | 0.614 | 1.088 | 0.883 | 0.943 | 0.664 | 0.928 |
| 44 | 0.684 | 0.160 | 0.891 | 0.916 | 0.970 | 0.942 |

The mode decomposition localizes the residual to the higher-mode
reconstructions rather than the carrier. The largest diagnostic high-frequency
amplitude loss is `(4,4)` (median 0.160); the largest holdout loss is `(2,1)`
(median 0.227). The full-band complex mismatch remains large for every
higher mode even after removing a constant complex convention, so a single
global phase or normalization fix cannot explain the residual. Some modes
also develop frequency-dependent phase disagreement above the inspiral band;
this is consistent with the unresolved intermediate and ringdown
reconstruction dependencies recorded in the earlier clean-room audit.

## Source cross-check and decision

The primary source is García-Quirós et al.,
[arXiv:2001.10914v2](https://arxiv.org/abs/2001.10914v2). The earlier
source-only amplitude audit in
[pass 17](pass-17-amplitude-next-primary.md) checked Eq. (4.4), Tables II,
IV, and V, the inverse-polynomial construction, and the ringdown ansatz. It
found no source-complete runtime defect in the 21, 33, 32, or 44 amplitude and
ringdown construction. It also recorded the literal Table II 32 denominator
ambiguity and explicitly left it unchanged because no allowed primary asset
resolves whether it is a typo. The earlier phase audit in
[pass 2](pass-2-higher-modes.md) likewise leaves the intermediate fit-target
conversion, absolute phase/time gauge, and 32 mixing-entry mapping
unresolved; inferring those values from this residual would violate the clean
room rule.

Therefore this pass makes **no speculative runtime change**. The quantitative
blocker is the missing source-resolved calibration/reconstruction metadata,
not an identified equation defect. No deferred NR job was started.

## Validation

Focused tests:

```text
PYTHONPATH=src:tests PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /Users/xangma/miniconda3/envs/pycbcorig313/bin/python -m pytest -q \
  tests/test_phenomxhm_reconstruction.py tests/test_imrphenom_hm.py
```

Result: **16 passed in 4.54 s**.

`git diff --check` passed. Only this report was added by pass 33; the existing
dirty source file was not modified.

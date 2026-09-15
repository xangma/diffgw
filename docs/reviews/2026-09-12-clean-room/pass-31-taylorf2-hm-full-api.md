# Pass 31: TaylorF2_HM full-mode external API audit

This is a clean-room validation audit.  It used only the installed public
`lal`/`lalsimulation` Python APIs as black boxes, the current TorchWave
TaylorF2_HM module, and the published García-Quirós et al. equations.  No LAL
or PyCBC implementation source was inspected, copied, or translated.

## Result

The public API does not provide a stable, mode-resolved TaylorF2 reference for
TorchWave's current higher-mode inventory.

The installed public `SimInspiralTaylorF2` entry point returns one
`COMPLEX16FrequencySeries`.  Supplying public mode-array metadata for
`[(2,2)]`, `[(3,2)]`, `[(2,2),(3,2)]`, or `[(2,1),(3,3),(4,4)]` produced an
array exactly equal to the no-mode-array carrier output for the same physical
parameters.  Thus mode-array selection cannot be used to extract individual
TaylorF2 harmonics or to construct a full-mode reference.

The public `SimInspiralChooseFDModes` dispatcher was also exercised with the
TaylorF2 approximant.  It rejected the request because the selected generator
does not provide frequency-domain modes.  The installed public approximant
lookup maps `TaylorF2HM` and `TaylorF2_HM` to the same TaylorF2 approximant
identifier rather than exposing a distinct multipolar frequency-domain
generator.

The public PN mode functions (`SimInspiralPNMode21`, `SimInspiralPNMode32`,
and related entries) are time-domain mode constructors.  They do not define
the frequency-domain stationary-phase interpolation, harmonic-dependent
normalization, stored negative-m convention, phase connection, cutoff, and
observer projection required to compare TorchWave's frequency-domain
TaylorF2_HM output.  Treating a separately transformed time-domain PN mode as
that reference would introduce an unapproved adapter convention.

## Validation guard

`tests/test_taylorf2_hm_external_api.py` records these black-box interface
properties without depending on implementation source.  The focused command

```text
PYTHONPATH=src:tests PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /Users/xangma/miniconda3/envs/pycbcorig313/bin/python -m pytest -q \
  tests/test_taylorf2_hm.py \
  tests/test_taylorf2_hm_mode_inventory.py \
  tests/test_taylorf2_hm_external_api.py
```

completed with **15 passed**.  `git diff --check` is clean.

The existing strict parity case therefore remains explicitly scoped to
`dominant_mode_only` and retains the unresolved full-mode oracle status.  Its
1e-6 complex-L2 and 1e-8 mismatch gates are unchanged; no carrier-only result
is promoted to full-mode parity.


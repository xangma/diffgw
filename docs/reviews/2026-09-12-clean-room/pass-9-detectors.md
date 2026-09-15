# Pass 9: detector overload precision audit

This focused follow-up reviewed the existing detector report in
`pass-3-detectors.md`, `src/torchwave/detectors/network.py`, and the detector
tests. The prior GMST, leap-boundary, arm normalization, antenna convention,
geometric-delay sign, and frequency-batch fixes were retained. No LAL/PyCBC
source, binary waveform, or oracle output was used.

## Finding and fix

`Detector.project_td_strain` supports both
`(h_plus, h_cross, delta_t, ...)` and `(times, h_plus, h_cross, ...)`. Its
dispatch previously treated any tensor with `numel() == 1` as `delta_t`.
Consequently a valid one-sample time-domain waveform, whose `h_cross` has
shape `(1,)`, was interpreted as the legacy overload. For example, with
`times.shape == hp.shape == hc.shape == (1,)`, the method projected `times` and
`hp` instead of `hp` and `hc`.

The dispatch now treats Python scalars and zero-dimensional tensors as
the legacy `delta_t` argument. A length-one tensor is treated as a waveform
when the caller supplies the optional `delta_t=` keyword, while the
historical one-element tensor form remains accepted as `delta_t` when that
keyword is absent. This follows the public convention that time series
retain their final sample axis while preserving the previously accepted
scalar-tensor callers. A focused regression test checks the one-sample result
published linear response `h_det = F_+ h_+ + F_\times h_\times` (the geometric
shift is zero for the explicitly supplied one-sample spacing).

## Validation

```sh
PYTHONPATH=src:tests PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /Users/xangma/miniconda3/envs/pycbcorig313/bin/python -m pytest \
  tests/test_clean_room_detectors.py tests/test_detectors.py -k 'not lal' -q
```

Result: **31 passed, 1 deselected**. `git diff --check` passes for the edited
detector files. The focused test also manually confirmed that Python-float
and zero-dimensional tensor `delta_t` calls retain the legacy overload.

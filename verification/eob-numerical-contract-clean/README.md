# SEOBNRv4 numerical contract clean-room checks

This record covers the bounded, opt-in numerical attachment implementation in
`torchwave.approximants.eob.numerical_attachment`.  The checks use analytic
and synthetic Torch tensors only: unequal-knot natural-cubic systems, spline
and matching gradients, an explicit floor attachment fixture, the geometric
step rule, the independent 107-knot angular QNM cache, and the published
Pan cubic-spin term.  No LAL/reference implementation, decompiled code,
exposure file, fitted residual, or long trajectory is used.

Run the focused checks with:

```text
PYTHONPATH=src:tests:scripts:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /Users/xangma/miniconda3/envs/pycbcorig313/bin/python -m pytest -q \
  tests/test_eob_numerical_contract_clean.py
```

The helper remains opt-in.  The default published EOB route and paper-only
spin-amplitude path are unchanged.  Public-binary waveform residuals and the
100M post-attachment control values from the review are intentionally not
reported here; those require a separately authorized trajectory diagnostic.

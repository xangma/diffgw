# Clean adaptive EOB driver verification

This report records the bounded, independent Fehlberg 4/5 adaptive driver in
`torchwave.approximants.eob.adaptive_driver_clean`.  The implementation keeps
accepted states and derivatives as the integration history, and constructs the
requested uniform output with a natural cubic spline.  EOB coarse and fine
stages use accepted-state event checks; the fine stage is restarted from a
real coarse sample and its local trajectory is downsampled and spliced into
the coarse prefix.  The optional `numerical_compatibility=True` published
route calls this implementation; the default fixed-step published route is
unchanged.

The focused suite covers analytic exponential ODEs under both controllers,
state-scaled controller threshold and candidate-state checks, rejected-state
restoration, accepted-state event termination, immediate-previous accepted
derivative comparisons, natural cubic interpolation on nonuniform knots,
tolerance refinement, an explicit intermediate-stage domain failure, and a
distinct-parameter batch with diverging event histories:

```text
PYTHONPATH=src:tests:scripts:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /Users/xangma/miniconda3/envs/pycbcorig313/bin/python -m pytest -q \
  tests/test_eob_adaptive_driver_clean.py
```

Result: `14 passed in 0.77s`.

The bounded physical smoke probe was run with 30+30 solar-mass inputs,
`omega_start=0.05`, `dt_geom=1`, a 1000 accepted-step bound, and 10M of
ringdown support.  It completed in 12.26 seconds on the local host.  The
coarse stage used 78 accepted and 2 rejected steps (479 field evaluations),
and the fine stage used 79 accepted and 2 rejected steps (485 evaluations)
with `fine_step=0.015625` and refinement factor 64.  The actual trajectory
produced 175 splice samples, 169 valid trajectory samples, restart index 18,
peak time 167.60096M, and match time 163.99521M; all returned mode values
were finite.  This is a bounded cost/shape probe, not a 20Hz BNS run or an
oracle comparison.  This recorded probe predates the controller and batch
contract correction; rerun the bounded physical probe after review before
treating its work counts as current.

The implementation choices are disclosed in the module docstring: Fehlberg
4/5 coefficients from NASA TR R-315, the preserved generic controller, and
the explicitly named `gsl_state_scaled` controller.  The latter uses
`atol + rtol*abs(y5)`, safety factor 0.9, strict 0.5/1.1 thresholds, q=5
exponents, and bounded factors.  Generic RHS calls default to the full
flattened batch shape and select active rows afterwards, which supports
distinct captured parameters after event histories diverge; an explicit
`batch_mode="active"` escape hatch remains available for subset-aware RHS
callables.  EOB coarse and fine trajectories use scalar row closures and
stack only equal-length rows.  Ragged physical waveforms therefore raise a
clear equal-length error until a padded waveform representation is supported.
Defaults are `atol=1e-10` and `rtol=1e-9`.  Domain failures,
rejected-step/evaluation/accepted-step budgets, duration limits, and
unresolved events raise explicit errors.

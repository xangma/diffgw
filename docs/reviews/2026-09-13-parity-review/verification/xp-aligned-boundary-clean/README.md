# XP v223 aligned boundary diagnosis

This clean-room report isolates and corrects the exact aligned and zero-spin
convention-1 chart residual in the clean MSA v223 composition.  It uses public
LALSimulation source-frame, angle, remnant, carrier, and XP waveform APIs as
black boxes, together with the local clean source-frame/MSA helpers.  The
production endpoint correction is implemented in
`src/torchwave/approximants/precession_msa_waveform_clean.py`; this diagnostic
is read-only with respect to runtime/helper physics and does not edit that
helper or the production adapter.

Run it from the repository root with:

```sh
PYTHONPATH=src:tests:scripts:. /Users/xangma/miniconda3/envs/pycbcorig313/bin/python \
  scripts/diagnose_xp_aligned_boundary_clean.py
```

The authoritative JSON is [report.json](report.json).  It records audited
fixtures separately from fresh fixtures, preserves full three-component spin
inputs, records clean-helper hashes before and after the run, and leaves the
unchanged `1e-6` strict gate visible.

The leading attribution is a source-frame chart mismatch at the endpoint.  In
the public convention-1 chart, exact alignment gives

```text
delta_alpha = phi_ref - pi/2 = -phi_J,
delta_gamma = +phi_J,
delta_alpha + delta_gamma = 0.
```

The m′=2 term in Table IV / Eq. (45) therefore has the same leading
`alpha+gamma` phase.  The clean helper nevertheless retains a finite,
root-regularized opening angle of about `3e-5`–`4e-5` radians.  The
m′≠2 Wigner terms are O(beta), so they expose the opposite alpha and gamma
chart offsets and produced the observed `2.5e-5`–`4.1e-5` relative complex L2
residual.  The native v223 endpoint now applies the same absolute chart
relation before projection, lowering the exact aligned and zero-spin rows to
the `1e-8` or smaller level without a fitted phase or threshold change.

The remnant scalar and the opaque XAS carrier are not the cause: the clean
v223 remnant differs from the public aligned fit by about `1e-10`, and the
carrier comparisons stay at numerical precision.  Convention 0 is an
ordinary control and remains within the gate for the exact aligned and
zero-spin fixtures.  The tiny nonzero transverse neighbor remains a separate
known MSA internal failure and is reported visibly; it is not captured by the
exact-zero selector.

The reusable correction contract is: for exact aligned v223 convention 1,
apply the paired endpoint chart mapping before the Eq. (45) projection,
`delta_alpha = -phi_J` and `delta_gamma = +phi_J`.  Retain the current finite
`cos(beta)` and root behavior, keep exact-zero selection separate from tiny
nonzero transverse spins, and retain the Table IV carrier phases.  The
minimal regression is
`tests/test_xp_aligned_boundary_clean.py`; its convention-1 assertion now
passes with the production gate unchanged.

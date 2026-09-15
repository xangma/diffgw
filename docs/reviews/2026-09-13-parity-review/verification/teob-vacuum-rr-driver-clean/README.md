# Bounded clean TEOB vacuum RR driver

This verification covers the bounded integration of the clean nonspinning
TEOB radiation component into the native TEOBResumS equations.  The route is
selected only when both aligned spins and all stored quadrupolar/octupolar
tide couplings are identically zero.  Spin or tidal systems continue through
the existing TEOBResumS radiation fallback.

The clean component evaluates the reviewed 35-mode infinity sum at `l <= 8`
and optionally adds the nonspinning horizon flux.  `include_horizon_absorption`
is an explicit boolean on `TEOBResumSRadiationReaction` and
`TEOBVacuumRadiationDriver`; the compatibility vacuum constructor enables it
by default.  The force handoff is

```text
F_r* = 0
F_phi = - Jdot_clean / nu
```

where `Jdot_clean` is the physical angular flux returned by
`TEOBRadiationClean`.  The clean driver supplies the state-dependent inputs
directly from the Hamiltonian:

```text
H_eff = H_eff(r, p_r*, p_phi)
H     = H_real(r, p_r*, p_phi) / M
Hcirc = H_real(r, 0, p_phi) / M
A'    = partial A(r) / partial r |_momenta
```

Thus `Hcirc` evolves with the current `p_phi`; a circular angular momentum is
not substituted.  `A'` is obtained by differentiating a fresh radius node,
which keeps it a coordinate partial.  The real total energy normalization is
kept separate from the effective energy.  The radiation component then forms
`rOmega` and `vphi = rOmega * Omega` from these inputs.  Both the fixed-step
trajectory wrapper and the native frequency-peak event integrator use this
same force path at every RK stage.

The focused checks are:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src:tests:scripts:. \
  /Users/xangma/miniconda3/envs/pycbcorig313/bin/python -m pytest -q \
  tests/test_teob_radiation_clean.py tests/test_teob_vacuum_driver_clean.py
```

Result: `14 passed`.

The bounded diagnostic is:

```bash
PYTHONPATH=src:tests:scripts:. \
  /Users/xangma/miniconda3/envs/pycbcorig313/bin/python \
  scripts/diagnose_teob_vacuum_rr_clean.py
```

It reports 35 modes, finite scalar/batch quantities, a finite gradient, zero
radial radiation force, a nonzero horizon increment, and a short `4 M` fixed
step refinement (`dt=0.5 M` versus `0.25 M`) with endpoint difference about
`3.75e-15` in the current environment.

This is a component and short-trajectory verification.  It does not run a
1.4+1.4 solar-mass 30 Hz inspiral, whose duration spans millions of `M`.
Full-duration work still needs a costed integration plan, domain/event
handling through plunge, NQC input extraction, merger/ringdown conventions,
and validation of those conventions.  The bounded driver makes no claim of
full TEOBResumS parity.

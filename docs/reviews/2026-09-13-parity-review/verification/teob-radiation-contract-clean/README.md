# Clean-room TEOB radiation contract

This verification records the bounded nonspinning radiation changes in
`src/torchwave/approximants/eob/teob_radiation_clean.py`.  The implementation
was reconstructed from the rendered primary equations and uses no LAL or
author implementation source, source-only coefficient table, numerical
oracle, or calibration fit.

The equation checks used the local rendered texts
`/tmp/torchwave-clean-teob-primary/1212.4357v2.txt` and
`/tmp/torchwave-clean-teob-primary/1806.01772.txt`.

## Contracts applied

* The Newtonian mass polynomial is
  `X2**(ell+epsilon-1) + (-1)**m*X1**(ell+epsilon-1)`.  Thus odd `ell=5`
  modes already contain `(X2-X1)*(1-2*nu)`.  The default no longer applies a
  second `sqrt(1-4*nu)` factor.  `legacy_l5_odd=True` remains an explicit
  compatibility path for old serialized runs.
* For `(3,2)`, `(4,2)`, and `(4,4)`, the displayed highest `v**9` terms are
  retained at `nu=0` and omitted at positive `nu`, following the paragraph
  before Eq. (D35) of 1212.4357v2.  The printed `(2,1)` Padé denominator keeps
  its literal `2456` coefficient.
* Horizon absorption follows 1212.4357v2 Eqs. (36)--(45).  With `x=vphi**2`,
  the dimensionless correction is

  ```text
  (1 - 4*nu + 2*nu**2) *
      (x**4 * Heff**2 * rhoH22**4 +
       x**5 * (pphi/(rOmega*vphi))**2 * rhoH21**4).
  ```

  The 4PN `rhoH22` and `rhoH21` coefficients are the printed Table-I values.
  `horizon_flux` multiplies this ratio by
  `(32/5)*nu**2*rOmega**4*Omega**5`, matching the physical normalization of
  the waveform angular-flux sum.  `radiation_reaction_torque` divides the
  total emitted angular flux by `nu`, giving the reduced force in Eq. (36)
  with zero radial force.  The `nu**2` factor occurs once in the physical
  flux, so the horizon term is not double-counted.

## Independent checks

`tests/test_teob_radiation_clean.py` covers the equal-mass zero, linear
odd-`ell=5` behavior, exchange sign, phase truncations, leading horizon
powers, the two source squares, exchange and `lmax` independence, additive
flux and torque normalization, dtype convergence, and autograd.

The reproducible diagnostic is
`scripts/diagnose_teob_radiation_contract_clean.py`.  Its float64 output on
2026-09-14 was:

```json
{
  "horizon_exchange_relative": 0.0,
  "horizon_float32_relative": 5.6141852027264025e-08,
  "horizon_gradient": 7.760604115337935e-13,
  "horizon_leading_ratio": 1.0003609414187322,
  "horizon_source_quadratic_residual": 3.1554436208840472e-30,
  "horizon_torque_contract_residual": 0.0,
  "horizon_total_additivity_residual": 5.134207910940304e-22,
  "l5_equal_mass_abs": 0.0,
  "l5_exchange_relative": 0.0,
  "l5_linear_ratio_d2_over_d1": 1.0000000001174487,
  "phase_v9_residuals": {
    "(3, 2)": 0.0,
    "(4, 2)": 0.0,
    "(4, 4)": 0.0
  }
}
```

The focused suite passed with the project Python environment:

```text
9 passed in 0.68s
```

The related TEOB radiation-reaction, potential, and published-driver checks
also passed (`34 passed in 1.54s`).

## Driver coordination and limits

`TEOBResumSRadiationReaction` currently receives only `(Omega, pr_star,
pphi, r, hamiltonian)` and routes its nonspinning point-mass path through a
separate factorized energy-flux helper.  The clean component requires the
explicit `Hcirc` and `Aprime` state needed for `rOmega`, so production driver
wiring remains a next-driver task.  The planned choice is an explicit
`include_horizon_absorption=False` driver option, enabled only on the
nonspinning vacuum route after those quantities are supplied; the present
clean torque API defaults to including absorption when called directly.

For the canonical zero-spin, zero-tides 1.4+1.4 solar-mass gate at 30 Hz,
the remaining work is to pass the evolving circular `Hcirc` and `Aprime`,
select that horizon policy, feed the returned `(Fhat_rstar, Fhat_phi)` into
`dpr_star/dt` and `dpphi/dt`, and validate the full-duration merger driver.

The 2018 paper's Appendix E was read for the published speedy-tail option.
It describes an implementation optimization for the phase of the Gamma tail
factor (Eqs. (E1)--(E6)); it does not alter the radiation contracts here, so
no speedy-tail approximation or calibration code was copied.  NQC,
calibrated potentials, spins, tides, merger attachment, ringdown, and a full
TEOB driver remain outside this component.

## Primary provenance

* Damour, Nagar & Bernuzzi, arXiv:1212.4357v2, Sec. II.D Eqs. (36)--(45),
  Table I, and Appendix D Eqs. (D35)--(D48).
* Nagar et al., arXiv:1806.01772v2, Sec. II.B.1 (the historical odd-`ell=5`
  factor representation) and Appendix E (speedy-tail implementation notes).
* Nagar et al., arXiv:1406.6913v1, Eqs. (69)--(76) for the non-Keplerian
  velocity and waveform angular-flux normalization.

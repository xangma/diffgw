# Clean-room MSA composition diagnostic

Run from the TorchWave checkout with:

```sh
PYTHONPATH=src:tests:scripts \
  /Users/xangma/miniconda3/envs/pycbcorig313/bin/python \
  scripts/diagnose_msa_waveform_composition_clean.py
```

The command writes `report.json` in this directory. It evaluates cases A,
B, C and an aligned control on 20--512 Hz at 1 Hz, for `phic=0` and `0.2` and
conventions 0 and 1. Every waveform metric is a raw complex `h_+`/`h_x`
relative L2 and maximum absolute error. No phase, time, norm, affine, or
other residual correction is fitted.

The public black-box calls are recorded with their exact inputs in each row:

* `SimIMRPhenomXPMSAAngles(frequency_vector, m1_SI, m2_SI, S1x,S1y,S1z,
  S2x,S2y,S2z, inclination, f_ref, mprime=2, lal.Dict)` returns
  `(alpha,gamma,cos(beta))`.
* `SimIMRPhenomXPCalculateModelParametersFromSourceFrame(m1_SI, m2_SI,
  f_ref, phiRef, incl, S1x,S1y,S1z,S2x,S2y,S2z, lal.Dict)` returns
  `(chi1,chi2,chip,thetaJN,alpha0,phiJL0,zeta)`.
* `SimInspiralChooseFDWaveform(..., IMRPhenomXP)` is the direct waveform
  target on the same physical masses, full spin vectors, distance, and
  frequency grid.

One shared SI physical-argument record feeds both public metadata and
waveform calls, and every inserted public option is read back with its lookup
API. The primary reference is public `prec_version=223`, expansion order 5,
`FinalSpinMod=0` for composition isolation; public versions 220, 223, and
224 are retained as metadata samples.

The identified frame contract is:

1. Map `phiJ=-phiJL0` and use public `thetaJN` and `alpha0`.
2. The public convention-0 `zeta` needs the Appendix C ABFO/Kidder basis
   relation `+pi/2` before entering the clean projector. Public convention-1
   metadata already includes that basis shift, so no second metadata shift is
   applied. The convention-1 composition then applies its separate Table-IV
   projector shift `-pi/2` exactly once. Since polarization depends on `2zeta`,
   geometry comparisons also report zeta modulo `pi`.
3. The angle API has no `phiRef` argument. For a requested `phic`, transport
   its absolute triplet from the `phic=0` frame by
   `alpha += alpha0(phic)-alpha0(0)` and, for convention 1,
   `gamma -= phiJ(phic)-phiJ(0)`, with wrapped public source-frame deltas.
   When the transverse spins vanish, `phiJL0` is gauge-degenerate and the
   convention-1 gamma transport is omitted; the aligned projected limit is
   reported explicitly.

This contract closes the generic composition boundary. Across all generic
cases, phases, and conventions, transported public angles with mapped public
geometry give direct-LAL `FinalSpinMod=0` raw relative-L2 errors from
`3.7e-14` to `7.3e-14` in either polarization. Leaving the angle triplet
unchanged at `phic=0.2` gives `0.321`--`0.428`; the equation-level transport
removes that reference-frame offset. Holding the transported angles fixed
while using the fresh clean geometry leaves `1.57e-4`--`1.72e-3`, isolating
the remaining small source-geometry approximation rather than a carrier
projection error. The aligned control remains below `1e-12`; its convention-1
`phic=0.2` branch requires the alpha transport and carries the documented
gauge-degenerate gamma handling.

The report also audits the source-frame components independently of waveform
residuals. It inverts the Appendix C rotations to obtain `Jhat`, then solves
`J=S+L zhat` for a conditional effective `L/M^2`, with the transverse
component residual recorded for every public tuple. For the three precessing
cases, public versions 220 and 224 reproduce the clean source-frame 3PN
`pn_version=102, compat_102=False` direction to at most `2.3e-16` and the
corresponding `L/M^2` to at most `3.0e-15`. Version 223 instead differs in
`Jhat` by `8.75e-5`, `6.63e-4`, and `4.34e-4` for A, B, and C, with conditional
effective-L offsets `1.30e-3`, `2.30e-3`, and `3.24e-3`. Its source tuple also
changes `thetaJN`, `alpha0`, and `zeta` while leaving `phiJL0` fixed. The
paper's version table identifies a distinct v223 `phi_z,0`/`zeta_z,0`
prescription, so these v223 values remain a diagnostic conditional inversion;
no undocumented L expression or fitted correction is promoted. At `phic=.2`,
the independently inverted `Jhat` returns to the zero-phase direction within
`3.0e-16` and reconstructs `N_L` within `3.7e-16`, validating the coordinate
transport. The aligned control reports the `J_perp=0` branch explicitly,
where direction alone cannot determine `L`.

The independent Eq. (45) alpha covariance and Eq. (146) zeta half-turn sign
checks are included in `report.json`; their maximum absolute errors are below
`2e-16` and `6e-17`, respectively.

The native wrapper is explicitly experimental. It uses the clean order-5 MSA
assembler with Newtonian `cos(beta)` by default (`beta_pn=True` selects the
separate clean PN beta path) and makes no public `prec_version` claim.

The report records Python, Torch, LAL and LALSimulation versions, SHA256 hashes
for the clean source modules and imported dependency modules, and the primary
paper TeX hash. The final-spin rows show both source-total and remnant-mass
normalizations implied by Eqs. `MSA_finalSpin`, `afin_L_fin`, and the printed
`finalspin_3` equation.

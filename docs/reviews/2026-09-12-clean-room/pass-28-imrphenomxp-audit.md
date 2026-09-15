# Pass 28: IMRPhenomXP clean-room residual audit

Date: 2026-09-13

## Scope and source boundary

This pass audited the current `IMRPhenomXP` carrier, twist-up, frame-angle,
remnant, and public frequency/phase conventions against Pratten et al.,
Phys. Rev. D 103, 104056 (arXiv:2004.06503v2), especially Eqs. (11)--(12),
(30)--(33), and (39)--(42), and the PhenomXAS source equations in
arXiv:2001.11412. No LALSimulation or PyCBC implementation source was read.
The installed public LAL API was used only for black-box convention and
residual diagnostics.

## Findings

The carrier and twist-up are not the remaining uniquely identifiable defect.
The XP implementation supplies the XAS carrier at face-on inclination, uses
the independent Table-IV carrier phase, and applies the Wigner `d^2` factors,
`exp(-i m alpha)`, `exp(2 i gamma)`, and observer-basis rotation. This is the
structure of Pratten et al. Eqs. (11)--(12). The aligned convention-1 branch
also uses the XAS waveform directly, with `phi_c=phic-phi0`.

The strongest source-supported discrepancy is the MSA final-spin prescription.
The paper's FinalSpinMod3 Eq. (42) is

```
|a_fin| = sqrt(<S_fin^2>_pr) / M^2,
```

whereas Eq. (31) defines `L_fin` through `M_fin^2 a_fin,parallel`.
With all spins aligned this gives the exact continuation

```
a_Mod3 = (M_fin/M)^2 a_parallel.
```

The public aligned XP output differs from XAS by about 5.23% in L2 norm for
the checked `(35,25,0.2,-0.1)` case, consistent with this remnant choice.
This identifies a source-defined physical effect, but not a safe patch for the
current API: the wrapper currently defaults to NNLO version 102, while the
paper's MSA default is version 223 with FinalSpinMod3. Existing tests and
reports intentionally preserve that NNLO default for compatibility.

The generic MSA remnant is not fully determined for every runtime row by the
published text. `compute_msa_averaged_couplings` returns the Eq. (4.20)
precession-averaged quantities only where its elliptic-root initialization is
nondegenerate. The paper itself notes the equal-mass singularity; it does not
specify a continuation for equal masses, one-spin cases, aligned roots, or
other invalid rows. Applying instantaneous spin sums or an arbitrary masked
value would therefore be an implementation convention rather than a
source-complete correction. Applying Mod3 only on valid rows would leave the
aligned source limit discontinuous. No runtime patch was made for this reason.

The frozen residuals also do not reduce to a phase gauge or polarization
rotation. A black-box ratio decomposition against public XP output gave:

* catalog: amplitude nonlinear RMS 0.037 (plus) / 0.047 (cross), nonlinear
  phase RMS 0.042 / 0.052 rad after removing a fitted constant and affine time
  term;
* holdout: amplitude nonlinear RMS 0.186 / 0.258, nonlinear phase RMS 0.239 /
  0.144 rad, with different fitted affine slopes in plus and cross.

Thus a `phiRef`, `tc`, `f_ref`, or polarization-gauge adjustment cannot be
uniquely inferred from these cases. The holdout residual is also too nonlinear
to justify changing the carrier phase or observer azimuth convention.

## Verification

Focused clean-room precession and MSA tests pass:

```
97 passed in 10.78s
```

No source or runtime file was changed in this pass. The precise blocker for a
strict runtime correction is the undocumented invalid-MSA remnant continuation
(and, separately, the deliberate API choice of NNLO rather than the paper's
MSA default). A future patch can safely add an explicit, opt-in Mod3 path only
after its invalid-row policy is specified from an independent primary source.

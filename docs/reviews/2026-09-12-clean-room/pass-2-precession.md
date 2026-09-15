# Clean-room precession reconstruction, pass 2

The generic two-spin MSA branch now includes spin nutation, the nutation
phase, secular Euler phases, and leading oscillatory Euler corrections.
This replaces the inherited secular-only approximation. It is a reconstruction
of the published analytical prescription, **not a claim of LAL parity or of
exact equivalence to LAL's numbered implementation variants**.

Only `precession_msa.py`, `imrphenom_xp.py`, `imrphenom_xphm.py`, the new
`tests/test_clean_room_msa.py`, and this report changed in this pass. The
previous pass's frame, Wigner, polarization, SPA-frequency, and ODE corrections
remain in place. No commits were made by this worker.

## Boundary and primary sources

This pass used the current implementation, newly obtained primary-paper PDF
and TeX equations, and independent equation/invariant tests. It did not read
LAL/PyCBC implementation source, ports, historical implementation audits, binary
oracle artifacts, or oracle residuals; it did not call a waveform oracle or fit
parameters to one. The primary-paper source archives were used only for their
paper TeX and bibliography, not bundled numerical notebooks or waveform data.
All downloaded source assets are outside the repository.

| Source | Equations used | SHA-256 of downloaded asset |
|---|---|---|
| [Chatziioannou et al., arXiv:1703.03967v2](https://arxiv.org/pdf/1703.03967v2) | Conservative cubic, elliptic solution, MSA equations and appendices | `a054f16caa31560b632f03f7a2280e576a9c4b5f49f4d8d32c058bb7fc403b7c` |
| [Corrected paper TeX, 1703.03967v3](https://arxiv.org/src/1703.03967v3) | D6–D9 corrections and all phase formulas | `64f6114885f1c6d5ab5afdaaa0d6cf44d65eb3a97f7a5b84df324397a39fdf92` |
| [Pratten et al., 2004.06503v2 paper TeX](https://arxiv.org/src/2004.06503v2) | Sec. IV, G10, Tables III–IV | `5295de1720953660c50b64cb4a340ac3ae1075e334c8203eb9cce412d92143bf` |
| [Chatziioannou et al., 1408.5158 paper TeX](https://arxiv.org/src/1408.5158) | Appendix A radiation-reaction spin couplings | `bcc73d175989589fc042ec6b7b9309ff61fd3369bd32ef14cb2b7556b05ff64c` |
| [Bohé et al., 1212.5520v2](https://arxiv.org/pdf/1212.5520v2) | Eq. 4.7 orbital angular momentum | `c28993b44523c7378c5181e62a13fc216d6596c81c56277de1b0be451e6ac7eb` |
| [Bohé et al. paper TeX, latest v2](https://arxiv.org/src/1212.5520) | `equations/LiCO.tex`, Eq. 4.7 | `ded07742fca678ffc8594ae42832ebfb7d964dbda762f539518f4a505c30f664` |
| [Khan et al., 1809.10113 paper TeX](https://arxiv.org/src/1809.10113) | Sec. II E series truncation and PN L for beta; follow-up review | `0394a6eb3032838e0977309ef4941f4d32a5193add57ad3359b315dffa3ef611` |

The elliptic algorithms use the mathematical identities in
[DLMF 19.26](https://dlmf.nist.gov/19.26),
[19.36](https://dlmf.nist.gov/19.36), and
[22.20](https://dlmf.nist.gov/22.20). They are tensor-native implementations
of Carlson duplication for incomplete F and descending Landen/AGM for Jacobi
sn, cn, dn, with fixed iteration counts and autograd support. These counts are
numerical convergence choices, not waveform calibration parameters.

## Implemented mathematics

1. Initialize geometric spins `S_i = (m_i/M)^2 chi_i`, conserved effective
   spin xi, and the Newtonian angular momentum `L_N = eta/v0`, as required by
   the MSA derivation. Solve the Appendix B cubic for
   `S3² <= Sminus² <= Splus²`. Set `Sav²=(Splus²+Sminus²)/2` and determine
   `c1=v0*(J0²-L0²-Sav²)/2` from the supplied reference spin vectors.
2. Implement the precession averages of Appendix A, including the variance
   terms in the squared projections and covariance. Use the published mass
   factors in beta3 and beta5: the coefficient multiplying primary geometric
   spin contains `q=m2/m1`; the secondary coefficient contains `1/q`.
3. Compute the 1PN nutation phase of Eq. 51. Its expansion coefficients are
   derived by expanding the defining cubic with Vieta's relations: the two
   finite roots and the `v^-2`, `v^-1`, and constant terms of the third root
   determine the expansion of `sqrt(Splus²-S3²)`. This is algebraically
   equivalent to expanding Eq. 50 and avoids nested cancellation in C3.
4. Initialize the inverse-elliptic phase from the supplied total spin
   magnitude; the scalar triple product fixes its sign through Eq. 28.
   Evaluate frequency-dependent roots and
   `S²=Splus²+(Sminus²-Splus²)*sn(psi|m)²`, with
   `m=(Splus²-Sminus²)/(Splus²-S3²)`.
5. Evaluate all six secular alpha primitives D21–D26. Each differentiates to
   `J*v^(n-3)`. The default truncation retains n=0,...,4, the O(5) prescription
   of Pratten Sec. IV B. Evaluate the matching epsilon primitives, including
   the logarithmic n=3 primitive.
6. Derive the leading oscillatory corrections directly by real partial
   fractions of Eq. 30 and F1 at m=0. Both are periodic primitives with zero
   cycle mean. Their fast-time derivatives are exactly
   `(Omega-mean(Omega))/psi_dot`, including the factor J. This avoids the
   phase/prefactor ambiguities in the printed F19 expression.
7. Anchor secular plus oscillatory angles at v0. For the opening angle,
   insert the selected PN L into the averaged relation from Eq. 40,
   `J_PN²=L_PN²+2*c1*L_PN/eta+Sav²`, then apply Eq. 8 / Pratten 4.12,
   `cos(beta)=(J_PN²+L_PN²-S²)/(2*J_PN*L_PN)`.
   Conservative roots and the angle phases retain Newtonian L. The supplied
   reference-spin cone is exact in the Newtonian limit. The PN prescription
   does not impose this additional constraint; see the follow-up below.

## Paper corrections and derivation checks

The latest v3 source corrects the delta placement in D6, the sign of D7's
`Rm²` term, and adds the D9 coefficient. Those corrections are included.
Additional inconsistencies are resolved from the defining equations:

* **D4:** `a1=1/2+3/(4 eta)`, rather than the printed `1/2+3 eta/4`.
  This is the leading constant from expanding Eq. 30/B5. The printed
  expression would even reverse alpha evolution for ordinary examples.
* **D22:** its logarithmic coefficient contains `c1²-eta² Sav²`, not
  `c1-eta² Sav²`. Differentiation of the corrected primitive gives `J/v²`;
  the printed form does not. All six primitives have derivative tests.
* **F5/F9:** the n=3 term multiplies `log(v)`, because its rate is proportional
  to `v^-1`; treating it as a constant loses that term entirely.
* **Nutation sign:** use positive `psi_dot=|A| sqrt(Splus²-S3²)/2` and the
  Eq. 28 scalar-triple-product branch. The negative A in B1 conflicts with
  that branch relation and the positive derivative of the Eq. 51 solution.
* **Oscillatory Euler terms:** direct integration retains the J prefactor
  from Eq. 30, and fixes epsilon from F1. Numerical differentiation is checked
  against the independently expressed geometrical Eq. 29 rate and a dense
  cycle average, with relative agreement better than 1e-10.
* **Pratten G10h:** the coefficient of delta*Sigma at 3.5PN is `-81/16`.
  The original derivation, Bohé Eq. 4.7, supplies this value instead of the
  printed `-81/6`. It also gives the Kerr test-mass linear-spin expansion
  `-10/3*v³ - 7*v⁵ - 81/4*v⁷`, verified independently.

## API and limits

The default API is now honestly named: `precession_method='nnlo'`,
`prec_version=None`, resolving to 102 and preserving the prior default
waveform behavior. Explicit `precession_method='msa'` with an omitted version
resolves to 223. An explicit supported version takes precedence over method.
Unsupported method/version values raise, including on aligned scalar calls.
For XPHM, an explicit numbered version is rejected with the experimental ODE
method because it does not apply to that method.

| Versions | Implemented behavior |
|---|---|
| 101 | NNLO angles, 2PN nonspinning L |
| 102 | NNLO angles, 3PN spinning L |
| 103 | NNLO angles, 4PN L and 3.5PN spin-orbit L |
| 104 | As 103, with leading quadratic/cubic spin terms G10i–j |
| 220, 223, 224 | Published generic MSA angles; NNLO/3PN fallback on failed initialization |
| 221, 222 | Same analytical MSA equations; error on failed initialization |

The discriminant validity criterion uses floating-point relative error bounds,
not an absolute spin or empirical mass-ratio threshold. Equal-mass, one-spin,
repeated-root, and nonreal secular-coefficient initializations select the
stated fallback. In particular, the frozen-root expansion requires real
`sqrt(cp*cm)`: clamping a negative product to an arbitrarily small positive
number previously generated enormous phases. The caller now rejects that MSA
initialization for the whole binary. Masked rows use a regular surrogate
internally to keep gradients of the selected NNLO branch finite.

The following remain explicitly unresolved:

* The papers describe labels 222/223 as closer to Pv3HM and label 224 as
  mixing prescriptions, without publishing the exact numerical distinctions.
  These labels currently share one mathematical solution. They must not be
  advertised as exact LAL-version reproductions.
* The published elliptic spin solution has period `2K(m)`, while the leading
  m=0 oscillatory corrections have period pi. Inverse-elliptic branches can
  consequently differ by formally omitted O(m) terms near a turning point.
  No fitted phase redefinition was introduced. Parameter differentiability
  is established at regular initializations, not at every branch boundary.
* Dedicated analytic equal-mass/single-spin MSA limits are not implemented;
  these use the documented NNLO fallback where the selected version allows it.
* The PN/MSA approximations can become inaccurate near alignment, spin-orbit
  resonances, and strong-field merger. This pass supplies the stated analytical
  continuation, not a new merger calibration.
* The pass-1 offset-convention limitations remain: default convention 1 uses
  C16 XP offsets instead of the distinct Pv2 alpha offset in Table IV; the
  other convention options are not complete reproductions of that table.
* Precession-dependent remnant parameters and carrier conventions are outside
  this MSA implementation and still affect full-waveform parity.

## Verification

Final command:

```text
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /Users/xangma/miniconda3/envs/pycbcorig313/bin/python -m pytest tests/test_clean_room_msa.py tests/test_clean_room_precession_equations.py tests/test_imrphenomp.py tests/test_imrphenom_xphm.py tests/test_precession_ode.py tests/test_phenomxp_msa_reconstruction.py -k 'not lal' -q
```

Result: **54 passed, 2 oracle tests deselected** in 5.04 seconds. Oracle tests
are reserved for the separate verifier; no parity threshold changed here.
The new tests cover elliptic inverse identities and differential equations,
the six secular primitives, both oscillatory rates and zero means, 1PN
nutation-phase convergence, reference geometry, frequency-grid independence,
intrinsic/frequency gradients, mixed regular/degenerate batches, fallback/error
policies, explicit method/version behavior, and the 4PN Kerr spin limit.

A separate deterministic 256-binary probe spanning mass ratios 1–10 and
generic physical spins found finite outputs and all 256 finite intrinsic
gradient rows on v={0.12,0.22,0.32,0.45}. This was a numerical diagnostic,
not an oracle comparison or a calibration dataset. `git diff --check` passes
for the changed code and focused test file.

## Follow-up: series order, PN opening angle, and branch periodicity

The separate verifier requested this bounded equation audit after reporting
residual discrepancies. No binary was called and no oracle arrays, target
angle values, or residual artifacts were read. The verifier's summary only
identified the questions to check. Defaults and parity thresholds are unchanged.

**Series order: no change.** Pratten Sec. IV B and Khan Sec. II E explicitly
number frequency terms as `f^(-4/3+k/3)`, `k=1,...,6`. This is a different
index from the coefficient index `n=0,...,5` in Pratten Eq. 4.11 / Chatziioannou
Eq. 66. Both papers state that the highest term is omitted in their chosen
model. Thus O(5) retains coefficient indices 0–4, through the linear-v term in
epsilon. Interpreting the setting as including coefficient 5 would retain all
six terms and contradict this prose. This resolves the paper interpretation;
it does not establish the semantics of any undocumented binary interface.

**PN opening angle: corrected.** The first implementation upgraded L while
preserving instantaneous spin projections reconstructed from the Newtonian
triangle. This implied

```text
J_old² = L_PN² + (L_PN/L_N)*(J_N²-L_N²-S²) + S².
```

Consequently J_old oscillated with S² whenever L_PN differed from L_N.
This conflicts with the precession-averaged J used in Chatziioannou Sec. IV B.
Equation 40, with the retained constant Sav², instead gives

```text
J_PN² = L_PN² + 2*c1*L_PN/eta + Sav².
```

This solves the defining averaged equation
`dJ/dL=(J²+L²-Sav²)/(2JL)` and has no nutation-phase dependence.
The code now inserts this J and L into the opening-angle triangle. The
initial roots, c1, Sav², spin solution, and secular/oscillatory alpha and gamma
are unchanged. Independent tests recover J from beta at different S² and
differentiate it to the defining averaged equation; a reference test checks
the PN triangle and exact Newtonian supplied-spin limit.

Using Eq. 40 for the PN substitution is an inference from the defining
averaged equation and the papers' instruction to upgrade L for beta. The
papers do not spell out every intermediate quantity used by numbered
implementation variants. In particular, substituting PN L only in selected
occurrences of the velocity-expanded Eq. 41, or refitting c1 after that
substitution, defines other higher-order prescriptions. Exact distinctions
between 220 and 223 remain unresolved; the corrected code does not claim
that the paper fully specifies their numerical implementation.

**Oscillatory branch: no change.** Chatziioannou Sec. IV D 2 explicitly sets
`m=0` in the first correction and substitutes the phase of Eq. 51 into
Eq. 67. Its trigonometric correction therefore has period pi. The elliptic
spin solution retains period 2K(m). The implemented atan2 primitive is
continuous across its trigonometric turning points and pi-periodic, as new
tests verify; the Jacobi spin term independently satisfies its 2K period.
Rescaling the phase to force these periods to coincide would add terms not
specified by Eq. 67. The mismatch of inverse-elliptic branches by omitted
O(m) terms remains a limitation of this mixed-order prescription, including
at exact coplanar branch boundaries. A globally smooth repair requires a
consistent higher-order or unexpanded oscillatory solution and is not
introduced solely to alter a binary residual.

Follow-up focused tests: **19 passed**. The full command above now gives
**56 passed, 2 oracle tests deselected**, in 4.98 seconds.

# Independent review of the Kerr eigenproblem solver

No unresolved equation or numerical defect was found in the reviewed angular
solver or the new radial solver. This review adds 49 equation-based tests; it
does not modify either implementation. Full generated-grid coverage,
interpolation accuracy and runtime migration are separate checks owned by the
generator review.

## Scope and evidence boundary

The target is spin weight -2, units M=1 and the time convention
`exp(-i omega t)`, with positive real frequency and negative imaginary
frequency. The requested branches are `(ell,m)=(2,2),(2,1),(3,3),(3,2),(4,4)`,
overtones `n=0,1,2`, spins through `a=0.99999`, and the signed 220 branch down
to `a=-0.9999`. The returned angular quantity is `A`, whose spherical value is
`ell*(ell+1)-2`; the radial equation instead uses
`lambda=A+a*a*omega*omega-2*m*a*omega`.

The implementation files reviewed were
`src/torchwave/approximants/kerr_angular.py` and
`scripts/kerr_leaver_solver.py`. No external QNM or waveform implementation,
port, executable oracle or numerical QNM table supplied solver inputs,
initial guesses, expected test values or review comparisons. During source
discovery, the previous generated author's-data module header and its first
spin-knot line incidentally surfaced; neither was used in this work. The
primary WKB publication also contains adjacent numerical comparisons, which
were not used. Thus the review's claim is independence of the derivation and
numerical inputs, rather than an assertion that no unrelated numbers ever
appeared on a source page.

| Primary source | Material checked | Downloaded PDF SHA-256 |
|---|---|---|
| [Cook and Zalutskiy, arXiv:1410.7698v2](https://arxiv.org/pdf/1410.7698v2) | Radial Eq. (5a), transformations and recurrence (23)-(44), angular equations (49)-(56), qualitative branch discussion in III A | `33973382059461835989057a47e93336e5233cccede48529b4b19cdc306d7330` |
| [Schutz and Will, ApJ 291 L33 (1985)](https://articles.adsabs.harvard.edu/pdf/1985ApJ...291L..33S) | Barrier quantization (6)-(9), converted from the paper's `exp(+i sigma t)` convention | `56beaa8a32107b8b6f2f22b67905557439a2ff7f5ed28efd8d578a88776e6aaf` |

## Equation and arithmetic review

The angular operator is the projection of `A0-c*c*x*x-4*c*x`, with
`c=a*omega`. Both implementations include an extra spherical basis function
before forming the quadratic moment, then truncate; this retains the final
diagonal contribution that squaring an already-truncated matrix would lose.
Sphere quadrature independently reproduces the operator. Perturbation at
`c=0` gives `A'=-8*m/[ell*(ell+1)]`, and its second derivative agrees with
off-diagonal second-order perturbation theory. Complex gradients obey the
complex-symmetric Hellmann--Feynman formula using a transpose, without a
conjugate transpose. Mirroring `m,c` preserves the required eigenvalue and
phase-fixed coefficient symmetries.

For the radial equation, an independent symbolic substitution of
`R=(r-r_plus)^xi*(r-r_minus)^(eta-alpha)*exp(i*omega*r)*Y(y)`, where
`y=(r-r_plus)/(r-r_minus)`, reduces Cook Eq. (5a) directly to the three
polynomial coefficients of Eq. (30). All three symbolic differences vanish.
The committed numerical test evaluates this substitution from logarithmic
derivatives of the original radial ansatz at several radii, including a
near-extremal spin, rather than reproducing the intermediate recurrence
definitions. The chosen inner-horizon exponent gives `alpha=1-4*i*omega`,
avoiding cancellation in that quantity near extremality.

The right continued fraction starts with the asymptotic `r_N`, descends
through `N,...,n+1`, and therefore returns `r_n`. Its assembly as
`beta_n-left+alpha_n*r_n` has the correct inversion indexing and signs.
The asymptotic square root explicitly selects the minimal solution with
negative real part. The left fraction is finite and uses the corresponding
previous-index alpha and beta. Increasing fraction depth, angular basis
size and arithmetic precision provides distinct numerical checks.

The WKB seed comes from the analytic Regge--Wheeler potential, evaluated at
its analytically determined maximum. An autograd calculation independently
verifies that its curvature is with respect to tortoise radius. The complex
conjugation relative to Schutz--Will gives damping in the declared time
convention. A separate homotopy starting at `ell=16` with spacing 0.05 agrees
with production's `ell=20`, spacing 0.1 for all nine Schwarzschild roots at
`ell=2,3,4`, `n=0,1,2`. These roots satisfy inversions 0 through 4, with
residuals normalized by their independently evaluated frequency derivatives
below `2e-12`; they are distinct and damping-ordered at zero spin.

The extended-precision angular path recomputes matrix coefficients and
`c=a*omega` in GMP arithmetic. Its double-precision eigenvector is used as a
direction in a stationary complex-symmetric Rayleigh quotient. An independent
65-decimal mpmath full eigensolve at three near-extremal and signed-spin
complex points agrees with the 160-bit quotient to at worst `5.47e-30`.
This checks the new precision path without importing any QNM solution.

## Branch tracking and numerical checks

A converged continued fraction does not establish an overtone label: different
inversions can share the same zero. Production labels instead originate in
large-ell WKB continuation followed by spin continuation. The final seedless
spin path uses `x=-log(1-abs(a))`, spacing at most 0.025 and a secant predictor,
with a correction bound relative to the preceding damping scale. A caller
providing a frequency seed remains responsible for that seed's branch.

An early review probe deliberately used only 40 uniform steps in
`u=sqrt(1-a)` descending from 0.99999 to 0.99. The 322 branch jumped to 321
while continuing to satisfy the root residual. Its initial relative horizon
step was too large; this observation motivated relative near-extremal
resolution and reverse-continuation checks. It is not an unresolved failure
of the final continuation implementation.

The refined independent check follows 322 from 0.99 to 0.99999 and back over
401 geometrically spaced `u` points. It predicts in `u`, unlike production's
logarithmic-coordinate predictor. Every reverse point agrees with its forward
counterpart within `1.33e-14`; the largest predictor correction divided by
the previous damping is 0.00869. This is a targeted branch check, not a
replacement for reverse checking all generated rows. The qualitative
coexistence of damped and zero-damping branches for 21 and 32 is consistent
with Cook section III A; sorting by damping independently at every spin
would not preserve the Schwarzschild labels.

The latest focused test command passed **65 tests, with one author-fixture
test deliberately deselected, in 3.48 seconds**:

```sh
/Users/xangma/miniconda3/envs/pycbcorig313/bin/python -m pytest -q \
  tests/test_kerr_independent_angular.py \
  tests/test_kerr_independent_radial.py \
  tests/test_clean_room_kerr_angular.py -k 'not fresh_author'
```

The 49 new tests cover angular sphere projection, first/second derivatives,
complex gradients, mirror phases and finer continuation; direct radial-ODE
substitution; WKB curvature; independent Schwarzschild labeling; cross-inversion
residuals; and distinct roots. The precision and 322 roundtrip probes ran
separately with `/Users/xangma/miniconda3/bin/python`, which supplies gmpy2.
The focused Torch environment does not supply that generation-only package.

No reference frequency is embedded in the new tests. Completion of this
review does not certify every generated interpolation cell, frequencies
beyond the declared domain or agreement with an external waveform model.

# Independent Kerr QNM solver implementation

`scripts/kerr_leaver_solver.py` now computes original spin -2 Kerr roots from
the published eigenproblem. It reads no external QNM implementation,
waveform oracle, author frequency table, or pretrained/interpolated spectrum.
The numerical inputs are mode labels, spin and continuation seeds obtained
from earlier solves of the same equations.

The frozen solver used for the first full independent generation has SHA-256
`2439ed8b2a2331656fff758cefaf578acc3eed5b0a3d443e9e98a66a252d95b0`.
The generation environment is `/Users/xangma/miniconda3/bin/python`, with
NumPy 2.5.2, SciPy 1.16.3 and gmpy2 2.3.1. These supply generic linear
algebra, root finding and arithmetic, respectively. gmpy2 is only needed
for near-extremal damped or counterrotating solves; the ordinary solver
path uses NumPy/SciPy.

## Equations and conventions

[Cook and Zalutskiy, arXiv:1410.7698v2](https://arxiv.org/pdf/1410.7698v2)
provides the radial Teukolsky transformation, recurrence and asymptotic
remainder in Eqs. (23)-(44), and the spherical spectral angular construction
in Eqs. (49)-(56). The downloaded primary PDF has SHA-256
`33973382059461835989057a47e93336e5233cccede48529b4b19cdc306d7330`.
The radial code chooses outgoing behavior at infinity and ingoing behavior
at the event horizon. The consistent inner-horizon exponent gives
`alpha=1-4*i*omega`. The square-root branch of the asymptotic remainder is
chosen explicitly to have negative real part, selecting the minimal
recurrence solution.

The time dependence is `exp(-i*omega*t)` and M=1: frequencies have positive
real part and negative imaginary part. The returned angular constant is
`A`, with `A(0)=ell*(ell+1)-2`; it is not the radial combination
`lambda=A+a*a*omega*omega-2*a*m*omega`.

The initial branch labels use parabolic-barrier WKB of the analytic
Regge--Wheeler potential, as in
[Schutz and Will's WKB construction](https://ntrs.nasa.gov/citations/19850052499).
Both its maximum and its second derivative with respect to tortoise radius
are evaluated analytically. WKB only initializes a root at ell=20; a
continuous separation-constant homotopy with ell spacing 0.1 labels the
low-ell Schwarzschild roots. The WKB frequencies are not used as final data.

## Continuation and accuracy checks

The seedless API follows the Schwarzschild label along
`x=-log(1-abs(a))`, with spacing no greater than 0.025 and a secant predictor.
The correction must remain within 10% of the previous damping scale.
Supplied seeds explicitly transfer branch responsibility to the caller:
the inversion number alone does not label an overtone. The generator must
therefore reverse its paths and reject overtone collisions.

Every root recomputes the angular eigenvalue inside the radial objective.
The angular representation includes one additional spherical basis element
before constructing the quadratic moment. Final roots are re-evaluated
with four additional basis elements and at twice the continued-fraction
depth. Convergence requires both frequency stability and a small residual
converted to a frequency error using the local derivative of the coupled
radial/angular equation. The result records the raw residual, depth,
remainder error, angular refinement error, arithmetic precision, root
evaluation count and convergence flag.

Near-extremal damped branches need their minimal-solution decay scale
resolved before the first Newton correction. Starting shallow and only
refining after moving the root can converge to another valid branch.
The initial depth now accounts for this decay scale. Where double
arithmetic loses accuracy, the recurrence uses 113-bit GMP arithmetic;
strict verification uses 160 bits. The angular matrix is also recomputed
at that precision. Its double-precision eigenvector supplies a direction
for a stationary complex-symmetric Rayleigh quotient, so direction errors
enter quadratically. Basis enlargement remains a separate check.

Representative results below are fresh outputs of this implementation,
not imported expected values:

| Mode at a=0.99999 | Complex M omega | Ordinary/strict difference |
|---|---|---:|
| 2,1,0 | `0.5814252142780002-0.03826480068711239j` | 0 |
| 2,1,1 | `0.5388633469726795-0.11863773497535467j` | 0 |
| 3,2,0 | `1.0285267472052337-0.018606364756720593j` | `2.49e-16` |
| 2,-2,0 | `0.29155401590590324-0.08802586385929487j` | 0 |

The counterrotating and 211 endpoint checks have estimated frequency
errors about `2e-17`; raw residual size varies with the recurrence's
conditioning. Seeded strict endpoint timings on the local Mac were about
1.0 seconds and 0.5 seconds respectively after tuning the initial depth.
These timings do not include the entire Schwarzschild-to-endpoint path.

## Independent review and limits

[The independent solver review](pass-7-kerr-solver-review.md) checks the
original radial differential equation directly, repeats the Schwarzschild
homotopy from ell=16 with spacing 0.05, and independently evaluates the
angular eigenproblem with 65-decimal mpmath linear algebra. The latter
agrees with the extended Rayleigh quotient within `5.47e-30` at the three
review points. A separate 322 continuation predicts in a different spin
coordinate and agrees on its entire reverse path within `1.33e-14`.

The API currently accepts ell=2..12, n=0..2, |m|<=ell, finite |a|<1 and
|a*omega|<=2. It may report failure where its finite depth or branch guard
cannot certify a root. Generation targets the explicitly validated low
modes through |a|=0.99999. The scalar solve accuracy does not itself certify
interpolation error, a complete QNM spectrum, all branches at arbitrarily
high spin, or a final waveform. Those require the generator's separate
holdout, endpoint, reverse-path and runtime checks.

Regression tests in `tests/test_kerr_leaver_solver.py` use only equations,
symmetries, continuation and numerical refinement. All **22 tests passed
in 17.26 seconds** with the generation environment:

```sh
/Users/xangma/miniconda3/bin/python -m pytest -q tests/test_kerr_leaver_solver.py
```

The two endpoint arithmetic tests skip explicitly when gmpy2 is unavailable;
the remaining checks exercise the generic NumPy/SciPy path. No reference
QNM frequency is embedded in the tests. The usual Torch test environment,
`/Users/xangma/miniconda3/envs/pycbcorig313/bin/python`, passed the remaining
**20 tests, with the two expected skips, in 12.15 seconds**.

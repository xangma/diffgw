# Standalone (2,2) nonquasicircular calibration, pass 3

`src/torchwave/approximants/eob/nqc.py` implements the published SEOBNRv4
nonquasicircular (NQC) factor, its five local matching conditions, the printed
NR target fits, and the printed matching-time offset. It provides trajectory
postprocessing mathematics. It does not select a peak, evolve a trajectory,
attach merger/ringdown, or change the waveform generator.

## Primary source ledger

The implementer freshly retrieved the primary PDF on 2026-09-12 and inspected
its equations directly. No implementation source, source port, inherited
coefficient table, prior calibration report, parity data, binary oracle, or
residual fitting was used. Only these three new NQC files were edited.

| Primary source | Equations and locations | Retrieved PDF SHA-256 |
| --- | --- | --- |
| [Bohé et al., arXiv:1611.03703v3](https://arxiv.org/pdf/1611.03703v3) | NQC basis (2.7), spin combination (2.8), and matching-time definition in Sec. II B, printed pp. 3–4; nonspinning offset (4.9), p. 9; spin-dependent offset (4.15), p. 10; peak amplitude (A1)–(A2), curvature (A3)–(A5), frequency (A6)–(A9), frequency slope (A10)–(A12), pp. 18–19 | `db9dc054368f131bdc152b319e54441c665547668221891723b309b005ac23ba` |

This paper supplies every physical basis and target used here. Additional
primary-paper prescriptions are not required to define these standalone
equations. The sampling/interpolation method below is an explicit numerical
choice, not a prescription attributed to the paper.

## Conventions and five conditions

Use `tau=t/M`, `r=R/M`, `pr_star=p_r*/mu`, and
`orbital_frequency=M*Omega`. Amplitude is the distance-rescaled dimensionless
mode amplitude `(D_L/M)*abs(h22)`. All supplied and returned derivatives are
with respect to `tau`. GW frequency is angular frequency, with positive
convention `omega22=-d(arg(h22))/d tau`, so `h22=A exp(-i phi22)`.

Equation (2.7) gives

```text
x = pr_star / (r * orbital_frequency)
q = x^2 (1, 1/r, 1/r^(3/2))
p = x (1, pr_star^2)
N22 = (1 + a.q) exp(+i b.p)
```

Thus the corrected amplitude is `A(1+a.q)` on the positive-amplitude branch,
and its positive frequency is `omega_base-b.p'`. The minus sign follows by
differentiating the phase of the complex corrected mode; it is not a choice
of coefficient sign. For matching targets `(A_t, A'_t, A''_t, omega_t,
omega'_t)`, the two linear systems are

```text
M_A[k,i] = d^k(A q_i)/d tau^k,        k=0,1,2; i=1,2,3
rhs_A[k] = d^k A_t/d tau^k - d^k A/d tau^k

M_phase[k,j] = d^k p_j/d tau^k,       k=1,2; j=1,2
rhs_phase[k] = d^(k-1) omega_base/d tau^(k-1)
               - d^(k-1) omega_t/d tau^(k-1)
```

The published peak targets set `A'_t=0`; Appendix A supplies the other four
values. The generic solver also accepts a nonzero first derivative for a
caller-specified local matching condition. It requires positive base and
target amplitudes and frequencies. The factor evaluator preserves the signed
amplitude multiplier printed in (2.7); it does not clip negative multipliers
that may occur away from the calibrated neighborhood.

## Printed fits and matching time

`seobnrv4_nqc_parameters(m1,m2,chi1,chi2)` computes

```text
nu = m1*m2/(m1+m2)^2
chi = (m1^2*chi1 + m2^2*chi2)/(m1^2+m2^2)
```

The second expression is algebraically Eq. (2.8). It avoids choosing a mass
ordering or differentiating a square root of `1-4 nu`, and is smooth for equal
masses. All fit decimals are transcribed at their printed precision.

For (A2), (A5), and (A12), the coefficient linear in `nu` is printed separately.
The other two quadratic coefficients are uniquely determined by imposing the
test-particle endpoint at `nu=0.001` and the equal-mass endpoint at `nu=0.25`.
The code solves these two constraints algebraically. For amplitude and
curvature it interpolates the normalized quantity `f/nu`, then multiplies by
`nu`; frequency slope is interpolated directly. This yields finite expressions
at `nu=0` without dividing by it.

**Different frequency endpoint:** the literal formula (A9) interpolates the
logarithm's two coefficients linearly between `nu=0` and `nu=0.25`. It therefore
recovers (A6) at zero, although the Appendix's introductory discussion describes
Teukolsky input at `nu=0.001`. The implementation follows the unambiguous printed
(A8)–(A9) for frequency and the explicitly specified `0.001` constraints for
the other three fits. It does not silently force a common endpoint.

`seobnrv4_nqc_time_offset` implements (4.9) plus (4.15), returning
`Delta t22_peak/M` with its printed sign. Sec. II B defines

```text
tau_match = tau_Omega_peak + seobnrv4_nqc_time_offset(nu, chi)
```

This requires a resolved orbital-frequency peak. The primary prescription
does not identify a trajectory's final retained sample with that peak.
Termination at a fixed radius or number of steps is insufficient by itself.
Sec. II B also applies NQC after the dynamics calculation, which is consistent
with keeping this implementation as standalone postprocessing.

The fit domain checks enforce physical parameter bounds `0<=nu<=1/4` and
`|chi|<=1`; they do not assert NR validation over that whole range. At `nu=0`,
the amplitude targets vanish, and the positive-amplitude calibration solver
appropriately cannot calibrate a zero-amplitude mode.

## Sample calibration and conditioning

`calibrate_nqc_22` accepts local samples of time, radius, tortoise radial
momentum, orbital frequency, uncorrected mode amplitude, and uncorrected
positive GW frequency, plus a matching time and targets. The final axis is
the sample axis. Leading batch axes, scalar/shared trajectories, matching
times, and target batches broadcast. Inputs must share a float32/float64 dtype
and device; float64 is preferable for local derivative matching.

The implementation centers time at the matching point and scales it by the
stencil span. It fits a polynomial (default degree four) to `A`, the three
products `A*q_i`, the two phase basis functions, and the base GW frequency.
Polynomial coefficients provide values and first/second derivatives at the
matching point, with the time scaling explicitly undone. Matching derivatives
of the products directly avoids replacing the product rule with independently
interpolated factors. The sample polynomial uses an ordinary full-rank
least-squares solve, retaining gradients through samples and sample times.

Both matching matrices are equilibrated by row and column norms before their
condition numbers are checked. They are then solved without regularization,
coefficient clipping, or pseudoinverse fallback. Returned diagnostics contain
the scaled polynomial's condition number, the equilibrated matching condition
numbers, and residuals in the original matching equations. Rank-deficient,
nonfinite or overly ill-conditioned systems raise `ValueError`. The default
maximum condition number is `1e10`, additionally bounded by the dtype's
numerical rank limit. Duplicate times, an exterior matching time, insufficient
samples, and invalid physical basis inputs are rejected. A circular stencil
cannot identify the five coefficients and is rejected by the matching solve.

Differentiability applies on successful, well-conditioned branches. Rejection
tests are discrete. Small matching residuals certify the fitted polynomial's
equations, not the accuracy of derivatives inferred from a poorly resolved
trajectory. The caller must establish stencil convergence.

Example structure, with an already resolved orbital-frequency peak:

```python
nu, chi = seobnrv4_nqc_parameters(m1, m2, chi1, chi2)
targets = seobnrv4_nqc_targets(nu, chi)
tau_match = tau_Omega_peak + seobnrv4_nqc_time_offset(nu, chi)
result = calibrate_nqc_22(
    tau_local, r_local, pr_star_local, Omega_local,
    A22_base_local, omega22_base_local, tau_match, targets,
)
N22 = nqc_22_factor(r, pr_star, Omega, result.coefficients, sample_dims=1)
h22_corrected = h22_base * N22
```

`nqc_22_factor` uses explicit `sample_dims=1` for trajectories, including a
shared one-dimensional trajectory evaluated with multiple coefficient sets.
Its default `sample_dims=0` evaluates individual states with ordinary batch
broadcasting. This avoids ambiguous guesses about whether an axis is time or
parameter batch.

## Validation and remaining inputs

The thirteen collected tests pass:

```text
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /Users/xangma/miniconda3/envs/pycbcorig313/bin/python -m pytest -q \
  tests/test_clean_room_eob_nqc.py
13 passed
```

They check printed endpoint values, component exchange, common mass scaling,
the equal-mass spin combination, the signed time offset, the circular factor
limit, and complex conjugation under radial-momentum reversal. Independently
constructed analytic complex signals recover known coefficients and satisfy
all five matching conditions when differentiated directly. A separate
quadratic amplitude peak tests zero slope and prescribed negative curvature.
Further checks cover batch/scalar agreement, cancellation of common amplitude
normalization, time translation, local-stencil refinement, finite-difference
autograd checks through trajectory inputs and fits, and failure diagnostics.

No physical NQC fit coefficients are missing from this standalone component.
To use it in a calibrated waveform, the caller still needs a consistent
uncorrected `(2,2)` mode and its positive frequency, a sufficiently extended
trajectory resolving the orbital-frequency maximum, a resolved local stencil
around the shifted matching time, and a compatible merger/ringdown attachment.
Those inputs and their model-level validation are outside this change. No full
SEOBNRv4 waveform or implementation parity is claimed.

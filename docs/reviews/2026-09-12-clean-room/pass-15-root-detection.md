# Pass 15: polynomial interval root detection

The XHM interval guard now lives in `torchwave.approximants.polynomial_interval`.
`polynomial_has_nonpositive(coefficients, lower, upper)` accepts ascending
powers through degree five, broadcasts batch dimensions, and returns a native
Torch boolean tensor.  It checks both closed interval endpoints and every
real stationary point of the polynomial.

The implementation recursively isolates derivative roots.  Derivative extrema
partition the interval into monotone pieces; each piece is refined with a
fixed number of native Torch bisection steps.  Residual checks at derivative
extrema retain repeated and tangent roots.  This covers lower-degree and
constant polynomials through the same padded representation and avoids a
fixed sampling grid, including the former eleven-node blind spot.

Decisions run under `torch.no_grad()` and are detached, so the discrete
selection cannot enter downstream gradients.  CPU, CUDA, and MPS tensors use
the same arithmetic without NumPy or host-side root calls.  Nonfinite
coefficients or bounds return `True` conservatively.  The root and residual
checks use machine-precision-scaled tolerances after affine normalization to
`[-1, 1]` (rather than a relative five-percent floor), so uniform coefficient
rescaling does not create a unit-sized threshold.  The result is therefore a
bounded numerical guard, not an exact algebraic certificate for arbitrary
floating-point inputs.

Focused validation is in `tests/test_pass15_polynomial_interval.py`, covering
narrow dips, tangent roots, positive near-zero minima, mixed batches,
degenerate degrees, scaling, detached decisions, nonfinite inputs, and
available accelerators.

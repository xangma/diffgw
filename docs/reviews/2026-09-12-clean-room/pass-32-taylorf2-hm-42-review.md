# Pass 32: independent TaylorF2_HM (4,2) review

This review checks the newly added `(4,2)` leading term using only Mezzasoma
and Yunes, *Theory-agnostic framework for inspiral tests of general
relativity with higher-harmonic gravitational waves*, Phys. Rev. D 106,
024026 (2022), arXiv:2203.15934, the current TorchWave source,
and an independent Wigner-d derivation.  No LAL, PyCBC, TEOBResumS, or other
implementation source was inspected or used.

## Amplitude and SPA normalization

The primary source gives `H_22 = 1` in Eq. (2.21b) and

\[
 H_{42}=\frac{\sqrt{5}}{63}(1-3\nu)x
\]

in Eq. (2.21g), with `x` evaluated at the harmonic's stationary frequency.
Its Eq. (2.26) gives

\[
 x_m=\left(\frac{2\pi M f}{m}\right)^{2/3}.
\]

For `m=2`, this is exactly `v22**2 = (pi*M*f)**(2/3)`.  Since `(4,2)` and
`(2,2)` have the same harmonic index, their SPA prefactors and frequency
mapping are identical.  Therefore the required mode ratio is exactly

\[
 \frac{\tilde h_{42}}{\tilde h_{22}}
 =\frac{\sqrt{5}}{63}(1-3\nu)v_{22}^2.
\]

The implementation uses this coefficient and multiplies it by the existing
`sqrt(16*pi/5) * A0` carrier normalization.  Independently multiplying that
normalization by the `(2,2)` spin-weighted projection reproduces the existing
TaylorF2 Newtonian plus/cross amplitude, so the `(4,2)` normalization is
consistent with the carrier rather than introducing a second amplitude
convention.

The source's `H_42` is real and has no extra `i` factor.  The implementation
uses the same phase as the `m=2` carrier: its phase connection is zero at
`m=2`, and the `f_ref` mapping is unchanged.  Thus the complex ratio is real
and positive, as required by Eq. (2.21g), rather than merely matching in
magnitude.

## Angular projection

Using the standard spin-weight `-2` Wigner-d definition independently gives

\[
{}_{-2}Y_{42}=\frac{3}{8\sqrt{\pi}}(1+c)^2
 (7c^2-7c+1)e^{2i\phi},
\]

and the negative-m partner used by the implementation,

\[
{}_{-2}Y_{4,-2}=\frac{3}{8\sqrt{\pi}}(1-c)^2
 (7c^2+7c+1)e^{-2i\phi},
\]

where `c=cos(iota)`.  These expressions have the same normalization and
equatorial-symmetry convention as the existing `(2,2)` projection.  The
`d_plus` and `d_cross` combinations therefore have the correct negative-m
contribution and preserve the carrier convention.

## Boundary behavior and verification

The coefficient does not vanish for equal masses: at `nu=1/4` it is
`sqrt(5)/252`, which is consistent with the even-m `(4,2)` harmonic.  The
implementation remains finite for equal masses, supports scalar and batched
inputs, and retains autograd paths through masses, inclination, and azimuth.

I strengthened the inventory test to check the signed complex `(4,2)/(2,2)`
ratio, including its zero imaginary part, instead of checking only the
magnitude.  The focused verification completed successfully:

```text
15 passed: tests/test_taylorf2_hm.py tests/test_taylorf2_hm_mode_inventory.py
py_compile: passed
git diff --check: clean
```

Conclusion: the `(4,2)` coefficient, SPA normalization, phase mapping,
negative-m projection, equal-mass behavior, batching, and gradients are
source-consistent.  No implementation defect was found.

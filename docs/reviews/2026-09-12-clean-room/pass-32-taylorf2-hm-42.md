# Pass 32: clean-room TaylorF2_HM (4,2) mode

## Result

The `(4,2)` mode is source-complete at the leading nonspinning PN order used by
the existing TaylorF2_HM higher modes, so the runtime implementation is
enabled.  The exact amplitude ratio, phase mapping, and angular projection are
all fixed by published formulas; no waveform implementation source was read.

## Primary derivation

Mezzasoma & Yunes, *Phys. Rev. D* **106**, 024026 (2022),
[arXiv:2203.08939](https://arxiv.org/abs/2203.08939), gives the Fourier-domain
mode normalization in Eq. (2.20) and the leading mode coefficient

\[
H_{42}=\frac{\sqrt{5}}{63}(1-3\nu)x
\]

in Eq. (2.21g).  Their frequency variable is
\(x_m=(2\pi M f/m)^{2/3}\) (Eq. (2.26)).  For `m=2`, this is exactly
`v22**2` in TorchWave.  Comparing Eq. (2.20) with the existing
`sqrt(16*pi/5) * A0` quadrupole normalization therefore fixes

```text
A_42 / (sqrt(16*pi/5) A0)
    = sqrt(5)/63 * (1 - 3*eta) * v22**2.
```

Because `m=2`, the SPA phase uses the same frequency and phase as the
quadrupole (`delta_phi = 0`); the published coefficient is real at this order.
No additional phase or fitted constant is introduced.

For the projection, the standard normalized spin-weighted spherical harmonics
with spin weight `s=-2` give

```text
Y_4,2  = 3/(8*sqrt(pi)) (1+c)^2 (7*c^2 - 7*c + 1) exp(2 i phi0)
Y_4,-2 = 3/(8*sqrt(pi)) (1-c)^2 (7*c^2 + 7*c + 1) exp(-2 i phi0),
```

where `c = cos(inclination)`.  Substitution into the existing positive-
frequency projection convention fixes `d_plus` and `d_cross`; the resulting
polynomials are normalized by the standard SWSH inner product.

## Changes and validation

- `src/torchwave/approximants/taylorf2_hm.py`: added the `(4,2)` SWSH branch,
  published leading-PN amplitude, source citation, and supported-mode entry.
- `tests/test_taylorf2_hm.py`: replaced the stale unsupported-mode assertion
  with a finite-output acceptance check.
- `tests/test_taylorf2_hm_mode_inventory.py`: added exact amplitude-ratio and
  SWSH projection invariants plus an autograd gradient check.

Focused validation passed:

```text
15 passed in 0.63s
git diff --check: clean
py_compile: clean
```

This implements the source-supported leading `(4,2)` term.  The cited result
does not by itself specify a higher-PN or spin-dependent `(4,2)` amplitude;
adding such terms would require additional source-complete formulas and is not
part of this patch.

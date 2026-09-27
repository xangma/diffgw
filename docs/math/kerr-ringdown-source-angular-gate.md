# Kerr 220 source-spectral angular composition

Specification ID: `kerr-220-source-spectral-angular-composition/v1`.

## Scope and status

This is a selected-input, direct-call composition gate for Torch and JAX
`kerr220_td/fd` with `angular_convention="source_spectral"`. It checks the
physical-energy (2,2,0) carrier with an equation-defined angular factor.
The default `"public_compat"` route and its [public-clean whole-waveform
blocker](kerr-ringdown-public-clean-gate.md) are unchanged. This gate makes
no binary-parity, dispatcher, spin-gradient, or full-domain claim.

## Inputs, conventions, and options

The frozen `(spin, mass/Msun, distance/Mpc, energy fraction, inclination,
phic, tc/s)` cases are `(0.35,53,230,0.017,1.1,0.41,0.002)` and
`(0.7,92,430,0.005,0.55,-0.27,-0.001)`. The TD grid is
`tc+(-0.001,0,0.002,0.008)` seconds; the signed FD grid is
`(-130,0,35,110,270)` Hz. CPU FP64 is required. The angular solution has
spin weight `-2`, `exp(+2i phic)`, unit-sphere norm, and a phase rotation
making its value at `(theta,phi)=(pi/2,0)` positive real. The other factor
uses `pi-inclination`. The TD step has `H(0)=1`, and the FD kernel is
`exp(-2 pi i f t)`.

## Mathematics and components

The independent frequency is the continued-fraction 220 root specified in
[scalar QNM qualification](kerr-qnm-scalar-qualification.md). With
`c=spin*(M omega)` for complex `M omega`, the angular function solves
[Cook and Zalutskiy, arXiv:1410.7698v2](https://arxiv.org/html/1410.7698v2)
Eq. (4) by the convergent spherical expansion of Eqs. (49)–(57). Its
complex phase is arbitrary; the equatorial-positive rule above fixes it
without using a public binary. The same paper's Eq. (15b) gives a
convergent Frobenius expansion, but it does not prescribe the absolute
`0.01` first-small-term stop in the inferred compatibility path. The
[conditional carrier](kerr-ringdown.md) specifies the single-mode energy
amplitude, causal polarization pair, and analytic two-pole FD transform.

## Implementation mapping

The source-spectral option evaluates the existing native spherical
eigensolver in `torchwave.approximants.kerr_angular` or
`jaxwave.approximants.kerr_ringdown`, then applies the equatorial phase
rotation before calling the existing carrier. The independent reference
uses `diffgw_reference.kerr_qnm`, `diffgw_reference.kerr_angular`, and
`diffgw_reference.kerr_ringdown`; it does not take native QNM or angular
values as inputs. The `angular_convention` option is explicit and leaves
the inferred public-compatible default unchanged.

## Evidence and limitations

For each case and backend, the predeclared gates are `3e-10` maximum
absolute error for either complex angular factor and `3e-9` relative L2
error across both polarizations on each TD and FD grid. The scalar reference
uses 40-digit mpmath, radial depth 400, Galerkin `ell_max=10`, and
Gauss–Legendre order 32. Its angular convergence check compares
`ell_max=10` with `12` and must be below `1e-11`. Native `ell_max=18` is
fixed. The grid checks a composition at selected spins, not a continuous
spin domain or every angle. JAX's spectral eigenvector callback has no
derivative through spin; no intrinsic-gradient claim is made. A separate
fresh pinned binary holdout and applicable gradients would be needed to
qualify a public waveform route.

## Reproduce the gate

On `len` with Torch/JAX CPU FP64, SymPy, mpmath and pytest:

```sh
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTHONPATH=src:tests python -m pytest -q -ra \
  --reference-device=cpu tests/test_kerr_source_angular_math_reference.py
```

This focused selection omits `--require-math-reference`, whose full-suite
contract requires all reference files.

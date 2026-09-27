# NRTidalv2 Pv2 positive-22 projection boundary

## Scope and status

Specification **`nrtidalv2-pv2-positive-22-projection/v1`** covers the
positive-frequency projection of a *supplied* aligned-frame `(2,2)` mode into
complex plus/cross polarizations. It is an executable, conditional component
of the Pv2 construction, not an `IMRPhenomPv2_NRTidalv2` waveform. The
[prior aligned-frame gate](nrtidalv2-pv2-aligned-additive-terms.md) covers
the signed finite-size term. [Dietrich et al. (2019), Sec. III.4](https://arxiv.org/html/1905.06011)
places that term into the aligned carrier *before* the precession rotation.

## Inputs, conventions, and options

The input is a complex positive-frequency `h^L_{2,-2}(f)`, with supplied
Euler angles `alpha`, `beta`, `gamma`, line-of-sight polar angle `theta_JN`,
and polarization-basis angle `zeta`, in radians. All are evaluated pointwise
on the same grid. The spin-weighted harmonics have azimuth zero. The output
is `(h_plus,h_cross)` in the inverse-rotated polarization basis. This gate
uses arbitrary normalized mode values and nondegenerate angles. It does not
infer the angle dynamics or the dimensional conversion from Eq. (24)'s
absolute tidal amplitude to `h^L_{2,-2}`.

## Mathematics and components

For the supplied mode `H`, let
`Y_m = sqrt(5/(4 pi)) d^2_{m,2}(theta_JN)` and
`A_{m,m'} = exp(-i m alpha) d^2_{m,m'}(beta) Y_m`.
The positive-frequency polarization pair is

\[
 p=\tfrac12 H e^{2i\gamma}\sum_{m=-2}^{2}
 (A_{m,-2}+A^*_{m,2}),\qquad
 c=\tfrac i2 H e^{2i\gamma}\sum_{m=-2}^{2}
 (A_{m,-2}-A^*_{m,2}).
\]

The returned pair is
`(cos(2 zeta) p + sin(2 zeta) c, cos(2 zeta) c - sin(2 zeta) p)`.
This is [Pratten et al. (2020), Eqs. (11)–(13), (53), and the inverse of
Eqs. (93)–(94)](https://arxiv.org/html/2004.06503). The reference evaluates
Eq. (53)'s factorial sum at 80 decimal digits; the native implementations
evaluate their own tensor expressions. Because this projection is linear in
`H`, an aligned-frame correction `D` must obey `P(H+D)=P(H)+P(D)`.

| Component ID/version | Source | Variant | Status | Scalar reference | Torch | JAX | Check |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `nrtidalv2-pv2-positive-22-projection/v1` | Pratten (2020), Eqs. (11)–(13), (53), (93)–(94) | Supplied positive-frequency `(2,-2)` mode | Conditional executable | `nrtidalv2_pv2_projection.project_positive_22` | `project_twisted_positive_modes` | `phenompv2_waveform._project` | `test_nrtidalv2_pv2_projection_math_reference.py` |

## Implementation mapping

`src/diffgw_reference/nrtidalv2_pv2_projection.py` has no native imports.
The native Torch and JAX projectors are used by their existing Pv2 BBH
waveforms. Neither frontend currently provides the tidal Pv2 carrier
assembly or its public BNS dispatch. This test calls the projector boundary
directly, with no LAL source or calibration.

## Evidence and limitations

The frozen cases are three complex `H` values, three independent complex
perturbations `D`, and nonzero frequency-dependent `alpha`, `beta`, `gamma`,
`theta_JN`, and `zeta`, as recorded in the maintained test. Torch and JAX
must each match both complex polarizations from the scalar reference for
`H`, `D`, and `H+D` to `1e-12` after normalization by maximum input-mode
magnitude. Each backend must also satisfy the pre-rotation linearity identity
to `1e-12`. CPU FP64 is required; there is no phase/time alignment, fitting,
or output rescaling. The [execution receipt](../validation/nrtidalv2_pv2_positive_22_projection_20260927.md)
records the run.

This operator gate does not establish the source-frame angle map, the
IMRPhenomD-to-Pv2 mode normalization, Eq. (24)'s absolute-amplitude injection,
the native tidal Pv2 assembly, a pinned-binary match, or physical validation.
The historical paper-D and `binary_compatible` variants are unchanged.

## Reproduce the checks

With mpmath, Torch, JAX, and pytest available, run CPU FP64:

```sh
PYTHONPATH=src:tests JAX_ENABLE_X64=1 python -m pytest -q \
  tests/test_nrtidalv2_pv2_projection_math_reference.py
```

Changing the sign, mode, basis, or angle convention requires a new spec
version and a new independent reference comparison.

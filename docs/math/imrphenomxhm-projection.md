# IMRPhenomXHM positive-frequency mode projection

## Scope and status

This is a paper-equation gate for the **complete five-mode projection** of
aligned-spin XHM's positive-frequency carriers. It specifies the angular
coefficients and the common time/orbital phase factor, treating each intrinsic
complex carrier as an input. It does not qualify those carriers, their
calibration fits, the precessing XPHM rotation, or a full LAL waveform.
Component ID: `xhm.positive-frequency.mode-pair-projection.v1`.

## Inputs, conventions, and options

[García-Quirós et al., arXiv:2001.10914v2](https://arxiv.org/html/2001.10914v2),
Sec. II, Eqs. (2), (4)–(9), defines the Fourier transform with
`exp(-2πift)`, the equatorial-reflection symmetry
`h_{ℓm}(t)=(-1)^ℓ h*_{ℓ,-m}(t)`, and the **positive-frequency** carrier
`H_{ℓm}(f)=\tilde h_{ℓ,-m}(f)` for `m>0`.

The native `H_{ℓm}` also carries `exp[-i(2πft_c-mφ_c)]`; the observer
azimuth enters the harmonics as `exp(±imφ_0)`. The supported mode set is
`(2,2), (2,1), (3,3), (3,2), (4,4)`. The scalar reference constructs the
spin-`−2` harmonics from Wigner `d` matrices, normalized by
`{}_{-2}Y_{22}=sqrt(5/(64π))(1+cos ι)^2 exp(2iφ_0)`.

## Mathematics and components

The paper's two polarization contributions are

\[
\tilde h_{+,\ell m} = \frac{1}{2}
  \left({}_{-2}Y_{\ell,-m} + (-1)^\ell {}_{-2}Y^*_{\ell m}\right) H_{\ell m},
\qquad
\tilde h_{\times,\ell m} = \frac{i}{2}
  \left({}_{-2}Y_{\ell,-m} - (-1)^\ell {}_{-2}Y^*_{\ell m}\right) H_{\ell m}.
\]

## Implementation mapping

[`imrphenomxhm_projection.py`](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/imrphenomxhm_projection.py)
implements the two source coefficients without importing either frontend.
The shared HM spin-weighted-harmonic helper instead returns coefficients for
the **positive-`m`** carrier. XHM evaluates that helper at `−φ_0`; the
source's negative-`m` carrier requires the additional factor `(-1)^ℓ` in
**both** plus and cross. Thus the existing even-ℓ coefficients stay unchanged,
while the `(3,3)` and `(3,2)` coefficients change sign in
[Torch](https://github.com/xangma/diffgw/blob/main/src/torchwave/approximants/imrphenom_xhm.py)
and [JAX](https://github.com/xangma/diffgw/blob/main/src/jaxwave/approximants/imrphenom_xhm.py).
The shared HM helpers are unchanged, so HM projections are unaffected.
XPHM's scalar aligned-spin fast path delegates to XHM and inherits this
correction; its precessing twist-up requests unprojected XHM carriers, which
this change leaves untouched.

## Evidence and limitations

[`test_xhm_projection_math_reference.py`](https://github.com/xangma/diffgw/blob/main/tests/test_xhm_projection_math_reference.py)
checks each complex plus/cross contribution and their sum at three viewing
angles and three positive frequencies using each frontend's unprojected
carrier as the independent projection input. It also checks the source's
quadrupole normalization and the common time and mode-dependent orbital phase
factor in Torch and JAX FP64. The test does not compare carrier amplitudes or
phases to LAL, and it cannot certify the XHM full waveform or XPHM rotation.
The frozen [XHM/XPHM release gates](../validation/xhm_xphm_release_qualification_20260925.json)
retain their strict tolerances and failing status pending separate work.

## Reproduce the checks

With SymPy, Torch, JAX, and pytest installed, run
`PYTHONPATH=src JAX_ENABLE_X64=1 python -m pytest -q tests/test_xhm_projection_math_reference.py`.

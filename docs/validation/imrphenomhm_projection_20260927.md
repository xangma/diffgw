# IMRPhenomHM six-mode projection gate — 2026-09-27

The bounded `hm.projection.london-eq1.v1` gate checks the projection of six
public positive-frequency HM mode carriers into complex plus/cross. It does
not validate the intrinsic carriers or qualify the complete waveform.

Source derivation: [London et al., Eq. (1)](https://arxiv.org/pdf/1708.00404v4)
sets `h=h+−ih×=Σh_lm {}₋₂Y_lm`; [García-Quirós et al., Sec. II, Eqs.
(2), (4)–(9)](https://arxiv.org/html/2001.10914v2) fix the
`exp(-2πift)` Fourier kernel and positive-frequency negative-`m` mode-pair
formula. SymPy Wigner matrices generate the harmonics; mpmath evaluates the
coefficients at 50 digits. The native public carrier adapter and azimuth sign
are stated in the [component specification](../math/imrphenomhm-projection.md).

The frozen specification preceded the reference/test implementation in commit
`b8a8e5f`, based on integration commit `7ecf8ab`. The test reads public
`return_unprojected_modes` and `return_modes` values from both frontends. At
three frequencies and four angles (face-on, edge-on, and two generic), it
checks each of six complex plus/cross contributions and their sum. It also
checks the pole selection and odd-ℓ sign, azimuth/orbital-phase/time
covariance, body-exchange `(-1)^m` parity, and symmetric odd-mode zero.

Host `len`; working copy
`/home/xangma/tmp/diffgw-hm-projection-gate-20260927`; Python
`/home/xangma/.cache/diffgw-taylorf2-hm-venv/bin/python`; Torch `2.14.0+cpu`,
JAX `0.11.2`, SymPy `1.14.0`, mpmath `1.3.0`, pytest `9.1.1`.

```sh
PYTHONPATH=src JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu \
  /home/xangma/.cache/diffgw-taylorf2-hm-venv/bin/python -m pytest -q \
  tests/test_imrphenomhm_projection_math_reference.py \
  --require-math-reference --reference-device=cpu -ra
```

The initial focused run passed `13/13` in 44.77 s. The 50-digit mpmath rerun
passed `13/13` in 43.49 s. The strict gate passed `13/13`, with zero skips,
expected failures, or deselections, in 40.89 s; its output is
`projection-pytest-strict.log` in the working copy. No red projection defect
was observed, so no native implementation was
changed. These results do not establish mode calibration, QNM accuracy,
negative-frequency reconstruction, LAL mode agreement, or physical accuracy.

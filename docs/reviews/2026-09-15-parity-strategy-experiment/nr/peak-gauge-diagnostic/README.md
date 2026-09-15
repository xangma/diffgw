# SXS:BBH:2488 peak-gauge diagnostic

This is a read-only diagnostic of the fresh quality pair `Lev5/N3` and
`Lev4/N3`. It does not read model coefficients or residual arrays, score a
model, refit anything, or modify the frozen runner/basis. The exact replay is
in [diagnose_peak_gauge.py](diagnose_peak_gauge.py), and its machine-readable
output is [report.json](report.json).

## Method

The original branch copies the frozen strategy: the time sample maximizing
`|h22|` defines the epoch, the h22 phase defines one shared orbital rotation
`exp(-i m phi)`, and the h33 value at that sample resolves the remaining pi
branch. The continuous branch fits one concave quadratic to the three
`|h22|` samples surrounding that same maximum. It linearly interpolates the
complex h22 and h33 values at the vertex, uses the resulting shared rotation,
and applies the same frozen `transform_complex` FFT and fresh holdout 512-bin
frequency grid. This is a single local gauge test, not a new alignment model.

Run from the repository root with:

```sh
PYTHONPATH=. python docs/reviews/2026-09-15-parity-strategy-experiment/nr/peak-gauge-diagnostic/diagnose_peak_gauge.py
```

## Result

The sampled-to-quadratic peak shifts are `-0.038976 M` (Lev5/N3) and
`+0.034447 M` (Lev4/N3), with left/right sample spacings
`(0.100003, 0.100003) M` and `(0.099979, 0.099979) M`. Thus the
Lev5-minus-Lev4 peak spacing changes from `51.166590 M` to `51.093168 M`, a
`-0.073423 M` correction. The orbital-angle changes are `+0.005960 rad` and
`-0.005262 rad`, with the same h33 branch in both gauges.

Using Lev5/N3 as reference and Lev4/N3 as candidate, the raw complex L2
changes are:

| gauge | (2,2) | (3,3) | mean |
|---|---:|---:|---:|
| sampled | 0.09812335 | 0.03584077 | 0.06698206 |
| quadratic | 0.07369870 | 0.01782249 | 0.04576060 |

The amplitude L2 changes are below `1e-8`, as expected for a gauge-only
change. The existing affine phase fit reports `(constant_phase,
time_shift, affine_phase_rms)` of `(0.090635, 0.251906 M, 0.007590)` for 22
and `(0.034920, 0.198543 M, 0.002037)` for 33 in the sampled branch. The
quadratic branch moves the fitted time shifts to `0.178483 M` and `0.125120 M`,
while affine residual RMS changes by less than `5.4e-10`.

The sampled branch reproduces the stored fresh-holdout Lev4/N3 control
metrics to a maximum absolute difference of `6.9e-15`.

Conclusion: sampled peak timing is a substantial contributor to the raw
complex sensitivity (about 25% lower 22 L2 and 50% lower 33 L2 in this one
resolution comparison), but it does not explain the post-affine phase
residual. This identifies a real gauge contribution without claiming that
peak interpolation alone resolves the full sensitivity.

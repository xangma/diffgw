# NSBH remnant printed-precision interval — 27 September 2026

The [six-section v1 specification](../math/nsbh-remnant-rounding-interval.md)
was frozen as commit `6e18fd4` **before** native Torch/JAX comparison. It
uses [Jiménez-Forteza et al. (2017)](https://arxiv.org/html/1611.00332v2),
Eqs. (7)–(28) and Tables 1–4, 7–10, for the BBH final mass/spin, and
[Zappa et al. (2019)](https://arxiv.org/html/1903.11622v2), Supplementary
Eqs. (4)–(6) and Table 1, for the NSBH correction. Each printed coefficient
is interpreted as a closed conventional round-to-nearest last-digit interval;
the independent scalar reference propagates these intervals by exact rational
arithmetic. This bounds possible missing digits under that declared printing
convention; it does not recover those digits or use LAL source/binary output.

The frozen cases `(q, chi_BH, Lambda) = (3,0,2500), (3,0.5,2500),
(5,0.5,1000), (5,0.75,100), (5,0.75,2500)` use `m_NS=1.4`, `chi_NS=0`,
and lie in Zappa's reported calibration ranges. For each case, source-only
interval arithmetic proved both tidal factors strictly below one, so native
extrapolation guards cannot affect this comparison. The upper bounds on the
mass/spin factors, and the resulting final mass/spin interval widths, are:

| Case | Factor upper bounds | Final mass width ($M_\odot$) | Final spin width |
| --- | --- | ---: | ---: |
| (3, 0, 2500) | 0.986386 / 0.969853 | 0.141473 | 0.016499 |
| (3, 0.5, 2500) | 0.983935 / 0.991051 | 0.155104 | 0.024938 |
| (5, 0.5, 1000) | 0.998851 / 0.996706 | 0.092735 | 0.012886 |
| (5, 0.75, 100) | 0.998414 / 0.999317 | 0.016482 | 0.006170 |
| (5, 0.75, 2500) | 0.980784 / 0.984092 | 0.184287 | 0.022871 |

On `len`, Linux x86-64 CPU FP64, in
`/home/xangma/tmp/diffgw-nsbh-remnant-rounding-20260927`, the focused command
`PYTHONPATH=src /home/xangma/.cache/diffgw-xp-pycbc-venv/bin/python -m pytest -q
tests/test_nsbh_remnant_rounding_math_reference.py` passed **5/5** in 4.64 s.
The neighboring `./run-neighbors.sh` command passed **42/42** in 23.59 s;
its `neighbor.log` and `neighbor.exit=0` are in the same scratch directory.
That suite covers remnant, London QNM, explicit phase, branch amplitude fits,
and joining frequencies. The environment was Python 3.12.3, Torch
2.14.0+cpu, JAX 0.11.2, pytest 9.1.1, with PyCBC available. Neither
backend skipped or xfailed. The native literal audit confirms every BBH/Zappa
fit coefficient rounds to its matching paper entry and the two dependent
BBH coefficients satisfy the printed Eq. (13)/(26) interval constraints.
The same frozen $2\times10^{-12}$ absolute FP64 endpoint allowance was used
for all output comparisons; no tolerance or case changed after evaluation.

The intervals are deliberately conservative, and the reported widths show
their practical resolution. Passing is implementation consistency with the
published fit at printed precision, not a tight match to an unpublished
coefficient vector. The Zappa extrapolation/fallback policy, remnant
calibration against NR data, coefficient provenance beyond printed digits,
derivatives, FP32, and remnant-driven whole-waveform composition remain open.

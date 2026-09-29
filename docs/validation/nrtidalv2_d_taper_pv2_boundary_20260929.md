# NRTidalv2 D taper and Pv2 dispatch follow-up — 29 September 2026

**Disposition:** No source-backed production change. The pinned
`binary_compatible` D amplitude remains a historical binary-compatibility
choice, while the published 2019 D composition remains phase-only. The
nonzero-tide D taper binary gate retains its frozen `5e-5` limit and strict
expected failures. Complete Pv2 BNS dispatch remains pending. This receipt
adds a reproducible black-box taper discriminator and a source boundary audit;
it does not promote either approximant's qualification status.

## Sources and competing explanations

[Dietrich et al. (2019), Table II and Eqs. (24)–(25)](https://arxiv.org/html/1905.06011v2)
assign the additive tidal amplitude to Pv2, not D, and taper the amplitude
from the merger frequency to `1.2 f_mrg`. The older
[Dietrich et al. (2018), Eqs. (10)–(11)](https://arxiv.org/pdf/1804.02235)
print the merger-frequency fit, saying that `kappa_eff^T` replaces
`kappa_2^T` in the LAL implementation. The independent SymPy/mpmath reference
in `src/diffgw_reference/nrtidalv2_amplitude.py` uses those printed
coefficients and the published Planck function. The earlier
[provenance audit](nrtidalv2_provenance_20260927.md) algebraically rejected
the coupling substitution as a complete explanation: it changes the
equal-mass case by exactly zero and the unequal-mass printed merger fit by
`-594.064 ppm`, compared with observed effective shifts near `-25 ppm`.
It also found no published normalization for the default D tidal amplitude.

This follow-up checked three further possibilities without reading LAL
implementation source or deriving constants from its binary:

| Hypothesis | Decisive observation | Status |
| --- | --- | --- |
| A fixed merger-frequency rescaling explains the binary taper | Effective shift is zero at zero tide, then varies from `-10.0140` to `-60.1647 ppm` across equal-mass tidal strengths | Rejected as a single fixed scale |
| The discrepancy is specific to LALSuite 7.26.15 | Isolated 7.25.1 and 7.26.15 binary wheels give identical printed diagnostic rows | Rejected for these two releases only |
| Unprinted precision in the published merger coefficients could explain it | Independent interval arithmetic around the last printed digit allows approximately `±190 ppm` at `kappa_eff^T=75` and `±253 ppm` at `225`, holding the zero-tide prefactor fixed | Numerically possible; no unique source-backed replacement coefficients |

For that interval check, each of the four nonconstant Eq. (11) coefficients
was independently moved by at most half a unit of its last printed decimal:
`n1=0.03354±0.000005`, `n2=0.00004315±0.000000005`,
`d1=0.07542±0.000005`, `d2=0.0002236±0.00000005`. Since the numerator
increases and denominator decreases monotonically with those choices for
positive coupling, the endpoint calculation bounds all combinations within
those decimal intervals. It establishes only that rounding can contain the
observed shift, not that the binary uses a particular unprinted value.

## Dense black-box result

On `len`, the checked-out `fcff4626` tree was copied to
`/home/xangma/tmp/diffgw-nrtidalv2-d-pv2-20260929`. The historical,
one-off diagnostic sampled 151 frequencies from `0.9` to `1.2` times each
*printed-paper* merger frequency. It queried only the LAL binary API for
`|h_plus(D_NRTidalv2)|/|h_plus(D)|`, divided by the native **untapered**
quotient, and compared the resulting effective window with the independently
specified Planck function. The diagnostic then inverted that window on 61
interior bins to express the mismatch as an equivalent boundary displacement.
This inversion is conditional on the native untapered quotient continuing to
approximate the binary quotient inside the taper; its median is not an
estimate of a production coefficient.

| `(m1,m2; Lambda1,Lambda2)` | Printed `f_mrg` Hz | Pre-taper maximum quotient error | Interior maximum window error | Median equivalent shift ppm |
| --- | ---: | ---: | ---: | ---: |
| `(1.35,1.35; 0,0)` | 4291.579158 | 0 | `2.22e-16` | 0 |
| `(1.35,1.35; 100,100)` | 2830.445551 | `1.03e-7` | `1.10e-4` | `-10.0140` |
| `(1.35,1.35; 200,200)` | 2401.755992 | `1.38e-7` | `1.68e-4` | `-15.2494` |
| `(1.35,1.35; 400,400)` | 2037.930727 | `1.79e-7` | `2.81e-4` | `-25.5376` |
| `(1.35,1.35; 800,800)` | 1732.535794 | `2.27e-7` | `4.90e-4` | `-44.4509` |
| `(1.35,1.35; 1200,1200)` | 1572.344865 | `2.54e-7` | `6.63e-4` | `-60.1647` |
| `(1.6,1.2; 200,800)` | 1709.953767 | `1.21e-7` | `2.77e-4` | `-25.1381` |

Rows agree at displayed precision for LALSuite **7.26.15**
(`lal 7.7.1`, `lalsimulation 6.2.1`) and isolated **7.25.1**
(`lal 7.6.1`, `lalsimulation 6.1.0`). Across the interior inversion bins,
the equivalent shifts are nearly constant within each case (for example,
`[-25.5536,-25.4512] ppm` at `Lambda=400`); a median-shifted Planck
function leaves a maximum window residual of `2.69e-7` there. This pattern
localizes the large mismatch to a merger-boundary-like difference. It does
not identify the binary's internal formula, full-precision coefficients, or
the default D amplitude's provenance.

The frozen focused binary comparison still gives maximum *amplitude quotient*
errors `0.000153757344` for equal masses and `0.000189164604` for unequal
masses inside the nonzero-tide taper, above the unchanged `0.00005` limit.
The zero-tide taper passes. These numbers are a separate, predeclared grid
from the dense effective-window diagnostic and should not be interchanged.

## Pv2 public dispatch boundary

Dietrich et al. (2019), Sec. III.4, directs finite-size corrections to the
aligned point-particle carrier before Pv2 rotation. The conditional carrier
and supplied-angle positive-22 projection have scalar references and
Torch/JAX checks in the existing
[public assembly boundary](../math/nrtidalv2-pv2-public-assembly-boundary.md).
The [public Pv2 technical note](https://dcc.ligo.org/public/0122/T1500602/004/PhenomPv2_technicalnotes.pdf)
specifies BBH frame and angle ingredients but does not select a BNS Fourier
carrier time/phase origin or merger-slope rule. Its NNLO spin input and
printed orbital-angular-momentum coefficient also differ from the current
native BBH compatibility drivers, as mapped in that boundary document.
Consequently those checked components do not justify a complete, unaligned
`IMRPhenomPv2_NRTidalv2` BNS dispatch. No angle, origin, or coefficient
was chosen from a binary residual.

## Historical run and maintained checks

The remote environment was
`/home/xangma/.cache/diffgw-taylorf2-hm-venv/bin/python` with
`JAX_ENABLE_X64=1`, `OMP_NUM_THREADS=1`, and `PYTHONPATH=src:tests` from
the scratch checkout. The dense scan's one-off replay script is not
maintained; the frozen amplitude comparison below remains a maintained
regression and does not replay the per-bin frequency inversion.

For isolated LALSuite 7.25.1, `PYTHONPATH=lalsuite-7.25.1:src:tests` was
used, with the binary wheel installed in the scratch target directory. The
full outputs are retained on `len` at
`/home/xangma/tmp/diffgw-nrtidalv2-d-pv2-20260929/taper_probe_7.26.15_clean.log`
and `taper_probe_7.25.1_clean.log`. The frozen comparison was reproduced with
`DIFFGW_REQUIRE_LAL_BINARY=1 PYTHONPATH=src:tests JAX_ENABLE_X64=1
OMP_NUM_THREADS=1 python -m pytest -q -s
tests/test_nrtidalv2_amplitude_lal_binary.py -k frontend_taper_amplitude`:
**2 passed, 6 strict xfailed, 24 deselected** on 7.26.15. The expected
failures remain failures. The full amplitude binary file yielded **14 passed,
18 strict xfailed**, recorded in `focused_binary_7.26.15.log`. The amplitude
math-reference, conditional Pv2 aligned-carrier, and supplied-angle
projection files yielded **31 passed**, recorded in `focused_math_pv2.log`.
No model, gate, CI, catalogue, or shared qualification-status file was
changed; nothing was deployed.

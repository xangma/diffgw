# Pass 17: XHM source-audit closure

Date: 2026-09-13. Pass 17 closes the current bounded XHM audit without a
runtime or test change. Full LAL parity remains unfinished.

## Outcome

The candidate numeric `(2,1)` amplitude veto was rejected. The paper defines
the threshold on geometric Fourier strain, but the allowed sources do not
establish the complete conversion from the native normalized fit quantities.
Parent validation also showed that removing the existing documented
four-value monotonicity convention fails the ordinary XHM smoke gate. The
runtime and pass-15 veto test were restored byte-for-byte to their frozen
pass-15 versions. See the [candidate disposition](pass-16-veto-implementation.md)
and [independent review](pass-16-veto-independent-review.md).

The phase audit archived 19 byte-identical official Mathematica assets for
the published intermediate systems and confirmed that every required native
fit matches its source expression. The nonmixing default uses
`p1,p2,p3,p4,p6`; the published alternate uses `p1,p2,p4,p5,p6`, but the
paper supplies no numerical dispatch region. The `(3,2)` system uses `p1`--
`p4` plus its two ringdown derivative conditions. See the
[phase-asset audit](pass-17-phase-fit-assets-primary.md).

The paper unambiguously selects the high-mass-ratio inspiral phase fit at
`q >= 100`, equivalent to `eta <= 100/10201`. Its official `Lambda.m` files,
however, contain several distinct exported fit records, and the allowed
source does not identify the production record or fully resolve the odd-mode
signed-amplitude phase mapping. The branch was therefore not enabled. See the
[primary audit](pass-16-highq-phase-primary.md) and
[independent review](pass-17-highq-phase-independent-review.md).

The literal Table II `(3,2)` amplitude cut contains the ratio
`f_ring32/f_ring32`, whereas the native code uses `f_ring32/f_ring22`.
No allowed primary erratum or official amplitude asset corroborates whether
the repeated denominator is intentional. That expression remains unchanged;
see the [amplitude audit](pass-17-amplitude-next-primary.md).

## Verification

The final ordinary suite passed **1,140 tests, with 9 skipped, in 81.34 s**:

```text
PYTHONPATH=src:tests PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
LAL_DATA_PATH=/tmp/torchwave-oracle-data \
/Users/xangma/miniconda3/envs/pycbcorig313/bin/python -m pytest -q
```

The final SHA-256 values are:

| File | SHA-256 |
| --- | --- |
| `src/torchwave/approximants/imrphenom_xhm.py` | `c54a2f8f3eb74f5b49836feca3a524d939f02991c71f8c089da58d362021aa31` |
| `tests/test_pass15_xhm_vetoes.py` | `2dde9599c1bf1737b479191d9674b1e52e1680560cebe042270c7a2b00290f9d` |

Both hashes exactly match the frozen pass-15 inputs. The strict verifier was
not rerun because no runtime or verifier input changed. The pass-15 strict
measurements therefore remain **11/21 catalog**, **4/13 holdouts**,
**3/34 components**, and **4/4 mode-sum controls**. No acceptance threshold,
calibration coefficient, or external case was changed.

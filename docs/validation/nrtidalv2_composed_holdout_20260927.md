# NRTidalv2 premerger complex-composition holdout — 27 September 2026

## Frozen gate before binary evaluation

This checks the **default `binary_compatible`** `IMRPhenomD_NRTidalv2`
composition against the pinned public LALSuite 7.26.15 binary, plus the
published nonspinning tidal phase against the independent scalar reference.
It does **not** claim that the default tidal-amplitude normalization is
paper-derived. The [phase specification](../math/nrtidalv2-phase.md) and
[amplitude/taper specification](../math/nrtidalv2-amplitude-taper.md) state
that distinction and document the separate failed gates.

The untouched systems `(m1,m2;Lambda1,Lambda2)` in solar masses are
`(1.25,1.25;900,900)`, `(1.48,1.22;600,250)`, and its body-exchanged copy.
They are absent from the earlier NRTidalv2 component cases and the strict
oracle holdout inventory. Frequencies are `30:10:1200` Hz; the test requires
the published merger fit to place the entire grid before taper onset. The
polarization and reference settings match the existing scalar/binary gates:
`h+`, distance 100 Mpc, inclination 0.4 rad, reference frequency 30 Hz,
zero orbital phase and time. CPU FP64 only.

The unaligned complex quotient is `h+_BNS/h+_BBH`, with each frontend's
own `IMRPhenomD` carrier and the corresponding public-binary carrier. No
phase or time shift, amplitude rescaling, or fit is allowed. The frozen
limits are relative complex L2 at most `1e-6`, maximum pointwise phase
residual at most `1e-5` rad, and maximum absolute quotient-modulus residual
at most `5e-5`. The pinned binary and both native unit quotients must also
agree with the independent [Dietrich et al. (2019), Eqs. (17)–(21)](https://arxiv.org/pdf/1905.06011v2)
scalar phase within `1e-5` rad. The complex ratio combines the tidal
phase and default amplitude in one check instead of multiplying separate
pass/fail summaries.

The test is `tests/test_nrtidalv2_composed_holdout_lal_binary.py`, selected
by the pinned-binary CI job. It requires `DIFFGW_REQUIRE_LAL_BINARY=1`,
JAX FP64, and the pinned LALSuite distribution; missing dependencies fail
under that strict setting.

## Result

**PASS: 3/3 strict cases, no skips, xfails, or deselections.** The thresholds
and systems above were recorded before the first binary run. Each case checks
both Torch and JAX against the public binary and the scalar phase reference.

| System | Printed-fit merger Hz | Worst complex relative L2 | Worst phase residual rad | Worst quotient-modulus residual |
| --- | ---: | ---: | ---: | ---: |
| Equal high tide | 1819.55874 | `1.04399722e-7` | `3.40393584e-7` | `6.27390887e-8` |
| Unequal mixed tide | 1766.11592 | `8.20676659e-8` | `2.55607708e-7` | `4.48940485e-8` |
| Exchanged mixed tide | 1766.11592 | `8.20676659e-8` | `2.55607708e-7` | `4.48940485e-8` |

The largest pinned-binary versus independent paper-phase residual was
`3.40393575e-7` rad; the largest native versus paper-phase residual was
`2.1318057e-14` rad. The entire test grid is below all three printed-fit
merger frequencies. This establishes named-binary agreement of the **default
nonspinning premerger complex tidal quotient on these holdouts**, not a
paper-grounded whole-model claim, a physical/NR validation, or agreement
within the nonzero-tide taper. The unresolved amplitude-normalization
provenance and previously failed high-frequency taper limits in the
[amplitude receipt](nrtidalv2_amplitude_20260926.md) remain unchanged.

The run used host `len`, cwd
`/home/xangma/tmp/diffgw-nrtidalv2-composed-20260927`, Linux CPU FP64,
Python 3.12.3, Torch 2.14.0+cpu, JAX 0.11.2, SymPy 1.14.0,
NumPy 2.5.3, and pinned LALSuite 7.26.15. No LAL implementation source was
inspected. The loaded public `lalsimulation` extension SHA-256 was
`e816879d23e652e8b3928496d6fd261de0832b5c3781c50e33a458042495b42e`.
The exact command was:

```sh
DIFFGW_REQUIRE_LAL_BINARY=1 JAX_ENABLE_X64=1 OMP_NUM_THREADS=1 \
PYTHONPATH=src:tests:/home/xangma/diffgw-pr45-318b37f7/.oracle-deps \
/home/xangma/.cache/diffgw-taylorf2-hm-venv/bin/python -m pytest -q -s \
  tests/test_nrtidalv2_composed_holdout_lal_binary.py
```

Output is in `composed_holdout.log` under that cwd. The command ran in the
foreground and completed in 9.06 s; no remote process remains or stop
command is needed. CI now selects this test in the pinned-binary job.

A follow-up strict focused run of the five existing phase/amplitude reference
and public-binary modules together with this new module returned **84 passed,
18 strict xfailed** in 31.78 s, with no skips. The 18 expected failures are
the previously documented paper-D versus binary and nonzero-tide taper
disagreements; none belongs to this new holdout. Its output is
`composed_focus.log` in the same remote cwd.

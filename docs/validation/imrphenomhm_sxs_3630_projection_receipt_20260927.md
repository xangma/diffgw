# IMRPhenomHM SXS:BBH:3630 projected-strain holdout — red receipt

**The frozen physical qualification fails.** The Lev2–Lev3 numerical
resolution screen passes after common time/orbital-phase alignment, and both
native frontends pass the two projected-match floors. At inclination `1.2`,
the model/NR PSD-weighted norm ratio is `0.840432`, below the predeclared
`0.85` floor. The [protocol](imrphenomhm_sxs_3630_projection_protocol_20260927.md)
was committed as `82675c7fb6e50973162675e198b1abf0e83fea20` before any
3630 strain was retrieved or decoded. No conditioning, optimizer, threshold,
or production coefficient was changed after the waveform was seen. The prior
[3619 red result](imrphenomhm_sxs_3619_projection_receipt_20260927.md)
remains unchanged.

## Source and method audit

The [official SXS:BBH:3630v3.0 record](https://doi.org/10.26138/SXS:BBH:3630v3.0)
provided the six exact files frozen in the protocol. The case was chosen from
official version-3 metadata only: q≈6, negligible spins, about 21.25 orbits,
and a 2022 run date. Published byte sizes and MD5 values were checked before
decoding. The raw SHA-256 identities, in protocol order, are:

| File | SHA-256 |
| --- | --- |
| `Lev3:Strain_N2.h5` | `4565b8600b13ba5d995c9d746e0fcc128d6efba9341b7757e71deed764f7d5e5` |
| `Lev3:Strain_N2.json` | `307042b593400e9c351659f30612d2c8839b062c0cece1e055e73cfb638f8c3a` |
| `Lev3:metadata.json` | `671b19d3ede11e6981373d9af659860361d5063bbb43ae1b3513646e2ee54f7a` |
| `Lev2:Strain_N2.h5` | `e63d83ba5aafe19c15a063d1a3785a8402597bba3cd066fea8aff20799ea6a24` |
| `Lev2:Strain_N2.json` | `47b05cbed8ac246382376551ce3b0ed48f1951d2238dc45cd05b69743f01cc4c` |
| `Lev2:metadata.json` | `bfed0d9ecfb998fb611dba5a0f319e74a9a28943b3ed4175f4efb3d9fd583c02` |

Both sidecars record extrapolation, space translations, boosts, memory
additions, and truncation. The memory integration starts are `778 M` (Lev3)
and `218 M` (Lev2); no correction was reversed. The SXS decoder reported
inertial spin-`-2` strain and all `2≤ell≤5` modes. The external fixture has
SHA-256 `8047eeee7b12aa7bc1d2304aa7c7987e65625bfaf103b436213b4c604992d234`.
Its Lev2 orbital basis sums at zero phase reproduce independently integrated
plus/cross spectra to relative maximum discrepancies of `1.81–2.04e-13`,
well below the frozen `1e-10` check.

For the NR resolution screen, the [SXS 2019 catalog Appendix B](https://arxiv.org/html/1904.04831#S.app2)
motivates a single common orbital-phase shift and a time shift of the Lev2
waveform, without a polarization retuning. Here the plus/cross channels are
summed as a two-detector network under the fixed band and conditioning of
this one-case protocol. The separate model/NR detector comparison follows
[London et al. (2018)](https://arxiv.org/html/1708.00404v4): fixed intrinsic
parameters and inclination, optimized template arrival time, orbital phase,
and polarization. The [documented PyCBC aLIGO design PSD](https://pycbc.org/pycbc/latest/html/psd.html)
was generated at evaluation time; its comparison-bin SHA-256 is
`0d20d2092bfaeb6073b0d656f4a3eb495bc3adbeb350ef279420cb4ffd0d67ea`.
No LAL source code or waveform array was copied. The two known-shift unit
checks recover exact one-channel and network time shifts. At all reported
optima, direct inner-product evaluation agrees with the inverse-FFT match
within `2.23e-16`; the four model optimizer seeds spread by at most
`6.34e-6`, within the predeclared `1e-5` feasibility check.

## Frozen-gate result

`M` is optimized model/Lev3 detector match (`≥0.980`), `A` its PSD-weighted
norm ratio (`0.85–1.15`), and `R` the Lev2–Lev3 plus/cross network match
(`≥0.995`). Torch values appear below; JAX matches differ by less than
`2e-15`, and its norm ratios by less than `3e-10`.

| Source `(iota,psi)` | M | A | R | Result |
| --- | ---: | ---: | ---: | --- |
| `(0.7,0.0)` | `0.991887103097` | `1.010480880219` | `0.999966664207` | pass |
| `(1.2,0.4)` | `0.980782452584` | **`0.840431923454`** | `0.999957014684` | **fail: A<0.85** |

The NR network norm ratios are `0.999850472100` and `0.999593922210`.
Both resolution optima use `-0.0064697265625 s` and orbital phases near
`5.9615 rad`; the 2048-point grid had 15 periodic local maxima at each
angle, and the eight highest were refined. This case is therefore resolvable
at the frozen NR fidelity threshold, but **does not qualify the model** under
the full predeclared gates. The result applies only to the frozen case,
100-solar-mass system, band, PSD, angles, modes, and conditioning.

## Reproduction

Host `len`; cwd `/tmp/diffgw-hm-sxs-3630-20260927/repo`; external files in
`/tmp/diffgw-hm-sxs-3630-20260927/data`. Preparation used Python 3.12.3,
NumPy 2.5.3, SciPy 1.18.1, SXS 2026.0.6, SymPy 1.14.0. Evaluation used
NumPy 2.3.5, SciPy 1.16.3, PyCBC 2.11.0, Torch 2.14.0, JAX 0.11.2.
Commands:

```sh
/tmp/diffgw-hm-sxs-2265-20260927/.venv/bin/python \
  -m scripts.prepare_imrphenomhm_sxs_3630_projection \
  --data-dir /tmp/diffgw-hm-sxs-3630-20260927/data
DIFFGW_REQUIRE_HM_SXS_3630_PROJECTION=1 \
  DIFFGW_HM_SXS_3630_PROJECTION_DIR=/tmp/diffgw-hm-sxs-3630-20260927/data \
  /tmp/diffgw-waveform-library-1315f6e/venv-latest/bin/python \
  -m pytest -q -s tests/test_imrphenomhm_sxs_3630_projection.py
```

Preparation completed successfully (launcher PID `1293985`; log
`/tmp/diffgw-hm-sxs-3630-20260927/prepare.log`). The final strict runner
PID `1323822` finished with **2 failed, 2 passed** in 23.74 s; the two
expected assertion failures are the red physical gates, not execution
errors. Its log is `/tmp/diffgw-hm-sxs-3630-20260927/strict.log`, and
the complete external metrics are
`data/hm_sxs_3630_{torch,jax}_projection_metrics.json`. No job remains
active, so there is no next check or stop command; during a rerun, stop its
active PID with `kill <PID>` on `len`. Raw SXS files and derived arrays stay
outside Git. The source-only run passed its two synthetic checks and skipped
the two physical gates without the strict flag. Ruff and syntax checks passed.

## Post-result mode diagnostic (non-gating)

After committing the red receipt, a read-only Torch diagnostic at
`(iota,psi_s)=(1.2,0.4)` decomposed the six *model* components at the
already-selected template orbital phase and polarization. Removing one mode
at a time held those angles fixed and remaximized only the discrete arrival
shift. The `(2,1)` component has norm `0.05286` relative to full NR and
incoherent power `0.00396` relative to the full model. Removing it raises the
model/NR norm ratio from `0.840432` to `0.849778`, still below the frozen
`0.85` floor, while reducing the match from `0.980782` to `0.972037`.
Removing `(3,3)` gives ratio `0.752416` and match `0.892565`; removing
`(4,4)` gives `0.824919` and `0.966130`. The `(4,3)` ablation slightly
raises match to `0.981162` but lowers ratio to `0.838589`. These are
coherent-waveform ablations: individual mode powers do not sum to total
power, and this fixed-optimum probe does not assign physical error to a
specific mode or alter acceptance.

The external script and complete output are
`/tmp/diffgw-hm-sxs-3630-20260927/ablation.py` and `ablation.log` on `len`.
It ran from the same remote cwd with
`PYTHONPATH=.:src:tests /tmp/diffgw-waveform-library-1315f6e/venv-latest/bin/python
/tmp/diffgw-hm-sxs-3630-20260927/ablation.py`; it completed in 6.6 s.

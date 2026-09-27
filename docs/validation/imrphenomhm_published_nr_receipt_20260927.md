# Published-calibration IMRPhenomHM: two-case NR result

Both prospective cases fail the frozen physical qualification gate. Each passes
its two-resolution NR screen and its first model view; each fails both the
match and amplitude-norm gates at the second view. Torch and JAX give the same
outcomes. No case, coefficient, option, conditioning step or threshold was
changed after strain inspection.

The [protocol](imrphenomhm_published_nr_protocol_20260927.md), both manifests,
runner and data-free tests were committed and pushed as
`4449a7340a7198b25a29c662c2030b91f001b002` before either strain download.
The exact native option is `calibration="published"`, all three named
`mishra_eq12b`/`mishra_eq12d`/`mishra_eq12g` PN choices, and
`hm_mapping_transitions="london_2018"`. The
[independent mathematical composition](../math/imrphenomhm-waveform.md)
remains supported by its separate implementation evidence; these physical
results do not establish the model's accuracy against NR.

## Results

The score uses full NR modes through ell=5 at 100 solar masses and 500 Mpc,
30–1024 Hz, the aLIGO design PSD, fixed intrinsic parameters and inclination,
and optimized template arrival time, orbital phase and polarization. A norm
ratio is the PSD-weighted template norm divided by the NR norm; it is not a
pointwise amplitude error. The frozen model gates require match >=0.980,
norm ratio in [0.85,1.15], and four-seed match spread <=1e-5.

Torch and JAX agree to the digits below, so one row reports both:

| Case | View `(iota,psi_s)` rad | Model/NR match | Model/NR norm ratio | Result |
| --- | --- | ---: | ---: | --- |
| SXS:BBH:2374, q≈5, nonspinning | (0.7,0.0) | 0.988815248 | 0.939174835 | Pass |
| SXS:BBH:2374 | (1.2,0.4) | 0.971607892 | 0.750351161 | Match and norm fail |
| SXS:BBH:2163, q≈3, spins≈(+0.6,+0.6) | (0.7,0.0) | 0.990008030 | 0.986805776 | Pass |
| SXS:BBH:2163 | (1.2,0.4) | 0.972272538 | 1.223703469 | Match and norm fail |

One common time/orbital rotation per case aligns the lower resolution to the
higher resolution jointly across both plus/cross views. Each view requires
match >=0.995 and norm ratio in [0.95,1.05]:

| Case and lower/higher levels | View | NR match | NR norm ratio | Result |
| --- | --- | ---: | ---: | --- |
| 2374 Lev2/Lev3 | (0.7,0.0) | 0.999954892 | 1.000316136 | Pass |
| 2374 Lev2/Lev3 | (1.2,0.4) | 0.999933731 | 1.000611953 | Pass |
| 2163 Lev3/Lev4 | (0.7,0.0) | 0.999901587 | 1.000104032 | Pass |
| 2163 Lev3/Lev4 | (1.2,0.4) | 0.999891293 | 1.000155881 | Pass |

The largest four-seed match spread is 2.00e-10, well below 1e-5.
Every recorded optimizer trial reports success. Maximum mode-scaled
Torch/JAX complex differences are below 6.61e-13. Direct/FFT rechecks pass;
mode-basis reconstruction discrepancies are below 3.80e-13.
Thus neither the tested NR resolution difference, backend disagreement nor
observed optimizer-seed variability explains these failed gates. This does
not prove the absence of a shared convention or composition error.

At the second view, dropping modes outside the six model mode pairs from NR
alone gives full/six-pair plus/cross network matches 0.997324 (2374) and
0.998825 (2163), and norm ratios 0.991622 and 1.000575. These network
diagnostics differ from the optimized scalar detector projections used for
model acceptance; they cannot quantitatively attribute the failed model gates
to omitted modes. The results do not uniquely identify calibration precision,
a source equation, phase convention or mapping choice as the cause.
No diagnostic changes the qualification result. A useful next discriminating
study is a component-by-component source-option comparison on these now-seen
cases, explicitly exploratory and followed by new untouched validation if the
model changes.

## Evidence and provenance

Complete maintained metrics retain both backends, every seed, NR alignment
candidates, source SHA-256 values, source-tree hash, package versions,
conditioning and omitted-mode diagnostics:

- [2374 metrics](imrphenomhm_published_sxs_2374_metrics_20260927.json),
  [frozen manifest](imrphenomhm_published_sxs_2374_protocol_20260927.json).
- [2163 metrics](imrphenomhm_published_sxs_2163_metrics_20260927.json),
  [frozen manifest](imrphenomhm_published_sxs_2163_protocol_20260927.json).

The metadata-only selection and rerun caveat are in the protocol. These cases
were unseen local prospective holdouts, not proven independent of every
historical model-development dataset. Official CaltechDATA size/MD5 checks
passed for all twelve files before decoding. Raw and conditioned arrays remain
external on `len` under `/tmp/diffgw-hm-published-nr-20260927/{2374,2163}`;
the software license does not relicense those data. See
[dataset attribution](https://github.com/xangma/diffgw/blob/codex/waveform-evidence-integration/THIRD_PARTY_NOTICES.md#scope-of-this-notice).

| Case | Manifest SHA-256 | Conditioned fixture SHA-256 |
| --- | --- | --- |
| 2374 | `4381ddf5537e328c0708ce9f77152ea985c075302a5d7f1f09e62fd2cb13fccd` | `e163d466ac0f56f6506d09ce438b4c52cd919fefab7197884038e038f085d0b1` |
| 2163 | `477083ba0fa8ba6346008d946cd41504abc1111d9f727e82ff8bdbc42ed80b58` | `bb6cc4dc43f726aca4c9209c76d7ba615167aaa89e9fe06e6bbe841a366ae707` |

## Reproduction and checks

Use the frozen revision, whose source hashes are enforced before data access.
The run completed on `len` between 22:36:04 and 22:37:18 UTC on 2026-09-27,
in `/tmp/diffgw-hm-published-nr-20260927/repo`. For each case, substitute the
manifest hash from the table:

```sh
export PYTHONPATH=src JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
CASE=2374
HASH=4381ddf5537e328c0708ce9f77152ea985c075302a5d7f1f09e62fd2cb13fccd
PREP=/tmp/diffgw-hm-sxs-2265-20260927/.venv/bin/python
EVAL=/tmp/diffgw-waveform-library-1315f6e/venv-latest/bin/python
ARGS=(scripts/run_imrphenomhm_published_nr.py
  --manifest "docs/validation/imrphenomhm_published_sxs_${CASE}_protocol_20260927.json"
  --manifest-sha256 "$HASH"
  --data-dir "/tmp/diffgw-hm-published-nr-20260927/$CASE"
  --source-revision 4449a7340a7198b25a29c662c2030b91f001b002)
"$PREP" "${ARGS[@]}" --prepare-only
"$EVAL" "${ARGS[@]}" --evaluate-only
```

Preparation used Python 3.12.3, NumPy 2.5.3, SciPy 1.18.1, SXS 2026.0.6
and SymPy 1.14.0. Evaluation used Python 3.12.3, NumPy 2.3.5, SciPy 1.16.3,
PyCBC 2.11.0, Torch 2.14.0, JAX/JAXlib 0.11.2 and SymPy 1.14.0.
The separate environments retain current SXS and PyCBC-compatible dependencies.

Before retrieval, the maintained manifest/option/level/spin tests plus the
existing 3144 harness tests passed **16/16** on len; Ruff passed. A data-free
native check exercised all six modes in both cases on both frontends.
The campaign exited successfully as an execution job; its stored
`qualified=false` values are scientific failures, not execution errors.
An independent audit rechecked every gate and the actual remote bytes of all
twelve raw files, both fixtures and both metrics files. Both generated-document
checks and the final strict MkDocs build passed on len.
All earlier NR results remain unchanged. Whole-family physical qualification
and an error bound over the parameter domain remain open.

The completed [source-option and reference-frequency diagnostic](imrphenomhm_published_options_20260927.md)
records controlled ablations and a source-derived covariance defect. It leaves
this frozen result unchanged; no complete physical repair is established.

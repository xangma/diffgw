# IMRPhenomHM SXS:BBH:2158 aligned-spin holdout — green receipt

The [prospective protocol](imrphenomhm_sxs_2158_prospective_protocol_20260927.md)
and [runner](https://github.com/xangma/diffgw/blob/4ec1383/scripts/run_imrphenomhm_sxs_2158.py) were committed as
`4ec1383` **before** retrieving or decoding any 2158 strain. The case,
conditioning, optimization, and thresholds were not changed after seeing
the data. The frozen two-resolution screen and both native frontend model
screens pass. This is one case-specific physical result, not full-domain
qualification or a repair of the earlier red SXS cases. The
[complete machine-readable metrics](imrphenomhm_sxs_2158_metrics_20260927.json)
retain all source hashes, alignment candidates, optimization seeds, norms,
and mode diagnostics.

## Source and execution identity

The metadata-only filter and sort in the protocol selected
[SXS:BBH:2158v3.0](https://doi.org/10.26138/SXS:BBH:2158v3.0) from the
official SXS v3.0.0 catalog, SHA-256
`132d38520b3562c8f254b3c9aaf8efd4678428f94dbbabfef4436bc50c1c87c6`.
All six Lev4/Lev3 N2 strain files came from the official
[CaltechDATA record](https://data.caltech.edu/records/3gb3n-e5t37).
Their byte sizes and published MD5 values matched the precommitted manifest
before decoding; their SHA-256 values are in the metrics. The prepared
fixture SHA-256 is
`003806ce22416cd1ecf607c33471aaa1fa4ff93e9bf7a24ea2a3c626141bc0be`;
the runner SHA-256 is
`c1e32275d709d2b5a2020a37fd0ade4f263da6f6eff22478a6e5cae42d3bd431`.
Raw and derived strain remain outside Git on `len` under
`/tmp/diffgw-hm-sxs-2158-20260927/data`.

On `len`, from `/tmp/diffgw-hm-source-audit-20260927/repo`, the preparation
used `PYTHONPATH=src` and
`/tmp/diffgw-hm-sxs-2265-20260927/.venv/bin/python
scripts/run_imrphenomhm_sxs_2158.py --data-dir
/tmp/diffgw-hm-sxs-2158-20260927/data --prepare-only --source-revision
4ec1383`. The evaluation used the same runner and directory with
`/tmp/diffgw-waveform-library-1315f6e/venv-latest/bin/python` and
`--evaluate-only --source-revision 4ec1383`, with `JAX_ENABLE_X64=1` and
`JAX_PLATFORMS=cpu`. The preparation environment had Python 3.12.3,
NumPy 2.5.3, SciPy 1.18.1, `sxs` 2026.0.6, and SymPy 1.14.0. The
evaluation environment had Python 3.12.3, NumPy 2.3.5, SciPy 1.16.3,
PyCBC 2.11.0, Torch 2.14.0, JAX/JAXlib 0.11.2, and SymPy 1.14.0.
The 30–1024 Hz PSD SHA-256 was
`0d20d2092bfaeb6073b0d656f4a3eb495bc3adbeb350ef279420cb4ffd0d67ea`.

The official inertial, spin-weight `-2` strain contained every signed-
`m` mode through `ell=5`, including `m=0`. The common interval was
`1410.0–6820.165009984545 M`, with 21,830 uniform samples. The
Lev4 `|h_22|` peak at `6672.998518942362 M` set time zero. The
mode-basis FFT reconstructed all direct Riemann spectra to relative
maximum error below `3.58e-13`; the synthetic time-shift and all direct
inner-product checks passed.

## Frozen gates

One Lev3-to-Lev4 two-view plus/cross alignment selected orbital phase
`6.225823345700972 rad` and arrival shift `-0.0009765625 s`.
The combined network match was `0.999996513679` and norm ratio
`1.000530895011`. That *same* alignment gave:

| Fixed `(iota,psi_s)` | Lev3/Lev4 match (floor 0.995) | Norm ratio (0.95–1.05) |
| --- | ---: | ---: |
| `(0.7,0.0)` | `0.999996706940` | `1.000412490033` |
| `(1.2,0.4)` | `0.999996028926` | `1.000877198583` |

With catalog mass ratio `3.0001323450539252`, fixed aligned spins
`(0.49997718539303,0.5000516952582452)`, total mass `100 M_sun`,
and no intrinsic or amplitude fitting, the independently optimized
Torch and JAX templates gave:

| Frontend | View | Model/Lev4 match (floor 0.980) | Norm ratio (0.85–1.15) | Four-seed match spread |
| --- | --- | ---: | ---: | ---: |
| Torch | `(0.7,0.0)` | `0.994088357164` | `1.003313403120` | `5.16e-11` |
| JAX | `(0.7,0.0)` | `0.994088357164` | `1.003313403120` | `5.16e-11` |
| Torch | `(1.2,0.4)` | `0.986499006522` | `1.074226637473` | `3.14e-11` |
| JAX | `(1.2,0.4)` | `0.986499006522` | `1.074226637473` | `3.14e-11` |

All optimizer seeds met the precommitted `1e-5` agreement rule. The full
NR signal minus its six template-mode pairs had weighted residual norm
fractions `0.01647` and `0.04887` in the two views; that is a diagnostic,
not a fitted correction. The weakest model match is `0.98650`, above the
frozen floor. This fixed-view result is not directly comparable to London's
polarization/orbital-phase population averages. The existing
[2265](imrphenomhm_sxs_2265_receipt_20260927.md),
[3631](imrphenomhm_sxs_3631_receipt_20260927.md),
[3619](imrphenomhm_sxs_3619_projection_receipt_20260927.md), and
[3630](imrphenomhm_sxs_3630_projection_receipt_20260927.md) failures
remain unchanged. This green result supports only the stated aligned-spin
case, mass, band, PSD, two views, finite resolutions, and fixed native
six-mode option; it does not establish all-sky or model-family accuracy.

## Post-result diagnostic and case closeout

The maintained `scripts/diagnose_imrphenomhm_sxs_2158.py` reproduces this
audit from the frozen inputs and best-fit Torch alignment. It checks source,
fixture, runner, metrics, and PSD hashes before evaluation. Neither mode nor
frequency-band diagnostics fit a new phase, time, amplitude, or intrinsic
parameter. The [diagnostic evidence](imrphenomhm_sxs_2158_diagnostic_20260927.json)
records both views, all six projected mode pairs, package versions, and
the diagnostic's SHA-256. These are post-result diagnostics, not new
prospective gates. The runner and original metrics remain unchanged.

For the weaker `(1.2,0.4)` view, replacing the native modes with modes from
the public **LALSuite 7.26.15 binary**, using the previously fixed mode
adapter and the same projection and alignment, gives:

| Candidate / reference | Match | Norm ratio |
| --- | ---: | ---: |
| Native HM / full NR | 0.986499006522 | 1.074226637473 |
| LAL HM / full NR | 0.986498963403 | 1.074226679116 |
| Native HM / six-pair NR | 0.987735823627 | 1.070556561918 |
| Six-pair NR / full NR | 0.998824938077 | 1.003428193975 |

Across the six complex carriers, native/LAL relative L2 is at most
`1.985e-5`; relative error on bins above 1% of each LAL mode's peak is at
most `1.252e-5`. Torch/JAX relative L2 is below `3.404e-13`. The LAL row
uses the independent reference projector, not LAL's public polarization
API; that separate API is covered by the maintained
[polarization holdout](imrphenomhm_lal_polarizations_20260927.md).
No external implementation source was inspected, and no production
coefficient was fitted to binary or NR output.

This near-identical binary result points away from a native frontend
discrepancy as the explanation for this match. Restricting NR to the
model's six pairs still leaves substantial disagreement. Missing modes
alone therefore do not explain the result. Shared model approximations
and comparison conventions remain possible sources; this diagnostic does
not uniquely identify a physical cause.

For the full detector spectrum, the exact pointwise identity

\[
|r-c|^2=(|r|-|c|)^2+4|r||c|\sin^2[(\arg c-\arg r)/2]
\]

splits squared residual into amplitude and phase terms, integrated with
the frozen PSD. Normalizing by full NR power gives total `0.03451584736`,
amplitude `0.02308207006` (66.87%), and phase `0.01143377730` (33.13%).
This is **squared complex residual**, not mismatch `1-match`; the norm
offset contributes to it. Amplitude/phase here describe the coherently
summed detector signal, not the source of individual mode errors.

| Frequency bins (Hz) | Share of total squared residual |
| --- | ---: |
| 30–59 | 12.44% |
| 60–99 | 19.47% |
| 100–199 | 43.94% |
| 200–399 | 20.41% |
| 400–1024 | 3.75% |

These disjoint bands partition the residual; their boundaries are
diagnostic choices, not waveform transition frequencies. Projected
mode-pair diagnostics expose a `(4,4)` model/NR norm ratio of `1.35083`
and substantial `(3,2)` phase disagreement under the shared alignment.
They are not intrinsic amplitude measurements or independently optimized
mode matches. Mode errors interfere coherently, so their powers cannot
be added or interpreted as unique causal shares. In particular, the
shared alignment compromises across modes.
Likewise, the JSON's `nr_power_fraction_of_full` is a per-pair/full-signal
power ratio: destructive interference permits values above one, and these
ratios do not sum to one.

[London et al., arXiv:1708.00404v4](https://arxiv.org/html/1708.00404v4)
describe approximate quadrupole mappings and omitted spherical–spheroidal
mixing. These provide relevant model limitations, not proof that a
particular approximation causes this case's residual. Improving the
physics would require a separately specified, independently sourced
model change and new held-out validation, rather than a correction tuned
to this result.

**Disposition:** the 2158 case and its diagnostic are complete. Its frozen
gates pass, with no change to thresholds or production waveforms. Whole-HM
qualification remains open for the inherited fitted-input provenance and
physical coverage listed in the
[family ledger](../waveforms/research/phenom-higher-modes.md#evidence-and-limitations);
the earlier red cases retain their original status.

### Reproduce the diagnostic

On `len`, from `/tmp/diffgw-hm-source-audit-20260927/repo`:

```bash
PYTHONPATH=src JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu \
  /tmp/diffgw-waveform-library-1315f6e/venv-latest/bin/python \
  -m scripts.diagnose_imrphenomhm_sxs_2158 \
  --data-dir /tmp/diffgw-hm-sxs-2158-20260927/data \
  --output /tmp/diffgw-hm-sxs-2158-20260927/diagnostic.json
```

The data directory can instead be regenerated with the frozen preparation
command above. Nine maintained analytical tests in
`tests/test_hm_residual_budget.py` pass on `len`, covering known amplitude
and phase offsets, frequency partitioning, coherent cancellation,
zero model support above a mode cutoff, and invalid PSD rejection.

# Conditional joint complex NR comparison

The selected reduced basis does **not** justify replacing the existing Eq.42
amplitude model. Model selection used only the five original training families;
the seven previously examined cases below are regression checks. One additional
fresh family passed the unchanged amplitude gates and was evaluated without refitting.

## Frozen comparison

The candidate jointly represents the complex `(2,2)` and `(3,3)` spectra with a
small SVD basis. Both modes share one orbital rotation and one time origin.
The comparator preserves the existing full-training Eq.42 `(3,3)` amplitude
fit and supplies a new, simple training-only phase interpolation; `(2,2)` uses
log-amplitude/phase interpolation. Thus this is a comparison with a diagnostic
completion of the amplitude baseline, not with an existing full waveform model.

Both models use q plus an NR remnant supplying an independent Kerr `(3,3)` QNM.
The experiment covers only `0.95*f_ring,33 <= Mf < 0.3`, including `(2,2)` on
that same high-frequency interval. It does not establish full waveform,
detector-strain, spin, or LAL parity. There was no LAL fitting or runtime change.

Selection chose rank 3 and degree 2 for the complex basis, and phase degree 2
for the comparator. Mean leave-one-training-family-out complex error, averaged
equally over modes, was **12.15% versus 6.39%**. The endpoint folds extrapolate.
The selected candidate lost on every training fold.

## Existing regression cases

The six interpolation cases are q approximately 3, 3.5, 4.5, 5.5, 6, and 7.
All scores preserve the shared gauge; no per-case alignment improves them.

| Six interpolation cases | Reduced basis | Completed Eq.42 baseline |
|---|---:|---:|
| Median equally averaged mode complex error | 4.94% | 4.47% |
| Worst equally averaged mode complex error | 7.14% | 7.59% |
| Median `(3,3)` complex error | 1.78% | 1.04% |
| Worst `(3,3)` complex error | 3.87% | 2.91% |
| Median `(3,3)` amplitude error | 1.07% | 0.67% |
| Worst `(3,3)` amplitude error | 1.27% | 0.80% |
| Worst equal-weight coherent-sum complex error | 4.15% | 3.43% |

The candidate has higher `(3,3)` complex error on all seven regression cases,
including the q approximately 10 extrapolation case. It introduces four
interpolation cases above the declared 1% `(3,3)` amplitude threshold. It fails
the predeclared win criterion; it was not tuned after seeing these results.

## Fresh-family check

Metadata-only selection chose SXS:BBH:2488 at q approximately 6.5, plus
SXS:BBH:0201 at q approximately 2.316 as a deterministic fallback. The latter
failed the unchanged extraction (2.72%) and resolution (7.93%) amplitude gates;
it was not scored by the frozen models. No replacement was selected.
Two metadata-ranking implementation errors were corrected and audited without
using waveform/model results: an incomplete prior-q list, and the tolerance/
available-level filters. The intended rules select the same two families;
the initial artifacts and correction audit are preserved in the data directory.

SXS:BBH:2488 passes the amplitude gates and declared affine phase diagnostics.
Its result is mixed:

| Fresh q = 6.49955 | Reduced basis | Completed Eq.42 baseline |
|---|---:|---:|
| `(2,2)` complex error | 2.02% | 3.35% |
| `(3,3)` complex error | 2.09% | 0.95% |
| Equally averaged mode complex error | 2.06% | 2.15% |
| `(3,3)` amplitude error | 0.90% | 0.72% |
| Equal-weight coherent-sum complex error | 2.10% | 1.03% |
| Relative mode phase RMS | 0.0164 rad | 0.0213 rad |

The raw resolution/extraction complex sensitivities reach 9.81% for `(2,2)`
and 3.58% for `(3,3)`, exceeding the differences between models. One qualified
fresh family does not fill the requested two-family coverage, and does not
establish a reliable overall improvement. The candidate remains unpromoted.

## What limits this conclusion

Numerical implementation checks pass: full-rank training reconstruction error
is `2.91e-15`, and conditional query JVP errors against centered finite differences
are below `1.34e-10`. Rank-3 projection error on the scaled training spectra is
0.75%. That projection norm differs from the physical scoring norm, but the
large increase in cross-validation error points toward sparse parameter
interpolation and phase/data conditioning as candidates for further diagnosis.
It does not prove reduced-basis waveform models are generally inferior.

All five training families pass the declared affine phase-residual diagnostic.
Nevertheless, their raw shared-gauge `(2,2)` resolution/extraction sensitivities
reach 8.30%. For regression families q approximately 3, 6, and 7, the `(2,2)`
affine diagnostic itself fails. Across regression controls, raw `(2,2)` complex
sensitivity reaches 15.92%, and `(3,3)` reaches 3.59%. These are sensitivity
comparisons, not rigorous error bars. Removing an affine phase difference in
a diagnostic can hide differences that remain in the shared-gauge model score.

The comparison therefore supports **stopping this particular candidate**, and
strengthening prospective shared-phase data qualification before expanding to
spins or choosing a more flexible interpolator. A future experiment should
declare its complex error budget before choosing data or inspecting holdouts.

A separate post-validation [peak-gauge diagnostic](peak-gauge-diagnostic/README.md)
found that continuous quadratic peak timing reduces the fresh family's raw
inter-resolution complex difference from 9.81% to 7.37% for `(2,2)` and from
3.58% to 1.78% for `(3,3)`. It leaves the affine phase residuals essentially
unchanged. This identifies a contribution from sampled timing without refitting
either frozen model or changing the reported scores.

## Evidence and reproduction

- [Predeclared protocol](../protocol.md) and [protocol hashes](../protocol-freeze.json).
- [Training cross-validation](cross-validation.json), [training controls](training-quality.json),
  [frozen models](models.json), and [model/source hashes](model-freeze.json).
- [Per-case regression results and controls](regression.json).
- [Fresh-family results and controls](fresh-holdout.json),
  [qualified inputs](../data/qualified-manifest.json), and
  [screening failures and coverage gap](../data/holdout-status.json).

Run from the repository root with the `pycbcorig313` environment and `PYTHONPATH=src`:

```sh
python -m scripts.run_nr_strategy_comparison train --output /tmp/new-nr-strategy-run
python -m scripts.run_nr_strategy_comparison evaluate --output /tmp/new-nr-strategy-run --manifest docs/reviews/2026-09-15-independent-phase-benchmark/input-manifest.json --label regression
python -m scripts.run_nr_strategy_comparison evaluate --output /tmp/new-nr-strategy-run --manifest docs/reviews/2026-09-15-parity-strategy-experiment/data/qualified-manifest.json --label fresh-holdout
python -m pytest -q tests/test_nr_complex_reduced_basis.py tests/test_nr_strategy_comparison.py
```

The commands require the hash-verified cached inputs referenced by the existing
manifests. Training and evaluation refuse to overwrite existing artifacts.
Evaluation verifies frozen source/model hashes and never refits the model.

# Replacement NR33 calibration: preparation corrected; five of seven qualify

## Result

**No new calibration was fitted.** Five replacement simulations pass every
predeclared check. The q≈6 and q≈8 cases fail only the minimum post-peak duration:
139.89M and 138.87M, respectively, against a frozen requirement of 150M. Here M
is the total reference mass in geometric time units (G=c=1).

This leaves four training simulations and one holdout, short of the required
five training simulations and two holdouts. Neither failed target has a frozen
alternate. The [actual screen](quality-screen-round1.json) therefore emits no
qualified fitting manifest, and the fitting wrapper rejects it. No new
coefficients, freeze, or LAL comparison were produced.

| Target q | SXS:BBH ID | Highest / next Lev | Post-peak M (≥150) | N2/N3 amplitude difference (≤1%) | Resolution difference (≤1%) | Result |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 1.5 | 2331 | 3 / 2 | 170.39 | 0.01297% | 0.05965% | Pass |
| 2 | 1166 | 3 / 2 | 181.78 | 0.71230% | 0.14647% | Pass |
| 3, holdout | 2265 | 3 / 2 | 183.49 | 0.00584% | 0.41375% | Pass |
| 4 | 2485 | 5 / 4 | 153.52 | 0.17288% | 0.04107% | Pass |
| 5 | 3619 | 3 / 2 | 154.59 | 0.02167% | 0.11418% | Pass |
| 6, holdout | 3630 | 3 / 2 | **139.89** | 0.02139% | 0.12850% | Duration fails |
| 8 | 3622 | 3 / 2 | **138.87** | 0.02307% | 0.11191% | Duration fails |

All seven pass spin, eccentricity, matching-metadata, extraction, resolution,
and preprocessing checks. The largest preprocessing variation is 0.00360%,
below the 0.2% limit. Differences are relative L2 norms of Fourier **amplitude**
over the frozen band, not complex-waveform mismatches or bounds on true error.

The short durations do not themselves prove the two simulations inaccurate:
both have stable spectra in these checks. They make the simulations ineligible
under this experiment's predeclared design. The next dataset requirement is
longer eligible q≈6 and q≈8 simulations; any further search needs a new recorded
candidate selection, while preserving the holdouts and numerical thresholds.

## Scope

This experiment corrects the resolution/metadata pairing error found in the
[original pilot diagnosis](../2026-09-15-independent-calibration/extraction-diagnosis/README.md).
It tests whether replacement simulations can support a new independent fit of
the nonspinning (3,3) Fourier-amplitude model. It does not establish full LAL parity.

The [quality policy](quality-policy.json) was frozen before replacement waveform
inspection. Its SHA256 is
`1a863811b90c410cd5a9b0b52a0bd53696b869a0d503cb8fa61d9c054842138d`.
Targets q≈1.5,2,4,5,8 are training cases; q≈3,6 are holdouts. Historical pilot
simulations are excluded. Replacement runs can share a physical family with
historical simulations, so this is not a claim of independent NR generation.

## Corrections

- Preparation binds each strain H5 to its exact record, resolution, waveform
  sidecar, and physical metadata file, with inventory size/MD5 and local SHA256
  checks. Catalog-level scalar defaults cannot substitute for matching metadata.
- Mass ratio comes from the reference masses. Remnant mass and code-unit times
  are normalized by their sum; spin magnitudes come from the vectors. The memory
  integration start remains separate from relaxation time.
- The screen compares the highest two resolutions and N2/N3 extraction orders
  on a shared frequency grid. It also checks spin, eccentricity, ringdown
  duration, sampling, and taper sensitivity.
- The fitting wrapper requires all seven quality-qualified simulations, verifies
  the predeclared holdouts against the unchanged fitter, pins numerical source
  hashes, and refuses to overwrite a freeze. LAL validation is permitted only
  after a successful new freeze.

## Source handling and interpretation

The [candidate inventory](candidate-inventory.json) freezes 13 candidates and
their order. Discovery used a catalog snapshot dated 2025-05-14 and exact
CaltechDATA records retrieved on 2026-09-15; it is not a claim of an exhaustive
2026 catalog search. The [round-one selection](selection-round1.json) chose
seven using metadata alone, before downloading their waveform H5 files.

Current records publish standalone N2 strain; N3 strain is stored in
`ExtraWaveforms.h5` under `/Strain_N3.dir`, declared in the matching
`ExtraWaveforms.json`. Initial standalone-only availability checks were
incomplete. The embedded layout was identified and recorded before downloading
the replacement waveforms. This changes file handling, not extraction order or
the numerical policy. The inventory's `decode pending` fields describe its
pre-download state; the completed [28-case manifest](nr33-manifest-round1.json)
records the actual decoded sources and hashes.

All 84 cached source files passed inventory MD5/size checks; the
[download record](download-verification.json) also saves SHA256 hashes. The
28 decodes cover each simulation's highest two available resolutions and both
N2/N3. All decoded sources report an inertial frame. Metadata remains bound to
its exact record and resolution, including published aliases referring to an
older simulation family.

The decoder warns that a generic `raw/metadata.json` is absent because metadata
files have unique record/resolution filenames here. The pinned loader decodes
the waveform before optionally attaching metadata and retains all times by
default. A [controlled check](decoder-metadata-check.json) explicitly attached
the matching metadata for both duration failures: decoded time, h22, and h33
arrays were exactly equal to the saved arrays. The warning does not explain
either failed duration.

The replacement q≈3 extraction difference is 0.00584%, a substantial improvement
in internal consistency over the problematic historical case. This supports
using better controlled NR inputs, but does not measure agreement with LAL or
show that the original fit discrepancy has disappeared. No fitting residuals
were used for selecting these data.

SXS describes cross-resolution and cross-extrapolation comparisons as numerical
accuracy diagnostics, and notes that extrapolation performance depends on the
waveform regime. These checks follow that principle with thresholds fixed for
this experiment. [SXS waveform guidance](https://data.black-holes.org/waveforms/)

The project's MIT license and runtime are unchanged. Raw H5 and decoded NPZ
files remain outside the repository. Current record metadata did not expose a
per-record license value; no old record's license is assigned to these new
records. See [data attribution and record links](DATA_ATTRIBUTION.md).

## Reproduction

The preparation, screening, and guarded fitting entry points are
`scripts/prepare_nr33_calibration_v2.py`,
`scripts/screen_nr33_calibration_v2.py`, and `scripts/calibrate_nr33_v2.py`.
Each accepts `--help`. Preparation requires the pinned SXS decoder version
2026.0.3. Raw files belong in `/tmp/torchwave-nr33-v2/`, outside this repository.

From the repository root:

```sh
PYTHONPATH=src /tmp/torchwave-nr33-pilot/venv/bin/python -m scripts.prepare_nr33_calibration_v2 \
  --selection docs/reviews/2026-09-15-independent-calibration-v2/selection-round1.json \
  --inventory docs/reviews/2026-09-15-independent-calibration-v2/candidate-inventory.json \
  --output /tmp/torchwave-nr33-v2/reproduced-manifest.json \
  --npz-dir /tmp/torchwave-nr33-v2/reproduced-npz \
  --expected-policy-sha256 1a863811b90c410cd5a9b0b52a0bd53696b869a0d503cb8fa61d9c054842138d

PYTHONPATH=src python -m scripts.screen_nr33_calibration_v2 \
  --manifest /tmp/torchwave-nr33-v2/reproduced-manifest.json \
  --policy docs/reviews/2026-09-15-independent-calibration-v2/quality-policy.json \
  --output /tmp/torchwave-nr33-v2/reproduced-screen.json \
  --qualified-output /tmp/torchwave-nr33-v2/reproduced-qualified.json
```

Do not invoke the numerical fitter for this failed dataset. Preparation logs are
at `/tmp/torchwave-nr33-v2/prepare-round1.log`; screening logs are at
`/tmp/torchwave-nr33-v2/screen-round1.log`. Both jobs completed on RONI2 from
`/Users/xangma/repos/torchwave`; no job remains running. Exact decoding command,
timestamps, Python path, and source hash are in the manifest. Screening used
Python 3.13.9, NumPy 2.3.5, and SciPy 1.16.3; its source hash is in verification.

## Validation and integrity

The six focused suites passed: **39 tests** covering the existing fitter and
validation plus preparation, screening, provenance, and admission guard. Luna
reviews caught and corrected embedded group dispatch and primary-split handling
before the final screen.

The [verification report](verification.json), produced by [verify-run.py](verify-run.py),
checks all 28 NPZ hashes, finite arrays and monotonic times, recomputes post-peak
durations directly from decoded h22, verifies the absence of frozen alternates,
and confirms that the actual failed quality report is rejected before fitting.

The [initial integrity snapshot](integrity-start.json) pins all 37 artifacts and
scripts of the previous calibration experiment plus 139 runtime inputs. The
final audit confirms **37/37 historical artifacts and 139/139 runtime inputs
unchanged**. New work is confined to separate experimental scripts, tests, and
this report.

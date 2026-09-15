# Expanded validation of the frozen NR33 calibration

**Four new cases inside the training range retain subpercent amplitude agreement. The q≈10 extrapolation does not.** The existing coefficients were unchanged throughout; no fitting or external contact occurred.

## Results

| Target mass ratio q | Fit vs NR | Fit vs LAL | NR vs LAL | Interpretation |
|---:|---:|---:|---:|---|
| 3.5 | 0.5304% | 0.6950% | 0.4206% | Interpolation |
| 4.5 | 0.6095% | 0.4695% | 0.4224% | Interpolation |
| 5.5 | 0.6764% | 0.4817% | 0.5302% | Interpolation |
| 7 | 0.7484% | 0.5777% | 0.4682% | Interpolation |
| 10 | 1.1114% | 2.2101% | 1.2479% | Extrapolation beyond training q≤8 |

Each percentage is the relative amplitude L2 difference `norm(A-B)/norm(B)`, on the case's native frequency grid over `0.95*f_ring_33 ≤ Mf ≤ 0.3`. These are nonspinning `(3,3)` **ringdown amplitude** results. They do not measure phase accuracy, complex-waveform mismatch, or detector-weighted mismatch. Small spin residuals in the NR metadata are approximated as zero for LAL. Actual mass ratios appear in [verification.json](verification.json); the q≈10 case is q=9.98950857.

LAL 7.7.0 / LALSimulation 6.2.0 uses the public IMRPhenomXHM one-mode frequency-sequence API, release **122022**, with positive-frequency mode argument `(3,-3)`. The calibration SHA remains `ade377c62a57a5d9cff513e34f22b0fe43677177a0c1577a9a4af5ad3b8aa561`.

Combined with the earlier q≈3 and q≈6 holdouts, six tested cases inside the training range show subpercent agreement. This supports the tested slice, not continuous coverage of the entire parameter domain. SXS IDs are disjoint from our prior simulations; this does not establish independence from LAL's original training catalog or from related simulation families.

The [full comparison plot](validation/nr33-external-comparison.png) shows large fractional differences in the very weak high-frequency tails. Integrated subpercent errors are not pointwise bounds.

## Coverage and quality

The [protocol](protocol.json) fixed eight targets before waveform downloads. Every new primary is a holdout; there are no new training cases. The original per-simulation numerical quality policy was reused unchanged.

| Target q | SXS BBH ID | Primary | Post-peak duration M | Extraction L2 | Resolution L2 | Outcome |
|---:|---|---|---:|---:|---:|---|
| 1.2 | 0198 | Lev5/N3 | 297.676 | 2.5684% | 7.4914% | Fails both 1% gates; excluded from prediction/LAL comparison |
| 1.8 | — | — | — | — | — | No candidate passes the catalog metadata filters |
| 2.5 | — | — | — | — | — | No candidate passes the catalog metadata filters |
| 3.5 | 2483 | Lev5/N3 | 161.283 | 0.1944% | 0.0538% | Pass |
| 4.5 | 2484 | Lev5/N3 | 152.195 | 0.1741% | 0.2632% | Pass |
| 5.5 | 2486 | Lev5/N3 | 150.088 | 0.1680% | 0.1511% | Pass |
| 7 | 2491 | Lev5/N3 | 152.489 | 0.2261% | 0.2095% | Pass |
| 10 | 0185 | Lev3/N3 | 413.871 | 0.1619% | 0.1026% | Pass |

Five of eight targets have qualified waveforms. The q≈5.5 duration passes narrowly, by 0.088M. The largest preprocessing-variation L2 among accepted cases is 0.00119%, below the fixed 0.2% threshold. The highest-resolution reference spin and eccentricity gates remain `1e-4` and `1e-3`.

[Discovery](discovery.md) records all 181 ranked catalog entries. Catalog metadata are a preliminary filter; exact highest-resolution metadata were fetched for the six selected candidates. The absent targets are gaps in this bounded catalog search. They are not proven absent from every NR source. No candidate was replaced or selected using fit/LAL residuals. The unused q≈4.5 alternative was not advanced because the first candidate passed.

## Provenance incident and replay

The metadata agent inadvertently rewrote the inventory while the first preparation job ran. The original inventory bytes, SHA `dfe5c6…`, could not be recovered. The original selection, manifest, quality report, and available replacement snapshot are retained. **The original full-inventory freeze is incomplete.**

We repaired reproducibility through an explicitly labeled replay, without changing the selection:

- All **181 candidate-order entries and 24 selection rows** exactly match the selection frozen before waveform downloads.
- All **72 selected source-file tuples** (filename, URL, MD5, size) in the replacement inventory match the original prepared manifest.
- Preparation was repeated for all 24 waveforms, including the failed q≈1.2 case, into a new directory.
- All replay NPZ bytes, decoded arrays, metadata, and numerical quality results are identical to the original run.

The replay has a consistent selection/inventory/manifest hash chain. It does not recover the missing original inventory. [Independent incident audit](inventory-incident-audit.md), [replay selection](selection-replay.json), and [machine verification](verification.json) preserve this distinction. The calibration, protocol, and scientific selection remained unchanged. No LAL validation occurred until after the replay and [external-input freeze](validation-freeze.json).

## Verification and reproduction

**51 focused tests pass.** The audit checks 72 raw files and 24 waveforms. All 139 runtime files, 37 historical artifacts, 15 v2 artifacts, 32 v3 artifacts, four existing pipeline files, and `LICENSE` match the starting snapshot. The new adapters reuse the existing numerical screen and frozen prediction formula; [adapter review](adapter-review.md) and [scientific audit](final-scientific-audit.md) cover these boundaries.

Exact preparation commands and completed job status are in [prepare-job.json](prepare-job.json) and [prepare-replay-job.json](prepare-replay-job.json). The prediction/LAL command is in [the validation report](validation/nr33-external-validation.json). Raw/NPZ caches are outside the repository under `/tmp/torchwave-nr33-expanded`; retain them or re-fetch the checksum-pinned sources for reproduction. Use new output paths when repeating the adapters.

```sh
PYTHONPATH=src python -m scripts.screen_nr33_external_validation \
  --manifest docs/reviews/2026-09-15-independent-calibration-expanded/nr33-manifest-replay.json \
  --protocol docs/reviews/2026-09-15-independent-calibration-expanded/protocol.json \
  --expected-protocol-sha256 e337fac253a230da3250593d0edfad8bcd2a9cd7540d0e6f811010d9c8c1f67a \
  --output /tmp/nr33-expanded-recheck/quality.json \
  --qualified-output /tmp/nr33-expanded-recheck/qualified.json
```

The MIT source license is unchanged; [data attribution](DATA_ATTRIBUTION.md) is recorded separately. No runtime promotion occurred. Next work is converged data at missing targets and an explicit phase/time convention for complex `(3,3)` validation, followed by spin and other-mode calibration. [Remaining scope](remaining-parity-scope.md) identifies the required data and runtime interfaces.

# Independent benchmark audit

**Status: PASS.** Validation command: `python scripts/audit_nr33_phase_benchmark.py --output docs/reviews/2026-09-15-independent-phase-benchmark/results/audit-results.json`.

The replay uses direct NumPy/SciPy calculations on the seven saved NPZ inputs. It does not import the frozen metric implementation, call LAL, fit calibration parameters, or tune thresholds. The runner computes its PASS/FAIL status from the checks below, verifies every numeric NPZ array and reported NPZ SHA256, and exits nonzero on failure.

- All 7 required holdouts are present and report `phase_quality_pass=true`.
- Saved frequencies, complex strains, quadrature weights, and masks are finite; all quadrature weights and PSD values are positive. The minimum phase-mask retained reference norm fraction is `0.999999468173` (not exactly one).
- Independent weighted constant and affine phase fits reproduce every saved NR-control and LAL raw/constant/affine phase RMS and full-band relative L2 metric to at most `6.601e-16`. The largest NR-control affine residuals are `0.00304304` rad and `0.00516899`.
- Phase RMS uses the preregistered NR amplitude support and weights `quadrature_weight * |reference|²`; complex L2 uses the full common band, so weak tails remain represented in waveform error.
- Direct replay of both white and PSD-weighted overlap at all 21 saved time shifts agrees to at most `2.220e-16`. All white and PSD maxima are interior. A 4001-point independent dense grid for SXS:BBH:2265 at 20 Msun refined to `dt=-0.721335300`, overlap `0.999987889655`, consistent with the saved interior maximum.
- `new_calibration=false` and `runtime_promoted=false`; this replay did not mutate calibration state. Direct affine fit coefficients agree with saved coefficients within `7.39e-12`, and recomputed quality-gate booleans match all saved flags.
- Negative control: a copied report with one saved metric increased by `0.1` returned `status=FAIL` and exit code `1`; the repository report was untouched.

No metric or numerical-consistency blocker was found. This file does not assess repository provenance or source-file hashes; those checks are handled separately by the root audit.

Final coordinator replay also checks both actual optimizer time bounds, LAL affine coefficients, optimizer success, all primary holdout splits, aggregate required gates, and the overlap replay tolerance. [Real audit log](../logs/audit.log); [negative-control record](../logs/audit-negative.json).

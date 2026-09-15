# Independent parity audit and initial repair pass

12 September 2026. Starting revision: `84ef9b3`. Results below describe the
uncommitted working-tree changes from this repair pass. Full LAL parity is
**not achieved**.

## Result

- The ordinary test suite passes: **383 passed, 2 warnings in 28.18 s**. The
  warnings explicitly identify the incomplete experimental MSA implementation.
- The strict catalog fails: **11/21 cases pass, 6 fail numerically, and 4 lack
  independent waveform references**. TaylorF2_HM's passing carrier case does
  not validate its higher modes, leaving **5 model-level reference gaps**.
- The separate strict holdouts fail: **4/13 cases pass**. These are the two
  XAS cases and the two NRTidal cases.
- Replaying the same final harness against the starting code gives the same
  pass counts. No strict case was newly fixed or newly broken by this pass.
  Several numerical errors decreased; some increased, as shown below.

The old 21/21 result was not a strict external-parity certificate: several
models used relaxed thresholds, four cases used non-independent references,
and TaylorF2_HM tested only `(2,2)`. The old result is retained as historical
evidence in `verification/baseline-reported-catalog.json`, not relabeled as a
strict result. The manifest now distinguishes tested cases, parity gaps and
missing independent references, and no longer asserts 100% catalog parity.

## Implemented changes

Three fresh-context implementers worked with disjoint file ownership. Their
source records, derivations, equation tests and unresolved dependencies are
linked here. The primary agent reviewed their diffs and ran external binary
verification separately.

| Area | Changes | Detailed record |
| --- | --- | --- |
| PhenomD | Stable peak-frequency expression and analytic merger-amplitude derivative for C1 matching; independent QNM data checks without changing its calibration coefficients. | [Baseline](baseline.md) |
| HM/XHM | Published HM transition frequencies and missing Eq. (9) ringdown-phase modification; consistent carrier quadrupole QNMs; SPA amplitude factors; XHM time/phase covariance, equal-mass body exchange and finite boundary gradients. | [Higher modes](higher-modes.md) |
| Pv2/XP/XPHM | Effective-spin geometry, Wigner/projection signs, angular-momentum order, body ordering, DC/batching behavior, mode azimuth/SPA factors; experimental ODE reference anchoring and minimal-rotation sign. | [Precession](precession.md) |
| NSBH | Dimensionally consistent amplitude-window widths and BH-spin disruption variables from the primary model equations. | [NSBH](nsbh.md) |
| Verification | Uniform strict gates; rejection of weakened thresholds or altered frozen specifications; independent-oracle requirements; 13 distinct fixed holdouts and CI reporting. | [Harness](../../../tests/parity_harness.py) |

The HM amplitude adjustment implements the alternative prescription stated
after London Eq. (4); its compatibility with the target binary's prescription
remains unresolved. The changes are equation-supported reconstruction work,
not a claim of monotonic improvement against LAL.

## Matched before/after measurements

Numbers below are the maximum unaligned complex relative L2 error over plus
and cross. They are dimensionless fractions, not percentages. Acceptance also
checks relative norm, flat/PSD mismatches and detector projections. All direct
L2/norm ceilings are `1e-6`; all mismatch ceilings are `1e-8`. No amplitude,
time or phase alignment is used to erase raw residuals. Optimized matches are
additional diagnostics and cannot replace these bounds.

Both sides use identical case parameters, frequency grids, external binary
versions and the final strict harness. The baseline was a temporary export of
`84ef9b3` source; the current worktree was not reset.

### Canonical models touched by this pass

| Case | Before | After | Strict result |
| --- | ---: | ---: | --- |
| IMRPhenomD_diagnostic | 4.94767e-07 | 4.94765e-07 | PASSED |
| IMRPhenomHM_diagnostic | 0.00982258 | 0.0376679 | FAILED |
| IMRPhenomXHM_diagnostic | 0.0981123 | 0.0926912 | FAILED |
| IMRPhenomPv2_diagnostic | 0.304654 | 0.257234 | FAILED |
| IMRPhenomXP_diagnostic | 0.180301 | 0.0754449 | FAILED |
| IMRPhenomXPHM_diagnostic | 0.194955 | 0.0819732 | FAILED |
| IMRPhenomNSBH_diagnostic | 9.77268e-06 | 2.95419e-06 | FAILED |

### All fixed holdouts

| Case | Before | After | Strict result |
| --- | ---: | ---: | --- |
| IMRPhenomD_holdout_0 | 0.000118628 | 0.000118628 | FAILED |
| IMRPhenomD_holdout_1 | 8.9629e-07 | 8.96288e-07 | FAILED |
| IMRPhenomD_holdout_2 | 0.000432261 | 0.000432261 | FAILED |
| IMRPhenomXAS_holdout_3 | 3.83807e-14 | 3.83807e-14 | PASSED |
| IMRPhenomXAS_holdout_4 | 9.49094e-09 | 9.49094e-09 | PASSED |
| IMRPhenomHM_holdout_5 | 0.0691883 | 0.0518421 | FAILED |
| IMRPhenomXHM_holdout_6 | 0.238816 | 0.233172 | FAILED |
| IMRPhenomD_NRTidal_holdout | 1.60214e-07 | 1.60214e-07 | PASSED |
| IMRPhenomD_NRTidalv2_holdout | 1.57588e-07 | 1.57588e-07 | PASSED |
| IMRPhenomNSBH_holdout | 0.0124657 | 0.00132946 | FAILED |
| IMRPhenomPv2_holdout | 1.39985 | 1.41831 | FAILED |
| IMRPhenomXP_holdout | 0.671627 | 0.384571 | FAILED |
| IMRPhenomXPHM_holdout | 0.715531 | 0.405437 | FAILED |

HM's canonical error increases from about `0.00982` to `0.03767`, while its
holdout improves. Pv2's holdout increases from `1.39985` to `1.41831`; XP and
XPHM improve substantially but remain far outside strict bounds. PhenomD's
largest holdout error remains about `4.32e-4`. Its 23/21 holdout passes the
polarization L2 bound but fails a detector projection at `1.02137e-6`.

The NSBH holdout improves from `0.012466` to `0.001329`. An earlier progress
message incorrectly called this a regression by comparing it with a different
Lambda=600 probe. The matched comparison above uses Lambda=2600 on both sides.
Rechecking the original Lambda=600 probe also improves, from `8.11840e-6` to
`5.67460e-6`, while still failing. Its full result is recorded separately in
`verification/nsbh-original-probe-after.json`.

## Remaining implementation queue

1. **Carrier and calibration provenance:** resolve PhenomD's noncanonical
   residuals, shared remnant/time/phase conventions, and independently source
   the remaining inherited high-precision calibration assets. Matching an
   external number does not establish its origin.
2. **Precession prescriptions:** implement complete two-spin MSA with nutation
   and oscillatory corrections; reconcile the published defaults/version
   semantics; derive prescription-specific Pv2/XP frames and precession-aware
   remnant quantities. Experimental ODE evolution remains a separate incomplete
   approximation. See the explicit missing terms and options in the precession
   record.
3. **Higher-mode reconstruction:** complete XHM PN phase corrections,
   derivative/alignment joins, fitted-amplitude normalization and collocation
   rules, and spherical/spheroidal 32 mixing. Resolve HM's inherited phase
   factors, reference term and 43 spin truncation. Source each missing author
   coefficient independently before accepting it.
4. **NSBH model definition:** resolve remnant/classification coupling,
   phase/reference conventions, the printed low-Lambda discontinuity, PN spin
   reduction and remnant-fit provenance. Current improvements do not close
   its strict error budget.
5. **Independent reference coverage:** define and verify TaylorF2 higher modes,
   full Kerr-ringdown conventions, and specific calibrated EOB target versions.
   The native EOB prototypes and their surrogate are not equivalent to a LAL
   model merely because their names resemble one. See [coverage gaps](independent-coverage.md).

This queue includes substantial model reconstruction and unavailable or
unverified source data, beyond the bounded corrections completed here. It must
not be treated as solved by lowering tolerances or by a self-comparison.

## Reproduction and evidence

Host `RONI2`, cwd `/Users/xangma/repos/torchwave`; CPU float64/complex128.
Python 3.13.9, PyTorch 2.9.1, NumPy 2.3.5, LAL 7.7.0, LALSimulation 6.2.0,
PyCBC 2.10.0. Commands used the interpreter
`/Users/xangma/miniconda3/envs/pycbcorig313/bin/python` with
`PYTHONPATH=src:tests` and `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`:

```sh
python -m pytest -q
python tests/parity_harness.py --strict --json-out docs/reviews/2026-09-12-clean-room/verification/catalog.json
python tests/parity_harness.py --scope=holdout --strict --json-out docs/reviews/2026-09-12-clean-room/verification/holdouts.json
```

The commands exit 0, 1 and 1 respectively. Full logs, exact process commands,
environment records, per-case parameters and metrics are in [verification](verification/).
`baseline-strict-catalog.json` and `baseline-strict-holdout.json` use the final
harness with starting source; `baseline-comparison-run.json` records that setup.
`worktree-sha256.json` identifies the changed source, tests and documentation.
`git diff --check` passes. All verification processes and implementers have
finished; no commit or pull request was created.

## Clean-room boundary and limits

Implementers were instructed to use primary papers and author mathematical
data, avoid external implementation source/translated ports, and exclude old
source audits and the existing source-derived calibration specification.
The [work plan](work-plan.md) records this instruction boundary; it was not
enforced by operating-system isolation. Papers/assets are cited with versions,
hashes and equation locations in the individual records.

The precession implementer disclosed unsolicited source snippets appearing in
an initial broad search; those results were not opened or used. Its searches
were subsequently restricted to primary papers. A reverted frame experiment
was restored from the current starting source, not derived from external code.
Inherited TorchWave source remained available and its provenance has not been
certified. Therefore this pass cannot certify the entire repository's prior
clean-room or licensing claims. No external implementation code, binary-fitted
coefficient adjustment or runtime waveform-oracle dependency was added.

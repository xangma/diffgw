# Parity without upstream contact

Date: 2026-09-15. Three source-isolated Luna agents performed bounded
investigations; the coordinator reviewed their methods and independently
checked material evidence. Spark was unavailable in this session.

## Acceptance baseline

The immutable snapshot passed **17/21 catalog and 11/13 holdout cases**.
Inventory and manifest checks passed, and source hashes remained stable.
The six failures remain XHM and XPHM in both scopes, plus SEOBNR_ODE and
TEOBResumS in the catalog. Previously inspected holdouts are regressions,
not unseen validation.

Evidence: [catalog](verification/baseline/catalog.json),
[holdout](verification/baseline/holdout.json),
[commands, environment and source hashes](verification/baseline/source-record.json).

## Findings

### SEOBNR: trajectory substitution narrows the discrepancy

On 291 strictly pre-attachment samples, passing clean and public trajectory
states through the same native factorized-mode and NQC equations changes
relative complex L2 against the public mode from `5.41750455e-5` to
`2.18544424e-5` (59.66% lower). The complete common fine time axis is identical;
both paths use the same targets and captured clean event time. Historical
clean base-mode replay agrees with recomputation to `4.46e-15`.

Trajectory state differences therefore contribute, but this substitution
does not identify an implementable fix. Event timing, target precision and
mode conventions still need independent specification. Public states remain
diagnostic inputs; no public-state provider or correction enters the runtime.

Evidence: [experiment](seob-first-divergence.json),
[paired arrays](seob-first-divergence-arrays.npz),
[independent metric and hash verification](verification/seob-coordinator.json).

### TEOB: processing amplifies small tolerance-dependent changes

Three solver-tolerance levels reuse one postadiabatic (PA) handoff. Between
the two tighter levels, the pre-NQC mode changes by relative L2 `2.68e-8`,
while the processed mode changes by `2.16e-3` and attachment time shifts by
`-0.0274 M`. NQC is the non-quasi-circular correction applied near merger.
Full-mode error against the public reference is nonmonotonic across levels:
`1.7625e-3`, `4.0396e-3`, `2.0543e-3`.

This directs investigation to amplification during NQC and attachment
processing. It does not establish solver convergence or exclude derivative
and interpolation amplification of integration errors. A control holding
event time and auxiliary grid fixed was not run: the current diagnostic API
derives both internally. This approximately 14.45 Hz time-domain experiment
is distinct from the earlier 20 Hz frequency-domain comparison.

Evidence: [reviewed conclusion](teob-ode-convergence-clean/conclusion.md),
[successive-level report](teob-ode-convergence-clean/reviewed-report.json),
[independent metric and hash verification](verification/teob-coordinator.json).
The [initial NQC comparison](teob-nqc-attribution-clean/conclusion.md) shows
the correction improves agreement; it does not isolate the remaining cause.

### XHM/XPHM: current calibration remains unavailable in checked sources

The metadata check identifies the arXiv `2001.10914v2` ancillary archive,
including 37 amplitude-fit and 33 phase-fit files. Its hash and inventory
were independently reproduced. This paper-era artifact does not establish
the current `122022` calibration or an admissible redistribution basis.
The search was bounded; it does not prove no suitable release exists elsewhere.

The new checker requires an expected digest before declaring identity
verified. It neither extracts nor executes ancillary code. See the
[source review and limitations](xhm-no-contact-route.md) and
[verification](verification/xhm-archive.json).

## MIT boundary and next work

The new diagnostic code is independent. This pass does not clear inherited
PhenomD/XAS/XHM calibration provenance or official ROM data terms. Numerical
passes and absence of bundled upstream files are insufficient to clear those
existing questions. See the [provenance ledger](../../provenance_ledger.md).

Next numerical controls, in order:

1. TEOB: build a diagnostic assembly that independently fixes event time and
   auxiliary grid while replaying the saved trajectories. Compare derivatives,
   solved NQC coefficients and attachment outputs to locate the amplification.
2. SEOB: independently specify trajectory evolution and event conventions;
   keep targets and the mode pipeline fixed while varying one convention at
   a time. Require a fresh native waveform improvement before promotion.

For XHM/XPHM, exact current-release parity still needs release-matched
calibration with admissible provenance. A new calibration from suitable
public data is a separate scientific model and requires its own validation;
it cannot be presented as demonstrated LAL parity.

## Verification

All **13 focused diagnostic/checker tests pass**; see the
[test log](verification/focused-tests.log). Coordinator checks reproduced
SEOB/TEOB metrics and source hashes and the XHM archive identity/inventory.
All **139 frozen acceptance inputs remain identical** to the completed strict
baseline; see [source identity](verification/runtime-baseline-identity.json).
No new strict pass was obtained, and these diagnostics do not certify MIT
provenance for the existing runtime.

No upstream contact, runtime default change, acceptance-threshold change,
commit or publication was made in this pass.

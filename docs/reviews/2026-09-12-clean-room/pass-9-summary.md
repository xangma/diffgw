# Pass 9: parameter and precision consistency

Three GPT-5.6 Luna implementers completed bounded native-code reviews. The
primary agent reviewed their diffs and performed independent binary verification.
Existing calibration tables remain in use; the deferred NR run was not restarted.

- [PhenomD/NSBH](pass-9-baseline.md): preserve input precision in the peak-phase
  slope helper, fixing float32 batches unexpectedly producing complex128 outputs.
  The scalar/batch regression uses zero absolute tolerance and double-precision
  norm accumulation to avoid vacuous comparisons at physical strain amplitudes.
- [Precession](pass-9-precession.md): explicit `None` now selects the MSA
  evaluator's declared default 223. Method-specific wrapper defaults are unchanged.
- [Detectors](pass-9-detectors.md): an explicit sampling interval disambiguates
  a one-sample time-series call from the legacy scalar-interval overload.

Verification on `RONI2` passed **1,063 tests, with eight skipped, in 72.21 s**.
Strict catalog acceptance remains **11/21**, and frozen holdouts remain **4/13**.
Polarization and detector metrics are unchanged from pass 7. These fixes improve
API consistency; they add no strict passing cases. Uniform FP64 thresholds and
frozen case definitions remain unchanged.

The [run record](verification/pass-9/verification-run.json) captures commands,
PIDs, timestamps, exit codes and stable source hashes. The ordinary suite exited
0; both strict suites exited 1 because the existing gaps remain. All verification
processes completed. See [catalog](verification/pass-9/catalog.json),
[holdouts](verification/pass-9/holdout.json) and
[comparison](verification/pass-9/comparison.json).

The follow-up remnant review distinguishes the paper-specified NNLO prescription
from unresolved MSA evaluation-state and ODE/fallback policies. NNLO remnant
implementation is pursued separately in pass 10; it is not part of this snapshot.

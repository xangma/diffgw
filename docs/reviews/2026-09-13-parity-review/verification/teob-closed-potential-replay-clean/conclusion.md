# Closed orbital-potential replay

Luna independently reduced the published logarithmic PN coefficient-matching
system to a closed `P^1_5` rational function. The coordinator reviewed the
algebra; tests compare values and radial derivatives with an independent
80-digit linear solve. The helper remains opt-in and is substituted only by
the explicit diagnostic wrapper. See the [derivation](../teob-orbital-potential-closed-clean/derivation.md).

The series is documented in Nagar et al., [1506.08457v2](https://arxiv.org/abs/1506.08457v2),
with 4PN terms from Bini–Damour, [1305.4884](https://arxiv.org/abs/1305.4884),
and 5PN logarithms from Damour–Jaranowski–Schäfer,
[1502.07245](https://arxiv.org/abs/1502.07245). No reference implementation or
source-only coefficient table was supplied to the implementer.

## Whole-waveform result

| PA policy | Existing potential | Closed potential | Change |
| --- | ---: | ---: | ---: |
| Analytic | 3.2708438016984477e-4 | 3.36530021663527e-4 | 2.89% worse |
| Direct quadratic | 4.153177115176164e-4 | 4.433664330821569e-4 | 6.75% worse |

These are combined-polarization FD relative L2 errors on 2,009 exact bins,
20–1024 Hz, for the same 30+20 nonspinning input. No time/phase alignment,
fitting or threshold change was used. Both fresh public arrays are bitwise
equal to the frozen public reference. Runs completed in 32.58 and 32.46 seconds.
Root recomputed each metric and checked arrays and hashes:
[analytic verification](root-verification.json),
[direct-quadratic verification](direct-quadratic/root-verification.json).

The later complex-input rejection fix touched only `_inputs`. The verification
records distinguish the replayed and final helper hashes; 211 valid-input
values and gradients were bitwise equal before/after that guard. Waveforms
were not rerun after this validation-only edit.

## Validation and decision

The final root run passed **60 tests** across the potential, vacuum driver,
continuation, merger NQC, ringdown assembly, direct quadratic and MIT packaging.
Separate wrapper smoke checks verified restoration on success/failure and that
`--help` or a pre-report failure cannot relabel an older report.

Neither potential substitution is promoted. These results do not support
replacing the linear solve as a parity fix. Frozen acceptance remains
17/21 catalog and 11/13 holdout; no new full strict suite was run.
No dataset or reference source was added to the package, and license metadata
was not changed. The broader inherited calibration-provenance limitations in
the implementation status remain unresolved.

# Pass 12: bounded XHM correction and explicit remnant selection

Completed 13 September 2026 with two fresh-context **GPT-5.6 Luna** subagents
and parent integration/review. **Full LAL parity remains unfinished. This pass
adds no strict passing case, and no frozen parity error changes.**

## Retained changes

- XHM removes both intermediate amplitude nodes in the unconditional published
  parameter regions, selecting its cubic boundary reconstruction. A production
  mixed-batch test proves the affected row ignores an altered inner fit value;
  a control row changes. Gradients remain finite away from region boundaries.
- XP/XPHM accepts `final_spin_mod=0` independently of the Euler-angle version,
  forwarding the existing NNLO remnant to XAS/XHM even with MSA angles.
  Omitting the keyword preserves prior behavior. Unsupported remnant choices
  reject explicitly. Full MSA FinalSpinMod3 remains unimplemented.

The [patch](verification/pass-12/changes-from-pass11.patch) contains three
runtime files and two new test files. Earlier dirty work is preserved.
No calibration table, acceptance case, grid or threshold was changed.

Parent review narrowed both candidates before final verification. XHM's
proposed absolute thresholds had no demonstrated conversion from strain
amplitude to the native ratio, and its conditional veto union did not preserve
ordered node-survival semantics. Those checks and the new quartic/linear
systems were removed. The proposed MSA remnant combined initial- and final-mass
normalizations and failed the aligned Kerr-spin limit; it was removed.
The retained precession change exposes the already-derived NNLO formula.

See [XHM derivation and exclusions](pass-12-higher-modes.md),
[precession source review](pass-12-precession.md) and
[primary source audit](verification/pass-12/source-audit.json).

## Verification

| Check | Result |
| --- | --- |
| Focused integration | 72 passed |
| Full ordinary suite | **1,112 passed, 8 skipped, 76.14 seconds** |
| Frozen strict catalog | **11/21 passed**; 9 numerical failures, 1 unresolved reference |
| Frozen strict holdouts | **4/13 passed** |
| Separate explicit component checks | **3/34 passed**, unchanged |
| Mode-sum closure controls | **4/4 passed**, unchanged |

All frozen catalog, holdout and supplemental component metrics are identical
to their prior values. The new XHM parameter-region behavior is exercised by
its independent native production test; the existing fixed parity inventory
does not gain a passing case from this change. New remnant selection is opt-in
and does not affect default comparisons. Twenty-five added test cases validate
these changes and their boundaries; they are not additional parity successes.

The [comparison](verification/pass-12/comparison.json) asserts unchanged case
inventories, parameters, grids, gates and verifier definitions. The full suite
has six CUDA skips and two generation-only gmpy2 skips; MPS tests pass.
`git diff --check` passes. All 171 runtime/test/script/manifest hashes remained
stable during verification. A separately hashed manifest update records results.

A separate [public-default control](verification/pass-12/public-defaults.json)
finds bitwise equality between LAL defaults and explicit 223/FinalSpinMod3 for
all four fixed XP/XPHM cases. Explicit 102/0 and 320/2 differ. Lookup defaults
300/4 also reproduce the default outputs. This is finite-case output evidence,
not proof of internal dispatch. The primary Pratten paper independently
documents 223/3/convention1. Native NNLO102 and partial MSA are distinct
prescriptions; switching angle defaults alone would not finish compatibility.
No binary residuals were given to the Luna implementers.

## Remaining work and process correction

The [remaining-work assessment](pass-12-remaining-work.md) separates Phenom
corrections from the larger unfinished EOB, ROM and tidal implementations.
The immediate milestone is a complete compatible Phenom waveform prescription:
resolve XHM fit/phase definitions and the MSA remnant convention from primary
evidence, then verify the full waveform. Repeated helper-only passes cannot
substitute for that milestone. The current case counts do not provide a
reliable completion percentage or time estimate.

Existing tables remain in use. The user-deferred NR job was not restarted or
scheduled. No commit or pull request was created.

The final verifier ran on RONI2 in `/Users/xangma/repos/torchwave`, using
`/Users/xangma/miniconda3/envs/pycbcorig313/bin/python` and the
[driver](verification/pass-12/verify.py), PID 78180, from 09:24:20 to
09:25:56 UTC. Test/catalog/holdout/component exits were 0/1/1/1; all children
finished. Exact commands, PIDs, logs and hashes are in the
[run record](verification/pass-12/verification-run.json).

Implementers were instructed to use primary papers, author mathematical data
and inherited native TorchWave code, excluding LAL/PyCBC implementation source,
translated ports and reference residuals. Parent public-binary verification
remained separate from their derivations. This is an instruction boundary;
it does not certify inherited provenance or provide OS-enforced isolation.

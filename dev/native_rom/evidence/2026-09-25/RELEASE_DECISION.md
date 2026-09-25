# Native SEOBNRv4 and proposed ROM: release decision (2026-09-25)

**Verdict: unqualified.** Keep the native generator and proposed independent ROM
on the development branch and outside the release catalog. Backend parity on
two rows is established, but it is not a physical-validity result.

The smallest existing prospectively fixed physical pass/fail check is the
[third five-case SXS plan](../2026-09-24/sxs_third_holdout_plan.json) with its
[weighted gate](../2026-09-24/sxs_third_holdout_weighted_plan.json): a complex
dominant-mode `h22` comparison with independent NR at total masses 10–200
solar masses in steps of one. Each selected case must have detector-weighted
overlap at least 0.99 after one bounded global time and phase shift, with no
amplitude or intrinsic-parameter fit. LAL supplies only the pinned noise PSD,
not waveform targets. This is a finite selected-case proxy, not a whole-domain
or fixed-chart full-complex-waveform release guarantee. The existing fixed-chart
complex waveform comparisons predeclare no physical pass threshold.

The [original frozen result](../2026-09-24/sxs_third_holdout_weighted.json)
fails one of five cases: SXS:BBH:2569 has minimum overlap
`0.9891417104357128` at 22 solar masses, against `0.9999241175571614` for
the low/high-resolution NR control. The numerical-convergence diagnostic
[for that case](../2026-09-24/sxs_2569_convergence.json) reports a
`6.65e-8` complex-waveform relative L2 change under tenfold tighter ODE
tolerances; this does not explain the overlap deficit. A fresh
[stored-array replay](sxs_third_holdout_weighted_replay.json) uses the same
frozen plans and input array hashes. Its five minimum overlaps agree with the
original within `3.4e-16`; it also fails 2569 at `0.9891417104357124`.
The replay ran on RONI2 using the local Python 3.13 environment rather than
the original Python 3.12 environment, so its binary fingerprints and runner
hash differ; it is a reproducibility check of stored data, not a new holdout.

**Current-candidate limitation:** the original anchored waveforms identify
generator source-manifest hash
`8ca6d152b4b07bb89bbbd97734d29814023148b79f0af448fda67079957ca532`,
whereas the current source snapshot hashes to
`2062f70072d9953a7f9b968336c80512b8034497eb2e4cb0fea7ad2633b7d2cf`.
Generator files changed between those snapshots. Replaying the stored arrays
cannot establish whether today's candidate passes or fails the NR gate.
The five NR cases have been inspected, so reusing them would be a diagnostic,
not an untouched release holdout.

The [source specification](../../reference/SPEC.md) remains incomplete: six
of 42 manifest components are `pending`, including flux, QNM interpolation,
ODE initial conditions, TD/FD conventions, and the ROM construction. The
published model does not supply independently derived ROM bases and spline
tables, and no qualified generator/training set or independent ROM holdout
exists. Full-waveform gradients and domain-wide convergence remain open.

Two different release claims must stay separate:

- **Historical SEOBNRv4 fidelity:** a port may be described as faithful only
  after a frozen, independently checked full-waveform source-fidelity and
  backend contract is met. It would need to disclose the observed sub-0.99 NR
  proxy result rather than imply that historical parity guarantees that gate.
- **Physical NR accuracy at the 0.99 gate:** a release under this claim needs a
  prospectively frozen current candidate, a new untouched independent NR
  holdout with valid start frequency and cap, and full-complex waveform,
  convergence, and applicable gradient checks. If a faithful historical model
  cannot pass the gate, a calibrated successor needs its own model identity.

The shortest useful next step is to finish the missing clean-room component
provenance and freeze one generator revision and its numerical/gradient
contract, then predeclare a fixed-chart complex-waveform threshold and a
detector-weighted overlap gate on new untouched NR rows before running them.
Only after that generator passes should independent ROM data be trained and
tested on reserved holdouts. Neither criterion currently permits release.

The completed replay ran on host RONI2, cwd
`/Users/xangma/.codex/worktrees/native-seobnr-rom/diffgw`, shell job/session
`67491`, and exited zero. Its log is
`/tmp/diffgw-sxs-third-holdout-weighted-replay.log`; no process remains or next
check is needed. The command was:

```sh
env PYTHONPATH=src:. /Users/xangma/repos/diffgw/.venv/bin/python \
  -m dev.native_rom.measure_sxs_third_holdout_weighted \
  --case-plan dev/native_rom/evidence/2026-09-24/sxs_third_holdout_plan.json \
  --weighted-plan dev/native_rom/evidence/2026-09-24/sxs_third_holdout_weighted_plan.json \
  --comparison dev/native_rom/evidence/2026-09-24/sxs_third_holdout_comparison.json \
  --decoded-dir /tmp/diffgw-sxs-third-holdout/decoded \
  --archive-dir /tmp/diffgw-sxs-third-holdout/archive \
  --json-out dev/native_rom/evidence/2026-09-25/sxs_third_holdout_weighted_replay.json \
  > /tmp/diffgw-sxs-third-holdout-weighted-replay.log 2>&1
```

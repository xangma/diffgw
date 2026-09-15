# TEOB trajectory controls: correction and final interpolation check

## Outcome

**The proposed common-grid/fixed-event replay had already been done.** My
previous recommendation missed the later
[strategy experiment](../2026-09-15-parity-strategy-experiment/eob/README.md).
That experiment held the actual selected NQC time and ringdown attachment
fixed, although its earlier joined interpolation knots still differed.
This report supersedes the “not yet run” statement in the
[target audit](../2026-09-15-teob-target-audit/README.md); that historical report
and its manifest remain unchanged.

The missing final-grid controls are now complete. Holding all final assembly
grids fixed changes the two-trajectory difference only slightly. It does
**not close the public-waveform residual or justify a runtime change**.

## Results

These are the historical nonspinning 30+20 solar-mass trajectories, with the
original radiation policy, 2017 NQC targets and 0.5 M TD scoring grid. Each row
compares the same two cached tolerance levels. All 24,938 public samples are
scored, without fitted time/phase alignment.

| Postprocessing control | Trajectory-to-trajectory relative L2 |
|---|---:|
| Native grids and events | 2.16470e-3 |
| Historical common auxiliary grid and fixed events | 9.03979e-7 |
| Same, authoritative 0.1 M NQC derivative spacing | 9.03979e-7 |
| Also use the `tight_0p1` joined/extended grid | 8.98926e-7 |
| Also use the `tight_0p01` joined/extended grid | 8.94091e-7 |

The first common-grid replay already removed most of the native inter-level
spread. Adding fixed final interpolation grids changes either trajectory's
waveform by at most **9.34e-8** relative L2. Correcting the derivative scalar
from `0.1000000000003638` to `0.1` changes either waveform by **2.77e-12**.
Neither is a substantial remaining effect in this case.

Error against the public reference remains separate:

| Control | `tight_0p1` vs public | `tight_0p01` vs public |
|---|---:|---:|
| Native | 4.03964e-3 | 2.05432e-3 |
| Common grid/events, explicit spacing | 4.81192e-3 | 4.81278e-3 |
| All grids fixed to `tight_0p1` | 4.81192e-3 | 4.81278e-3 |
| All grids fixed to `tight_0p01` | 4.81192e-3 | 4.81278e-3 |

Common-grid/event controls deliberately change the numerical construction;
they are attribution probes, not improvements selected for public agreement.
Their small inter-trajectory difference does not establish that the common
result is correct.

The pairwise denominator here is the `tight_0p1` waveform norm. The historical
report used `tight_0p01`; independent validation reproduces the historical
values with that denominator too. Public comparisons always divide by the
public waveform norm.

## What was held fixed

The [protocol](protocol.md) preceded generated waveforms. Ten outputs comprise
five variants for each trajectory. The last two variants use opposite
reference prefixes, so the conclusion does not depend on selecting the
coarser or finer accepted grid.

- Common auxiliary grid: 200 knots, `12298.65990319574` to
  `12318.559903195739 M`.
- Requested NQC time: `12315.487269626203 M`; actual selected time:
  `12315.35990319574 M`.
- Ringdown peak: `12316.531451506382 M`; actual attachment:
  `12315.35990319574 M`.
- Both prefix controls have identical auxiliary, joined, extended, native
  output and public scoring grids across trajectories. Joined/extended lengths
  are 1089/2589 or 1573/3073 according to the chosen prefix.
- Changing only the prefix leaves all auxiliary inputs and NQC coefficients
  bitwise unchanged. The trajectory supplying the prefix retains its output
  bitwise. Candidate prefixes reject extrapolation and interpolate signed
  amplitude and lifted phase separately.

The original accepted-to-auxiliary spline knots remain trajectory-dependent.
Thus residual integration and interpolation errors, and their effects on NQC
coefficients, are still part of each trajectory approximation. This does not
isolate exact ODE solution error or establish asymptotic convergence.

## Verification and provenance

- **28 focused tests passed**, including independent SciPy interpolation,
  exact-knot preservation, extrapolation rejection, explicit-step precision,
  prefix identity, and existing NQC/ringdown controls.
- Native outputs reproduce the cached processed modes to at most **3.08e-16**
  relative L2. Historical pair metrics and public scores reproduce.
- The explicit-spacing diagnostic reproduces native NQC coefficients and
  original/auxiliary corrected arrays **bitwise** on each own native grid.
- Independent NumPy/SciPy validation verifies all ten output hashes and
  recalculates public/pairwise scores and grid/event identities. SciPy
  reconstruction of saved extended amplitude/phase agrees to **2.44e-14**
  relative L2; public-score differences are zero.
- All **130 runtime files**, **60 previous manifest entries**, and all **198
  recorded source/input paths** remain unchanged. No public waveform values
  enter construction, and no upstream implementation source was inspected.

The first preflight correctly stopped at a historical source-hash mismatch.
NQC and ringdown gained the already documented continuous-matching branch
after the old experiment. Current sources are pinned to the latest frozen
target-audit record; replay helper and trajectory driver retain their original
hashes. Numerical legacy reproduction establishes compatibility for these
controls, not identity of the old and current source files. The protocol
amendment and [failed preflight log](preflight-source-guard.log) preserve this.

Execution finished locally on **RONI2**, PID **78396**, in **1.64 seconds**, cwd
`/Users/xangma/repos/torchwave`. No jobs remain running. Run command:

```sh
PYTHONPATH=src OMP_NUM_THREADS=1 /Users/xangma/miniconda3/envs/pycbcorig313/bin/python scripts/teob_trajectory_grid_controls.py
```

See [run log](run.log), [full controls](controls.json),
[independent validation](artifact-validation.json), [test log](tests.log), and
[artifact manifest](artifact-manifest.json). Existing completed output is
protected against overwrite. Luna performed narrow, token-limited reviews;
the primary agent verified claims and ran the experiments.

## Decision for parity work

Close this historical replay branch: another prefix-grid or derivative-step
adjustment has little support from these results. The earlier continuous
matching option also already received strict FD testing and failed; do not
present it again as an untried route to parity.

The useful trajectory question belongs in the **current strict FD contract**:
measure convergence while preserving its physical inputs, radiation policy,
observer epoch, conditioning and exact comparison bins, and compare native
events with fixed sampled events. A new study must inventory existing runs
first, then regenerate only missing compatible tolerance levels and retain
derived quantities in its cache. The old r~20 TD results cannot stand in for
that test. PA handoff remains a separate factor, not something to vary in the
same initial comparison.

No current strict FD gate was rerun or changed here. The four previously
reported failures remain unresolved. This result closes a diagnostic gap;
it does not claim progress toward parity by lowering the wrong metric.

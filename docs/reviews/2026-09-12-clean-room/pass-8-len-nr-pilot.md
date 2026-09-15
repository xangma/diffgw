# Own numerical-relativity pilot on len

The user selected `len` on 2026-09-12 to execute the
[prepared NR pilot](pass-7-own-nr-pilot.md). This is an execution record for
simulation data generated with SpECTRE, not a claim of a new NR solver or a
completed waveform calibration.

The finite pilot completed successfully at 03:22:39 UTC (04:22 UK time) on
2026-09-13. The 32-PE P11 solve, P9/P11 initial-data comparison, binary
checkpoint/restart and final validation all passed. The v3 continuation took
2 h 33 min. The next 100 M Lev1 performance evolution ran on 48 workers from
06:46:48 to 07:42:59 UTC on 2026-09-13, then stopped at the user's request.
The user chose to defer NR work and use the existing calibration tables for now.
The full waveform campaign and new calibration remain deferred.

## User-requested deferral

At 07:42 UTC (08:42 UK time), the user requested: “Let's run this later, and use
the tables for now.” Stopping `torchwave-nr-performance-48pe.service` also stopped
its active solver child. Both services have PID 0; Docker confirms the solver
is no longer running and was not OOM-killed. The driver records `status: stopped`
and `Stop requested`. Systemd's exit-code result and Docker exit 137 reflect the
requested termination, not a completed or numerically failed evolution.

The last logged control-system time was approximately 2.878 M after 56 minutes.
This is partial progress, not a validated 100 M result. No evolution checkpoint
had been written; a later execution must start a new run from the saved
controlled initial data. Inputs, the pinned image, prior completed pilot evidence,
logs and approximately 258 MB of partial performance files are retained.
No future restart time was specified or scheduled. The automatic continuation
driver is stopped. The [operator stop record](verification/pass-8/provenance/performance-48pe/performance-48pe-operator-stop.json)
contains the terminal process state and retained-file inventory.

For now, use TorchWave's existing calibration-table path. The NR performance run
has not supplied any replacement coefficients. Table sources and existing
precision/provenance limits remain recorded in the
[calibration inventory](pass-7-independent-calibration-inventory.md); this choice
does not change the recorded model acceptance results.

## 100 M performance evolution on 48 cores — historical launch

The user authorized the pending run on 48 cores. The prepared next stage is a
100 M Lev1 evolution from the accepted controlled-P9 initial data, with outer
radius 1500 M and Bondi output radii 200, 300 and 400 M. The P11 comparison and
32-worker checkpoint/restart validation passed before this launch. This stage
measures sustained performance; it does not establish waveform convergence or
eccentricity control.

Read-only preflight found 213 GiB available RAM and 287 GiB free disk. Logical
CPUs 0–47 map to 48 distinct physical cores. The solver uses `+p48`, a Docker
48-CPU quota, CPU affinity 0–47 and a 160 GiB memory limit without additional
swap. The runtime is the repaired, pinned image
`sha256:0ba0f3bb70b90e83c9b50316ecd37f1b55af4537160eafbe3392a6f67c4e9d77`.

- Host: `len`.
- Campaign cwd: `/home/xangma/simulations/torchwave-nr/2026-09-12-pilot`.
- Solver cwd: `/home/xangma/simulations/torchwave-nr/2026-09-12-pilot/performance/Lev1/Segment_0000`
  (mounted as `/campaign/performance/Lev1/Segment_0000`).
- Driver: `torchwave-nr-performance-48pe.service`, initial PID `2599638`.
- Initial solver stage: `torchwave-nr-performance-48pe-segment-0000.service`;
  container init host PID `2600143`.
- Driver command: `/usr/bin/python3 -u /home/xangma/simulations/torchwave-nr/2026-09-12-pilot/scripts/run-performance-48pe.py`.
- Solver command: `EvolveGhBinaryBlackHole +p48 --input-file /campaign/performance/Lev1/Segment_0000/Inspiral.48pe.yaml`.
- Driver log: `/home/xangma/simulations/torchwave-nr/2026-09-12-pilot/logs/performance-48pe.log`.
- Solver log: `/home/xangma/simulations/torchwave-nr/2026-09-12-pilot/logs/performance-48pe-segment-0000.log`.
- State: `/home/xangma/simulations/torchwave-nr/2026-09-12-pilot/provenance/performance-48pe-state.json`.
- Original check policy: driver polled every 10 seconds; checkpoint was scheduled
  after about 11.5 wall-hours plus synchronization/write time. The user stopped
  the run before this checkpoint; no check or restart is now pending.
- Stop driver and its active child: `ssh len 'systemctl --user stop torchwave-nr-performance-48pe.service'`.

The original rendered input is preserved. `Inspiral.48pe.yaml` changes only the
wall-clock checkpoint interval (23.5 to 11.5 hours) and its trigger frequency
(every 10 slabs to every slab). The actual executable accepted both inputs with
`--check-options`. The runtime input SHA256 is
`ceada3a7aa09b5b4610c3d12d96521d96ce54caafdb0abf4095f8bc31727372d`.
The locked [launch plan](verification/pass-8/provenance/performance-48pe/performance-48pe-plan.json)
records helper/input hashes and limits; the
[job record](verification/pass-8/provenance/performance-48pe/performance-48pe-job.json)
contains the exact systemd command.

Each segment has a 12.5-hour watchdog. Successful checkpoint exits (code 2)
are validated for finite reductions and Bondi data, forward time progress and
nonempty checkpoint files with SHA256 hashes, then resumed in a new segment
directory using the same input and 48 workers. Final exit 0 must pass the
100 M completion checks. Other exit codes, OOM, failed validation or exhausted
resources stop the driver. It allows at most eight segments and 104 wall-hours;
all checkpoints remain on disk. Independent child service limits and systemd
dependencies also stop the solver if its driver is stopped.

Disk guards poll every 10 seconds: stop above 192 GiB of performance output or
below 80 GiB free space. These are stop thresholds, with room for writes between
checks. The previous 1 M pilot's approximately 12 GB output was dominated by
checkpoint and initial-volume data, so it is not a linear storage forecast.

Verification covered Python compilation, accepted exits 0/2, rejected exit 1,
watchdog termination, valid 100 M synthetic HDF5 data, and rejection of nonfinite
Bondi data. An independent helper review checked continuation and stop behavior.
The first live sample confirmed a running container with CPU affinity 0–47,
about 4709% CPU usage (47 cores) and no OOM. No sustained-throughput or finish-time
estimate is claimed from this startup sample.

## Allocation and isolation

Read-only checks confirmed an AMD Threadripper PRO 3995WX with 64 physical
cores / 128 logical CPUs and approximately 214 GiB available RAM. `/home`
initially had 331 GiB free; Docker's storage also resides on that filesystem.
Docker 25.0.2 and cgroup v2 are available; there is no host Slurm installation.

- Host: `len`, SSH alias `len`.
- Working directory: `/home/xangma/simulations/torchwave-nr/2026-09-12-pilot`.
- Initial CPU / memory limit: 24 CPUs / 96 GiB, without additional swap.
- First stage: validate the environment, generate equal-mass nonspinning XCTS
  initial data, and measure a finite short evolution before the full campaign.
- Existing host services are outside this campaign's resource controls.

## Pinned environment

The official Linux amd64 deployment image resolves to:

```text
sxscollaboration/spectre@sha256:698ea4fe0d997d66113a3ee7e869a736003577753dc70a62372000bc7436e658
```

It contains the XCTS solver and CCE tools. Binary evolution requires an
additional build. The source is the official release `v2026.09.05`, commit
`ce221e8b4bf8f4ca1d0acad98453aed5b454f469`. The image's precompiled header was
created for an Intel processor and cannot be reused for this AMD build. All
required SpECTRE targets were therefore rebuilt with an explicit
`-DOVERRIDE_ARCH=znver2` and `--clean-first`. The image supplies the dependency
environment. No calibration waveform catalog is an input to this setup.

## Image pull

The pull was launched at 2026-09-12 18:33:43 UTC as the user service
`torchwave-nr-image-pull.service`, initial PID `928085`:

```bash
systemd-run --user --unit=torchwave-nr-image-pull \
  --property=RuntimeMaxSec=2h \
  --working-directory=/home/xangma/simulations/torchwave-nr/2026-09-12-pilot \
  /bin/bash /home/xangma/simulations/torchwave-nr/2026-09-12-pilot/pull-image.sh
```

The script invokes `docker pull` with the fixed digest above. Its log is
`logs/image-pull.log`, and its full command record is
`provenance/image-pull-job.json` beneath the remote working directory.
Stop it with `systemctl --user stop torchwave-nr-image-pull.service` on len.

The image pull completed successfully. The following two failed build launches
are retained in the provenance record: an unavailable `/usr/bin/time` command
(exit 127), and the incompatible precompiled header (exit 2). Neither produced
simulation data.

## Clean build

The clean build used container `torchwave-nr-build-zen2`, user service
`torchwave-nr-build-len.service`, initial host PID `1175261`. Its command is
recorded in [build-len-job.json](verification/pass-8/provenance/build-len-job.json).
It runs the following inside `/work/spectre/build` in the pinned image:

```bash
cmake -S /work/spectre -B /work/spectre/build \
  -DOVERRIDE_ARCH=znver2 -DUSE_GIT_HOOKS=OFF
cmake --build /work/spectre/build --clean-first --target \
  SolveXcts cli EvolveGhBinaryBlackHole EvolveGhSingleBlackHole \
  CharacteristicExtract -j12
```

The container is limited to 24 CPUs, 96 GiB RAM, no extra swap, and no network.
The service has a four-hour outer limit. The remote log is
`/home/xangma/simulations/torchwave-nr/2026-09-12-pilot/logs/build-len.log`.
Stop it on len with `systemctl --user stop torchwave-nr-build-len.service`.

The clean build completed at 19:31:51 UTC with exit 0, no OOM, and elapsed
time 2932.436 s. All four executables and the CLI/Python bindings built
successfully. The stopped container was preserved as immutable image
`sha256:4d03ce10bf6deb922444b293b39e0c3753d92ac69c70f99ecb9cef49ba64b42f`.
Its [complete environment record](verification/pass-8/provenance/runtime-all-original-2-environment.json)
contains the executable hashes and successful binding/linkage checks.

## Simulation gates prepared during compilation

At 19:08 UTC the clean `SolveXcts` and CLI targets were complete, including
successful Python binding imports. They were frozen in a local image while the
remaining evolution targets continued compiling:

```text
sha256:506cbbdd23fe617f9dfbf0243cc99cdede63c277ceaa6be636928bd0ffc235cb
```

The [environment manifest](verification/pass-8/provenance/runtime-id-environment.json)
records packages, compiler settings, source status, linked libraries and the
rebuilt solver hash. Source status reports only a missing
`containers/Dockerfile.buildenv`, with no numerical source changes.

The smoke solve started at 19:08:55 UTC under
`torchwave-nr-id-smoke.service`, initial host PID `1878366`. It uses 8 CPUs,
32 GiB RAM, a 6900-second inner time limit, a 32 GiB output bound and a 50 GiB
minimum free-space guard. The full command is recorded in
[id-smoke-job.json](verification/pass-8/provenance/id-smoke-job.json):

```bash
spectre schedule /campaign/smoke/id/InitialData.yaml \
  --no-schedule --num-procs 8 --submit
```

Its container working directory is `/campaign/smoke/id`; the corresponding host
directory is `smoke/id` beneath the campaign root. The log is
`logs/id-smoke.log`. Stop it on len with
`systemctl --user stop torchwave-nr-id-smoke.service`.

The rendered smoke input targets q=1, zero spin, separation 15.366 M, at
initial-data L1/P5. It has no mass/spin correction loop or automatic evolution.
The finite PN orbital guesses are angular velocity
`0.01533628349983242` and radial expansion `-4.827331550791904e-05`.
An ancillary PN duration integration emitted an LSODA warning; its duration is
not used as a validated forecast. The original render and warning are retained.

The L1/P5 operational smoke completed with exit code 0 after 356.8917 s. Its
five Newton iterations were recorded; the final Newton row was iteration 5 with residual
`1.5499295579965185e-9`; peak cgroup memory was 9,085,911,040 bytes (8.462
GiB), and the output grew to 75,878,883 bytes (about 75.9 MB). The measured
horizon masses were `0.5001092653` and `0.5001092801`; spin norms were
`0.0013393658` and `0.0013389525`. This passes the initial-data operational
smoke only. The spin result does not meet the calibrated zero-spin target of
`5e-5`. It has not produced an evolution waveform.

The archived inputs, HDF5 outputs, and log are under
`verification/pass-8/inputs/smoke-id/` and `verification/pass-8/logs/`; their
hashes are recorded in `verification/pass-8/provenance/id-smoke-quality.json`.
The subsequent controlled L1/P9 stage is ongoing. Its recorded launch used
16 CPUs, 96 GiB, a 13,800-second limit, and at most six correction solves; the
launch PID was 2074611. The first L1/P9 solve completed Newton iteration 5
at 20:10:41 UTC with residual `9.96951189346986e-10` and solver wall time
3246.342 s, using about 54 GiB at peak. Horizon postprocessing completed at
20:12:06 UTC. Measured masses were `0.5001074903965722` and
`0.500107490416049`; spin norms were about `0.00133753` and `0.00133766`.
The largest target residual, `0.0013376588607675122`, exceeds `5e-5`.
Correction solve 1 is running in `controlled-id/ControlParams_001`; control
convergence is not yet established.
These operational checks cannot establish low
eccentricity or waveform convergence.

The next controlled initial-data stage uses a fresh L1/P9 seed and at most six
correction solves. It controls both masses and both spin vectors, requires
finite measured values, and checks the largest absolute residual against
`5e-5`. The pinned upstream control loop returns normally when its iteration
limit is exhausted, so its return status alone is insufficient. Its inner
solves also default to all visible CPUs; a recorded runtime adapter will pass
the intended processor count explicitly. No numerical solver source is patched.

The [launcher](verification/pass-8/scripts/launch-stage.py) and
[measurement wrapper](verification/pass-8/scripts/measure-stage.py) keep each
stage in a separate container, run as UID/GID 1000 with a read-only root,
and record commands, timing, peak memory, CPU accounting and output growth.
The wrapper also enforces a finite wall time and storage bounds. The
initial-data smoke allocation was 8 CPUs / 32 GiB; the evolution retry uses
8 CPUs / 64 GiB. Longer allocations depend on measured behavior.

The intended checkpoint smoke uses the same binary and eight processing elements before
and after restart, as required by the pinned checkpoint implementation. Its
intentional exit code 2 must be accompanied by a nonempty checkpoint and an
accepted watchdog result. A recorded overlay disables the time-limit checkpoint
on the second segment while preserving the phase-control type needed for restart.

The first evolution smoke was launched at 19:20:20 UTC with eight processing
elements from immutable image
`sha256:838e4ff26967644cc66bebcfc97eab341495d15bf58503874a6d083d77498021`.
The input requested a checkpoint at 0.5 M and completion at 1 M. It failed in
0.7215 s with a segmentation fault while constructing
`ah::Storage::LockedPreviousSurface<Frame::Distorted>` in `GlobalCache`, before
producing HDF5 data or a checkpoint. The wrapper recorded exit 1, no watchdog
termination, and no OOM. The failed input, log, metrics, and binary manifest are
archived under `verification/pass-8/`; restart validation remains pending.

A separate eight-process debugger probe reproduced the startup fault at an
aligned AVX store (`vmovaps %ymm0,0x260(%rbx)`). The destination object was at
`0x59cad115a330`, which is only 16-byte aligned, while the lock member requires
64-byte alignment. The parent Charm nodegroup allocation was also misaligned.
This identifies a runtime allocation/alignment problem. The one-process
debugger probe passed that construction step, then hit its deliberately small
8 GiB memory limit; it is not a successful evolution. Both debugger logs and
their resource records are retained. The initial-data volume exists and was
not the cause of the startup fault.

The bounded runtime repair changes only Charm 7's group and nodegroup
allocation to free-compatible `posix_memalign(..., 64, ...)`. The
[patch and hashes](verification/pass-8/diagnostics/charm7-alignment/patch-provenance.json)
record the exact change against Charm commit
`c5f1dcda82440082bf7d507a63e63270acc390df`. Both ordinary creation and
checkpoint restoration reach these allocators. Separate main, ordinary and
array chare allocations remain outside this repair's scope.

A separate container rebuilt `libck.a` and relinked all four executables in
51.765 s with exit 0, no OOM, and peak cgroup memory 1,959,219,200 bytes.
Each executable hash changed; SpECTRE's numerical source status was unchanged.
The first repair launch's bind-directory permission failure is retained.
The successful repaired image is immutable:
`sha256:0ba0f3bb70b90e83c9b50316ecd37f1b55af4537160eafbe3392a6f67c4e9d77`.
Its [environment record](verification/pass-8/provenance/runtime-aligned-environment.json)
passes Python import and dynamic-library checks.

The same minimal eight-processing-element nodegroup regression aborted for
misalignment against the original library, then completed with all 16
allocations aligned to 64 bytes against the repaired library. The
[before](verification/pass-8/diagnostics/aligned-probe-before-2/probe-result.json)
and [after](verification/pass-8/diagnostics/aligned-probe-after/probe-result.json)
records preserve commands, source hashes and allocation observations.

The evolution retry uses the repaired image, a byte-identical input in fresh
directory `smoke/evolution-aligned/Segment_0000`, eight processing elements,
and a 64 GiB cap. Its input options pass validation. Service
`torchwave-nr-evolution-aligned.service` launched as PID 2781926; its log is
`logs/evolution-aligned.log` and stop command is
`systemctl --user stop torchwave-nr-evolution-aligned.service`.
It passed the original construction fault and evolved both horizons, then
failed at 20:08:15 UTC after 1428.639 s, around simulation time `0.149386 M`.
`DemandOutgoingCharSpeeds` detected incoming characteristic speed at Object B's
excision boundary. The SizeB diagnostics also recorded a negative minimum
characteristic speed. The last recorded norms at `0.1 M` were finite; this
does not establish stability at later times. Peak cgroup memory was
18,756,780,032 bytes, with no OOM or watchdog termination. Exit 1 was rejected
by the checkpoint gate; no checkpoint was produced and no restart was attempted.
The [closed-output diagnostics](verification/pass-8/provenance/evolution-aligned-diagnostics.json)
separate observations from the possible contribution of coarse resolution.
The input, log, reductions and surface data are retained locally; the larger
volume output is retained on len with its hash in the archive manifest.

The [next binary smoke preparation](verification/pass-8/scripts/prepare-controlled-p9-lev0-evolution.py)
requires accepted controlled P9 data and renders Lev0 with 16 processing
elements, checkpoint at `0.5 M`, and completion at `1 M`. It preserves the
upstream boundary conditions and tolerances and disables automatic continuation.
The input was prepared and checked successfully on 2026-09-13. A checkpoint
and successful binary restart are still required. P9 denotes the source initial
data; the pinned evolution Lev0 uses base polynomial order 7 and region-specific
grid extents from the upstream template.

The proposed eccentricity-measurement evolution can be rendered independently
of the upstream automatic control pipeline. Inspection of the pinned
`EccentricityControl.py` found that its fitter requires a compatible SpEC
`OmegaDotEccRemoval` installation. That fitter is not installed or used here.
Retain horizon trajectories for an independently implemented fit; generating
those trajectories alone does not establish the requested eccentricity tolerance.

## Subsequent finite checks

The [P11 preparation helper](verification/pass-8/scripts/prepare-independent-p11.py)
requires a successful controlled P9 record and rechecks its data hashes and
mass/spin residuals. It renders the same numerical options with each grid
extent increased by two, retaining the apparent-horizon postprocessing but
disabling control and evolution. Every other numerical option must agree
exactly with P9. The bounded options check precedes a separately launched solve.
The [comparison helper](verification/pass-8/scripts/compare-independent-p11.py)
will record masses, spin vectors, P11-minus-P9 differences, nonlinear residuals,
and hashes. Both levels must satisfy the proposed `5e-5` mass/spin target gate;
two initial-data resolutions do not establish waveform convergence. These
helpers rendered and validated the input on 2026-09-13; the P11 solve is running
and comparison remains pending.

The [finite evolution preparation](verification/pass-8/scripts/prepare-performance-evolution.py)
also requires controlled P9 data. It renders a Lev1 run ending at 100 M with
outer boundary 1500 M and extraction radii 200, 300 and 400 M. It disables
automatic continuation and validates the generated input. Its 100 M extent
is a performance measurement, insufficient for the proposed two-to-three-orbit
eccentricity fit. Actual allocation remains conditional on the smoke measurements.

## Completed single-hole checkpoint and reference

The pinned `tests/InputFiles/GeneralizedHarmonic/KerrSchild.yaml` template was
rendered with a checkpoint trigger at time >=0.001 and completion at 0.002.
Only these triggers changed. The recorded restart overlay disables that
checkpoint trigger while preserving the phase-control type. All three runs
used the repaired immutable image and four processing elements.

The checkpoint exited intentionally with code 2 after 1.718827 s; the restart
completed with code 0 after 1.168885 s. The uninterrupted reference completed
in 1.721879 s. All had accepted watchdog results and zero OOM counters. Eleven
checkpoint files were recorded with SHA-256 hashes. The final Norms, Errors
and ApparentHorizon rows at 0.002 agree exactly between restart and reference.
The [v2 validation proof](verification/pass-8/provenance/single-kerrschild-validation-v2.json)
checks the actual image, commands, processor counts, exits, finite reduction
tables and final-row comparisons. Surface files are retained and hashed; their
contents were not part of the numerical comparison. This is a small runtime
restart test and does not substitute for a successful binary evolution.

## First finite continuation (stopped)

At 20:34:02 UTC, `torchwave-nr-finite-followup.service` started on `len` with
PID `3997865`. Its [job record](verification/pass-8/provenance/finite-followup-job.json)
and [fixed plan](verification/pass-8/provenance/finite-followup-plan.json) preserve
the command, helper hashes, image and prerequisite proof. Host cwd:
`/home/xangma/simulations/torchwave-nr/2026-09-12-pilot`.

```bash
/usr/bin/python3 -u /home/xangma/simulations/torchwave-nr/2026-09-12-pilot/scripts/run-finite-followup.py
```

It waits for `torchwave-nr-controlled-id.service` (PID `2074611`), whose command
inside `/campaign/controlled-id` is
`python -u /campaign/scripts/control-id.py --num-procs 16`. Correction 1 was
still CPU-bound around 20:33 UTC, using about 53 GiB. The continuation polls
every 30 seconds and launches these stages sequentially:

1. Require completed controlled P9 data, finite diagnostics, unchanged hashes
   and mass/spin target residuals <=`5e-5`.
2. Prepare and solve independent P11 data, then require the recorded comparison
   to pass. Allocation: 16 CPUs, 160 GiB, 13,800 s. The memory cap is an estimate
   from P9's measured peak, not a validated P11 measurement.
3. Prepare controlled P9/Lev0 evolution, checkpoint at >=0.5 M, restart through
   1 M, then validate the closed outputs and unchanged checkpoint manifest.
   Each evolution segment has 16 CPUs, 96 GiB and a 6,900 s limit.

The driver has a 48,000 s total limit (13 h 20 min), with an outer service
limit of 48,120 s. Each launch requires 16 GiB available-memory headroom above
its cap and at least 80 GiB free disk. It rejects changed helpers, stale metrics,
unaccepted exits, OOMs and watchdog terminations; it does not retry stages.
[Fifteen isolated checks](verification/pass-8/provenance/finite-followup-isolated-tests.json)
covered these failure paths, path restrictions, helper hashes and stopping the
owned service. Helper syntax and uploaded SHA-256 hashes were verified.

Logs are `logs/finite-followup.log` and `logs/controlled-id.log` beneath the
host cwd. The live state is `provenance/finite-followup-state.json`. Initial
inspection confirmed the driver active and waiting for controlled ID; the
next automatic check is within 30 seconds. Stop the continuation and current
controlled solve on len with:

```bash
systemctl --user stop torchwave-nr-finite-followup.service torchwave-nr-controlled-id.service
```

Stopping the driver also stops a downstream service it launched. The 100 M
performance test, independent eccentricity fit, full multi-resolution evolution,
waveform extraction/convergence and calibration remain pending. No waveform
or completed calibration is claimed.


## Controlled-data completion and repaired continuation

Controlled ID completed successfully at 21:58:26 UTC on 2026-09-12 after
9,723.83 seconds, with two mass/spin correction solves. The final Newton
residual was `7.277966467926099e-10`; the maximum absolute target residual was
`2.560646462079763e-5`, below `5e-5`. The final masses were
`0.5000256064491514` and `0.5000256064646208`, with spin magnitudes approximately
`5.17e-7` and `5.13e-7`. Peak cgroup memory was 59,146,649,600 bytes; all OOM
counters were zero and no watchdog fired. The final input, reductions, horizons
and [quality record](verification/pass-8/inputs/controlled-id-final/final-quality.json)
are archived locally with verified hashes; volume output remains on len.

The first continuation failed at P11 preparation at 21:58:37 UTC because the
helper omitted the pinned scheduler API's required `scheduler` argument.
It stopped before launching a P11 solve or binary evolution. The
[first-attempt records](verification/pass-8/provenance/finite-followup-first-attempt/finite-followup-state.json)
and original helpers were preserved. The corrective change passes
`scheduler=None` while retaining `no_schedule=True` and `submit=False` in all
three preparation helpers; numerical inputs and tolerances are unchanged.

Actual preparation on the pinned image passed at 00:31 UTC on 2026-09-13:
P11 preparation in 1.51748 s and binary evolution preparation in 1.31671 s,
both exit 0 with no OOM or watchdog event. Both executable options checks
returned zero. P11 differs from controlled P9 only by increasing each grid
extent by two; the helper checks every other numerical option for exact equality.
The rendered evolution imports controlled P9 volumes, uses upstream Lev0,
checkpoints at >=0.5 M and completes at 1 M. Its region-specific initial grid
extents are preserved in the
[archived input](verification/pass-8/provenance/finite-followup-v2-prepared/controlled-p9-lev0-Inspiral.yaml).

The [v2 plan](verification/pass-8/provenance/finite-followup-v2-plan.json) pins
14 completed files plus all continuation helpers. The driver verifies the
launcher's plan hash, rejects changed plans/inputs/helpers, and reuses the
preparation jobs only after matching their accepted metrics, image, commands,
working directories and resource allocations. Fourteen
[isolated checks](verification/pass-8/provenance/finite-followup-v2-isolated-tests.json)
passed, including changed plans/inputs, path traversal, escaping symlinks,
nonaccepted exits and OOM rejection. Uploaded hashes and `git diff --check`
passed. Resource caps, failure-stop behavior and no automatic retries remain
as in the first plan.

The v2 launch record (superseded by v3 below):

- Host: `len`; host cwd: `/home/xangma/simulations/torchwave-nr/2026-09-12-pilot`.
- Driver: `torchwave-nr-finite-followup-v2.service`, PID `1829427`, launched
  00:34:30 UTC. Command:
  `/usr/bin/python3 -u /home/xangma/simulations/torchwave-nr/2026-09-12-pilot/scripts/run-finite-followup-v2.py`.
- Driver log: `/home/xangma/simulations/torchwave-nr/2026-09-12-pilot/logs/finite-followup-v2.log`.
  State: `provenance/finite-followup-v2-state.json` beneath the host cwd.
- Current stage: `torchwave-nr-p11-solve.service`, PID `1830314`;
  container cwd `/campaign/independent-p11`. Command:
  `spectre schedule /campaign/independent-p11/InitialData.yaml --no-schedule --num-procs 16 --submit`.
- Solver log: `/home/xangma/simulations/torchwave-nr/2026-09-12-pilot/logs/p11-solve.log`.
  Initial inspection confirmed the solver started in multicore mode with 16 PEs.
- Expected next check: driver polls within 30 seconds. P11 is capped at
  13,800 seconds (3 h 50 min), through approximately 04:24:32 UTC; this is a
  time limit, not a completion estimate.
- Stop on len: `systemctl --user stop torchwave-nr-finite-followup-v2.service`.
  The driver stops its current owned solver service when stopped.

The [driver job](verification/pass-8/provenance/finite-followup-v2-job.json) and
[P11 job](verification/pass-8/provenance/p11-solve-job.json) retain exact launch
commands and limits. The continuation still covers P9/P11 initial-data checks
and a 1 M binary restart only. The 100 M performance test, independent
eccentricity fit, full evolution, waveform convergence and calibration remain
unlaunched.


## 32-PE allocation requested by the user

At the user's request to use more cores where faster, read-only checks confirmed
64 physical cores / 128 hardware threads on len. The active P11 solver used
approximately 1,600% CPU with 16 worker threads, and about 115 GiB RAM. The
runtime's `+p` option sets the number of processing elements at launch
([Charm++ runtime documentation](https://charm.readthedocs.io/en/rtd.yaml/charm%2B%2B/manual.html)).
The current multicore process had no resumable checkpoint. A 32-worker trial
was selected, leaving CPU capacity for other host services and retaining the
160 GiB memory limit. No twofold speedup was assumed.

The actual 16-PE solver received SIGTERM at 00:47:44 UTC. Its measurement
wrapper closed at 00:48:00 UTC after 808.01685 s, with rejected exit 1, no
watchdog and zero OOM counters. Peak memory was 123,308,122,112 bytes. The old
continuation stopped at this rejected result and launched no downstream checks.
The interrupted output and exact launch/stop records were archived on len in
`archive/p11-16pe-interrupted` and locally under
[the interrupted-run archive](verification/pass-8/provenance/p11-16pe-interrupted/archive-manifest.json).
Partial reductions are timing diagnostics only and are not an accepted solution.
The original P11 input was restored byte-for-byte to a fresh output directory.

The [v3 plan](verification/pass-8/provenance/finite-followup-v3-plan.json) and
versioned driver launch P11 and both subsequent binary evolution segments with
32 processing elements / 32 CPUs. The binary restart preparer and validator
require matching 32-PE checkpoint/restart commands. Historical preparation
records retain their original 16-PE requested allocation; their numerical YAML
is unchanged and hash-verified. Preparation containers still use one CPU.
All existing memory, time, provenance and scientific gates remain in force.
The orchestration service's 10% CPU limit applies to the driver alone; the
independent solver container was observed using approximately 3,212% CPU.
Thirteen [focused checks](verification/pass-8/provenance/finite-followup-v3-isolated-tests.json)
passed, including allocation contracts and changed-input/plan rejection.

Launch record (now complete):

- Host: `len`; host cwd: `/home/xangma/simulations/torchwave-nr/2026-09-12-pilot`.
- Driver: `torchwave-nr-finite-followup-v3.service`, PID `2211621`, launched
  00:49:26 UTC. Command:
  `/usr/bin/python3 -u /home/xangma/simulations/torchwave-nr/2026-09-12-pilot/scripts/run-finite-followup-v3.py`.
- Driver log: `/home/xangma/simulations/torchwave-nr/2026-09-12-pilot/logs/finite-followup-v3.log`.
  State: `provenance/finite-followup-v3-state.json` beneath the host cwd.
- Current solver: `torchwave-nr-p11-solve-32pe.service`, PID `2212503`;
  container cwd `/campaign/independent-p11`. Command:
  `spectre schedule /campaign/independent-p11/InitialData.yaml --no-schedule --num-procs 32 --submit`.
- Solver log: `/home/xangma/simulations/torchwave-nr/2026-09-12-pilot/logs/p11-solve-32pe.log`.
- Expected next check: driver polls every 30 seconds; the initial timing
  comparison follows the first comparable solver iterations. P11 is capped at
  13,800 seconds, through approximately 04:39:29 UTC, with no completion-time
  guarantee.
- Stop on len: `systemctl --user stop torchwave-nr-finite-followup-v3.service`.
  The driver stops its currently owned solver service.

Exact [driver](verification/pass-8/provenance/finite-followup-v3-job.json) and
[solver](verification/pass-8/provenance/p11-solve-32pe-job.json) job records are
retained locally. This remains the finite prerequisite run; longer waveform
work is still pending.

The bounded timing observation completed at 00:56:54 UTC. With the same input
and runtime image, the first two GMRES residuals agreed to floating-point
precision:

| Initial GMRES iteration reached | 16-PE elapsed time | 32-PE elapsed time | Speedup |
| --- | ---: | ---: | ---: |
| 1 | 588.12 s | 342.43 s | 1.72× |
| 2 | 660.29 s | 408.79 s | 1.62× |

Reaching the second iteration took 38.1% less time. The incremental time from
iteration 1 to 2 improved from 72.17 s to 66.36 s (1.09×), so the cumulative
speedup includes substantial initial setup savings and is not a full-run
forecast. The 32-PE allocation is retained. At 00:55 UTC the container used
approximately 3,215% CPU and 114.8 GiB of its 160 GiB memory cap.
The [comparison record](verification/pass-8/provenance/p11-core-scaling-comparison.json)
identifies the exact diagnostic snapshot and its hash;
the [comparison helper](verification/pass-8/scripts/compare-p11-core-scaling.py)
records the bounded read-only measurement procedure. Both runs' partial
reductions are diagnostic evidence only; no accepted P11 solution or completed
binary restart was claimed at that observation. The local timing observer
exited successfully; the remote finite continuation was still active then.

## Completed finite pilot

Read-only checks at 06:27 UTC on 2026-09-13 confirmed the driver and P11 solver
services had stopped successfully, with no active owned stage. The final
[state](verification/pass-8/provenance/completed-32pe/finite-followup-v3-state.json)
records completion at 03:22:39 UTC. All eight stage records, including the two
reused preparation stages, have accepted exits, no watchdog termination and
zero OOM counters. Their hashes match the final state; the
[retrieval manifest](verification/pass-8/provenance/completed-32pe/retrieval-manifest.json)
records the retained final provenance.

- The P11 solve completed in 7,672.85 s (2 h 8 min), peaking at approximately
  119.5 GiB RAM. Its final nonlinear residual was 8.69e-10. Both P9 and P11 met
  the 5e-5 mass/spin target tolerance; the maximum P11 target residual was
  2.561e-5. The per-hole P9/P11 mass differences were approximately 3.17e-9.
  This [comparison](verification/pass-8/provenance/completed-32pe/independent-p11-comparison.json)
  is an initial-data check, without a spectral-convergence claim.
- The binary checkpoint segment took 793.44 s and returned its expected
  checkpoint exit 2. The 32-PE restart took 596.58 s and exited successfully.
  The [final validation](verification/pass-8/provenance/completed-32pe/controlled-p9-lev0-smoke-quality.json)
  verified input and checkpoint hashes, finite reduction data, and restart
  observations from 0.5 M through the 1 M target. Norm/time-step observations
  reached 1.1 M and horizon observations reached 1 M. The two evolution
  segments consumed 23 min 10 s total wall time.

The previously measured 38.1% early elapsed-time reduction remains the only
direct 16/32-PE timing comparison; the interrupted 16-PE run provides no full
solve baseline. The finite operational pilot is complete. A longer evolution,
steady-throughput measurement, eccentricity fit, waveform convergence study
and calibration remain pending.

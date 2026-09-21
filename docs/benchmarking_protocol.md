# Benchmarking protocol

This protocol defines the evidence required for DiffGW performance comparisons.
The [published FP64 campaign](package_benchmarks.md) records its actual workload,
adapters and limitations. Requirements for additional workloads below do not
imply that those workloads have been measured.

## 1. Reference & Scientific Scope

### 1.1 Multi-Backend Comparison Matrix

The package-comparison runner accepts five backend labels:

| Label | Timed implementation |
| --- | --- |
| `torch` | DiffGW PyTorch eager execution |
| `torch_compiled` | DiffGW with `torch.compile` |
| `jax` | DiffGW with `jax.jit` |
| `lal` | LALSimulation public API on CPU, with a Python loop for batches |
| `ripple` | Ripple public JAX API with JIT and vectorization |

Select the supported subset for each model and record omissions. Match physical
parameters, precision, modes, conventions and sampling grids. These execution
backends are not five independent physical models: DiffGW's frontends share
physics inputs. Record frontend consistency, same-model external-reference
agreement and independent NR validation separately; see the
[package comparison](comparison.md#physics-coverage-and-scientific-agreement).

### 1.2 The Gatekeeper Rule (Parity First)

A timing result is qualified only for the model, reference, precision and
configurations that passed its declared accuracy checks.

For the package campaign's **strict same-model LAL comparison**:

- Compare both polarizations of every timed binary with the pinned LAL model
  and options on the same physical grid.
- Require direct relative L2 error at most `1e-11`, without alignment, and
  mismatch at most `1e-14`. This mismatch uses flat frequency weights and
  maximizes constant phase, without maximizing time.
- Retain errors per binary/polarization, reference identity and configuration.
  Reject nonfinite, zero-norm, wrong-shape or wrong-dtype outputs.
- Stop on failure, or retain the timing explicitly as diagnostic and
  unqualified. Do not relax tolerances merely to obtain a timing.

These are implementation-agreement thresholds for this campaign. Passing them
does not establish physical accuracy throughout a calibration domain; failing
them does not alone decide suitability for an astrophysical analysis.
Models with independent calibrations require an appropriate external reference
or held-out NR test, with separately justified tolerances.

`scripts/compare_packages.py` implements these checks for FP64 `TaylorF2`,
`IMRPhenomD` and `IMRPhenomXAS`. It retains failed timings only with
`--record-unqualified`. It measures forward waveform generation, not
time-domain, derivative, Fisher, likelihood or memory workloads.

**Legacy harness limitation:** `scripts/run_benchmark_campaign.py` first uses
JAXWave as its reference, falls back to LAL with relative L2 at most `1e-3`
and mismatch at most `1e-6`, and can fall back to self-consistency. Its pass
status must not be labelled a strict LAL pass. Obtain explicit same-configuration
LAL errors before reusing those timings as qualified results.

### 1.3 Precision and device qualification

Begin with FP64 where supported. Evaluate FP32 separately, reporting its errors
and acceptance criteria alongside its timings. Record real and complex dtypes,
accelerator and compiler settings, including mixed precision. An FP64 pass does
not qualify FP32, and framework device support alone does not qualify a
particular model/device combination.

## 2. Timing Boundaries & Instrumentation

### 2.1 Boundary Definitions

The published campaign times calls through public waveform adapters, from
resident inputs to completed outputs. It excludes setup, input transfers and
output conversion to NumPy. Necessary Python dispatch, parameter handling and
batch assembly inside an adapter remain timed. For example, LAL's boundary
includes its scalar-call loop, padded grid and slicing/stacking. Document these
differences rather than describe the interval as kernel-only execution.

For each configuration:

1. Record at least three untimed warmups and at least 15 timed calls. Warmup
   count alone does not prove that compilation or initialization has finished.
2. Ensure previous device work has finished before starting the timer, then
   wait for every measured output before stopping it. Use the appropriate
   PyTorch device synchronization or JAX `block_until_ready`; apply this to
   asynchronous CPU execution as well as GPU execution.
3. Measure first-call cost separately and state whether caches were cleared.
   A first call can include evaluation and cache loading; it is not automatically
   a cold-start or isolated compilation measurement.
4. Keep profiling sessions separate from latency measurements. Record inference
   and gradient settings; the published Torch forward adapter uses
   `torch.inference_mode()`.
5. Record compilation settings and evidence that compiled graphs were captured.
   Do not infer kernel fusion or zero Python overhead from a compiler flag.

Dynamic shapes and bucketing can reduce repeated compilation, but a new shape,
bucket, dtype or static configuration can still compile. A claim of no
recompilation requires compiler counters or logs for the stated workload.

### 2.2 Derivative, analysis and memory workloads

Derivative comparisons must use the same objective, differentiated parameters,
precision and derivative order, and validate derivatives independently of
waveform values. Distinguish directly timed backward execution from a difference
between separately measured forward and value-and-gradient times. Fisher and
likelihood comparisons must also match PSDs, detector projections, integration
rules and approximation settings. Synchronize their outputs before stopping
the timer, and save their own receipts.

Memory reports must identify the measured quantity, device, precision, batch,
autograd mode and allocator settings. PyTorch's `max_memory_allocated` measures
peak tensor allocations tracked by its allocator; it is not total device memory
use. Report reserved memory separately and use consistent byte units: MiB/KiB
for division by powers of 1024, MB/KB for powers of 1000. Call a batch the
largest tested batch unless an actual capacity limit has been measured.

## 3. Workload Scaling & Convergence

Use batches of 1, 10, 100 and 1,000 to distinguish scalar latency from batch
throughput. Larger sweeps may characterize scaling, but saturation and the
limiting hardware resource require measurements rather than inference from
batch size alone.

The published campaign uses 20–512 Hz inclusive at 0.25 Hz spacing: 1,969 bins.
Record the exact grid and parameters for every campaign, including whether a
batch repeats one binary or contains distinct binaries. Match physical inputs
across packages and disclose parameter conversions excluded from timing.
Legacy model defaults need not use this grid.

## 4. Host Conditions & Hardware Provenance

Record hostname, OS, Python and package versions, CPU model, process affinity,
thread settings, GPU model/memory, driver/runtime versions and source/runner
hashes. Describe clock and reservation controls, or their absence. A requested
commit label is not proof that the working tree matches it; independently
verify the source used. Report CPU and GPU comparisons separately and explain
that cross-device ratios combine hardware and implementation differences.

## 5. Statistical Reporting and Receipts

Retain every raw timing with its model, backend, device, dtype, batch,
configuration and qualification result. Report the median complete-batch
latency and a 95% percentile-bootstrap interval using 2,000 seeded resamples.
The published campaign uses 25 timed calls. Its intervals describe variation
within that run, not across machines or independent campaigns. Additional
percentiles and IQR can be calculated from the saved samples.

For batch size $B$ and median batch latency $t$ in seconds:

- Throughput is $B/t$ waveforms per second.
- Amortized time per waveform is $t/B$ seconds.
- Only $B=1$ directly measures a single-waveform request's latency.

The committed package receipts use `diffgw-package-comparison/v1`; the legacy
runner uses `diffgw-benchmark-receipt/v1`. A SHA-256 digest detects content
changes when checked against a trusted copy. It does not authenticate the
experiment or qualify its scientific result. Publish the raw data, source and
runner provenance alongside any numerical claims.

## 6. Presenting Comparisons

Tables and plots must identify the model, workload, device, precision, batch,
timing boundary and qualification status. Show failures and omissions explicitly.
Ratios should compare matched batch workloads; multiplying a scalar baseline
by a batch size is an extrapolation, not a measured batch baseline.

Do not derive end-to-end inference speedups from waveform-generation timings,
or memory, derivative or physical-accuracy claims from a forward-only campaign.
Legacy plotting scripts do not independently establish qualification. See the
[performance summary](performance.md) and [complete package results](package_benchmarks.md)
for the supported measurements.

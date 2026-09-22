# DiffGW performance results

The published [package benchmarks](package_benchmarks.md) measure **FP64 forward
waveform generation** in DiffGW, Ripple and LALSimulation. They include raw
timings, numerical errors and environment metadata in committed
[CPU](benchmarks/mac-20260921-cpu.json) and
[GPU](benchmarks/mac-20260921-gpu.json) receipts.

## Measured scope

The campaign ran on 21 September 2026 using DiffGW commit `5226141`, a Ryzen 9
7950X3D host and an RTX 5090. CPU execution and GPU host dispatch were pinned to
one logical CPU with numerical-library thread limits. This was a workstation
run without exclusive reservation or locked clocks.

The workload covers `TaylorF2`, `IMRPhenomD` and `IMRPhenomXAS`, at batches of
1, 10, 100 and 1,000 distinct binaries. Every call returns both polarizations
on 1,969 frequency bins: 20–512 Hz inclusive at 0.25 Hz spacing. The report
specifies the sampled mass/spin ranges, fixed extrinsic parameters, software
versions and public-API adapters.

Before timing, both polarizations of every binary were compared with LAL.
The campaign requires direct relative L2 at most `1e-11` and flat-weight,
constant-phase-maximized mismatch at most `1e-14`. All measured DiffGW
PhenomD/XAS configurations pass; all measured DiffGW and Ripple TaylorF2
configurations miss the L2 threshold. Failed configurations remain explicitly
labelled diagnostic in the plots and full tables. This gate measures implementation
agreement for the sampled inputs, not physical accuracy across a model's domain.

## Reading the results

The [comparison plot](comparison.md#measured-batch-performance) shows
complete-batch latencies at **batch 1,000**. Compiled DiffGW has the lowest
measured CPU batch latency among the tested implementations for IMRPhenomD
and IMRPhenomXAS at this batch size. LAL has the lowest
single-waveform CPU latency for all three measured models. Results depend on
model, batch size, backend and hardware; the
[scaling plots and complete tables](package_benchmarks.md#warmed-batch-latency) include Ripple,
eager execution, smaller batches, failed accuracy checks and 95% bootstrap
intervals. Those intervals describe variation within this run.

Timings use resident inputs and wait for completed outputs. Setup, transfers
and output conversion to NumPy are excluded. The LAL adapter includes a Python
batch loop, its padded uniform-grid API and slicing/stacking. Ripple accepts
chirp mass and symmetric mass ratio prepared before timing; DiffGW accepts
component masses. Ratios therefore apply to the documented adapters.
GPU-to-CPU ratios additionally combine hardware and implementation differences.

For batch size $B$ and batch latency $t$ in seconds, throughput is $B/t$ and
amortized time per waveform is $t/B$. Only $B=1$ directly measures a
single-waveform request; a large batch's amortized time is not request latency.

## Compilation and derivatives

The published Torch adapter checks that `torch.compile` captured at least one
graph; timed calls may still include Python outside compiled graphs. First-call
costs include evaluation and compilation or cache loading, with imports and
setup excluded. Compiler caches were not cleared before the campaign, so these
costs do not measure cold start or isolated compilation.

`dynamic=True` can reduce PyTorch recompilation when shapes change, but does
not guarantee one compilation for every batch size, dtype or model.
[JAX shape bucketing](usage.md#shape-bucketing-for-mcmc-sampling-bucketedwaveformevaluator)
reuses a JIT evaluator for each padded bucket. A newly encountered bucket or
input signature can require compilation; warm the relevant signatures before
measuring repeated evaluation.

Supported waveform paths provide automatic differentiation. Its availability
and order depend on the model and sampling path. The published campaign does
not measure derivative, Fisher, likelihood, time-domain, FP32 or peak-memory
workloads, and supplies no speedup or memory claim for them. Such comparisons
need separate matched workloads, numerical checks and saved measurements.

## Reproduction

Use the [package benchmark reproduction commands](package_benchmarks.md#reproduction-and-receipts)
with the declared source revision, runner hash and environment. The
`--source-commit` argument records a label; verify source contents independently.
`--record-unqualified` retains failed cases without changing the accuracy gate.
See the [benchmark protocol](benchmarking_protocol.md) for timing boundaries,
qualification, precision and reporting requirements.

The legacy `run_benchmark_campaign.py` can use JAXWave or self-consistency as
its reference and relaxed LAL tolerances; its pass status does not establish
the strict LAL gate. Its timings require independent qualification before reuse.

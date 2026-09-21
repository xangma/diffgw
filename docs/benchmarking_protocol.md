# Controlled Executable Benchmark Protocol & Methodology for DiffGW

This document defines the formal benchmarking methodology and protocol for gravitational-wave approximants in **`diffgw`** (`torchwave` and `jaxwave`), establishing consistent performance tracking across releases, architectures, and hardware environments.

The protocol adapts the controlled scientific benchmarking principles established in PyCBC's `torch_benchmark_protocol.rst`. These requirements describe qualified evidence; the legacy harness limitations below must be accounted for when interpreting existing results.

---

## 1. Reference & Scientific Scope

### 1.1 Multi-Backend Comparison Matrix
The legacy campaign supports five backend labels. Select the supported subset
for each model and record omissions; match physical parameters, precision,
sampling rates and frequency grids:

1. **`torchwave` (PyTorch Eager)**: Differentiable PyTorch approximants evaluated in eager execution (`torch.inference_mode()` for forward-only timing). Record the actual tested device; framework support alone does not qualify a model/device combination.
2. **`torchwave_compiled` (PyTorch Compiled)**: Differentiable PyTorch approximants compiled via `torch.compile()` with TorchDynamo and TorchInductor/Triton kernel fusion.
3. **`jaxwave` (JAX)**: Differentiable JAX approximants compiled via `jax.jit` and vectorized via `jax.vmap` on GPU and CPU.
4. **`lalsimulation` (C / LALSuite)**: An external waveform reference and CPU single-waveform baseline. Pin the LAL version and exact approximant options before claiming same-model agreement.
5. **`ripple` (JAX)**: External differentiable JAX approximants from the [active GW-JAX-Team project](https://github.com/GW-JAX-Team/ripple), providing an additional CPU/GPU implementation where models intersect.

These are execution backends, not five independent physical models. DiffGW's
PyTorch and JAX frontends share physics inputs. Record frontend agreement,
same-model external-reference agreement and independent NR validation separately;
see the [package comparison](comparison.md#physics-coverage-and-scientific-agreement).

### 1.2 The Gatekeeper Rule (Parity First)
> [!IMPORTANT]
> **A timing result is scientifically qualified only for the model, reference, precision and configurations that passed its declared accuracy checks.**

For a **strict same-model LAL comparison**, the required Gate 1 checks are:

- Compare candidate waveforms ($h_+, h_\times$) against the pinned LAL model and options across the specified grid and parameter cases. Record conventions and any alignment.
- **Strict FP64 tolerances**:
  - Max relative $L_2$ error: $\frac{\|h_{\text{candidate}} - h_{\text{oracle}}\|_2}{\|h_{\text{oracle}}\|_2} \le 10^{-11}$
  - Max phase mismatch: $1 - \mathcal{M} \le 10^{-14}$
- Retain the reference identity and raw errors. Stop the qualified campaign on failure, or label retained timings as diagnostic and unqualified. Do not relax tolerances merely to record timings.

**Legacy harness limitation:** `scripts/run_benchmark_campaign.py` does not
enforce this strict LAL contract. It first uses JAXWave as its reference, falls
back to LAL with $L_2 \le 10^{-3}$ and mismatch $\le 10^{-6}$ tolerances, and can
fall back to self-consistency. Leaving its gate enabled is preferable to skipping
it, but a legacy pass must not be labelled a strict LAL pass. Obtain an explicit
same-configuration LAL comparison before making that claim.

The narrower `scripts/compare_packages.py` runner performs explicit LAL checks
for FP64 `TaylorF2`, `IMRPhenomD` and `IMRPhenomXAS`. It saves errors for both
polarizations of every binary. Its mismatch uses flat frequency weights and
maximizes constant phase, but not time; its relative $L_2$ error uses the direct
complex waveforms without alignment. Unqualified timings are retained only with
`--record-unqualified` and remain labelled as such. It does not measure
time-domain, derivative, Fisher or memory workloads.

For models with independent calibrations, record the appropriate external
reference or held-out NR test and its justified tolerances instead. Passing
Torch/JAX agreement is not independent physical validation. A receipt hash
does not change the meaning of the reference or qualification checks.

### 1.3 Precision and device qualification

Begin comparisons with FP64 where supported. Evaluate FP32 separately and report
its errors and acceptance criteria alongside its timings; an FP64 pass cannot
qualify an FP32 result. Record real and complex dtypes, accelerator, compiler
options and any mixed-precision operations. Where a device cannot execute the
required precision, mark that configuration unavailable rather than silently
changing precision or thresholds.

---

## 2. Timing Boundaries & Instrumentation

### 2.1 Boundary Definitions
To ensure meaningful comparisons between interpreted Python frontends, compiled kernels, and C libraries:

1. **Calculation-Stage Timing**:
   - Measures only the core mathematical generation of $h_+$ and $h_\times$ over the frequency grid.
   - Excludes initialization, parameter parsing, disk I/O and JIT compilation. Record the actual boundary for each adapter and report cold-start/compilation costs separately.
   - For GPU executions, the timing interval **strictly synchronizes** device queues at both boundaries:
     - PyTorch: `torch.cuda.synchronize(device)` (or `torch.mps.synchronize()`)
     - JAX: `res[0].block_until_ready(); res[1].block_until_ready()`
2. **Warm-Up Runs**:
   - To eliminate JIT compilation overhead, PyTorch kernel caching, and CUDA driver initialization, every configuration executes $W \ge 3$ unrecorded warm-up evaluations before timing starts.
3. **Unprofiled Execution**:
   - Official timing runs are executed **completely uninstrumented** (no PyTorch Profiler, cProfile, or memory profilers active). Profiling sessions are run separately and documented in isolated sections.
4. **Inference Mode**:
   - When measuring pure forward generation (`measure_backward=False`), execution runs inside `torch.inference_mode()`, preventing autograd graph accumulation and memory runaway.

---

## 3. Workload Scaling & Convergence

### 3.1 Geometric Batch Sweeps
Gravitational-wave parameter estimation (PE) and matched-filter searches rely heavily on batched waveform generation. Performance must be characterized across geometric batch sizes:
$$B \in \{1, 10, 100, 1000\}$$

- **Batch 1**: Measures single-waveform latency (relevant for low-latency search alerts and scalar MCMC steps).
- **Batch 10–100**: Represents parallel MCMC walker steps, particle swarm optimizations, and small matched-filter template batches.
- **Batch 1000**: Measures larger-batch throughput. Saturation and the limiting hardware resource must be measured rather than inferred from batch size.

### 3.2 Fixed Frequency Grids & Parameters
Comparisons must be matched on identical frequency grids:
- Canonical inspection grid: $f_{\text{low}} = 20\text{ Hz}$, $f_{\text{high}} = 512\text{ Hz}$, $\Delta f = 0.25\text{ Hz}$ ($N_{\text{freq}} = 1,969$).
- Parameter definitions (masses, spins, distances, reference frequencies) are frozen across all backends.

---

## 4. Host Conditions & Hardware Provenance

Benchmark receipts should capture machine-readable hardware and environment provenance:
1. **Host**: Hostname, OS platform, kernel version, Python runtime.
2. **CPU**: Model name, physical core count, logical core count, frequency governor, thread limits (`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`).
3. **GPU**: Device model, VRAM capacity, CUDA driver version, CUDA runtime version.
4. **Software Revisions**: Git commit SHA, PyTorch version, JAX version, LALSimulation version, Ripple version.

---

## 5. Statistical Rigor & Cryptographic Sealing

### 5.1 Metrics Reported
For each configuration $( \text{model}, \text{backend}, \text{device}, \text{dtype}, B )$ across $N \ge 15$ iterations:
- **Median Latency ($t_{50}$)**: Primary timing metric robust against system jitter.
- **Bootstrap 95% Confidence Interval**: Calculated via 2,000 deterministic percentile-bootstrap resamples of the median.
- **Percentiles**: $p_{05}, p_{16}, p_{50}, p_{84}, p_{95}$ and interquartile range (IQR).
- **Throughput**: Waveforms per second:
  $$\text{Throughput} = \frac{B}{t_{50}}$$
- **Batch vs. amortized latency**: $t_{50}$ is the latency of the whole batch;
  $t_{50}/B = 1/\text{Throughput}$ is the amortized time per waveform. Only
  $B=1$ measures a single-waveform request directly.

### 5.2 Cryptographic Artifact Sealing (`SCHEMA_VERSION = 1`)
The legacy campaign emits JSON receipts using schema
`diffgw-benchmark-receipt/v1` with a SHA-256 digest over canonicalized content.
This allows detection of content changes when the digest is checked against a
trusted copy; it does not authenticate the experiment or certify its scientific
qualification. Preserve raw timings, reference metrics and software provenance
with published results.

---

## 6. Publication-Grade Plotting Standards

Benchmark receipts feed directly into automated plotting tools (`scripts/plot_benchmark_results.py`), generating four standardized figures:

1. **Figure 1: Throughput vs. Batch Size ($B$)**:
   - Log-log plot showing throughput ($\text{wf/s}$) vs $B$ from $1$ to $1000$.
   - Highlights GPU scaling curves vs CPU ceilings and LALSimulation baseline.
2. **Figure 2: Per-Waveform Latency vs. Batch Size**:
   - Shows any measured CPU-to-GPU amortization crossover for the chosen workload.
3. **Figure 3: Speedup vs. LALSimulation**:
   - For a batch time $t_{\text{backend}}$ and scalar LAL time $t_{\text{LAL}}$, report $B\,t_{\text{LAL}}/t_{\text{backend}}$. Label GPU-to-single-core-CPU ratios explicitly and compare qualified configurations.
4. **Figure 4: Cross-Model Approximant Breakdown**:
   - Multi-model bar chart comparing non-spinning (`TaylorF2`), aligned-spin (`IMRPhenomD`, `IMRPhenomXAS`), precessing (`IMRPhenomXP`), and higher-harmonic (`IMRPhenomXHM`) models.

For a detailed walkthrough of the generated figures, hardware scaling on the RTX 5090, and physical interpretation of the results, see [**Performance Report**](performance.md).

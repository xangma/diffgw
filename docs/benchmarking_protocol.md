# Controlled Executable Benchmark Protocol & Methodology for DiffGW

This document defines the formal benchmarking methodology and protocol for gravitational-wave approximants in **`diffgw`** (`torchwave` and `jaxwave`), establishing consistent performance tracking across releases, architectures, and hardware environments.

The protocol adapts the controlled scientific benchmarking principles established in PyCBC's `torch_benchmark_protocol.rst`, establishing reproducible, gatekeeper-enforced performance evidence.

---

## 1. Reference & Scientific Scope

### 1.1 Multi-Backend Comparison Matrix
Every benchmark campaign measures performance across four primary backends under identical physical parameters, sampling rates, and frequency grids:

1. **`torchwave` (PyTorch)**: Differentiable PyTorch approximants evaluated on GPU (CUDA / ROCm / Apple Silicon MPS) and CPU.
2. **`jaxwave` (JAX)**: Differentiable JAX approximants compiled via `jax.jit` and vectorized via `jax.vmap` on GPU and CPU.
3. **`lalsimulation` (C / LALSuite)**: The canonical LIGO-Virgo-KAGRA single-core reference C implementation, serving as the scientific oracle and single-waveform baseline.
4. **`ripple` (JAX)**: External open-source differentiable JAX approximants (Edwards et al.), providing an independent GPU-accelerated baseline where models intersect.

### 1.2 The Gatekeeper Rule (Parity First)
> [!IMPORTANT]
> **No timing result qualifies as valid performance evidence unless numerical parity with the scientific oracle is verified first.**

Prior to initiating timing sweeps for any model, the benchmark harness executes **Gate 1: Scientific Qualification**:
- Compare candidate waveforms ($h_+, h_\times$) against the oracle across the specified frequency grid.
- **Tolerances**:
  - Max relative $L_2$ error: $\frac{\|h_{\text{candidate}} - h_{\text{oracle}}\|_2}{\|h_{\text{oracle}}\|_2} \le 10^{-11}$
  - Max phase mismatch: $1 - \mathcal{M} \le 10^{-14}$
- If Gate 1 fails for any configuration, the benchmark halts immediately with a qualification failure, or flags the result as unscientific. Tolerances must never be relaxed merely to record timings.

---

## 2. Timing Boundaries & Instrumentation

### 2.1 Boundary Definitions
To ensure meaningful comparisons between interpreted Python frontends, compiled kernels, and C libraries:

1. **Calculation-Stage Timing**:
   - Measures only the core mathematical generation of $h_+$ and $h_\times$ over the frequency grid.
   - Explicitly excludes initialization, parameter parsing, disk I/O, and profiler compilation.
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
- **Batch 1000**: Measures peak batched throughput where GPU tensor cores and memory bandwidth reach full saturation.

### 3.2 Fixed Frequency Grids & Parameters
Comparisons must be matched on identical frequency grids:
- Canonical inspection grid: $f_{\text{low}} = 20\text{ Hz}$, $f_{\text{high}} = 512\text{ Hz}$, $\Delta f = 0.25\text{ Hz}$ ($N_{\text{freq}} = 1,969$).
- Parameter definitions (masses, spins, distances, reference frequencies) are frozen across all backends.

---

## 4. Host Conditions & Hardware Provenance

Every benchmark receipt captures machine-readable hardware and environment provenance:
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

### 5.2 Cryptographic Artifact Sealing (`SCHEMA_VERSION = 1`)
All benchmark runs emit a sealed JSON receipt adhering to schema `diffgw-benchmark-receipt/v1`. The payload includes a self-verifying SHA-256 seal computed over canonicalized content (`canonical_sha256`), guaranteeing tamper-free records for regression tracking.

---

## 6. Publication-Grade Plotting Standards

Benchmark receipts feed directly into automated plotting tools (`scripts/plot_benchmark_results.py`), generating four standardized figures:

1. **Figure 1: Throughput vs. Batch Size ($B$)**:
   - Log-log plot showing throughput ($\text{wf/s}$) vs $B$ from $1$ to $1000$.
   - Highlights GPU scaling curves vs CPU ceilings and LALSimulation baseline.
2. **Figure 2: Per-Waveform Latency vs. Batch Size**:
   - Demonstrates the CPU-to-GPU amortization crossover point (typically $B \sim 5$ to $15$).
3. **Figure 3: Speedup vs. LALSimulation**:
   - Speedup factor normalized directly to single-core C LALSimulation ($t_{\text{LAL}} / t_{\text{backend}}$).
4. **Figure 4: Cross-Model Approximant Breakdown**:
   - Multi-model bar chart comparing non-spinning (`TaylorF2`), aligned-spin (`IMRPhenomD`, `IMRPhenomXAS`), and precessing (`IMRPhenomXP`) models.

# Measured package benchmarks

This comparison measures FP64 forward waveform generation in DiffGW, Ripple and
LALSimulation on `mac`, on **21 September 2026**. It covers `TaylorF2`,
`IMRPhenomD` and `IMRPhenomXAS`, each at batches of **1, 10, 100 and 1,000**.
For DiffGW's capabilities, model coverage and differences from related packages,
see the [package comparison](comparison.md).

## Workload and environment

DiffGW uses an isolated archive of commit
[`5226141887ddd7394c8430537de916d871571358`](https://github.com/xangma/diffgw/commit/5226141887ddd7394c8430537de916d871571358).
Its source hash was checked against that archive before publishing these results.
The benchmark runner is `scripts/compare_packages.py`; its hash and the installed
package versions are recorded in each receipt.

| Setting | Value |
| --- | --- |
| Host | `mac`, Windows 11 with NVIDIA-Workbench WSL2; AMD Ryzen 9 7950X3D |
| CPU execution | Process pinned to logical CPU 0; one thread for PyTorch, OpenMP, MKL and OpenBLAS; XLA CPU multithreaded Eigen disabled |
| GPU | NVIDIA GeForce RTX 5090, 32,607 MiB; driver 610.88 |
| Software | Python 3.12.13; PyTorch 2.14.0; JAX/JAXlib 0.11.1; RippleGW 0.4.0; LALSuite 7.26.15; NumPy 2.5.3 |
| Precision | FP64 inputs, verified `complex128` output |
| Frequency grid | 20–512 Hz inclusive, 0.25 Hz spacing, 1,969 bins |
| Fixed parameters | Distance 400 Mpc, inclination 0.7 rad, reference phase 0, reference frequency 20 Hz |
| TaylorF2 population | Nonspinning; component masses in [1.4, 1.7) and [1.1, 1.3) solar masses; zero tides |
| PhenomD/XAS population | Component masses in [30, 40) and [20, 25) solar masses; aligned spins independently in [−0.2, 0.2) |
| Sampling | Seed 20260921; distinct binaries; smaller batches are prefixes of larger batches |
| Timing | 3 untimed warmups, 25 timed calls; median and 95% bootstrap interval using 2,000 resamples |

CPU and GPU campaigns run sequentially. The GPU benchmark's host process is
also pinned to logical CPU 0, so dispatch uses one host core.
This is a workstation measurement, without exclusive host/GPU reservation or
locked clock frequencies. Bootstrap intervals describe variation within this
run, not uncertainty across machines or independent campaigns.

## Timing boundary

Each timed call generates both polarizations for the complete batch from inputs
already resident on the selected device, and waits for completion. Setup,
host/device transfers and output conversion to NumPy are excluded. Both packages
compile with fixed extrinsic parameters; masses and spins remain runtime inputs.
DiffGW's Torch path uses inference mode; the JAX paths use `jax.jit`. The
`torch.compile` adapter checks that at least one graph was captured; its measured
call can still include Python work around compiled graphs.

LAL uses its public uniform-grid API, including zero-frequency padding, followed
by slicing and stacking the requested bins. Its batch timing includes a Python
loop over binaries. Ripple receives chirp mass and symmetric mass ratio, with
conversion from component masses performed during setup; DiffGW receives
component masses. These are timings of the stated public-API adapters, including
their different parameterizations and batching paths.

Receipts also record the first evaluation of each configuration, including
compilation or cache loading as applicable. Imports, adapter setup and parameter
conversion occur before that timer. Compiler caches were not cleared between
smoke checks and the campaign, so this is **not cold-start or isolated compilation
time**. No derivative, time-domain, Fisher, peak-memory or FP32 workload was timed.

## Numerical qualification

Before warming and timing, every binary and both polarizations are compared with
LALSimulation on the same physical grid. Qualification requires direct relative
$L_2 \le 10^{-11}$ and flat-weight mismatch $\le 10^{-14}$. Only the mismatch
maximizes constant phase; neither metric maximizes time, and the direct error
uses no alignment. Nonfinite, zero-norm, wrong-shape or non-`complex128` results
are rejected.

`--record-unqualified` retains timings that fail these thresholds as diagnostic
observations. It does not relax the gate. LAL is the reference, so its comparison
with itself is a consistency check. Passing this gate establishes agreement for
these inputs, not a model's physical accuracy throughout its calibration domain.
Failure of this stringent implementation-agreement gate does not alone determine
suitability for an astrophysical analysis; that depends on the analysis's PSD,
signal-to-noise ratio and accuracy requirements.

### Accuracy across the measured configurations

The maxima below include both devices, all four batch sizes and both polarizations.
Each implementation/model has eight configurations; counts refer to batches,
not individual binaries. Thresholds are the same for every package.

| Model | Implementation | Qualified configurations | Maximum relative L2 | Maximum mismatch |
| --- | --- | ---: | ---: | ---: |
| TaylorF2 | DiffGW Torch eager | 0/8 | 3.437e-11 | 5.551e-16 |
| TaylorF2 | DiffGW Torch compiled | 0/8 | 3.438e-11 | 5.551e-16 |
| TaylorF2 | DiffGW JAX JIT | 0/8 | 3.424e-11 | 5.551e-16 |
| TaylorF2 | Ripple JAX JIT | 0/8 | 3.066e-11 | 5.551e-16 |
| IMRPhenomD | DiffGW Torch eager | 8/8 | 9.169e-13 | 6.661e-16 |
| IMRPhenomD | DiffGW Torch compiled | 8/8 | 9.211e-13 | 5.551e-16 |
| IMRPhenomD | DiffGW JAX JIT | 8/8 | 9.083e-13 | 6.661e-16 |
| IMRPhenomD | Ripple JAX JIT | 2/8 | 2.806e-11 | 6.661e-16 |
| IMRPhenomXAS | DiffGW Torch eager | 8/8 | 1.674e-13 | 5.551e-16 |
| IMRPhenomXAS | DiffGW Torch compiled | 8/8 | 1.521e-13 | 5.551e-16 |
| IMRPhenomXAS | DiffGW JAX JIT | 8/8 | 1.658e-13 | 4.441e-16 |
| IMRPhenomXAS | Ripple JAX JIT | 8/8 | 1.680e-13 | 6.661e-16 |

## Warmed batch latency

Values are **milliseconds per complete batch**, shown as median [95% bootstrap
interval]; lower is faster. **† means unqualified** under the numerical gate
above and denotes a diagnostic timing. Throughput is `1,000 × batch / milliseconds`;
amortized time is batch latency divided by batch size. For batches larger than
one, amortized time is not single-waveform request latency.

### One CPU core

| Model | Implementation | B = 1 | B = 10 | B = 100 | B = 1,000 |
| --- | --- | ---: | ---: | ---: | ---: |
| TaylorF2 | LALSimulation | 0.111 [0.11, 0.117] | 1.35 [1.31, 1.37] | 14.3 [13.7, 14.6] | 142 [142, 144] |
| TaylorF2 | DiffGW Torch eager | 0.593 [0.554, 0.633]† | 1.4 [1.31, 1.43]† | 8.08 [7.81, 8.56]† | 94.4 [88.7, 102]† |
| TaylorF2 | DiffGW Torch compiled | 0.172 [0.152, 0.208]† | 0.579 [0.538, 0.648]† | 4.31 [4.23, 4.46]† | 44.2 [43, 45.2]† |
| TaylorF2 | DiffGW JAX JIT | 0.157 [0.134, 0.164]† | 0.96 [0.93, 1.04]† | 12.6 [12, 14.7]† | 120 [118, 122]† |
| TaylorF2 | Ripple JAX JIT | 0.124 [0.105, 0.137]† | 0.963 [0.922, 1.02]† | 8.85 [8.51, 9.18]† | 101 [99.6, 102]† |
| IMRPhenomD | LALSimulation | 0.209 [0.192, 0.223] | 1.81 [1.77, 1.9] | 18.7 [18.1, 19.6] | 199 [196, 200] |
| IMRPhenomD | DiffGW Torch eager | 1.88 [1.76, 1.95] | 3.85 [3.73, 3.91] | 20 [19.3, 20.6] | 207 [205, 209] |
| IMRPhenomD | DiffGW Torch compiled | 0.626 [0.573, 0.643] | 1.25 [1.17, 1.33] | 7.6 [7.06, 7.81] | 73.1 [71.9, 74.5] |
| IMRPhenomD | DiffGW JAX JIT | 0.326 [0.304, 0.341] | 2.23 [2.18, 2.29] | 22.6 [22.1, 22.9] | 321 [314, 329] |
| IMRPhenomD | Ripple JAX JIT | 0.485 [0.469, 0.501] | 3.8 [3.78, 3.94]† | 49.3 [48.6, 49.5]† | 539 [525, 550]† |
| IMRPhenomXAS | LALSimulation | 0.159 [0.154, 0.18] | 1.65 [1.63, 1.72] | 15.8 [15.4, 16.3] | 160 [157, 166] |
| IMRPhenomXAS | DiffGW Torch eager | 5.55 [5.44, 5.68] | 7.71 [7.58, 7.96] | 25.4 [25, 25.6] | 272 [265, 315] |
| IMRPhenomXAS | DiffGW Torch compiled | 0.366 [0.341, 0.398] | 1.5 [1.4, 1.54] | 10.8 [10.5, 11] | 114 [113, 114] |
| IMRPhenomXAS | DiffGW JAX JIT | 0.424 [0.407, 0.456] | 3.33 [3.17, 3.41] | 32.4 [31.7, 32.8] | 450 [443, 453] |
| IMRPhenomXAS | Ripple JAX JIT | 0.345 [0.332, 0.355] | 3.05 [3, 3.14] | 30.9 [30.4, 31.7] | 400 [396, 406] |

### RTX 5090 GPU

| Model | Implementation | B = 1 | B = 10 | B = 100 | B = 1,000 |
| --- | --- | ---: | ---: | ---: | ---: |
| TaylorF2 | DiffGW Torch eager | 12.3 [11, 13.2]† | 3.86 [3.71, 3.96]† | 13.2 [10.6, 13.4]† | 12.2 [4.86, 18.8]† |
| TaylorF2 | DiffGW Torch compiled | 0.293 [0.277, 0.333]† | 0.322 [0.296, 0.335]† | 0.368 [0.365, 0.371]† | 1.46 [1.43, 1.47]† |
| TaylorF2 | DiffGW JAX JIT | 0.308 [0.286, 1.18]† | 0.114 [0.108, 1.17]† | 1.38 [1.36, 1.42]† | 1.23 [1.21, 1.24]† |
| TaylorF2 | Ripple JAX JIT | 0.246 [0.229, 0.277]† | 0.156 [0.149, 0.184]† | 0.264 [0.23, 0.277]† | 1.29 [1.26, 1.31]† |
| IMRPhenomD | DiffGW Torch eager | 12 [11.8, 12.6] | 11.7 [11.2, 14.5] | 36.8 [35.1, 40.8] | 32.9 [19.2, 40] |
| IMRPhenomD | DiffGW Torch compiled | 1.54 [1.48, 1.58] | 0.776 [0.742, 0.796] | 1.51 [1.41, 1.58] | 3.48 [3.39, 3.59] |
| IMRPhenomD | DiffGW JAX JIT | 1.95 [1.88, 1.97] | 2.85 [2.09, 2.95] | 1.57 [1.55, 1.6] | 3.78 [3.73, 3.81] |
| IMRPhenomD | Ripple JAX JIT | 1.24 [1.23, 1.27] | 1.39 [1.34, 1.43]† | 1.37 [1.35, 1.39]† | 4.54 [4.51, 4.57]† |
| IMRPhenomXAS | DiffGW Torch eager | 75.7 [60.2, 98.6] | 69.4 [51.1, 97.7] | 76.6 [48.5, 109] | 130 [113, 139] |
| IMRPhenomXAS | DiffGW Torch compiled | 2.76 [2.61, 2.86] | 2.31 [2.17, 2.37] | 1.54 [1.48, 2.4] | 5.75 [5.58, 5.76] |
| IMRPhenomXAS | DiffGW JAX JIT | 2.59 [2.57, 2.62] | 3.48 [3.22, 4] | 1.58 [1.54, 1.63] | 4.7 [3.79, 4.74] |
| IMRPhenomXAS | Ripple JAX JIT | 2.75 [2.5, 2.77] | 1.58 [1.56, 1.63] | 2.58 [2.56, 2.72] | 5.34 [5.23, 5.37] |

The LAL CPU baseline is intentionally absent from the GPU table. Compare GPU
implementations on that table; a ratio to the CPU baseline combines hardware
and implementation differences.

## Reproduction and receipts

The receipts contain actual parameters, errors per binary/polarization, all raw
timings, first-call costs, bootstrap intervals, software/hardware provenance and
content/source/runner SHA-256 hashes:

- [CPU receipt](benchmarks/mac-20260921-cpu.json)
- [GPU receipt](benchmarks/mac-20260921-gpu.json)

Use the declared DiffGW source revision and the runner matching the receipt's
hash. The `--source-commit` argument is a provenance label; independently check
the source contents, especially when running from a working tree.

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export JAX_ENABLE_X64=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS=--xla_cpu_multi_thread_eigen=false

taskset -c 0 python scripts/compare_packages.py \
    --models TaylorF2 IMRPhenomD IMRPhenomXAS \
    --backends lal torch torch_compiled jax ripple \
    --device cpu --batches 1 10 100 1000 \
    --source-commit 5226141887ddd7394c8430537de916d871571358 \
    --record-unqualified --output cpu.json

taskset -c 0 python scripts/compare_packages.py \
    --models TaylorF2 IMRPhenomD IMRPhenomXAS \
    --backends torch torch_compiled jax ripple \
    --device cuda --batches 1 10 100 1000 \
    --source-commit 5226141887ddd7394c8430537de916d871571358 \
    --record-unqualified --output gpu.json
```

`taskset` is Linux-specific; use an available core and record affinity on other
hosts. LAL remains the CPU accuracy reference during the GPU campaign. See the
[benchmark protocol](benchmarking_protocol.md) for the broader qualification
requirements and the [historical performance report](performance.md) for the
separate, earlier FP32/FP64 campaign.

# DiffGW Performance & Benchmark Analysis Guide

This document provides a comprehensive analysis of the performance benchmarks, scaling characteristics, and publication figures for **`diffgw`** (`torchwave` and `jaxwave`).

For the formal benchmarking protocol, Gate 1 scientific qualification rules, and statistical methodology, see [**docs/benchmarking_protocol.md**](benchmarking_protocol.md).

---

## 1. Executive Summary & Hardware Setup

All benchmarks reported here were executed on a dedicated high-performance workstation under controlled conditions:
- **GPU**: NVIDIA GeForce RTX 5090 (32 GB GDDR7, Blackwell Architecture, SM 12.0)
- **CPU**: AMD Ryzen 9 9950X 16-Core / 32-Thread Processor @ 4.3 GHz (Boost to 5.7 GHz)
- **Execution Environments**:
  - **Linux (WSL2 Ubuntu 24.04)**: PyTorch 2.6.0+cu128, JAX 0.5.1+cuda12.cudnn9, LALSuite 7.6.0
  - **Windows 11 Native**: PyTorch 2.6.0+cu128, Ripple 0.0.3, LALSimulation 7.6.0
- **Controlled Frequency Grid**: $f_{\text{low}} = 20\text{ Hz}$, $f_{\text{high}} = 512\text{ Hz}$, $\Delta f = 0.25\text{ Hz}$ ($N_{\text{freq}} = 1,969$ bins).
- **Parity Gate**: Verified double-precision agreement against LALSuite ($L_2 \le 10^{-11}$, mismatch $1 - \mathcal{M} \le 10^{-14}$) prior to timing sweeps.

### Key Performance Highlights

| Milestone | Backend / Device | Approximant | Throughput | Amortized Latency |
| :--- | :--- | :--- | :--- | :--- |
| **Peak FP64 Throughput** | JAXWave CUDA (WSL2) | `TaylorF2` | **1,374,700 wf/s** | $0.73\ \mu\text{s / wf}$ |
| **Peak FP32 Throughput** | TorchWave CUDA (Native) | `TaylorF2` | **1,068,449 wf/s** | $0.94\ \mu\text{s / wf}$ |
| **Aligned-Spin Peak** | JAXWave CUDA (WSL2) | `IMRPhenomD` | **405,816 wf/s** | $2.46\ \mu\text{s / wf}$ |
| **Higher-Harmonic Peak** | JAXWave CUDA (WSL2) | `IMRPhenomXHM` | **46,285 wf/s** | $21.6\ \mu\text{s / wf}$ |
| **Maximum Speedup vs LAL** | TorchWave CUDA | `TaylorF2` ($B=5000$) | **$55.7\times$** | single-core C baseline |
| **Maximum Speedup vs LAL** | JAXWave CUDA | `IMRPhenomD` ($B=5000$) | **$164.7\times$** | single-core C baseline |

---

## 2. Walkthrough of Publication Figures

### Figure 1: Waveform Throughput Scaling Across Workloads
![Figure 1: Waveform Throughput Scaling](../artifacts/benchmarks/plots/fig1_throughput_vs_batch.png)

#### What This Plot Shows
Figure 1 plots generated waveforms per second (throughput, logarithmic y-axis) as a function of batch size $B \in [1, 5000]$ (logarithmic x-axis) across all five canonical approximants:
1. **`TaylorF2`**: Post-Newtonian point-particle inspiral.
2. **`IMRPhenomD`**: 2nd-generation aligned-spin inspiral-merger-ringdown.
3. **`IMRPhenomXAS`**: 4th-generation multi-collocation aligned-spin model.
4. **`IMRPhenomXP`**: Precessing-spin model with multi-scale analytical (MSA) precession angles.
5. **`IMRPhenomXHM`**: Multi-harmonic model evaluating 5 physical multipoles $(2,\pm 2), (2,\pm 1), (3,\pm 3), (3,\pm 2), (4,\pm 4)$.

#### Physical & Architectural Interpretation
- **CPU Ceilings ($B \ge 100$)**: Both PyTorch and JAX on CPU hit a throughput plateau between $5,000$ and $20,000\text{ wf/s}$ due to multi-core host memory bandwidth and vector register saturation.
- **GPU Scaling Regimes**:
  - **Launch Overhead Bound ($B < 10$)**: At small batch sizes, GPU throughput is constrained by CUDA driver kernel launch overhead ($\sim 10\text{--}50\ \mu\text{s}$).
  - **Linear Amortization ($10 \le B \le 1000$)**: As batch size expands, tensor arithmetic parallelizes across the RTX 5090's 170 Streaming Multiprocessors, scaling almost linearly with batch size.
  - **Memory Bandwidth & Compute Saturation ($B \ge 5000$)**: Throughput reaches its hardware maximum.
- **JAX XLA JIT vs. PyTorch Eager Execution**: JAXWave compiled with `@jax.jit` fuses multi-stage mathematical expressions (e.g. polynomial expansions, exponential phase shifts) into monolithic GPU kernels, reducing intermediate global memory round-trips. This delivers a $3.7\times$ to $26\times$ throughput advantage over eager PyTorch at large batches.
- **LALSimulation Single-Core Baseline**: Indicated by the red dotted horizontal line. At $B=1$, single-core LAL C code is competitive with or faster than unamortized GPU calls; at $B \ge 100$, GPU execution decisively outpaces LAL.

---

### Figure 2: Per-Waveform Latency & GPU Amortization Crossover
![Figure 2: Per-Waveform Generation Latency](../artifacts/benchmarks/plots/fig2_latency_vs_batch.png)

#### What This Plot Shows
Figure 2 plots the amortized per-waveform generation latency in milliseconds ($t_{\text{median}} / B$) across batch sizes for PyTorch on CPU (orange triangles) versus CUDA GPU (blue circles).

#### Key Takeaways: When to Use CPU vs. GPU
- **The Crossover Point ($B_{\text{cross}}$)**:
  - For `TaylorF2` and `IMRPhenomD`, the CPU and GPU curves cross at **$B \approx 50\text{--}100$**.
  - For `IMRPhenomXP`, the crossover occurs earlier at **$B \approx 10\text{--}20$** due to the heavy computational cost of precession frames.
- **Scalar Workloads ($B = 1$)**: CPU generation is $5\times$ to $15\times$ faster in wall-clock latency ($0.5\text{ ms}$ on CPU vs $5\text{ ms}$ on CUDA) because it avoids PCIe host-to-device dispatch and CUDA context synchronization.
- **Batched Workloads ($B \ge 1000$)**: CUDA amortizes generation latency down to **$0.003\text{--}0.02\text{ ms / waveform}$** ($3\text{--}20\ \mu\text{s}$), making GPU execution **$50\times\text{--}100\times$ faster per waveform** than single-core CPU execution.

---

### Figure 3: GPU Speedup Factor vs. LALSimulation Single-Core Baseline
![Figure 3: GPU Speedup vs LAL](../artifacts/benchmarks/plots/fig3_speedup_vs_lal.png)

#### What This Plot Shows
Figure 3 displays the effective acceleration factor ($t_{\text{LAL}} / t_{\text{amortized}}$) relative to the canonical single-core C `lalsimulation` implementation. The dashed horizontal line represents parity ($1\times$).

#### Quantitative Speedup Breakdown
- At $B = 1$: GPU execution operates below parity ($0.003\times\text{--}0.05\times$) due to kernel launch latency.
- At $B = 10$: GPU approaches parity ($0.04\times\text{--}0.5\times$).
- At $B = 100$: All models cross parity, delivering **$3\times\text{--}5\times$** acceleration over single-core LAL.
- At $B = 1,000$: Speedups reach **$3\times\text{--}12\times$** for precessing/higher-order models and **$12\times\text{--}30\times$** for aligned-spin models.
- At $B = 5,000$: Speedup reaches **$55.7\times$** in TorchWave and **$>160\times$** in JAXWave.

---

### Figure 4: Cross-Model Complexity & Throughput Breakdown
![Figure 4: Model Breakdown Bar Chart](../artifacts/benchmarks/plots/fig4_model_breakdown_bar.png)

#### What This Plot Shows
Figure 4 compares throughput across the five canonical models at three distinct operational scales:
- **Batch 1 (Light blue)**: Single-waveform evaluation (scalar alerts, serial MCMC steps).
- **Batch 100 (Medium blue)**: Parallel MCMC ensemble walkers and particle swarm optimization.
- **Batch 1,000 (Dark blue)**: Matched-filter bank screening and nested sampling likelihood batches.

#### Computational Complexity Hierarchy
The throughput directly reflects the analytical complexity of the underlying physics:
1. **`TaylorF2`** ($\sim 65,000\text{ wf/s}$ at $B=1000$ in Torch, $>695,000$ in JAX): Pure analytical post-Newtonian phasing without collocation collocation solves.
2. **`IMRPhenomD`** ($\sim 23,000\text{ wf/s}$ in Torch, $>260,000$ in JAX): 3-region piecewise collocation with 5 intermediate collocation points.
3. **`IMRPhenomXAS`** ($\sim 8,000\text{ wf/s}$ in Torch, $>208,000$ in JAX): 4th-generation multi-scale piecewise phasing with higher-order pseudo-PN coefficients.
4. **`IMRPhenomXP`** ($\sim 2,000\text{ wf/s}$ in Torch, $>3,200$ in JAX): 3D spin precession requiring Euler angle integration ($\alpha, \beta, \gamma$) and frame rotation into the coprecessing frame.
5. **`IMRPhenomXHM`** ($\sim 640\text{ wf/s}$ in Torch, $>40,000$ in JAX): Simultaneous evaluation of 5 spherical harmonic multipoles $(l, |m|)$ with mode-mixing corrections for $(3,2)$.

---

## 3. Ultra-Large Batch Scaling ($B = 1,000 \to 50,000$) & Memory Limits

To establish the upper bounds of GPU scalability, we conducted an unconstrained batch sweep up to $B = 50,000$ on the RTX 5090 ($32\text{ GB}$ VRAM) using [`scripts/benchmark_batch_scaling.py`](../scripts/benchmark_batch_scaling.py).

### Single-Precision (FP32) Scaling & VRAM Consumption

| Model | Batch Size ($B$) | Latency (ms) | Throughput (wf/s) | Peak VRAM (MB) | VRAM / Waveform | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **TaylorF2** | $1,000$ | $3.72$ | $269,005$ | $124.8\text{ MB}$ | $127.8\text{ KB}$ | **PASS** |
| **TaylorF2** | $10,000$ | $14.21$ | $703,576$ | $1,178.5\text{ MB}$ | $120.7\text{ KB}$ | **PASS** |
| **TaylorF2** | $50,000$ | $46.80$ | **$1,068,449$** | $5,843.2\text{ MB}$ | $119.7\text{ KB}$ | **PASS** |
| **IMRPhenomD** | $1,000$ | $11.94$ | $83,725$ | $418.1\text{ MB}$ | $428.2\text{ KB}$ | **PASS** |
| **IMRPhenomD** | $10,000$ | $33.16$ | $301,561$ | $4,103.6\text{ MB}$ | $420.2\text{ KB}$ | **PASS** |
| **IMRPhenomD** | $50,000$ | $135.96$ | **$367,751$** | $20,331.3\text{ MB}$ | $416.4\text{ KB}$ | **PASS** |
| **IMRPhenomXAS** | $1,000$ | $48.14$ | $20,774$ | $272.4\text{ MB}$ | $278.9\text{ KB}$ | **PASS** |
| **IMRPhenomXAS** | $10,000$ | $62.92$ | $158,936$ | $2,677.1\text{ MB}$ | $274.1\text{ KB}$ | **PASS** |
| **IMRPhenomXAS** | $50,000$ | $173.74$ | **$287,782$** | $13,230.3\text{ MB}$ | $271.0\text{ KB}$ | **PASS** |
| **IMRPhenomXHM** | $1,000$ | $651.07$ | $1,536$ | $551.2\text{ MB}$ | $564.5\text{ KB}$ | **PASS** |
| **IMRPhenomXHM** | $10,000$ | $733.13$ | $13,640$ | $5,472.4\text{ MB}$ | $560.4\text{ KB}$ | **PASS** |
| **IMRPhenomXHM** | $50,000$ | $1,591.16$ | **$31,424$** | $27,189.1\text{ MB}$ | $556.8\text{ KB}$ | **PASS** |
| **IMRPhenomXP** | $1,000$ | $349.38$ | $2,862$ | $3,421.6\text{ MB}$ | $3,503.7\text{ KB}$ | **PASS** |
| **IMRPhenomXP** | $5,000$ | $1,349.83$ | **$3,704$** | $16,965.7\text{ MB}$ | $3,474.6\text{ KB}$ | **PASS** |
| **IMRPhenomXP** | $25,000$ | -- | -- | $>32\text{ GB}$ | -- | **OOM Contained** |

### Memory Sizing Guidelines
- **Linear Memory Footprint**: For all models, peak VRAM scales strictly linearly with batch size ($R^2 > 0.999$).
- **Memory Footprint per Waveform**:
  - `TaylorF2`: $\sim 120\text{ KB / waveform}$
  - `IMRPhenomXAS`: $\sim 270\text{ KB / waveform}$
  - `IMRPhenomD`: $\sim 420\text{ KB / waveform}$
  - `IMRPhenomXHM`: $\sim 560\text{ KB / waveform}$
  - `IMRPhenomXP`: $\sim 3,480\text{ KB / waveform}$ ($3.5\text{ MB / waveform}$ due to full 3D precession Euler angle arrays)
- **Safe Batch Size Limits on 16GB / 24GB / 32GB GPUs**:
  - On a **16 GB GPU** (e.g. RTX 4080): Safe batch size is $B \le 30,000$ for aligned-spin models, and $B \le 4,000$ for precessing `IMRPhenomXP`.
  - On a **24 GB GPU** (e.g. RTX 3090 / 4090): Safe batch size is $B \le 50,000$ for aligned-spin models, and $B \le 6,000$ for precessing `IMRPhenomXP`.
  - On a **32 GB GPU** (e.g. RTX 5090): Safe batch size is $B \le 75,000$ for aligned-spin models, and $B \le 8,000$ for precessing `IMRPhenomXP`.

---

## 4. Reproducing Benchmarks & Generating Plots

### Step 1: Run Multi-Backend Benchmark Campaign
To run a standardized benchmark campaign across PyTorch, JAX, LAL, and Ripple:

```bash
# Run benchmark campaign with Gate 1 verification and cryptographic sealing
python scripts/run_benchmark_campaign.py \
    --models TaylorF2,IMRPhenomD,IMRPhenomXAS,IMRPhenomXP,IMRPhenomXHM \
    --backends torchwave,jaxwave,lalsimulation,ripple \
    --devices cuda,cpu \
    --batch-sizes 1,10,100,1000,5000 \
    --iterations 25 \
    --warmup 5 \
    --output artifacts/benchmarks/benchmark_receipt.json
```

### Step 2: Render Publication-Grade Figures
To generate the four standardized figures (PNG at 300 DPI and vector PDF) from one or more merged receipts:

```bash
python scripts/plot_benchmark_results.py \
    --input artifacts/benchmarks/benchmark_receipt_mac_rtx5090.json,artifacts/benchmarks/benchmark_receipt_wsl2_rtx5090.json \
    --output-dir artifacts/benchmarks/plots
```

The script automatically:
1. Loads and deduplicates benchmark entries across receipts.
2. Formats all labels and colors consistently.
3. Moves legends outside the axes to prevent curve or data occlusion.
4. Generates both `.png` and `.pdf` versions suitable for inclusion in LaTeX papers.

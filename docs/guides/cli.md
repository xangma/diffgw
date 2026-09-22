# Command-line tools

Command-line operations and the reproducible benchmark entry points.

<a id="12-command-line-tools"></a>

TorchWave registers five standalone console executables:

### `torchwave-bench`
Benchmark model throughput, batch sizes, autograd backward passes, and cross-framework GPU speedups:
```bash
# Benchmark standard models across batch sizes
torchwave-bench --models popular --batch-sizes 1,10,100,1000

# Benchmark across backends (Torch eager, Torch compiled, JAX JIT, or all)
torchwave-bench --models IMRPhenomD --backend all --batch-sizes 1,10,100

# Benchmark compiled PyTorch with dynamic shapes (can reduce recompilation across batch sizes)
torchwave-bench --models TaylorF2,IMRPhenomD --compile --dynamic --batch-sizes 1,10,50,100

# Measure forward + backward autograd gradient time
torchwave-bench --models IMRPhenomD --backend torch --batch-sizes 100 --backward

# Compare against LALSimulation baseline and export Markdown table
torchwave-bench --models TaylorF2,IMRPhenomD,IMRPhenomXAS --compare-lal --markdown-out benchmark.md
```

### `torchwave-profile`
Profile operator kernels, execution bottlenecks, and memory allocation:
```bash
# Run comprehensive profiling (cProfile, torch profiler, memory)
torchwave-profile --model IMRPhenomD --mode all --top-k 15

# Export Chrome trace JSON for visualization in chrome://tracing
torchwave-profile --model TaylorF2 --trace-out trace_f2.json
```

### `torchwave-data`
Manage optional external ROM datasets outside git repositories and wheels:
```bash
# List available external datasets
torchwave-data list

# Inspect dataset metadata
torchwave-data info seobnrv4-rom-v3

# Fetch dataset to user cache or custom directory
torchwave-data fetch seobnrv4-rom-v3

# Verify pinned SHA-256 integrity
torchwave-data verify seobnrv4-rom-v3
```

### `torchwave-generate-qnm`
Generate Kerr quasinormal mode tables directly from the Leaver continued-fraction eigenproblem:
```bash
torchwave-generate-qnm --output-dir ./qnm_data --modes 2,2,0
```

### `torchwave-generate-london-qnm`
Recompute the London $N=61$ quasinormal mode polynomial fits:
```bash
torchwave-generate-london-qnm --output-dir ./london_qnm
```

---


## Controlled Benchmarking Protocol

To reproduce published results, use `scripts/compare_packages.py` with the
[dated campaign's commands and environment](../package_benchmarks.md#reproduction-and-receipts).
Regenerate its figures with the [maintained plot command](../package_benchmarks.md#regenerating-the-plots).
For a new experiment, follow the [benchmark methodology](../benchmarking_protocol.md).

### Running a Multi-Backend Benchmark Campaign

The [runner guide](../benchmarking_protocol.md#choose-a-runner) distinguishes the
maintained package comparison, local CLI experiments and legacy campaigns.
The older `run_benchmark_campaign.py` can use JAXWave or self-consistency as its
reference; its `gate_passed` flag alone does not establish LAL agreement.

### Generating Figures

Use the [published campaign's plot instructions](../package_benchmarks.md#regenerating-the-plots)
for maintained CPU/GPU figures. `plot_benchmark_results.py` reads the separate
legacy receipt format; see the [legacy qualification rules](../benchmarking_protocol.md#legacy-campaigns).

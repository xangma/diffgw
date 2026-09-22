# Execution and differentiation

Backend, device, precision, compilation, batching, and autodiff guidance.

<a id="3-hardware-acceleration-precision"></a>
## Hardware acceleration and precision

The [published FP64 benchmarks](../package_benchmarks.md) cover CPU and NVIDIA CUDA for both frontends. Other framework targets are not established by those results. PyTorch MPS does not support FP64; use CPU for the FP64 examples on Apple Silicon. Model operations and available precision determine device compatibility.

### PyTorch Device & Precision
```python
device = "cuda" if torch.cuda.is_available() else "cpu"

hp_f64, hc_f64 = torchwave.get_fd_waveform(
    approximant="IMRPhenomD",
    mass1=30.0, mass2=25.0,
    f_lower=20.0, f_final=512.0, delta_f=0.25,
    device=device,
    dtype=torch.float64,
)
```

---


<a id="jax-compilation-batching-and-transforms"></a>

<a id="5-jax-frontend-acceleration-jaxwave"></a>
## JAX frontend and acceleration (`jaxwave`)

`jaxwave` provides tensor waveform models for JAX scientific computing and machine learning workflows. Transformation and device support depend on the selected path; for example, the Kerr angular eigensolver uses a host CPU callback.

### Waveform Generation & `WaveformGenerator`
Generate waveforms using the direct functional interface or preconfigured generator:

```python
import jaxwave

# Direct functional interface
hp, hc = jaxwave.get_fd_waveform(
    approximant="IMRPhenomD",
    mass1=30.0, mass2=25.0,
    chi1=0.2, chi2=-0.1,
    f_lower=20.0, f_final=512.0, delta_f=0.25,
)

# Class-based generator for repeated evaluations (supports automatic JIT caching)
gen = jaxwave.WaveformGenerator(
    approximant="IMRPhenomXAS",
    f_lower=20.0, f_final=512.0, delta_f=0.25,
    use_jit=True,
)
hp, hc = gen.generate_fd_waveform(mass1=30.0, mass2=25.0, chi1=0.1, chi2=0.0)
```

### Just-In-Time Compilation (`@jax.jit`)
The following IMRPhenomD path supports JAX tracing and XLA compilation. Transformation support is model- and option-dependent; external-data callbacks and numerical solvers require separate checks:

```python
import jax
import jaxwave

@jax.jit
def fast_waveform(m1, m2, s1z, s2z):
    return jaxwave.get_fd_waveform(
        approximant="IMRPhenomD",
        mass1=m1, mass2=m2, chi1=s1z, chi2=s2z,
        f_lower=20.0, f_final=512.0, delta_f=0.5,
    )

# First call compiles; compatible subsequent calls reuse the compiled function
hp, hc = fast_waveform(35.0, 25.0, 0.2, -0.1)
```

### Vectorized Batch Evaluation (`jax.vmap`)
Map waveform generation across parameter grids without Python loops:

```python
import jax
import jax.numpy as jnp
import jaxwave

m1_grid = jnp.array([20.0, 30.0, 40.0])
m2_grid = jnp.array([15.0, 25.0, 35.0])
s1_grid = jnp.array([0.1, 0.2, -0.3])
s2_grid = jnp.array([-0.1, 0.0, 0.2])

vmapped_fn = jax.vmap(fast_waveform, in_axes=(0, 0, 0, 0))
hp_batch, hc_batch = vmapped_fn(m1_grid, m2_grid, s1_grid, s2_grid)
print("JAX batched shape:", hp_batch.shape)  # (3, 985)
```

### Autodiff Parameter Gradients (`jax.grad`)
Compute derivatives $\partial h / \partial \theta$ of the implemented numerical model using automatic differentiation. Piecewise branches, interpolation, solver settings, and supported derivative orders still matter:

```python
import jax
import jax.numpy as jnp

def waveform_power(m1, m2):
    hp, hc = fast_waveform(m1, m2, 0.2, -0.1)
    return jnp.sum(jnp.abs(hp) ** 2)

# Compute gradient vector [dPower/dm1, dPower/dm2]
grad_fn = jax.grad(waveform_power, argnums=(0, 1))
dm1, dm2 = grad_fn(30.0, 25.0)
print(f"Gradients: dm1={dm1:.4e}, dm2={dm2:.4e}")
```

### Shape Bucketing for MCMC Sampling (`BucketedWaveformEvaluator`)

In MCMC sampling, parallel tempering, or ensemble walkers where batch sizes vary dynamically, standard `@jax.jit` functions recompile whenever a new batch size is encountered.

`jaxwave.inference.BucketedWaveformEvaluator` reduces the number of distinct batch shapes by padding inputs to discrete buckets, caching each bucket evaluator, and unpadding its outputs. A new bucket compiles on first use; changes to dtype or other traced shapes may also trigger compilation:

```python
from jaxwave.inference import BucketedWaveformEvaluator
import jax.numpy as jnp

evaluator = BucketedWaveformEvaluator(
    approximant="IMRPhenomD",
    f_lower=20.0, f_final=512.0, delta_f=0.5,
    buckets=(1, 4, 8, 16, 32, 64, 128),
)

# Evaluating 13 walkers pads to bucket 16, using the cached JIT kernel
m1_walkers = jnp.array([30.0] * 13)
m2_walkers = jnp.array([25.0] * 13)
hp, hc = evaluator.evaluate(mass1=m1_walkers, mass2=m2_walkers)
print(f"Output walkers: {hp.shape[0]}")  # 13
```

### Unified Dispatch via `diffgw`
Use `diffgw.get_fd_waveform` to select backends dynamically:

```python
import diffgw

hp_t, hc_t = diffgw.get_fd_waveform("IMRPhenomD", mass1=30.0, mass2=25.0, backend="torch")
hp_j, hc_j = diffgw.get_fd_waveform("IMRPhenomD", mass1=30.0, mass2=25.0, backend="jax")
```

---


<a id="9-automatic-differentiation-in-pytorch"></a>
## Automatic differentiation in PyTorch

```python
import torch
import torchwave

m1 = torch.tensor(35.0, requires_grad=True, dtype=torch.float64)
m2 = torch.tensor(25.0, requires_grad=True, dtype=torch.float64)

hp, _ = torchwave.get_fd_waveform(
    approximant="IMRPhenomD",
    mass1=m1, mass2=m2,
    f_lower=20.0, f_final=256.0, delta_f=1.0,
)
loss = torch.sum(torch.abs(hp) ** 2)
loss.backward()

print("d(loss)/dm1:", m1.grad.item())
print("d(loss)/dm2:", m2.grad.item())
```

---

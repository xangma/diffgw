# Parameter estimation

`diffgw.inference` provides a public PE workflow for both native frontends:
device-resident dense detector likelihoods, normalized uniform box priors,
differentiable posteriors and NUTS sampling. Torch uses diffGW's native
`torchwave.NUTS`; JAX uses the optional NumPyro sampler.

## Install and choose a device

```bash
pip install 'diffgw[torch]'    # native Torch PE
pip install 'diffgw[pe-jax]'   # JAX PE with NumPyro
```

Install a CUDA-enabled Torch or JAX runtime for your GPU using the
[PyTorch installation guide](https://pytorch.org/get-started/locally/) or
[JAX installation guide](https://docs.jax.dev/en/latest/installation.html).
The PE extra provides software dependencies; it does not install GPU drivers.

Use `backend="torch", device="cuda"` or
`backend="jax", device="gpu"`. Indexed devices (`cuda:1`, `gpu:1`) are supported.
An explicit unavailable device raises an error. With no device argument, Torch
uses CPU and JAX uses its default device. PE uses float64/complex128; MPS is not
supported by this interface. Keep your waveform grids, detector network and
likelihood inputs on the same device as the posterior.

## Define a posterior and sample

This small analytic example exercises the same public sampler API as a waveform
likelihood. The parameter order follows the prior dictionary's insertion order.

```python
from diffgw.inference import UniformBoxPrior, Posterior, run_nuts

prior = UniformBoxPrior({"x": (-5., 5.), "y": (-6., 8.)})
posterior = Posterior(
    lambda p: -0.5 * ((p["x"] - 0.7) / 0.8)**2
              -0.5 * ((p["y"] + 0.4) / 1.1)**2,
    prior, backend="torch", device="cuda",
)
result = run_nuts(
    posterior, [[0., 0.], [1.2, -1.]],
    num_warmup=500, num_samples=1000, seed=54,
)
samples = result.posterior_samples   # named native arrays, shape (chains, draws)
host = result.to_numpy()             # explicit GPU-to-host export
```

The likelihood callback receives named native scalar parameters and must return
one real scalar. Preserve its autodiff graph; converting parameters to Python
floats or NumPy breaks gradients. JAX callbacks must also support JIT. Torch
sampling rejects a detached initial likelihood. For a deliberately constant
likelihood, connect a zero term to a parameter, such as `0 * p["x"]`.

`Posterior.log_prob(theta)` evaluates the physical density with the normalized
prior, returning negative infinity outside support. NUTS uses an unconstrained
sigmoid chart and includes the transform Jacobian exactly once. Initial positions
are physical values and must lie strictly inside every prior bound. This first
interface supports independent finite uniform priors; correlated, periodic and
other prior families require a separately defined posterior through the native
frontend APIs.

Chains execute sequentially on the selected device. Both samplers adapt their
step sizes during warmup and keep the diagonal inverse mass fixed. An optional
`inverse_mass` vector refers to unconstrained chart coordinates. Torch's sampler
has Python control flow and device synchronization, so GPU execution alone does
not guarantee a speedup; profile your actual likelihood and chain workload.

## Torch target execution

Torch CUDA posteriors compile their complete unconstrained log density by default
with `torch.compile(fullgraph=True, dynamic=False, mode="reduce-overhead")`.
The initial example already uses this path. Torch CPU stays eager by default;
JAX uses NumPyro JIT. NUTS tree building, random draws and adaptation stay unchanged.

```python
# Explicit eager control or a callback unsupported by the Torch compiler:
eager = run_nuts(posterior, [0., 0.], torch_compile=False)

# Ordinary compilation, without the reduce-overhead mode:
compiled = run_nuts(
    posterior, [0., 0.], torch_compile=True, torch_compile_mode="default",
)
```

`torch_compile=None` and `torch_compile_mode=None` select these device-aware
defaults. `torch_compile=True` enables compilation explicitly: CUDA uses
`reduce-overhead`, CPU uses ordinary compilation. `torch_compile=False` disables
Torch compilation. An explicit `torch_compile_mode="default"` selects ordinary
compilation when enabled; `reduce-overhead` requires enabled Torch compilation
on CUDA. `torch_compile=True` rejects JAX posteriors because NumPyro already
JIT-compiles their target.

The callback and its waveform operations must support the installed Torch compiler
for both values and gradients. Compiler errors propagate; the API provides no
managed fallback or model substitution. Use `torch_compile=False` explicitly
when eager execution is intended. Initial gradient and finite-value checks still
run eagerly. Compilation has a cold cost and may produce separate backward or
input-context graphs; measure it separately from warmed target/chain throughput.
This option does not establish a GPU speedup or PE qualification.
CUDA graph mode requires callable `torch.compiler.cudagraph_mark_step_begin`
support. Unsupported Torch runtimes raise an error; upgrade Torch or select
`torch_compile=False` explicitly.

For `reduce-overhead`, the native sampler marks each complete target
value/gradient pair as a new graph step and copies both outputs so retained tree
states survive later replays. This adds copy cost and does not promise graph
capture or a speedup for every callback/runtime.
`metadata["target_execution"]` records `eager`, `torch_compile`, or
`jax_numpyro_jit`; `metadata["torch_compile_mode"]` records the resolved compiled
mode, or `None` when this API does not compile the target.

## Connect detector data

`WhittleLikelihood` constructs the existing native likelihood after placing its
data and cached terms on the selected device. Data and PSDs are matching detector
dictionaries on a shared one-dimensional frequency grid. Data are complex128;
PSDs must be finite and positive.

```python
from diffgw.inference import WhittleLikelihood, Posterior, UniformBoxPrior

likelihood = WhittleLikelihood(
    data=data, psds=psds, df=delta_f, backend="torch", device="cuda",
)
posterior = Posterior(
    lambda p: likelihood(make_detector_templates(p)),
    UniformBoxPrior({"mass1": (29.5, 30.5), "mass2": (24.5, 25.5)}),
    backend="torch", device="cuda",
)
```

Supply **either** `df` for rectangular integration **or** `frequencies` for
trapezoidal integration. These have different endpoint weights. Frequency cuts
require explicit frequencies. Construct the likelihood on its intended device;
use the returned native object for advanced likelihood operations. See
[detector and inference APIs](inference.md) for projections and existing relative
binning, Fisher and optimization interfaces. Validate any approximate likelihood
against a trustworthy dense reference over the intended posterior region.

The maintained [waveform example](../tutorials/04_public_parameter_estimation.md)
samples the two component masses of IMRPhenomD from zero-noise H1/L1 data. It runs
in either frontend and exports samples, metadata and diagnostics.

## Interpret results

`PEResult.samples` has shape `(chains, draws, parameters)` in physical units.
`posterior_samples` provides named native arrays. `diagnostics` retains sampler
transition data; `metadata` records backend, device, versions, prior bounds,
seed and sampler options. `to_numpy()` explicitly copies arrays to host memory.

Torch diagnostics include warmup and retained transitions, with a `warmup` mask.
NumPyro diagnostics contain retained transitions only; its warmup divergences
are unobserved here. Metadata identifies the diagnostic phase. Inspect divergences,
tree depth, rank-normalized split R-hat, ESS, MCSE and independent posterior
functionals for your scientific run; successful execution is not a convergence
decision. This interface does not calculate evidence or automatically qualify
a waveform.

The [current PE results](../validation/inference/parameter-estimation-results.md)
explain the separately scoped reference, sampler and calibration evidence. Its
restricted two-parameter self-injections do not establish full-dimensional
inference or physical accuracy against independent signals. Package API tests,
GPU smoke tests and campaign qualification remain separate claims.

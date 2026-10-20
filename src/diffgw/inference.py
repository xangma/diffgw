"""Parameter estimation with independently optional Torch and JAX backends.

Likelihoods and samples stay on the selected device in float64/complex128.
Importing this module does not import either framework or a sampler dependency.
"""

from dataclasses import dataclass
from importlib import import_module
from types import MappingProxyType
from typing import Callable, Mapping

import numpy as np

__all__ = ["UniformBoxPrior", "Posterior", "WhittleLikelihood", "run_nuts", "PEResult"]


def _runtime(backend, device):
    from diffgw import _backend

    _backend(backend)
    framework = import_module(backend)
    if backend == "torch":
        selected = framework.device("cpu" if device is None else device)
        if selected.type not in ("cpu", "cuda"):
            raise ValueError("PE supports float64 on 'cpu' or 'cuda[:index]'")
        if selected.type == "cpu":
            if selected.index not in (None, 0):
                raise ValueError(f"Requested PE device {selected} is unavailable")
            selected = framework.device("cpu")
        if selected.type == "cuda":
            if not framework.cuda.is_available():
                raise ValueError(f"Requested PE device {selected} is unavailable")
            index = selected.index if selected.index is not None else framework.cuda.current_device()
            if index >= framework.cuda.device_count():
                raise ValueError(f"Requested PE device {selected} is unavailable")
            selected = framework.device("cuda", index)
        return framework, selected
    # The existing JAX frontend also enables x64. Make the public PE contract explicit.
    framework.config.update("jax_enable_x64", True)
    if device is None:
        selected = framework.devices()[0]
        if selected.platform not in ("cpu", "gpu"):
            raise ValueError("PE supports CPU or GPU devices")
        return framework, selected
    if not isinstance(device, str):
        platform = getattr(device, "platform", None)
        if platform not in ("cpu", "gpu") or device not in framework.devices(platform):
            raise ValueError("Requested JAX PE device is unavailable")
        return framework, device
    platform, _, suffix = device.partition(":")
    platform = "gpu" if platform == "cuda" else platform
    if platform not in ("cpu", "gpu"):
        raise ValueError("PE supports 'cpu' or 'gpu[:index]' for JAX")
    try:
        index = int(suffix) if suffix else 0
        devices = framework.devices(platform)
        if index < 0:
            raise ValueError
        selected = devices[index]
    except (RuntimeError, ValueError, IndexError) as exc:
        raise ValueError(f"Requested PE device {device!r} is unavailable") from exc
    return framework, selected


def _array(value, backend, framework, device, *, complex=False):
    if not complex:
        if backend == "torch":
            is_complex = framework.is_complex(value) if isinstance(value, framework.Tensor) else np.iscomplexobj(value)
        else:
            is_complex = framework.numpy.iscomplexobj(value)
        if is_complex:
            raise ValueError("Parameters, PSDs and frequency grids must be real")
    if backend == "torch":
        dtype = framework.complex128 if complex else framework.float64
        return framework.as_tensor(value, dtype=dtype, device=device)
    with framework.default_device(device):
        array = framework.numpy.asarray(value, dtype=framework.numpy.complex128 if complex else framework.numpy.float64)
        return framework.device_put(array, device)


def _numpy(value):
    """Explicit host export; never used on a traced sampler target."""
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


@dataclass(frozen=True, init=False)
class UniformBoxPrior:
    """Normalized independent uniform priors in insertion-ordered physical units.

    ``UniformBoxPrior({"mass1": (25, 35), "mass2": (15, 24)})`` defines
    parameter order as ``("mass1", "mass2")``. Bounds must be finite and distinct.
    """

    names: tuple
    lower: tuple
    upper: tuple

    def __init__(self, bounds: Mapping[str, tuple]):
        if not bounds or any(not isinstance(name, str) or not name for name in bounds):
            raise ValueError("Provide a nonempty mapping of parameter names to bounds")
        if np.iscomplexobj(list(bounds.values())):
            raise ValueError("Prior bounds must be real")
        pairs = np.asarray(list(bounds.values()), dtype=np.float64)
        if pairs.shape != (len(bounds), 2) or not np.isfinite(pairs).all() or not (pairs[:, 0] < pairs[:, 1]).all():
            raise ValueError("Each prior bound must be a finite (lower, upper) pair with lower < upper")
        if not np.isfinite(pairs[:, 1] - pairs[:, 0]).all():
            raise ValueError("Prior widths must be finite")
        object.__setattr__(self, "names", tuple(bounds))
        object.__setattr__(self, "lower", tuple(pairs[:, 0]))
        object.__setattr__(self, "upper", tuple(pairs[:, 1]))

    @property
    def bounds(self):
        return MappingProxyType(dict(zip(self.names, zip(self.lower, self.upper))))


class Posterior:
    """A differentiable scalar log likelihood and a uniform box prior.

    The callback receives a dictionary of native scalar arrays, keyed by prior
    name. Keep waveform grids, templates and likelihood inputs on ``device``;
    do not convert callback values to NumPy or Python floats. JAX callbacks must
    also support JIT. Sampling uses a sigmoid chart with its exact Jacobian.
    """

    def __init__(self, log_likelihood: Callable, prior: UniformBoxPrior, *, backend="torch", device=None):
        if not callable(log_likelihood) or not isinstance(prior, UniformBoxPrior):
            raise TypeError("Posterior requires a callable likelihood and UniformBoxPrior")
        self.backend = backend
        self._framework, self.device = _runtime(backend, device)
        self.prior = prior
        self.names = prior.names
        self.log_likelihood = log_likelihood
        self._lower = self.asarray(prior.lower)
        self._upper = self.asarray(prior.upper)
        self._width = self.asarray(np.subtract(prior.upper, prior.lower))

    def asarray(self, value):
        """Convert setup values to float64 on this posterior's device."""
        return _array(value, self.backend, self._framework, self.device)

    def _check_shape(self, value, *, vector=False):
        if value.ndim < 1 or value.shape[-1] != len(self.names) or (vector and value.ndim != 1):
            raise ValueError(f"Expected {'a vector' if vector else 'last dimension'} of {len(self.names)} parameters")

    def parameters(self, theta):
        """Map a physical parameter vector to native scalar values."""
        self._check_shape(theta, vector=True)
        return dict(zip(self.names, theta))

    def _likelihood(self, theta):
        value = self.log_likelihood(self.parameters(theta))
        if self.backend == "torch":
            if not isinstance(value, self._framework.Tensor) or value.numel() != 1 or value.is_complex():
                raise ValueError("Likelihood must return one real native scalar")
            if value.device != self.device:
                raise ValueError("Likelihood output must be on the posterior device")
        else:
            if (not isinstance(value, (self._framework.Array, self._framework.core.Tracer))
                    or value.size != 1 or self._framework.numpy.iscomplexobj(value)):
                raise ValueError("Likelihood must return one real native scalar")
            if not isinstance(value, self._framework.core.Tracer) and value.device != self.device:
                raise ValueError("Likelihood output must be on the posterior device")
        return value.reshape(())

    def from_unconstrained(self, z):
        """Map chart coordinates to physical units; accepts leading sample axes."""
        self._check_shape(z)
        xp = self._framework if self.backend == "torch" else self._framework.numpy
        sigmoid = self._framework.sigmoid if self.backend == "torch" else self._framework.nn.sigmoid
        return xp.where(z <= 0, self._lower + self._width * sigmoid(z),
                        self._upper - self._width * sigmoid(-z))

    def to_unconstrained(self, theta):
        """Inverse chart. Values must lie strictly inside the prior for sampling."""
        self._check_shape(theta)
        xp = self._framework if self.backend == "torch" else self._framework.numpy
        # Distances preserve representable interior points close to either edge;
        # dividing by width first can round a valid interior coordinate to one.
        return xp.log(theta - self._lower) - xp.log(self._upper - theta)

    def log_prob(self, theta):
        """Physical log posterior, normalized prior, unnormalized likelihood."""
        self._check_shape(theta, vector=True)
        xp = self._framework if self.backend == "torch" else self._framework.numpy
        inside = xp.all((theta >= self._lower) & (theta <= self._upper)) & xp.all(xp.isfinite(theta))
        if self.backend == "torch":
            if not bool(inside):
                return self.asarray(float("-inf"))
            return self._likelihood(theta) - xp.log(self._width).sum()
        return self._framework.lax.cond(
            inside, lambda t: self._likelihood(t) - xp.log(self._width).sum(),
            lambda t: xp.asarray(-xp.inf, dtype=xp.float64), theta,
        )

    def log_prob_unconstrained(self, z):
        """Chart density, including the sigmoid Jacobian exactly once."""
        self._check_shape(z, vector=True)
        theta = self.from_unconstrained(z)
        # Normalized uniform density cancels the affine width in the Jacobian.
        if self.backend == "torch":
            softplus = self._framework.nn.functional.softplus
            return self._likelihood(theta) - (softplus(z) + softplus(-z)).sum()
        softplus = self._framework.nn.softplus
        return self._likelihood(theta) - (softplus(z) + softplus(-z)).sum()


def WhittleLikelihood(data, psds, *, backend="torch", device=None, df=None, frequencies=None, f_low=None, f_high=None):
    """Construct the native dense detector likelihood on a selected device.

    Data and PSDs are matching dictionaries of 1D arrays. Use ``df`` for a
    rectangular sum OR ``frequencies`` for trapezoidal integration. Data are
    complex128 and PSDs float64. Frequency cuts require explicit frequencies.
    Returns the native frontend object, including its advanced likelihood API.
    """
    if not data or set(data) != set(psds):
        raise ValueError("Data and PSDs need matching nonempty detector keys")
    if (df is None) == (frequencies is None):
        raise ValueError("Provide exactly one of df or frequencies")
    if df is not None and (not np.isfinite(df) or df <= 0):
        raise ValueError("df must be finite and positive")
    if frequencies is None and (f_low is not None or f_high is not None):
        raise ValueError("Frequency cuts require frequencies")
    framework, selected = _runtime(backend, device)
    converted_data = {key: _array(value, backend, framework, selected, complex=True) for key, value in data.items()}
    converted_psds = {key: _array(value, backend, framework, selected) for key, value in psds.items()}
    shape = next(iter(converted_data.values())).shape
    for key in data:
        d, p = converted_data[key], converted_psds[key]
        # Setup-only validation; sampler evaluations never export to the host.
        if len(shape) != 1 or shape[0] < 2 or d.shape != shape or p.shape != shape:
            raise ValueError("Data and PSDs must share a 1D frequency grid of at least two points")
        if not np.isfinite(_numpy(d)).all() or not np.isfinite(_numpy(p)).all() or not (_numpy(p) > 0).all():
            raise ValueError("Data must be finite and PSDs finite and positive")
    f = None if frequencies is None else _array(frequencies, backend, framework, selected)
    if f is not None:
        host_f = _numpy(f)
        if f.shape != shape or not np.isfinite(host_f).all() or not (np.diff(host_f) > 0).all():
            raise ValueError("Frequencies must match data and be finite and strictly increasing")
        if any(cut is not None and not np.isfinite(cut) for cut in (f_low, f_high)):
            raise ValueError("Frequency cuts must be finite")
        if f_low is not None and f_high is not None and f_low > f_high:
            raise ValueError("f_low must not exceed f_high")
    native = import_module(f"{'torchwave' if backend == 'torch' else 'jaxwave'}.inference.likelihood")
    if backend == "jax":
        with framework.default_device(selected):
            return native.WhittleLikelihood(converted_data, converted_psds, df=df, frequencies=f, f_low=f_low, f_high=f_high)
    return native.WhittleLikelihood(converted_data, converted_psds, df=df, frequencies=f, f_low=f_low, f_high=f_high)


@dataclass
class PEResult:
    """Native physical samples (chain, draw, parameter) and sampler diagnostics.

    Diagnostics describe sampler transitions; they do not certify convergence.
    ``metadata['diagnostic_phase']`` identifies whether warmup was retained.
    """

    samples: object
    names: tuple
    diagnostics: dict
    metadata: dict

    @property
    def posterior_samples(self):
        return {name: self.samples[..., index] for index, name in enumerate(self.names)}

    def to_numpy(self):
        """Explicitly copy physical samples and diagnostics to host NumPy arrays."""
        return {"samples": _numpy(self.samples).copy(), "names": self.names,
                "diagnostics": {key: _numpy(value).copy() for key, value in self.diagnostics.items()},
                "metadata": dict(self.metadata)}


def _torch_nuts_type(framework, cuda_graphs=False):
    """Return native NUTS, owning cached outputs when CUDA graphs are selected."""
    from torchwave.inference.mcmc import NUTS

    if not cuda_graphs:
        return NUTS

    class _CudaGraphNUTS(NUTS):
        def value_and_grad(self, theta):
            framework.compiler.cudagraph_mark_step_begin()
            log_p, grad_v = super().value_and_grad(theta)
            return log_p.detach().clone(), grad_v.detach().clone()

    return _CudaGraphNUTS


def run_nuts(posterior, initial_positions, *, num_samples=1000, num_warmup=500, seed=0,
             target_accept=0.8, max_tree_depth=10, inverse_mass=None, torch_compile=None,
             torch_compile_mode=None):
    """Sample a Posterior using native Torch NUTS or optional NumPyro/JAX NUTS.

    Starts are physical values, shaped (parameters,) or (chains, parameters),
    strictly inside the prior. Chains run sequentially on the chosen device.
    The initial diagonal inverse mass is in chart coordinates and stays fixed
    in both backends; step sizes adapt during warmup. No convergence claim or
    automatic rerun is made. JAX requires ``diffgw[pe-jax]``.
    By default, Torch CUDA compiles the complete chart target with fullgraph=True,
    dynamic=False and mode="reduce-overhead"; Torch CPU stays eager and JAX uses
    NumPyro JIT. ``torch_compile=False`` explicitly disables Torch compilation.
    ``torch_compile=True`` enables it on Torch (CUDA: reduce-overhead; CPU: default).
    ``torch_compile_mode="default"`` selects ordinary compilation when enabled.
    Compiler errors propagate; this API provides no managed fallback.
    ``torch_compile_mode="reduce-overhead"`` requires a compiled Torch CUDA target.
    It marks each value/gradient pair as a CUDA-graph step and copies both outputs
    before NUTS retains them; the native algorithm and random draws stay unchanged.
    """
    if not isinstance(posterior, Posterior):
        raise TypeError("posterior must be a Posterior")
    if torch_compile is not None and type(torch_compile) is not bool:
        raise ValueError("torch_compile must be a bool or None")
    if torch_compile_mode is not None and (type(torch_compile_mode) is not str or torch_compile_mode not in ("default", "reduce-overhead")):
        raise ValueError("torch_compile_mode must be None, 'default' or 'reduce-overhead'")
    torch_cuda = posterior.backend == "torch" and posterior.device.type == "cuda"
    if torch_compile is None:
        torch_compile = torch_cuda
    if torch_compile_mode is None:
        torch_compile_mode = "reduce-overhead" if torch_compile and torch_cuda else "default"
    if torch_compile_mode == "reduce-overhead" and (not torch_compile or posterior.backend != "torch" or posterior.device.type != "cuda"):
        raise ValueError("torch_compile_mode='reduce-overhead' requires torch_compile=True and a Torch CUDA posterior")
    if torch_compile and posterior.backend != "torch":
        raise ValueError("torch_compile=True is only supported for Torch; JAX/NumPyro already compiles its target")
    if torch_compile_mode == "reduce-overhead" and not callable(
        getattr(getattr(posterior._framework, "compiler", None), "cudagraph_mark_step_begin", None)
    ):
        raise RuntimeError(
            "Torch reduce-overhead PE requires callable torch.compiler.cudagraph_mark_step_begin; "
            "upgrade Torch or explicitly set torch_compile=False for eager execution"
        )
    for name, value, minimum in (("num_samples", num_samples, 1), ("num_warmup", num_warmup, 0),
                                 ("max_tree_depth", max_tree_depth, 1), ("seed", seed, 0)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if seed >= 2**32 or not np.isfinite(target_accept) or not 0 < target_accept < 1:
        raise ValueError("seed must be < 2**32 and target_accept between 0 and 1")
    starts = posterior.asarray(initial_positions)
    if starts.ndim == 1:
        starts = starts[None, :]
    if starts.ndim != 2 or starts.shape[0] < 1:
        raise ValueError("Initial positions must be a vector or nonempty (chains, parameters) array")
    posterior._check_shape(starts)
    host_starts = _numpy(starts)
    if not np.isfinite(host_starts).all() or not (host_starts > posterior.prior.lower).all() or not (host_starts < posterior.prior.upper).all():
        raise ValueError("Initial positions must be finite and strictly inside the prior")
    mass = None if inverse_mass is None else posterior.asarray(inverse_mass)
    if mass is not None and (mass.shape != (len(posterior.names),) or not np.isfinite(_numpy(mass)).all() or not (_numpy(mass) > 0).all()):
        raise ValueError("inverse_mass must be a finite positive diagonal vector in chart coordinates")
    z0 = posterior.to_unconstrained(starts)
    framework = posterior._framework
    if posterior.backend == "torch":
        for start in z0:
            z = start.detach().clone().requires_grad_(True)
            likelihood = posterior._likelihood(posterior.from_unconstrained(z))
            if not likelihood.requires_grad or framework.autograd.grad(likelihood, z, allow_unused=True)[0] is None:
                raise ValueError("Torch likelihood must remain connected to its parameter gradients")
            value = posterior.log_prob_unconstrained(z)
            gradient = framework.autograd.grad(value, z)[0]
            if not bool(framework.isfinite(value) & framework.isfinite(gradient).all()):
                raise ValueError("Initial log posterior and gradients must be finite")
        target = posterior.log_prob_unconstrained
        if torch_compile:
            if torch_compile_mode == "reduce-overhead":
                target = framework.compile(target, fullgraph=True, dynamic=False, mode="reduce-overhead")
            else:
                target = framework.compile(target, fullgraph=True, dynamic=False)
        sampler_type = _torch_nuts_type(framework, cuda_graphs=torch_compile_mode == "reduce-overhead")
        sampler = sampler_type(target, target_accept=target_accept,
                       max_depth=max_tree_depth, inv_mass=mass)
        cuda_devices = [posterior.device.index] if posterior.device.type == "cuda" else []
        with framework.random.fork_rng(devices=cuda_devices):
            framework.random.default_generator.manual_seed(seed)
            if cuda_devices:
                with framework.cuda.device(posterior.device):
                    framework.cuda.manual_seed(seed)
            z, diagnostics = sampler.sample_parallel(z0, n_samples=num_samples, n_warmup=num_warmup, return_diagnostics=True)
        engine = "torchwave.NUTS"
        target_execution = "torch_compile" if torch_compile else "eager"
        phase = "warmup_and_sampling"
        versions = {"torch": framework.__version__}
    else:
        try:
            import numpyro
            from numpyro.infer import MCMC, NUTS
        except ModuleNotFoundError as exc:
            if exc.name and exc.name.startswith("numpyro"):
                raise ModuleNotFoundError("JAX NUTS requires NumPyro; install 'diffgw[pe-jax]'.", name=exc.name) from exc
            raise
        value_and_grad = framework.jit(framework.value_and_grad(posterior.log_prob_unconstrained))
        with framework.default_device(posterior.device):
            for start in z0:
                value, gradient = value_and_grad(start)
                if not np.isfinite(_numpy(value)).all() or not np.isfinite(_numpy(gradient)).all():
                    raise ValueError("Initial log posterior and gradients must be finite")
            sampler = NUTS(potential_fn=lambda z: -posterior.log_prob_unconstrained(z),
                           target_accept_prob=target_accept, max_tree_depth=max_tree_depth,
                           inverse_mass_matrix=mass, adapt_mass_matrix=False)
            mcmc = MCMC(sampler, num_samples=num_samples, num_warmup=num_warmup,
                        num_chains=len(z0), chain_method="sequential", progress_bar=False)
            mcmc.run(framework.random.PRNGKey(seed), init_params=z0[0] if len(z0) == 1 else z0,
                     extra_fields=("diverging", "accept_prob", "num_steps", "potential_energy", "adapt_state.step_size"))
            z = mcmc.get_samples(group_by_chain=True)
            diagnostics = mcmc.get_extra_fields(group_by_chain=True)
        engine = "numpyro.NUTS"
        target_execution = "jax_numpyro_jit"
        phase = "sampling_only"
        versions = {"jax": framework.__version__, "numpyro": numpyro.__version__}
    metadata = {"backend": posterior.backend, "device": str(posterior.device), "sampler": engine,
                "dtype": "float64", "chain_method": "sequential", "diagnostic_phase": phase,
                "target_execution": target_execution,
                "torch_compile_mode": torch_compile_mode if torch_compile else None,
                "num_samples": num_samples, "num_warmup": num_warmup, "num_chains": len(z0),
                "seed": seed, "target_accept": target_accept, "max_tree_depth": max_tree_depth,
                "adapt_mass_matrix": False, "inverse_mass": None if mass is None else _numpy(mass).tolist(),
                "prior_bounds": dict(posterior.prior.bounds), "versions": versions}
    return PEResult(posterior.from_unconstrained(z), posterior.names, diagnostics, metadata)

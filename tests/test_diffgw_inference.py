"""Public PE contracts, exact densities, backend parity and actual recovery."""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from diffgw.inference import Posterior, UniformBoxPrior, WhittleLikelihood, run_nuts


def posterior(backend, *, device="cpu"):
    pytest.importorskip(backend)
    return Posterior(lambda p: -0.5 * ((p["x"] - 0.7) / 0.8) ** 2 - 0.5 * ((p["y"] + 0.4) / 1.1) ** 2,
                     UniformBoxPrior({"x": (-5, 5), "y": (-6, 8)}), backend=backend, device=device)


def host(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def test_public_import_is_backend_and_sampler_independent():
    src = str(Path(__file__).resolve().parents[1] / "src")
    code = '''
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'torchwave', 'jax', 'jaxwave', 'numpyro'):
            raise ModuleNotFoundError(fullname, name=fullname)
sys.meta_path.insert(0, Block())
import diffgw
prior = diffgw.inference.UniformBoxPrior({'x': (0, 1)})
assert prior.names == ('x',)
assert not any(x in sys.modules for x in ('torch', 'jax', 'numpyro'))
for backend in ('torch', 'jax'):
    try:
        diffgw.inference.Posterior(lambda p: 0, prior, backend=backend)
    except ModuleNotFoundError as e:
        assert 'diffgw[' + backend + ']' in str(e)
    else:
        raise AssertionError('missing dependency must be reported')
'''
    subprocess.run([sys.executable, "-c", code], env={**os.environ, "PYTHONPATH": src}, check=True)
    # Available frameworks must remain lazy too; blocking imports alone would
    # miss an eager import whose missing dependency is swallowed by the core.
    code = '''
import sys, diffgw
from diffgw.inference import UniformBoxPrior
assert UniformBoxPrior({'x': (0, 1)}).names == ('x',)
assert not any(x in sys.modules for x in ('torch', 'jax', 'numpyro'))
'''
    subprocess.run([sys.executable, "-c", code], env={**os.environ, "PYTHONPATH": src}, check=True)


@pytest.mark.parametrize("bounds", [{}, {"x": (1, 1)}, {"x": (2, 1)}, {"x": (0, np.inf)}, {"x": (0, 1, 2)}, {"": (0, 1)}])
def test_invalid_prior(bounds):
    with pytest.raises(ValueError):
        UniformBoxPrior(bounds)


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_strict_interior_chart_remains_finite_near_prior_edges(backend):
    pytest.importorskip(backend)
    p = Posterior(lambda p: 0 * p["x"], UniformBoxPrior({"x": (-1., 1.)}), backend=backend, device="cpu")
    theta = p.asarray([[np.nextafter(-1., 0.)], [np.nextafter(1., 0.)]])
    assert np.isfinite(host(p.to_unconstrained(theta))).all()
    skewed = Posterior(lambda p: 0 * p["x"], UniformBoxPrior({"x": (-1e20, 1.)}), backend=backend, device="cpu")
    assert np.isneginf(host(skewed.log_prob(skewed.asarray([2.]))))
    assert host(skewed.from_unconstrained(skewed.asarray([100.])))[0] == 1.


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_real_setup_quantities_reject_complex_values(backend):
    p = posterior(backend)
    with pytest.raises(ValueError, match="real"):
        p.asarray([1 + 2j, 0])
    with pytest.raises(ValueError, match="real"):
        WhittleLikelihood({"H1": [1j, 2j]}, {"H1": [1 + 2j, 1]}, df=1, backend=backend)


def test_jax_cpu_device_object_is_supported():
    jax = pytest.importorskip("jax")
    device = jax.devices("cpu")[0]
    assert posterior("jax", device=device).device == device


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_physical_density_chart_and_gradients(backend):
    p = posterior(backend)
    theta = p.asarray([0.2, -0.1])
    expected_ll = -0.5 * ((0.2 - 0.7) / 0.8) ** 2 - 0.5 * ((-0.1 + 0.4) / 1.1) ** 2
    np.testing.assert_allclose(host(p.log_prob(theta)), expected_ll - np.log(140), atol=1e-13)
    z = p.to_unconstrained(theta)
    np.testing.assert_allclose(host(p.from_unconstrained(z)), host(theta), atol=2e-15)
    u = (host(theta) - np.array([-5, -6])) / [10, 14]
    np.testing.assert_allclose(host(p.log_prob_unconstrained(z)), expected_ll + np.log(u * (1 - u)).sum(), atol=1e-13)
    expected_grad = -(host(theta) - [0.7, -0.4]) / np.square([0.8, 1.1]) * [10, 14] * u * (1 - u) + 1 - 2 * u
    if backend == "torch":
        import torch
        z.requires_grad_(True)
        gradient = torch.autograd.grad(p.log_prob_unconstrained(z), z)[0]
    else:
        import jax
        gradient = jax.jit(jax.grad(p.log_prob_unconstrained))(z)
        np.testing.assert_allclose(host(jax.jit(p.log_prob)(theta)), expected_ll - np.log(140), atol=1e-13)
    np.testing.assert_allclose(host(gradient), expected_grad, atol=2e-13)
    assert np.isneginf(host(p.log_prob(p.asarray([10, 0]))))
    batch = p.asarray(np.zeros((2, 3, 2)))
    assert p.from_unconstrained(batch).shape == (2, 3, 2)
    with pytest.raises(ValueError, match="vector"):
        p.log_prob(batch)


@pytest.mark.parametrize("backend", ["torch", "jax"])
@pytest.mark.parametrize("integration", ["rectangle", "trapezoid"])
def test_complex_likelihood_and_derivative(backend, integration):
    pytest.importorskip(backend)
    data = {"H1": np.array([1 + 2j, 3 - 1j, -1 + 0.5j])}
    psds = {"H1": np.array([2., 3., 4.])}
    kwargs = {"df": 0.5} if integration == "rectangle" else {"frequencies": np.array([20., 20.5, 21.5])}
    likelihood = WhittleLikelihood(data, psds, backend=backend, device="cpu", **kwargs)
    p = Posterior(lambda params: likelihood({"H1": params["scale"] * d}),
                  UniformBoxPrior({"scale": (0, 2)}), backend=backend, device="cpu")
    d = p._framework.as_tensor(data["H1"], dtype=p._framework.complex128) if backend == "torch" else p._framework.numpy.asarray(data["H1"])
    integrand = abs(data["H1"]) ** 2 / psds["H1"]
    norm = 4 * (integrand.sum() * 0.5 if integration == "rectangle" else np.trapezoid(integrand, kwargs["frequencies"]))
    theta = p.asarray([0.6])
    np.testing.assert_allclose(host(p._likelihood(theta)), -0.5 * 0.4**2 * norm, atol=1e-13)
    if backend == "torch":
        import torch
        theta.requires_grad_(True)
        gradient = torch.autograd.grad(p._likelihood(theta), theta)[0]
        assert likelihood.data_H1.dtype == torch.complex128
    else:
        import jax
        gradient = jax.grad(p._likelihood)(theta)
    np.testing.assert_allclose(host(gradient), [0.4 * norm], atol=1e-13)


@pytest.mark.parametrize("kwargs", [{}, {"df": 1, "frequencies": [1, 2]}, {"df": -1}, {"df": 1, "f_low": 1}])
def test_likelihood_integration_must_be_explicit(kwargs):
    with pytest.raises(ValueError):
        WhittleLikelihood({"H1": [1, 2]}, {"H1": [1, 1]}, **kwargs)


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_missing_gpu_is_rejected(backend):
    framework = pytest.importorskip(backend)
    if backend == "torch" and framework.cuda.is_available():
        pytest.skip("GPU present; exercised by the GPU contract test")
    if backend == "jax" and any(d.platform == "gpu" for d in framework.devices()):
        pytest.skip("GPU present; exercised by the GPU contract test")
    with pytest.raises(ValueError, match="unavailable"):
        posterior(backend, device="cuda" if backend == "torch" else "gpu")


@pytest.mark.parametrize("backend", ["torch", "jax"])
@pytest.mark.parametrize("starts", [[-5, 0], [6, 0], [np.nan, 0], [], [[0]]])
def test_invalid_initial_positions(backend, starts):
    with pytest.raises(ValueError):
        run_nuts(posterior(backend), starts, num_samples=1, num_warmup=0)


def test_detached_torch_likelihood_is_rejected():
    torch = pytest.importorskip("torch")
    p = Posterior(lambda params: torch.tensor(-float(params["x"].detach()) ** 2), UniformBoxPrior({"x": (-1, 1)}))
    with pytest.raises(ValueError, match="connected"):
        run_nuts(p, [0.2], num_samples=1, num_warmup=0)


def test_numpy_callback_is_rejected_by_jax():
    pytest.importorskip("jax")
    p = Posterior(lambda p: np.array(0.), UniformBoxPrior({"x": (-1, 1)}), backend="jax")
    with pytest.raises(ValueError, match="native scalar"):
        p.log_prob_unconstrained(p.asarray([0.]))


def test_missing_numpyro_reports_pe_extra():
    pytest.importorskip("jax")
    code = '''
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'torchwave', 'numpyro'):
            raise ModuleNotFoundError(fullname, name=fullname)
sys.meta_path.insert(0, Block())
from diffgw.inference import Posterior, UniformBoxPrior, run_nuts
p = Posterior(lambda p: -p['x']**2, UniformBoxPrior({'x': (-1, 1)}), backend='jax')
try:
    run_nuts(p, [0.], num_samples=1, num_warmup=0)
except ModuleNotFoundError as e:
    assert 'diffgw[pe-jax]' in str(e)
else:
    raise AssertionError('missing NumPyro must be reported')
assert 'torch' not in sys.modules
'''
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_nonfinite_initial_target_is_rejected(backend):
    pytest.importorskip(backend)
    if backend == "jax":
        pytest.importorskip("numpyro")
    p = Posterior(lambda p: p["x"] * np.nan, UniformBoxPrior({"x": (-1, 1)}), backend=backend)
    with pytest.raises(ValueError, match="finite"):
        run_nuts(p, [0.2], num_samples=1, num_warmup=0)


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_actual_nuts_recovery_and_result_contract(backend):
    if backend == "jax":
        pytest.importorskip("numpyro")
    p = posterior(backend)
    if backend == "torch":
        import torch
        rng_before = torch.random.get_rng_state().clone()
    result = run_nuts(p, [[0., 0.], [1.2, -1.]], seed=54, num_samples=240, num_warmup=160, max_tree_depth=7)
    assert result.samples.shape == (2, 240, 2)
    exported = result.to_numpy()
    assert set(result.posterior_samples) == {"x", "y"}
    assert np.isfinite(exported["samples"]).all()
    np.testing.assert_allclose(exported["samples"].mean(axis=(0, 1)), [0.7, -0.4], atol=0.23)
    np.testing.assert_allclose(exported["samples"].std(axis=(0, 1)), [0.8, 1.1], atol=0.21)
    assert exported["metadata"]["adapt_mass_matrix"] is False
    assert result.metadata["target_execution"] == ("eager" if backend == "torch" else "jax_numpyro_jit")
    json.dumps(exported["metadata"])
    if backend == "torch":
        assert torch.equal(rng_before, torch.random.get_rng_state())
        assert result.metadata["diagnostic_phase"] == "warmup_and_sampling"
        assert result.diagnostics["warmup"].shape == (2, 400)
        assert int(result.diagnostics["warmup"].sum()) == 320
    else:
        assert result.metadata["diagnostic_phase"] == "sampling_only"
        assert result.diagnostics["diverging"].shape == (2, 240)


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_explicit_gpu_samples_and_likelihood_placement(backend):
    framework = pytest.importorskip(backend)
    if backend == "torch" and not framework.cuda.is_available():
        pytest.skip("requires CUDA")
    if backend == "jax":
        pytest.importorskip("numpyro")
        if not any(d.platform == "gpu" for d in framework.devices()):
            pytest.skip("requires JAX GPU plugin")
    device = "cuda" if backend == "torch" else "gpu"
    p = posterior(backend, device=device)
    ll = WhittleLikelihood({"H1": [1j, 2j]}, {"H1": [1., 1.]}, df=1, backend=backend, device=device)
    resident = ll.data_H1 if backend == "torch" else ll.data["H1"]
    assert (resident.device.type == "cuda") if backend == "torch" else (resident.device.platform == "gpu")
    result = run_nuts(p, [0, 0], num_samples=8, num_warmup=8, seed=1, max_tree_depth=4)
    assert (result.samples.device.type == "cuda") if backend == "torch" else (result.samples.device.platform == "gpu")
    assert np.isfinite(result.to_numpy()["samples"]).all()


@pytest.mark.parametrize("value", [0, 1, "true", np.bool_(True)])
def test_torch_compile_option_requires_bool_or_none(value):
    with pytest.raises(ValueError, match="torch_compile must be a bool or None"):
        run_nuts(posterior("torch"), [0., 0.], num_samples=1, num_warmup=0, torch_compile=value)


def test_torch_compile_option_rejects_jax():
    with pytest.raises(ValueError, match="only supported for Torch"):
        run_nuts(posterior("jax"), [0., 0.], num_samples=1, num_warmup=0, torch_compile=True)


def test_default_and_explicit_eager_target_do_not_compile(monkeypatch):
    torch = pytest.importorskip("torch")

    def unexpected_compile(*args, **kwargs):
        raise AssertionError("eager target must not compile")

    monkeypatch.setattr(torch, "compile", unexpected_compile)
    p = posterior("torch")
    options = dict(num_samples=8, num_warmup=4, seed=7, max_tree_depth=3)
    default = run_nuts(p, [0., 0.], **options)
    explicit = run_nuts(p, [0., 0.], torch_compile=False, **options)
    assert torch.equal(default.samples, explicit.samples)
    assert default.metadata == explicit.metadata
    assert default.metadata["target_execution"] == "eager"


@pytest.mark.parametrize("invalid", ["detached", "nonfinite"])
def test_compile_option_preserves_initial_target_checks(monkeypatch, invalid):
    torch = pytest.importorskip("torch")

    def unexpected_compile(*args, **kwargs):
        raise AssertionError("invalid initial target must fail before compilation")

    monkeypatch.setattr(torch, "compile", unexpected_compile)
    likelihood = (lambda p: -p["x"].detach().square()) if invalid == "detached" else (lambda p: p["x"] * float("nan"))
    p = Posterior(likelihood, UniformBoxPrior({"x": (-1., 1.)}), backend="torch", device="cpu")
    with pytest.raises(ValueError, match="connected" if invalid == "detached" else "finite"):
        run_nuts(p, [0.2], num_samples=1, num_warmup=0, torch_compile=True)


@pytest.mark.parametrize("compiler_backend", ["aot_eager", "inductor"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_real_compiled_torch_target_values_gradients_and_sampling(monkeypatch, compiler_backend, device):
    torch = pytest.importorskip("torch")
    import shutil

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if compiler_backend == "inductor" and device == "cpu" and not any(shutil.which(tool) for tool in ("c++", "g++", "clang++")):
        pytest.skip("actual CPU Inductor test requires a C++ compiler")
    p = posterior("torch", device=device)
    compile_target = torch.compile
    compiled = []

    def record_compile(target, **kwargs):
        assert target.__self__ is p
        assert target.__func__ is Posterior.log_prob_unconstrained
        assert kwargs == {"fullgraph": True, "dynamic": False}
        result = compile_target(target, backend=compiler_backend, **kwargs)
        compiled.append(result)
        return result

    monkeypatch.setattr(torch, "compile", record_compile)
    rng_before = torch.random.get_rng_state().clone()
    cuda_rng_before = torch.cuda.get_rng_state(p.device).clone() if device == "cuda" else None
    try:
        result = run_nuts(p, [[0., 0.], [1.2, -1.]], num_samples=8, num_warmup=4,
                          seed=7, max_tree_depth=3, torch_compile=True, torch_compile_mode="default")
    except torch._dynamo.exc.BackendCompilerFailed as error:
        message = str(error)
        if (compiler_backend == "inductor" and sys.platform == "darwin"
                and "Library not loaded:" in message
                and any(library in message for library in ("libc++.1.dylib", "libomp.dylib"))):
            pytest.skip("Installed CPU Inductor cannot load its C++/OpenMP runtime: " + message.splitlines()[0])
        raise
    assert len(compiled) == 1
    assert torch.equal(rng_before, torch.random.get_rng_state())
    if cuda_rng_before is not None:
        assert torch.equal(cuda_rng_before, torch.cuda.get_rng_state(p.device))
    assert result.samples.shape == (2, 8, 2)
    assert result.samples.dtype == torch.float64
    assert result.samples.device.type == device
    assert result.metadata["target_execution"] == "torch_compile"
    assert torch.isfinite(result.samples).all()
    assert torch.all(result.samples > p.asarray(p.prior.lower))
    assert torch.all(result.samples < p.asarray(p.prior.upper))
    for point in ([0.2, -0.1], [-1., 1.], [2., -2.]):
        z = p.to_unconstrained(p.asarray(point)).detach().requires_grad_(True)
        eager = p.log_prob_unconstrained(z)
        actual = compiled[0](z)
        eager_gradient = torch.autograd.grad(eager, z)[0]
        actual_gradient = torch.autograd.grad(actual, z)[0]
        torch.testing.assert_close(actual, eager, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(actual_gradient, eager_gradient, rtol=1e-12, atol=1e-12)


def test_real_compile_failure_is_not_replaced_by_eager_sampling():
    torch = pytest.importorskip("torch")
    import torch._dynamo

    def untraceable(params):
        if params["x"].item() > 0:
            return -params["x"].square()
        return -2 * params["x"].square()

    p = Posterior(untraceable, UniformBoxPrior({"x": (-1., 1.)}), backend="torch", device="cpu")
    eager = run_nuts(p, [0.2], num_samples=2, num_warmup=0, seed=8, max_tree_depth=2)
    assert torch.isfinite(eager.samples).all()
    rng_before = torch.random.get_rng_state().clone()
    with torch._dynamo.config.patch(capture_scalar_outputs=False):
        with pytest.raises((torch._dynamo.exc.Unsupported, torch._dynamo.exc.UserError),
                           match="Tensor.item|data-dependent"):
            run_nuts(p, [0.2], num_samples=2, num_warmup=0, seed=8,
                     max_tree_depth=2, torch_compile=True)
    assert torch.equal(rng_before, torch.random.get_rng_state())


@pytest.mark.parametrize("mode", [True, 0, [], "fast", "max-autotune"])
def test_torch_compile_mode_rejects_invalid_values(mode):
    with pytest.raises(ValueError, match="torch_compile_mode must be"):
        run_nuts(posterior("torch"), [0., 0.], torch_compile_mode=mode)


@pytest.mark.parametrize("backend,compile_target", [("torch", False), ("torch", True), ("jax", False), ("jax", True)])
def test_reduce_overhead_requires_compiled_torch_cuda(backend, compile_target):
    with pytest.raises(ValueError, match="requires torch_compile=True and a Torch CUDA posterior"):
        run_nuts(posterior(backend), [0., 0.], torch_compile=compile_target,
                 torch_compile_mode="reduce-overhead")


@pytest.mark.parametrize("capability", ["missing_compiler", "missing_marker", "noncallable_marker"])
def test_cuda_default_requires_graph_step_capability_before_execution(monkeypatch, capability):
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    p = posterior("torch")
    monkeypatch.setattr(p, "device", torch.device("cuda:0"))
    compiler = {"missing_compiler": None, "missing_marker": SimpleNamespace(),
                "noncallable_marker": SimpleNamespace(cudagraph_mark_step_begin=None)}[capability]
    monkeypatch.setattr(torch, "compiler", compiler)

    def unexpected_execution(*args, **kwargs):
        raise AssertionError("unsupported runtime must fail before device setup or compilation")

    monkeypatch.setattr(p, "asarray", unexpected_execution)
    monkeypatch.setattr(torch, "compile", unexpected_execution)
    with pytest.raises(RuntimeError, match=r"requires callable torch.compiler.cudagraph_mark_step_begin.*torch_compile=False"):
        run_nuts(p, [0., 0.], num_samples=1, num_warmup=0)


def test_cuda_graph_nuts_factory_owns_retained_pair_and_marks_before_target(monkeypatch):
    torch = pytest.importorskip("torch")
    from diffgw.inference import _torch_nuts_type
    from torchwave.inference.mcmc import NUTS

    assert _torch_nuts_type(torch) is NUTS
    events = []
    shared_logp = torch.empty((), dtype=torch.float64)

    class BufferedTarget(torch.autograd.Function):
        @staticmethod
        def forward(ctx, theta):
            events.append("target")
            ctx.save_for_backward(theta)
            shared_logp.copy_(-0.5 * theta.square().sum())
            return shared_logp

        @staticmethod
        def backward(ctx, grad_output):
            events.append("backward")
            return -ctx.saved_tensors[0] * grad_output

    monkeypatch.setattr(torch.compiler, "cudagraph_mark_step_begin", lambda: events.append("mark"))
    points = [torch.tensor(point, dtype=torch.float64) for point in ([1., 2.], [3., -1.], [-2., 4.])]
    unsafe = NUTS(BufferedTarget.apply)
    old_logp, _ = unsafe.value_and_grad(points[0])
    old_value = old_logp.clone()
    unsafe.value_and_grad(points[1])
    assert not torch.equal(old_logp, old_value)  # Reproduces the cached scalar alias.

    events.clear()
    safe = _torch_nuts_type(torch, cuda_graphs=True)(BufferedTarget.apply)
    retained = [safe.value_and_grad(point) for point in points]
    assert events == ["mark", "target", "backward"] * len(points)
    for point, (value, gradient) in zip(points, retained):
        torch.testing.assert_close(value, -0.5 * point.square().sum(), rtol=0, atol=0)
        torch.testing.assert_close(gradient, point, rtol=0, atol=0)
        assert not value.requires_grad and not gradient.requires_grad
        assert value.data_ptr() != shared_logp.data_ptr()
    assert len({gradient.data_ptr() for _, gradient in retained}) == len(points)

    def mark_failure():
        raise RuntimeError("graph step failed")

    monkeypatch.setattr(torch.compiler, "cudagraph_mark_step_begin", mark_failure)
    with pytest.raises(RuntimeError, match="graph step failed"):
        safe.value_and_grad(points[0])


def _check_cuda_reduce_overhead_target_chain_rng_and_retained_states(monkeypatch, compile_options):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA and real Inductor CUDA graphs")
    from diffgw.inference import _torch_nuts_type
    from torchwave.inference.mcmc import NUTS

    p = posterior("torch", device="cuda")
    import diffgw.inference as inference

    original_factory = inference._torch_nuts_type
    factory_calls = []

    def record_factory(framework, cuda_graphs=False):
        selected = original_factory(framework, cuda_graphs=cuda_graphs)
        factory_calls.append(cuda_graphs)
        assert (selected is NUTS) if not cuda_graphs else (selected.__bases__ == (NUTS,))
        return selected

    monkeypatch.setattr(inference, "_torch_nuts_type", record_factory)
    original_compile = torch.compile
    compiled = []

    def record_compile(target, **kwargs):
        assert target.__self__ is p
        assert kwargs == {"fullgraph": True, "dynamic": False, "mode": "reduce-overhead"}
        result = original_compile(target, **kwargs)  # Real default Inductor backend.
        compiled.append(result)
        return result

    monkeypatch.setattr(torch, "compile", record_compile)
    options = dict(num_samples=8, num_warmup=4, seed=7, max_tree_depth=3)
    starts = [[0., 0.], [1.2, -1.]]
    rng_before = torch.random.get_rng_state().clone()
    cuda_rng_before = torch.cuda.get_rng_state(p.device).clone()
    eager = run_nuts(p, starts, torch_compile=False, **options)
    assert len(compiled) == 0
    result = run_nuts(p, starts, **compile_options, **options)
    assert len(compiled) == 1
    assert factory_calls == [False, True]
    assert torch.equal(rng_before, torch.random.get_rng_state())
    assert torch.equal(cuda_rng_before, torch.cuda.get_rng_state(p.device))
    assert result.metadata["torch_compile_mode"] == "reduce-overhead"
    assert eager.metadata["torch_compile_mode"] is None
    assert result.metadata["target_execution"] == "torch_compile"
    assert result.samples.shape == (2, 8, 2)
    assert result.samples.dtype == torch.float64 and result.samples.device.type == "cuda"
    torch.testing.assert_close(result.samples, eager.samples, rtol=1e-10, atol=1e-10)
    assert result.diagnostics.keys() == eager.diagnostics.keys()
    for name, value in result.diagnostics.items():
        torch.testing.assert_close(value, eager.diagnostics[name], rtol=1e-9, atol=1e-10)

    safe = _torch_nuts_type(torch, cuda_graphs=True)(compiled[0])
    native_eager = NUTS(p.log_prob_unconstrained)
    points = [p.to_unconstrained(p.asarray(point)) for point in ([0.2, -0.1], [-1., 1.], [2., -2.])]
    for _ in range(3):
        safe.value_and_grad(points[0])
    retained = [safe.value_and_grad(point) for point in points * 2]
    for point, (value, gradient) in zip(points * 2, retained):
        expected_value, expected_gradient = native_eager.value_and_grad(point)
        torch.testing.assert_close(value, expected_value, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(gradient, expected_gradient, rtol=1e-12, atol=1e-12)
        assert not value.requires_grad and not gradient.requires_grad
    assert len({value.data_ptr() for value, _ in retained}) == len(retained)
    assert len({gradient.data_ptr() for _, gradient in retained}) == len(retained)



def test_cuda_reduce_overhead_target_chain_rng_and_retained_states(monkeypatch):
    _check_cuda_reduce_overhead_target_chain_rng_and_retained_states(
        monkeypatch, {"torch_compile": True, "torch_compile_mode": "reduce-overhead"})


@pytest.mark.parametrize("compile_options", [{}, {"torch_compile": True}], ids=["implicit", "explicit_true"])
def test_cuda_default_acceleration_chain_rng_and_retained_states(monkeypatch, compile_options):
    _check_cuda_reduce_overhead_target_chain_rng_and_retained_states(monkeypatch, compile_options)


def test_cpu_explicit_none_options_remain_eager(monkeypatch):
    torch = pytest.importorskip("torch")

    def unexpected_compile(*args, **kwargs):
        raise AssertionError("CPU automatic default must stay eager")

    monkeypatch.setattr(torch, "compile", unexpected_compile)
    p = posterior("torch")
    options = dict(num_samples=8, num_warmup=4, seed=7, max_tree_depth=3)
    implicit = run_nuts(p, [0., 0.], **options)
    explicit_none = run_nuts(p, [0., 0.], torch_compile=None, torch_compile_mode=None, **options)
    explicit_default = run_nuts(p, [0., 0.], torch_compile_mode="default", **options)
    assert torch.equal(implicit.samples, explicit_none.samples)
    assert torch.equal(implicit.samples, explicit_default.samples)
    assert implicit.metadata == explicit_none.metadata == explicit_default.metadata
    assert implicit.metadata["target_execution"] == "eager"
    assert implicit.metadata["torch_compile_mode"] is None


def test_jax_automatic_compile_options_preserve_numpyro_jit():
    pytest.importorskip("numpyro")
    p = posterior("jax")
    result = run_nuts(p, [0., 0.], num_samples=2, num_warmup=0, seed=7, max_tree_depth=2)
    assert result.metadata["target_execution"] == "jax_numpyro_jit"
    assert result.metadata["torch_compile_mode"] is None
    assert np.isfinite(result.to_numpy()["samples"]).all()


def test_cuda_default_phenomd_initializes_cold_coefficient_cache_before_compile(monkeypatch):
    """Public eager preflight must keep native coefficients outside CUDA graphs."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA and real Inductor CUDA graphs")
    import diffgw
    import torchwave.approximants.imrphenomd as phenomd
    from torch._dynamo.utils import counters
    from torchwave.detectors import DetectorNetwork, aLIGOZeroDetHighPower
    from torchwave.inference.mcmc import NUTS
    from diffgw.inference import _torch_nuts_type

    prior = UniformBoxPrior({"mass1": (29.5, 30.5), "mass2": (24.5, 25.5)})
    p = Posterior(lambda params: likelihood(templates(params)), prior, backend="torch", device="cuda")
    frequencies = p.asarray(np.arange(20., 129., 2.))
    network = DetectorNetwork(["H1", "L1"]).to(p.device)
    psd = aLIGOZeroDetHighPower(frequencies)

    def templates(params):
        hp, hc = diffgw.get_fd_waveform(
            "IMRPhenomD", backend="torch", sample_frequencies=frequencies,
            mass1=params["mass1"], mass2=params["mass2"], distance=1000.,
            chi1=0., chi2=0., inclination=0.4,
        )
        return network.project_fd_strain(
            f=frequencies, h_plus=hp, h_cross=hc,
            ra=1.5, dec=0.3, psi=0.8, t_gps=1126259462.,
        )

    data = {name: value.reshape(-1) for name, value in templates(
        p.parameters(p.asarray([30., 25.]))).items()}
    likelihood = WhittleLikelihood(data, {name: psd for name in data},
                                   df=2., backend="torch", device=p.device)
    # Data construction may populate the module cache. Isolate this test's cold
    # cache afterward; no target evaluation may prime it before public run_nuts.
    cold_cache = {}
    monkeypatch.setattr(phenomd, "_TABLE_V_CACHE", cold_cache)
    key = (str(p.device), torch.float64, "inherited")
    assert key not in cold_cache
    real_compile = torch.compile
    compiled = []
    coefficient_state = []

    def record_compile(target, **kwargs):
        assert target.__self__ is p
        assert target.__func__ is Posterior.log_prob_unconstrained
        assert kwargs == {"fullgraph": True, "dynamic": False, "mode": "reduce-overhead"}
        # This assertion runs before compilation; only run_nuts's own eager AD
        # preflight can have initialized this real native coefficient matrix.
        assert phenomd._TABLE_V_CACHE is cold_cache
        assert set(cold_cache) == {key}
        matrix = cold_cache[key]
        assert matrix.shape == (18, 11)
        assert matrix.dtype == torch.float64 and matrix.device == p.device
        assert not matrix.requires_grad and matrix.grad_fn is None
        assert not torch.is_inference(matrix)
        coefficient_state.append((matrix, matrix.data_ptr(), matrix.detach().clone()))
        result = real_compile(target, **kwargs)  # Actual default Inductor backend.
        compiled.append(result)
        return result

    monkeypatch.setattr(torch, "compile", record_compile)

    def compiler_snapshot():
        # Strict positive witnesses use the counter schema observed in the
        # pinned CUDA qualification runtime; metadata alone is insufficient.
        return {
            "graphs": counters["stats"]["unique_graphs"],
            "calls": counters["stats"]["calls_captured"],
            "inductor": counters["inductor"]["fxgraph_cache_hit"] + counters["inductor"]["fxgraph_cache_miss"],
            "cuda_graph_inputs": counters["inductor"]["cudagraph_recorded_non_static_inputs"],
        }

    before = compiler_snapshot()
    rng_before = torch.random.get_rng_state().clone()
    cuda_rng_before = torch.cuda.get_rng_state(p.device).clone()
    starts = [[29.9, 25.1], [30.1, 24.9]]
    options = dict(num_samples=8, num_warmup=4, seed=7, max_tree_depth=3)
    result = run_nuts(p, starts, **options)  # Exercise implicit public CUDA defaults.
    assert len(compiled) == len(coefficient_state) == 1
    assert result.metadata["target_execution"] == "torch_compile"
    assert result.metadata["torch_compile_mode"] == "reduce-overhead"
    assert result.samples.shape == (2, 8, 2)
    assert result.samples.dtype == torch.float64 and result.samples.device == p.device
    assert torch.isfinite(result.samples).all()
    assert torch.all(result.samples > p.asarray(prior.lower))
    assert torch.all(result.samples < p.asarray(prior.upper))
    assert torch.equal(rng_before, torch.random.get_rng_state())
    assert torch.equal(cuda_rng_before, torch.cuda.get_rng_state(p.device))
    after_run = compiler_snapshot()
    assert after_run["graphs"] - before["graphs"] == 1
    assert after_run["calls"] > before["calls"]
    assert after_run["inductor"] > before["inductor"]
    assert after_run["cuda_graph_inputs"] > before["cuda_graph_inputs"]

    helper = _torch_nuts_type(torch, cuda_graphs=True)(compiled[0])
    reference = NUTS(p.log_prob_unconstrained)
    physical_points = [[30., 25.], [29.9, 25.1], [30.1, 24.9], [29.51, 24.51],
                       [30.49, 25.49], [29.7, 25.3], [30.3, 24.7], [29.8, 24.9]]
    points = [p.to_unconstrained(p.asarray(point)) for point in physical_points]
    held = []
    snapshots = []
    pair_calls = 0

    def pair(point):
        nonlocal pair_calls
        pair_calls += 1
        return helper.value_and_grad(point)

    for point in points:
        value, potential_gradient = pair(point)
        held.append((value, potential_gradient))
        snapshots.append((value.clone(), potential_gradient.clone()))
        expected_value, expected_gradient = reference.value_and_grad(point)
        torch.testing.assert_close(value, expected_value, rtol=1e-8, atol=1e-7)
        torch.testing.assert_close(potential_gradient, expected_gradient, rtol=1e-4, atol=1e-4)
        for axis in range(2):
            offset = torch.zeros_like(point)
            offset[axis] = 1e-5
            plus, _ = pair(point + offset)
            minus, _ = pair(point - offset)
            difference = (plus - minus) / (2e-5)
            torch.testing.assert_close(-potential_gradient[axis], difference, rtol=1e-4, atol=1e-4)
    for point in points * 2:
        pair(point)
    assert pair_calls == 56  # Eight held, 32 central-FD, sixteen intervening pairs.
    for (value, gradient), (old_value, old_gradient) in zip(held, snapshots):
        torch.testing.assert_close(value, old_value, rtol=0, atol=0)
        torch.testing.assert_close(gradient, old_gradient, rtol=0, atol=0)
        assert not value.requires_grad and not gradient.requires_grad
    assert len({value.data_ptr() for value, _ in held}) == len(held)
    assert len({gradient.data_ptr() for _, gradient in held}) == len(held)

    after = compiler_snapshot()
    assert after["graphs"] == after_run["graphs"]
    eager = run_nuts(p, starts, torch_compile=False, **options)
    torch.testing.assert_close(result.samples, eager.samples, rtol=1e-10, atol=1e-10)
    assert result.diagnostics.keys() == eager.diagnostics.keys()
    for name in result.diagnostics:
        torch.testing.assert_close(result.diagnostics[name], eager.diagnostics[name], rtol=1e-9, atol=1e-10)
    assert torch.equal(rng_before, torch.random.get_rng_state())
    assert torch.equal(cuda_rng_before, torch.cuda.get_rng_state(p.device))
    matrix, pointer, contents = coefficient_state[0]
    assert phenomd._TABLE_V_CACHE is cold_cache and cold_cache[key] is matrix
    assert matrix.data_ptr() == pointer
    torch.testing.assert_close(matrix, contents, rtol=0, atol=0)

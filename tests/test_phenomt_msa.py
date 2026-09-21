"""Native PhenomT MSA geometry contracts."""

import numpy as np
import pytest
import torch
import jax
import jax.numpy as jnp

from torchwave.approximants._phenomt_msa import msa_geometry as torch_geometry
from torchwave.approximants._phenomt_msa import msa_final_spin as torch_final_spin
from jaxwave.approximants._phenomt_msa import msa_geometry as jax_geometry
from jaxwave.approximants._phenomt_msa import msa_final_spin as jax_final_spin


jax.config.update("jax_enable_x64", True)


def _parameters(backend):
    a = backend.asarray if backend is jnp else torch.as_tensor
    return dict(
        mass1=a([30.0, 20.0], dtype=torch.float64 if backend is torch else jnp.float64),
        mass2=a([20.0, 30.0], dtype=torch.float64 if backend is torch else jnp.float64),
        spin1x=a([0.2, 0.0], dtype=torch.float64 if backend is torch else jnp.float64),
        spin1y=a([0.1, 0.0], dtype=torch.float64 if backend is torch else jnp.float64),
        spin1z=a([0.4, 0.3], dtype=torch.float64 if backend is torch else jnp.float64),
        spin2x=a([-0.1, 0.0], dtype=torch.float64 if backend is torch else jnp.float64),
        spin2y=a([0.2, 0.0], dtype=torch.float64 if backend is torch else jnp.float64),
        spin2z=a([-0.3, 0.2], dtype=torch.float64 if backend is torch else jnp.float64),
        inclination=a([0.8, 0.5], dtype=torch.float64 if backend is torch else jnp.float64),
        phic=a([0.2, 0.1], dtype=torch.float64 if backend is torch else jnp.float64),
        f_ref=a([20.0, 20.0], dtype=torch.float64 if backend is torch else jnp.float64),
    )


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_msa_geometry_batch_and_aligned_limits(backend):
    t = np.linspace(-0.02, 0.02, 16)
    f = np.linspace(20.0, 100.0, 16)
    if backend == "torch":
        out = torch_geometry(torch.as_tensor(t), torch.as_tensor(f), **_parameters(torch))
    else:
        out = jax_geometry(jnp.asarray(t), jnp.asarray(f), **_parameters(jnp))
    assert [tuple(x.shape) for x in out] == [(2, 16), (2, 16), (2, 16), (2,), (2,)]
    assert all(np.all(np.isfinite(np.asarray(x))) for x in out)


def test_msa_geometry_accepts_batched_time_grid_and_frequency_callback():
    t = torch.stack((torch.linspace(-0.02, 0.02, 8), torch.linspace(-0.01, 0.03, 8)))
    p = _parameters(torch)
    out = torch_geometry(t, torch.full_like(t, 30.0), **p,
                         frequency_at_time=lambda q: 30.0 + 5.0 * torch.exp(q))
    assert all(x.shape == (2, 8) for x in out[:3])
    assert all(torch.all(torch.isfinite(x)) for x in out)


def test_msa_final_spin_gradients_are_finite():
    m1 = torch.tensor(30.0, dtype=torch.float64, requires_grad=True)
    af = torch_final_spin(m1, 20.0, 0.0, 0.0, 0.4, 0.0, 0.0, -0.3)
    torch.autograd.grad(af, m1)
    assert torch.isfinite(af)

    def scalar_spin(m):
        return jax_final_spin(m, 20.0, 0.0, 0.0, 0.4, 0.0, 0.0, -0.3)

    assert np.isfinite(jax.grad(scalar_spin)(30.0))


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_msa_rejects_nonaligned_equal_mass_rows(backend):
    fn = torch_final_spin if backend == "torch" else jax_final_spin
    xp = torch if backend == "torch" else jnp
    args = (30.0, 30.0, 0.2, -0.1, 0.4, -0.12, 0.08, -0.3)
    with pytest.raises(ValueError, match="requires unequal masses"):
        fn(*args)
    if backend == "jax":
        compiled = jax.jit(fn)(*map(xp.asarray, args))
        assert np.isnan(np.asarray(compiled))
    aligned = fn(30.0, 30.0, 0.0, 0.0, 0.4, 0.0, 0.0, -0.3)
    assert np.isfinite(np.asarray(aligned))


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_msa_rejects_invalid_spin_domain(backend):
    fn = torch_final_spin if backend == "torch" else jax_final_spin
    args = (30.0, 20.0, 0.2, -0.1, 1.1, -0.12, 0.08, -0.3)
    with pytest.raises(ValueError, match="MSA final spin"):
        fn(*args)
    if backend == "jax":
        assert np.isnan(np.asarray(jax.jit(fn)(*map(jnp.asarray, args))))


def test_msa_jit_preserves_invalid_root_flag(monkeypatch):
    import jaxwave.approximants._phenomt_msa as module

    original = module.compute_msa_averaged_couplings
    def invalid_roots(*args):
        sav, s1, s2, valid = original(*args)
        return sav, s1, s2, jnp.zeros_like(valid)
    monkeypatch.setattr(module, "compute_msa_averaged_couplings", invalid_roots)
    args = (30.0, 20.0, 0.2, -0.1, 0.4, -0.12, 0.08, -0.3)
    with pytest.raises(ValueError, match="root domain"):
        module.msa_final_spin(*args)
    # Use a fresh trace so a previous compilation cannot retain the original
    # unpatched coupling function.
    result = jax.jit(lambda *values: module.msa_final_spin(*values))(*map(jnp.asarray, args))
    assert np.isnan(np.asarray(result))


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_msa_supplied_final_spin_preserves_domain(backend):
    xp = torch if backend == "torch" else jnp
    fn = torch_geometry if backend == "torch" else jax_geometry
    t = xp.asarray([-0.1, 0.01], dtype=xp.float64)
    f = xp.asarray([20.0, 80.0], dtype=xp.float64)
    def evaluate(m2, af):
        return fn(t, f, 30.0, m2, 0.2, -0.1, 0.4, -0.12, 0.08, -0.3,
                  0.8, 0.3, 20.0, final_spin=af, precession_version=22300)
    with pytest.raises(ValueError, match="requires unequal masses"):
        evaluate(30.0, 0.7)
    if backend == "jax":
        out = jax.jit(evaluate)(30.0, 0.7)
        assert all(np.isnan(np.asarray(value)).all() for value in out)
    # Analytical option 00 must propagate an invalid remnant even though
    # its angle expression itself does not depend on the ringdown spin.
    assert all(np.isnan(np.asarray(value)).all() for value in evaluate(20.0, float("nan")))


def test_msa_geometry_rejects_supplied_spin_for_equal_mass_precession():
    p = _parameters(jnp)
    p.update(mass1=jnp.asarray([30.0, 30.0]), mass2=jnp.asarray([30.0, 30.0]))
    t = jnp.asarray([-0.02, 0.01])
    f = jnp.asarray([20.0, 80.0])
    with pytest.raises(ValueError, match="requires unequal masses"):
        jax_geometry(t, f, **p, final_spin=jnp.asarray([0.7, 0.7]))
    result = jax.jit(lambda tt, ff: jax_geometry(
        tt, ff, **p, final_spin=jnp.asarray([0.7, 0.7])
    ))(t, f)
    for value in result:
        assert np.isnan(np.asarray(value)[0]).all()
        assert np.isfinite(np.asarray(value)[1]).all()


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_msa_nonpositive_reference_frequency_uses_20hz(backend):
    p = _parameters(torch if backend == "torch" else jnp)
    xp = torch if backend == "torch" else jnp
    t = xp.asarray([-0.1, -0.01, 0.01, 0.02], dtype=xp.float64)
    f = xp.asarray([20.0, 40.0, 80.0, 100.0], dtype=xp.float64)
    p0 = dict(p)
    p0["f_ref"] = xp.zeros_like(p["f_ref"])
    out0 = (torch_geometry if backend == "torch" else jax_geometry)(
        t, f, **p0, frequency_peak=xp.asarray([100.0, 100.0]), precession_version=22300,
    )
    out20 = (torch_geometry if backend == "torch" else jax_geometry)(
        t, f, **p, frequency_peak=xp.asarray([100.0, 100.0]), precession_version=22300,
    )
    for a, b in zip(out0, out20):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=0.0, atol=2e-13)


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_msa_22300_keeps_analytic_angles_through_ringdown(backend):
    xp = torch if backend == "torch" else jnp
    fn = torch_geometry if backend == "torch" else jax_geometry
    p = _parameters(xp)
    t = xp.asarray([-0.1, -0.01, 0.01, 0.02], dtype=xp.float64)
    f = xp.asarray([20.0, 40.0, 80.0, 100.0], dtype=xp.float64)
    kwargs = dict(p, frequency_peak=xp.asarray([100.0, 100.0]))
    analytic = fn(t, f, **kwargs, precession_version=22300)
    continued = fn(t, f, **kwargs, precession_version=22310)
    # The second half of the supplied carrier is deliberately below the
    # peak.  22300 follows it, while 22310 applies the peak-frequency RD
    # continuation.
    assert np.max(np.abs(np.asarray(analytic[0][:, 2:] - continued[0][:, 2:]))) > 1e-4


@pytest.mark.parametrize("backend", ["torch", "jax"])
@pytest.mark.parametrize("version", [22311, 22321])
def test_msa_numerical_options_are_grid_independent_and_batch_safe(backend, version):
    xp = torch if backend == "torch" else jnp
    fn = torch_geometry if backend == "torch" else jax_geometry
    p = _parameters(xp)
    short = xp.asarray([-0.12, -0.02, 0.005], dtype=xp.float64)
    long = xp.asarray([-0.2, -0.12, -0.05, -0.02, 0.0, 0.005, 0.02], dtype=xp.float64)
    ref = xp.asarray([-0.1, -0.1], dtype=xp.float64)
    meco = xp.asarray([-0.03, -0.03], dtype=xp.float64)
    def callback(q):
        return 30.0 + 5.0 * xp.exp(q)
    def evaluate(times):
        return fn(
            times, xp.full_like(times, 30.0), **p,
            frequency_peak=xp.asarray([100.0, 100.0]),
            precession_version=version, frequency_at_time=callback,
            reference_time=ref, meco_time=meco,
        )
    small = evaluate(short)
    large = evaluate(long)
    for a, b in zip(small[:3], large[:3]):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b)[..., [1, 3, 5]], rtol=2e-11, atol=2e-12)
    batched_times = xp.stack((short, short + 0.01))
    batched = evaluate(batched_times)
    assert [tuple(x.shape) for x in batched[:3]] == [(2, 3)] * 3
    assert all(np.isfinite(np.asarray(x)).all() for x in batched)


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_msa_zero_spin_geometry_gradients_are_finite(backend):
    xp = torch if backend == "torch" else jnp
    fn = torch_geometry if backend == "torch" else jax_geometry
    t = xp.asarray([-0.1, -0.01, 0.01], dtype=xp.float64)
    f = xp.asarray([20.0, 40.0, 80.0], dtype=xp.float64)
    if backend == "torch":
        spins = [torch.tensor(0.0, dtype=torch.float64, requires_grad=True) for _ in range(6)]
        out = fn(t, f, 30.0, 20.0, *spins, 0.8, 0.3, 20.0, precession_version=22310)
        grads = torch.autograd.grad(out[0].sum(), spins, allow_unused=True)
        assert all(g is None or torch.isfinite(g).all() for g in grads)
    else:
        def scalar_spins(s):
            return fn(t, f, 30.0, 20.0, *s, 0.8, 0.3, 20.0, precession_version=22310)[0].sum()
        grads = jax.grad(scalar_spins)(jnp.zeros(6, dtype=jnp.float64))
        assert np.isfinite(np.asarray(grads)).all()

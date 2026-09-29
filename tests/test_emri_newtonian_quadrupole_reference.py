"""Scoped reference and native checks for the weak-field eccentric EMRI limit."""

import math

import pytest

sp = pytest.importorskip("sympy")
mp = pytest.importorskip("mpmath")
torch = pytest.importorskip("torch")
jax = pytest.importorskip("jax")
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from diffgw_reference import emri_newtonian_quadrupole as ref
from torchwave.approximants.emri_kludge import NewtonianEccentricEMRI as TorchEMRI
from jaxwave.approximants.emri_kludge import NewtonianEccentricEMRI as JaxEMRI


def test_exact_circular_harmonics_and_observer():
    expected_plus = 2*ref.amp*(1+ref.u**2)*sp.cos(2*(ref.phi+ref.gamma))/ref.distance
    expected_cross = -4*ref.amp*ref.u*sp.sin(2*(ref.phi+ref.gamma))/ref.distance
    actual = ref.circular_limit()
    assert sp.simplify(actual[0]-expected_plus) == 0
    assert sp.simplify(actual[1]-expected_cross) == 0
    for harmonic in (1, 3, 4):
        assert sp.simplify(ref.plus_n.subs({ref.n: harmonic, ref.e: 0})) == 0
        assert sp.simplify(ref.cross_n.subs({ref.n: harmonic, ref.e: 0})) == 0


@pytest.mark.parametrize("ecc,inc,gamma", ((0.0, 0.7, 0.0),
                                           (0.2, 0.7, 0.3),
                                           (0.55, 1.2, -0.4)))
def test_scalar_reference_torch_jax_values(ecc, inc, gamma, reference_device):
    assert reference_device == "cpu"
    times = [0.0, 73.0, 146.0, 219.0, 292.0]
    kw = dict(total_mass=1e6, secondary_mass=10.0, p0=16.0, e0=ecc,
              inclination=inc, periapsis=gamma, phase0=0.2, distance=1000.0)
    expected = np.array([[float(x) for x in pair] for pair in
                         ref.polarizations(times, inclination=inc, periapsis=gamma,
                                           e0=ecc, phase0=0.2)])
    thp, thx = TorchEMRI().polarizations(torch.tensor(times, dtype=torch.float64), **kw)
    jhp, jhx = JaxEMRI().polarizations(jnp.array(times), **kw)
    for actual in ((thp.detach().numpy(), thx.detach().numpy()),
                   (np.asarray(jhp), np.asarray(jhx))):
        np.testing.assert_allclose(np.stack(actual, axis=-1), expected,
                                   rtol=3e-12, atol=2e-34)


@pytest.mark.parametrize("parameter", ("p0", "e0", "inclination", "periapsis"))
def test_independent_parameter_gradients(parameter, reference_device):
    assert reference_device == "cpu"
    times = [0.0, 61.0, 122.0, 183.0]
    defaults = dict(p0=16.0, e0=0.2, inclination=0.7, periapsis=0.3)
    position = dict(p0=2, e0=3, inclination=4, periapsis=5)[parameter]

    def scalar(value):
        args = [1e6, 10.0, defaults["p0"], defaults["e0"],
                defaults["inclination"], defaults["periapsis"]]
        args[position] = value
        return ref.polarizations(times, mass_solar=args[0], secondary_solar=args[1],
                                 p0=args[2], e0=args[3], inclination=args[4],
                                 periapsis=args[5])[-1][0]

    # Fixed high-precision five-point stencil avoids mp.diff shrinking its step
    # beyond the evaluator's fixed 70-digit working precision.
    with mp.workdps(60):
        center, h = mp.mpf(str(defaults[parameter])), mp.mpf("1e-4")
        expected = float((scalar(center-2*h)-8*scalar(center-h)
                          +8*scalar(center+h)-scalar(center+2*h))/(12*h))
    x = torch.tensor(defaults[parameter], dtype=torch.float64, requires_grad=True)
    tkw = {**defaults, parameter: x}
    actual_t = torch.autograd.grad(
        TorchEMRI().polarizations(torch.tensor(times, dtype=torch.float64), **tkw)[0][-1], x)[0]
    jkw = {**defaults}
    actual_j = jax.grad(lambda v: JaxEMRI().polarizations(
        jnp.array(times), **{**jkw, parameter: v})[0][-1])(
            jnp.asarray(defaults[parameter]))
    assert math.isclose(float(actual_t), expected, rel_tol=4e-10, abs_tol=1e-34)
    assert math.isclose(float(actual_j), expected, rel_tol=4e-10, abs_tol=1e-34)


def test_batch_jit_and_refinement(reference_device):
    assert reference_device == "cpu"
    times = np.linspace(0, 1200, 17)
    p_values = np.array([15.0, 17.0])
    e_values = np.array([0.1, 0.4])
    t_model, j_model = TorchEMRI(6), JaxEMRI(6)
    tb = t_model.polarizations(torch.tensor(times, dtype=torch.float64),
                               p0=torch.tensor(p_values), e0=torch.tensor(e_values))
    jb = jax.jit(lambda p, e: j_model.polarizations(jnp.array(times), p0=p, e0=e))(
        jnp.array(p_values), jnp.array(e_values))
    for idx in range(2):
        expected = ref.polarizations(times, p0=p_values[idx], e0=e_values[idx],
                                     n_harmonics=6)
        for native in (tb, jb):
            np.testing.assert_allclose(np.stack([np.asarray(native[0][idx]),
                                                 np.asarray(native[1][idx])], axis=-1),
                                       np.asarray(expected, dtype=float),
                                       rtol=3e-12, atol=2e-34)
    # RK4 should converge on the same orbital evolution, not just match a shared grid.
    coarse = j_model.trajectory(jnp.linspace(0, 6000, 21), 1e6, 10, 16, 0.4)[-1][-1]
    medium = j_model.trajectory(jnp.linspace(0, 6000, 41), 1e6, 10, 16, 0.4)[-1][-1]
    fine = j_model.trajectory(jnp.linspace(0, 6000, 81), 1e6, 10, 16, 0.4)[-1][-1]
    assert abs(float(coarse-medium)) < 1e-12
    assert abs(float(medium-fine)) < 1e-12


def test_harmonic_series_at_domain_edge(reference_device):
    assert reference_device == "cpu"
    times = [0.0, 10.0, 20.0]
    expected = np.asarray(ref.polarizations(times, p0=12.1, e0=0.6,
                                             n_harmonics=8), dtype=float)
    for actual in (TorchEMRI(8).polarizations(torch.tensor(times, dtype=torch.float64),
                                                   p0=12.1, e0=0.6),
                   JaxEMRI(8).polarizations(jnp.array(times), p0=12.1, e0=0.6)):
        np.testing.assert_allclose(np.stack([np.asarray(actual[0]),
                                             np.asarray(actual[1])], axis=-1),
                                   expected, rtol=3e-12, atol=2e-34)


def test_domain_rejects_unqualified_inputs():
    t = torch.linspace(0, 10, 3, dtype=torch.float64)
    with pytest.raises(ValueError):
        TorchEMRI().polarizations(t, p0=8)
    with pytest.raises(ValueError):
        JaxEMRI().polarizations(jnp.asarray(t.numpy()), e0=0.7)
    with pytest.raises(ValueError):
        TorchEMRI(n_harmonics=9)
    with pytest.raises(ValueError):
        JaxEMRI(n_harmonics=5.5)

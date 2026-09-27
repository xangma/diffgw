"""Published-precision 220 QNM fit versus native NSBH Torch and JAX."""

from itertools import product

import mpmath
import numpy as np
import pytest

from diffgw_reference.nsbh_london_qnm import london_qnm_220


MASSES = (2.8, 8.4, 30.0)
SPINS = (-0.9, -0.5, 0.0, 0.5, 0.9, 0.99)
CASES = tuple(product(MASSES, SPINS))
RELATIVE_LIMIT = 3e-12


@pytest.mark.parametrize("mass,spin", CASES)
def test_scalar_precision_and_positive_damping(mass, spin):
    low = london_qnm_220(mass, spin, dps=70)
    high = london_qnm_220(mass, spin, dps=90)
    ctx = mpmath.mp.clone()
    ctx.dps = 100
    for name in ("frequency_hz", "quality_factor"):
        a = ctx.mpf(getattr(low, name))
        b = ctx.mpf(getattr(high, name))
        assert abs((a - b) / b) <= ctx.mpf("1e-50")
        assert b > 0
    assert high.omega.imag > 0


@pytest.mark.parametrize("backend", ("torch", "jax"))
def test_native_london_220_against_paper_scalar(backend, reference_device):
    if reference_device != "cpu":
        pytest.fail("London QNM gate is frozen for CPU FP64")
    mass, spin = (np.asarray(x, dtype=np.float64) for x in zip(*CASES, strict=True))
    if backend == "torch":
        torch = pytest.importorskip("torch")
        from torchwave.approximants.imrphenom_nsbh import london_ringdown_properties

        f_rd, q = london_ringdown_properties(
            torch.as_tensor(mass, dtype=torch.float64),
            torch.as_tensor(spin, dtype=torch.float64),
        )
        actual = (f_rd.detach().cpu().numpy(), q.detach().cpu().numpy())
    else:
        jax = pytest.importorskip("jax")
        previous = jax.config.x64_enabled
        jax.config.update("jax_enable_x64", True)
        try:
            import jax.numpy as jnp
            from jaxwave.approximants.imrphenom_nsbh import london_ringdown_properties

            with jax.default_device(jax.devices("cpu")[0]):
                f_rd, q = london_ringdown_properties(
                    jnp.asarray(mass, dtype=jnp.float64),
                    jnp.asarray(spin, dtype=jnp.float64),
                )
                actual = (np.asarray(f_rd), np.asarray(q))
        finally:
            jax.config.update("jax_enable_x64", previous)
    expected = tuple(np.asarray(values) for values in zip(*(
        (float(result.frequency_hz), float(result.quality_factor))
        for result in (london_qnm_220(m, s) for m, s in CASES)
    ), strict=True))
    for measured, truth in zip(actual, expected, strict=True):
        assert np.all(np.isfinite(measured))
        assert np.all(measured > 0)
        assert np.max(np.abs(measured - truth) / truth) <= RELATIVE_LIMIT
    for i, spin_value in enumerate(SPINS):
        assert np.max(np.abs(actual[0][i::len(SPINS)] * mass[i::len(SPINS)]
                             / (actual[0][i] * mass[0]) - 1)) <= RELATIVE_LIMIT
        assert np.max(np.abs(actual[1][i::len(SPINS)] / actual[1][i] - 1)) <= RELATIVE_LIMIT


def test_scalar_rejects_unphysical_inputs():
    for mass, spin in ((0, 0), (-1, 0), (2.8, -1), (2.8, 1), (2.8, float("nan"))):
        with pytest.raises(ValueError):
            london_qnm_220(mass, spin)

# SPDX-License-Identifier: MIT
"""Independent-reference and frontend checks for XHM carrier rescaling."""

import numpy as np
import pytest

sp = pytest.importorskip("sympy")

from diffgw_reference import imrphenomxhm_inspiral_carrier as ref


@pytest.mark.parametrize("mode_m", (1, 2, 3, 4))
@pytest.mark.parametrize("index", (0, 2, 5, 7, 8))
def test_appendix_d_coefficient_transform_is_exact(mode_m, index):
    direct = ref.directly_rescaled_term(index=index, mode_m=mode_m)
    transformed = ref.rescaled_term_from_coefficients(index=index, mode_m=mode_m)
    assert sp.simplify(direct - transformed) == 0


@pytest.fixture(params=("torch", "jax"))
def frontend(request):
    if request.param == "torch":
        pytest.importorskip("torch")
        from torchwave.approximants.xhm_inspiral_phase_clean import (
            rescale_carrier_phase,
            rescale_carrier_phase_coefficients,
        )

        return request.param, rescale_carrier_phase, rescale_carrier_phase_coefficients

    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    from jaxwave.approximants.xhm_inspiral_phase_clean import (
        rescale_carrier_phase,
        rescale_carrier_phase_coefficients,
    )

    return request.param, rescale_carrier_phase, rescale_carrier_phase_coefficients


@pytest.mark.parametrize("mode_m", (1, 2, 3, 4))
def test_frontends_match_scalar_reference(frontend, mode_m):
    _, rescale_phase, rescale_coefficients = frontend
    indices = np.asarray((0.0, 2.0, 5.0, 7.0, 8.0))
    c22 = np.asarray((0.2, -1.1, 0.7, 2.3, -0.4))
    d22 = np.asarray((-0.3, 0.5, 1.2, -0.8, 0.1))
    cm, dm = rescale_coefficients(mode_m, indices, c22, d22)
    expected = [
        tuple(float(v.evalf(45)) for v in ref.rescaled_coefficients(int(i), mode_m, c, d))
        for i, c, d in zip(indices, c22, d22)
    ]
    np.testing.assert_allclose(np.asarray(cm), np.asarray(expected)[:, 0], rtol=2e-13, atol=2e-14)
    np.testing.assert_allclose(np.asarray(dm), np.asarray(expected)[:, 1], rtol=2e-13, atol=2e-14)

    phase = np.asarray((-8.0, 0.0, 3.5))
    np.testing.assert_allclose(
        np.asarray(rescale_phase(phase, mode_m)), (mode_m / 2.0) * phase,
        rtol=0.0, atol=0.0,
    )


def test_torch_gradient_and_jax_jit():
    torch = pytest.importorskip("torch")
    from torchwave.approximants.xhm_inspiral_phase_clean import rescale_carrier_phase as torch_rescale

    phase = torch.tensor((1.0, -2.0), dtype=torch.float64, requires_grad=True)
    torch_rescale(phase, 3).sum().backward()
    torch.testing.assert_close(phase.grad, torch.full_like(phase, 1.5), rtol=0.0, atol=0.0)

    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from jaxwave.approximants.xhm_inspiral_phase_clean import rescale_carrier_phase as jax_rescale

    got = jax.jit(lambda x: jax_rescale(x, 3))(jnp.asarray((1.0, -2.0), dtype=jnp.float64))
    np.testing.assert_array_equal(np.asarray(got), np.asarray((1.5, -3.0)))


@pytest.mark.parametrize("bad_mode", (0, -1, 1.5, True))
def test_frontends_reject_invalid_mode(frontend, bad_mode):
    _, rescale_phase, _ = frontend
    with pytest.raises(ValueError, match="positive integer"):
        rescale_phase(1.0, bad_mode)

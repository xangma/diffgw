"""Published positive-22 projection of a supplied aligned-frame Pv2 mode."""

import numpy as np
import pytest

from diffgw_reference.nrtidalv2_pv2_projection import project_positive_22
from test_nrtidalv2_phase_math_reference import require_module


# Frozen nondegenerate angles and arbitrary, unit-normalized complex mode data.
# These are an operator gate, not a physical BNS carrier or an Eq. (24) scaling.
H22 = np.array([1.2 + 0.4j, -0.3 + 0.8j, 0.6 - 1.1j])
DELTA = np.array([-0.2 + 0.7j, 0.9 - 0.5j, -0.4 - 0.3j])
ALPHA = np.array([0.2, 1.1, -0.7])
GAMMA = np.array([-0.4, 0.8, 0.3])
BETA = np.array([0.35, 0.9, 1.2])
THETA = np.array([0.6, 1.1, 2.2])
ZETA = np.array([0.15, -0.25, 0.45])
NORMALIZED_LIMIT = 1e-12


def _reference(h):
    pairs = [project_positive_22(value, *angles) for value, angles in zip(
        h, zip(ALPHA, GAMMA, BETA, THETA, ZETA))]
    return tuple(np.asarray([pair[index] for pair in pairs]) for index in (0, 1))


def _native(backend, h):
    if backend == "torch":
        torch = require_module("torch")
        from torchwave.approximants.precession_twisting_clean import (
            project_twisted_positive_modes,
        )
        real = lambda x: torch.as_tensor(x, dtype=torch.float64)
        mode = torch.as_tensor(h, dtype=torch.complex128)
        result = project_twisted_positive_modes(
            {(2, 2): mode}, *map(real, (ALPHA, GAMMA, BETA, THETA, ZETA)))
        return tuple(value.detach().cpu().numpy() for value in result)
    jax = require_module("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from jaxwave.approximants.phenompv2_waveform import _project
    result = _project(jnp.asarray(h, dtype=jnp.complex128),
                      *(jnp.asarray(angle, dtype=jnp.float64) for angle in
                        (ALPHA, GAMMA, BETA, THETA, ZETA)))
    return tuple(np.asarray(value) for value in result)


@pytest.mark.parametrize("backend", ("torch", "jax"))
def test_published_positive_22_projection_and_pre_rotation_addition(backend):
    for mode in (H22, DELTA, H22 + DELTA):
        expected = _reference(mode)
        actual = _native(backend, mode)
        for observed, target in zip(actual, expected):
            assert np.max(np.abs(observed - target)) / np.max(np.abs(mode)) <= NORMALIZED_LIMIT

    carrier = _native(backend, H22)
    correction = _native(backend, DELTA)
    composed = _native(backend, H22 + DELTA)
    # This angle set makes rotation observable; direct polarization addition
    # of the aligned-mode samples would be a different operation.
    assert np.max(np.abs(correction[0] - DELTA)) > 0.1
    assert np.max(np.abs(correction[1] - DELTA)) > 0.1
    for carrier_pol, correction_pol, composed_pol in zip(carrier, correction, composed):
        assert np.max(np.abs(composed_pol - carrier_pol - correction_pol)) / np.max(
            np.abs(DELTA)) <= NORMALIZED_LIMIT

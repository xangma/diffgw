"""Pin the inherited 21 inspiral fit wiring to public 122019 mode nodes.

The fixed dimensionless amplitudes were measured through the installed
LALSuite 7.26.15 binary's public FrequencySequenceOneMode API with (2,-1),
then divided by the paper's leading A22 strain. No LAL dependency is needed
when running this regression.
"""

import math

import numpy as np
import pytest
import torch

from diffgw_core.constants import C_SI, MPC_SI, MSUN_SEC
from torchwave.approximants.imrphenom_xhm import IMRPhenomXHM as TorchXHM


CASES = (
    (36.0, 29.0, -0.1, 0.2,
     (30.213420749419974, 45.320131124129965, 60.42684149883995),
     (0.012720485439561854, 0.015365845796237962, 0.01831301981569129)),
    (40.0, 20.0, 0.3, -0.1,
     (33.41541154173331, 50.123117312599966, 66.83082308346663),
     (0.016991088788359386, 0.01888657869031196, 0.021664278296433737)),
)


def _normalized_amplitude(strain, frequency_hz, m1, m2):
    eta = m1 * m2 / (m1 + m2) ** 2
    leading = (
        math.sqrt(16.0 * math.pi / 5.0)
        * math.sqrt(5.0 * eta / 24.0)
        / (math.pi ** (2.0 / 3.0) * (100.0 * MPC_SI / C_SI))
        * ((m1 + m2) * MSUN_SEC) ** (5.0 / 6.0)
        * frequency_hz ** (-7.0 / 6.0)
    )
    return np.abs(strain) / leading


@pytest.mark.parametrize("m1,m2,chi1,chi2,frequencies,expected", CASES)
def test_torch_21_inspiral_nodes_match_public_122019(
    m1, m2, chi1, chi2, frequencies, expected,
):
    frequency = torch.tensor(frequencies, dtype=torch.float64)
    with torch.no_grad():
        _, _, modes = TorchXHM(modes=((2, 1),))(
            frequency, m1, m2, chi1, chi2, 100.0, 0.6, 0.0, 0.0, 20.0,
            return_unprojected_modes=True,
        )
    actual = _normalized_amplitude(modes[(2, 1)].numpy(),
                                   frequency.numpy(), m1, m2)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=0.0)


@pytest.mark.parametrize("m1,m2,chi1,chi2,frequencies,expected", CASES)
def test_jax_21_inspiral_nodes_match_public_122019(
    m1, m2, chi1, chi2, frequencies, expected,
):
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from jaxwave.approximants.imrphenom_xhm import IMRPhenomXHM as JaxXHM

    frequency = jnp.asarray(frequencies, dtype=jnp.float64)
    _, _, modes = JaxXHM(modes=((2, 1),))(
        frequency, m1, m2, chi1, chi2, 100.0, 0.6, 0.0, 0.0, 20.0,
        return_unprojected_modes=True,
    )
    actual = _normalized_amplitude(np.asarray(modes[(2, 1)]),
                                   np.asarray(frequency), m1, m2)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=0.0)

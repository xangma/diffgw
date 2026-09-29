"""Published inherited Pv2 in-plane remnant-spin component for BNS inputs."""

import pytest

from diffgw_reference import nrtidalv2_pv2_remnant as reference
from test_nrtidalv2_phase_math_reference import require_module


CASES = (
    # Equal/unequal BNS masses and both possible dominant in-plane spins.
    (1.4, 1.4, 0.16, -0.09, -0.07, 0.04, 0.68),
    (1.7, 1.2, 0.04, 0.03, -0.27, 0.14, 0.61),
    (1.6, 1.1, -0.31, 0.13, 0.05, -0.08, 0.74),
)
LIMIT = 2e-15


@pytest.mark.parametrize("backend", ("torch", "jax"))
@pytest.mark.parametrize("case", CASES)
def test_pv2_inherited_remnant_spin(backend, case):
    m1, m2, s1x, s1y, s2x, s2y, aligned = case
    expected_chip = reference.effective_chip(m1, m2, s1x, s1y, s2x, s2y)
    expected_final = reference.final_spin(aligned, m1, m2, expected_chip)

    if backend == "torch":
        torch = require_module("torch")
        from torchwave.approximants.phenompv2_waveform_clean import (
            _precessing_final_spin,
        )
        from torchwave.approximants.precession_nnlo_clean import effective_chip
        values = [torch.tensor(value, dtype=torch.float64) for value in case]
        chip = effective_chip(*values[:2], *values[2:6])
        actual = _precessing_final_spin(values[6], values[0], values[1], chip)
    else:
        jax = require_module("jax")
        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp
        from jaxwave.approximants.phenompv2_waveform import (
            _effective_chip, _precessing_final_spin,
        )
        values = [jnp.asarray(value, dtype=jnp.float64) for value in case]
        chip = _effective_chip(*values[:2], *values[2:6])
        actual = _precessing_final_spin(values[6], values[0], values[1], chip)

    assert abs(float(chip) - float(expected_chip)) <= LIMIT
    assert abs(float(actual) - float(expected_final)) <= LIMIT

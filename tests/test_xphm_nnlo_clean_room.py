"""Clean-room checks for the published IMRPhenomXPHM NNLO-102 path."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from diffgw_core.constants import MSUN_SEC
from jaxwave.approximants._phenomt_precession import (
    _raw_angle_coefficients as jax_angle_coefficients,
    frequency_domain_nnlo_geometry as jax_nnlo_geometry,
)
from jaxwave.approximants.imrphenom_xphm import IMRPhenomXPHM as JaxPhenomXPHM
from torchwave.approximants._phenomt_precession import (
    _angle_coefficients as torch_angle_coefficients,
    frequency_domain_nnlo_geometry as torch_nnlo_geometry,
)
from torchwave.approximants.imrphenom_xphm import IMRPhenomXPHM as TorchPhenomXPHM
from torchwave.approximants.precession_nnlo_waveform_clean import waveform as scalar_waveform

jax.config.update("jax_enable_x64", True)


FROZEN_CASES = (
    {
        "mass1": 35.0,
        "mass2": 25.0,
        "spin1x": 0.2,
        "spin1y": 0.1,
        "spin1z": 0.2,
        "spin2x": -0.1,
        "spin2y": 0.15,
        "spin2z": -0.1,
        "distance": 500.0,
        "inclination": 0.7,
        "phic": 0.3,
        "f_ref": 20.0,
    },
    {
        "mass1": 42.0,
        "mass2": 17.0,
        "spin1x": 0.3,
        "spin1y": -0.15,
        "spin1z": -0.3,
        "spin2x": 0.12,
        "spin2y": 0.08,
        "spin2z": 0.2,
        "distance": 500.0,
        "inclination": 1.1,
        "phic": 0.71,
        "f_ref": 35.0,
    },
)


def _relative_l2(actual, expected):
    return np.linalg.norm(np.asarray(actual) - np.asarray(expected)) / np.linalg.norm(np.asarray(expected))


def test_published_nnlo_coefficients_match_frozen_paper_checkpoint():
    # Frozen numerical checkpoints evaluated from Pratten et al. (2021),
    # Eqs. (120)--(122), then stored at float64 precision. No implementation
    # output was used to construct them; this is not an executable SymPy oracle.
    alpha_expected = np.array(
        [
            -0.15997023809523809523809523809523809523809523809524,
            -0.043191964285714285714285714285714285714285714285714,
            -1.9168658470332577475434618291761148904006046863190,
            3.1321318081233502657019225230199923309987836511602,
            -1.8810917126853623117430321535582419957695889793337,
        ]
    )
    epsilon_expected = np.array(
        [
            -0.15997023809523809523809523809523809523809523809524,
            -0.043191964285714285714285714285714285714285714285714,
            -1.8933502220332577475434618291761148904006046863190,
            3.1163189097581882286648854859829552939617466141232,
            -1.8796807751853623117430321535582419957695889793337,
        ]
    )
    args = (35 / 144, -1 / 6, 7 / 12, 9 / 70, math.sqrt(5) / 10)
    torch_args = tuple(torch.tensor(x, dtype=torch.float64) for x in args)
    torch_alpha, torch_epsilon = torch_angle_coefficients(*torch_args)
    jax_alpha, jax_epsilon = jax_angle_coefficients(*(jnp.asarray(x, dtype=jnp.float64) for x in args))

    for actual, expected in (
        (torch_alpha.detach().numpy(), alpha_expected),
        (torch_epsilon.detach().numpy(), epsilon_expected),
        (np.asarray(jax_alpha), alpha_expected),
        (np.asarray(jax_epsilon), epsilon_expected),
    ):
        np.testing.assert_allclose(actual, expected, rtol=3e-15, atol=3e-15)


@pytest.mark.parametrize("case", FROZEN_CASES)
def test_nnlo_geometry_matches_between_tensor_backends(case):
    frequencies = np.array([20.0, 30.0, 40.0, 100.0])
    total_mass_seconds = (case["mass1"] + case["mass2"]) * MSUN_SEC
    v = (math.pi * total_mass_seconds * frequencies) ** (1 / 3)
    v0 = (math.pi * total_mass_seconds * case["f_ref"]) ** (1 / 3)
    names = (
        "mass1",
        "mass2",
        "spin1x",
        "spin1y",
        "spin1z",
        "spin2x",
        "spin2y",
        "spin2z",
        "inclination",
        "phic",
    )
    torch_result = torch_nnlo_geometry(
        torch.tensor(v, dtype=torch.float64),
        torch.tensor(v0, dtype=torch.float64),
        *(torch.tensor(case[name], dtype=torch.float64) for name in names),
        convention=1,
    )
    jax_result = jax_nnlo_geometry(
        jnp.asarray(v, dtype=jnp.float64),
        jnp.asarray(v0, dtype=jnp.float64),
        *(jnp.asarray(case[name], dtype=jnp.float64) for name in names),
        convention=1,
    )
    for torch_value, jax_value in zip(torch_result, jax_result):
        np.testing.assert_allclose(torch_value.detach().numpy(), np.asarray(jax_value), rtol=2e-12, atol=2e-12)


def test_opposed_j_domain_is_nan_in_torch_eager_compile_and_jax_jit():
    case = {
        "mass1": 100.0,
        "mass2": 5.0,
        "spin1x": 1e-9,
        "spin1y": 0.0,
        "spin1z": -0.95,
        "spin2x": 0.0,
        "spin2y": 0.0,
        "spin2z": 0.0,
        "inclination": 0.7,
        "phic": 0.3,
        "f_ref": 40.0,
    }
    frequencies = np.array([25.0, 40.0, 80.0])
    total_mass_seconds = (case["mass1"] + case["mass2"]) * MSUN_SEC
    v = (math.pi * total_mass_seconds * frequencies) ** (1 / 3)
    v0 = (math.pi * total_mass_seconds * case["f_ref"]) ** (1 / 3)
    names = (
        "mass1",
        "mass2",
        "spin1x",
        "spin1y",
        "spin1z",
        "spin2x",
        "spin2y",
        "spin2z",
        "inclination",
        "phic",
    )
    torch_args = (
        torch.tensor(v, dtype=torch.float64),
        torch.tensor(v0, dtype=torch.float64),
        *(torch.tensor(case[name], dtype=torch.float64) for name in names),
    )
    jax_args = (
        jnp.asarray(v, dtype=jnp.float64),
        jnp.asarray(v0, dtype=jnp.float64),
        *(jnp.asarray(case[name], dtype=jnp.float64) for name in names),
    )
    eager = torch_nnlo_geometry(*torch_args, convention=1)
    compiled = torch.compile(
        lambda *args: torch_nnlo_geometry(*args, convention=1),
        backend="eager",
        fullgraph=True,
    )(*torch_args)
    jitted = jax.jit(lambda *args: jax_nnlo_geometry(*args, convention=1))(*jax_args)

    for result in (eager, compiled, jitted):
        for value in result[:3]:
            assert np.isnan(np.asarray(value)).all()
        for value in result[3:]:
            assert np.isfinite(np.asarray(value)).all()


@pytest.mark.parametrize("case", FROZEN_CASES)
def test_quadrupole_waveform_matches_scalar_reference_and_jax(case):
    frequencies = np.arange(20.0, 301.0, 1.0)
    torch_frequencies = torch.tensor(frequencies, dtype=torch.float64)
    torch_model = TorchPhenomXPHM(
        modes=[(2, 2)], precession_method="nnlo", prec_version=102, convention=1, final_spin_mod=0
    )
    torch_result = torch_model(f=torch_frequencies, **case)
    scalar_result = scalar_waveform(torch_frequencies, **case, prec_version=102, convention=1, final_spin_mod=0)
    for actual, expected in zip(torch_result, scalar_result):
        assert _relative_l2(actual.detach().numpy(), expected.detach().numpy()) < 5e-13

    jax_model = JaxPhenomXPHM(
        modes=((2, 2),), precession_method="nnlo", prec_version=102, convention=1, final_spin_mod=0
    )
    jax_result = jax_model(f=jnp.asarray(frequencies, dtype=jnp.float64), **case)
    for torch_value, jax_value in zip(torch_result, jax_result):
        assert _relative_l2(np.asarray(jax_value), torch_value.detach().numpy()) < 5e-12

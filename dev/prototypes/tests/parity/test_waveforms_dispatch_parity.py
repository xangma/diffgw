"""Cross-framework parity tests: torchwave.waveforms vs jaxwave.waveforms.

Tests get_fd_waveform and WaveformGenerator dispatch parity across all 13 supported approximants:
1. TaylorF2
2. TaylorF2_HM
3. TaylorF2_Eccentric
4. IMRPhenomD
5. IMRPhenomP
6. IMRPhenomPv2
7. IMRPhenomD_NRTidal
8. IMRPhenomD_NRTidalv2
9. IMRPhenomHM
10. IMRPhenomXAS
11. IMRPhenomXHM
12. IMRPhenomNSBH
13. KerrRingdown
"""

import numpy as np
import pytest
import torch
import jax.numpy as jnp

from torchwave.waveforms import (
    get_fd_waveform as torch_get_fd,
    WaveformGenerator as TorchWaveformGenerator,
)
from jaxwave.waveforms import (
    get_fd_waveform as jax_get_fd,
    WaveformGenerator as JaxWaveformGenerator,
)


APPROXIMANTS_AND_PARAMS = [
    (
        "TaylorF2",
        {"mass1": 1.4, "mass2": 1.2, "chi1": 0.05, "chi2": -0.02, "lambda1": 300.0, "lambda2": 400.0},
        {"f_lower": 30.0, "f_final": 500.0, "delta_f": 1.0},
        1e-10,
    ),
    (
        "TaylorF2_HM",
        {"mass1": 25.0, "mass2": 10.0, "chi1": 0.2, "chi2": -0.1, "inclination": 0.6},
        {"f_lower": 25.0, "f_final": 400.0, "delta_f": 1.0},
        1e-10,
    ),
    (
        "TaylorF2_Eccentric",
        {"mass1": 1.4, "mass2": 1.3, "chi1": 0.0, "chi2": 0.0, "eccentricity": 0.05},
        {"f_lower": 30.0, "f_final": 300.0, "delta_f": 1.0},
        1e-10,
    ),
    (
        "IMRPhenomD",
        {"mass1": 30.0, "mass2": 20.0, "chi1": 0.3, "chi2": -0.2},
        {"f_lower": 20.0, "f_final": 512.0, "delta_f": 1.0},
        1e-12,
    ),
    (
        "IMRPhenomP",
        {"mass1": 35.0, "mass2": 25.0, "chi1": 0.2, "chi2": -0.1, "inclination": 0.5},
        {"f_lower": 20.0, "f_final": 400.0, "delta_f": 1.0},
        1e-10,
    ),
    (
        "IMRPhenomPv2",
        {"mass1": 35.0, "mass2": 25.0, "chi1": 0.2, "chi2": -0.1, "inclination": 0.5},
        {"f_lower": 20.0, "f_final": 400.0, "delta_f": 1.0},
        1e-10,
    ),
    (
        "IMRPhenomD_NRTidal",
        {"mass1": 1.4, "mass2": 1.2, "chi1": 0.0, "chi2": 0.0, "lambda1": 350.0, "lambda2": 450.0},
        {"f_lower": 30.0, "f_final": 800.0, "delta_f": 2.0},
        1e-11,
    ),
    (
        "IMRPhenomD_NRTidalv2",
        {"mass1": 1.4, "mass2": 1.2, "chi1": 0.0, "chi2": 0.0, "lambda1": 350.0, "lambda2": 450.0},
        {"f_lower": 30.0, "f_final": 800.0, "delta_f": 2.0},
        1e-11,
    ),
    (
        "IMRPhenomHM",
        {"mass1": 40.0, "mass2": 15.0, "chi1": 0.4, "chi2": -0.3, "inclination": 0.7},
        {"f_lower": 20.0, "f_final": 600.0, "delta_f": 1.0},
        1e-10,
    ),
    (
        "IMRPhenomXAS",
        {"mass1": 30.0, "mass2": 20.0, "chi1": 0.5, "chi2": -0.4},
        {"f_lower": 20.0, "f_final": 512.0, "delta_f": 1.0},
        1e-12,
    ),
    (
        "IMRPhenomXHM",
        {"mass1": 35.0, "mass2": 20.0, "chi1": 0.3, "chi2": -0.2, "inclination": 0.6},
        {"f_lower": 20.0, "f_final": 500.0, "delta_f": 1.0},
        1e-10,
    ),
    (
        "IMRPhenomNSBH",
        {"mass1": 5.0, "mass2": 1.4, "chi1": 0.2, "chi2": 0.0, "lambda2": 400.0},
        {"f_lower": 20.0, "f_final": 600.0, "delta_f": 1.0},
        1e-10,
    ),
    (
        "KerrRingdown",
        {"mass1": 50.0, "mass2": 50.0, "chi1": 0.0, "chi2": 0.0, "inclination": 0.5},
        {"f_lower": 100.0, "f_final": 1024.0, "delta_f": 2.0},
        1e-10,
    ),
    (
        "IMRPhenomXP",
        {"mass1": 30.0, "mass2": 20.0, "chi1": 0.2, "chi2": -0.1, "inclination": 0.5, "precession_method": "nnlo"},
        {"f_lower": 20.0, "f_final": 400.0, "delta_f": 1.0},
        1e-10,
    ),
    (
        "IMRPhenomXPHM",
        {"mass1": 35.0, "mass2": 20.0, "chi1": 0.3, "chi2": -0.2, "inclination": 0.6},
        {"f_lower": 20.0, "f_final": 400.0, "delta_f": 1.0},
        1e-10,
    ),
    (
        "seobnrv5rom",
        {"mass1": 30.0, "mass2": 20.0, "chi1": 0.2, "chi2": -0.1},
        {"f_lower": 20.0, "f_final": 300.0, "delta_f": 1.0},
        1e-10,
    ),
    (
        "SEOBNR_ODE",
        {"mass1": 30.0, "mass2": 20.0, "chi1": 0.0, "chi2": 0.0, "num_steps": 10},
        {"f_lower": 30.0, "f_final": 100.0, "delta_f": 5.0},
        1e-10,
    ),
    (
        "TEOBResumS",
        {"mass1": 1.4, "mass2": 1.3, "chi1": 0.0, "chi2": 0.0, "lambda1": 300.0, "lambda2": 400.0},
        {"f_lower": 30.0, "f_final": 400.0, "delta_f": 2.0},
        1e-10,
    ),
]


@pytest.mark.parametrize("approximant, params, grid, tol", APPROXIMANTS_AND_PARAMS)
def test_get_fd_waveform_parity(approximant, params, grid, tol):
    """Test get_fd_waveform parity across all 13 approximants."""
    hp_t, hc_t = torch_get_fd(approximant=approximant, **params, **grid)
    hp_j, hc_j = jax_get_fd(approximant=approximant, **params, **grid)

    assert hp_t.shape == hp_j.shape
    assert hc_t.shape == hc_j.shape

    hp_t_np = hp_t.detach().numpy()
    hc_t_np = hc_t.detach().numpy()
    hp_j_np = np.asarray(hp_j)
    hc_j_np = np.asarray(hc_j)

    np.testing.assert_allclose(hp_j_np, hp_t_np, rtol=tol, atol=1e-30)
    np.testing.assert_allclose(hc_j_np, hc_t_np, rtol=tol, atol=1e-30)


@pytest.mark.parametrize("approximant, params, grid, tol", [
    (
        "IMRPhenomD",
        {"mass1": 32.0, "mass2": 18.0, "chi1": 0.25, "chi2": -0.15},
        {"f_lower": 20.0, "f_final": 500.0, "delta_f": 1.0},
        1e-12,
    ),
    (
        "IMRPhenomXHM",
        {"mass1": 40.0, "mass2": 20.0, "chi1": 0.4, "chi2": -0.3, "inclination": 0.7},
        {"f_lower": 20.0, "f_final": 500.0, "delta_f": 1.0},
        1e-10,
    ),
    (
        "TaylorF2",
        {"mass1": 1.5, "mass2": 1.3, "chi1": 0.02, "chi2": -0.01},
        {"f_lower": 30.0, "f_final": 400.0, "delta_f": 1.0},
        1e-10,
    ),
])
def test_waveform_generator_parity(approximant, params, grid, tol):
    """Test WaveformGenerator class-based interface parity."""
    gen_t = TorchWaveformGenerator(approximant=approximant, **grid)
    gen_j = JaxWaveformGenerator(approximant=approximant, **grid)

    # Call with kwargs
    hp_t, hc_t = gen_t(**params)
    hp_j, hc_j = gen_j(**params)

    np.testing.assert_allclose(np.asarray(hp_j), hp_t.detach().numpy(), rtol=tol, atol=1e-30)
    np.testing.assert_allclose(np.asarray(hc_j), hc_t.detach().numpy(), rtol=tol, atol=1e-30)

    # Call with dict parameters and alternative parameter names
    dict_params = {
        "mass_1": params["mass1"],
        "mass_2": params["mass2"],
        "spin_1z": params.get("chi1", 0.0),
        "spin_2z": params.get("chi2", 0.0),
    }
    if "inclination" in params:
        dict_params["theta_jn"] = params["inclination"]

    hp_t_dict, hc_t_dict = gen_t(dict_params)
    hp_j_dict, hc_j_dict = gen_j(dict_params)

    np.testing.assert_allclose(np.asarray(hp_j_dict), hp_t_dict.detach().numpy(), rtol=tol, atol=1e-30)
    np.testing.assert_allclose(np.asarray(hc_j_dict), hc_t_dict.detach().numpy(), rtol=tol, atol=1e-30)


def test_sample_frequencies_grid_dispatch():
    """Verify arbitrary sample_frequencies grid dispatch parity."""
    f_np = np.geomspace(20.0, 500.0, 100)

    hp_t, hc_t = torch_get_fd(
        approximant="IMRPhenomXHM",
        mass1=36.0,
        mass2=18.0,
        chi1=0.2,
        chi2=-0.1,
        sample_frequencies=torch.tensor(f_np, dtype=torch.float64),
    )
    hp_j, hc_j = jax_get_fd(
        approximant="IMRPhenomXHM",
        mass1=36.0,
        mass2=18.0,
        chi1=0.2,
        chi2=-0.1,
        sample_frequencies=jnp.array(f_np),
    )

    np.testing.assert_allclose(np.asarray(hp_j), hp_t.detach().numpy(), rtol=1e-10, atol=1e-30)
    np.testing.assert_allclose(np.asarray(hc_j), hc_t.detach().numpy(), rtol=1e-10, atol=1e-30)


@pytest.mark.parametrize(
    "alias_name, canonical_name",
    [
        ("taylorf2hm", "TaylorF2_HM"),
        ("imrphenomd", "IMRPhenomD"),
        ("phenomxas", "IMRPhenomXAS"),
        ("xas", "IMRPhenomXAS"),
        ("imrphenomxhm", "IMRPhenomXHM"),
        ("nsbh", "IMRPhenomNSBH"),
        ("ringdown", "KerrRingdown"),
    ],
)
def test_approximant_alias_names(alias_name, canonical_name):
    """Verify that alias names dispatch identically to canonical names in jaxwave."""
    f = jnp.linspace(30.0, 400.0, 50)
    params = {"mass1": 30.0, "mass2": 20.0, "chi1": 0.1, "chi2": -0.1}

    hp_alias, hc_alias = jax_get_fd(approximant=alias_name, sample_frequencies=f, **params)
    hp_canon, hc_canon = jax_get_fd(approximant=canonical_name, sample_frequencies=f, **params)

    np.testing.assert_allclose(np.asarray(hp_alias), np.asarray(hp_canon), rtol=1e-14, atol=1e-30)
    np.testing.assert_allclose(np.asarray(hc_alias), np.asarray(hc_canon), rtol=1e-14, atol=1e-30)

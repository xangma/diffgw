"""Parity tests between torchwave.approximants.rom and jaxwave.approximants.rom."""

import numpy as np
import torch
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from torchwave.approximants.rom import (
    OfficialSEOBNRv4ROM as TorchOfficialSEOBNRv4ROM,
    bspline_basis as torch_bspline_basis,
    tensor_product_basis as torch_tensor_product_basis,
    cubic_spline_interpolate as torch_cubic_spline_interpolate,
    independent_ringdown_mf as torch_independent_ringdown_mf,
    table_convention_ringdown_mf as torch_table_convention_ringdown_mf,
)
from jaxwave.approximants.rom import (
    OfficialSEOBNRv4ROM as JaxOfficialSEOBNRv4ROM,
    bspline_basis as jax_bspline_basis,
    tensor_product_basis as jax_tensor_product_basis,
    cubic_spline_interpolate as jax_cubic_spline_interpolate,
    independent_ringdown_mf as jax_independent_ringdown_mf,
    table_convention_ringdown_mf as jax_table_convention_ringdown_mf,
)

def test_bspline_basis_parity():
    breaks = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    x = np.linspace(0.01, 0.99, 50)
    order = 4

    t_b = torch_bspline_basis(torch.tensor(x, dtype=torch.float64),
                              torch.tensor(breaks, dtype=torch.float64), order=order)
    j_b = jax_bspline_basis(jnp.array(x, dtype=jnp.float64),
                            jnp.array(breaks, dtype=jnp.float64), order=order)

    np.testing.assert_allclose(np.asarray(j_b), t_b.numpy(), rtol=1e-14, atol=1e-15)


def test_tensor_product_basis_parity():
    g1 = np.array([0.0, 0.5, 1.0])
    g2 = np.array([-1.0, 0.0, 1.0])
    g3 = np.array([-1.0, 0.0, 1.0])
    eta = np.array([0.2, 0.24])
    chi1 = np.array([-0.5, 0.4])
    chi2 = np.array([0.1, -0.3])
    order = 3

    t_tp = torch_tensor_product_basis(
        torch.tensor(eta, dtype=torch.float64),
        torch.tensor(chi1, dtype=torch.float64),
        torch.tensor(chi2, dtype=torch.float64),
        (
            torch.tensor(g1, dtype=torch.float64),
            torch.tensor(g2, dtype=torch.float64),
            torch.tensor(g3, dtype=torch.float64),
        ),
        order=order,
    )
    j_tp = jax_tensor_product_basis(
        jnp.array(eta, dtype=jnp.float64),
        jnp.array(chi1, dtype=jnp.float64),
        jnp.array(chi2, dtype=jnp.float64),
        (
            jnp.array(g1, dtype=jnp.float64),
            jnp.array(g2, dtype=jnp.float64),
            jnp.array(g3, dtype=jnp.float64),
        ),
        order=order,
    )

    np.testing.assert_allclose(np.asarray(j_tp), t_tp.numpy(), rtol=1e-14, atol=1e-15)


def test_cubic_spline_interpolate_parity():
    x_nodes = np.array([10.0, 30.0, 60.0, 120.0, 250.0])
    y_nodes = np.sin(x_nodes / 50.0)
    x_eval = np.linspace(10.0, 250.0, 100)

    t_y = torch_cubic_spline_interpolate(
        torch.tensor(x_eval, dtype=torch.float64),
        torch.tensor(x_nodes, dtype=torch.float64),
        torch.tensor(y_nodes, dtype=torch.float64),
    )
    j_y = jax_cubic_spline_interpolate(
        jnp.array(x_eval, dtype=jnp.float64),
        jnp.array(x_nodes, dtype=jnp.float64),
        jnp.array(y_nodes, dtype=jnp.float64),
    )

    np.testing.assert_allclose(np.asarray(j_y), t_y.numpy(), rtol=1e-14, atol=1e-15)


def test_ringdown_mf_parity():
    test_params = [
        (30.0, 20.0, 0.2, -0.1),
        (10.0, 10.0, 0.0, 0.0),
        (50.0, 10.0, 0.8, 0.5),
        (80.0, 20.0, -0.6, 0.3),
    ]
    for m1, m2, s1, s2 in test_params:
        t_ind = torch_independent_ringdown_mf(
            torch.tensor(m1, dtype=torch.float64),
            torch.tensor(m2, dtype=torch.float64),
            torch.tensor(s1, dtype=torch.float64),
            torch.tensor(s2, dtype=torch.float64),
        )
        j_ind = jax_independent_ringdown_mf(
            jnp.array(m1, dtype=jnp.float64),
            jnp.array(m2, dtype=jnp.float64),
            jnp.array(s1, dtype=jnp.float64),
            jnp.array(s2, dtype=jnp.float64),
        )
        np.testing.assert_allclose(float(j_ind), t_ind.item(), rtol=1e-14, atol=1e-15)

        t_tab = torch_table_convention_ringdown_mf(
            torch.tensor(m1, dtype=torch.float64),
            torch.tensor(m2, dtype=torch.float64),
            torch.tensor(s1, dtype=torch.float64),
            torch.tensor(s2, dtype=torch.float64),
        )
        j_tab = jax_table_convention_ringdown_mf(
            jnp.array(m1, dtype=jnp.float64),
            jnp.array(m2, dtype=jnp.float64),
            jnp.array(s1, dtype=jnp.float64),
            jnp.array(s2, dtype=jnp.float64),
        )
        np.testing.assert_allclose(float(j_tab), t_tab.item(), rtol=1e-14, atol=1e-15)


def test_official_seobnrv4_rom_parity(official_rom_path):
    t_wrapper = TorchOfficialSEOBNRv4ROM(official_rom_path, flow=20.0, fhigh=256.0)
    j_wrapper = JaxOfficialSEOBNRv4ROM(official_rom_path, flow=20.0, fhigh=256.0)

    f = np.array([20.0, 30.0, 50.0, 100.0, 200.0])
    test_cases = [
        (30.0, 20.0, 0.2, -0.1),
        (40.0, 30.0, 0.0, 0.0),
    ]

    for m1, m2, s1, s2 in test_cases:
        t_hp, t_hc = t_wrapper(
            torch.tensor(f, dtype=torch.float64),
            mass1=m1,
            mass2=m2,
            chi1=s1,
            chi2=s2,
            distance=100.0,
            inclination=0.6,
            phic=0.2,
            f_ref=20.0,
        )
        j_hp, j_hc = j_wrapper(
            jnp.array(f, dtype=jnp.float64),
            mass1=m1,
            mass2=m2,
            chi1=s1,
            chi2=s2,
            distance=100.0,
            inclination=0.6,
            phic=0.2,
            f_ref=20.0,
        )

        np.testing.assert_allclose(np.asarray(j_hp), t_hp.detach().numpy(), rtol=1e-12, atol=1e-25)
        np.testing.assert_allclose(np.asarray(j_hc), t_hc.detach().numpy(), rtol=1e-12, atol=1e-25)

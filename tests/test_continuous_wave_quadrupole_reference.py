"""Paper-to-scalar and scalar-to-Torch/JAX checks for the CW 2f carrier."""

import math

import jax
import jax.numpy as jnp
import mpmath as mp
import numpy as np
import pytest
import torch

from diffgw_reference import continuous_wave_quadrupole as ref
from jaxwave.approximants import continuous_wave as j_cw
from torchwave.approximants import continuous_wave as t_cw

jax.config.update("jax_enable_x64", True)


def test_jks_si_benchmark_and_frequency_convention():
    # JKS v1 PDF Eq. (25) rounds Eq. (23) to 4.23e-25 at f_rot=100 Hz,
    # epsilon=1e-5, Izz=1e45 g cm², d=1 kpc.
    kpc_m = 3.085677581491367e19
    with mp.workdps(60):
        h0 = ref.h0_from_rotation_frequency(100, 1e-5, kpc_m, 1e38)
        assert math.isclose(float(h0), 4.23e-25, rel_tol=2e-3)
        assert ref.h0_from_gw_frequency(200, 1e-5, kpc_m, 1e38) == h0


@pytest.mark.parametrize("f0,eps,distance,inertia", [
    (20.0, 1e-6, 1e19, 1e38),
    (200.0, 1e-5, 3.085677581491367e19, 1.2e38),
    (700.0, 3e-7, 5e20, 0.8e38),
])
def test_h0_and_frequency_derivative_against_independent_scalar(f0, eps, distance, inertia):
    with mp.workdps(60):
        target = float(ref.h0_from_gw_frequency(f0, eps, distance, inertia))
        target_grad = float(ref.dh0_df0(f0, eps, distance, inertia))

    ft = torch.tensor(f0, dtype=torch.float64, requires_grad=True)
    tv = t_cw.continuous_wave_h0(ft, eps, distance, i_zz=inertia)
    tg = torch.autograd.grad(tv, ft)[0]
    jf = lambda freq: j_cw.continuous_wave_h0(freq, eps, distance, i_zz=inertia)
    jv, jg = jax.jit(jax.value_and_grad(jf))(jnp.asarray(f0, dtype=jnp.float64))
    np.testing.assert_allclose([tv.item(), float(jv)], target, rtol=1e-13, atol=0)
    np.testing.assert_allclose([tg.item(), float(jg)], target_grad, rtol=1e-13, atol=0)


def test_nondefault_inertia_reaches_both_class_wrappers():
    t = np.linspace(0.0, 0.1, 16, dtype=np.float64)
    kw = dict(ra=1.1, dec=-0.2, f0=30.0, ellipticity=1e-5,
              distance=1e20, cos_iota=0.6, psi=0.3, phi0=0.2)
    inertia = 1.7e38
    with mp.workdps(60):
        h0 = float(ref.h0_from_gw_frequency(kw["f0"], kw["ellipticity"], kw["distance"], inertia))
    direct_kw = {key: value for key, value in kw.items() if key not in ("ellipticity", "distance")}
    for module, samples in ((t_cw, torch.from_numpy(t)), (j_cw, jnp.asarray(t))):
        via_inertia = module.ContinuousWave(i_zz=inertia)(samples, **kw)
        via_h0 = module.ContinuousWave()(samples, h0=h0, **direct_kw)
        np.testing.assert_allclose(np.asarray(via_inertia), np.asarray(via_h0), rtol=1e-13, atol=1e-35)

"""Development-only source-choice parity at fixed physical states."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from dev.native_rom.eob.jax.source_calibration_hamiltonian import (
    SourceCalibratedHistoricalHamiltonian as JaxHam,
)
from dev.native_rom.eob.torch.source_calibration_hamiltonian import (
    SourceCalibratedHistoricalHamiltonian as TorchHam,
)
from jaxwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux as JaxFlux
from torchwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux as TorchFlux

jax.config.update("jax_enable_x64", True)

SOURCE_FLUX = dict(
    spin_truncation="tabulated", retain_pan_cubic_spin=True,
    mode_21_higher_spin="rho_hybrid", mode_31_higher_spin="lal_v4_rho",
    mode_32_spin_square="lal_v4_zero", mode_33_higher_spin="lal_v4_rho",
    mode_43_higher_spin="lal_v4_rho",
)


@pytest.mark.parametrize("m1,m2,chi1,chi2,r,pr,pphi", [
    (30., 20., 0., 0., 9., -.01, 3.4),
    (30., 20., .8, .6, 6.5, -.02, 3.1),
    (45., 5., -.8, .3, 11., -.003, 3.7),
])
def test_source_hamiltonian_flux_and_derivatives_match_torch(
        m1, m2, chi1, chi2, r, pr, pphi):
    nu = m1*m2/(m1+m2)**2
    delta = (m1-m2)/(m1+m2)
    jh = JaxHam(nu, chi1, chi2, delta=delta)
    th = TorchHam(*(torch.tensor(x, dtype=torch.float64)
                    for x in (nu, chi1, chi2)),
                  delta=torch.tensor(delta, dtype=torch.float64))
    jf, tf = JaxFlux(**SOURCE_FLUX), TorchFlux(**SOURCE_FLUX)
    js = jnp.asarray([r, 0., pr, pphi])
    ts = torch.tensor([r, 0., pr, pphi], dtype=torch.float64)
    for jvalue, tvalue in zip(jh.calibration_parameters(js[0]),
                              th.calibration_parameters(ts[0])):
        np.testing.assert_allclose(jvalue, tvalue.detach(), rtol=1e-13, atol=1e-12)
    jeff = jax.jit(jh.effective_hamiltonian)(js[0], js[2], js[3])
    teff = th.effective_hamiltonian(ts[0], ts[2], ts[3])
    np.testing.assert_allclose(jeff, teff.detach(), rtol=3e-11, atol=3e-11)
    jq, tq = jf.from_hamiltonian(js, jh), tf.from_hamiltonian(ts, th)
    for key in ("omega", "omega_zero_radial", "heff_over_mu", "hreal_over_M"):
        np.testing.assert_allclose(jq[key], tq[key].detach(), rtol=3e-11, atol=3e-11)
    jm, tm = jf._signed_amplitudes(js, **jq), tf._signed_amplitudes(ts, **tq)
    assert set(jm) == set(tm)
    for mode in jm:
        np.testing.assert_allclose(jm[mode], tm[mode].detach(), rtol=5e-11,
                                   atol=2e-13, err_msg=str(mode))
    jtorque = jax.jit(lambda state: jf.radiation_reaction_forces(
        jq["omega"], state[2], state[3], r=state[0], hamiltonian=jh)[1])(js)
    ttorque = tf.radiation_reaction_forces(
        tq["omega"], ts[2], ts[3], r=ts[0], hamiltonian=th)[1]
    np.testing.assert_allclose(jtorque, ttorque.detach(), rtol=5e-11, atol=2e-13)

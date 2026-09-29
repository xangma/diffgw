"""Frozen ppE inspiral adapter gate on real Torch and JAX TaylorF2 carriers.

Yunes--Pretorius, arXiv:0909.3328v2, Eqs. (1), (11), (47), specifies the
inspiral multiplier below f_IM. The complex ratio cancels the GR carrier and
therefore tests only the adapter and its parameter derivatives.
"""

import importlib

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import mpmath as mp
import numpy as np
import torch

from diffgw_reference import ppe as ref


t_ppe = importlib.import_module("torchwave.approximants.ppe")
j_ppe = importlib.import_module("jaxwave.approximants.ppe")
t_taylor = importlib.import_module("torchwave.approximants.taylorf2")
j_taylor = importlib.import_module("jaxwave.approximants.taylorf2")

FREQUENCIES = np.array([30.0, 40.0, 55.0, 70.0])
MASS = 25.0
DISTANCE = 200.0
INCLINATION = 0.4
REDSHIFT = 0.2
ALPHA = 0.4
A = 0.5
BETA = 0.001
N = -2
LAMBDA_G = 1e17


def _source_targets():
    with mp.workdps(70):
        mass_sec = ref.chirp_mass_seconds(MASS, MASS)
        total_sec = mp.mpf(2 * MASS) * mp.mpf(ref._G) * mp.mpf(ref._MSUN) / mp.mpf(ref._C)**3
        eta = mp.mpf(1) / 4
        f_im = (mp.mpf("0.29") * eta**2 + mp.mpf("0.045") * eta + mp.mpf("0.096")) / (mp.pi * total_sec)
        f_isco = 1 / (6 * mp.sqrt(6) * mp.pi * total_sec)
        assert FREQUENCIES[-1] < f_im
        assert FREQUENCIES[-1] < f_isco

        d0 = ref.d0_from_luminosity_distance(DISTANCE, "0.2")
        values, d_alpha, d_beta = [], [], []
        for frequency in FREQUENCIES:
            f = float(frequency)
            phase_factor = ref.carrier_multiplier(f, MASS, MASS, "0.001", N,
                                                  d0, "1e17", "0.2")
            ratio = ref.carrier_amplitude_multiplier(f, MASS, MASS, "0.4", "0.5",
                                                     "0.001", N, d0, "1e17", "0.2")
            u = mp.pi * mass_sec * mp.mpf(f)
            values.append(complex(ratio))
            d_alpha.append(complex(u ** mp.mpf("0.5") * phase_factor))
            d_beta.append(complex(-1j * u ** (mp.mpf(N) / 3) * ratio))
    return np.array(values), d_alpha[2], d_beta[2]


def test_real_taylorf2_inspiral_ratio_and_parameter_derivatives():
    expected, expected_alpha, expected_beta = _source_targets()
    kwargs = dict(mass1=MASS, mass2=MASS, distance=DISTANCE,
                  inclination=INCLINATION)
    modified = dict(alpha_ppe=ALPHA, a_ppe=A, beta_ppe=BETA, b_ppe=N,
                    lambda_g=LAMBDA_G, redshift=REDSHIFT)

    ft = torch.tensor(FREQUENCIES, dtype=torch.float64)
    t_carrier = t_taylor.TaylorF2()
    t_model = t_ppe.ParameterizedPostEinstein(t_carrier)
    t_base = t_carrier(f=ft, **kwargs)
    assert all(torch.all(abs(h) > 0) for h in t_base)
    alpha_t = torch.tensor(ALPHA, dtype=torch.float64, requires_grad=True)
    beta_t = torch.tensor(BETA, dtype=torch.float64, requires_grad=True)
    t_modified = t_model(ft, **kwargs, **(modified | dict(alpha_ppe=alpha_t, beta_ppe=beta_t)))
    t_ratios = tuple(h / h0 for h, h0 in zip(t_modified, t_base))

    fj = jnp.asarray(FREQUENCIES, dtype=jnp.float64)
    j_carrier = j_taylor.TaylorF2()
    j_model = j_ppe.ParameterizedPostEinstein(j_carrier)
    j_base = j_carrier(f=fj, **kwargs)
    assert all(np.all(np.abs(np.asarray(h)) > 0) for h in j_base)

    def j_ratios(alpha, beta):
        hp, hc = j_model(fj, **kwargs, **(modified | dict(alpha_ppe=alpha, beta_ppe=beta)))
        return hp / j_base[0], hc / j_base[1]

    j_ratio_values = jax.jit(j_ratios)(jnp.asarray(ALPHA), jnp.asarray(BETA))
    for ratios in (*t_ratios, *j_ratio_values):
        np.testing.assert_allclose(np.asarray(ratios.detach() if isinstance(ratios, torch.Tensor) else ratios),
                                   expected, rtol=5e-12, atol=5e-13)

    expected_derivatives = np.array([[expected_alpha.real, expected_alpha.imag],
                                     [expected_beta.real, expected_beta.imag]])
    for ratio in t_ratios:
        components = (ratio[2].real, ratio[2].imag)
        gradients = [torch.autograd.grad(part, (alpha_t, beta_t), retain_graph=True)
                     for part in components]
        actual = np.array([[gradients[0][0].item(), gradients[1][0].item()],
                           [gradients[0][1].item(), gradients[1][1].item()]])
        np.testing.assert_allclose(actual, expected_derivatives, rtol=1e-11, atol=1e-12)

    def j_components(alpha, beta):
        return jnp.stack([jnp.array([ratio[2].real, ratio[2].imag])
                          for ratio in j_ratios(alpha, beta)])

    j_dalpha, j_dbeta = jax.jit(jax.jacfwd(j_components, argnums=(0, 1)))(
        jnp.asarray(ALPHA), jnp.asarray(BETA)
    )
    for index in range(2):
        actual = np.array([np.asarray(j_dalpha[index]), np.asarray(j_dbeta[index])])
        np.testing.assert_allclose(actual, expected_derivatives, rtol=1e-11, atol=1e-12)

    t_gr = t_model(ft, **kwargs, alpha_ppe=0.0, beta_ppe=0.0, lambda_g=float("inf"))
    j_gr = jax.jit(lambda: j_model(fj, **kwargs, alpha_ppe=0.0,
                                  beta_ppe=0.0, lambda_g=float("inf")))()
    for deformed, base in (*zip(t_gr, t_base), *zip(j_gr, j_base)):
        quotient = deformed / base
        np.testing.assert_allclose(np.asarray(quotient), np.ones(len(FREQUENCIES)),
                                   rtol=0, atol=5e-13)

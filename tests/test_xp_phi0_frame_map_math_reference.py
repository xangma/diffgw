"""Appendix-C azimuth map, checked independently of the binary oracle."""

from __future__ import annotations

import mpmath
import numpy as np
import pytest

from diffgw_reference.xp_msa_generic_boundary import source_chart


def _rotate_spin(spin, angle):
    cosine, sine = mpmath.cos(angle), mpmath.sin(angle)
    x, y, z = spin
    return (cosine*x-sine*y, sine*x+cosine*y, z)


@pytest.mark.parametrize("azimuth", ("0.4", "-0.7", "6.683185307179586"))
def test_scalar_appendix_c_chart_is_covariant_under_source_z_rotation(azimuth):
    # Eq. (67): N has source azimuth pi/2-(phic-phi0). Rotating the whole
    # source frame by -phi0 gives a zero-phi0 chart with both spins rotated.
    with mpmath.workdps(90):
        phase, phi0 = mpmath.mpf("0.3"), mpmath.mpf(azimuth)
        spin = tuple(mpmath.mpf(value) for value in ("0.08", "-0.03", "0.1"))
        original = source_chart(
            spin, orbital_l="0.8", inclination="0.65",
            phi_ref=str(phase-phi0),
        )
        mapped = source_chart(
            _rotate_spin(spin, -phi0), orbital_l="0.8",
            inclination="0.65", phi_ref="0.3",
        )
        assert abs(mapped.theta_jn-original.theta_jn) < mpmath.mpf("1e-80")
        for left, right in ((mapped.alpha0, original.alpha0),
                            (mapped.zeta, original.zeta)):
            assert abs(mpmath.exp(1j*(left-right))-1) < mpmath.mpf("1e-80")
        assert abs(mpmath.exp(1j*(mapped.phi_j-original.phi_j+phi0))-1) < mpmath.mpf("1e-80")


@pytest.mark.parametrize("convention, azimuth", ((0, -0.7), (1, 0.4), (6, 0.4)))
def test_torch_jax_generic_waveform_obeys_paper_frame_map(convention, azimuth):
    torch = pytest.importorskip("torch")
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from torchwave.approximants.imrphenom_xp import IMRPhenomXP as TorchXP
    from jaxwave.approximants.imrphenom_xp import IMRPhenomXP as JAXXP

    params = dict(
        mass1=35.0, mass2=25.0,
        spin1x=0.2, spin1y=0.1, spin1z=0.3,
        spin2x=-0.1, spin2y=0.05, spin2z=-0.2,
        distance=150.0, inclination=0.7, phic=0.3, tc=0.0,
        f_ref=20.0, prec_version=223, convention=convention, final_spin_mod=3,
    )
    rotated = dict(params)
    for body in ("spin1", "spin2"):
        x, y = params[f"{body}x"], params[f"{body}y"]
        rotated[f"{body}x"] = np.cos(azimuth)*x+np.sin(azimuth)*y
        rotated[f"{body}y"] = -np.sin(azimuth)*x+np.cos(azimuth)*y
    frequencies = np.linspace(25.0, 100.0, 24)
    t_frequency = torch.as_tensor(frequencies, dtype=torch.float64)
    t_original = TorchXP()(t_frequency, **params, phi0=azimuth)
    t_mapped = TorchXP()(t_frequency, **rotated, phi0=0.0)
    j_original = JAXXP.evaluate(jnp.asarray(frequencies), **params, phi0=azimuth)
    j_mapped = JAXXP.evaluate(jnp.asarray(frequencies), **rotated, phi0=0.0)
    factor = np.exp(-2j*azimuth)
    for original, mapped in ((t_original, t_mapped), (j_original, j_mapped)):
        for actual, zero_azimuth in zip(original, mapped):
            actual, zero_azimuth = np.asarray(actual), np.asarray(zero_azimuth)
            assert np.linalg.norm(actual-factor*zero_azimuth)/np.linalg.norm(actual) < 5e-12
    for torch_value, jax_value in zip(t_original, j_original):
        left, right = np.asarray(torch_value), np.asarray(jax_value)
        assert np.linalg.norm(left-right)/np.linalg.norm(left) < 5e-12


def test_torch_jax_nonzero_azimuth_derivative():
    torch = pytest.importorskip("torch")
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from torchwave.approximants.imrphenom_xp import IMRPhenomXP as TorchXP
    from jaxwave.approximants.imrphenom_xp import IMRPhenomXP as JAXXP

    params = dict(
        mass1=35.0, mass2=25.0,
        spin1x=0.2, spin1y=0.1, spin1z=0.3,
        spin2x=-0.1, spin2y=0.05, spin2z=-0.2,
        distance=150.0, inclination=0.7, phic=0.3, tc=0.0,
        f_ref=20.0, prec_version=223, convention=1, final_spin_mod=3,
    )
    frequencies = np.linspace(30.0, 80.0, 12)
    t_phi = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
    t_plus, _ = TorchXP()(torch.as_tensor(frequencies), **params, phi0=t_phi)
    t_grad = torch.autograd.grad((t_plus[5].real*1e23), t_phi)[0].item()

    def j_observable(phi0):
        plus, _ = JAXXP.evaluate(jnp.asarray(frequencies), **params, phi0=phi0)
        return jnp.real(plus[5])*1e23

    j_grad = float(jax.grad(j_observable)(jnp.asarray(0.4)))
    epsilon = 1e-5
    finite = (float(j_observable(0.4+epsilon))-float(j_observable(0.4-epsilon)))/(2*epsilon)
    assert np.isfinite(t_grad) and np.isfinite(j_grad)
    assert np.isclose(t_grad, j_grad, rtol=1e-6, atol=1e-8)
    assert np.isclose(j_grad, finite, rtol=1e-6, atol=1e-8)

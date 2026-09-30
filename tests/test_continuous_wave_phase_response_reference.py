"""Paper-literal JKS limits against the two native CW carriers."""

import math

import jax
import jax.numpy as jnp
import mpmath as mp
import numpy as np
import pytest
import torch

from diffgw_reference import continuous_wave_phase_response as ref
from jaxwave.approximants import continuous_wave as j_cw
from torchwave.approximants import continuous_wave as t_cw

jax.config.update("jax_enable_x64", True)


def test_jks_ideal_beam_and_2f_projection():
    """An analytic equatorial detector checks signs and polarization rotation."""
    with mp.workdps(60):
        for psi, fp_expected, fx_expected in ((0, 1, 0), (mp.pi / 4, 0, -1)):
            fp, fx = ref.jks_beam_patterns(0, 0, psi, 0, mp.pi / 4, mp.pi / 2, 0)
            assert mp.almosteq(fp, fp_expected, abs_eps=mp.mpf("1e-58"))
            assert mp.almosteq(fx, fx_expected, abs_eps=mp.mpf("1e-58"))
        # At theta=pi/2, the paper's f branch vanishes and the 2f term remains.
        assert mp.almosteq(ref.jks_2f_strain(2, 0, 0, 1, 0), 1)
        assert mp.almosteq(ref.jks_2f_strain(2, 1, mp.pi / 2, 0, -1), -2)


@pytest.mark.parametrize("ra,dec,psi,gmst", [
    (0.0, 0.0, 0.0, 0.0),
    (1.1, -0.3, 0.27, 2.4565038851564722),
    (2.7, 0.6, -0.4, 4.2),
])
def test_jks_arm_tensor_reproduces_closed_beam_formula(ra, dec, psi, gmst):
    """PDF Eqs. (1)-(6) and (10)-(13) agree under Table 1 bisector angles."""
    with mp.workdps(60):
        lat = mp.radians(mp.mpf("46.45"))
        west_lon = mp.radians(mp.mpf("119.41"))
        gamma = mp.radians(mp.mpf("171.8"))
        lst = mp.mpf(str(gmst)) - west_lon
        lon = -west_lon
        north = (-mp.sin(lat) * mp.cos(lon), -mp.sin(lat) * mp.sin(lon), mp.cos(lat))
        east = (-mp.sin(lon), mp.cos(lon), mp.mpf("0"))
        # Table 1 gamma runs east to the arm bisector; the ordered arms are
        # gamma +/- zeta/2, with x cross y pointing outward.
        x_angle, y_angle = gamma - mp.pi / 4, gamma + mp.pi / 4
        x_arm = tuple(mp.sin(x_angle) * n + mp.cos(x_angle) * e for n, e in zip(north, east))
        y_arm = tuple(mp.sin(y_angle) * n + mp.cos(y_angle) * e for n, e in zip(north, east))
        from_arms = ref.jks_arm_tensor_beam_patterns(ra, dec, psi, gmst, x_arm, y_arm)
        closed = ref.jks_beam_patterns(ra, dec, psi, lat, gamma, mp.pi / 2, lst)
        for actual, expected in zip(from_arms, closed):
            assert mp.almosteq(actual, expected, abs_eps=mp.mpf("1e-57"))


@pytest.mark.parametrize("dt", [0.0, 1234.0, 86400.0])
def test_monochromatic_orbit_phase_matches_paper_given_zero_orbit_phase_epoch(dt):
    """The carrier's declared circular orbit has a paper-literal f1=f2=0 limit."""
    t0, ra, dec, f0, phi0 = 1126259462.0, 1.1, -0.3, 30.0, 0.37
    with mp.workdps(60):
        tau = ref.jks_circular_roemer_delay(
            ra, dec, orbit_phase=t_cw.OMEGA_ORBIT * dt,
            orbital_radius_m=t_cw.AU_SI, obliquity=t_cw.OBLIQUITY_RAD,
        )
        expected = float(ref.jks_2f_phase(dt, tau, f0, phi0=phi0))
    tt = torch.tensor([t0 + dt], dtype=torch.float64)
    jt = jnp.asarray([t0 + dt], dtype=jnp.float64)
    actual_t = t_cw.continuous_wave_phase(tt, ra, dec, f0, phi0=phi0, t0=t0)
    actual_j = j_cw.continuous_wave_phase(jt, ra, dec, f0, phi0=phi0, t0=t0)
    np.testing.assert_allclose([actual_t.item(), float(actual_j[0])], expected, rtol=0, atol=2e-9)


def test_monochromatic_frequency_gradient_against_paper_scalar():
    """Eq. (14) gives d phase / d GW frequency = 2 pi (dt + tau)."""
    t0, dt, ra, dec, f0 = 1126259462.0, 1234.0, 1.1, -0.3, 30.0
    with mp.workdps(60):
        tau = ref.jks_circular_roemer_delay(
            ra, dec, orbit_phase=t_cw.OMEGA_ORBIT * dt,
            orbital_radius_m=t_cw.AU_SI, obliquity=t_cw.OBLIQUITY_RAD,
        )
        expected = float(2 * mp.pi * (dt + tau))
    ft = torch.tensor(f0, dtype=torch.float64, requires_grad=True)
    tv = t_cw.continuous_wave_phase(torch.tensor(t0 + dt, dtype=torch.float64), ra, dec, ft, t0=t0)
    tg = torch.autograd.grad(tv, ft)[0]
    jg = jax.grad(lambda frequency: j_cw.continuous_wave_phase(
        jnp.asarray(t0 + dt), ra, dec, frequency, t0=t0))(jnp.asarray(f0))
    np.testing.assert_allclose([tg.item(), float(jg)], expected, rtol=1e-14, atol=0)


def test_spindown_residual_is_known_powers_of_roemer_delay():
    """Record the source mismatch algebraically; do not tune the phase to pass."""
    t0, dt, ra, dec = 1126259462.0, 43200.0, 1.1, -0.3
    f0, f1, f2 = 30.0, 2e-10, -1e-15
    with mp.workdps(60):
        tau = ref.jks_circular_roemer_delay(
            ra, dec, orbit_phase=t_cw.OMEGA_ORBIT * dt,
            orbital_radius_m=t_cw.AU_SI, obliquity=t_cw.OBLIQUITY_RAD,
        )
        paper = ref.jks_2f_phase(dt, tau, f0, f1, f2)
        residual = 2 * mp.pi * (f1 * tau**2 / 2 + f2 * dt * tau**2 / 2 + f2 * tau**3 / 6)
        expected_carrier = float(paper + residual)
        assert abs(float(residual)) > 1e-5
    tt = torch.tensor([t0 + dt], dtype=torch.float64)
    jt = jnp.asarray([t0 + dt], dtype=jnp.float64)
    actual_t = t_cw.continuous_wave_phase(tt, ra, dec, f0, f1, f2, t0=t0)
    actual_j = j_cw.continuous_wave_phase(jt, ra, dec, f0, f1, f2, t0=t0)
    np.testing.assert_allclose([actual_t.item(), float(actual_j[0])], expected_carrier, rtol=0, atol=2e-9)


@pytest.mark.xfail(strict=True, reason="JKS PDF Eq. 14/131 truncates tau² and tau³; native uses an untruncated barycentric polynomial")
def test_spindown_paper_phase_identity_gate():
    t0, dt, ra, dec = 1126259462.0, 43200.0, 1.1, -0.3
    with mp.workdps(60):
        tau = ref.jks_circular_roemer_delay(
            ra, dec, orbit_phase=t_cw.OMEGA_ORBIT * dt,
            orbital_radius_m=t_cw.AU_SI, obliquity=t_cw.OBLIQUITY_RAD,
        )
        paper = float(ref.jks_2f_phase(dt, tau, 30, 2e-10, -1e-15))
    actual = t_cw.continuous_wave_phase(
        torch.tensor([t0 + dt], dtype=torch.float64), ra, dec, 30,
        2e-10, -1e-15, t0=t0,
    )
    np.testing.assert_allclose(actual.item(), paper, rtol=0, atol=2e-9)


@pytest.mark.parametrize("module,detector,tensor", [
    (t_cw, "H1", lambda t: torch.tensor(t, dtype=torch.float64)),
    (j_cw, "H1", lambda t: jnp.asarray(t, dtype=jnp.float64)),
])
def test_2f_projection_conditional_on_native_detector_pattern(module, detector, tensor):
    """Independent source projection; antenna geometry and timing remain inputs."""
    t0, t = 1126259462.0, [1126259462.0, 1126259462.125]
    ra, dec, f0, h0, ci, psi, phi0 = 1.1, -0.3, 30.0, 2e-25, 0.4, 0.27, 0.13
    samples = tensor(t)
    det = module.Detector(detector)
    fp, fx = det.antenna_patterns(ra, dec, psi, t_gps=samples)
    phase = module.continuous_wave_phase(samples, ra, dec, f0, phi0=phi0, t0=t0, detector=detector)
    expected = np.array([
        float(ref.jks_2f_strain(h0, ci, ph, plus, cross))
        for ph, plus, cross in zip(np.asarray(phase), np.asarray(fp), np.asarray(fx))
    ])
    actual = module.continuous_wave_strain(
        samples, ra, dec, f0, h0=h0, cos_iota=ci, psi=psi, phi0=phi0,
        t0=t0, detector=detector,
    )
    # The native basis evaluates the large phase before inserting phi0; a
    # few ulps of phase rounding correspond to ~1e-37 strain here.
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-12, atol=5e-37)


@pytest.mark.parametrize("module,tensor", [
    (t_cw, lambda t: torch.tensor(t, dtype=torch.float64)),
    (j_cw, lambda t: jnp.asarray(t, dtype=jnp.float64)),
])
def test_h1_arms_and_tensor_response_against_published_geometry(module, tensor):
    """Anderson Appendix B Table I arms; JKS Eqs. (1)-(6) response algebra."""
    det = module.Detector("H1")
    # Anderson et al. gr-qc/0008066v1, Appendix B Table I prints four decimals.
    published_x = np.array([-0.2239, 0.7998, 0.5569])
    published_y = np.array([-0.9140, 0.0261, -0.4049])
    np.testing.assert_allclose(np.asarray(det.x_arm), published_x, rtol=0, atol=5e-5)
    np.testing.assert_allclose(np.asarray(det.y_arm), published_y, rtol=0, atol=5e-5)

    for t, ra, dec, psi in [
        (1126259462.0, 1.1, -0.3, 0.27),
        (1126302662.0, 2.7, 0.6, -0.4),
        (1126345862.0, 0.2, -0.8, 1.2),
    ]:
        gmst = float(module.greenwich_mean_sidereal_time(tensor(t)))
        with mp.workdps(60):
            expected = ref.jks_arm_tensor_beam_patterns(
                ra, dec, psi, gmst, np.asarray(det.x_arm), np.asarray(det.y_arm)
            )
        actual = det.antenna_patterns(ra, dec, psi, t_gps=tensor(t))
        np.testing.assert_allclose(
            np.asarray([float(x) for x in actual]),
            np.asarray([float(x) for x in expected]), rtol=0, atol=2e-15,
        )


@pytest.mark.xfail(strict=True, reason="JKS Table 1 inherits Allen 1996 LIGO arm angles corrected by Anderson 2001 Appendix B")
@pytest.mark.parametrize("module,tensor", [
    (t_cw, lambda t: torch.tensor(t, dtype=torch.float64)),
    (j_cw, lambda t: jnp.asarray(t, dtype=jnp.float64)),
])
def test_h1_beam_against_historical_jks_table_1_geometry_diagnostic(module, tensor):
    """Freeze the obsolete Allen-1996 site mismatch, conditioned on native GMST."""
    t, ra, dec, psi = 1126259462.0, 1.1, -0.3, 0.27
    gmst = float(module.greenwich_mean_sidereal_time(tensor(t)))
    local_sidereal = gmst - math.radians(119.41)  # west longitude in JKS Table 1
    expected = ref.jks_beam_patterns(
        ra, dec, psi, math.radians(46.45), math.radians(171.8),
        math.pi / 2, local_sidereal,
    )
    actual = module.Detector("H1").antenna_patterns(ra, dec, psi, t_gps=tensor(t))
    np.testing.assert_allclose(np.asarray([float(x) for x in actual]),
                               np.asarray([float(x) for x in expected]), rtol=0, atol=5e-4)

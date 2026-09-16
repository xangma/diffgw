"""Cross-framework parity tests: torchwave.detectors vs jaxwave.detectors.

Validates exact mathematical parity across:
1. Analytical PSDs (aLIGO, AdvVirgo, KAGRA, CosmicExplorer, EinsteinTelescope, get_psd)
2. GMST and Leap Seconds
3. Detector geometry, arm vectors, and response tensors (H1, L1, V1, K1)
4. Antenna patterns F+ and Fx, and geometric time delays
5. Detector network frequency-domain and time-domain projections
6. LISA constellation geometry, response tensors, TDI channels, and PSDs
7. PTA geometry, Hellings-Downs correlation, and log-likelihood
8. JAX JIT and gradient compatibility
"""

import math
import numpy as np
import pytest
import torch
import jax
import jax.numpy as jnp

# Torchwave detectors
import torchwave.detectors as td_torch
from torchwave.detectors.network import (
    Detector as TorchDetector,
    DetectorNetwork as TorchDetectorNetwork,
    greenwich_mean_sidereal_time as torch_gmst,
    gps_to_utc_seconds as torch_gps_to_utc,
)
from torchwave.detectors.lisa import (
    LISANetwork as TorchLISANetwork,
    lisa_psd as torch_lisa_psd,
)
from torchwave.detectors.pta import (
    PTANetwork as TorchPTANetwork,
    hellings_downs_correlation as torch_hd,
    stochastic_gwb_spectrum as torch_gwb,
    pulsar_line_of_sight as torch_los,
    angular_separation as torch_ang_sep,
)

# Jaxwave detectors
import jaxwave.detectors as td_jax
from jaxwave.detectors.network import (
    Detector as JaxDetector,
    DetectorNetwork as JaxDetectorNetwork,
    greenwich_mean_sidereal_time as jax_gmst,
    gps_to_utc_seconds as jax_gps_to_utc,
)
from jaxwave.detectors.lisa import (
    LISANetwork as JaxLISANetwork,
    lisa_psd as jax_lisa_psd,
)
from jaxwave.detectors.pta import (
    PTANetwork as JaxPTANetwork,
    hellings_downs_correlation as jax_hd,
    stochastic_gwb_spectrum as jax_gwb,
    pulsar_line_of_sight as jax_los,
    angular_separation as jax_ang_sep,
)


# ==============================================================================
# 1. Analytical PSDs Parity Tests
# ==============================================================================

@pytest.mark.parametrize("model", [
    "aLIGOZeroDetHighPower",
    "aLIGOEarlyHighPower",
    "AdvancedVirgo",
    "KAGRA",
    "EinsteinTelescope",
    "CosmicExplorer",
])
def test_psd_models_parity(model):
    """Verify exact parity between torchwave and jaxwave for all analytical PSDs."""
    torch_fn = getattr(td_torch, model)
    jax_fn = getattr(td_jax, model)

    f_np = np.logspace(0, 3.7, 500, dtype=np.float64)  # 1 Hz to ~5000 Hz
    f_t = torch.from_numpy(f_np)
    f_j = jnp.asarray(f_np)

    psd_t = torch_fn(f_t)
    psd_j = jax_fn(f_j)

    np.testing.assert_allclose(
        psd_t.numpy(),
        np.asarray(psd_j),
        rtol=1e-14,
        atol=1e-45,
    )


@pytest.mark.parametrize("name", [
    "aLIGO",
    "ligo",
    "aligodesign",
    "aligoearlyhighpower",
    "AdvancedVirgo",
    "virgo",
    "KAGRA",
    "EinsteinTelescope",
    "et",
    "etd",
    "CosmicExplorer",
    "ce",
    "ce40km",
])
def test_get_psd_parity(name):
    """Verify get_psd parity across detector name aliases and custom bounds."""
    f_np = np.linspace(5.0, 3000.0, 300, dtype=np.float64)
    f_t = torch.from_numpy(f_np)
    f_j = jnp.asarray(f_np)

    psd_t = td_torch.get_psd(name, f_t, f_low=15.0, f_high=2048.0)
    psd_j = td_jax.get_psd(name, f_j, f_low=15.0, f_high=2048.0)

    np.testing.assert_allclose(
        psd_t.numpy(),
        np.asarray(psd_j),
        rtol=1e-14,
        atol=1e-45,
    )


# ==============================================================================
# 2. GMST and Leap Seconds Parity
# ==============================================================================

@pytest.mark.parametrize("gps", [
    0.0,
    46828800.25,
    46828801.0,
    630763213.0,
    1024012816.5,
    1025136015.25,
    1025136016.0,
    1126259462.4,
    1167264016.0,
    1167264017.25,
    1167264018.0,
    1472688018.123456,
])
def test_gmst_and_leap_parity(gps):
    """Verify exact parity of GMST and UTC conversion across GPS epochs."""
    utc_t = torch_gps_to_utc(torch.tensor(gps, dtype=torch.float64))
    utc_j = jax_gps_to_utc(jnp.array(gps, dtype=jnp.float64))
    np.testing.assert_allclose(utc_t.numpy(), np.asarray(utc_j), rtol=0, atol=0)

    gmst_t = torch_gmst(torch.tensor(gps, dtype=torch.float64))
    gmst_j = jax_gmst(jnp.array(gps, dtype=jnp.float64))
    np.testing.assert_allclose(gmst_t.numpy(), np.asarray(gmst_j), rtol=1e-15, atol=1e-15)


# ==============================================================================
# 3. Detector Geometry and Response Tensor Parity
# ==============================================================================

@pytest.mark.parametrize("prefix", ["H1", "L1", "V1", "K1"])
def test_detector_geometry_parity(prefix):
    """Verify site coordinates, arm vectors, and response tensors match between frameworks."""
    det_t = TorchDetector(prefix)
    det_j = JaxDetector(prefix)

    np.testing.assert_allclose(det_t.vertex.numpy(), np.asarray(det_j.vertex), rtol=1e-15, atol=1e-15)
    np.testing.assert_allclose(det_t.arm_x.numpy(), np.asarray(det_j.arm_x), rtol=1e-15, atol=1e-15)
    np.testing.assert_allclose(det_t.arm_y.numpy(), np.asarray(det_j.arm_y), rtol=1e-15, atol=1e-15)
    np.testing.assert_allclose(det_t.response.numpy(), np.asarray(det_j.response), rtol=1e-15, atol=1e-15)


# ==============================================================================
# 4. Antenna Patterns and Time Delays Parity
# ==============================================================================

@pytest.mark.parametrize("prefix", ["H1", "L1", "V1", "K1"])
@pytest.mark.parametrize("ra,dec,psi", [
    (1.95, -1.27, 0.82),
    (0.0, 0.0, 0.0),
    (3.5, 0.7, 1.2),
    (5.8, -0.4, 2.5),
])
def test_antenna_patterns_and_time_delays_scalar(prefix, ra, dec, psi):
    """Verify antenna factors and time delays for scalar angles and times."""
    t_gps = 1126259462.4
    det_t = TorchDetector(prefix)
    det_j = JaxDetector(prefix)

    fp_t, fc_t = det_t.antenna_patterns(ra, dec, psi, t_gps)
    fp_j, fc_j = det_j.antenna_patterns(ra, dec, psi, t_gps)

    np.testing.assert_allclose(fp_t.numpy(), np.asarray(fp_j), rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(fc_t.numpy(), np.asarray(fc_j), rtol=1e-14, atol=1e-15)

    dt_t = det_t.time_delay_from_earth_center(ra, dec, t_gps)
    dt_j = det_j.time_delay_from_earth_center(ra, dec, t_gps)
    np.testing.assert_allclose(dt_t.numpy(), np.asarray(dt_j), rtol=1e-14, atol=1e-15)


def test_antenna_patterns_vectorized_parity():
    """Verify batched/vectorized sky positions antenna patterns and time delays."""
    ra_np = np.linspace(0.0, 2.0 * math.pi, 25, dtype=np.float64)
    dec_np = np.linspace(-math.pi / 2.0 + 0.05, math.pi / 2.0 - 0.05, 25, dtype=np.float64)
    psi_np = np.linspace(0.0, math.pi, 25, dtype=np.float64)
    t_gps_np = np.linspace(1126259462.0, 1167264018.0, 25, dtype=np.float64)

    ra_t, dec_t = torch.from_numpy(ra_np), torch.from_numpy(dec_np)
    psi_t, t_t = torch.from_numpy(psi_np), torch.from_numpy(t_gps_np)

    ra_j, dec_j = jnp.asarray(ra_np), jnp.asarray(dec_np)
    psi_j, t_j = jnp.asarray(psi_np), jnp.asarray(t_gps_np)

    for prefix in ["H1", "L1", "V1", "K1"]:
        det_t = TorchDetector(prefix)
        det_j = JaxDetector(prefix)

        fp_t, fc_t = det_t.antenna_patterns(ra_t, dec_t, psi_t, t_t)
        fp_j, fc_j = det_j.antenna_patterns(ra_j, dec_j, psi_j, t_j)

        np.testing.assert_allclose(fp_t.numpy(), np.asarray(fp_j), rtol=1e-13, atol=1e-14)
        np.testing.assert_allclose(fc_t.numpy(), np.asarray(fc_j), rtol=1e-13, atol=1e-14)

        dt_t = det_t.time_delay_from_earth_center(ra_t, dec_t, t_t)
        dt_j = det_j.time_delay_from_earth_center(ra_j, dec_j, t_j)
        np.testing.assert_allclose(dt_t.numpy(), np.asarray(dt_j), rtol=1e-13, atol=1e-14)


# ==============================================================================
# 5. Frequency-Domain and Time-Domain Projection Parity
# ==============================================================================

def test_fd_strain_projection_parity():
    """Verify frequency-domain strain projection for single detector and network."""
    f_np = np.linspace(20.0, 500.0, 150, dtype=np.float64)
    hp_np = np.sin(f_np / 10.0) + 1j * np.cos(f_np / 15.0)
    hc_np = 0.5 * np.cos(f_np / 10.0) - 0.7j * np.sin(f_np / 20.0)

    f_t, hp_t, hc_t = torch.from_numpy(f_np), torch.from_numpy(hp_np), torch.from_numpy(hc_np)
    f_j, hp_j, hc_j = jnp.asarray(f_np), jnp.asarray(hp_np), jnp.asarray(hc_np)

    ra, dec, psi, t_gps = 1.3, -0.4, 0.7, 1126259462.4

    # Single detector (H1)
    h1_t = TorchDetector("H1")
    h1_j = JaxDetector("H1")

    proj_t1 = h1_t.project_fd(f_t, hp_t, hc_t, ra, dec, psi, t_gps)
    proj_j1 = h1_j.project_fd(f_j, hp_j, hc_j, ra, dec, psi, t_gps)
    np.testing.assert_allclose(proj_t1.numpy(), np.asarray(proj_j1), rtol=1e-13, atol=1e-15)

    # Alternate argument order: (hp, hc, f, ...)
    proj_t2 = h1_t.project_fd(hp_t, hc_t, f_t, ra, dec, psi, t_gps)
    proj_j2 = h1_j.project_fd(hp_j, hc_j, f_j, ra, dec, psi, t_gps)
    np.testing.assert_allclose(proj_t2.numpy(), np.asarray(proj_j2), rtol=1e-13, atol=1e-15)

    # Full network
    net_t = TorchDetectorNetwork(["H1", "L1", "V1", "K1"])
    net_j = JaxDetectorNetwork(["H1", "L1", "V1", "K1"])

    strains_t = net_t.project_fd(f_t, hp_t, hc_t, ra, dec, psi, t_gps)
    strains_j = net_j.project_fd(f_j, hp_j, hc_j, ra, dec, psi, t_gps)

    assert set(strains_t.keys()) == set(strains_j.keys())
    for key in strains_t:
        np.testing.assert_allclose(
            strains_t[key].numpy(),
            np.asarray(strains_j[key]),
            rtol=1e-13,
            atol=1e-15,
        )


def test_fd_strain_projection_batched_sky_parity():
    """Verify frequency projection broadcasts sky batches over shared frequency grid."""
    f_np = np.linspace(20.0, 180.0, 20, dtype=np.float64)
    hp_np = np.cos(f_np / 12.0) + 1j * np.sin(f_np / 17.0)
    hc_np = 0.3 * hp_np

    ra_np = np.array([0.2, 1.0, 2.8], dtype=np.float64)
    dec_np = np.array([-0.7, 0.2, 0.9], dtype=np.float64)
    times_np = np.array([1126259462.4, 1167264017.25, 1167264018.25], dtype=np.float64)
    psi = 0.4

    f_t, hp_t, hc_t = torch.from_numpy(f_np), torch.from_numpy(hp_np), torch.from_numpy(hc_np)
    ra_t, dec_t, times_t = torch.from_numpy(ra_np), torch.from_numpy(dec_np), torch.from_numpy(times_np)

    f_j, hp_j, hc_j = jnp.asarray(f_np), jnp.asarray(hp_np), jnp.asarray(hc_np)
    ra_j, dec_j, times_j = jnp.asarray(ra_np), jnp.asarray(dec_np), jnp.asarray(times_np)

    det_t = TorchDetector("L1")
    det_j = JaxDetector("L1")

    actual_t = det_t.project_fd(f_t, hp_t, hc_t, ra_t, dec_t, psi, times_t)
    actual_j = det_j.project_fd(f_j, hp_j, hc_j, ra_j, dec_j, psi, times_j)

    assert actual_j.shape == (3, 20)
    np.testing.assert_allclose(actual_t.numpy(), np.asarray(actual_j), rtol=1e-13, atol=1e-15)


def test_td_strain_projection_parity():
    """Verify time-domain strain projection for single detector."""
    dt = 1.0 / 2048.0
    times_np = np.arange(0.0, 0.1, dt, dtype=np.float64)
    hp_np = np.sin(2.0 * math.pi * 100.0 * times_np)
    hc_np = np.cos(2.0 * math.pi * 100.0 * times_np)

    times_t, hp_t, hc_t = torch.from_numpy(times_np), torch.from_numpy(hp_np), torch.from_numpy(hc_np)
    times_j, hp_j, hc_j = jnp.asarray(times_np), jnp.asarray(hp_np), jnp.asarray(hc_np)

    ra, dec, psi, t_gps = 1.2, -0.4, 0.7, 1126259462.4

    det_t = TorchDetector("H1")
    det_j = JaxDetector("H1")

    res_t = det_t.project_td(times_t, hp_t, hc_t, ra, dec, psi, t_gps, delta_t=dt)
    res_j = det_j.project_td(times_j, hp_j, hc_j, ra, dec, psi, t_gps, delta_t=dt)

    np.testing.assert_allclose(res_t.numpy(), np.asarray(res_j), rtol=1e-14, atol=1e-15)

    # Also test legacy calling convention: project_td(hp, hc, dt, ...)
    res_t2 = det_t.project_td(hp_t, hc_t, dt, ra, dec, psi, t_gps)
    res_j2 = det_j.project_td(hp_j, hc_j, dt, ra, dec, psi, t_gps)
    np.testing.assert_allclose(res_t2.numpy(), np.asarray(res_j2), rtol=1e-14, atol=1e-15)


# ==============================================================================
# 6. LISA Suite Parity
# ==============================================================================

@pytest.mark.parametrize("channel", ["A", "E", "T", "X", "sky_averaged"])
@pytest.mark.parametrize("include_conf", [False, True])
def test_lisa_psd_parity(channel, include_conf):
    """Verify LISA analytical PSD parity across all TDI channels."""
    f_np = np.logspace(-4, 0, 100, dtype=np.float64)
    f_t = torch.from_numpy(f_np)
    f_j = jnp.asarray(f_np)

    psd_t = torch_lisa_psd(f_t, channel=channel, include_galactic_confusion=include_conf)
    psd_j = jax_lisa_psd(f_j, channel=channel, include_galactic_confusion=include_conf)

    np.testing.assert_allclose(psd_t.numpy(), np.asarray(psd_j), rtol=1e-13, atol=1e-45)


def test_lisa_detector_tensors_parity():
    """Verify LISA TDI response tensors in the ecliptic frame."""
    t_t = TorchLISANetwork.detector_tensors()
    t_j = JaxLISANetwork.detector_tensors()

    for ch in ("X", "Y", "Z", "A", "E", "T"):
        np.testing.assert_allclose(t_t[ch].numpy(), np.asarray(t_j[ch]), rtol=1e-14, atol=1e-15)


@pytest.mark.parametrize("channel", ["A", "E", "T"])
@pytest.mark.parametrize("th,ph,ps", [
    (1.2, 2.4, 0.6),
    (0.5, 1.0, 0.0),
    (math.pi / 3.0, math.pi / 4.0, 0.2),
])
def test_lisa_antenna_patterns_parity(channel, th, ph, ps):
    """Verify LISA antenna patterns for A, E, T channels."""
    lisa_t = TorchLISANetwork()
    lisa_j = JaxLISANetwork()

    fp_t, fc_t = lisa_t.antenna_patterns(th, ph, ps, channel=channel)
    fp_j, fc_j = lisa_j.antenna_patterns(th, ph, ps, channel=channel)

    np.testing.assert_allclose(fp_t.numpy(), np.asarray(fp_j), rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(fc_t.numpy(), np.asarray(fc_j), rtol=1e-14, atol=1e-15)


def test_lisa_project_fd_parity():
    """Verify LISA frequency-domain projection for TDI channels."""
    f_np = np.linspace(1e-4, 5e-2, 100, dtype=np.float64)
    hp_np = np.sin(f_np * 100.0) + 1j * np.cos(f_np * 50.0)
    hc_np = 0.8 * np.cos(f_np * 100.0) - 0.4j * np.sin(f_np * 50.0)

    f_t, hp_t, hc_t = torch.from_numpy(f_np), torch.from_numpy(hp_np), torch.from_numpy(hc_np)
    f_j, hp_j, hc_j = jnp.asarray(f_np), jnp.asarray(hp_np), jnp.asarray(hc_np)

    th, ph, ps = 1.0, 2.0, 0.5

    lisa_t = TorchLISANetwork()
    lisa_j = JaxLISANetwork()

    for ch in ("A", "E"):
        h_t = lisa_t.project_fd_strain(hp_t, hc_t, f_t, th, ph, ps, channel=ch)
        h_j = lisa_j.project_fd_strain(hp_j, hc_j, f_j, th, ph, ps, channel=ch)
        np.testing.assert_allclose(h_t.numpy(), np.asarray(h_j), rtol=1e-13, atol=1e-15)


# ==============================================================================
# 7. PTA Suite Parity
# ==============================================================================

def test_pta_functions_parity():
    """Verify line of sight, angular separation, and Hellings-Downs correlation."""
    ra_np = np.array([0.13, 1.21, 4.51], dtype=np.float64)
    dec_np = np.array([0.08, -0.82, 0.13], dtype=np.float64)

    los_t = torch_los(torch.from_numpy(ra_np), torch.from_numpy(dec_np))
    los_j = jax_los(jnp.asarray(ra_np), jnp.asarray(dec_np))
    np.testing.assert_allclose(los_t.numpy(), np.asarray(los_j), rtol=1e-14, atol=1e-15)

    zeta_np = np.linspace(0.0, math.pi, 50, dtype=np.float64)
    zeta_t = torch.from_numpy(zeta_np)
    zeta_j = jnp.asarray(zeta_np)

    hd_cross_t = torch_hd(zeta_t, is_auto=False)
    hd_cross_j = jax_hd(zeta_j, is_auto=False)
    np.testing.assert_allclose(hd_cross_t.numpy(), np.asarray(hd_cross_j), rtol=1e-14, atol=1e-15)

    hd_auto_t = torch_hd(zeta_t, is_auto=True)
    hd_auto_j = jax_hd(zeta_j, is_auto=True)
    np.testing.assert_allclose(hd_auto_t.numpy(), np.asarray(hd_auto_j), rtol=1e-14, atol=1e-15)

    # GWB spectrum
    f_np = np.array([1e-8, 3.16875e-8, 1e-7], dtype=np.float64)
    s_t = torch_gwb(torch.from_numpy(f_np), a_gwb=2.0e-15)
    s_j = jax_gwb(jnp.asarray(f_np), a_gwb=2.0e-15)
    np.testing.assert_allclose(s_t.numpy(), np.asarray(s_j), rtol=1e-14, atol=1e-35)


def test_pta_network_covariance_and_likelihood_parity():
    """Verify PTANetwork covariance matrix and log-likelihood match PyTorch."""
    names = ["J0030+0451", "J0437-4715", "J1713+0747", "J1744-1134", "J1909-3744"]
    ra = [0.13, 1.21, 4.51, 4.64, 5.01]
    dec = [0.08, -0.82, 0.13, -0.20, -0.65]
    noise_std = [1e-7, 8e-8, 5e-8, 1.2e-7, 6e-8]

    pta_t = TorchPTANetwork(names, ra, dec, noise_std=noise_std)
    pta_j = JaxPTANetwork(names, ra, dec, noise_std=noise_std)

    a_gwb = 2.5e-15
    gamma = 13.0 / 3.0

    cov_t = pta_t.covariance_matrix(a_gwb, gamma=gamma)
    cov_j = pta_j.covariance_matrix(a_gwb, gamma=gamma)
    np.testing.assert_allclose(cov_t.numpy(), np.asarray(cov_j), rtol=1e-14, atol=1e-30)

    # Residuals
    np.random.seed(42)
    res_np = np.random.randn(5).astype(np.float64) * 1e-7
    res_t = torch.from_numpy(res_np)
    res_j = jnp.asarray(res_np)

    ll_t = pta_t.log_likelihood(res_t, a_gwb=a_gwb, gamma=gamma)
    ll_j = pta_j.log_likelihood(res_j, a_gwb=a_gwb, gamma=gamma)
    np.testing.assert_allclose(ll_t.numpy(), np.asarray(ll_j), rtol=1e-13, atol=1e-14)


# ==============================================================================
# 8. JAX JIT and Gradient Verification
# ==============================================================================

def test_jax_jit_compilation():
    """Verify JIT compatibility of detector functions and network operations."""
    det = JaxDetector("H1")

    @jax.jit
    def jit_antenna(ra, dec, psi, t_gps):
        return det.antenna_patterns(ra, dec, psi, t_gps)

    @jax.jit
    def jit_time_delay(ra, dec, t_gps):
        return det.time_delay_from_earth_center(ra, dec, t_gps)

    fp, fc = jit_antenna(1.5, -0.5, 0.3, 1126259462.0)
    dt = jit_time_delay(1.5, -0.5, 1126259462.0)

    assert jnp.isfinite(fp)
    assert jnp.isfinite(fc)
    assert jnp.isfinite(dt)

    # JIT project_fd_strain
    f_grid = jnp.linspace(20.0, 200.0, 50, dtype=jnp.float64)
    hp = jnp.ones_like(f_grid, dtype=jnp.complex128)
    hc = 0.5j * hp

    @jax.jit
    def jit_proj(ra, dec, psi, t_gps):
        return det.project_fd(f_grid, hp, hc, ra, dec, psi, t_gps)

    s = jit_proj(1.5, -0.5, 0.3, 1126259462.0)
    assert s.shape == f_grid.shape
    assert jnp.all(jnp.isfinite(s))


def test_jax_gradient_parity():
    """Verify exact parity between PyTorch autograd and JAX grad for antenna patterns and delay."""
    ra_val, dec_val, psi_val, t_val = 1.5, -0.5, 0.3, 1126259462.0

    # PyTorch gradients
    ra_t = torch.tensor(ra_val, dtype=torch.float64, requires_grad=True)
    dec_t = torch.tensor(dec_val, dtype=torch.float64, requires_grad=True)
    psi_t = torch.tensor(psi_val, dtype=torch.float64, requires_grad=True)
    t_t = torch.tensor(t_val, dtype=torch.float64, requires_grad=True)

    det_t = TorchDetector("H1")
    fp_t, fc_t = det_t.antenna_patterns(ra_t, dec_t, psi_t, t_t)
    loss_t = fp_t * fp_t + 2.0 * fc_t * fc_t
    loss_t.backward()

    # JAX gradients
    det_j = JaxDetector("H1")

    def loss_fn(ra, dec, psi, t_gps):
        fp, fc = det_j.antenna_patterns(ra, dec, psi, t_gps)
        return fp * fp + 2.0 * fc * fc

    grad_fn = jax.grad(loss_fn, argnums=(0, 1, 2, 3))
    g_ra, g_dec, g_psi, g_t = grad_fn(ra_val, dec_val, psi_val, t_val)

    np.testing.assert_allclose(ra_t.grad.item(), float(g_ra), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(dec_t.grad.item(), float(g_dec), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(psi_t.grad.item(), float(g_psi), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(t_t.grad.item(), float(g_t), rtol=1e-12, atol=1e-14)

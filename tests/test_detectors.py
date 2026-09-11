"""Tests for clean-room differentiable detector responses and detector networks.

Literature citations:
- Anderson et al. (2001), Phys. Rev. D 63, 042003 (arXiv:gr-qc/0008066)
- Schutz (2011), Class. Quant. Grav. 28, 125023 (arXiv:1102.5421)
- Allen et al. (2012), Phys. Rev. D 85, 122006 (arXiv:gr-qc/0509116)
"""

import math
import pytest
import torch

from torchwave.detectors import (
    Detector,
    DetectorNetwork,
    greenwich_mean_sidereal_time,
)
from torchwave.waveforms import get_fd_waveform, get_td_waveform


def test_detector_initialization():
    """Verify built-in detectors initialize with correct unit orthogonal arms."""
    for prefix in ["H1", "L1", "V1", "K1"]:
        det = Detector(prefix)
        assert det.prefix == prefix
        # Arms must be unit vectors
        assert torch.isclose(torch.linalg.norm(det.arm_x), torch.tensor(1.0, dtype=torch.float64))
        assert torch.isclose(torch.linalg.norm(det.arm_y), torch.tensor(1.0, dtype=torch.float64))
        # Arms must be orthogonal (within ~1e-4 on Earth geoid)
        dot = torch.dot(det.arm_x, det.arm_y)
        assert torch.abs(dot) < 1e-4


def test_antenna_patterns_bounded():
    """Verify antenna factors satisfy F_+^2 + F_x^2 <= 1 everywhere."""
    network = DetectorNetwork(["H1", "L1", "V1", "K1"])
    ra = torch.linspace(0.0, 2.0 * math.pi, 20, dtype=torch.float64)
    dec = torch.linspace(-math.pi / 2.0 + 0.05, math.pi / 2.0 - 0.05, 20, dtype=torch.float64)
    psi = torch.linspace(0.0, math.pi, 10, dtype=torch.float64)
    t_gps = 1126259462.0

    for det in network.detectors.values():
        for r in ra[:5]:
            for d in dec[:5]:
                for p in psi[:5]:
                    fp, fc = det.antenna_patterns(r, d, p, t_gps)
                    power = fp ** 2.0 + fc ** 2.0
                    assert power.item() <= 1.0000001, f"Antenna power {power} > 1 for {det.name}"


def test_geometric_time_delays_bounded():
    """Verify geometric time delays are bounded by Earth radius / c ~ 0.0425 s."""
    max_delay_earth = 2.0 * 6378137.0 / 299792458.0  # ~ 0.04255 s

    network = DetectorNetwork(["H1", "L1", "V1", "K1"])
    ra = torch.linspace(0.0, 2.0 * math.pi, 15, dtype=torch.float64)
    dec = torch.linspace(-math.pi / 2.0 + 0.1, math.pi / 2.0 - 0.1, 15, dtype=torch.float64)
    t_gps = 1126259462.0

    for det in network.detectors.values():
        for r in ra:
            for d in dec:
                dt = det.time_delay_from_earth_center(r, d, t_gps)
                assert abs(dt.item()) <= max_delay_earth

    # Relative time delays between detector pairs
    h1 = network.detectors["H1"]
    l1 = network.detectors["L1"]
    for r in ra:
        for d in dec:
            dt_h1 = h1.time_delay_from_earth_center(r, d, t_gps)
            dt_l1 = l1.time_delay_from_earth_center(r, d, t_gps)
            # Baseline H1-L1 separation distance ~ 3000 km -> max light travel time ~ 10 ms
            dt_rel = abs((dt_h1 - dt_l1).item())
            assert dt_rel <= 0.015, f"H1-L1 time delay {dt_rel} > 15 ms"


def test_antenna_patterns_pycbc_lal_parity():
    """Verify exact numerical parity of antenna factors and time delays against PyCBC/LAL."""
    try:
        from pycbc.detector import Detector as PyCBCDetector
    except ImportError:
        pytest.skip("PyCBC not installed")

    t_gps = 1126259462.4  # GW150914 trigger time
    ra_val = 1.95
    dec_val = -1.27
    psi_val = 0.82

    try:
        import lal
        HAS_LAL = True
    except ImportError:
        HAS_LAL = False

    for prefix in ["H1", "L1", "V1"]:
        pycbc_det = PyCBCDetector(prefix)
        tw_det = Detector(prefix)

        # Antenna factors
        fp_pycbc, fc_pycbc = pycbc_det.antenna_pattern(ra_val, dec_val, psi_val, t_gps)
        fp_tw, fc_tw = tw_det.antenna_patterns(ra_val, dec_val, psi_val, t_gps)

        if HAS_LAL:
            lal_idx = {
                "H1": lal.LALDetectorIndexLHODIFF,
                "L1": lal.LALDetectorIndexLLODIFF,
                "V1": lal.LALDetectorIndexVIRGODIFF,
            }[prefix]
            ref_det = lal.CachedDetectors[lal_idx]
            gmst_lal = lal.GreenwichMeanSiderealTime(lal.LIGOTimeGPS(t_gps))
            rfp, rfc = lal.ComputeDetAMResponse(ref_det.response, ra_val, dec_val, psi_val, gmst_lal)
            assert math.isclose(fp_tw.item(), rfp, abs_tol=1e-6), f"{prefix} LAL F+ mismatch"
            assert math.isclose(fc_tw.item(), rfc, abs_tol=1e-6), f"{prefix} LAL Fx mismatch"

        assert math.isclose(fp_tw.item(), fp_pycbc, abs_tol=3e-5), f"{prefix} PyCBC F+ mismatch"
        assert math.isclose(fc_tw.item(), fc_pycbc, abs_tol=3e-5), f"{prefix} PyCBC Fx mismatch"

        # Time delay from Earth center
        dt_pycbc = pycbc_det.time_delay_from_earth_center(ra_val, dec_val, t_gps)
        dt_tw = tw_det.time_delay_from_earth_center(ra_val, dec_val, t_gps)
        assert math.isclose(dt_tw.item(), dt_pycbc, abs_tol=1e-6), f"{prefix} time delay mismatch"


def test_detector_network_projections():
    """Verify detector network projection of FD and TD polarizations."""
    network = DetectorNetwork(["H1", "L1", "V1"])
    ra = 1.5
    dec = -0.5
    psi = 0.3
    t_gps = 1126259462.0

    # FD projection
    f = torch.linspace(20.0, 300.0, 100, dtype=torch.float64)
    hp_fd, hc_fd = get_fd_waveform(
        approximant="IMRPhenomD",
        mass1=30.0,
        mass2=25.0,
        sample_frequencies=f,
    )

    strains_fd = network.project_fd(hp_fd, hc_fd, f, ra, dec, psi, t_gps)
    assert set(strains_fd.keys()) == {"H1", "L1", "V1"}
    for name, s in strains_fd.items():
        assert s.shape == f.shape
        assert torch.all(torch.isfinite(s))
        assert torch.any(s.abs() > 0.0)

    # TD projection
    hp_td, hc_td = get_td_waveform(
        approximant="IMRPhenomD",
        mass1=30.0,
        mass2=25.0,
        delta_t=1.0 / 2048.0,
        f_lower=25.0,
    )
    strains_td = network.project_td(hp_td, hc_td, 1.0 / 2048.0, ra, dec, psi, t_gps)
    assert set(strains_td.keys()) == {"H1", "L1", "V1"}
    for name, s in strains_td.items():
        assert s.shape == hp_td.shape
        assert torch.all(torch.isfinite(s))


def test_detector_autograd_gradients():
    """Verify end-to-end autograd differentiability w.r.t. sky coordinates and time."""
    ra = torch.tensor(1.5, dtype=torch.float64, requires_grad=True)
    dec = torch.tensor(-0.5, dtype=torch.float64, requires_grad=True)
    psi = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    t_gps = torch.tensor(1126259462.0, dtype=torch.float64, requires_grad=True)

    f = torch.linspace(25.0, 200.0, 50, dtype=torch.float64)
    hp, hc = get_fd_waveform(
        approximant="IMRPhenomD",
        mass1=30.0,
        mass2=25.0,
        sample_frequencies=f,
    )

    h1 = Detector("H1")
    strain = h1.project_fd(hp, hc, f, ra, dec, psi, t_gps)

    loss = (strain.abs() ** 2.0).sum()
    loss.backward()

    for p, name in [(ra, "ra"), (dec, "dec"), (psi, "psi"), (t_gps, "t_gps")]:
        assert p.grad is not None, f"Gradient for {name} is None"
        assert torch.isfinite(p.grad), f"Gradient for {name} is not finite: {p.grad}"
        assert p.grad.item() != 0.0, f"Gradient for {name} is zero"

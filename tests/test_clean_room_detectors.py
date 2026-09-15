"""Independent time/geometry checks; no waveform implementation or binary oracle.

References: Aoki et al. (1982), Eqs. (13),(19); IERS Bulletin C 72;
Anderson et al., gr-qc/0008066v1, Appendix B; published detector surveys.
"""

from datetime import datetime, timezone
from decimal import Decimal, localcontext
import math

import pytest
import torch

from torchwave.constants import C_SI
from torchwave.detectors import Detector, DetectorNetwork, greenwich_mean_sidereal_time
from torchwave.detectors.network import GPS_LEAP_SECONDS, gps_to_utc_seconds


DTYPE = torch.float64
ROUNDOFF = 8 * torch.finfo(DTYPE).eps
GPS_EPOCH = datetime(1980, 1, 6, tzinfo=timezone.utc)
# UTC effective dates and TAI-UTC from IERS Leap_Second.dat (Bulletin C 72).
IERS_TRANSITIONS = (
    (1981, 7, 1, 20), (1982, 7, 1, 21), (1983, 7, 1, 22),
    (1985, 7, 1, 23), (1988, 1, 1, 24), (1990, 1, 1, 25),
    (1991, 1, 1, 26), (1992, 7, 1, 27), (1993, 7, 1, 28),
    (1994, 7, 1, 29), (1996, 1, 1, 30), (1997, 7, 1, 31),
    (1999, 1, 1, 32), (2006, 1, 1, 33), (2009, 1, 1, 34),
    (2012, 7, 1, 35), (2015, 7, 1, 36), (2017, 1, 1, 37),
)


def _calendar_boundaries():
    return [
        (int((datetime(y, m, d, tzinfo=timezone.utc) - GPS_EPOCH).total_seconds()), tai - 19)
        for y, m, d, tai in IERS_TRANSITIONS
    ]


def _aoki_decimal(gps):
    """50-digit evaluation of the printed equations, independent calendar dates."""
    offset = 0
    for calendar, new_offset in _calendar_boundaries():
        if gps >= calendar + new_offset:
            offset = new_offset
    with localcontext() as ctx:
        ctx.prec = 50
        utc = Decimal.from_float(float(gps)) - offset
        days, seconds = divmod(utc, Decimal(86400))
        centuries = (days - Decimal("7300.5")) / 36525
        angle = (Decimal("24110.54841") + Decimal("8640184.812866") * centuries
                 + Decimal("0.093104") * centuries**2 - Decimal("0.0000062") * centuries**3)
        rate = (Decimal("1.002737909350795") + Decimal("5.9006e-11") * centuries
                - Decimal("5.9e-15") * centuries**2)
        angle = (angle + seconds * rate) % 86400
        if angle < 0:
            angle += 86400
        return float(angle / 86400 * Decimal.from_float(2 * math.pi)), float(rate * Decimal.from_float(2 * math.pi) / 86400)


def test_leap_table_uses_iers_effective_midnight_and_preserves_inserted_second():
    expected = tuple((float(calendar + offset), offset) for calendar, offset in _calendar_boundaries())
    assert GPS_LEAP_SECONDS == expected
    for calendar, offset in _calendar_boundaries():
        boundary = calendar + offset
        gps = torch.tensor([boundary - 2, boundary - 1, boundary, boundary + 1])
        utc = gps_to_utc_seconds(gps)
        assert utc.dtype == DTYPE
        torch.testing.assert_close(utc, torch.tensor([calendar - 1, calendar, calendar, calendar + 1], dtype=DTYPE), rtol=0, atol=0)


@pytest.mark.parametrize("gps", [0.0, 46828800.25, 46828801.0, 630763213.0,
                                      1024012816.5, 1025136015.25, 1025136016.0,
                                      1126259462.4, 1167264016.0, 1167264017.25,
                                      1167264018.0, 1472688018.123456])
def test_sidereal_angle_matches_high_precision_aoki_equations(gps):
    expected, _ = _aoki_decimal(gps)
    actual = greenwich_mean_sidereal_time(gps)
    assert actual.dtype == DTYPE
    assert abs(actual.item() - expected) < 5e-14


def test_sidereal_time_resolves_microseconds_and_has_published_rate():
    gps = torch.tensor(1126259462.0, dtype=DTYPE, requires_grad=True)
    later = gps + 2**-18
    start = greenwich_mean_sidereal_time(gps)
    stop = greenwich_mean_sidereal_time(later)
    _, rate = _aoki_decimal(gps.item())
    assert abs((stop - start).item() - rate * (later - gps).item()) < 2e-15
    derivative, = torch.autograd.grad(start, gps)
    assert math.isclose(derivative.item(), rate, rel_tol=2e-15)
    # Integer epochs must not silently route through float32 arithmetic.
    torch.testing.assert_close(greenwich_mean_sidereal_time(gps.detach().long()), start.detach(), rtol=0, atol=0)


def test_float32_detector_preserves_full_precision_epoch_until_angle_conversion():
    gps = 1126259462.4
    double = Detector("H1")
    single = Detector("H1").float()
    for ra, dec, psi in [(1.2, -0.7, 0.8), (3.0, 0.3, 1.9)]:
        actual = single.antenna_patterns(ra, dec, psi, gps)
        expected = double.antenna_patterns(ra, dec, psi, gps)
        for a, e in zip(actual, expected):
            assert a.dtype == torch.float32
            torch.testing.assert_close(a.double(), e, rtol=0, atol=3e-7)
        delay = single.time_delay_from_earth_center(ra, dec, gps)
        torch.testing.assert_close(delay.double(), double.time_delay_from_earth_center(ra, dec, gps), rtol=0, atol=3e-9)


@pytest.mark.parametrize("prefix", ["H1", "L1", "V1", "K1"])
def test_normalized_survey_arms_give_trace_free_michelson_tensor(prefix):
    detector = Detector(prefix)
    assert abs(torch.linalg.vector_norm(detector.arm_x).item() - 1) < 3e-16
    assert abs(torch.linalg.vector_norm(detector.arm_y).item() - 1) < 3e-16
    assert abs(detector.response.trace().item()) < 2e-16
    # Unit-arm Michelson tensor norm also holds for a non-right arm angle.
    angle_cos = torch.dot(detector.arm_x, detector.arm_y)
    torch.testing.assert_close(torch.sum(detector.response**2), (1 - angle_cos**2) / 2, rtol=0, atol=ROUNDOFF)


def test_detector_site_inputs_agree_with_printed_survey_precision():
    # LIGO T980044-08 Tables 4 and 18; Anderson Appendix B Table I (Virgo);
    # KAGRA JGW-T1605295-v1 Tables 1, 4 and 5.  Tolerances reflect the printed
    # survey digits, not a more precise external implementation.
    surveys = {
        "H1": ((-2161414.9, -3834695.2, 4600350.2), (-.223892, .799831, .556905), (-.913978, .0260945, -.404923), .051, 1e-6),
        "L1": ((-74276.044, -5496283.721, 3224257.018), (-.954574, -.14158, -.262189), (.297741, -.48791, -.820545), .005, 1e-6),
        "V1": ((4546374., 842990., 4378577.), (-.7005, .2085, .6826), (-.0538, -.9691, .2408), .51, 5e-5),
        "K1": ((-3777336.024, 3484898.411, 3765313.697), (-.3759040, -.8361583, .3994189), (.7164378, .01114076, .6975620), .003, 1e-7),
    }
    for prefix, (vertex, arm_x, arm_y, vertex_tol, arm_tol) in surveys.items():
        detector = Detector(prefix)
        torch.testing.assert_close(detector.vertex, torch.tensor(vertex, dtype=DTYPE), rtol=0, atol=vertex_tol)
        torch.testing.assert_close(detector.arm_x, torch.tensor(arm_x, dtype=DTYPE), rtol=0, atol=arm_tol)
        torch.testing.assert_close(detector.arm_y, torch.tensor(arm_y, dtype=DTYPE), rtol=0, atol=arm_tol)


def test_ideal_xy_detector_matches_anderson_closed_form():
    detector = Detector("H1")
    detector.response.copy_(torch.diag(torch.tensor([0.5, -0.5, 0.0], dtype=DTYPE)))
    phi = torch.tensor([0.0, 0.2, 1.4, 2.8], dtype=DTYPE)
    theta = torch.tensor([0.0, 0.8, 1.4, math.pi], dtype=DTYPE)
    psi = torch.tensor([0.0, 0.7, 1.9, 2.1], dtype=DTYPE)
    gps = 1126259462.4
    ra = greenwich_mean_sidereal_time(gps) + phi
    fp, fc = detector.antenna_patterns(ra, math.pi / 2 - theta, psi, gps)
    a = 0.5 * (1 + torch.cos(theta)**2) * torch.cos(2 * phi)
    b = torch.cos(theta) * torch.sin(2 * phi)
    torch.testing.assert_close(fp, -a * torch.cos(2 * psi) - b * torch.sin(2 * psi), rtol=0, atol=8e-16)
    torch.testing.assert_close(fc, a * torch.sin(2 * psi) - b * torch.cos(2 * psi), rtol=0, atol=8e-16)


def test_polarization_rotation_and_geometric_delay_extrema():
    detector = Detector("K1")
    gps = 1167264018.25
    fp, fc = detector.antenna_patterns(1.2, -0.4, 0.7, gps)
    delta = 0.36
    rotated_p, rotated_c = detector.antenna_patterns(1.2, -0.4, 0.7 + delta, gps)
    torch.testing.assert_close(rotated_p, fp * math.cos(2 * delta) + fc * math.sin(2 * delta), rtol=0, atol=ROUNDOFF)
    torch.testing.assert_close(rotated_c, fc * math.cos(2 * delta) - fp * math.sin(2 * delta), rtol=0, atol=ROUNDOFF)
    radius = torch.linalg.vector_norm(detector.vertex)
    ra = greenwich_mean_sidereal_time(gps) + torch.atan2(detector.vertex[1], detector.vertex[0])
    dec = torch.asin(detector.vertex[2] / radius)
    torch.testing.assert_close(detector.time_delay_from_earth_center(ra, dec, gps), -radius / C_SI, rtol=0, atol=1e-17)
    torch.testing.assert_close(detector.time_delay_from_earth_center(ra + math.pi, -dec, gps), radius / C_SI, rtol=0, atol=1e-17)


@pytest.mark.parametrize("shared_waveform", [False, True])
def test_frequency_projection_broadcasts_sky_batches_over_shared_grid(shared_waveform):
    detector = Detector("L1")
    frequencies = torch.linspace(20, 180, 17, dtype=DTYPE)
    base = torch.polar(1 + frequencies / 100, frequencies / 13)
    hp = base if shared_waveform else torch.stack((base, 2 * base, 3 * base))
    hc = (0.3 + 0.8j) * hp
    ra = torch.tensor([0.2, 1.0, 2.8], dtype=DTYPE, requires_grad=True)
    dec = torch.tensor([-0.7, 0.2, 0.9], dtype=DTYPE)
    psi = 0.4
    times = torch.tensor([1126259462.4, 1167264017.25, 1167264018.25], dtype=DTYPE)
    actual = detector.project_fd(frequencies, hp, hc, ra, dec, psi, times)
    expected = torch.stack([
        detector.project_fd(frequencies, hp if shared_waveform else hp[i], hc if shared_waveform else hc[i], ra[i], dec[i], psi, times[i])
        for i in range(3)
    ])
    assert actual.shape == (3, 17)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    derivative, = torch.autograd.grad(actual.real.sum(), ra)
    assert torch.all(torch.isfinite(derivative))
    assert torch.all(derivative != 0)
    network = DetectorNetwork([detector])
    torch.testing.assert_close(network.project_fd(hp, hc, frequencies, ra, dec, psi, times)["L1"], expected, rtol=0, atol=0)


def test_projection_gradients_and_earth_rotation_generator():
    detector = Detector("V1")
    ra, dec, psi, gps = [torch.tensor(value, dtype=DTYPE, requires_grad=True)
                         for value in [1.3, -0.4, 0.7, 1126259462.4]]
    frequencies = torch.linspace(20, 200, 17, dtype=DTYPE)
    hp = torch.polar(torch.ones_like(frequencies), frequencies / 71)
    hc = (0.2 + 0.4j) * hp
    # Elliptical polarization and a phase-sensitive loss avoid exact physical
    # invariants such as circular-polarization power being independent of psi.
    projected = detector.project_fd(frequencies, hp, hc, ra, dec, psi, gps)
    gradients = torch.autograd.grad(projected.real.sum(), (ra, dec, psi, gps))
    assert all(torch.isfinite(g) and g != 0 for g in gradients)
    _, omega = _aoki_decimal(gps.item())
    torch.testing.assert_close(gradients[3], -omega * gradients[0], rtol=2e-14, atol=0)
    fp, fc = detector.antenna_patterns(ra, dec, psi, gps)
    invariant_gradient, = torch.autograd.grad(fp**2 + fc**2, psi)
    assert abs(invariant_gradient.item()) < ROUNDOFF


def test_time_projection_keeps_one_sample_waveform_overload_unambiguous():
    detector = Detector("H1")
    times = torch.tensor([0.0], dtype=DTYPE)
    hp = torch.tensor([1.25], dtype=DTYPE)
    hc = torch.tensor([-0.75], dtype=DTYPE)
    ra, dec, psi, gps = 1.2, -0.4, 0.7, 1126259462.4

    actual = detector.project_td(times, hp, hc, ra, dec, psi, gps, delta_t=0.1)
    fp, fc = detector.antenna_patterns(ra, dec, psi, gps)
    expected = fp * hp + fc * hc
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

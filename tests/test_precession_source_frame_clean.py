import torch

from torchwave.approximants.precession_source_frame_clean import source_frame_geometry


def test_geometry_is_finite_and_orthogonal_observer_basis():
    out = source_frame_geometry(40.125, 19.875, .3, .2, .4, -.2, .1, -.2,
                                .7, .0, 40.)
    assert all(torch.isfinite(x).all() for x in out)
    assert all(x.dtype == torch.float64 for x in out)


def test_aligned_limit_has_zero_opening_and_float64_batch_safe_scalars():
    alpha, theta, zeta, phi_j = source_frame_geometry(
        40., 20., 0., 0., .4, 0., 0., -.2, .7, .0, 40.)
    assert torch.isfinite(alpha + theta + zeta + phi_j)
    assert 0.0 <= float(theta) <= 3.141592653589793


def test_aligned_continuous_limit_tracks_observer_and_azimuth_gauge():
    iota, phi_ref = 0.7, 0.3
    alpha, theta, zeta, phi_j = source_frame_geometry(
        40., 20., 0., 0., .4, 0., 0., -.2, iota, phi_ref, 40.)
    assert abs(float(theta) - iota) < 1e-14
    assert abs(float(phi_j) - (3.141592653589793 / 2 - phi_ref)) < 1e-14
    assert all(torch.isfinite(x) for x in (alpha, theta, zeta, phi_j))

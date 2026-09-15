"""Source-level checks for the clean-room nonspinning TEOB radiation stage."""

import math

import torch

from torchwave.approximants.eob.teob_radiation_clean import (
    TEOBRadiationClean,
    _residual_phase,
    _residual_phase_taylor,
    non_keplerian_radius,
    non_keplerian_velocity,
    teob22_mode,
)


def _inputs(*, dtype=torch.float64, device="cpu", r=10.0, unequal=True):
    r = torch.tensor(r, dtype=dtype, device=device)
    omega = r.pow(-1.5)
    masses = (0.6, 0.4) if unequal else (0.5, 0.5)
    return dict(
        r=r,
        Omega=omega,
        phi=torch.tensor(0.17, dtype=dtype, device=device),
        prstar=torch.tensor(-0.03, dtype=dtype, device=device),
        pphi=torch.tensor(3.0, dtype=dtype, device=device),
        Heff=torch.tensor(1.0, dtype=dtype, device=device),
        H=torch.tensor(1.0, dtype=dtype, device=device),
        Hcirc=torch.tensor(1.0, dtype=dtype, device=device),
        Aprime=2 / r.square(),
        m1=torch.tensor(masses[0], dtype=dtype, device=device),
        m2=torch.tensor(masses[1], dtype=dtype, device=device),
    )


def test_nonspinning_radius_and_circular_kepler_identity():
    args = _inputs(r=10.0)
    r_omega = non_keplerian_radius(args["r"], args["Hcirc"], args["Aprime"])
    vphi = non_keplerian_velocity(args["r"], args["Omega"], args["Hcirc"], args["Aprime"])
    assert torch.allclose(r_omega, args["r"], rtol=1e-13, atol=0)
    assert torch.allclose(vphi, args["r"] * args["Omega"], rtol=1e-13, atol=0)
    assert torch.allclose(vphi.square() * r_omega, torch.ones_like(r_omega), rtol=1e-13, atol=0)


def test_plunge_velocity_uses_hcirc_and_is_not_keplerian():
    args = _inputs(r=8.0)
    args["Hcirc"] = torch.tensor(1.12, dtype=args["r"].dtype)
    r_omega = non_keplerian_radius(args["r"], args["Hcirc"], args["Aprime"])
    vphi = non_keplerian_velocity(args["r"], args["Omega"], args["Hcirc"], args["Aprime"])
    assert torch.allclose(r_omega, args["r"] * args["Hcirc"].pow(2 / 3), rtol=1e-13, atol=0)
    assert not torch.allclose(vphi, args["r"] * args["Omega"], rtol=1e-6, atol=1e-8)


def test_weak_field_22_amplitude_and_flux_normalization():
    args = _inputs(r=1.0e6, unequal=False)
    h22 = teob22_mode(**args)
    nu = 0.25
    expected_amplitude = 8 * math.sqrt(math.pi / 5) * nu / args["r"]
    # Tail and residual phase approach one and zero, respectively, as r grows.
    assert torch.isclose(h22.abs(), torch.as_tensor(expected_amplitude, dtype=h22.real.dtype),
                         rtol=3e-6, atol=0)
    leading_flux = (32 / 5) * nu**2 * args["Omega"] ** (10 / 3)
    flux22 = (2 * args["Omega"])**2 * h22.abs().square() / (8 * math.pi)
    assert torch.isclose(flux22, leading_flux, rtol=4e-6, atol=0)
    radiation = TEOBRadiationClean(2)
    jdot22 = radiation.flux(**args)
    edot22 = radiation.energy_flux(**args)
    assert torch.isclose(jdot22, 4 * args["Omega"] * h22.abs().square() / (8 * math.pi),
                         rtol=1e-12, atol=0)
    assert torch.isclose(edot22, flux22, rtol=1e-12, atol=0)
    assert torch.isclose(edot22, args["Omega"] * jdot22, rtol=1e-12, atol=0)


def test_updated_phase_equation_and_pade_choice():
    y = torch.tensor(0.04, dtype=torch.float64)
    nu = torch.tensor(0.24, dtype=torch.float64)
    v = torch.sqrt(y)
    expected = (7 / 3 * v**3 *
                (808920 * nu * math.pi * v + 137388 * math.pi**2 * v**2
                 + 35 * nu**2 * (136080 + (154975 - 1359276 * nu) * v**2)) /
                (808920 * nu * math.pi * v + 137388 * math.pi**2 * v**2
                 + 35 * nu**2 * (136080 + (154975 + 40404 * nu) * v**2)))
    assert torch.allclose(_residual_phase((2, 2), y, nu), expected, rtol=1e-14, atol=0)

    # The explicitly retained test-mass terms are used at nu=0.
    nu0 = torch.zeros_like(y)
    expected0 = 7 / 3 * v**3 + 428 * math.pi / 105 * v**6 + (1712 * math.pi**2 / 315 - 2203 / 81) * v**9
    assert torch.allclose(_residual_phase((2, 2), y, nu0), expected0, rtol=1e-14, atol=0)


def test_finite_mass_324244_phases_omit_test_mass_v9_terms():
    y = torch.tensor(0.04, dtype=torch.float64)
    nu = torch.tensor(0.24, dtype=torch.float64)
    nu0 = torch.zeros_like(nu)
    v9 = torch.sqrt(y)**9
    highest = {
        (3, 2): 208 / 63 * math.pi**2 - 9112 / 405,
        (4, 2): 25136 / 10395 * math.pi**2 - 6893 / 375,
        (4, 4): 201088 / 10395 * math.pi**2 - 55144 / 375,
    }
    for mode, coefficient in highest.items():
        finite_mass = _residual_phase(mode, y, nu)
        full_taylor = _residual_phase_taylor(mode, y, nu)
        assert torch.allclose(
            finite_mass, full_taylor - coefficient * v9, rtol=1e-14, atol=0
        )
        # The printed highest terms remain on the exact test-mass branch.
        assert torch.allclose(
            _residual_phase(mode, y, nu0), _residual_phase_taylor(mode, y, nu0),
            rtol=1e-14, atol=0,
        )


def test_all_35_modes_symmetry_and_linear_l5_odd_mass_polynomial():
    args = _inputs(unequal=True)
    radiation = TEOBRadiationClean(8)
    modes = radiation.modes(**args)
    assert len(modes) == 35
    assert all(value.is_complex() and torch.isfinite(value).all() for value in modes.values())
    for ell, m in ((2, 2), (3, 3), (5, 1)):
        positive = radiation.mode(ell, m, **args)
        negative = radiation.mode(ell, -m, **args)
        assert torch.allclose(negative, (-1)**ell * positive.conj(), rtol=1e-13, atol=0)

    # c_(5,odd) = X2**4-X1**4 = (X2-X1)*(1-2*nu): the default has one
    # mass-difference factor and vanishes linearly at equal mass.
    default = radiation.mode(5, 1, **args)
    equal_mass = radiation.mode(5, 1, **_inputs(unequal=False))
    assert torch.equal(equal_mass, torch.zeros_like(equal_mass))
    for delta in (1e-4, 2e-4):
        near_equal = _inputs(unequal=False)
        near_equal["m1"] = torch.tensor(0.5 + delta / 2, dtype=torch.float64)
        near_equal["m2"] = torch.tensor(0.5 - delta / 2, dtype=torch.float64)
        value = radiation.mode(5, 1, **near_equal).abs()
        if delta == 1e-4:
            first = value / delta
        else:
            second = value / delta
    assert torch.allclose(first, second, rtol=2e-4, atol=0)

    exchanged = radiation.mode(5, 1, **dict(args, m1=args["m2"], m2=args["m1"]))
    assert torch.allclose(exchanged, -default, rtol=1e-13, atol=0)

    # Keep the pre-contract extra sqrt(X_AB) only as an explicit legacy
    # control for comparisons with old serialized runs.
    legacy = TEOBRadiationClean(5, legacy_l5_odd=True).amplitudes(**args)[(5, 1)]
    expected_legacy = default.abs() * torch.sqrt(1 - 4 * args["m1"] * args["m2"])
    assert torch.allclose(legacy, expected_legacy, rtol=1e-13, atol=0)


def test_dtype_device_autograd_and_nqc22():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = _inputs(dtype=torch.float32, device=device, r=12.0)
    args["r"].requires_grad_(True)
    mode = teob22_mode(**args, nqc22=torch.tensor(1.2, dtype=torch.float32, device=device))
    assert mode.dtype == torch.complex64
    mode.real.square().sum().backward()
    assert args["r"].grad is not None and torch.isfinite(args["r"].grad).all()


def _horizon_inputs(*, x=1e-2, unequal=True, dtype=torch.float64):
    args = _inputs(dtype=dtype, r=1.0, unequal=unequal)
    args["Omega"] = torch.sqrt(torch.tensor(x, dtype=dtype))
    return args


def test_horizon_flux_leading_powers_sources_and_total_normalization():
    radiation = TEOBRadiationClean(2)
    args = _horizon_inputs(x=1e-4)
    # Hold S_odd=p_phi/(rOmega*vphi) finite while taking x to zero so that
    # the displayed x**4 and x**5 powers can be separated from source scaling.
    args["pphi"] = args["Omega"].clone()
    nu = args["m1"] * args["m2"]
    scale = (32 / 5) * nu**2 * args["r"]**4 * args["Omega"]**5
    ratio = radiation.horizon_flux(**args) / scale
    leading = (1 - 4 * nu + 2 * nu**2) * args["Omega"]**8 * args["Heff"]**2
    # Eq. (44) starts with x**4 S_eff^(0)**2; x=vphi**2=Omega**2 here.
    assert torch.allclose(ratio / leading, torch.ones_like(ratio), rtol=2e-3, atol=0)

    # The odd source enters only as S_odd**2*x**5.  Holding the kinematics
    # fixed makes its quadratic scaling independently testable.
    base = radiation.horizon_flux(**dict(args, pphi=torch.tensor(0.0, dtype=args["pphi"].dtype)))
    one = radiation.horizon_flux(**dict(args, pphi=torch.tensor(1.0, dtype=args["pphi"].dtype)))
    two = radiation.horizon_flux(**dict(args, pphi=torch.tensor(2.0, dtype=args["pphi"].dtype)))
    assert torch.allclose(two - base, 4 * (one - base), rtol=1e-13, atol=0)

    infinity = radiation.flux(**args)
    horizon = radiation.horizon_flux(**args)
    assert torch.allclose(
        radiation.flux(**args, include_horizon=True), infinity + horizon,
        rtol=1e-13, atol=0,
    )
    assert torch.allclose(
        radiation.total_flux(**args), infinity + horizon, rtol=1e-13, atol=0,
    )
    torque = radiation.radiation_reaction_torque(**args)
    assert torch.allclose(torque, -(infinity + horizon) / nu, rtol=1e-13, atol=0)
    radial, integrated_torque = radiation.radiation_reaction_forces(**args)
    assert torch.equal(radial, torch.zeros_like(radial))
    assert torch.allclose(integrated_torque, torque, rtol=1e-13, atol=0)


def test_horizon_flux_exchange_lmax_and_gradient_contracts():
    args = _horizon_inputs(x=0.02)
    radiation2 = TEOBRadiationClean(2)
    radiation8 = TEOBRadiationClean(8)
    swapped = dict(args, m1=args["m2"], m2=args["m1"])
    assert torch.allclose(radiation2.horizon_flux(**args), radiation2.horizon_flux(**swapped),
                          rtol=1e-13, atol=0)
    assert torch.allclose(radiation2.horizon_flux(**args), radiation8.horizon_flux(**args),
                          rtol=1e-13, atol=0)

    omega = args["Omega"].clone().requires_grad_(True)
    differentiated = radiation2.horizon_flux(**dict(args, Omega=omega))
    gradient = torch.autograd.grad(differentiated, omega)[0]
    assert torch.isfinite(gradient).all() and gradient > 0

    args32 = _horizon_inputs(x=0.02, dtype=torch.float32)
    flux32 = radiation2.horizon_flux(**args32)
    flux64 = radiation2.horizon_flux(**args)
    assert torch.allclose(flux32.double(), flux64, rtol=2e-6, atol=0)

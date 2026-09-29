"""Gaussian transform and gauge proofs; no native waveform or NR oracle."""

from math import gcd

import pytest

mp = pytest.importorskip("mpmath")
sp = pytest.importorskip("sympy")

from diffgw_reference import imrphenomhm_spa_conventions as ref


@pytest.mark.parametrize("m", (1, 2, 3, 4))
def test_symbolic_conjugation_completion_and_stationary_point(m):
    t, f, phi0, omega = sp.symbols("t f phi0 omega", real=True)
    k, epsilon, amplitude = sp.symbols("k epsilon A", positive=True)
    coefficient = sp.Symbol("C", complex=True)
    phi = phi0 + omega * t + k * t**2 / 2
    td = coefficient * amplitude * sp.exp(-sp.I * m * phi)
    integrand = sp.conjugate(td) * sp.exp(-2 * sp.pi * sp.I * f * t)
    expected = sp.conjugate(coefficient) * amplitude * sp.exp(
        sp.I * (m * phi - 2 * sp.pi * f * t))
    assert sp.simplify(integrand - expected) == 0
    a = epsilon - sp.I * m * k
    delta = m * omega - 2 * sp.pi * f
    # Complete the square independently before applying the Gaussian identity.
    exponent = -epsilon * t**2 / 2 + sp.I * (m * phi - 2 * sp.pi * f * t)
    completed = -a * (t - sp.I * delta / a)**2 / 2 - delta**2 / (2 * a) + sp.I * m * phi0
    assert sp.simplify(exponent - completed) == 0
    exact = sp.conjugate(coefficient) * amplitude * sp.sqrt(2 * sp.pi / a) * sp.exp(
        completed.subs(t, sp.I * delta / a))
    assert sp.simplify(exact - ref.symbolic_gaussian_carrier(
        m, f, phi0, omega, k, epsilon, coefficient, amplitude)) == 0
    stationary = sp.solve(sp.diff(m * phi - 2 * sp.pi * f * t, t), t)[0]
    assert sp.simplify(stationary - (2 * sp.pi * f / m - omega) / k) == 0
    assert sp.diff(m * phi, t, 2) == m * k
    phase = (m * phi - 2 * sp.pi * f * t).subs(t, stationary) + sp.pi / 4
    assert sp.simplify(phase - ref.symbolic_spa_phase(m, f, phi0, omega, k)) == 0


@pytest.mark.parametrize("m", (1, 2, 3, 4))
def test_symbolic_primitive_cancels_dynamics_orbit_and_time(m):
    f, phi0, omega, k, alpha, tau, gamma, theta, theta22 = sp.symbols(
        "f phi0 omega k alpha tau gamma theta theta22", real=True)
    a, b = ref.primitive_powers(m)
    assert a * m == 2 * b and gcd(a, b) == 1
    phase = ref.symbolic_spa_phase(m, f, phi0, omega, k) - theta
    phase22 = ref.symbolic_spa_phase(2, 2 * f / m, phi0, omega, k) - theta22
    angle = a * (phase + m * alpha - 2 * sp.pi * f * tau + gamma)
    angle -= b * (phase22 + 2 * alpha - 2 * sp.pi * (2 * f / m) * tau + gamma)
    assert sp.simplify(angle - ref.symbolic_relative_angle(m, theta, theta22, gamma)) == 0
    assert sp.diff(sp.expand(angle), gamma) == sp.Rational(2 - m, gcd(2, m))


@pytest.mark.parametrize("m", (1, 2, 3, 4))
def test_exact_gaussian_against_direct_td_quadrature(m):
    with mp.workdps(75):
        f, phi0, omega, k, epsilon, amplitude = map(
            mp.mpf, (".23", ".41", ".38", ".17", "1.2", "1.7"))
        coefficient = mp.mpc("-.7", ".9")

        def integrand(t):
            td = coefficient * amplitude * mp.exp(
                -mp.j * m * (phi0 + omega * t + k * t**2 / 2))
            return mp.conj(td) * mp.exp(-epsilon * t**2 / 2 - 2 * mp.pi * mp.j * f * t)

        numerical = mp.quad(integrand, [-20, -12, -6, 0, 6, 12, 20])
        exact = ref.gaussian_carrier(m, f, phi0, omega, k, epsilon, coefficient, amplitude)
        # Absolute tail < 2 |C| A exp(-epsilon*20^2/2)/(epsilon*20).
        tail = 2 * abs(coefficient) * amplitude * mp.exp(-epsilon * 200) / (epsilon * 20)
        assert tail < mp.mpf("1e-100")
        assert abs(numerical - exact) < mp.mpf("1e-55")


@pytest.mark.parametrize("m", (1, 2, 3, 4))
def test_zero_damping_sequence_converges_to_spa(m):
    with mp.workdps(70):
        phi0, omega, k, tstar = map(mp.mpf, (".41", ".38", ".7", ".4"))
        f = m * (omega + k * tstar) / (2 * mp.pi)
        coefficient = mp.mpc("-.7", ".9")
        limit = ref.spa_carrier(m, f, phi0, omega, k, coefficient)
        errors = []
        for power in (2, 4, 6, 8, 10):
            epsilon = mp.mpf(10)**(-power)
            exact = ref.gaussian_carrier(m, f, phi0, omega, k, epsilon, coefficient)
            error = abs(exact / limit - 1)
            assert error < epsilon
            errors.append(error)
        assert all(later < earlier / 90 for earlier, later in zip(errors, errors[1:]))
        assert errors[-1] < mp.mpf("1e-10")


@pytest.mark.parametrize("m", (1, 2, 3, 4))
def test_spa_relative_phase_and_common_strain_response(m):
    with mp.workdps(65):
        f, phi0, omega, k, alpha, tau, gamma = map(
            mp.mpf, (".37", ".41", ".38", ".7", ".53", ".29", ".61"))
        coefficient, coefficient22 = mp.mpc("-.7", ".9"), mp.mpc(".3", "-.8")
        mode = ref.spa_carrier(m, f, phi0, omega, k, coefficient)
        carrier = ref.spa_carrier(2, 2 * f / m, phi0, omega, k, coefficient22)
        expected = ref.spa_relative_phase(coefficient, coefficient22, m)
        assert abs(ref.primitive_relative_phase(mode, carrier, m) - expected) < mp.mpf("1e-60")
        transformed = ref.primitive_relative_phase(
            mode * mp.exp(mp.j * (m * alpha - 2 * mp.pi * f * tau + gamma)),
            carrier * mp.exp(mp.j * (2 * alpha - 2 * mp.pi * 2 * f / m * tau + gamma)), m)
        a, b = ref.primitive_powers(m)
        assert abs(transformed - expected * mp.exp(mp.j * (a - b) * gamma)) < mp.mpf("1e-60")


def test_sign_observability_and_odd_mode_orbital_ambiguity():
    with mp.workdps(65):
        mode, carrier = mp.mpc(".7", ".3"), mp.mpc("-.4", ".9")
        for m in (1, 3):
            invariant = ref.primitive_relative_phase(mode, carrier, m)
            assert abs(ref.primitive_relative_phase(-mode, carrier, m) - invariant) < mp.mpf("1e-60")
            assert abs(ref.primitive_relative_phase(mp.j * mode, carrier, m) + invariant) < mp.mpf("1e-60")
        invariant44 = ref.primitive_relative_phase(mode, carrier, 4)
        flipped44 = ref.primitive_relative_phase(-mode, carrier, 4)
        assert abs(flipped44 + invariant44) < mp.mpf("1e-60")
        # Squaring the primitive recovers the old doubled invariant: blind to 44 sign.
        assert abs(flipped44**2 - invariant44**2) < mp.mpf("1e-60")
        for m in (1, 2, 3, 4):
            assert abs(mp.exp(mp.j * m * mp.pi) - (-1)**m) < mp.mpf("1e-60")


@pytest.mark.parametrize("common_td_sign", (1, -1))
def test_source_phases_and_london_22_preserving_difference(common_td_sign):
    with mp.workdps(65):
        modes = ((2, 2), (2, 1), (3, 3), (3, 2), (4, 4), (4, 3))
        expected_phases = (1, mp.j, -mp.j, 1, -1, -mp.j)
        c22 = ref.leading_td_coefficient_phase(2, 2, common_td_sign=common_td_sign)
        phase22 = -mp.arg(c22) + mp.pi / 4
        for (ell, m), expected_c in zip(modes, expected_phases):
            coefficient = ref.leading_td_coefficient_phase(ell, m, common_td_sign=common_td_sign)
            assert coefficient == common_td_sign * expected_c
            source_angle = -mp.arg(coefficient) + mp.pi / 4 - mp.mpf(m) / 2 * phase22
            # Native exp(-i*Delta22)=exp(-2*pi*i) is a trivial complex phasor.
            delta = mp.pi * (3 * ell + (ell + m) % 2 - 2) / 2
            native_angle = -delta
            correction = ref.london_orbital_phase_correction(m, common_td_sign=common_td_sign)
            assert abs(mp.exp(mp.j * source_angle) - mp.exp(mp.j * (native_angle + correction))) < mp.mpf("1e-60")
        assert ref.london_orbital_phase_correction(2, common_td_sign=common_td_sign) == 0
        expected_invariant = mp.exp(mp.j * (3 if common_td_sign == 1 else -1) * mp.pi / 4)
        for ell, m in ((3, 3), (4, 4)):
            coefficient = ref.leading_td_coefficient_phase(ell, m, common_td_sign=common_td_sign)
            assert abs(ref.spa_relative_phase(coefficient, c22, m) - expected_invariant) < mp.mpf("1e-60")
            delta = mp.pi * (3 * ell + (ell + m) % 2 - 2) / 2
            native_invariant = ref.primitive_relative_phase(mp.exp(-mp.j * delta), mp.exp(-2 * mp.j * mp.pi), m)
            assert abs(native_invariant + 1) < mp.mpf("1e-60")


def test_22_phase_branch_change_is_an_odd_mode_sign():
    with mp.workdps(65):
        # Changing arg(C22) by 2*pi changes its half-angle orbital alignment by pi.
        for m in (1, 2, 3, 4):
            change = mp.exp(mp.j * m * mp.pi)
            assert abs(change - (-1)**m) < mp.mpf("1e-60")


def test_spherical_basis_strain_sign_conversion():
    # Blanchet (8.2)-(8.3), observer azimuth pi/2; SXS App. C (C.21)-(C.24).
    iota = sp.Symbol("iota", real=True)
    p = sp.Matrix((1, 0, 0))
    q = sp.Matrix((0, sp.cos(iota), -sp.sin(iota)))
    etheta, ephi = q, -p
    xx, xy, xz, yy, yz, zz = sp.symbols("xx xy xz yy yz zz", real=True)
    h = sp.Matrix(((xx, xy, xz), (xy, yy, yz), (xz, yz, zz)))

    def strain(a, b):
        return ((a.T * h * a)[0] - (b.T * h * b)[0]) / 2 - sp.I * (a.T * h * b)[0]

    assert sp.simplify(strain(etheta, ephi) + strain(p, q)) == 0


@pytest.mark.parametrize("common_td_sign", (1, -1))
def test_conditional_conversion_is_common_carrier_phase_plus_orbit(common_td_sign):
    # Algebraic carrier decomposition, not a proof about real polarizations.
    with mp.workdps(65):
        k = -mp.pi / 4 if common_td_sign == 1 else 3 * mp.pi / 4
        for m in (1, 2, 3, 4):
            correction = ref.london_orbital_phase_correction(m, common_td_sign=common_td_sign)
            assert abs(mp.exp(mp.j * correction) -
                       mp.exp(-mp.j * k) * mp.exp(mp.j * m * k / 2)) < mp.mpf("1e-60")


def test_invalid_leading_phase_inputs():
    with pytest.raises(ValueError, match="common_td_sign"):
        ref.leading_td_coefficient_phase(2, 2, common_td_sign=0)
    with pytest.raises(ValueError, match="unavailable"):
        ref.leading_td_coefficient_phase(5, 5, common_td_sign=1)


@pytest.mark.parametrize("m", (0, -1, 1.5, True))
def test_invalid_harmonic(m):
    with pytest.raises(ValueError, match="positive integer"):
        ref.primitive_powers(m)


@pytest.mark.parametrize("value", (0, mp.inf, mp.nan))
def test_undefined_unit_phase(value):
    with pytest.raises(ValueError, match="finite nonzero"):
        ref.unit_phase(value)


def test_invalid_gaussian_and_spa_domains():
    with pytest.raises(ValueError, match="epsilon"):
        ref.gaussian_carrier(3, ".2", 0, ".3", ".4", 0)
    with pytest.raises(ValueError, match="positive"):
        ref.spa_carrier(3, ".2", 0, ".3", 0)

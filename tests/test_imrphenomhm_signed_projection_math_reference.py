"""Signed harmonic projection derived from real TD polarizations.

No production or NR values supply the oracle. A damped chirp deliberately
has nonzero Fourier leakage, so a one-sided projection cannot pass the TD
quadrature checks accidentally.
"""

import pytest

mp = pytest.importorskip("mpmath")
sp = pytest.importorskip("sympy")

from diffgw_reference import imrphenomhm_projection as projection
from diffgw_reference import imrphenomhm_signed_projection as ref
from diffgw_reference import imrphenomhm_spa_conventions as spa


@pytest.mark.parametrize("ell,m", ref.MODES)
def test_exact_projection_from_strain_and_fourier_conjugation(ell, m):
    # F[h]=Q Y_lm + (-1)^ell P Y_l,-m; F[conj(h)] has conjugate Y.
    p, q = sp.symbols("P Q", complex=True)
    yplus = projection.harmonic(ell, m).subs(projection.AZIMUTH, -projection.AZIMUTH)
    yminus = projection.harmonic(ell, -m).subs(projection.AZIMUTH, -projection.AZIMUTH)
    strain = q * yplus + (-1)**ell * p * yminus
    conjugate_strain = p * sp.conjugate(yplus) + (-1)**ell * q * sp.conjugate(yminus)
    # h=hplus-i*hcross, so hcross=i*(h-conj(h))/2.
    expected = ((strain + conjugate_strain) / 2,
                sp.I * (strain - conjugate_strain) / 2)
    result = ref.symbolic_project_pair(ell, m, p, q)
    assert all(sp.simplify(a - b) == 0 for a, b in zip(result, expected))
    one_sided = ref.symbolic_project_pair(ell, m, p, 0)
    assert all(sp.simplify(a - d * p) == 0 for a, d in zip(
        one_sided, projection.native_carrier_coefficients(ell, m)))


@pytest.mark.parametrize("ell,m", ref.MODES)
def test_phase_transport_preserves_signed_symmetry_and_real_polarizations(ell, m):
    p, q, h = sp.symbols("P Q h", complex=True)
    constant = sp.Symbol("K", real=True)
    factor = ref.symbolic_phase_factor(m, constant)
    positive, opposite = ref.symbolic_transport_pair(m, constant, p, q)
    changed_h = sp.conjugate(factor) * h
    changed_partner = factor * (-1)**ell * sp.conjugate(h)
    assert sp.simplify(changed_partner - (-1)**ell * sp.conjugate(changed_h)) == 0
    assert sp.simplify(positive - factor * p) == 0
    assert sp.simplify(opposite - sp.conjugate(factor) * q) == 0
    # At -f, P(-f)=conj(Q(f)), Q(-f)=conj(P(f)).
    minus_p, minus_q = ref.symbolic_transport_pair(
        m, constant, sp.conjugate(q), sp.conjugate(p))
    plus = ref.symbolic_project_pair(ell, m, positive, opposite)
    minus = ref.symbolic_project_pair(ell, m, minus_p, minus_q)
    assert all(sp.simplify(a - sp.conjugate(b)) == 0 for a, b in zip(minus, plus))
    if m == 2:
        assert factor == 1
        assert plus == ref.symbolic_project_pair(ell, m, p, q)


@pytest.mark.parametrize("ell,m", ref.MODES)
@pytest.mark.parametrize("inclination", (".83", "2.47"))
def test_transport_against_real_td_quadrature_with_leakage(ell, m, inclination):
    with mp.workdps(65):
        inclination, azimuth = mp.mpf(inclination), mp.mpf(".37")
        f, phi0, omega, k, epsilon, amplitude = map(
            mp.mpf, (".17", ".41", ".31", ".13", "1.2", "1.7"))
        coefficient = mp.mpc("-.7", ".9")
        constant = 3 * mp.pi / 4
        factor = mp.exp(mp.j * (mp.mpf(m) / 2 - 1) * constant)
        p, q = ref.gaussian_pair(m, f, phi0, omega, k, epsilon, coefficient, amplitude)
        assert abs(q) > mp.mpf(".01") * abs(p)
        # Independent construction: TD harmonic sum, then real and imaginary parts.
        y = [sp.lambdify((projection.IOTA, projection.AZIMUTH),
                        projection.harmonic(ell, signed_m), "mpmath")(
                            inclination, -azimuth) for signed_m in (m, -m)]

        def td_strain(t):
            mode = (mp.conj(factor) * coefficient * amplitude
                    * mp.exp(-epsilon * t**2 / 2
                             - mp.j * m * (phi0 + omega * t + k * t**2 / 2)))
            return mode * y[0] + (-1)**ell * mp.conj(mode) * y[1]

        expected = ref.project_pair(ell, m, inclination, azimuth, factor * p,
                                    mp.conj(factor) * q)
        measured = tuple(mp.quad(
            lambda t: component(td_strain(t)) * mp.exp(-2 * mp.pi * mp.j * f * t),
            [-20, -12, -6, 0, 6, 12, 20]) for component in (mp.re, lambda z: -mp.im(z)))
        tail = (2 * abs(coefficient) * amplitude * sum(map(abs, y))
                * mp.exp(-epsilon * 200) / (epsilon * 20))
        assert tail < mp.mpf("1e-100")
        assert all(abs(a - b) < mp.mpf("1e-48") for a, b in zip(measured, expected))
        one_sided = ref.project_pair(ell, m, inclination, azimuth, factor * p, 0)
        assert max(abs(a - b) for a, b in zip(expected, one_sided)) > mp.mpf("1e-4")
        # Directly evaluate both signed frequencies rather than assigning conjugates.
        minus_p, minus_q = ref.gaussian_pair(
            m, -f, phi0, omega, k, epsilon, coefficient, amplitude)
        minus = ref.project_pair(ell, m, inclination, azimuth,
                                 factor * minus_p, mp.conj(factor) * minus_q)
        assert all(abs(a - mp.conj(b)) < mp.mpf("1e-60") for a, b in zip(minus, expected))


def test_common_fd_phase_is_not_generic_real_polarization_rotation():
    # Every real polarization rotation preserves hp^2+hx^2 (without conjugation).
    # A generic elliptical positive-frequency signal has this invariant nonzero.
    a, b, angle = sp.symbols("a b angle", real=True)
    vector = sp.Matrix((a, -sp.I * b))
    rotation = sp.Matrix(((sp.cos(angle), -sp.sin(angle)),
                          (sp.sin(angle), sp.cos(angle))))
    rotated = rotation * vector
    invariant = (vector.T * vector)[0]
    assert sp.simplify((rotated.T * rotated)[0] - invariant) == 0
    common_phase = sp.exp(sp.I * 3 * sp.pi / 4)
    assert sp.simplify((common_phase * vector).dot(common_phase * vector)
                       - common_phase**2 * invariant) == 0
    # Generic 22 angle cos(iota)=1/3; remove its common nonzero normalization.
    elliptical = invariant.subs({a: sp.Rational(10, 9), b: sp.Rational(2, 3)})
    assert elliptical == sp.Rational(64, 81)
    assert sp.simplify((common_phase**2 - 1) * elliptical) != 0
    # The exceptional circular limit explains why a pole-only check is insufficient.
    assert invariant.subs(b, a) == 0


@pytest.mark.parametrize("inclination", (".83", "2.47"))
def test_both_hemispheres_have_nonzero_polarization_rotation_obstruction(inclination):
    with mp.workdps(60):
        hp, hx = ref.project_pair(2, 2, inclination, 0, 1, 0)
        invariant = hp**2 + hx**2
        assert abs(invariant) > mp.mpf(".001")
        factor = mp.exp(-mp.j * 3 * mp.pi / 4)
        assert abs((factor * hp)**2 + (factor * hx)**2 - invariant) > mp.mpf(".001")


@pytest.mark.parametrize("m", (1, 2, 3, 4))
def test_gaussian_pair_reuses_positive_frequency_reference(m):
    with mp.workdps(60):
        args = (m, ".17", ".41", ".31", ".13", "1.2", mp.mpc("-.7", ".9"), "1.7")
        positive, _ = ref.gaussian_pair(*args)
        assert abs(positive - spa.gaussian_carrier(*args)) < mp.mpf("1e-55")
        zero_p, zero_q = ref.gaussian_pair(m, 0, *args[2:])
        assert abs(zero_q - mp.conj(zero_p)) < mp.mpf("1e-55")


@pytest.mark.parametrize("m", (1, 2, 3, 4))
def test_controlled_opposite_frequency_suppression(m):
    # At 2*pi*f=m*omega, completing the square gives this exact leakage ratio.
    # Increasing omega shifts the Fourier peak away from the opposite support;
    # it does not alter damping or chirp rate to conceal a projection error.
    with mp.workdps(65):
        k, epsilon = mp.mpf(".13"), mp.mpf("1.2")
        ratios = []
        for omega in map(mp.mpf, (".5", "1", "2", "4")):
            frequency = m * omega / (2 * mp.pi)
            p, q = ref.gaussian_pair(m, frequency, ".41", omega, k, epsilon)
            ratio = abs(q / p)
            exact = mp.exp(-2 * m**2 * omega**2 * epsilon / (epsilon**2 + m**2 * k**2))
            assert abs(ratio / exact - 1) < mp.mpf("1e-59")
            ratios.append(ratio)
        assert all(later < earlier for earlier, later in zip(ratios, ratios[1:]))
        assert ratios[-1] < mp.mpf("1e-10")


def test_invalid_gaussian_domain():
    with pytest.raises(ValueError, match="finite"):
        ref.gaussian_pair(2, mp.nan, 0, 1, 1, 1)
    with pytest.raises(ValueError, match="positive"):
        ref.gaussian_pair(2, 0, 0, 1, 1, 0)
    with pytest.raises(ValueError, match="positive integer"):
        ref.gaussian_pair(0, 0, 0, 1, 1, 1)

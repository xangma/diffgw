"""Independent scalar ppE and massive-graviton equations from primary papers.

The paper phase and the project's carrier multiplier have deliberately separate
functions. This module does not import either waveform frontend or LAL.
"""

import mpmath as mp

SPEC_VERSION = "ppe-phase-dispersion/v1"
AMPLITUDE_SPEC_VERSION = "ppe-inspiral-amplitude/v1"
SOURCE_PPE = "Yunes & Pretorius, PRD 80 122003 (2009), arXiv:0909.3328, Eq. (1)"
SOURCE_DISPERSION = (
    "Mirshekari, Yunes & Will, PRD 85 024041 (2012), "
    "arXiv:1110.2720, Eqs. (14)-(16), (28)-(31)"
)

# SI constants stated independently of diffgw_core.constants.
_C = "299792458"
_G = "6.67430e-11"
_MSUN = "1.9884098706980507e30"
_MPC = "3.085677581491367e22"


def chirp_mass_seconds(mass1_solar, mass2_solar):
    """Chirp mass of the supplied positive masses, in geometric seconds."""
    m1, m2 = mp.mpf(mass1_solar), mp.mpf(mass2_solar)
    if m1 <= 0 or m2 <= 0:
        raise ValueError("component masses must be positive")
    chirp_solar = (m1 * m2) ** (mp.mpf(3) / 5) / (m1 + m2) ** (mp.mpf(1) / 5)
    return mp.mpf(_G) * mp.mpf(_MSUN) * chirp_solar / mp.mpf(_C)**3


def paper_ppe_phase(f_hz, mass1_solar, mass2_solar, beta, paper_b):
    """Yunes--Pretorius positive Fourier-phase term beta*(pi*Mc*f)^b."""
    f = mp.mpf(f_hz)
    if f <= 0:
        raise ValueError("paper inspiral frequency must be positive")
    return mp.mpf(beta) * (mp.pi * chirp_mass_seconds(mass1_solar, mass2_solar) * f) ** mp.mpf(paper_b)


def paper_ppe_amplitude_factor(f_hz, mass1_solar, mass2_solar, alpha, paper_a):
    """Yunes--Pretorius Eqs. (1), (47): signed inspiral factor 1+alpha*u**a."""
    f = mp.mpf(f_hz)
    if f <= 0:
        raise ValueError("paper inspiral frequency must be positive")
    u = mp.pi * chirp_mass_seconds(mass1_solar, mass2_solar) * f
    return 1 + mp.mpf(alpha) * u ** mp.mpf(paper_a)


def _cosmology_integral(z, power, omega_m, omega_lambda):
    z = mp.mpf(z)
    if z < 0:
        raise ValueError("redshift must be nonnegative")
    om, ol = mp.mpf(omega_m), mp.mpf(omega_lambda)
    if om < 0 or ol < 0 or om + ol <= 0:
        raise ValueError("cosmological density parameters must be nonnegative")
    return mp.quad(lambda zp: (1 + zp) ** power / mp.sqrt(om * (1 + zp) ** 3 + ol), [0, z])


def cosmological_d0(z, h0_km_s_mpc=72, omega_m="0.3", omega_lambda="0.7"):
    """Mirshekari et al. Eq. (15) for D_0, in meters."""
    z = mp.mpf(z)
    h0 = mp.mpf(h0_km_s_mpc) * 1000 / mp.mpf(_MPC)
    if h0 <= 0:
        raise ValueError("Hubble constant must be positive")
    return (1 + z) * mp.mpf(_C) / h0 * _cosmology_integral(z, -2, omega_m, omega_lambda)


def cosmological_luminosity_distance(z, h0_km_s_mpc=72, omega_m="0.3", omega_lambda="0.7"):
    """Mirshekari et al. Eq. (16), flat matter/Lambda D_L in meters."""
    z = mp.mpf(z)
    h0 = mp.mpf(h0_km_s_mpc) * 1000 / mp.mpf(_MPC)
    if h0 <= 0:
        raise ValueError("Hubble constant must be positive")
    return (1 + z) * mp.mpf(_C) / h0 * _cosmology_integral(z, 0, omega_m, omega_lambda)


def d0_from_luminosity_distance(distance_mpc, z, omega_m="0.3", omega_lambda="0.7"):
    """Eliminate H0 between the paper's D0 and D_L integrals (Eqs. 15-16)."""
    distance = mp.mpf(distance_mpc) * mp.mpf(_MPC)
    z = mp.mpf(z)
    if distance <= 0 or z < 0:
        raise ValueError("distance must be positive and redshift nonnegative")
    if z == 0:
        return distance  # Continuous ratio limit when D_L is supplied independently.
    numerator = _cosmology_integral(z, -2, omega_m, omega_lambda)
    denominator = _cosmology_integral(z, 0, omega_m, omega_lambda)
    return distance * numerator / denominator


def paper_dispersion_phase(f_hz, d0_m, lambda_g_m, z):
    """Eq. (28)-(31) massive-graviton delta-Psi, with SI c restored."""
    f, d0, wavelength, z = map(mp.mpf, (f_hz, d0_m, lambda_g_m, z))
    if f <= 0 or d0 < 0 or wavelength <= 0 or z < 0:
        raise ValueError("requires f>0, D0>=0, lambda_g>0, z>=0")
    return -mp.pi * d0 * mp.mpf(_C) / (wavelength**2 * (1 + z) * f)


def carrier_phase_shift(f_hz, mass1_solar, mass2_solar, beta, native_b, d0_m, lambda_g_m, z):
    """Map native integer velocity exponent to paper b=native_b/3."""
    return paper_ppe_phase(f_hz, mass1_solar, mass2_solar, beta, mp.mpf(native_b) / 3) + paper_dispersion_phase(f_hz, d0_m, lambda_g_m, z)


def carrier_multiplier(f_hz, mass1_solar, mass2_solar, beta, native_b, d0_m, lambda_g_m, z):
    """Project adapter: exp(-i delta-Psi) for native TaylorF2 carrier sign."""
    return mp.exp(-1j * carrier_phase_shift(f_hz, mass1_solar, mass2_solar, beta, native_b, d0_m, lambda_g_m, z))


def carrier_amplitude_multiplier(f_hz, mass1_solar, mass2_solar, alpha, paper_a,
                                 beta, native_b, d0_m, lambda_g_m, z):
    """Project adapter: paper amplitude times negative-phase carrier multiplier."""
    return (paper_ppe_amplitude_factor(f_hz, mass1_solar, mass2_solar, alpha, paper_a)
            * carrier_multiplier(f_hz, mass1_solar, mass2_solar, beta, native_b,
                                 d0_m, lambda_g_m, z))

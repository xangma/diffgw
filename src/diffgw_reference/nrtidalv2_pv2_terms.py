"""Published aligned-frame Pv2 finite-size terms before the precession rotation.

This is a signed additive tidal-amplitude contribution, not a complete
IMRPhenomPv2_NRTidalv2 waveform or a multiplicative BBH quotient.
"""

import mpmath as mp

from diffgw_reference import nrtidalv2_amplitude, nrtidalv2_phase, nrtidalv2_self_spin


SPEC_VERSION = "nrtidalv2-pv2-aligned-additive-terms/v1"
ALIGNED_CARRIER_SPEC_VERSION = "nrtidalv2-pv2-aligned-carrier/v1"


def additive_tidal_term(f, m1, m2, chi1, chi2, lam1, lam2,
                        m_sec, d_sec, f_merger, *, dps=80):
    """A_T W exp[-i(psi_T+S_2+S_3+S_3.5)] in the paper's aligned frame."""
    with mp.workdps(dps):
        frequency = mp.mpf(str(f))
        mass_sec = mp.mpf(str(m_sec))
        x = (mp.pi * mass_sec * frequency) ** (mp.mpf(2) / 3)
        delta = (mp.mpf(str(m1)) - mp.mpf(str(m2))) / (
            mp.mpf(str(m1)) + mp.mpf(str(m2)))
        tidal = nrtidalv2_phase.evaluate(
            frequency, mass_sec, delta, lam1, lam2, dps=dps)
        spin = sum(nrtidalv2_self_spin.components(
            x, str(m1), str(m2), str(chi1), str(chi2), str(lam1), str(lam2)))
        amplitude = nrtidalv2_amplitude.amplitude(
            frequency, mass_sec, d_sec, delta, lam1, lam2, dps=dps)
        window = nrtidalv2_amplitude.taper(frequency, f_merger, dps=dps)
        return complex(amplitude * window * mp.exp(-mp.j * (tidal + spin)))


def assembled_carrier_from_bbh(f, bbh_face_on, m1, m2, chi1, chi2,
                               lam1, lam2, m_sec, d_sec, f_merger, *, dps=80):
    """Published aligned Pv2 BNS carrier, conditional on a supplied BBH carrier.

    The reference composes independently defined scalar terms; it has no
    native waveform imports. Live bins require a nonzero BBH carrier.
    """
    with mp.workdps(dps):
        frequency = mp.mpf(str(f))
        bbh = mp.mpc(str(bbh_face_on.real), str(bbh_face_on.imag))
        window = nrtidalv2_amplitude.taper(frequency, f_merger, dps=dps)
        if window == 0:
            return 0j
        if bbh == 0:
            raise ValueError("supplied BBH carrier must be nonzero on live bins")
        mass_sec = mp.mpf(str(m_sec))
        delta = (mp.mpf(str(m1)) - mp.mpf(str(m2))) / (
            mp.mpf(str(m1)) + mp.mpf(str(m2)))
        x = (mp.pi * mass_sec * frequency) ** (mp.mpf(2) / 3)
        phase = nrtidalv2_phase.evaluate(
            frequency, mass_sec, delta, lam1, lam2, dps=dps)
        spin = sum(nrtidalv2_self_spin.components(
            x, str(m1), str(m2), str(chi1), str(chi2), str(lam1), str(lam2)))
        amplitude = nrtidalv2_amplitude.amplitude(
            frequency, mass_sec, d_sec, delta, lam1, lam2, dps=dps)
        return complex(window * (bbh + amplitude * bbh / abs(bbh))
                       * mp.exp(-mp.j * (phase + spin)))

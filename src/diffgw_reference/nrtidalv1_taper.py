"""Independent scalar composition of the NRTidal v1 phase and merger taper.

The merger fit and window are shared published components, already encoded in
``nrtidalv2_amplitude`` from Dietrich et al. (2018), Eq. (11), and Dietrich et
al. (2019), Eq. (25). The latter explicitly says the taper is unchanged from
the previous NRTidal implementation. The descending Planck branch is from
McKechan et al. (2010), Eq. (7), cited by Dietrich et al. (2018). No native or
LAL implementation is used.
"""

import mpmath as mp

from diffgw_reference.nrtidalv1_phase import evaluate as phase
from diffgw_reference.nrtidalv2_amplitude import merger_frequency, taper


SPEC_VERSION = "nrtidalv1-nonspinning-merger-taper/v1"
COMPONENT_IDS = {
    "merger_frequency": "nrtidal.shared.merger_frequency.eq11.v1",
    "taper": "nrtidal.shared.planck_taper.v1",
    "factor": "nrtidalv1.composed_tidal_factor.v1",
}


def factor(f, m_sec, delta, lambda1, lambda2, *, carrier_rounded=False,
           dps=80):
    """Return exp(-i psi_T) W(f), with the explicitly selected v1 phase."""
    with mp.workdps(dps):
        f_merger = merger_frequency(m_sec, delta, lambda1, lambda2, dps=dps)
        psi = phase(f, m_sec, delta, lambda1, lambda2,
                    carrier_rounded=carrier_rounded, dps=dps)
        return +(mp.exp(-mp.j * psi) * taper(f, f_merger, dps=dps))

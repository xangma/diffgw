"""NRTidal fit values published in the scientific literature.

Pure Python float fitting coefficients. Zero external framework dependencies.
Values and constraints are transcribed from:
- Dietrich et al. (2017), Phys. Rev. D 96, 121501(R), arXiv:1706.02969 Eq. 7 [NRTidal v1]
- Dietrich et al. (2019), Phys. Rev. D 100, 044003, arXiv:1905.06011 Eqs. 21--24 [NRTidal v2]
- Dietrich et al. (2018), Phys. Rev. D 99, 024029, arXiv:1804.02235 Eq. 11 [BNS merger frequency]
"""

import math

# Dietrich et al. (2017) Eq. 7
NRTIDAL_V1_PADÉ_COEFFS = {
    "c_newt": 2.4375,
    "n_1": -17.428,
    "n_3over2": 31.867,
    "n_2": -26.414,
    "n_5over2": 62.362,
    "d_1": -19.924,
    "d_3over2": 36.089,
}
NRTIDAL_V1_PADE_COEFFS = NRTIDAL_V1_PADÉ_COEFFS

_n52, _n3, _d1, _d2 = 90.550822, -60.253578, -15.111208, 8.0641096
_c1, _c32, _c2, _c52 = (
    3115.0 / 1248.0,
    -math.pi,
    28024205.0 / 3302208.0,
    -4283.0 * math.pi / 1092.0,
)

# Dietrich et al. (2019) Eqs. 21--23
NRTIDAL_V2_PHASE_COEFFS = {
    "c_newt": 2.4375,
    "n_1": _c1 + _d1,
    "n_3over2": (_c1 * _c32 - _c52 - _c32 * _d1 + _n52) / _c1,
    "n_2": _c2 + _c1 * _d1 + _d2,
    "n_5over2": _n52,
    "n_3": _n3,
    "d_1": _d1,
    "d_3over2": -(_c52 + _c32 * _d1 - _n52) / _c1,
    "d_2": _d2,
}

# Dietrich et al. (2019) Eq. 24; the paper prints d only to one decimal place.
NRTIDAL_V2_AMP_COEFFS = {
    "n_1": 449.0 / 108.0,
    "n_289": 22672.0 / 9.0,
    "d": 13477.8,
}

# Dietrich et al. (2018) Eq. 11: BNS merger frequency fit coefficients
BNS_MERGER_FREQ_COEFFS = {
    "n1": 3.354e-2,
    "n2": 4.315e-5,
    "d1": 7.542e-2,
    "d2": 2.236e-4,
    "m_omega_0": 0.3586,
}

__all__ = [
    "NRTIDAL_V1_PADÉ_COEFFS",
    "NRTIDAL_V1_PADE_COEFFS",
    "NRTIDAL_V2_PHASE_COEFFS",
    "NRTIDAL_V2_AMP_COEFFS",
    "BNS_MERGER_FREQ_COEFFS",
]

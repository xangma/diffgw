"""NRTidal fit values published in the scientific literature.

Values and constraints are transcribed from Dietrich et al. (2017),
arXiv:1706.02969 Eq. 7, and Dietrich et al. (2019), arXiv:1905.06011
Eqs. 21--24. The papers publish fitted decimals at the precision shown;
constrained coefficients are evaluated from their printed equations.
"""
import math

NRTIDAL_V1_PADÉ_COEFFS = {"c_newt": 2.4375, "n_1": -17.428, "n_3over2": 31.867, "n_2": -26.414, "n_5over2": 62.362, "d_1": -19.924, "d_3over2": 36.089}
_n52, _n3, _d1, _d2 = 90.550822, -60.253578, -15.111208, 8.0641096
_c1, _c32, _c2, _c52 = 3115.0 / 1248.0, -math.pi, 28024205.0 / 3302208.0, -4283.0 * math.pi / 1092.0
NRTIDAL_V2_PHASE_COEFFS = {
    "c_newt": 2.4375, "n_1": _c1 + _d1,
    "n_3over2": (_c1 * _c32 - _c52 - _c32 * _d1 + _n52) / _c1,
    "n_2": _c2 + _c1 * _d1 + _d2, "n_5over2": _n52, "n_3": _n3,
    "d_1": _d1, "d_3over2": -(_c52 + _c32 * _d1 - _n52) / _c1, "d_2": _d2,
}
# Eq. 24; the paper prints d only to one decimal place.
NRTIDAL_V2_AMP_COEFFS = {"n_1": 449.0 / 108.0, "n_289": 22672.0 / 9.0, "d": 13477.8}
__all__ = ["NRTIDAL_V1_PADÉ_COEFFS", "NRTIDAL_V2_PHASE_COEFFS", "NRTIDAL_V2_AMP_COEFFS"]

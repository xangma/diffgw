"""Published PhenomD phenomenological-coefficient fits.

The coefficients below are a clean-room transcription of Table V of
Khan et al., *Phys. Rev. D* 93, 044007 (2016), arXiv:1508.07253v2.
``chi_pn`` is the *unscaled* reduced-spin coordinate used by that table,
not ``chi_eff`` and not the normalized ``hat{chi}`` coordinate.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch

PARAMETER_NAMES = (
    "rho_1", "rho_2", "rho_3", "v_2",
    "gamma_1", "gamma_2", "gamma_3",
    "sigma_1", "sigma_2", "sigma_3", "sigma_4",
    "beta_1", "beta_2", "beta_3",
    "alpha_1", "alpha_2", "alpha_3", "alpha_4", "alpha_5",
)

# Printed Table-V literals are intentionally kept as strings: converting them
# to a tensor happens only at evaluation time, without invented digits.
TABLE_V_ROWS: Tuple[Tuple[str, ...], ...] = (
    ("3931.9", "-17395.8", "3132.38", "343966.", "-1.21626e6", "-70698.", "1.38391e6", "-3.96628e6", "-60017.5", "803515.", "-2.09171e6"),
    ("-40105.5", "112253.", "23561.7", "-3.47618e6", "1.13759e7", "754313.", "-1.30848e7", "3.64446e7", "596227.", "-7.42779e6", "1.8929e7"),
    ("83208.4", "-191238.", "-210916.", "8.71798e6", "-2.69149e7", "-1.98898e6", "3.0888e7", "-8.39087e7", "-1.4535e6", "1.70635e7", "-4.27487e7"),
    ("0.814984", "2.57476", "1.16102", "-2.36278", "6.77104", "0.757078", "-2.72569", "7.11404", "0.176693", "-0.797869", "2.11624"),
    ("0.0069274", "0.0302047", "0.00630802", "-0.120741", "0.262716", "0.00341518", "-0.107793", "0.27099", "0.000737419", "-0.0274962", "0.0733151"),
    ("1.01034", "0.000899312", "0.283949", "-4.04975", "13.2078", "0.103963", "-7.02506", "24.7849", "0.030932", "-2.6924", "9.60937"),
    ("1.30816", "-0.00553773", "-0.0678292", "-0.668983", "3.40315", "-0.0529658", "-0.992379", "4.82068", "-0.00613414", "-0.384293", "1.75618"),
    ("2096.55", "1463.75", "1312.55", "18307.3", "-43534.1", "-833.289", "32047.3", "-108609.", "452.251", "8353.44", "-44531.3"),
    ("-10114.1", "-44631.", "-6541.31", "-266959.", "686328.", "3405.64", "-437508.", "1.63182e6", "-7462.65", "-114585.", "674402."),
    ("22933.7", "230960.", "14961.1", "1.19402e6", "-3.10422e6", "-3038.17", "1.87203e6", "-7.30915e6", "42738.2", "467502.", "-3.06485e6"),
    ("-14621.7", "-377813.", "-9608.68", "-1.71089e6", "4.33292e6", "-22366.7", "-2.50197e6", "1.02745e7", "-85360.3", "-570025.", "4.39684e6"),
    ("97.8975", "-42.6597", "153.484", "-1417.06", "2752.86", "138.741", "-1433.66", "2857.74", "41.0251", "-423.681", "850.359"),
    ("-3.2827", "-9.05138", "-12.4154", "55.4716", "-106.051", "-11.953", "76.807", "-155.332", "-3.41293", "25.5724", "-54.408"),
    ("-2.51564e-5", "1.97503e-5", "-1.83707e-5", "2.18863e-5", "8.25024e-5", "7.15737e-6", "-5.578e-5", "1.91421e-4", "5.44717e-6", "-3.22061e-5", "7.97402e-5"),
    ("43.3151", "638.633", "-32.8577", "2415.89", "-5766.88", "-61.8546", "2953.97", "-8986.29", "-21.5714", "981.216", "-3239.57"),
    ("-0.0702021", "-0.162698", "-0.187251", "1.13831", "-2.83342", "-0.17138", "1.71975", "-4.53972", "-0.0499834", "0.606207", "-1.68277"),
    ("9.59881", "-397.054", "16.2021", "-1574.83", "3600.34", "27.0924", "-1786.48", "5152.92", "11.1757", "-577.8", "1808.73"),
    ("-0.0298949", "1.40221", "-0.0735605", "0.833701", "0.224001", "-0.0552029", "0.566719", "0.718693", "-0.0155074", "0.157503", "0.210768"),
    ("0.997441", "-0.00788445", "-0.0590469", "1.39587", "-4.51663", "-0.0558534", "1.75166", "-5.99021", "-0.0179453", "0.59651", "-2.06089"),
)

_ROW_MAP: Dict[str, Tuple[str, ...]] = dict(zip(PARAMETER_NAMES, TABLE_V_ROWS))


def eval_table_v_fit(param_name: str, eta: torch.Tensor, chi_pn: torch.Tensor) -> torch.Tensor:
    """Evaluate Eq. (31), preserving broadcast shape, dtype, and device."""
    if param_name not in _ROW_MAP:
        raise KeyError(f"unknown Table V parameter {param_name!r}")
    if not torch.is_tensor(eta):
        eta = torch.as_tensor(eta, dtype=torch.get_default_dtype())
    if not torch.is_tensor(chi_pn):
        chi_pn = torch.as_tensor(chi_pn, dtype=eta.dtype, device=eta.device)
    dtype = torch.promote_types(eta.dtype, chi_pn.dtype)
    if not dtype.is_floating_point:
        raise TypeError("eta and chi_pn must have floating-point dtype")
    eta, chi_pn = torch.broadcast_tensors(eta.to(dtype=dtype), chi_pn.to(dtype=dtype, device=eta.device))
    c = torch.as_tensor([float(x) for x in _ROW_MAP[param_name]], dtype=dtype, device=eta.device)
    x = chi_pn - 1.0
    return c[0] + c[1] * eta + x * (c[2] + c[3] * eta + c[4] * eta * eta) + x * x * (c[5] + c[6] * eta + c[7] * eta * eta) + x * x * x * (c[8] + c[9] * eta + c[10] * eta * eta)


__all__ = ["PARAMETER_NAMES", "TABLE_V_ROWS", "eval_table_v_fit"]

"""Shared framework-agnostic physics core for differentiable gravitational waveforms.

Contains physical constants, published calibration fits, and kinematic formulas.
Zero external heavy dependencies (pure Python & standard library math).
"""

from .constants import (
    C_SI,
    G_SI,
    MSUN_SI,
    MSUN_SEC,
    MPC_SI,
    EULER_GAMMA,
    PI,
)
from .fits import (
    TABLE_V_UNROUNDED_COEFFS,
    AMP_FJOIN_INS,
    PHI_FJOIN_INS,
    F_CUT,
    MIN_FINAL_SPIN,
    MAX_ALLOWED_MASS_RATIO,
    FINAL_SPIN_ETA_POLY,
    FINAL_SPIN_S_COEFFS,
    ERAD_ETA_NUMERATOR,
    ERAD_S_NUMERATOR,
    ERAD_S_DENOMINATOR,
    eval_table_v_fit,
)
from .conversions import (
    component_masses_to_chirp_mass,
    component_masses_to_eta,
    component_masses_to_delta,
    spin_combinations,
    total_spin_for_remnant,
)

from .nrtidal_fits import (
    NRTIDAL_V1_PADÉ_COEFFS,
    NRTIDAL_V1_PADE_COEFFS,
    NRTIDAL_V2_PHASE_COEFFS,
    NRTIDAL_V2_AMP_COEFFS,
    BNS_MERGER_FREQ_COEFFS,
)

__all__ = [
    "C_SI",
    "G_SI",
    "MSUN_SI",
    "MSUN_SEC",
    "MPC_SI",
    "EULER_GAMMA",
    "PI",
    "TABLE_V_UNROUNDED_COEFFS",
    "AMP_FJOIN_INS",
    "PHI_FJOIN_INS",
    "F_CUT",
    "MIN_FINAL_SPIN",
    "MAX_ALLOWED_MASS_RATIO",
    "FINAL_SPIN_ETA_POLY",
    "FINAL_SPIN_S_COEFFS",
    "ERAD_ETA_NUMERATOR",
    "ERAD_S_NUMERATOR",
    "ERAD_S_DENOMINATOR",
    "eval_table_v_fit",
    "component_masses_to_chirp_mass",
    "component_masses_to_eta",
    "component_masses_to_delta",
    "spin_combinations",
    "total_spin_for_remnant",
    "NRTIDAL_V1_PADÉ_COEFFS",
    "NRTIDAL_V1_PADE_COEFFS",
    "NRTIDAL_V2_PHASE_COEFFS",
    "NRTIDAL_V2_AMP_COEFFS",
    "BNS_MERGER_FREQ_COEFFS",
]

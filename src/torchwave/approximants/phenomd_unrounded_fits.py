"""Inherited unrounded IEEE FP64 fitting coefficients for IMRPhenomD.

The independent source and licensing of these extra digits remain under review.
The cited literature identifies the model, but does not establish provenance
of this unrounded representation. Moving the tables into the shared core does
not change that status.

Re-exports pure data tables from diffgw_core.fits and provides torch evaluation helper.
"""

from typing import Dict, Tuple, Union
import torch

from diffgw_core.fits import (
    AMP_FJOIN_INS,
    PHI_FJOIN_INS,
    F_CUT,
    MIN_FINAL_SPIN,
    MAX_ALLOWED_MASS_RATIO,
    TABLE_V_UNROUNDED_COEFFS,
    FINAL_SPIN_ETA_POLY,
    FINAL_SPIN_S_COEFFS,
    ERAD_ETA_NUMERATOR,
    ERAD_S_NUMERATOR,
    ERAD_S_DENOMINATOR,
    eval_table_v_fit as _eval_table_v_fit_float,
)


def eval_table_v_fit(
    param_name: str,
    eta: Union[torch.Tensor, float],
    chi_pn: Union[torch.Tensor, float],
) -> Union[torch.Tensor, float]:
    """Evaluate Table V hierarchical polynomial fit (Eq. 31 of Khan et al. 2016).

    Args:
        param_name: One of the 19 phenomenological parameter names.
        eta: Symmetric mass ratio eta = m1*m2/(m1+m2)^2 in (0, 0.25].
        chi_pn: PN reduced spin parameter chi_pn.

    Returns:
        Evaluated phenomenological parameter.
    """
    if isinstance(eta, torch.Tensor):
        coeffs = TABLE_V_UNROUNDED_COEFFS[param_name]
        c = [torch.tensor(val, dtype=eta.dtype, device=eta.device) for val in coeffs]
        eta2 = eta * eta
        xi = chi_pn - 1.0
        xi2 = xi * xi
        xi3 = xi2 * xi
        return (
            c[0] + c[1] * eta
            + xi * (c[2] + c[3] * eta + c[4] * eta2)
            + xi2 * (c[5] + c[6] * eta + c[7] * eta2)
            + xi3 * (c[8] + c[9] * eta + c[10] * eta2)
        )
    return _eval_table_v_fit_float(param_name, eta, chi_pn)

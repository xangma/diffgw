"""Scalar HM ringdown phase term with supplied QNMs and PhenomD coefficients.

London et al., arXiv:1708.00404v4, Eq. (14), modifying the last term of
Khan et al., arXiv:1508.07253v2, Eq. (14). No fitted surface, QNM table, or
frontend is imported here.
"""

from __future__ import annotations

import mpmath as mp


SPEC_VERSION = "imrphenomhm-ringdown-phase/v1"
COMPONENT_SOURCES = {
    "hm.phase.ringdown.eq9-c1.v1": (
        "https://arxiv.org/pdf/1708.00404v4",
        "Eq. (14); Khan et al. Eq. (14) and phase join at Mf=0.5 f_RD",
    ),
}
COMPONENT_BINDINGS = {
    "hm.phase.ringdown.eq9-c1.v1": {
        "reference": "carrier_delta_c1",
        "torch": ("torchwave.approximants.imrphenom_hm", "_production_eq14_delta_c1"),
        "jax": ("jaxwave.approximants.imrphenom_hm", "_production_eq14_delta_c1"),
        "tests": (
            "test_paper_limit",
            "test_frontend_values_and_gradients",
            "test_c1_join_and_22_control",
        ),
    },
}


def _number(value):
    return value if isinstance(value, mp.mpf) else mp.mpf(str(value))


def _parameters(eta, alpha4, alpha5, hm_fr22, hm_flm, hm_fd22, hm_fdlm,
                base_fr22, base_fd22):
    eta, a4, a5, h22, hlm, d22, dlm, b22, bd22 = map(
        _number, (eta, alpha4, alpha5, hm_fr22, hm_flm, hm_fd22, hm_fdlm,
                  base_fr22, base_fd22)
    )
    if not (eta > 0 and h22 > 0 and hlm > 0 and d22 > 0 and dlm > 0
            and b22 > 0 and bd22 > 0):
        raise ValueError("eta and all QNM frequencies/damping widths must be positive")
    return eta, a4, a5, h22, hlm, d22, dlm, b22, bd22


def paper_hm_term(f, eta, alpha4, alpha5, fr22, frlm, fdlm):
    """The changed arctangent term in London's Eq. (14), literal QNM convention."""
    f, eta, a4, a5, q22, qlm, damp = map(
        _number, (f, eta, alpha4, alpha5, fr22, frlm, fdlm)
    )
    if not (f > 0 and eta > 0 and q22 > 0 and qlm > 0 and damp > 0):
        raise ValueError("f, eta, and QNM frequencies/damping width must be positive")
    ratio = q22 / qlm
    return a4 / eta * ratio * mp.atan((f - a5 * q22) / (ratio * damp))


def phenomd_term(f, eta, alpha4, alpha5, fr22, fd22):
    """The original arctangent term in Khan et al.'s Eq. (14)."""
    f, eta, a4, a5, q22, damp = map(
        _number, (f, eta, alpha4, alpha5, fr22, fd22)
    )
    if not (f > 0 and eta > 0 and q22 > 0 and damp > 0):
        raise ValueError("f, eta, and QNM frequency/damping width must be positive")
    return a4 / eta * mp.atan((f - a5 * q22) / damp)


def carrier_delta(f, eta, alpha4, alpha5, hm_fr22, hm_flm, hm_fd22,
                  hm_fdlm, base_fr22, base_fd22):
    """HM-minus-D arctangent term using the live D-carrier QNM convention.

    The literal paper difference is recovered when both supplied 22 QNM pairs
    coincide. Otherwise HM QNMs supply only the RD and damping *ratios*, while
    the center, base width, and attachment frequency use the D-carrier pair.
    """
    f = _number(f)
    if f <= 0:
        raise ValueError("f must be positive")
    eta, a4, a5, h22, hlm, d22, dlm, b22, bd22 = _parameters(
        eta, alpha4, alpha5, hm_fr22, hm_flm, hm_fd22, hm_fdlm,
        base_fr22, base_fd22
    )
    ratio = h22 / hlm
    width = bd22 * ratio * dlm / d22
    center = a5 * b22
    return a4 / eta * (ratio * mp.atan((f - center) / width)
                       - mp.atan((f - center) / bd22))


def carrier_delta_c1(f, eta, alpha4, alpha5, hm_fr22, hm_flm, hm_fd22,
                     hm_fdlm, base_fr22, base_fd22):
    """Tangent-subtracted HM correction above the D phase join (Khan Eq. 37)."""
    f, join = _number(f), _number(base_fr22) / 2
    if f <= 0:
        raise ValueError("f must be positive")
    if f <= join:
        return mp.mpf("0")

    def raw(x):
        return carrier_delta(x, eta, alpha4, alpha5, hm_fr22, hm_flm,
                             hm_fd22, hm_fdlm, base_fr22, base_fd22)

    return raw(f) - raw(join) - mp.diff(raw, join) * (f - join)

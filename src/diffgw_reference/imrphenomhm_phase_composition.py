"""Source-composed HM phase with supplied D carrier, QNMs, and alpha fits.

London et al., arXiv:1708.00404v4, Methods Eqs. (3), (5)--(9),
composed with Khan et al., arXiv:1508.07253v2, Eq. (14). This module
contains no fitted surface, remnant model, QNM table, or frontend import.
"""

from __future__ import annotations

from .imrphenomhm_phase_offsets import mapped_phase
from .imrphenomhm_ringdown_phase import carrier_delta_c1


COMPONENT = "hm.phase.public-composition.eq3-14.v1"
SPEC_VERSION = "imrphenomhm-phase-composition/v1"
COMPONENT_SOURCES = {
    COMPONENT: (
        "https://arxiv.org/pdf/1708.00404v4",
        "Methods Eqs. (3), (5)--(9); Khan et al. Eq. (14)",
    ),
}
COMPONENT_BINDINGS = {
    COMPONENT: {
        "reference": "composed_phase",
        "torch": ("torchwave.approximants.imrphenom_hm", "IMRPhenomHM"),
        "jax": ("jaxwave.approximants.imrphenom_hm", "IMRPhenomHM"),
        "tests": ("test_public_phase_values_and_one_sided_slopes",),
    },
}


def composed_phase(
    f, f0_phi, fr_lm, fr22, ell, m, carrier_phase, *, eta, alpha4,
    alpha5, fd22, fd_lm, base_fr22, base_fd22,
):
    """Return the mapped intrinsic phase in positive dimensionless Mf.

    ``carrier_phase`` is the supplied D phase in the caller's frozen gauge.
    The London Eq. (9) correction is attached to that same D carrier before the
    three-branch London map/offset construction. All fitted inputs are supplied.
    """

    def corrected_carrier(g):
        return carrier_phase(g) + carrier_delta_c1(
            g, eta, alpha4, alpha5, fr22, fr_lm, fd22, fd_lm,
            base_fr22, base_fd22,
        )

    return mapped_phase(f, f0_phi, fr_lm, fr22, ell, m,
                        corrected_carrier)

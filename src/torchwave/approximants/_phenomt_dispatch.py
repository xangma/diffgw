"""Shared TP/TPHM routing for native precession and matching remnant spins."""

from ._phenomt_precession import nnlo_final_spin, nnlo_geometry


def prepare_precession(method, m1, m2, x1, y1, z1, x2, y2, z2, f_ref,
                       *, final_spin=None, prepared=None):
    """Prepare intrinsic evolution once, including the matching remnant spin."""
    if method == "nnlo":
        fs = nnlo_final_spin(m1, m2, x1, y1, z1, x2, y2, z2) if final_spin is None else final_spin
        return fs, None
    if method == "msa":
        from ._phenomt_msa import msa_final_spin
        fs = msa_final_spin(m1, m2, x1, y1, z1, x2, y2, z2, f_ref) if final_spin is None else final_spin
        return fs, None
    from ._phenomt_numerical import prepare_numerical_trajectory
    if prepared is None:
        prepared = prepare_numerical_trajectory(m1, m2, x1, y1, z1, x2, y2, z2, f_ref)
    return prepared["final_spin"] if final_spin is None else final_spin, prepared


def precession_geometry(*args, precession_method="nnlo", precession_version=10210,
                        prepared=None, **kwargs):
    """Evaluate an explicitly selected prescription on physical times."""
    if precession_version % 100 in (11, 21):
        from .imrphenomt import IMRPhenomT
        from .imrphenom_xas import XASParameterFits
        from diffgw_core.constants import MSUN_SEC
        import torch

        _, _, m1, m2, _, _, z1, _, _, z2, _, _, fr = args
        fr = torch.where(fr > 0, fr, torch.full_like(fr, 20.0))
        carrier = IMRPhenomT()
        kwargs["reference_time"] = carrier.time_at_frequency(fr, m1, m2, z1, z2)
        if precession_version % 100 == 21:
            eta = m1 * m2 / (m1 + m2)**2
            fmeco = XASParameterFits.fit_f_meco(eta, z1, z2) / ((m1 + m2) * MSUN_SEC)
            kwargs["meco_time"] = carrier.time_at_frequency(fmeco, m1, m2, z1, z2)
    if precession_method == "nnlo":
        return nnlo_geometry(*args, precession_version=precession_version, **kwargs)
    if precession_method == "msa":
        from ._phenomt_msa import msa_geometry
        return msa_geometry(*args, precession_version=precession_version, **kwargs)
    from ._phenomt_numerical import numerical_geometry
    return numerical_geometry(*args, prepared=prepared, **kwargs)


def carrier_frequency(model, state, m1, m2, query):
    """Evaluate the carrier with any number of trailing quadrature axes."""
    from diffgw_core.constants import MSUN_SEC
    import math

    flat = query.reshape(m1.shape + (-1,))
    ms = (m1 + m2)[..., None] * MSUN_SEC
    frequency = model._evaluate(flat / ms, state)[0] / (2 * math.pi * ms)
    return frequency.reshape(query.shape)

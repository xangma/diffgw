"""Published nonspinning TEOBResumS 22 postmerger component.

This module is deliberately a standalone component.  It implements the
nonspinning fits and the QNM-rescaled amplitude/phase ansatz in Nagar et al.,
arXiv:1806.01772v2, Appendix ``NR-informed description of merger and
postmerger`` (Eqs. (amp_temp)--(QNM_g) and Tables ``merger_fit``, ``QNM_fits``
and ``rng``).  The NQC extraction-point fits are included from the same
Appendix (Eqs. (amp_zero_NQC)--(df_orb_NQC)).

The final mass and spin are explicit inputs because the paper delegates them
to the separate Jiménez-Forteza et al. fit, and because this component does
not decide how an EOB trajectory finds its merger or NQC anchor.  In
particular, this module does not attach itself to a trajectory and does not
apply an NQC factor.  Those are integration layers, not hidden defaults.

The returned waveform uses the paper's convention
``hbar = exp((alpha + i omega) tau + i phase0) h22 / nu``.  Thus the
positive-frequency phase decreases after merger, as in the TEOBResumS
factorized mode convention.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch


def _tensor(value: torch.Tensor | float, like: torch.Tensor) -> torch.Tensor:
    # Supply the destination dtype at construction time.  Materializing a
    # Python coefficient with torch's default float32 and casting afterward
    # permanently loses precision for float64 callers.
    raw = torch.as_tensor(value, device=like.device, dtype=like.dtype)
    if raw.is_complex():
        raise ValueError("TEOBResumS postmerger inputs must be real")
    return raw.to(dtype=like.dtype)


def _broadcast_nu(nu: torch.Tensor | float) -> torch.Tensor:
    out = torch.as_tensor(nu) if isinstance(nu, torch.Tensor) else torch.as_tensor(float(nu))
    if out.is_complex():
        raise ValueError("TEOBResumS postmerger inputs must be real")
    if not out.is_floating_point():
        out = out.to(dtype=torch.get_default_dtype())
    return out


def _validate_nu(nu: torch.Tensor) -> torch.Tensor:
    if torch.any(~torch.isfinite(nu)) or torch.any((nu <= 0) | (nu > 0.25)):
        raise ValueError("symmetric mass ratio nu must satisfy 0 < nu <= 1/4")
    return nu


def _validate_spin(chi: torch.Tensor) -> torch.Tensor:
    if torch.any(~torch.isfinite(chi)) or torch.any(torch.abs(chi) >= 1):
        raise ValueError("final spin must satisfy |chi_final| < 1")
    return chi


def merger_amplitude(nu: torch.Tensor | float) -> torch.Tensor:
    """Return the published nonspinning ``Ahat_22^mrg`` fit."""
    nu = _validate_nu(_broadcast_nu(nu))
    return _tensor(1.43842, nu) + _tensor(0.100709, nu) * nu + _tensor(1.82657, nu) * nu.square()


def merger_frequency(nu: torch.Tensor | float) -> torch.Tensor:
    """Return the published nonspinning ``omega_22^mrg`` fit in ``1/M``."""
    nu = _validate_nu(_broadcast_nu(nu))
    return _tensor(0.273813, nu) + _tensor(0.223977, nu) * nu + _tensor(0.481959, nu) * nu.square()


def qnm_fits(chi_final: torch.Tensor | float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(omega_1, alpha_1, alpha_21)`` from Table ``QNM_fits``.

    ``chi_final`` is the dimensionless remnant spin.  The fits are the
    rational functions in Eq. (QNM_g), with the coefficients printed in the
    paper.  The values are dimensionless and rescaled by the final mass.
    """
    chi = _validate_spin(_broadcast_nu(chi_final))

    def rational(coeffs: Tuple[float, ...]) -> torch.Tensor:
        y0, b1, b2, b3, c1, c2, c3 = (_tensor(x, chi) for x in coeffs)
        numerator = 1.0 + b1 * chi + b2 * chi.square() + b3 * chi.pow(3)
        denominator = 1.0 + c1 * chi + c2 * chi.square() + c3 * chi.pow(3)
        return y0 * numerator / denominator

    omega = rational((0.373672, -1.74085, 0.808214, -0.0598838,
                      -2.07641, 1.31524, -0.235896))
    alpha = rational((0.0889623, -1.82261, 0.701584, 0.121126,
                      -1.80020, 0.720117, 0.0811633))
    alpha21 = rational((0.184953, -1.41681, -0.0593166, 0.476420,
                        -1.35955, -0.0763529, 0.438558))
    return omega, alpha, alpha21


def nonspinning_ringdown_shape(nu: torch.Tensor | float) -> Dict[str, torch.Tensor]:
    """Return the three nonspinning global ringdown-shape fits.

    These are the zero-spin branch of Table ``rng``; ``X_AB`` and all spin
    terms vanish.  The returned values are ``c3A``, ``c3phi`` and ``c4phi``.
    """
    nu = _validate_nu(_broadcast_nu(nu))
    return {
        "c3A": _tensor(-0.561584, nu) + _tensor(0.829868, nu) * nu,
        "c3phi": _tensor(3.88838, nu) + _tensor(0.455847, nu) * nu,
        "c4phi": _tensor(1.49969, nu) + _tensor(2.08223, nu) * nu,
    }


def nqc_targets(nu: torch.Tensor | float) -> Dict[str, torch.Tensor]:
    """Return published nonspinning NQC extraction-point targets.

    The amplitude target is the Regge--Wheeler normalized ``|Psi_22|``.  The
    derivative fits follow Eqs. (fit_derv_amp)--(df_orb_NQC) exactly: the
    amplitude derivative is ``omega * dot_A_orb`` and the frequency derivative
    is ``dot_omega_orb``.  These are targets at ``t_mrg + 2M``.
    """
    nu = _validate_nu(_broadcast_nu(nu))
    amp = (_tensor(0.294888, nu) - _tensor(0.0427442, nu) * nu
           + _tensor(0.816756, nu) * nu.square()
           - _tensor(0.986204, nu) * nu.pow(3))
    amp_rate = -((_tensor(-0.00421428, nu) - _tensor(0.0847947, nu) * nu)
                  / (1.0 + _tensor(16.1559, nu) * nu))
    omega = (_tensor(0.286399, nu) + _tensor(0.251240, nu) * nu
             + _tensor(0.542717, nu) * nu.square())
    omega_rate = ((_tensor(0.00649349, nu) + _tensor(0.00452138, nu) * nu)
                  / (1.0 - _tensor(1.44664, nu) * nu))
    return {
        "amplitude": amp,
        "amplitude_rate": omega * amp_rate,
        "frequency": omega,
        "frequency_rate": omega_rate,
    }


def postmerger_22(
    time: torch.Tensor,
    nu: torch.Tensor | float,
    final_mass: torch.Tensor | float,
    final_spin: torch.Tensor | float,
    phase0: torch.Tensor | float = 0.0,
    merger_time: torch.Tensor | float = 0.0,
    merger_frequency_value: torch.Tensor | float | None = None,
) -> torch.Tensor:
    """Evaluate the published nonspinning TEOBResumS postmerger 22 mode.

    ``time`` is in units of the initial total mass ``M``; ``final_mass`` is
    ``M_BH/M`` and ``merger_frequency_value`` is ``M*omega_22^mrg``.  If the
    latter is omitted, the published nonspinning merger-frequency fit is used.
    The mode is normalized as ``h_22/M = nu * ...`` and is returned as a
    complex tensor with the phase convention documented above.
    """
    t = torch.as_tensor(time)
    if t.ndim == 0:
        raise ValueError("time must have a sample axis on the last dimension")
    if t.is_complex():
        raise ValueError("time must be real")
    if not t.is_floating_point():
        t = t.to(dtype=torch.get_default_dtype())

    raw = {
        "nu": _tensor(nu, t),
        "final_mass": _tensor(final_mass, t),
        "final_spin": _tensor(final_spin, t),
        "phase0": _tensor(phase0, t),
        "merger_time": _tensor(merger_time, t),
    }
    if merger_frequency_value is not None:
        raw["merger_frequency"] = _tensor(merger_frequency_value, t)
    batch_shape = t.shape[:-1]
    for value in raw.values():
        if value.ndim:
            batch_shape = torch.broadcast_shapes(batch_shape, value.shape)
    sample_count = t.shape[-1]
    t = t.expand(batch_shape + (sample_count,))

    def parameter(name: str) -> torch.Tensor:
        value = raw[name]
        return value.expand(batch_shape).unsqueeze(-1)

    nu = _validate_nu(parameter("nu"))
    mf = parameter("final_mass")
    chi = _validate_spin(parameter("final_spin"))
    phase = parameter("phase0")
    tm = parameter("merger_time")
    if torch.any(~torch.isfinite(mf)) or torch.any(mf <= 0):
        raise ValueError("final_mass must be positive")
    if torch.any(~torch.isfinite(phase)):
        raise ValueError("phase0 must be finite")
    if torch.any(~torch.isfinite(tm)) or torch.any(~torch.isfinite(t)):
        raise ValueError("time and merger_time must be finite")
    if torch.any(t < tm):
        raise ValueError("postmerger_22 requires time >= merger_time")
    if merger_frequency_value is None:
        wmrg = merger_frequency(nu)
    else:
        wmrg = parameter("merger_frequency")
        if torch.any(~torch.isfinite(wmrg)) or torch.any(wmrg <= 0):
            raise ValueError("merger_frequency_value must be finite and positive")
    omega1, alpha1, alpha21 = qnm_fits(chi)
    shape = nonspinning_ringdown_shape(nu)
    a_mrg = merger_amplitude(nu)

    c2a = 0.5 * alpha21
    c1a = a_mrg * alpha1 * torch.cosh(shape["c3A"]).square() / c2a
    c4a = a_mrg - c1a * torch.tanh(shape["c3A"])
    c2p = alpha21
    delta_omega = omega1 - mf * wmrg
    c1p = delta_omega * (1.0 + shape["c3phi"] + shape["c4phi"]) / (
        c2p * (shape["c3phi"] + 2.0 * shape["c4phi"]))

    tau = (t - tm) / mf
    # Appendix F.1 uses c2A=alpha21/2 for the amplitude and
    # c2phi=alpha21 for the logarithmic phase fit.
    exp_phase_tau = torch.exp(-c2p * tau)
    abar = c1a * torch.tanh(c2a * tau + shape["c3A"]) + c4a
    phase_num = (1.0 + shape["c3phi"] * exp_phase_tau
                 + shape["c4phi"] * exp_phase_tau.square())
    phase_den = 1.0 + shape["c3phi"] + shape["c4phi"]
    phibar = -c1p * torch.log(phase_num / phase_den)
    sigma_phase = omega1 * tau
    sigma_decay = alpha1 * tau
    return nu * abar * torch.exp(-sigma_decay) * torch.exp(1j * (phibar - sigma_phase - phase))


__all__ = [
    "merger_amplitude",
    "merger_frequency",
    "qnm_fits",
    "nonspinning_ringdown_shape",
    "nqc_targets",
    "postmerger_22",
]

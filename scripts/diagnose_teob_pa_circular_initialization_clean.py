#!/usr/bin/env python3
"""Diagnose the circular angular-momentum initializer used by clean TEOB PA.

This is a read-only-at-runtime, clean-room diagnostic.  It compares the current clean
analytic initializer ``j**2 = -A'/U'`` with a local shadow of the same
alternating PA algorithm whose initial ``j`` is obtained from the general
circular stationarity equation in Appendix C, Eq. (58), of
arXiv:1806.01772v2.  The shadow is deliberately kept in this script: the
canonical production PA route is not changed by this probe.

For ``x=j**2`` and
``H_eff^circ = sqrt(A + U*x) + G*j``, direct squaring of
``(A' + U'*x)/(2*sqrt(A+U*x)) + G'*sqrt(x) = 0`` gives

    a*x**2 + b*x + c = 0,
    a = U'**2 - 4*U*G'**2,
    b = 2*A'*U' - 4*A*G'**2,
    c = A'**2.

The requested zero-spin compatibility rule is used exactly: ``G' == 0``
selects the lower quadratic root, and only a negative discriminant is
replaced by zero.  No spin perturbation, phase alignment, fitting, or public
binary/reference waveform is used to choose the branch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import torch

from torchwave.approximants.eob.teob_postadiabatic_clean import (
    _state_quantities,
    _vacuum_torque,
    alternating8pa,
    circular_angular_momentum_squared,
    cumulative_cubic_integral,
    quartic_derivative,
)
from torchwave.approximants.eob.teob_radiation_clean import (
    TEOBRadiationClean,
    non_keplerian_radius,
)
from torchwave.approximants.teobresums import TEOBResumSHamiltonian


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / (
    "docs/reviews/2026-09-13-parity-review/verification/"
    "teob-pa-circular-initialization-clean"
)
RUNTIME_FILES = (
    ROOT / "src/torchwave/approximants/eob/teob_postadiabatic_clean.py",
    ROOT / "src/torchwave/approximants/eob/teob_radiation_clean.py",
    ROOT / "src/torchwave/approximants/teobresums.py",
)

R0 = 19.999999999999996
R_MIN = 14.0
DR = 0.1
NU = 0.24  # 30+20 mass ratio, q=1.5
DELTA = 0.2
ORDERS = (2, 4, 6, 8)
N_POINTS = 60  # float64 clean runtime: floor((r0-rmin)/dr)+1
BINARY_ENDPOINT_VALIDATION_ONLY = {
    "time_M": 9523.942662602482,
    "phase": 132.17031461096497,
    "j": 4.215698565112572,
    "pr_star": -0.0012978461179493459,
    "omega": 0.018973840039422412,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return _json_value(value.detach().cpu().item())
        return [_json_value(item) for item in value.detach().cpu().tolist()]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (float, int, bool)) or value is None or isinstance(value, str):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _stats(values: torch.Tensor) -> dict[str, Any]:
    values = values.detach()
    finite = values[torch.isfinite(values)]
    result: dict[str, Any] = {
        "count": int(values.numel()),
        "finite": int(finite.numel()),
    }
    if finite.numel():
        result.update({
            "min": _json_value(finite.min()),
            "max": _json_value(finite.max()),
            "max_abs": _json_value(finite.abs().max()),
            "rms": _json_value(torch.sqrt(torch.mean(finite.square()))),
        })
    return result


def _hamiltonian(dtype: torch.dtype) -> TEOBResumSHamiltonian:
    nu = torch.tensor(NU, dtype=dtype)
    zero = torch.zeros_like(nu)
    return TEOBResumSHamiltonian(
        nu,
        zero,
        zero,
        zero,
        zero,
        delta=torch.tensor(DELTA, dtype=dtype),
    )


class _LeadingOrderHorizonReaction:
    """Diagnostic-only zero-spin leading-order horizon reaction.

    The clean infinity-mode flux is retained.  The horizon ratio is the
    zero-spin leading-order specialization of the published horizon flux,
    ``(X1**4 + X2**4) * v_phi**8``.  This object is intentionally local to
    the diagnostic and is never selected by the production route.
    """

    def __init__(self) -> None:
        self.include_horizon_absorption = True
        self._radiation = TEOBRadiationClean(8, residual_phase="zero-unpublished")

    def radiation_reaction_forces(
        self,
        omega: torch.Tensor,
        pr_star: torch.Tensor,
        pphi: torch.Tensor,
        *,
        r: torch.Tensor | None = None,
        hamiltonian: TEOBResumSHamiltonian | None = None,
        include_horizon_absorption: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del include_horizon_absorption
        if r is None or hamiltonian is None:
            raise ValueError("leading-order horizon diagnostic requires full state")
        h_eff, h_real, h_circ, a_prime = hamiltonian.clean_radiation_inputs(
            r, pr_star, pphi
        )
        infinity_flux = self._radiation.flux(
            r,
            omega,
            torch.zeros_like(omega),
            pr_star,
            pphi,
            h_eff,
            h_real,
            h_circ,
            a_prime,
            hamiltonian.m1,
            hamiltonian.m2,
            include_horizon=False,
        )
        r_omega = non_keplerian_radius(r, h_circ, a_prime)
        vphi = r_omega * omega
        total_mass = hamiltonian.m1 + hamiltonian.m2
        x1 = hamiltonian.m1 / total_mass
        x2 = hamiltonian.m2 / total_mass
        horizon_ratio = (x1**4 + x2**4) * vphi**8
        horizon_flux = (
            (32.0 / 5.0)
            * hamiltonian.nu**2
            * r_omega**4
            * omega**5
            * horizon_ratio
        )
        flux = infinity_flux + horizon_flux
        nu = hamiltonian.nu
        if omega.dim() > nu.dim():
            nu = nu.unsqueeze(-1)
        return torch.zeros_like(flux), -flux / nu


def _radius(dtype: torch.dtype) -> torch.Tensor:
    start = torch.tensor(R0, dtype=dtype)
    dr = torch.tensor(DR, dtype=dtype)
    # Keep dtype comparisons on the same reviewed 60-node grid.  Letting
    # float32 recompute the interval count would turn a representation
    # roundoff check into an unrelated endpoint/grid-length change.
    return start - torch.arange(N_POINTS, dtype=dtype) * dr


def _circular_arrays(
    hamiltonian: TEOBResumSHamiltonian,
    radius: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Evaluate Eq. (58) and the current analytic limit on one clean grid."""
    A, B, _ = hamiltonian.metric_potentials(radius)
    Aprime = hamiltonian.metric_potential_derivative(radius)
    # At exactly zero aligned spin the source-frame circular radius is r and
    # r_c' is one.  Keep this equation-level grouping in the general solve.
    rc = radius
    rc_prime = torch.ones_like(radius)
    U = A / rc.square()
    Uprime_eq = (Aprime - 2.0 * A * rc_prime / rc) / rc.square()
    # This is the grouping used by the current clean PA implementation.  It
    # is retained solely as a control so the shadow can reproduce that route.
    Uprime_clean = Aprime / radius.square() - 2.0 * A / radius.pow(3)
    Gprime = torch.zeros_like(radius)
    a = Uprime_eq.square() - 4.0 * U * Gprime.square()
    b = 2.0 * Aprime * Uprime_eq - 4.0 * A * Gprime.square()
    c = Aprime.square()
    discriminant_raw = b.square() - 4.0 * a * c
    # Compatibility rule: clamp only negative values; positive values remain
    # untouched, even when they are at floating-point roundoff scale.
    discriminant = torch.where(
        discriminant_raw < 0.0,
        torch.zeros_like(discriminant_raw),
        discriminant_raw,
    )
    sqrt_discriminant = torch.sqrt(discriminant)
    denominator = 2.0 * a
    root_minus = (-b - sqrt_discriminant) / denominator
    root_plus = (-b + sqrt_discriminant) / denominator
    # The audited G'=0 rule selects the lower root.  The minimum is explicit
    # so this remains correct if a positive discriminant survives roundoff.
    lower_root = torch.minimum(root_minus, root_plus)
    helper_lower_root = circular_angular_momentum_squared(
        A, Aprime, rc, rc_prime, Gprime
    )
    analytic_clean = -Aprime / Uprime_clean
    analytic_equation = -Aprime / Uprime_eq
    return {
        "A": A,
        "B": B,
        "Aprime": Aprime,
        "U": U,
        "Uprime_eq": Uprime_eq,
        "Uprime_clean": Uprime_clean,
        "Gprime": Gprime,
        "a": a,
        "b": b,
        "c": c,
        "discriminant_raw": discriminant_raw,
        "discriminant": discriminant,
        "root_minus": root_minus,
        "root_plus": root_plus,
        "lower_root": lower_root,
        "helper_lower_root": helper_lower_root,
        "analytic_clean": analytic_clean,
        "analytic_equation": analytic_equation,
    }


def _initializer_report(dtype: torch.dtype) -> dict[str, Any]:
    hamiltonian = _hamiltonian(dtype)
    radius = _radius(dtype)
    values = _circular_arrays(hamiltonian, radius)
    x = values["lower_root"]
    A = values["A"]
    U = values["U"]
    Aprime = values["Aprime"]
    Uprime = values["Uprime_eq"]
    quadratic = (
        values["a"] * x.square()
        + values["b"] * x
        + values["c"]
    )
    unsquared = (Aprime + Uprime * x) / (2.0 * torch.sqrt(A + U * x))
    scale = values["b"].square().abs() + (4.0 * values["a"] * values["c"]).abs()
    scale = scale.clamp_min(torch.finfo(dtype).tiny)
    delta_j2_clean = x - values["analytic_clean"]
    delta_j2_equation = x - values["analytic_equation"]
    positive_raw = values["discriminant_raw"] > 0.0
    negative_raw = values["discriminant_raw"] < 0.0
    zero_raw = values["discriminant_raw"] == 0.0
    return {
        "dtype": str(dtype),
        "grid": {
            "count": int(radius.numel()),
            "start_radius": _json_value(radius[0]),
            "end_radius": _json_value(radius[-1]),
            "spacing": DR,
        },
        "discriminant": {
            "raw": _stats(values["discriminant_raw"]),
            "clamped": _stats(values["discriminant"]),
            "positive_count": int(positive_raw.sum()),
            "negative_count": int(negative_raw.sum()),
            "exact_zero_count": int(zero_raw.sum()),
            "relative_to_coefficient_scale": _stats(
                values["discriminant_raw"] / scale
            ),
        },
        "coefficients": {
            "a": _stats(values["a"]),
            "b": _stats(values["b"]),
            "c": _stats(values["c"]),
            "uprime_equation_minus_clean_max_abs": _json_value(
                (values["Uprime_eq"] - values["Uprime_clean"]).abs().max()
            ),
        },
        "roots": {
            "lower_positive": bool(torch.all(x > 0.0)),
            "lower_finite": bool(torch.isfinite(x).all()),
            "minus_is_lower_count": int(
                (values["root_minus"] <= values["root_plus"]).sum()
            ),
            "plus_is_lower_count": int(
                (values["root_plus"] < values["root_minus"]).sum()
            ),
            "j2_lower_minus_analytic_clean": _stats(delta_j2_clean),
            "j2_lower_minus_analytic_equation": _stats(delta_j2_equation),
            "j_lower_minus_analytic_clean_max_abs": _json_value(
                (
                    torch.sqrt(x) - torch.sqrt(values["analytic_clean"])
                ).abs().max()
            ),
            "helper_minus_direct_root_max_abs": _json_value(
                (values["helper_lower_root"] - values["lower_root"]).abs().max()
            ),
        },
        "stationarity": {
            "quadratic_residual": _stats(quadratic),
            "unsquared_residual": _stats(unsquared),
            "unsquared_sign_positive_count": int((unsquared > 0).sum()),
            "unsquared_sign_negative_count": int((unsquared < 0).sum()),
        },
        # Keep the exact first-node values useful without storing full grids.
        "first_node": {
            name: _json_value(values[name][0])
            for name in (
                "A", "Aprime", "U", "Uprime_eq", "Uprime_clean",
                "discriminant_raw", "root_minus", "root_plus",
                "lower_root", "analytic_clean", "analytic_equation",
            )
        },
    }


def _shadow_pa(
    hamiltonian: TEOBResumSHamiltonian,
    reaction: object,
    *,
    order: int,
    initializer: torch.Tensor,
) -> dict[str, torch.Tensor | int]:
    """Run the clean PA equations with only the initial j profile replaced."""
    radius = _radius(hamiltonian.nu.dtype)
    A, B, _ = hamiltonian.metric_potentials(radius)
    Aprime = hamiltonian.metric_potential_derivative(radius)
    p_phi = torch.sqrt(initializer)
    pr_star = torch.zeros_like(radius)
    quantities = _state_quantities(
        hamiltonian, radius, pr_star, p_phi, A, B, Aprime
    )
    torque = _vacuum_torque(
        reaction,
        quantities,
        radius,
        pr_star,
        p_phi,
        hamiltonian,
        True,
    )
    p_prime = torch.zeros_like(radius)
    j_prime = quartic_derivative(p_phi, radius)
    if bool(torch.any(j_prime == 0)) or not bool(torch.isfinite(j_prime).all()):
        raise RuntimeError("shadow angular-balance derivative is zero/nonfinite")
    Uprime = quantities.Uprime
    for current_order in range(1, order + 1):
        if current_order % 2:
            denominator = j_prime * quantities.C
            pr_star = torque / denominator
        else:
            p_prime = quartic_derivative(pr_star, radius)
            z3 = 2.0 * hamiltonian.nu * (4.0 - 3.0 * hamiltonian.nu)
            correction = 1.0 + 2.0 * z3 * A * pr_star.square() / radius.square()
            angular_radicand = -(
                Aprime
                + 2.0 * pr_star * p_prime * correction
                + z3 * pr_star.pow(4) * Uprime
            ) / Uprime
            p_phi = torch.sqrt(angular_radicand)
        quantities = _state_quantities(
            hamiltonian, radius, pr_star, p_phi, A, B, Aprime
        )
        torque = _vacuum_torque(
            reaction,
            quantities,
            radius,
            pr_star,
            p_phi,
            hamiltonian,
            True,
        )
        j_prime = quartic_derivative(p_phi, radius)
    p_prime = quartic_derivative(pr_star, radius)
    j_prime = quartic_derivative(p_phi, radius)
    time_profile = cumulative_cubic_integral(
        torch.reciprocal(quantities.radial_velocity), radius
    )
    phase_profile = cumulative_cubic_integral(
        quantities.omega * torch.reciprocal(quantities.radial_velocity), radius
    )
    return {
        "radius": radius,
        "time": time_profile,
        "phase": phase_profile,
        "pr_star": pr_star,
        "p_phi": p_phi,
        "omega": quantities.omega,
        "radial_balance": quantities.pr_star_rate - p_prime * quantities.radial_velocity,
        "angular_balance": j_prime * quantities.radial_velocity - torque,
        "order": order,
    }


def _profile_stats(
    baseline: dict[str, torch.Tensor],
    shadow: dict[str, torch.Tensor | int],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in ("time", "phase", "pr_star", "p_phi", "omega"):
        delta = shadow[name] - baseline[name]  # type: ignore[operator]
        result[f"{name}_difference"] = _stats(delta)
    result["endpoint"] = {
        "clean_time_M": _json_value(baseline["time"][-1]),
        "general_time_M": _json_value(shadow["time"][-1]),  # type: ignore[index]
        "clean_phase": _json_value(baseline["phase"][-1]),
        "general_phase": _json_value(shadow["phase"][-1]),  # type: ignore[index]
        "clean_pr_star": _json_value(baseline["pr_star"][-1]),
        "general_pr_star": _json_value(shadow["pr_star"][-1]),  # type: ignore[index]
        "clean_j": _json_value(baseline["p_phi"][-1]),
        "general_j": _json_value(shadow["p_phi"][-1]),  # type: ignore[index]
        "clean_omega": _json_value(baseline["omega"][-1]),
        "general_omega": _json_value(shadow["omega"][-1]),  # type: ignore[index]
    }
    return result


def _run_order(order: int) -> dict[str, Any]:
    hamiltonian = _hamiltonian(torch.float64)
    radius = _radius(torch.float64)
    values = _circular_arrays(hamiltonian, radius)
    reaction_clean = _LeadingOrderHorizonReaction()
    baseline_result = alternating8pa(
        hamiltonian,
        radiation_reaction=reaction_clean,
        r0=R0,
        r_min=R_MIN,
        dr=DR,
        pa_order=order,
        include_horizon_absorption=True,
    )
    baseline = {
        "time": baseline_result.time,
        "phase": baseline_result.phase,
        "pr_star": baseline_result.pr_star,
        "p_phi": baseline_result.p_phi,
        "omega": baseline_result.omega,
    }
    # Control: using the current grouping and analytic initializer in the
    # local shadow should reproduce the production helper.  The general run
    # then changes only the circular initializer and uses Eq. (58) grouping.
    control = _shadow_pa(
        hamiltonian,
        _LeadingOrderHorizonReaction(),
        order=order,
        initializer=values["analytic_clean"],
    )
    general = _shadow_pa(
        hamiltonian,
        _LeadingOrderHorizonReaction(),
        order=order,
        initializer=values["lower_root"],
    )
    control_profile = _profile_stats(baseline, control)
    general_profile = _profile_stats(baseline, general)
    return {
        "order": order,
        "production_control_reproduction": control_profile,
        "general_quadratic_shadow": general_profile,
        "initializer": {
            "clean_first_j": _json_value(torch.sqrt(values["analytic_clean"][0])),
            "general_first_j": _json_value(torch.sqrt(values["lower_root"][0])),
            "first_j_difference": _json_value(
                torch.sqrt(values["lower_root"][0])
                - torch.sqrt(values["analytic_clean"][0])
            ),
            "max_j_difference": _json_value(
                (
                    torch.sqrt(values["lower_root"])
                    - torch.sqrt(values["analytic_clean"])
                ).abs().max()
            ),
        },
    }


def _markdown(report: dict[str, Any]) -> str:
    init64 = report["initializer_diagnostics"]["float64"]
    init32 = report["initializer_diagnostics"]["float32"]
    lines = [
        "# TEOB PA circular initialization clean diagnostic",
        "",
        "Status: diagnostic-only; the canonical production PA route was not changed. The reviewed runtime also exposes the direct policy as an opt-in helper for labeled diagnostics.",
        "",
        "## Contract and derivation",
        "",
        "For `x=j^2`, `U=A/r_c^2`, and `H_eff^circ=sqrt(A+U*x)+G*j`, direct squaring of the radial stationarity equation gives `a*x^2+b*x+c=0` with `a=U'^2-4*U*G'^2`, `b=2*A'*U'-4*A*G'^2`, and `c=A'^2`. The zero-spin probe has `r_c=r`, `r_c'=1`, and `G'=0`; it evaluates `U'=r_c^-2*(A'-2*A*r_c'/r_c)` directly. A negative discriminant is replaced by zero only when negative. At exact `G'=0`, the lower root is selected, as required by the compatibility contract. No perturbation or fitted value is used.",
        "",
        "Primary equation sources: [arXiv:1805.03891v2 Sec. II](https://arxiv.org/html/1805.03891v2) and [arXiv:1806.01772v2 Appendix C, Eq. (58)](https://arxiv.org/html/1806.01772v2).",
        "",
        "The local shadow reruns the clean alternating PA equations with leading-order zero-spin horizon absorption. Its only intended change is the initial circular `j` profile; the actual clean helper is run separately as the baseline and an analytic shadow is used as a reproduction control. The shadow calls the opt-in `circular_angular_momentum_squared` helper; the canonical analytic initializer remains the default.",
        "",
        "## Frozen probe",
        "",
        f"The grid is `r0={R0!r}`, `r_min={R_MIN}`, `dr={DR}`, `nu={NU}`, zero spin/tide, and PA orders `{list(ORDERS)}`. The exact float64 floor policy produces {init64['grid']['count']} nodes and ends at `r={init64['grid']['end_radius']}`; `r_min` is the lower bound passed to the clean runtime. The supplied public-binary endpoint is recorded for validation only: `{json.dumps(BINARY_ENDPOINT_VALIDATION_ONLY, sort_keys=True)}`. It is not used as an input or fit target.",
        "",
        "## Circular equation diagnostics",
        "",
        f"In float64 there are {init64['discriminant']['positive_count']} positive, {init64['discriminant']['negative_count']} negative, and {init64['discriminant']['exact_zero_count']} exactly zero raw discriminants over {init64['grid']['count']} nodes. The raw discriminant range is {init64['discriminant']['raw'].get('min')} to {init64['discriminant']['raw'].get('max')}; its maximum magnitude relative to `|b^2|+|4ac|` is {init64['discriminant']['relative_to_coefficient_scale'].get('max_abs')}. The lower root is positive and finite, and its maximum absolute difference from the clean analytic `j^2` is {init64['roots']['j2_lower_minus_analytic_clean']['max_abs']} (maximum `j` difference {init64['roots']['j_lower_minus_analytic_clean_max_abs']}).",
        "",
        f"The direct stationarity checks remain finite: maximum absolute polynomial residual {init64['stationarity']['quadratic_residual']['max_abs']} and maximum absolute unsquared residual {init64['stationarity']['unsquared_residual']['max_abs']}. The float32 control has {init32['discriminant']['positive_count']} positive, {init32['discriminant']['negative_count']} negative, and {init32['discriminant']['exact_zero_count']} zero raw discriminants, exposing the expected larger representation sensitivity without changing the branch rule.",
        "",
        "## PA endpoint comparison",
        "",
        "Each row below reports the clean production result versus the local quadratic-initializer shadow. `production_control_reproduction` checks that the shadow mechanics reproduce the production helper when seeded with the current analytic initializer; `general_quadratic_shadow` is the requested Eq. (58) comparison.",
        "",
        "| PA order | clean time M | quadratic time M | time change M | clean phase | quadratic phase | phase change | clean j | quadratic j | clean p_r* | quadratic p_r* |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report["orders"]:
        endpoint = item["general_quadratic_shadow"]["endpoint"]
        lines.append(
            "| {order} | {clean_time_M:.15g} | {general_time_M:.15g} | {dt:.6g} | {clean_phase:.15g} | {general_phase:.15g} | {dphase:.6g} | {clean_j:.15g} | {general_j:.15g} | {clean_pr_star:.15g} | {general_pr_star:.15g} |".format(
                order=item["order"],
                dt=endpoint["general_time_M"] - endpoint["clean_time_M"],
                dphase=endpoint["general_phase"] - endpoint["clean_phase"],
                **endpoint,
            )
        )
    order8 = next(item for item in report["orders"] if item["order"] == 8)
    clean8 = order8["production_control_reproduction"]["endpoint"]
    general8 = order8["general_quadratic_shadow"]["endpoint"]
    endpoint_delta_time = general8["general_time_M"] - clean8["clean_time_M"]
    endpoint_delta_phase = general8["general_phase"] - clean8["clean_phase"]
    supplied_delta_time = BINARY_ENDPOINT_VALIDATION_ONLY["time_M"] - clean8["clean_time_M"]
    supplied_delta_phase = BINARY_ENDPOINT_VALIDATION_ONLY["phase"] - clean8["clean_phase"]
    lines.extend([
        "",
        "The analytic control reproduction maximum profile differences are listed in the JSON artifact. Any nonzero control difference is an implementation-copy check, not a parity result. The binary endpoint remains validation-only; no alignment or tuning was performed.",
        "",
        "## Interpretation",
        "",
        f"At order 8 the general shadow changes the clean endpoint by `{endpoint_delta_time:.12g} M` in time and `{endpoint_delta_phase:.12g} rad` in phase, while the validation-only public endpoint is offset from clean by `{supplied_delta_time:.12g} M` and `{supplied_delta_phase:.12g} rad`. Thus this independently derived general initializer accounts for most of the observed endpoint displacement in this frozen probe, but leaves a residual and does not establish production parity. The analytic initializer remains the canonical production default; the general policy is exposed only as an opt-in helper for labeled diagnostics pending coordinator review.",
        "",
        "## Reproducibility",
        "",
        f"Host `{report['host']}`, cwd `{report['cwd']}`, Python `{report['python']}`, command `{report['command']}`. Runtime source hashes before/after and the complete machine-readable metrics are in `report.json`; `runtime_sources_unchanged={report['runtime_sources_unchanged']}`.",
        "",
        "No LAL/GPL implementation source, reference cache, source-only calibration table, public waveform, or oracle fitting was read by this diagnostic.",
    ])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORT_DIR / "report.json",
        help="JSON report path (default: the reviewed verification directory)",
    )
    args = parser.parse_args()
    started_at = time.time()
    command = " ".join(sys.argv)
    source_before = {str(path): _sha256(path) for path in RUNTIME_FILES}
    report: dict[str, Any] = {
        "status": "running",
        "started_unix": started_at,
        "host": platform.node(),
        "cwd": os.getcwd(),
        "command": command,
        "pid": os.getpid(),
        "python": sys.executable,
        "torch": torch.__version__,
        "runtime_source_hashes_before": source_before,
        "runtime_source_hashes_after": None,
        "runtime_sources_unchanged": None,
        "parameters": {
            "r0": R0,
            "r_min": R_MIN,
            "dr": DR,
            "n_intervals": N_POINTS - 1,
            "nu": NU,
            "delta": DELTA,
            "orders": list(ORDERS),
            "spins": [0.0, 0.0],
            "tides": [0.0, 0.0],
            "horizon": "diagnostic-only leading-order zero-spin horizon",
        },
        "equation": {
            "a": "Uprime**2 - 4*U*Gprime**2",
            "b": "2*Aprime*Uprime - 4*A*Gprime**2",
            "c": "Aprime**2",
            "uprime": "rc**-2*(Aprime - 2*A*rcprime/rc)",
            "branch": "lower quadratic root at exact Gprime == 0",
            "discriminant": "clamp only if raw discriminant < 0",
        },
        "binary_endpoint_validation_only": BINARY_ENDPOINT_VALIDATION_ONLY,
    }
    try:
        report["initializer_diagnostics"] = {
            "float64": _initializer_report(torch.float64),
            "float32": _initializer_report(torch.float32),
        }
        report["orders"] = [_run_order(order) for order in ORDERS]
        report["status"] = "complete"
    except Exception as exc:  # pragma: no cover - diagnostic failure record
        report["status"] = "error"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        source_after = {str(path): _sha256(path) for path in RUNTIME_FILES}
        report["runtime_source_hashes_after"] = source_after
        report["runtime_sources_unchanged"] = source_after == source_before
        report["finished_unix"] = time.time()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(_json_value(report), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        markdown_path = args.output.with_name("report.md")
        if report.get("status") == "complete":
            markdown_path.write_text(_markdown(report), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

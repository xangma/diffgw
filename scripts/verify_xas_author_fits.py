"""Opt-in comparison with the pinned external XAS Mathematica fit deposit.

This script reads, parses, and evaluates the author's file at run time. The
file and its formula rows are never installed or copied into DiffGW. Run on
a trusted local copy with ``--asset PATH``; missing/wrong assets fail closed.
"""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import numpy as np
import sympy as sp
from sympy.parsing.mathematica import parse_mathematica


SHA256 = "94b9334c19f3a7909864891ea278ab9a411e4f33b7b7e17cc01d927056d9ac52"
SOURCE = "https://journals.aps.org/prd/supplemental/10.1103/PhysRevD.102.064001/IMRPhenomXAS_Parameter_Space_Fits_Public_Release_Candidate.m"

# A version is an explicit carrier choice, not selected by matching outputs.
FITS = {
    "fit_f_meco": ("fMECOFit", None),
    "fit_final_spin": ("FinalSpin2017", None),
    "fit_erad": ("Erad2017", None),
    "fit_rd_phase_v4": ("RDPhasev4Fit", "105"),
    "fit_rd_phase_d12": ("RDPhased12Fit", "105"),
    "fit_rd_phase_d24": ("RDPhased24Fit", "105"),
    "fit_rd_phase_d34": ("RDPhased34Fit", "105"),
    "fit_rd_phase_d54": ("RDPhased54Fit", "105"),
    "fit_int_phase_v3m_rdv4": ("IntPhasev3mRDv4Fit", "105"),
    "fit_int_phase_v2m_rdv4": ("IntPhasev2mRDv4Fit", "105"),
    "fit_int_phase_v2": ("IntPhasev2Fit", "105"),
    "fit_int_phase_d43": ("IntPhased43Fit", "105"),
    "fit_ins_phase_v3_104": ("InsPhasev3Fit", "104"),
    "fit_ins_phase_d13_104": ("InsPhased13Fit", "104"),
    "fit_ins_phase_d23_104": ("InsPhased23Fit", "104"),
    "fit_ins_phase_d43_104": ("InsPhased43Fit", "104"),
    "fit_rd_amp_v1": ("RDAmpv1Fit", "103"),
    "fit_rd_amp_gamma2": ("RDAmpgamma2Fit", "103"),
    "fit_rd_amp_gamma3": ("RDAmpgamma3Fit", "103"),
    "fit_int_amp_va": ("IntAmpvAFit", "104"),
    "fit_int_amp_v2": ("IntAmpv2Fit", "105"),
    "fit_int_amp_v3": ("IntAmpv3Fit", "105"),
    "fit_ins_amp_r2": ("InsAmpR2Fit", "103"),
    "fit_ins_amp_r3": ("InsAmpR3Fit", "103"),
    "fit_ins_amp_r4": ("InsAmpR4Fit", "103"),
}


def _blocks(source: str) -> dict[str, str]:
    matches = list(re.finditer(
        r"(?m)^([A-Za-z][A-Za-z0-9]*)\[.*?\]\s*:=\s*Module\[", source
    ))
    return {
        match.group(1): source[match.end():matches[i + 1].start() if i + 1 < len(matches) else len(source)]
        for i, match in enumerate(matches)
    }


def _formula(block: str, version: str | None) -> str:
    if re.search(r"\bout\s*=", block) is None:
        spin_assignment = re.search(r"(?m)^\s*S\s*=\s*[^;]+;", block)
        if spin_assignment is None:
            raise ValueError("external remnant fit has no spin assignment")
        return block[spin_assignment.end():].split("];", 1)[0].strip()
    selectors = list(re.finditer(r'"([0-9]+)"\s*,', block))
    if version is None:
        if selectors:
            raise ValueError("unversioned fit unexpectedly has Switch selectors")
        segment = block
    else:
        matches = [(i, item) for i, item in enumerate(selectors)
                   if item.group(1) == version]
        if len(matches) != 1:
            raise ValueError(f"expected one external fit for version {version}")
        i, item = matches[0]
        segment = block[item.end():selectors[i + 1].start() if i + 1 < len(selectors) else len(block)]
    assignment = re.search(r"\bout\s*=", segment)
    if assignment is None:
        raise ValueError(f"external fit version {version} has no out assignment")
    formula = segment[assignment.end():].split(";", 1)[0].strip()
    # Some final Switch branches omit the semicolon after `out = ...`.
    while formula.endswith("]") and formula.count("]") > formula.count("["):
        formula = formula[:-1].rstrip()
    return formula


def _parse(formula: str) -> sp.Expr:
    formula = " ".join(formula.split())
    for old, new in (
        (r"\[Eta]\[Eta]", "eta"), (r"\[Chi]\[Chi]1", "chi1"),
        (r"\[Chi]\[Chi]2", "chi2"), (r"\[Eta]", "eta"),
        (r"\[Chi]1", "chi1"), (r"\[Chi]2", "chi2"),
        (r"\[Delta]", "delta"), (r"\[Pi]", "Pi"),
        (r"\[Chi]", "spin"),
    ):
        formula = formula.replace(old, new)
    formula = re.sub(r"\bS\b", "spin", formula).replace("`", "")
    formula = re.sub(r"([0-9]+(?:\.[0-9]+)?)\*\^([+-]?[0-9]+)",
                     r"(\1*10^\2)", formula)
    return parse_mathematica(formula)


def _spin(kind: str, eta: float, chi1: float, chi2: float) -> float:
    delta = np.sqrt(1 - 4 * eta)
    m1, m2 = (1 + delta) / 2, (1 - delta) / 2
    if kind == "sTotR":
        return (m1 * m1 * chi1 + m2 * m2 * chi2) / (m1 * m1 + m2 * m2)
    if kind == "chiPNhat":
        return (m1 * chi1 + m2 * chi2 - (38 / 113) * (chi1 + chi2) * eta) / (1 - 76 * eta / 113)
    raise ValueError(f"unrecognized author spin parameter {kind}")


def _spin_kind(block: str) -> str:
    match = re.search(r"(?m)^\s*S\s*=\s*([^;]+);", block)
    if match is None:
        raise ValueError("external fit has no S assignment")
    assignment = match.group(1)
    if "chiPNhat" in assignment or r"\[Chi]PNhat" in assignment:
        return "chiPNhat"
    if "sTotR" in assignment:
        return "sTotR"
    if "m1" in assignment and "m2" in assignment:
        return "sTotR"
    raise ValueError("external fit has unknown S assignment")


def compare(asset: Path) -> list[tuple[str, str, float, float, str]]:
    if hashlib.sha256(asset.read_bytes()).hexdigest() != SHA256:
        raise ValueError("external XAS fit asset SHA-256 differs from pinned deposit")
    source = asset.read_text()
    blocks = _blocks(source)
    from torchwave.approximants.imrphenom_xas import XASParameterFits as TorchFits
    from jaxwave.approximants.imrphenom_xas import XASParameterFits as JaxFits
    import torch
    import jax
    import jax.numpy as jnp
    covered_methods = set(FITS) | {"fit_re_omega", "fit_im_omega",
                                   "fit_final_spin_nnlo"}
    for backend, native in (("Torch", TorchFits), ("JAX", JaxFits)):
        present = {name for name in vars(native) if name.startswith("fit_")}
        if present != covered_methods:
            raise ValueError(
                f"{backend} fit coverage drift: missing={sorted(present-covered_methods)}, "
                f"stale={sorted(covered_methods-present)}"
            )
    cases = (
        (1, 0.0, 0.0), (1, 0.6, 0.6), (1, 0.6, -0.6),
        (2, 0.5, -0.3), (3, 0.8, -0.7), (5, 0.6, 0.2),
        (8, -0.7, 0.8), (10, -0.4, -0.6), (20, 0.9, -0.9),
    )
    symbols = sp.symbols("eta chi1 chi2 spin delta m1 m2")
    results = []
    for native_name, (source_name, version) in FITS.items():
        block = blocks[source_name]
        try:
            expression = _parse(_formula(block, version))
        except Exception as exc:
            raise ValueError(f"could not parse {source_name} version {version}") from exc
        unknown = expression.free_symbols - set(symbols)
        if unknown:
            raise ValueError(f"{source_name} has unhandled symbols: {unknown}")
        evaluator = sp.lambdify(symbols, expression, modules="numpy")
        kind = _spin_kind(block)
        max_relative = 0.0
        max_derivative = 0.0
        for q, chi1, chi2 in cases:
            eta = q / (1 + q) ** 2
            def source_value(e, c1, c2, spin_kind=kind):
                delta = np.sqrt(1 - 4 * e)
                m1, m2 = (1 + delta) / 2, (1 - delta) / 2
                return evaluator(e, c1, c2, _spin(spin_kind, e, c1, c2),
                                 delta, m1, m2)

            expected = float(source_value(eta, chi1, chi2))
            args_t = (torch.tensor(eta, dtype=torch.float64),
                      torch.tensor(chi1, dtype=torch.float64),
                      torch.tensor(chi2, dtype=torch.float64))
            args_j = tuple(jnp.asarray(float(x), dtype=jnp.float64) for x in args_t)
            actual = (float(getattr(TorchFits, native_name)(*args_t)),
                      float(getattr(JaxFits, native_name)(*args_j)))
            for value in actual:
                relative = abs(value - expected) / max(1.0, abs(expected))
                max_relative = max(max_relative, relative)
            if native_name == "fit_int_amp_va":
                # This is an identified carrier choice: the author file says
                # chiPNhat, while binary-compatible carriers use sTotR.
                alternate = float(source_value(eta, chi1, chi2, "sTotR"))
                if max(abs(value - alternate) / max(1.0, abs(alternate))
                       for value in actual) > 2e-10:
                    raise AssertionError("carrier midpoint fit differs from its declared sTotR variant")
            if q in (2, 10):
                variant = "sTotR" if native_name == "fit_int_amp_va" else kind
                source_gradient = []
                for axis in range(3):
                    perturbed = [eta, chi1, chi2]
                    perturbed[axis] += 1e-24j
                    source_gradient.append(float(np.imag(source_value(*perturbed, variant)) / 1e-24))
                torch_args = tuple(x.detach().requires_grad_() for x in args_t)
                torch_gradient = torch.autograd.grad(
                    getattr(TorchFits, native_name)(*torch_args), torch_args
                )
                jax_gradient = jax.grad(
                    lambda e, c1, c2: getattr(JaxFits, native_name)(e, c1, c2),
                    argnums=(0, 1, 2),
                )(*args_j)
                for wanted, from_torch, from_jax in zip(
                    source_gradient, torch_gradient, jax_gradient
                ):
                    for value in (float(from_torch), float(from_jax)):
                        max_derivative = max(
                            max_derivative,
                            abs(value - wanted) / max(1.0, abs(wanted)),
                        )
        status = "matches" if max_relative <= 2e-10 and max_derivative <= 2e-8 else "differs"
        if native_name == "fit_int_amp_va" and max_derivative <= 2e-8:
            status = "carrier_variant" if max_relative > 2e-10 else "matches"
        results.append((native_name, kind, max_relative, max_derivative, status))
    for native_name, source_name in (
        ("fit_re_omega", r"re\[Omega]"),
        ("fit_im_omega", r"im\[Omega]"),
    ):
        start = source.index(source_name + "[2,2,")
        formula = source[source.index(":=", start) + 2:source.index(";", start)]
        expression = _parse(formula)
        if expression.free_symbols != {sp.Symbol("spin")}:
            raise ValueError(f"unexpected QNM fit symbols for {source_name}")
        evaluator = sp.lambdify(sp.Symbol("spin"), expression, modules="numpy")
        max_relative = 0.0
        max_derivative = 0.0
        for chi in (-0.95, -0.7, 0.0, 0.6, 0.95):
            expected = float(evaluator(chi))
            actual = (
                float(getattr(TorchFits, native_name)(torch.tensor(chi, dtype=torch.float64))),
                float(getattr(JaxFits, native_name)(jnp.asarray(chi, dtype=jnp.float64))),
            )
            max_relative = max(max_relative, *(abs(value - expected) / max(1.0, abs(expected))
                                               for value in actual))
            wanted = float(np.imag(evaluator(chi + 1e-24j)) / 1e-24)
            torch_arg = torch.tensor(chi, dtype=torch.float64, requires_grad=True)
            from_torch = float(torch.autograd.grad(
                getattr(TorchFits, native_name)(torch_arg), torch_arg
            )[0])
            from_jax = float(jax.grad(
                lambda c: getattr(JaxFits, native_name)(c)
            )(jnp.asarray(chi, dtype=jnp.float64)))
            max_derivative = max(
                max_derivative,
                *(abs(value - wanted) / max(1.0, abs(wanted))
                  for value in (from_torch, from_jax)),
            )
        results.append((native_name, "QNM", max_relative, max_derivative,
                        "matches" if max_relative <= 2e-10 and max_derivative <= 2e-8
                        else "differs"))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path, required=True)
    args = parser.parse_args()
    results = compare(args.asset)
    for name, kind, error, derivative, status in results:
        print(f"{name:32s} {kind:9s} {status:15s} value={error:.3g} derivative={derivative:.3g}")
    return int(any(status not in ("matches", "carrier_variant")
                   for _, _, _, _, status in results))


if __name__ == "__main__":
    raise SystemExit(main())

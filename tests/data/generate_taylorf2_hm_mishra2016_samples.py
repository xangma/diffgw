"""Regenerate TaylorF2_HM component fixtures from Mishra author source.

The input is the pinned arXiv v1 source bundle.  This script reads the
author-published Mathematica supplement directly from the archive, extracts
the lowest-PN term of each requested H[l,m], and retains the supplement's
nonspinning ``TF28Phasing`` expression.  It imports no diffGW implementation
code.  The accompanying paper's Eq. (11) supplies the absolute SPA
normalization and complex-mode composition used by the maintained test.
"""

import argparse
import hashlib
import json
import re
import tarfile
from pathlib import Path

import sympy as sp
from sympy.parsing.mathematica import parse_mathematica


SOURCE_URL = "https://export.arxiv.org/e-print/1601.05588v1"
ARCHIVE_SHA256 = "f0b6c6810da95f45141dbb9f0c5cccec0ba0742f60b836dba7698c90a25d5154"
MEMBER = "supl-mkaf16.m"
MEMBER_SHA256 = "501ea53c29b77e061585540b9ccfd69a4cd0283e805ea1e9c24c38800e627a70"
OUT = Path(__file__).with_name("taylorf2_hm_mishra2016_samples.json")
MODES = ((2, 2), (2, 1), (3, 1), (3, 2), (3, 3), (4, 2), (4, 3), (4, 4))
CASES = (
    ("q2p411", "41", "17", ("29", "53", "89")),
    ("q3", "45", "15", ("24", "61", "110")),
    ("q1p857", "39", "21", ("31", "67", "103")),
)

# CODATA 2018 and IAU 2015 Resolution B3, evaluated independently below.
G_SI = sp.Rational("6.67430e-11")
MSUN_SI = sp.Rational("1.9884098706980507e30")
C_SI = sp.Integer(299792458)
MPC_SI = sp.Rational("3.085677581491367e22")
MSUN_SEC = G_SI * MSUN_SI / C_SI**3


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _read_supplement(archive):
    raw = archive.read_bytes()
    if _sha256(raw) != ARCHIVE_SHA256:
        raise RuntimeError(f"unexpected arXiv source-bundle hash: {archive}")
    with tarfile.open(archive, mode="r:*") as bundle:
        member = bundle.extractfile(MEMBER)
        if member is None:
            raise RuntimeError(f"{MEMBER} is absent from {archive}")
        source = member.read()
    if _sha256(source) != MEMBER_SHA256:
        raise RuntimeError(f"unexpected author-supplement hash: {MEMBER}")
    return source.decode("utf-8")


def _mode_expressions(source):
    matches = re.finditer(
        r"^H\[(\d+),\s*(\d+)\]\s*=\s*(.*?)(?=^H\[|^\"-{10,})",
        source,
        flags=re.MULTILINE | re.DOTALL,
    )
    expressions = {}
    for match in matches:
        mode = tuple(map(int, match.group(1, 2)))
        text = match.group(3).strip()
        text = (
            text.replace(r"\[Delta]", "delta")
            .replace(r"\[Nu]", "eta")
            .replace(r"\[Chi]adl", "chia")
            .replace(r"\[Chi]sdl", "chis")
        )
        text = re.sub(r"Subscript\[V,\s*(\d+)\]", r"V\1", text)
        expressions[mode] = sp.expand(parse_mathematica(text))
    missing = set(MODES) - set(expressions)
    if missing:
        raise RuntimeError(f"author supplement is missing modes: {sorted(missing)}")
    return expressions


def _leading_terms(expressions):
    eta, delta, chia, chis = sp.symbols("eta delta chia chis")
    terms = {}
    for mode in MODES:
        velocity = sp.Symbol(f"V{mode[1]}")
        nonspinning = sp.expand(expressions[mode].subs({chia: 0, chis: 0}))
        polynomial = sp.Poly(nonspinning, velocity)
        powers = [power[0] for power, coefficient in polynomial.terms() if coefficient != 0]
        power = min(powers)
        coefficient = sp.simplify(polynomial.coeff_monomial(velocity**power))
        terms[mode] = (power, coefficient)
    return terms


def _nonspinning_orbital_phasing(source):
    """Extract the author's first-harmonic orbital SPA phase correction."""
    match = re.search(
        r"^TF28Phasing\s*=\s*(.*)\Z",
        source,
        flags=re.MULTILINE | re.DOTALL,
    )
    if match is None:
        raise RuntimeError("author supplement is missing TF28Phasing")
    text = (
        match.group(1)
        .strip()
        .replace(r"\[Eta]", "eta")
        .replace(r"\[Delta]", "delta")
        .replace(r"\[Chi]adl", "chia")
        .replace(r"\[Chi]sdl", "chis")
        .replace(r"\[Gamma]E", "EulerGamma")
    )
    expression = parse_mathematica(text).xreplace({sp.Symbol("EulerGamma"): sp.EulerGamma})
    expression = expression.subs({sp.Symbol("chia"): 0, sp.Symbol("chis"): 0})
    unexpected = expression.free_symbols - {sp.Symbol("eta"), sp.Symbol("v")}
    if unexpected:
        raise RuntimeError(f"unexpected symbols in nonspinning TF28Phasing: {sorted(map(str, unexpected))}")
    return sp.factor(expression)


def _decimal(value):
    return str(sp.N(value, 40))


def _build_fixture(source):
    eta_symbol, delta_symbol = sp.symbols("eta delta")
    leading = _leading_terms(_mode_expressions(source))
    orbital_phasing = _nonspinning_orbital_phasing(source)
    power22, coefficient22 = leading[(2, 2)]
    rows = []
    for name, mass1_text, mass2_text, frequencies in CASES:
        mass1, mass2 = sp.Rational(mass1_text), sp.Rational(mass2_text)
        total_mass = mass1 + mass2
        eta = mass1 * mass2 / total_mass**2
        delta = (mass1 - mass2) / total_mass
        expected = {f"{l}{m}": [] for l, m in MODES}
        for frequency_text in frequencies:
            frequency = sp.Rational(frequency_text)
            v22 = (sp.pi * total_mass * MSUN_SEC * frequency) ** sp.Rational(1, 3)
            h22 = coefficient22.subs({eta_symbol: eta, delta_symbol: delta}) * v22**power22
            for mode in MODES:
                _, m = mode
                power, coefficient = leading[mode]
                vm = (sp.Rational(2, m)) ** sp.Rational(1, 3) * v22
                h_lm = coefficient.subs({eta_symbol: eta, delta_symbol: delta}) * vm**power
                ratio = abs(h_lm / h22) * (vm / v22) ** sp.Rational(-7, 2)
                expected[f"{mode[0]}{mode[1]}"].append(_decimal(ratio))
        rows.append(
            {
                "name": name,
                "mass1_msun": mass1_text,
                "mass2_msun": mass2_text,
                "frequencies_hz": list(frequencies),
                "expected_abs_hlm_over_h22": expected,
            }
        )

    source_terms = {
        f"{l}{m}": {
            "velocity_power": power,
            "coefficient": str(coefficient),
        }
        for (l, m), (power, coefficient) in leading.items()
    }
    return {
        "format": "taylorf2-hm-mishra2016-leading-amplitude-v1",
        "description": (
            "All-eight leading SPA amplitude ratios and normalized nonspinning "
            "phase metadata independently derived from the author-published "
            "Mathematica supplement; no native imports."
        ),
        "source": {
            "paper": "https://arxiv.org/abs/1601.05588v1",
            "bundle_url": SOURCE_URL,
            "bundle_sha256": ARCHIVE_SHA256,
            "member": MEMBER,
            "member_sha256": MEMBER_SHA256,
            "scope": ("lowest nonspinning PN term of each H[l,m] and the nonspinning part of TF28Phasing"),
        },
        "adapter": ("V_m=(2/m)^(1/3)*v22 and |h_lm/h_22|=|H_lm^LO(V_m)|*(V_m/v22)^(-7/2)"),
        "constants": {
            "G_SI": str(G_SI),
            "MSUN_SI": str(MSUN_SI),
            "C_SI": str(C_SI),
            "MPC_SI": str(MPC_SI),
            "MSUN_SEC": _decimal(MSUN_SEC),
        },
        "complex_component": {
            "format": "mishra2016-leading-nonspinning-complex-modes-v1",
            "scope": (
                "unwindowed leading nonspinning raw modes with coalescence-phase "
                "origin; excludes f_ref subtraction, polarization projection, "
                "cutoffs, and spin-amplitude corrections"
            ),
            "paper_equations": ["4", "11", "12a--12h"],
            "tf28_phasing": str(orbital_phasing),
            "inputs": {
                "distance_mpc": "410",
                "phic": "0.23",
                "tc_seconds": "0.0041",
            },
            "author_mode": (
                "M_sec^2/D_sec*pi*sqrt(2*eta/3)*V_m^(-7/2)*H_lm_LO(V_m)*exp(-I*(m*(2*pi*(f/m)*tc-phic+TF28)+pi/4))"
            ),
            "native_over_author_adapter": "exp(I*pi*(m-3)/2)",
            "evidence_class": "symbolic author-source component gate, not external numerical parity",
        },
        "leading_terms": source_terms,
        "samples": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--archive",
        required=True,
        type=Path,
        help="path to the pinned arXiv 1601.05588v1 source bundle",
    )
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    fixture = _build_fixture(_read_supplement(args.archive))
    args.output.write_text(json.dumps(fixture, indent=2) + "\n")


if __name__ == "__main__":
    main()

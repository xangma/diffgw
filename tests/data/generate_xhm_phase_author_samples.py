"""Regenerate xhm_phase_author_samples.json from archived author math only."""

import json
import re
import hashlib
import ast
import os
from decimal import Decimal, getcontext
from pathlib import Path

getcontext().prec = 80
ROOT = Path(__file__).parents[2]
ASSET_ROOT = Path(os.environ.get("TORCHWAVE_EXTERNAL_SOURCE_ASSETS_DIR", Path.home() / ".cache/torchwave/external-validation/source-assets"))
ASSETS = ASSET_ROOT / "pass-14-phase"
OUT = Path(__file__).with_name("xhm_phase_author_samples.json")
RAW_INPUTS = [("0.12", "0.4", "-0.2"), ("0.18", "0.7", "0.1"),
              ("0.24", "-0.6", "0.8"), ("0.25", "0.3", "-0.5")]
EXPECTED_SHA = {
    "21": "5b3ab0e7f04092f99f8983b02f7cb985cf4bfd13b3e4022b16cf020d57987716",
    "33": "7cb4273b6b46f4be4f82b5a48e03705fa0f1d832effd71c79ade845c2d91afe8",
    "44": "6dd15b7cf9fded8c3cdd329c7535d688f3d25633a9508c00ebf7fad2b29d5c8f",
}


def source_expression(mode):
    text = (ASSETS / f"3Dfit_p6_lm_{mode}.m").read_text()
    path = ASSETS / f"3Dfit_p6_lm_{mode}.m"
    if hashlib.sha256(path.read_bytes()).hexdigest() != EXPECTED_SHA[mode]:
        raise RuntimeError(f"unexpected source hash for p6_{mode}")
    expr = re.search(r"^\{\{\{(.*?),\s*\{f", text, re.S | re.M).group(1)
    expr = expr.replace(r"\[Eta]", "eta").replace("*^", "e").replace("^", "**")
    # Turn every literal into a Decimal constructor; this is independent of
    # Torch and of the native implementation's evaluation order.
    return re.sub(
        r"(?<![\w.])(?:\d+\.\d*|\.\d+|\d+)(?:e[+-]?\d+)?",
        lambda m: f"Decimal('{m.group(0)}')",
        re.sub(r"\s+", "", expr),
    )


def make_spin(eta, chi1, chi2):
    delta = (Decimal(1) - Decimal(4) * eta).sqrt()
    m1, m2 = (Decimal(1) + delta) / 2, (Decimal(1) - delta) / 2
    return (m1 * m1 * chi1 + m2 * m2 * chi2) / (Decimal(1) - Decimal(2) * eta)


def main():
    rows = []
    for eta_text, chi1_text, chi2_text in RAW_INPUTS:
        eta, chi1, chi2 = map(Decimal, (eta_text, chi1_text, chi2_text))
        rows.append({"eta": float(eta), "S": float(make_spin(eta, chi1, chi2)),
                     "chi1": float(chi1), "chi2": float(chi2)})
    expected = {}
    for mode in ("21", "33", "44"):
        expr = source_expression(mode)
        tree = ast.parse(expr, mode="eval")
        allowed = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Add, ast.Sub,
                   ast.Mult, ast.Pow, ast.USub, ast.Call, ast.Name, ast.Load,
                   ast.Constant)
        if any(not isinstance(node, allowed) for node in ast.walk(tree)):
            raise RuntimeError(f"non-arithmetic author expression in p6_{mode}")
        fn = eval("lambda eta, S: " + expr, {"Decimal": Decimal, "__builtins__": {}}, {})
        expected[mode] = [float(fn(Decimal(str(row["eta"])),
                                    Decimal(str(row["S"])))) for row in rows]
    files = {}
    for mode in ("21", "33", "44"):
        path = ASSETS / f"3Dfit_p6_lm_{mode}.m"
        files[mode] = {"url": f"https://arxiv.org/src/2001.10914v2/anc/ParspaceFits/PhaseFits/{mode}/3Dfit_p6_lm_{mode}.m",
                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    data = {"description": "FP64 samples evaluated independently from the author Mathematica p6 expressions",
            "inputs": rows, "expected_p6": expected,
            "source": {"paper": "https://arxiv.org/abs/2001.10914",
                        "ancillary_base": "https://arxiv.org/src/2001.10914v2/anc/ParspaceFits/PhaseFits",
                        "files": files,
                        "evaluation": "Decimal(80) arithmetic over the first expanded expression; no network or native imports."}}
    OUT.write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    main()

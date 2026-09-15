"""Black-box diagnostic helper for the published-fit substitution.

The native evaluator is supplied by the caller. This module only inspects its
runtime signature and temporarily replaces the named attribute; it never reads
implementation source. A caller can pass its existing parity_harness case
runner as ``blackbox_runner`` to obtain the frozen-gate metrics.
"""

from __future__ import annotations

import inspect
import json
import hashlib
from pathlib import Path
from contextlib import contextmanager
from typing import Callable, Iterator

from torchwave.phenomd_published_fits import eval_table_v_fit
from torchwave.phenomd_published_fits import PARAMETER_NAMES, TABLE_V_ROWS

_NATIVE_TO_TABLE = {n.replace("_", ""): n for n in PARAMETER_NAMES}


def compare_table_v_tex(path: Path):
    rows, active = [], False
    for line in path.read_text().splitlines():
        if "\\rho_1" in line:
            active = True
        if not active or "&" not in line:
            continue
        fields = [x.strip() for x in line.split("&")]
        if len(fields) != 12:
            continue
        vals = [x.replace("$", "").replace("\\times 10^", "e").replace("\\hline", "").replace("\\", "").replace("{", "").replace("}", "").replace(" ", "") for x in fields[1:]]
        try:
            [float(x) for x in vals]
        except ValueError:
            continue
        rows.append(vals)
        if "\\alpha _5" in fields[0]:
            break
    return {"rows_found": len(rows), "columns_found": len(rows[0]) if rows else 0,
            "numeric_match": len(rows) == 19 and all(all(float(a) == float(b) for a, b in zip(row, TABLE_V_ROWS[i])) for i, row in enumerate(rows))}


@contextmanager
def temporary_fit_substitution(module, attribute: str = "eval_table_v_fit") -> Iterator[None]:
    """Monkeypatch one runtime evaluator attribute, restoring it on exit."""
    descriptor = inspect.getattr_static(module, attribute)
    native = getattr(module, attribute)
    signature = inspect.signature(native)
    required = {"param_name", "eta", "chi_pn"}
    if not required.issubset(signature.parameters):
        raise TypeError(f"{module.__name__}.{attribute} has incompatible signature {signature}")
    setattr(module, attribute, staticmethod(eval_table_v_fit))
    try:
        yield
    finally:
        setattr(module, attribute, descriptor)


def run_substitution_diagnostic(module, blackbox_runner: Callable[[], object], attribute: str = "eval_table_v_fit"):
    """Run a caller-provided canonical/holdout black-box harness under patch."""
    with temporary_fit_substitution(module, attribute):
        return blackbox_runner()


def run_gate_cases(cases, coordinate: str = "unscaled"):
    """Run D gate cases baseline and under a reversible fit substitution."""
    import parity_harness
    from torchwave.approximants.imrphenomd import IMRPhenomD
    native_module = IMRPhenomD

    attribute = "eval_table_v"
    descriptor = inspect.getattr_static(native_module, attribute)
    native = getattr(native_module, attribute)
    signature = inspect.signature(native)
    if set(("name", "eta", "chi_pn")) - set(signature.parameters):
        raise TypeError(f"unexpected native evaluator signature: {signature}")
    baseline = [parity_harness.run_oracle_comparison(c, strict=True) for c in cases]
    calls = {"count": 0, "first_input": None}

    p = cases[0]["params"]

    def replacement(name, eta, chi_pn):
        calls["count"] += 1
        if calls["first_input"] is None:
            e = float(eta.detach().cpu().reshape(-1)[0])
            z = float(chi_pn.detach().cpu().reshape(-1)[0])
            m1, m2, s1, s2 = p["mass1"], p["mass2"], p["chi1"], p["chi2"]
            et = m1 * m2 / (m1 + m2) ** 2
            eff = (m1 * s1 + m2 * s2) / (m1 + m2)
            pn = eff - 38.0 * et / 113.0 * (s1 + s2)
            hat = pn / (1.0 - 76.0 * et / 113.0)
            calls["first_input"] = {"eta": e, "incoming": z, "chiPN": pn, "hat_chi": hat,
                                     "abs_delta_chiPN": abs(z - pn), "abs_delta_hat_chi": abs(z - hat)}
        table_name = _NATIVE_TO_TABLE.get(name, name)
        # Interpret the incoming coordinate by proximity to the independently
        # computed physical coordinates, avoiding accidental double scaling.
        if coordinate == "normalized" and calls["first_input"]["abs_delta_hat_chi"] > calls["first_input"]["abs_delta_chiPN"]:
            chi_pn = chi_pn / (1.0 - 76.0 * eta / 113.0)
        elif coordinate == "unscaled" and calls["first_input"]["abs_delta_chiPN"] > calls["first_input"]["abs_delta_hat_chi"]:
            chi_pn = chi_pn * (1.0 - 76.0 * eta / 113.0)
        return eval_table_v_fit(table_name, eta, chi_pn)

    setattr(native_module, attribute, staticmethod(replacement))
    try:
        substituted = []
        probes = []
        per_case_calls = []
        for case in cases:
            p = case["params"]
            calls["first_input"] = None
            before_count = calls["count"]
            substituted.append(parity_harness.run_oracle_comparison(case, strict=True))
            probes.append(calls["first_input"])
            per_case_calls.append(calls["count"] - before_count)
    finally:
        setattr(native_module, attribute, descriptor)
    return {"coordinate": coordinate, "native_signature": str(signature), "baseline": baseline,
            "substituted": substituted, "replacement_calls": calls["count"], "per_case_replacement_calls": per_case_calls,
            "first_input_coordinate_probe": probes,
            "restored": inspect.getattr_static(native_module, attribute) is descriptor}


def _case_from_result(result):
    out = {k: result[k] for k in ("torch_name", "lal_name", "params", "grid")}
    out["expected_gate"] = result["gate"]
    out["id"] = result["case_id"]
    return out


def main():
    root = Path(__file__).resolve().parents[1]
    catalog = json.loads((root / "docs/reviews/2026-09-13-parity-review/verification/current-gates/catalog.json").read_text())
    holdout = json.loads((root / "docs/reviews/2026-09-13-parity-review/verification/current-gates/holdout.json").read_text())
    selected = [r for r in catalog["results"] + holdout["results"]
                if r["torch_name"] == "IMRPhenomD" and r["case_id"] not in {"IMRPhenomD_diagnostic"}]
    selected.insert(0, next(r for r in catalog["results"] if r["case_id"] == "IMRPhenomD_diagnostic"))
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    tracked = [root / "src/torchwave/phenomd_published_fits.py", Path(__file__), root / "tests/test_phenomd_published_fits.py", Path("/tmp/torchwave-PhenomPaper2.tex"), root / "src/torchwave/approximants/imrphenomd.py", root / "src/torchwave/approximants/phenomd_unrounded_fits.py", root / "docs/reviews/2026-09-13-parity-review/verification/current-gates/catalog.json", root / "docs/reviews/2026-09-13-parity-review/verification/current-gates/holdout.json"]
    hashes_before = {str(x): sha(x) for x in tracked}
    out = {"source": {"path": "/tmp/torchwave-PhenomPaper2.tex", "sha256": hashes_before[str(Path('/tmp/torchwave-PhenomPaper2.tex'))], "table_v_comparison": compare_table_v_tex(Path('/tmp/torchwave-PhenomPaper2.tex'))}, "cases": [r["case_id"] for r in selected],
           "unscaled": run_gate_cases([_case_from_result(r) for r in selected], "unscaled"),
           "normalized": run_gate_cases([_case_from_result(r) for r in selected], "normalized")}
    out["hashes_before"] = hashes_before
    out["hashes_after"] = {str(x): sha(x) for x in tracked}
    (root / "docs/reviews/2026-09-13-parity-review/verification/phenomd-published").mkdir(parents=True, exist_ok=True)
    (root / "docs/reviews/2026-09-13-parity-review/verification/phenomd-published/results.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()


__all__ = ["temporary_fit_substitution", "run_substitution_diagnostic"]

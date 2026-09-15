"""Summarize the unchanged frozen inventories and separate explicit component probes."""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def metrics(result):
    return {key: result.get(key) for key in (
        "plus", "cross", "psd_mismatches_plus", "psd_mismatches_cross", "projections"
    )}


def errors(result):
    return {
        "status": result["status"],
        "max_polarization_complex_relative_l2": max(
            result[p]["complex_relative_l2"] for p in ("plus", "cross")
        ),
        "max_projection_complex_relative_l2": max(
            p["complex_relative_l2"] for p in result["projections"].values()
        ),
    }


summary = {}
for scope in ("catalog", "holdout"):
    before = read(OUT.parent / "pass-10" / f"{scope}.json")
    after = read(OUT / f"{scope}.json")
    old = {r["case_id"]: r for r in before["results"]}
    assert set(old) == {r["case_id"] for r in after["results"]}
    changed = {}
    for result in after["results"]:
        previous = old[result["case_id"]]
        assert result["params"] == previous["params"]
        assert result["grid"] == previous["grid"]
        assert result.get("gate") == previous.get("gate")
        assert result.get("verification_scope") == previous.get("verification_scope")
        if metrics(result) != metrics(previous):
            changed[result["case_id"]] = dict(before=errors(previous), after=errors(result))
    summary[scope] = dict(passed=after["num_passed"], total=after["num_cases"],
                          changed_case_metrics_from_pass10=changed)

before = read(OUT / "components-before.json")
after = read(OUT / "components-after.json")
assert before["cases"] == after["cases"]
assert before["verifier_sha256"] == after["verifier_sha256"]
assert all(c["passed"] for c in before["mode_sum_controls"] + after["mode_sum_controls"])
old = {r["case_id"]: r for r in before["results"]}
summary["supplemental_explicit_components"] = dict(
    before_passed=before["num_passed"], after_passed=after["num_passed"],
    changed_case_metrics=[r["case_id"] for r in after["results"]
                          if metrics(r) != metrics(old[r["case_id"]])],
    total=after["num_cases"], mode_sum_controls_before=before["mode_sum_controls"],
    mode_sum_controls_after=after["mode_sum_controls"],
    cases={r["case_id"]: dict(before=errors(old[r["case_id"]]), after=errors(r))
           for r in after["results"]},
)
run = read(OUT / "verification-run.json")
summary["source_stable_during_verification"] = run["source_stable"]
(OUT / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps({k: {a:b for a,b in v.items() if a != "cases"} if isinstance(v,dict) else v for k,v in summary.items()}, indent=2))

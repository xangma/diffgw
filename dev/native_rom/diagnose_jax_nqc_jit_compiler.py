"""Dev-only CPU compiler-flag control for historical-spline NQC JIT parity."""

from __future__ import annotations

import json
import os

from . import qualify_generator as q
from . import diagnose_jax_nqc_jit_stages as stages


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-25/jax_nqc_jit_compiler_plan.json"
OUTPUT = PLAN.with_name("jax_nqc_jit_compiler.json")


def main() -> int:
    plan = json.loads(PLAN.read_text())
    if (plan["schema"] != "diffgw.seobnrv4.jax-nqc-jit-compiler-plan.v1"
            or plan["release_qualified"] is not False
            or plan["case"] != "high_start_mixed_spin"
            or os.environ.get("XLA_FLAGS") != plan["xla_flags"]):
        raise ValueError("frozen compiler control or XLA_FLAGS changed")
    baseline_path = q.ROOT / plan["baseline_result"]
    if q.sha256(baseline_path) != plan["baseline_sha256"]:
        raise ValueError("pinned baseline NQC evidence changed")
    baseline = json.loads(baseline_path.read_text())
    if (baseline["case"] != plan["case"]
            or baseline["schema"] != "diffgw.seobnrv4.jax-nqc-jit-stages.v1"):
        raise ValueError("baseline NQC stage evidence changed")
    stages.OUTPUT = OUTPUT.with_name("jax_nqc_jit_compiler_stages.json")
    stage_exit = stages.main()
    control = json.loads(stages.OUTPUT.read_text())
    limits = plan["checks"]
    differences = control["differences"]
    checks = {
        "amplitude_coefficients": (
            differences["amplitude_coefficients"]["max_abs"]
            <= limits["max_eager_jit_amplitude_coefficients_absolute_error"]),
        "phase_coefficients": (
            differences["phase_coefficients"]["max_abs"]
            <= limits["max_eager_jit_phase_coefficients_absolute_error"]),
        "base_mode": (differences["base_mode"]["relative_l2"]
                      <= limits["max_eager_jit_base_mode_relative_l2"]),
        "stage_checks": stage_exit == 0 and control["diagnostic_checks_passed"],
    }
    result = {
        "schema": "diffgw.seobnrv4.jax-nqc-jit-compiler.v1",
        "status": "diagnostic_complete", "release_qualified": False,
        "plan_sha256": q.sha256(PLAN), "script_sha256": q.sha256(__file__),
        "baseline_sha256": q.sha256(baseline_path),
        "stage_script_sha256": q.sha256(stages.__file__),
        "stage_result_sha256": q.sha256(stages.OUTPUT),
        "xla_flags": os.environ["XLA_FLAGS"],
        "case": plan["case"], "host": control["host"], "cwd": control["cwd"],
        "pid": control["pid"], "differences": differences,
        "baseline_amplitude_coefficients_max_abs": baseline[
            "differences"]["amplitude_coefficients"]["max_abs"],
        "checks": checks, "diagnostic_checks_passed": all(checks.values()),
        "elapsed_s": control["elapsed_s"],
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"checks": checks, "amplitude_coefficients_max_abs":
                      differences["amplitude_coefficients"]["max_abs"]},
                     sort_keys=True), flush=True)
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())

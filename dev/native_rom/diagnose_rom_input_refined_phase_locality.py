"""Post-hoc phase-locality analysis of the fourth FD sampling level."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from . import qualify_generator as q
from .diagnose_rom_input_phase_locality import _pair


BASE = q.ROOT / "dev/native_rom/evidence/2026-09-25"
OLD_SPECTRA = BASE / "rom_input_stability_spectra.npz"
NEW_SPECTRA = BASE / "rom_input_refined_cadence_spectra.npz"
NEW_REPORT = BASE / "rom_input_refined_cadence.json"
OLD_PLAN = BASE / "rom_input_stability_plan.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(NEW_REPORT.read_text())
    plan = json.loads(OLD_PLAN.read_text())
    if (report["schema"] != "diffgw.seobnrv4.rom-input-refined-cadence.v1"
            or report["spectra_sha256"] != q.sha256(NEW_SPECTRA)
            or report["prior_spectra_sha256"] != q.sha256(OLD_SPECTRA)):
        parser.error("refined or prior spectrum differs")
    result = {
        "schema": "diffgw.seobnrv4.rom-input-refined-phase-locality.v1",
        "status": "post-hoc diagnostic", "release_qualified": False,
        "refined_report_sha256": q.sha256(NEW_REPORT),
        "prior_spectra_sha256": q.sha256(OLD_SPECTRA),
        "refined_spectra_sha256": q.sha256(NEW_SPECTRA),
        "script_sha256": q.sha256(__file__),
        "interpretation": "The broad phase-line fit is descriptive, not a substitute for the required local-gauge numerical gate; both refined-cadence rows failed their predeclared convergence checks.",
        "rows": [],
    }
    with np.load(OLD_SPECTRA) as old, np.load(NEW_SPECTRA) as new:
        frequency = old["frequency_hz"]
        if not np.array_equal(frequency, new["frequency_hz"]):
            parser.error("refined frequency band differs")
        for row in report["rows"]:
            name = row["name"]
            first = old[name + "_ode_2_plus"]
            second = new[name + "_grid_3_plus"]
            if hashlib.sha256(second.tobytes()).hexdigest() != row["plus_sha256"]:
                parser.error(f"{name} refined spectrum hash differs")
            result["rows"].append({
                "name": name,
                "fine_to_refined": _pair(
                    frequency, first, second, plan["reference_frequency_hz"]
                ),
            })
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

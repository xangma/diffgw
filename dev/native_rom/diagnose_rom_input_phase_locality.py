"""Post-hoc phase-locality analysis of the failed ROM-input cadence pilot.

The broad-line fit is descriptive only. It is neither a changed gate nor a
replacement ROM phase/time conditioner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from . import qualify_generator as q
from .rom.conditioning import condition_modes


PILOT = q.ROOT / "dev/native_rom/evidence/2026-09-25/rom_input_stability.json"
SPECTRA = q.ROOT / "dev/native_rom/evidence/2026-09-25/rom_input_stability_spectra.npz"
PAIRS = (("grid_0", "grid_1"), ("grid_1", "ode_2"),
         ("ode_0", "ode_1"), ("ode_1", "ode_2"))


def _pair(frequency: np.ndarray, first: np.ndarray, second: np.ndarray,
          reference_frequency: float) -> dict:
    first_gauge = condition_modes(frequency, first, reference_frequency)
    second_gauge = condition_modes(frequency, second, reference_frequency)
    phase_difference = np.unwrap(np.angle(first)) - np.unwrap(np.angle(second))
    chart = frequency - reference_frequency
    broad_slope, broad_intercept = np.polyfit(chart, phase_difference, 1)
    broad_residual = phase_difference - (broad_intercept + broad_slope * chart)
    conditioned_difference = first_gauge.phase - second_gauge.phase
    local_slope = first_gauge.removed_phase_slope - second_gauge.removed_phase_slope
    return {
        "local_quartic_slope_difference_rad_per_hz": float(local_slope),
        "broad_line_slope_difference_rad_per_hz": float(broad_slope),
        "broad_line_equivalent_time_shift_seconds": float(broad_slope / (2 * math.pi)),
        "broad_line_intercept_difference_rad": float(broad_intercept),
        "raw_phase_difference_range_rad": float(np.ptp(phase_difference)),
        "broad_line_detrended_phase_rms_rad": float(np.sqrt(np.mean(broad_residual**2))),
        "required_local_gauge_phase_rms_rad": float(
            np.sqrt(np.mean(conditioned_difference**2))
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()
    pilot = json.loads(PILOT.read_text())
    if (pilot["schema"] != "diffgw.seobnrv4.rom-input-stability.v1"
            or pilot["spectra_sha256"] != q.sha256(SPECTRA)):
        parser.error("pilot or frozen raw spectra differ")
    plan = json.loads((q.ROOT / "dev/native_rom/evidence/2026-09-25/rom_input_stability_plan.json").read_text())
    result = {
        "schema": "diffgw.seobnrv4.rom-input-phase-locality.v1",
        "status": "post-hoc diagnostic", "release_qualified": False,
        "pilot_sha256": q.sha256(PILOT), "spectra_sha256": q.sha256(SPECTRA),
        "script_sha256": q.sha256(__file__),
        "interpretation": "Broad-line detrending is descriptive and was chosen after the frozen local-gauge cadence gate failed; it does not supersede that failure or qualify a new conditioner.",
        "rows": [],
    }
    with np.load(SPECTRA) as spectra:
        frequency = spectra["frequency_hz"]
        if (frequency.size != pilot["frequency_samples"]
                or frequency[0] != plan["comparison_band_hz"][0]
                or frequency[-1] != plan["comparison_band_hz"][1]):
            parser.error("frequency band differs from the frozen pilot")
        for row in pilot["rows"]:
            arrays = {}
            for level in row["levels"]:
                name = level["level"]
                array = spectra[f"{row['name']}_{name}_plus"]
                if hashlib.sha256(array.tobytes()).hexdigest() != level["plus_sha256"]:
                    parser.error(f"{row['name']}/{name} spectrum hash differs")
                arrays[name] = array
            comparisons = {}
            for first, second in PAIRS:
                comparisons[f"{first}_to_{second}"] = _pair(
                    frequency, arrays[first], arrays[second],
                    plan["reference_frequency_hz"],
                )
            result["rows"].append({"name": row["name"], "pairs": comparisons})
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

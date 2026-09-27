"""Regression checks for the frozen historical-Hamiltonian public-source gate."""

import json
from pathlib import Path
import unittest


class HistoricalHamiltonianSourceGateTest(unittest.TestCase):
    def test_incomplete_public_prescription_remains_pending(self):
        repository = Path(__file__).resolve().parents[4]
        evidence = repository / "dev/native_rom/evidence/2026-09-27"
        plan = json.loads(
            (evidence / "hamiltonian_historical_source_gate_plan.json").read_text()
        )
        audit = json.loads(
            (evidence / "hamiltonian_historical_source_audit.json").read_text()
        )
        result = json.loads(
            (evidence / "hamiltonian_historical_source_gate.json").read_text()
        )

        required = [item["id"] for item in plan["mandatory_requirements"]]
        audited = [item["id"] for item in audit["requirements"]]
        self.assertEqual(audited, required)
        self.assertEqual(result["status"], "diagnostic_complete")
        self.assertEqual(result["component_id"], "V4-HAM-1-3")
        self.assertFalse(result["source_gate_passed"])
        self.assertEqual(result["satisfied_requirement_count"], 4)
        self.assertEqual(result["mandatory_requirement_count"], 6)
        self.assertEqual(
            result["blocking_requirements"],
            [
                "historical_calibration_retained_precision",
                "historical_radial_log_derivative_convention",
            ],
        )
        self.assertFalse(any(result["clean_room"].values()))
        self.assertFalse(result["historical_fidelity_claim"])
        self.assertFalse(result["release_qualified"])


if __name__ == "__main__":
    unittest.main()

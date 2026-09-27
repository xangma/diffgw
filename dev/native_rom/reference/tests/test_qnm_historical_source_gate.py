"""Regression checks for the frozen historical-QNM public-source gate."""

import json
from pathlib import Path
import unittest


class HistoricalQNMSourceGateTest(unittest.TestCase):
    def test_incomplete_public_prescription_remains_pending(self):
        repository = Path(__file__).resolve().parents[4]
        evidence = repository / "dev/native_rom/evidence/2026-09-27"
        plan = json.loads(
            (evidence / "qnm_historical_source_gate_plan.json").read_text()
        )
        audit = json.loads(
            (evidence / "qnm_historical_source_audit.json").read_text()
        )
        result = json.loads(
            (evidence / "qnm_historical_source_gate.json").read_text()
        )

        required = [item["id"] for item in plan["mandatory_requirements"]]
        audited = [item["id"] for item in audit["requirements"]]
        self.assertEqual(audited, required)
        self.assertEqual(result["status"], "diagnostic_complete")
        self.assertEqual(result["component_id"], "V4-QNM-INTERPOLATION")
        self.assertFalse(result["source_gate_passed"])
        self.assertEqual(result["satisfied_requirement_count"], 1)
        self.assertEqual(result["mandatory_requirement_count"], 6)
        self.assertEqual(len(result["blocking_requirements"]), 5)
        self.assertFalse(any(result["clean_room"].values()))
        self.assertFalse(result["historical_fidelity_claim"])
        self.assertFalse(result["release_qualified"])


if __name__ == "__main__":
    unittest.main()

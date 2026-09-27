"""Regression checks for the nine-row flux-selector source closure."""

import json
import unittest
from pathlib import Path


class FluxSelectorSourceClosureTest(unittest.TestCase):
    def test_exact_row_outcomes_remain_frozen(self):
        repository = Path(__file__).resolve().parents[4]
        evidence = repository / "dev/native_rom/evidence/2026-09-27"
        plan = json.loads(
            (evidence / "flux_selector_source_closure_plan.json").read_text()
        )
        result = json.loads(
            (evidence / "flux_selector_source_closure.json").read_text()
        )

        self.assertEqual(result["status"], "diagnostic_complete")
        self.assertEqual(result["component_id"], "V4-FLUX-4-9")
        self.assertEqual(result["row_count"], 9)
        self.assertEqual(result["pass_count"], 5)
        self.assertEqual(result["fail_count"], 4)
        self.assertEqual(
            result["passed_modes"], ["2,1", "3,1", "3,3", "4,1", "4,3"]
        )
        self.assertEqual(result["failed_modes"], ["8,2", "8,4", "8,6", "8,8"])
        self.assertEqual(result["refined_selector_complete_mode_count"], 31)
        self.assertFalse(result["complete_final_mode_selector"])
        self.assertEqual(
            [row["mode"] for row in result["rows"]],
            [
                *plan["rows"]["bfh_pan_coexistence"]["modes"],
                *plan["rows"]["ell8_even_m_truncation"]["modes"],
            ],
        )
        for row in result["rows"]:
            if row["status"] == "pass":
                self.assertTrue(row["selector"])
                self.assertIsNone(row["missing_contract"])
            else:
                self.assertIsNone(row["selector"])
                self.assertTrue(row["missing_contract"])
        self.assertFalse(any(result["clean_room"].values()))
        self.assertFalse(result["historical_fidelity_claim"])
        self.assertFalse(result["production_change"])
        self.assertFalse(result["release_qualified"])


if __name__ == "__main__":
    unittest.main()

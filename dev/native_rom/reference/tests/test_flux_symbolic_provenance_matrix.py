"""Regression checks for the public-equation flux provenance matrix."""

import json
import unittest
from pathlib import Path


class FluxSymbolicProvenanceMatrixTest(unittest.TestCase):
    def test_primary_sources_resolve_only_rho43_placement(self):
        repository = Path(__file__).resolve().parents[4]
        evidence = repository / "dev/native_rom/evidence/2026-09-27"
        plan = json.loads(
            (evidence / "flux_symbolic_provenance_matrix_plan.json").read_text()
        )
        result = json.loads(
            (evidence / "flux_symbolic_provenance_matrix.json").read_text()
        )

        self.assertEqual(result["status"], "diagnostic_complete")
        self.assertEqual(result["component_id"], "V4-FLUX-4-9")
        self.assertEqual(
            result["requirements"],
            {
                "complete_final_mode_selector": False,
                "rho43_higher_spin_placement": True,
            },
        )
        self.assertTrue(
            result["rho43_symbolic_check"][
                "equal_through_published_spin_order_v5"
            ]
        )
        self.assertEqual(
            result["rho43_symbolic_check"]["first_difference_order"], 6
        )
        self.assertEqual(result["mode_count"], 35)
        self.assertEqual(result["selector_complete_mode_count"], 26)
        self.assertEqual(
            set(result["selector_incomplete_modes"]),
            {
                "2,1",
                "3,1",
                "3,3",
                "4,1",
                "4,3",
                "8,2",
                "8,4",
                "8,6",
                "8,8",
            },
        )
        self.assertEqual(
            [row["mode"] for row in result["matrix"]],
            plan["matrix"]["expected_modes"],
        )
        for row in result["matrix"]:
            for field in plan["matrix"]["required_fields"]:
                self.assertTrue(str(row[field]).strip(), f"{row['mode']}:{field}")
        self.assertFalse(any(result["clean_room"].values()))
        self.assertFalse(result["historical_fidelity_claim"])
        self.assertFalse(result["production_change"])
        self.assertFalse(result["release_qualified"])


if __name__ == "__main__":
    unittest.main()

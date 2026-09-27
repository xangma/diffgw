"""Independent exact-algebra checks for the Fujita ell=8 source contract."""

import json
import unittest
from fractions import Fraction
from pathlib import Path


class FluxEll8SourceContractTest(unittest.TestCase):
    def test_all_four_rho_and_f_coefficients_are_exact(self):
        repository = Path(__file__).resolve().parents[4]
        evidence = repository / "dev/native_rom/evidence/2026-09-27"
        result = json.loads(
            (evidence / "flux_selector_source_closure.json").read_text()
        )
        contract = result["ell8_source_contract"]
        expected_v2 = {
            "8,2": Fraction(-1231, 1368),
            "8,4": Fraction(-1333, 1368),
            "8,6": Fraction(-167, 152),
            "8,8": Fraction(-1741, 1368),
        }

        self.assertEqual(contract["factorization"], "f_lm=(rho_lm)^ell")
        self.assertEqual(contract["source_spin_variable"], "q=a/M")
        self.assertEqual(
            contract["archive_member"], "11PN_mathematica/rho8_11PN.m"
        )
        self.assertEqual(
            contract["archive_member_sha256"],
            "185a2e857e45b0f8799a2201253bee32d27b9c15b58dbb54b469424f4804cd9d",
        )
        self.assertEqual(
            [row["mode"] for row in contract["rows"]], list(expected_v2)
        )
        for row in contract["rows"]:
            mode = row["mode"]
            rho_q_v3 = Fraction(row["rho_q_v3"])
            self.assertEqual(Fraction(row["rho_v2"]), expected_v2[mode])
            self.assertEqual(rho_q_v3, Fraction(-2, 3))
            self.assertEqual(Fraction(row["f_q_v3"]), 8 * rho_q_v3)
            self.assertEqual(Fraction(row["f_q_v3"]), Fraction(-16, 3))
            self.assertEqual(Fraction(row["rho_v3_at_q_zero"]), 0)
            self.assertEqual(Fraction(row["f_v3_at_q_zero"]), 0)

    def test_unlicensed_archive_does_not_close_frozen_source_gate(self):
        repository = Path(__file__).resolve().parents[4]
        evidence = repository / "dev/native_rom/evidence/2026-09-27"
        audit = json.loads(
            (evidence / "flux_selector_source_closure_audit.json").read_text()
        )
        source = next(
            item
            for item in audit["sources"]
            if item["id"] == "fujita-11pn-mathematica"
        )

        self.assertEqual(source["archive_license"], "unspecified")
        self.assertFalse(audit["historical_fidelity_claim"])
        self.assertFalse(audit["release_qualified"])


if __name__ == "__main__":
    unittest.main()

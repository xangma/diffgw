"""Independent FP64 checks of the eight archived author amplitude expressions.

Expected values come from the independent author-expression generator. Normal
tests compare native raw fits with this fixture without requiring the ignored
source cache or deriving expected values from native output.
"""
import json
from pathlib import Path
import torch
from torchwave.approximants.xhm_fits import XHMParameterFits

DATA = Path(__file__).parent / "data" / "xhm_amplitude_author_samples.json"


def test_native_raw_fits_match_independent_author_samples():
    fixture = json.loads(DATA.read_text())
    for sample in fixture["samples"]:
        eta, chi1, chi2 = (
            torch.tensor(sample[key], dtype=torch.float64)
            for key in ("eta", "chi1", "chi2")
        )
        delta = torch.sqrt(1 - 4 * eta)
        m1, m2 = (1 + delta) / 2, (1 - delta) / 2
        spin = (m1.square() * chi1 + m2.square() * chi2) / (1 - 2 * eta)
        for key, value in sample["values"].items():
            mode, node = key.split("_")
            actual = getattr(XHMParameterFits, f"amp_{node}_{mode}")(
                eta, chi1, chi2, spin
            )
            expected = torch.tensor(value, dtype=torch.float64)
            torch.testing.assert_close(actual, expected, rtol=2e-12, atol=0)


def test_equal_mass_unequal_spin_odd_modes_can_be_nonzero():
    fixture = json.loads(DATA.read_text())
    sample = next(x for x in fixture["samples"] if x["name"] == "equal_mass_unequal_spin")
    assert any(abs(sample["values"][key]) > 0.0 for key in ("21_int1", "21_int2", "33_int1", "33_int2"))

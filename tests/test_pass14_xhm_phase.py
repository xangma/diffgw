import json
from pathlib import Path

import torch

from torchwave.approximants.xhm_fits import XHMParameterFits


_SAMPLES = json.loads(
    (Path(__file__).parent / "data" / "xhm_phase_author_samples.json").read_text()
)


def test_author_p6_phase_fits_match_independent_math_samples():
    """The three nonmixing p6 expressions reproduce author-math FP64 samples."""
    inputs = _SAMPLES["inputs"]
    eta = torch.tensor([row["eta"] for row in inputs], dtype=torch.float64)
    chi1 = torch.tensor([row["chi1"] for row in inputs], dtype=torch.float64)
    chi2 = torch.tensor([row["chi2"] for row in inputs], dtype=torch.float64)
    spin = torch.tensor([row["S"] for row in inputs], dtype=torch.float64)

    for mode in ("21", "33", "44"):
        expected = torch.tensor(_SAMPLES["expected_p6"][mode], dtype=torch.float64)
        actual = getattr(XHMParameterFits, f"phase_p6_{mode}")(eta, chi1, chi2, spin)
        torch.testing.assert_close(actual, expected, rtol=2e-12, atol=0)

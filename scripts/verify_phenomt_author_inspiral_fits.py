"""Verify PhenomT inspiral fit fixtures against pinned MIT author source.

The author file is supplied separately; its SHA-256 is checked before its two
arithmetic fit functions are evaluated. ``--emit-fixture`` prints a candidate
fixture for review, and the default mode verifies the committed fixture.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from diffgw_reference.phenomt_author_fit_inputs import (
    SOURCE_REVISION,
    SOURCE_SHA256,
    fit_inputs,
)


CASES = (
    (30.0, 30.0, 0.0, 0.0),
    (30.0, 30.0, 0.6, -0.2),
    (35.0, 25.0, 0.7, -0.5),
    (60.0, 6.0, 0.9, 0.1),
    (90.0, 5.0, -0.8, 0.6),
)
DEFAULT_FIXTURE = Path(__file__).resolve().parents[1] / "tests/data/phenomt_author_inspiral_fits_4121.json"


def generate(source_path):
    rows = []
    for case in CASES:
        t0, frequencies = fit_inputs(source_path, *case)
        rows.append({"inputs": list(case), "t0": t0, "frequencies": list(frequencies)})
    return {"source_revision": SOURCE_REVISION, "source_sha256": SOURCE_SHA256, "cases": rows}


def verify(actual, expected):
    if actual["source_revision"] != expected["source_revision"]:
        raise AssertionError("author revision mismatch")
    if actual["source_sha256"] != expected["source_sha256"]:
        raise AssertionError("author SHA-256 mismatch")
    if len(actual["cases"]) != len(expected["cases"]):
        raise AssertionError("case count mismatch")
    for index, (actual_row, expected_row) in enumerate(zip(actual["cases"], expected["cases"])):
        if actual_row["inputs"] != expected_row["inputs"]:
            raise AssertionError(f"case {index}: input grid mismatch")
        for key in ("t0", "frequencies"):
            actual_values = actual_row[key] if key == "frequencies" else [actual_row[key]]
            expected_values = expected_row[key] if key == "frequencies" else [expected_row[key]]
            if len(actual_values) != len(expected_values):
                raise AssertionError(f"case {index}: {key} length mismatch")
            for position, (observed, frozen) in enumerate(zip(actual_values, expected_values)):
                if not math.isclose(observed, frozen, rel_tol=2e-13, abs_tol=2e-13):
                    raise AssertionError(f"case {index}: {key}[{position}] source mismatch")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author-fits", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--emit-fixture", action="store_true")
    args = parser.parse_args()
    generated = generate(args.author_fits)
    if args.emit_fixture:
        print(json.dumps(generated, indent=2) + "\n", end="")
    else:
        verify(generated, json.loads(args.fixture.read_text()))
        print(f"verified {len(CASES)} cases and six fit outputs per case")


if __name__ == "__main__":
    main()

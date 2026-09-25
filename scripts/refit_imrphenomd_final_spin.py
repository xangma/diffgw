#!/usr/bin/env python3
"""Independent refit of the four free final-spin terms in Husa et al. Eq. (22).

The input is the *later* 427-case NR table accompanying Jiménez-Forteza et al.
(arXiv:1611.00332v2), not the original 106-case PhenomD calibration set.
Only the printed coefficients of Husa et al. (arXiv:1508.07250v2) enter the
model. No waveform implementation or LAL package is imported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from dataclasses import dataclass
from pathlib import Path

import numpy as np


DATA_URL = "https://arxiv.org/src/1611.00332v2/anc/FinalStateUIB2016_suppl_NRdata.txt"
DATA_SHA256 = "3a20968a77b2292754404c9cefcaab373f14fc3a47adc1546df1ec442fbea0a3"
PAPER_B = np.array([-0.085, 0.102, -1.355, -0.868], dtype=float)
PAPER_A = np.array([-4.399, 9.397, -13.181], dtype=float)
PAPER_C = np.array([-5.837, -2.097, 4.109, 2.064], dtype=float)
CATALOGS = ("BAM", "GaTech", "RIT", "SXS")


@dataclass(frozen=True)
class NRData:
    eta: np.ndarray
    chi1: np.ndarray
    chi2: np.ndarray
    final_spin: np.ndarray
    weight: np.ndarray
    tag: np.ndarray
    catalog: np.ndarray


def read_nr_table(path: Path, *, expected_sha256: str = DATA_SHA256) -> NRData:
    raw = path.read_bytes()
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(f"NR data SHA256 mismatch: {actual_sha256}")
    lines = raw.decode("utf-8").splitlines()
    if not lines or lines[0].strip() != "# eta chi1 chi2 Erad chif weight  tag code":
        raise ValueError("unexpected NR table header")
    rows = [line.split() for line in lines[1:] if line.strip() and not line.startswith("#")]
    if any(len(row) != 8 for row in rows):
        raise ValueError("NR table must have eight columns")
    numeric = np.asarray([[float(row[i]) for i in range(6)] for row in rows], dtype=float)
    if numeric.shape != (427, 6) or not np.all(np.isfinite(numeric)):
        raise ValueError("expected 427 finite NR cases")
    eta, chi1, chi2, _, final_spin, weight = numeric.T
    tag = np.asarray([row[6] for row in rows])
    catalog = np.asarray([row[7] for row in rows])
    if (np.any((eta <= 0) | (eta > 0.25)) or np.any(abs(chi1) > 1.001)
            or np.any(abs(chi2) > 1.001) or np.any(abs(final_spin) > 1)
            or np.any(weight <= 0) or set(catalog) != set(CATALOGS)):
        raise ValueError("NR table contains invalid physical inputs or catalogs")
    return NRData(eta, chi1, chi2, final_spin, weight, tag, catalog)


def total_spin(eta: np.ndarray, chi1: np.ndarray, chi2: np.ndarray) -> np.ndarray:
    """S=(m1/M)^2 chi1+(m2/M)^2 chi2, with m1 >= m2."""
    eta, chi1, chi2 = np.broadcast_arrays(eta, chi1, chi2)
    if np.any((eta < 0) | (eta > 0.25)):
        raise ValueError("eta must lie in [0, 1/4]")
    delta = np.sqrt(np.maximum(0.0, 1.0 - 4.0 * eta))
    return ((1 + delta) / 2) ** 2 * chi1 + ((1 - delta) / 2) ** 2 * chi2


def free_basis(eta: np.ndarray, spin: np.ndarray) -> np.ndarray:
    """The four free columns eta*S^k, k=1..4, of printed Eq. (22)."""
    return np.column_stack([eta * spin**k for k in range(1, 5)])


def fixed_part(eta: np.ndarray, spin: np.ndarray) -> np.ndarray:
    """Eq. (22) after removing its four eta*S^k terms."""
    return (spin + 2 * np.sqrt(3) * eta
            + sum(PAPER_A[k - 2] * eta**k for k in range(2, 5))
            + eta**2 * sum(PAPER_C[k - 1] * spin**k for k in range(1, 5)))


def predict(eta: np.ndarray, spin: np.ndarray, b: np.ndarray) -> np.ndarray:
    return fixed_part(eta, spin) + free_basis(eta, spin) @ np.asarray(b, dtype=float)


def fit_four(data: NRData, mask: np.ndarray, *, weighted: bool) -> dict:
    spin = total_spin(data.eta, data.chi1, data.chi2)
    basis = free_basis(data.eta, spin)[mask]
    target = (data.final_spin - fixed_part(data.eta, spin))[mask]
    weights = np.sqrt(data.weight[mask]) if weighted else np.ones(len(target))
    coefficients, _, rank, singular_values = np.linalg.lstsq(basis * weights[:, None], target * weights, rcond=None)
    if rank != 4:
        raise ValueError("four-term fit is rank deficient")
    return {"b": coefficients.tolist(), "rank": int(rank),
            "condition_number": float(singular_values[0] / singular_values[-1])}


def fit_b2_only(data: NRData, mask: np.ndarray, *, weighted: bool) -> float:
    """Sensitivity fit with all other coefficients fixed at Eq. (22) digits."""
    spin = total_spin(data.eta, data.chi1, data.chi2)
    column = (data.eta * spin**2)[mask]
    residual = (data.final_spin - predict(data.eta, spin, PAPER_B))[mask]
    weights = data.weight[mask] if weighted else np.ones(len(column))
    denominator = np.dot(weights * column, column)
    if denominator <= 0:
        raise ValueError("b2 is not identifiable from selected cases")
    return float(PAPER_B[1] + np.dot(weights * column, residual) / denominator)


def metrics(truth: np.ndarray, prediction: np.ndarray) -> dict:
    residual = prediction - truth
    return {"n": len(truth), "rmse": float(np.sqrt(np.mean(residual**2))),
            "max_abs": float(np.max(abs(residual))), "mean": float(np.mean(residual))}


def run(data: NRData, *, comparison_b2: float | None = None) -> dict:
    spin = total_spin(data.eta, data.chi1, data.chi2)
    all_cases = np.ones(len(data.eta), dtype=bool)
    printed = predict(data.eta, spin, PAPER_B)
    variants = {}
    for name, weighted in (("uniform", False), ("ancillary_weights", True)):
        fit = fit_four(data, all_cases, weighted=weighted)
        fit["all_cases"] = metrics(data.final_spin, predict(data.eta, spin, fit["b"]))
        fit["paper_all_cases"] = metrics(data.final_spin, printed)
        fit["b2_only_with_other_b_fixed"] = fit_b2_only(data, all_cases, weighted=weighted)
        variants[name] = fit

    folds = {}
    for catalog in CATALOGS:
        holdout = data.catalog == catalog
        train = ~holdout
        fit = fit_four(data, train, weighted=False)
        fold_prediction = predict(data.eta[holdout], spin[holdout], fit["b"])
        folds[catalog] = {"train_n": int(train.sum()), "holdout": metrics(data.final_spin[holdout], fold_prediction),
                          "paper_holdout": metrics(data.final_spin[holdout], printed[holdout]),
                          "train_b": fit["b"]}

    catalog_cv = {
        "refit_rmse": float(np.sqrt(sum(fold["holdout"]["n"] * fold["holdout"]["rmse"]**2
                                        for fold in folds.values()) / len(data.eta))),
        "paper_rmse": float(np.sqrt(sum(fold["paper_holdout"]["n"] * fold["paper_holdout"]["rmse"]**2
                                        for fold in folds.values()) / len(data.eta))),
    }

    report = {
        "protocol_version": "phenomd-final-spin-four-term-2017/v1",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "experiment": "PhenomD Eq. (22) four-free-coefficient refit on later 2017 NR data",
        "paper": "https://arxiv.org/html/1508.07250v2, Section III.1, Eq. (22)",
        "data_paper": "https://arxiv.org/html/1611.00332v2, Section II.1 and ancillary table",
        "data_url": DATA_URL, "data_sha256": DATA_SHA256,
        "sample_counts": {catalog: int(np.sum(data.catalog == catalog)) for catalog in CATALOGS},
        "fixed_coefficients": {"eta_2_to_4": PAPER_A.tolist(), "eta2_spin_1_to_4": PAPER_C.tolist()},
        "paper_b": PAPER_B.tolist(),
        "method": "linear least squares for eta*S^1..4; other Eq. (22) coefficients fixed at printed precision",
        "weight_convention": "ancillary_weights uses sqrt(weight) times each residual; uniform uses unit weights",
        "variants": variants,
        "catalog_holdouts": folds,
        "catalog_cv": catalog_cv,
        "versions": {"python": platform.python_version(), "numpy": np.__version__},
    }
    if comparison_b2 is not None:
        if not np.isfinite(comparison_b2):
            raise ValueError("comparison_b2 must be finite")
        compared = PAPER_B.copy()
        compared[1] = comparison_b2
        report["comparison_b2"] = {
            "value": comparison_b2,
            "all_cases": metrics(data.final_spin, predict(data.eta, spin, compared)),
            "max_prediction_difference_from_printed": float(np.max(abs(predict(data.eta, spin, compared) - printed))),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, help="downloaded ancillary NR table")
    parser.add_argument("--comparison-b2", type=float, help="external coefficient used only after fitting")
    parser.add_argument("--output", type=Path, help="write the JSON report here")
    args = parser.parse_args()
    report = run(read_nr_table(args.data), comparison_b2=args.comparison_b2)
    result = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(result)
    print(result, end="")


if __name__ == "__main__":
    main()

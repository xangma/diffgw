"""Checks for the independent Eq. (22) final-spin calibration experiment."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from refit_imrphenomd_final_spin import (  # noqa: E402
    CATALOGS,
    NRData,
    PAPER_B,
    fit_b2_only,
    fit_four,
    predict,
    read_nr_table,
    run,
    total_spin,
)


def test_total_spin_uses_mass_squared_spin_weights():
    np.testing.assert_allclose(total_spin(np.array([0.25]), np.array([0.8]), np.array([-0.4])), [0.1])
    # q=3 gives m1/M=3/4, m2/M=1/4, eta=3/16.
    np.testing.assert_allclose(total_spin(np.array([3 / 16]), np.array([0.8]), np.array([-0.4])), [0.425])


def test_four_coefficient_fit_recovers_synthetic_truth_and_catalog_holdouts():
    rng = np.random.default_rng(173)
    n = 160
    eta = rng.uniform(0.06, 0.249, n)
    chi1 = rng.uniform(-0.9, 0.9, n)
    chi2 = rng.uniform(-0.9, 0.9, n)
    expected = np.array([-0.12, 0.13, -1.25, -0.9])
    truth = predict(eta, total_spin(eta, chi1, chi2), expected)
    data = NRData(eta, chi1, chi2, truth, rng.uniform(0.05, 1, n),
                  np.array([f"synthetic-{i}" for i in range(n)]),
                  np.asarray(CATALOGS * (n // len(CATALOGS))))
    fit = fit_four(data, np.ones(n, dtype=bool), weighted=False)
    np.testing.assert_allclose(fit["b"], expected, rtol=0, atol=1e-11)
    report = run(data)
    for fold in report["catalog_holdouts"].values():
        np.testing.assert_allclose(fold["train_b"], expected, rtol=0, atol=1e-11)
        assert fold["holdout"]["rmse"] < 1e-14


def test_one_coefficient_sensitivity_recovers_injected_b2():
    eta = np.linspace(0.06, 0.24, 16)
    chi1 = np.linspace(-0.85, 0.9, 16)
    chi2 = np.linspace(0.75, -0.8, 16)
    coefficients = PAPER_B.copy()
    coefficients[1] = 0.115
    data = NRData(eta, chi1, chi2, predict(eta, total_spin(eta, chi1, chi2), coefficients),
                  np.linspace(0.1, 1, 16), np.array([str(i) for i in range(16)]),
                  np.asarray(CATALOGS * 4))
    for weighted in (False, True):
        assert fit_b2_only(data, np.ones(16, dtype=bool), weighted=weighted) == pytest.approx(0.115, abs=1e-13)


def test_input_hash_is_required(tmp_path):
    table = tmp_path / "nr.txt"
    table.write_text("# eta chi1 chi2 Erad chif weight  tag code\n")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        read_nr_table(table)

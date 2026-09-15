"""Tests for the observable public-epoch bisection diagnostic."""

import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from scripts.diagnose_eob_public_epoch_bisection_clean import (
    _bisection_hypothesis,
    _gps_quantization,
    _local_polynomial_derivative,
)


def test_fixed_polynomial_derivative_bisection_resolves_sign_bracket():
    time = np.arange(8, dtype=np.float64)
    omega = 1.0 - (time - 3.4) ** 2
    derivative = _local_polynomial_derivative(time, omega, 3, 3.4, 3)
    assert abs(derivative) < 1e-12

    result = _bisection_hypothesis(time, omega, 3, 3, iterations=14)
    assert result["initial_endpoint"]["bracket_status"] == "valid_sign_bracket"
    assert abs(result["candidate_time_M_initial_origin"] - 3.4) < 1e-4


def test_same_sign_endpoint_contraction_is_retained_as_nonbracketing():
    time = np.arange(8, dtype=np.float64)
    omega = 1.0 - (time - 2.8) ** 2
    result = _bisection_hypothesis(time, omega, 3, 6, iterations=10)
    endpoint = result["initial_endpoint"]
    assert endpoint["bracket_status"] == "nonbracketing_same_sign_negative"
    assert endpoint["valid_sign_bracket"] is False
    expected = 4.0 - 1.0 / 2048.0
    assert result["candidate_time_M_initial_origin"] == expected


def test_tolerance_hypothesis_reports_width_and_gps_interval():
    time = np.arange(8, dtype=np.float64)
    omega = 1.0 - (time - 2.8) ** 2
    result = _bisection_hypothesis(time, omega, 3, 3, tolerance_M=1e-5)
    assert result["stopped_by_tolerance"] is True
    assert result["final_interval_width_M"] <= 1e-5
    gps = _gps_quantization(-1.232042679)
    assert gps["nearest_integer_epoch_ns"] == -1232042679
    np.testing.assert_allclose(
        gps["nearest_ns_interval_epoch_s"],
        [-1.2320426795, -1.2320426785],
        atol=1e-15,
    )
    assert "does not establish" in gps["interpretation"]


def test_pure_helpers_import_without_optional_lal_bindings():
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        (
            str(repo_root / "src"),
            str(repo_root / "tests"),
            str(repo_root / "scripts"),
            str(repo_root),
        )
    )
    code = r'''
import builtins
import numpy as np

real_import = builtins.__import__

def blocked_lal(name, *args, **kwargs):
    if name.split(".", 1)[0] in {"lal", "lalsimulation"}:
        raise ModuleNotFoundError(name)
    return real_import(name, *args, **kwargs)

builtins.__import__ = blocked_lal
from scripts.diagnose_eob_public_epoch_bisection_clean import (
    _bisection_hypothesis,
    _gps_quantization,
    _local_polynomial_derivative,
)
assert _gps_quantization(1.0)["nearest_integer_epoch_ns"] == 1000000000
time = np.arange(8, dtype=float)
assert abs(_local_polynomial_derivative(
    time,
    1.0 - (time - 3.4) ** 2,
    3,
    3.4,
    3,
)) < 1e-12
print("pure helpers imported")
'''
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "pure helpers imported" in completed.stdout

"""Test the torchwave.qnm package API and CLI entry points."""

import shutil
import subprocess
import sys
import pytest
import torchwave
from torchwave import qnm
from torchwave.qnm import solve, KerrRoot, angular_eigenvalue, radial_residual, MODES, generate_main


def test_qnm_exported_in_torchwave():
    assert hasattr(torchwave, "qnm")
    assert "qnm" in torchwave.__all__
    assert qnm.solve is solve
    assert qnm.angular_eigenvalue is angular_eigenvalue
    assert qnm.radial_residual is radial_residual
    assert qnm.MODES is MODES



def test_qnm_solve_schwarzschild():
    root = solve(0.0, 2, 2, 0)
    assert isinstance(root, KerrRoot)
    assert root.converged
    assert abs(root.omega.real - 0.37367) < 1e-4
    assert abs(root.omega.imag - (-0.08896)) < 1e-4
    assert root.metadata["branch_origin"].startswith("large-ell WKB")


def test_qnm_solve_kerr_spin():
    root = solve(0.5, 2, 2, 0)
    assert isinstance(root, KerrRoot)
    assert root.converged
    assert root.omega.real > 0.37367  # spin increases frequency for corotating m=2
    assert root.omega.imag < 0


def test_qnm_generate_cli_help():
    with pytest.raises(SystemExit) as exc_info:
        generate_main(["--help"])
    assert exc_info.value.code == 0


def test_qnm_module_cli():
    res = subprocess.run(
        [sys.executable, "-m", "torchwave.qnm.generate", "--help"],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert "TorchWave Kerr QNM Data Generator" in res.stdout
    assert "--output-dir" in res.stdout


def test_qnm_console_script_cli():
    exe = shutil.which("torchwave-generate-qnm")
    if exe is None:
        pytest.skip("torchwave-generate-qnm is not installed in PATH")
    res = subprocess.run(
        [exe, "--help"],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert "TorchWave Kerr QNM Data Generator" in res.stdout
    assert "--output-dir" in res.stdout


def test_qnm_london_exported():
    assert hasattr(qnm, "london")
    assert "london" in qnm.__all__
    assert callable(qnm.london.exact_theta_grid)
    assert callable(qnm.london.exact_spin_grid)
    assert callable(qnm.london.check_theta_grid)
    assert callable(qnm.london.evaluate_coefficients)
    assert callable(qnm.london.fit_exact_rows)
    assert callable(qnm.london.generate_exact_training_grid)
    assert callable(qnm.london.main)


def test_qnm_london_cli_help():
    with pytest.raises(SystemExit) as exc_info:
        qnm.london.main(["--help"])
    assert exc_info.value.code == 0


def test_qnm_london_module_cli():
    res = subprocess.run(
        [sys.executable, "-m", "torchwave.qnm.london", "--help"],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert "TorchWave London N=61 Kerr QNM reconstruction" in res.stdout
    assert "--output-dir" in res.stdout


def test_qnm_london_console_script_cli():
    exe = shutil.which("torchwave-generate-london-qnm")
    if exe is None:
        pytest.skip("torchwave-generate-london-qnm is not installed in PATH")
    res = subprocess.run(
        [exe, "--help"],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert "TorchWave London N=61 Kerr QNM reconstruction" in res.stdout
    assert "--output-dir" in res.stdout


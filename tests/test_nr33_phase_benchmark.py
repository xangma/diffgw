"""Boundary checks for the frozen NR/LAL phase diagnostic adapter."""
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import run_nr33_phase_benchmark as runner
from torchwave.constants import C_SI, MPC_SI, MSUN_SEC


class FakeLAL:
    MSUN_SI = 1.988409870698051e30
    __version__ = "test-lal"

    @staticmethod
    def CreateREAL8Vector(n):
        return SimpleNamespace(data=np.zeros(n))

    @staticmethod
    def CreateDict():
        return {}


class FakeSimulation:
    __version__ = "test-simulation"
    arguments = None

    @staticmethod
    def SimInspiralWaveformParamsInsertPhenomXHMReleaseVersion(params, release):
        params["release"] = release

    @staticmethod
    def SimInspiralWaveformParamsLookupPhenomXHMReleaseVersion(params):
        return params["release"]

    @classmethod
    def SimIMRPhenomXHMFrequencySequenceOneMode(cls, *arguments):
        cls.arguments = arguments
        values = np.arange(len(arguments[0].data)) + 1j * np.arange(1, len(arguments[0].data) + 1)
        return SimpleNamespace(data=SimpleNamespace(data=values), epoch=0.)


def test_public_lal_mode_frequency_and_complex_scaling_contract():
    f = np.linspace(.1, .3, 8)
    values, hz, meta = runner.lal_complex(f, 3., 60., lal_module=FakeLAL, ls_module=FakeSimulation)
    args = FakeSimulation.arguments
    assert args[5:7] == (3, -3)
    assert args[-1]["release"] == 122022
    assert args[1] == 45. * FakeLAL.MSUN_SI
    assert args[2] == 15. * FakeLAL.MSUN_SI
    assert args[3:5] == (0., 0.)
    assert args[8:10] == (0., 20.)
    np.testing.assert_array_equal(hz, f / (60. * MSUN_SEC))
    np.testing.assert_array_equal(args[0].data, hz)
    scale = (60. * MSUN_SEC) ** 2 * C_SI / (100. * MPC_SI)
    np.testing.assert_allclose(values * scale, np.arange(8) + 1j * np.arange(1, 9))
    assert meta["mode"] == [3, -3]


def test_quadrature_integrates_constant_and_linear_on_nonuniform_grid():
    f = np.array([.1, .105, .12, .14, .17, .23, .26, .3])
    weights = runner.quadrature_weights(f)
    assert weights.sum() == pytest.approx(.2)
    assert np.sum(weights * f) == pytest.approx((.3 ** 2 - .1 ** 2) / 2)
    with pytest.raises(ValueError):
        runner.quadrature_weights(f[::-1])


def test_phase_support_does_not_reconnect_weak_tail_islands():
    reference = np.ones(20, dtype=complex)
    reference[12:15] = 1e-4
    mask = runner.phase_support(reference, .001)
    np.testing.assert_array_equal(mask, np.arange(20) < 12)
    with pytest.raises(ValueError, match="insufficient"):
        runner.phase_support(np.r_[np.ones(4), np.zeros(10)], .001)


def test_protocol_rejected_before_loading_inputs(tmp_path):
    protocol = tmp_path / "protocol.json"
    protocol.write_text("{}")
    with pytest.raises(ValueError, match="preregistered hash"):
        runner.verify_inputs(protocol, tmp_path / "missing-input", tmp_path / "missing-freeze")


def test_results_are_never_overwritten(tmp_path):
    path = tmp_path / "result.json"
    runner.write_json(path, {"original": True})
    with pytest.raises(FileExistsError):
        runner.write_json(path, {"original": False})
    assert runner.read_json(path) == {"original": True}

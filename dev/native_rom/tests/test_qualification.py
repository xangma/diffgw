"""Maintained tests of diagnostic gates; these do not validate EOB physics."""

import json
import math

import numpy as np
import pytest

from dev.native_rom import qualify_generator as q


def signal(dt=.5, length=9, perturbation=0.):
    time = np.arange(length)*dt
    mode = np.exp(-.1*time-1j*time)*(1+perturbation)
    return dict(time=time, mode=mode, dt_geom=dt,
                peak_time_M=2., match_time_M=2.1, boundary_time_M=2.)


def test_complex_metric_does_not_fit_away_phase_or_amplitude():
    reference = np.array([1, 1j, -1, -1j], dtype=complex)
    phase = q.mode_metrics(1j*reference, reference)
    assert phase["complex_relative_l2"] == pytest.approx(math.sqrt(2))
    assert phase["amplitude_relative_l2"] == 0
    assert phase["phase_rms_rad"] == pytest.approx(math.pi/2)
    assert phase["unmaximized_real_overlap"] == 0
    assert q.mode_metrics(2*reference, reference)["complex_relative_l2"] == 1


@pytest.mark.parametrize("bad", [np.array([np.nan, 1j]), np.array([np.inf, 1j]), np.zeros(2)])
def test_invalid_signals_fail(bad):
    with pytest.raises(ValueError):
        q.mode_metrics(bad, np.ones(2))


def test_nested_grids_use_exact_samples_and_include_missing_tails():
    assert q.compare_runs(signal(), signal(.25, 17))["complex_relative_l2"] == 0
    metrics = q.compare_runs(signal(length=6), signal())
    assert metrics["complex_relative_l2"] > .1
    assert metrics["common_samples"] == 6
    assert metrics["samples"] == 9
    shifted = signal()
    shifted["time"] += .1
    with pytest.raises(ValueError, match="starting at t=0"):
        q.compare_runs(shifted, signal())
    with pytest.raises(ValueError, match="nested"):
        q.compare_runs(signal(.3), signal(.5))


def test_stability_budgets_require_both_refinement_and_event_stability():
    coarse, medium, fine = signal(perturbation=.02), signal(perturbation=.001), signal()
    assert q.convergence_check(coarse, medium, fine, q.Settings(relative_l2_limit=.002))["numerical_check_passed"]
    fine["peak_time_M"] += 1.
    result = q.convergence_check(coarse, medium, fine, q.Settings(relative_l2_limit=.002))
    assert not result["numerical_check_passed"]
    assert not result["gates"]["peak_time_M"]
    result = q.convergence_check(signal(), signal(), signal(perturbation=.0001), q.Settings())
    assert not result["gates"]["error_decreased"]


def test_campaign_runs_independent_ode_and_grid_checks_with_one_shared_solve():
    calls = []
    def runner(row, level, settings):
        calls.append(level)
        dt = level["dt_geom"]
        return dict(**signal(dt, round(4/dt)+1), **{k: v for k, v in level.items() if k != "dt_geom"})
    report = q.qualify_row({}, q.Settings(), runner)
    assert report["numerical_checks_passed"]
    assert set(report["checks"]) == {"ode_tolerance", "output_grid"}
    assert len(calls) == 5
    assert len({call["dt_geom"] for call in calls[:3]}) == 1
    assert len({(call["atol"], call["rtol"]) for call in calls[2:]}) == 1
    assert q.SCOPE["physics_validated"] is False
    assert q.SCOPE["production_promotion_allowed"] is False
    json.dumps(report, allow_nan=False)


def test_failed_or_nonfinite_solve_cannot_pass_campaign():
    def runner(row, level, settings):
        run = signal()
        run["mode"][0] = np.nan
        return run
    report = q.qualify_row({}, q.Settings(), runner)
    assert not report["numerical_checks_passed"]
    assert len(report["errors"]) == 1
    assert report["runs"] == []
    json.dumps(report, allow_nan=False)


def test_requested_oracle_error_is_a_failure():
    def runner(row, level, settings):
        dt = level["dt_geom"]
        return dict(**signal(dt, round(4/dt)+1), name=level["name"])
    def oracle(*_):
        raise RuntimeError("oracle unavailable")
    report = q.qualify_row({}, q.Settings(), runner, oracle)
    assert not report["numerical_checks_passed"]
    assert report["errors"][0]["check"] == "lal"


def test_evidence_writer_rejects_production_paths_and_nonfinite_numbers(tmp_path):
    with pytest.raises(ValueError, match="no production asset"):
        q.write_evidence(q.ROOT / "src/asset.json", {})
    with pytest.raises(ValueError, match="no production asset"):
        q.write_evidence(tmp_path / "asset.npz", {})
    with pytest.raises(ValueError):
        q.write_evidence(tmp_path / "evidence.json", {"scope": q.SCOPE, "bad": float("nan")})
    with pytest.raises(ValueError, match="explicitly refuse"):
        q.write_evidence(tmp_path / "evidence.json", {"scope": {**q.SCOPE, "production_asset": True}})
    output = tmp_path / "evidence.json"
    q.write_evidence(output, {"scope": q.SCOPE, "status": "failed"})
    assert json.loads(output.read_text())["scope"]["training_generator_qualified"] is False


@pytest.mark.parametrize("pins", [["--lal-version", "7.7.1"],
                                 ["--lalsimulation-version", "6.2.1"], ["--compare-lal"]])
def test_cli_requires_both_reference_pins_only_with_requested_reference(pins, tmp_path):
    with pytest.raises(SystemExit) as error:
        q.main(["--json-out", str(tmp_path / "evidence.json"), "--print-plan", *pins])
    assert error.value.code == 2


def test_default_numerical_source_budget_remains_strict():
    assert q.Settings().relative_l2_limit == 1e-6


def test_tighter_tolerance_plan_preserves_declared_error_budget():
    atol, rtol = q.validate_tolerance_levels((1e-12, 1e-13, 1e-14),
                                              (1e-11, 1e-12, 1e-13))
    plan = q.run_plan(q.Settings(atol_levels=atol, rtol_levels=rtol))
    assert [run["rtol"] for run in plan[:3]] == list(rtol)
    assert q.Settings(atol_levels=atol, rtol_levels=rtol).relative_l2_limit == 1e-6
    with pytest.raises(ValueError, match="strictly decreasing"):
        q.validate_tolerance_levels((1e-12, 1e-12, 1e-14), rtol)


def test_source_snapshot_ignores_generated_egg_info_but_tracks_sources_and_assets(tmp_path, monkeypatch):
    monkeypatch.setattr(q, "ROOT", tmp_path)
    runner = tmp_path / "dev/native_rom/qualify_generator.py"
    monkeypatch.setattr(q, "__file__", str(runner))
    files = {"src/example/model.py": "equations", "src/example/data/table.npz": "numbers",
             "src/example.egg-info/SOURCES.txt": "generated listing",
             "dev/native_rom/qualify_generator.py": "runner"}
    for name, contents in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)
    before = q.source_snapshot()
    assert set(before) == set(files) - {"src/example.egg-info/SOURCES.txt"}
    (tmp_path / "src/example.egg-info/SOURCES.txt").write_text("updated by packaging")
    assert q.source_snapshot() == before
    for name in ("src/example/model.py", "src/example/data/table.npz"):
        (tmp_path / name).write_text("changed scientific input")
        assert q.source_snapshot()[name] != before[name]


@pytest.mark.parametrize("rows", [[], [{"mass1": [30]}], [dict(mass1=30, mass2=20, chi1=2, chi2=0, omega_start=.04)]])
def test_scalar_input_contract(rows):
    with pytest.raises(ValueError):
        q.validate_rows(rows)

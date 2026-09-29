"""Frozen black-box PV300 carrier-clock and Euler-chart discrimination.

The public binary supplies coprecessing (2,2) modes and Euler series, never
internal precession states.  A common time reparametrization scales the mode
phase rate and Euler-derived orbital-axis rate equally.  The published
minimal-rotation identity, d gamma = -cos(beta) d alpha, separately tests the
returned gamma chart.  These observations cannot distinguish a hidden ODE
change from a change in the returned alpha/beta map or a separate precession
clock.  No binary outputs are fitted or used as production coefficients.
"""

from __future__ import annotations

import datetime
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import lal
import lalsimulation as ls
import numpy as np
import scipy
import torch

from diffgw_core.constants import MSUN_SEC
from diffgw_reference.phenomt_numerical_trajectory import initial_state
from diffgw_reference.phenomt_spin_evolution import evaluate_spin_derivatives
from scripts.diagnose_phenomt_initial_rates import endpoint_rate
from scripts.validate_phenomt_precession import CASES
from torchwave.approximants.imrphenomt import IMRPhenomT


CASE = "high_mass_ratio"
RATE = 16384
PROBE_SECONDS = (0.0, 0.25, 0.5, 1.0, 2.0)
CHART_WINDOW_SECONDS = 2.0


def _center_rate(values: np.ndarray, index: int, dt: float) -> tuple[float, float]:
    """Return fixed fourth-/second-order centered derivatives."""
    high = (values[index - 2] - 8 * values[index - 1]
            + 8 * values[index + 1] - values[index + 2]) / (12 * dt)
    low = (values[index + 1] - values[index - 1]) / (2 * dt)
    return float(high), float(low)


def _public_series():
    m1, m2, spin1, spin2, inclination, phase, fref = CASES[CASE]
    options = lal.CreateDict()
    ls.SimInspiralWaveformParamsInsertPhenomXPrecVersion(options, 300)
    ls.SimInspiralWaveformParamsInsertPhenomXPFinalSpinMod(options, 4)
    modes, alpha, cosbeta, gamma, _ = ls.SimIMRPhenomTPHM_CoprecModes(
        m1 * lal.MSUN_SI, m2 * lal.MSUN_SI, *spin1, *spin2,
        100e6 * lal.PC_SI, inclination, 1 / RATE, fref, fref, phase, options, 1,
    )
    mode = ls.SphHarmTimeSeriesGetMode(modes, 2, 2)
    n = mode.data.length
    for series in (alpha, cosbeta, gamma):
        if (series.data.length != n or series.deltaT != mode.deltaT
                or series.epoch != mode.epoch):
            raise RuntimeError("public mode and Euler grids differ")
    return (
        np.asarray(mode.data.data),
        np.unwrap(np.asarray(alpha.data.data)),
        np.asarray(cosbeta.data.data),
        np.unwrap(np.asarray(gamma.data.data)),
        float(mode.epoch),
    )


def diagnose():
    m1, m2, spin1, spin2, _inclination, _phase, fref = CASES[CASE]
    mode, alpha, cosbeta, gamma, epoch = _public_series()
    dt = 1 / RATE
    inspiral_count = int(np.count_nonzero(epoch + dt * np.arange(len(mode)) < 0))
    if inspiral_count <= round(CHART_WINDOW_SECONDS * RATE) + 4:
        raise RuntimeError("public inspiral does not cover frozen chart window")
    if any(not np.isfinite(x[:inspiral_count]).all()
           for x in (mode, alpha, cosbeta, gamma)):
        raise RuntimeError("nonfinite public inspiral samples")
    beta = np.arccos(np.clip(cosbeta, -1, 1))
    phase = np.unwrap(np.angle(mode))

    total_seconds = (m1 + m2) * MSUN_SEC
    model = IMRPhenomT()
    carrier = model._coeff(
        torch.tensor(m1, dtype=torch.float64),
        torch.tensor(m2, dtype=torch.float64),
        torch.tensor(spin1[2], dtype=torch.float64),
        torch.tensor(spin2[2], dtype=torch.float64),
    )
    reference_seconds = float(model.time_at_frequency(fref, m1, m2, spin1[2], spin2[2]))

    def native_frequency(elapsed: float) -> float:
        time_over_m = (reference_seconds + elapsed) / total_seconds
        omega = float(model._evaluate(torch.tensor(time_over_m, dtype=torch.float64), carrier)[0])
        return omega / (2 * math.pi * total_seconds)

    probe_rows = []
    for elapsed in PROBE_SECONDS:
        index = round(elapsed * RATE)
        if index == 0:
            public_rate = float(endpoint_rate(phase[:9], dt))
            lower_rate = float(endpoint_rate(phase[:5], dt))
        else:
            public_rate, lower_rate = _center_rate(phase, index, dt)
        public_frequency = abs(public_rate) / (2 * math.pi)
        native = native_frequency(elapsed)
        probe_rows.append(dict(
            elapsed_seconds=elapsed,
            public_mode_frequency_hz=public_frequency,
            native_carrier_frequency_hz=native,
            relative_difference=abs(public_frequency / native - 1),
            derivative_order_difference_hz=abs(public_rate - lower_rate) / (2 * math.pi),
        ))

    axis = np.stack((np.sin(beta) * np.cos(alpha),
                     np.sin(beta) * np.sin(alpha), np.cos(beta)), axis=-1)
    public_axis_rate = endpoint_rate(axis[:9], dt)
    public_axis_rate_low = endpoint_rate(axis[:5], dt)
    x1, x2 = m1 / (m1 + m2), m2 / (m1 + m2)
    lhat, s1, s2, _ = np.split(initial_state(x1, x2, spin1, spin2), 4)
    vref = (math.pi * total_seconds * fref) ** (1 / 3)
    reference_rhs = evaluate_spin_derivatives(
        vref, lhat, s1, s2, x1, x2, include_n3lo_ss=False, dps=50
    )
    independent_axis_speed = float(np.linalg.norm(
        np.asarray(reference_rhs.dlhat, dtype=float) / total_seconds
    ))
    public_axis_speed = float(np.linalg.norm(public_axis_rate))

    stop = round(CHART_WINDOW_SECONDS * RATE) + 1
    alpha_increment = np.diff(alpha[:inspiral_count])
    expected_gamma_increment = -0.5 * (
        cosbeta[:inspiral_count - 1] + cosbeta[1:inspiral_count]
    ) * alpha_increment
    chart_error = (
        gamma[:inspiral_count] - gamma[0]
        - np.r_[0.0, np.cumsum(expected_gamma_increment)]
    )
    # The same integral on every second node bounds quadrature sensitivity.
    coarse = np.arange(0, stop, 2)
    coarse_increment = -0.5 * (
        cosbeta[coarse[:-1]] + cosbeta[coarse[1:]]
    ) * np.diff(alpha[coarse])
    fine_at_coarse = np.cumsum(expected_gamma_increment).reshape(-1)[coarse[1:] - 1]
    chart_quadrature_difference = float(np.max(abs(
        np.cumsum(coarse_increment) - fine_at_coarse
    )))

    return dict(
        case=CASE,
        parameters=dict(mass1=m1, mass2=m2, spin1=list(spin1), spin2=list(spin2),
                        f_ref_hz=fref, sample_rate_hz=RATE, precession_version=300,
                        final_spin_mod=4),
        conventions=dict(
            reference="TPHM Eqs. (12)-(14) and minimal-rotation Eq. (5); native T carrier from author-pinned fit",
            public="Unmodified public coprecessing (2,2) mode and Euler series; no internal state exposed",
            alignment="First public sample and native f_ref event identified; no fitted shift or phase",
            limitation="Common-clock test cannot exclude a separate hidden precession clock; gamma check cannot validate alpha/beta construction or hidden spin ODE",
            source_restriction="No LAL implementation source inspected or translated; no binary output fitted",
            production="No equation, coefficient, or gate change",
        ),
        event=dict(public_epoch_seconds=epoch, native_reference_seconds=reference_seconds,
                   difference_seconds=reference_seconds - epoch),
        carrier_probes=probe_rows,
        initial_axis_rate=dict(
            public_speed_per_second=public_axis_speed,
            independent_published_speed_per_second=independent_axis_speed,
            public_over_published=public_axis_speed / independent_axis_speed,
            derivative_order_difference_per_second=float(np.linalg.norm(
                public_axis_rate - public_axis_rate_low
            )),
            common_clock_ratio_from_mode=probe_rows[0]["public_mode_frequency_hz"] / fref,
        ),
        minimal_rotation_chart=dict(
            equation="gamma(t)-gamma(0) = -integral cos(beta) d alpha",
            window_seconds=CHART_WINDOW_SECONDS,
            window_samples=stop,
            max_residual_window_radians=float(np.max(abs(chart_error[:stop]))),
            final_residual_window_radians=float(chart_error[stop - 1]),
            max_residual_full_inspiral_radians=float(np.max(abs(chart_error))),
            two_grid_quadrature_difference_window_radians=chart_quadrature_difference,
            beta_range_window_radians=[float(np.min(beta[:stop])), float(np.max(beta[:stop]))],
        ),
    )


def main():
    if len(sys.argv) != 3 or sys.argv[1] != "--output":
        raise SystemExit("usage: diagnose_phenomt_pv300_clock_chart.py --output PATH")
    output = Path(sys.argv[2])
    receipt = dict(
        generated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        execution=dict(host=socket.gethostname(), working_directory=os.getcwd(),
                       command=f"PYTHONPATH=src {sys.executable} {' '.join(sys.argv)}"),
        versions=dict(python=platform.python_version(), numpy=np.__version__,
                      scipy=scipy.__version__, torch=torch.__version__,
                      lalsuite=importlib.metadata.version("lalsuite")),
        result=diagnose(),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()

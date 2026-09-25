"""Build scalar PV223 angle tables and isolate PhenomT continuations.

Public LAL binary angles are observations only.  Native raw PV223 angles feed
an independent scalar implementation of the paper's MECO, minimal-rotation,
and ringdown rules.  This separates analytical-inspiral residuals (22300)
from continuation implementation residuals without inspecting LAL source or
fitting any binary output.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import lal
import lalsimulation as ls
import numpy as np
import scipy
from scipy.interpolate import PchipInterpolator
import torch

from diffgw_core import phenomt_fits as fits
from diffgw_core.constants import MSUN_SEC
from diffgw_reference.phenomt_angle_continuation import apply_angle_option
from scripts.validate_phenomt_precession import CASES, parameters
from torchwave.approximants import _phenomt_dispatch as dispatch
from torchwave.approximants import _phenomt_msa as msa
from torchwave.approximants.imrphenom_xas import XASParameterFits
from torchwave.approximants.imrphenomt import IMRPhenomT


VERSIONS = (22300, 22310, 22311, 22321)


def _rotation_matrix(alpha, beta, gamma):
    alpha, beta, gamma = np.broadcast_arrays(alpha, beta, gamma)
    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta), np.sin(beta)
    cg, sg = np.cos(gamma), np.sin(gamma)
    return np.stack((
        ca * cb * cg - sa * sg, -ca * cb * sg - sa * cg, ca * sb,
        sa * cb * cg + ca * sg, -sa * cb * sg + ca * cg, sa * sb,
        -sb * cg, sb * sg, cb,
    ), axis=-1).reshape(alpha.shape + (3, 3))


def _rotation_distance(left, right):
    chord = np.linalg.norm(left - right, axis=(-2, -1)) / (2 * np.sqrt(2))
    return 2 * np.arcsin(np.clip(chord, 0, 1))


def _relative(rotations):
    return np.swapaxes(rotations[0], -1, -2) @ rotations


def _analytical_onset(times, meco_time, native, observed):
    """Locate raw-angle divergence before any suffix continuation can act."""
    times = np.asarray(times, dtype=float)
    native = np.asarray(native, dtype=float)
    observed = np.asarray(observed, dtype=float)
    use = times < meco_time
    if not np.any(use) or not use[0]:
        raise RuntimeError("public grid does not begin in the analytical inspiral")
    selected_times = times[use]
    selected_native = native[:, use]
    selected_observed = observed[:, use]
    distance = _rotation_distance(
        _relative(_rotation_matrix(*selected_native)),
        _relative(_rotation_matrix(*selected_observed)),
    )
    alpha_residual = (
        np.unwrap(selected_native[0]) - np.unwrap(selected_observed[0])
    )
    alpha_residual -= alpha_residual[0]
    beta_residual = selected_native[1] - selected_observed[1]
    gamma_residual = (
        np.unwrap(selected_native[2]) - np.unwrap(selected_observed[2])
    )
    gamma_residual -= gamma_residual[0]
    crossings = {}
    for threshold in (1e-4, 1e-3, 1e-2):
        indices = np.flatnonzero(distance >= threshold)
        if len(indices) == 0:
            crossings[f"{threshold:g}"] = None
            continue
        index = int(indices[0])
        crossings[f"{threshold:g}"] = dict(
            sample_index=index,
            time_seconds=float(selected_times[index]),
            elapsed_from_first_sample_seconds=float(selected_times[index] - selected_times[0]),
            relative_rotation_radians=float(distance[index]),
            alpha_anchored_residual_radians=float(alpha_residual[index]),
            beta_residual_radians=float(beta_residual[index]),
            gamma_anchored_residual_radians=float(gamma_residual[index]),
        )
    return dict(
        scope="PV22300 samples strictly before MECO; no MECO, peak, ringdown, or numerical-gamma rule applies",
        samples=int(np.count_nonzero(use)),
        first_time_seconds=float(selected_times[0]),
        last_time_seconds=float(selected_times[-1]),
        initial_absolute_rotation_radians=float(_rotation_distance(
            _rotation_matrix(*selected_native[:, :1]),
            _rotation_matrix(*selected_observed[:, :1]),
        )[0]),
        relative_rotation_max_radians=float(np.max(distance)),
        relative_rotation_final_radians=float(distance[-1]),
        first_relative_rotation_crossings=crossings,
    )


def _public_angles(version, case, rate):
    m1, m2, spin1, spin2, inclination, phase, fref = CASES[case]
    options = lal.CreateDict()
    ls.SimInspiralWaveformParamsInsertPhenomXPrecVersion(options, version)
    ls.SimInspiralWaveformParamsInsertPhenomXPFinalSpinMod(options, 3)
    modes, alpha, cosbeta, gamma, _ = ls.SimIMRPhenomTPHM_CoprecModes(
        m1 * lal.MSUN_SI, m2 * lal.MSUN_SI, *spin1, *spin2,
        100e6 * lal.PC_SI, inclination, 1 / rate, 20.0, fref, phase, options, 1,
    )
    mode = ls.SphHarmTimeSeriesGetMode(modes, 2, 2)
    times = float(mode.epoch) + mode.deltaT * np.arange(mode.data.length)
    for series in (alpha, cosbeta, gamma):
        if series.data.length != len(times) or series.deltaT != mode.deltaT:
            raise RuntimeError("public angle grids are inconsistent")
    angles = np.stack((
        alpha.data.data,
        np.arccos(np.clip(cosbeta.data.data, -1, 1)),
        gamma.data.data,
    ))
    return times, angles


def _selected_indices(times, reference_time, meco_time):
    positive = times[times >= 0]
    ring_mid = positive[len(positive) // 2] if len(positive) else times[-1]
    targets = (
        times[0],
        (reference_time + meco_time) / 2,
        meco_time - 1 / 2048,
        meco_time,
        meco_time / 2,
        times[times < 0][-1],
        times[times >= 0][0] if np.any(times >= 0) else times[-1],
        ring_mid,
        times[-1],
    )
    return np.asarray(sorted(set(int(np.argmin(abs(times - target))) for target in targets)))


def _segment(time, meco_time):
    if time < meco_time:
        return "analytical_inspiral"
    if time < 0:
        return "meco_to_peak"
    return "ringdown"


def _metrics(times, meco_time, actual, expected):
    result = {}
    rotations_actual = _rotation_matrix(*actual)
    rotations_expected = _rotation_matrix(*expected)
    for segment in ("analytical_inspiral", "meco_to_peak", "ringdown", "all"):
        mask = np.ones(len(times), dtype=bool) if segment == "all" else np.asarray([
            _segment(time, meco_time) == segment for time in times
        ])
        if not np.any(mask):
            continue
        left, right = rotations_actual[mask], rotations_expected[mask]
        result[segment] = dict(
            samples=int(np.count_nonzero(mask)),
            absolute_rotation_max_radians=float(np.max(_rotation_distance(left, right))),
            relative_rotation_max_radians=float(np.max(_rotation_distance(
                _relative(left), _relative(right)
            ))),
            alpha_anchored_max_abs_radians=float(np.max(abs(
                np.unwrap(actual[0, mask]) - np.unwrap(expected[0, mask])
                - (actual[0, mask][0] - expected[0, mask][0])
            ))),
            beta_max_abs_radians=float(np.max(abs(actual[1, mask] - expected[1, mask]))),
            gamma_anchored_max_abs_radians=float(np.max(abs(
                np.unwrap(actual[2, mask]) - np.unwrap(expected[2, mask])
                - (actual[2, mask][0] - expected[2, mask][0])
            ))),
        )
    return result


def diagnose(case, rate, table_size):
    m1, m2, spin1, spin2, inclination, phase, fref = CASES[case]
    values = [torch.tensor(value, dtype=torch.float64) for value in (
        m1, m2, *spin1, *spin2, inclination, phase, fref
    )]
    tm1, tm2, x1, y1, z1, x2, y2, z2, tinc, tphase, tfref = values
    total_seconds = (m1 + m2) * MSUN_SEC
    model = IMRPhenomT()
    carrier = model._coeff(tm1, tm2, z1, z2)
    reference_time = float(model.time_at_frequency(tfref, tm1, tm2, z1, z2))
    eta = tm1 * tm2 / (tm1 + tm2) ** 2
    f_meco = XASParameterFits.fit_f_meco(eta, z1, z2) / total_seconds
    meco_time = float(model.time_at_frequency(f_meco, tm1, tm2, z1, z2))
    frequency_peak = carrier[13] / (2 * math.pi * total_seconds)
    final_spin = msa.msa_final_spin(tm1, tm2, x1, y1, z1, x2, y2, z2, tfref)
    final_mass = fits.IMRPhenomX_FinalMass2017(eta, z1, z2, (tm1 - tm2) / (tm1 + tm2))
    ringdown_slope = float(
        2 * math.pi / total_seconds
        * (fits.IMRPhenomT_fringfit(final_spin, 22) - fits.IMRPhenomT_fringfit(final_spin, 21))
        / final_mass
    )
    vref = (math.pi * (m1 + m2) * MSUN_SEC * fref) ** (1 / 3)

    def carrier_frequency(time):
        omega = model._evaluate(torch.tensor(float(time) / total_seconds, dtype=torch.float64), carrier)[0]
        return float(omega / (2 * math.pi * total_seconds))

    def carrier_frequency_tensor(time):
        shape = time.shape
        flat = time.reshape(-1)
        omega = model._evaluate(flat / total_seconds, carrier)[0]
        return (omega / (2 * math.pi * total_seconds)).reshape(shape)

    def raw_at_frequency(frequency):
        velocity = (math.pi * total_seconds * max(frequency, np.finfo(float).tiny)) ** (1 / 3)
        alpha, beta, gamma, _, _ = msa._evaluate_msa_precession(
            *[torch.tensor(value, dtype=torch.float64) for value in (
                velocity, vref, m1, m2, *spin1, *spin2, inclination, phase
            )],
            prec_version=223,
        )
        return float(alpha), float(beta), float(gamma)

    def raw_analytic(time):
        return raw_at_frequency(carrier_frequency(time))

    public = {version: _public_angles(version, case, rate) for version in VERSIONS}
    base_times = public[22300][0]
    for version, (times, _angles) in public.items():
        if not np.array_equal(times, base_times):
            raise RuntimeError(f"public grids differ for PV22300 and PV{version}")
    full_query = torch.as_tensor(base_times, dtype=torch.float64)
    full_frequency = carrier_frequency_tensor(full_query)
    full_raw = dispatch.precession_geometry(
        full_query,
        full_frequency,
        tm1,
        tm2,
        x1,
        y1,
        z1,
        x2,
        y2,
        z2,
        tinc,
        tphase,
        tfref,
        frequency_peak=frequency_peak,
        final_spin=final_spin,
        precession_method="msa",
        precession_version=22300,
        frequency_at_time=carrier_frequency_tensor,
    )[:3]
    full_raw = np.stack([value.detach().numpy() for value in full_raw])
    analytical_onset = _analytical_onset(
        base_times, meco_time, full_raw, public[22300][1]
    )
    indices = _selected_indices(base_times, reference_time, meco_time)
    times = base_times[indices]
    table_start = min(reference_time, float(np.min(times[times < 0])))
    derivative_step = 2e-6 * max(1.0, abs(meco_time))
    table_nodes = np.unique(np.concatenate((
        np.linspace(table_start, 0.0, table_size),
        times[times <= 0],
        [reference_time, meco_time],
        meco_time + derivative_step * np.arange(-8, 9),
    )))
    table_nodes = table_nodes[(table_nodes >= table_start) & (table_nodes <= 0)]
    raw_table = np.asarray([raw_analytic(time) for time in table_nodes])

    def interpolator(nodes, values):
        functions = (
            PchipInterpolator(nodes, np.unwrap(values[:, 0])),
            PchipInterpolator(nodes, values[:, 1]),
            PchipInterpolator(nodes, np.unwrap(values[:, 2])),
        )

        def evaluate(time):
            return tuple(float(function(float(time))) for function in functions)

        derivatives = tuple(function.derivative() for function in functions)

        def derivative(time):
            return tuple(float(function(float(time))) for function in derivatives)

        return evaluate, derivative

    fine_raw, fine_derivative = interpolator(table_nodes, raw_table)
    coarse_indices = np.unique(np.concatenate((
        np.arange(0, len(table_nodes), 2),
        [len(table_nodes) - 1],
        [int(np.argmin(abs(table_nodes - target))) for target in np.concatenate((
            times[times <= 0], [reference_time, meco_time, 0.0]
        ))],
    )))
    coarse_raw, coarse_derivative = interpolator(
        table_nodes[coarse_indices], raw_table[coarse_indices]
    )
    table = []
    summaries = {}
    convergence = {}
    for version in VERSIONS:
        if version == 22300:
            independent = np.asarray([raw_analytic(time) for time in times]).T
            coarse = independent.copy()
        else:
            independent = np.stack(apply_angle_option(
                times,
                fine_raw,
                version,
                reference_time=reference_time,
                meco_time=meco_time,
                ringdown_alpha_slope=ringdown_slope,
                raw_derivatives=fine_derivative,
                integration_breakpoints=table_nodes,
            ))
            coarse = np.stack(apply_angle_option(
                times,
                coarse_raw,
                version,
                reference_time=reference_time,
                meco_time=meco_time,
                ringdown_alpha_slope=ringdown_slope,
                raw_derivatives=coarse_derivative,
                integration_breakpoints=table_nodes[coarse_indices],
            ))
        convergence[str(version)] = dict(
            coarse_nodes=len(coarse_indices),
            fine_nodes=len(table_nodes),
            max_rotation_radians=float(np.max(_rotation_distance(
                _rotation_matrix(*coarse), _rotation_matrix(*independent)
            ))),
        )
        query = torch.as_tensor(times, dtype=torch.float64)
        frequency = torch.as_tensor([carrier_frequency(time) for time in times], dtype=torch.float64)
        native = dispatch.precession_geometry(
            query, frequency, tm1, tm2, x1, y1, z1, x2, y2, z2,
            tinc, tphase, tfref,
            frequency_peak=frequency_peak,
            final_spin=final_spin,
            precession_method="msa",
            precession_version=version,
            frequency_at_time=carrier_frequency_tensor,
        )[:3]
        native = np.stack([value.detach().numpy() for value in native])
        observed = public[version][1][:, indices]
        native_rotation = _rotation_matrix(*native)
        independent_rotation = _rotation_matrix(*independent)
        observed_rotation = _rotation_matrix(*observed)
        native_independent_distance = _rotation_distance(native_rotation, independent_rotation)
        independent_public_relative = _rotation_distance(
            _relative(independent_rotation), _relative(observed_rotation)
        )
        summaries[str(version)] = dict(
            native_vs_independent=_metrics(times, meco_time, native, independent),
            independent_vs_public=_metrics(times, meco_time, independent, observed),
        )
        for offset, time in enumerate(times):
            table.append(dict(
                precession_version=version,
                sample_index=int(indices[offset]),
                time_seconds=float(time),
                segment=_segment(time, meco_time),
                native_angles_radians=native[:, offset].tolist(),
                independent_angles_radians=independent[:, offset].tolist(),
                public_binary_angles_radians=observed[:, offset].tolist(),
                native_minus_independent_rotation_radians=float(native_independent_distance[offset]),
                independent_minus_public_relative_rotation_radians=float(
                    independent_public_relative[offset]
                ),
            ))
    return dict(
        case=case,
        sample_rate_hz=rate,
        reference_time_seconds=reference_time,
        meco_time_seconds=meco_time,
        peak_time_seconds=0.0,
        ringdown_alpha_slope_radians_per_second=ringdown_slope,
        selected_samples=len(times),
        raw_scalar_table=dict(
            start_time_seconds=float(table_nodes[0]),
            end_time_seconds=float(table_nodes[-1]),
            fine_nodes=len(table_nodes),
            coarse_nodes=len(coarse_indices),
            interpolation="shape-preserving piecewise cubic; alpha and gamma unwrapped",
            meco_local_spacing_seconds=derivative_step,
        ),
        analytical_angle_onset=analytical_onset,
        table_convergence=convergence,
        summaries=summaries,
        scalar_table=table,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=tuple(CASES), default="high_mass_ratio")
    parser.add_argument("--sample-rate", type=int, default=2048)
    parser.add_argument("--table-nodes", type=int, default=4097)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.sample_rate <= 0 or args.table_nodes < 257 or args.table_nodes % 2 == 0:
        parser.error("sample rate must be positive; table nodes must be odd and at least 257")
    result = diagnose(args.case, args.sample_rate, args.table_nodes)
    receipt = dict(
        generated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        reproduction=(
            "PYTHONPATH=src python scripts/diagnose_phenomt_msa_scalar_angles.py "
            f"--case {args.case} --sample-rate {args.sample_rate} --table-nodes {args.table_nodes} "
            "--output receipt.json"
        ),
        versions=dict(
            python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
            torch=torch.__version__, lalsimulation=ls.__version__,
        ),
        dtype="float64",
        parameters=parameters(args.case),
        conventions=dict(
            analytical="Raw PV223 values come from the audited native evaluator; public PV22300 independently observes the analytical branch.",
            continuation="Independent scalar PCHIP derivatives and adaptive quadrature implement PhenomTP Eqs. (5), (20), and (24).",
            public="Public binary angles are not fitted and no LAL implementation source is inspected or translated.",
            relative_rotation="R(first)^T R(t) removes only the fixed first-sample frame within each reported segment.",
        ),
        result=result,
    )
    encoded = json.dumps(receipt, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()

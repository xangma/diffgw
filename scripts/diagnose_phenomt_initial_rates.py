"""Isolate the spin-even initial precession rate using public LAL binaries.

For one tilted spin, reversing that spin cancels the native spin-orbit terms
and retains the published self-quadrupole torque. This is a diagnostic of
binary behavior, not a reconstruction of its equations or a production fit.
Every case sets the companion spin to zero and f_lower=f_ref.
"""

import argparse
import datetime
import importlib
import json
import math
from pathlib import Path
import platform

import lal
import lalsimulation as ls
import numpy as np

from diffgw_core.constants import MSUN_SEC

try:
    from scripts.validate_phenomt_precession import CASES, parameters
except ModuleNotFoundError:
    from validate_phenomt_precession import CASES, parameters


def endpoint_rate(samples, dt):
    """Differentiate the degree-(n-1) interpolant at its first sample."""
    order = len(samples) - 1
    weights = np.array([(-1)**(i + 1) * math.comb(order, i) / i
                        for i in range(1, order + 1)])
    return weights @ (samples[1:] - samples[0]) / dt


def binary_rate(m1, m2, spin, inc, phase, fref, rate):
    options = lal.CreateDict()
    ls.SimInspiralWaveformParamsInsertPhenomXPrecVersion(options, 300)
    ls.SimInspiralWaveformParamsInsertPhenomXPFinalSpinMod(options, 4)
    result = ls.SimIMRPhenomTPHM_CoprecModes(
        m1 * lal.MSUN_SI, m2 * lal.MSUN_SI, *spin, 0., 0., 0.,
        100e6 * lal.PC_SI, inc, 1 / rate, fref, fref, phase, options, 1,
    )
    for series in result[1:4]:
        if series.data.length < 9 or not np.all(np.isfinite(series.data.data[:9])):
            raise RuntimeError("Insufficient finite reference samples")
        if not np.isclose(series.deltaT, 1 / rate, rtol=0, atol=1e-18) or not np.isclose(
            float(series.epoch), float(result[1].epoch), rtol=0, atol=1e-12
        ):
            raise RuntimeError("Inconsistent binary angle grids")
    alpha = result[1].data.data[:9]
    beta = np.arccos(np.clip(result[2].data.data[:9], -1., 1.))
    gamma0 = result[3].data.data[0]
    axis = np.stack((np.sin(beta) * np.cos(alpha),
                     np.sin(beta) * np.sin(alpha), np.cos(beta)), axis=-1)
    phi = np.arctan2(spin[1], spin[0])
    def rotation_z(angle):
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])

    rz = rotation_z(phi)
    c, s = np.cos(beta[0]), np.sin(beta[0])
    ry = np.array([[c, 0., -s], [0., 1., 0.], [s, 0., c]])
    # Map the source triad to the returned J-frame. Verify the entire initial
    # Euler rotation, including the tangent azimuth, for each spin sign.
    rotation = rz @ ry @ rz.T
    euler_rotation = rotation_z(alpha[0]) @ ry.T @ rotation_z(gamma0)
    triad_error = float(np.linalg.norm(rotation - euler_rotation))
    anchor_error = float(np.linalg.norm(rotation.T @ axis[0] - [0., 0., 1.]))
    if max(anchor_error, triad_error) > 1e-10:
        raise RuntimeError("Reference angle convention does not match the source triad")
    high = rotation.T @ endpoint_rate(axis, 1 / rate)
    low = rotation.T @ endpoint_rate(axis[:5], 1 / rate)
    mode = ls.SphHarmTimeSeriesGetMode(result[0], 2, 2)
    mode_phase = np.unwrap(np.angle(mode.data.data[:9]))
    initial_frequency = abs(float(endpoint_rate(mode_phase, 1 / rate))) / (2 * math.pi)
    return dict(rate_per_second=high.tolist(),
                stencil_difference_per_second=float(np.linalg.norm(high - low)),
                initial_axis_error=anchor_error,
                initial_triad_error=triad_error,
                initial_carrier_frequency_hz=initial_frequency,
                carrier_frequency_relative_error=abs(initial_frequency / fref - 1),
                epoch_seconds=float(result[1].epoch))


def native_rate(backend, m1, m2, spin, fref):
    prefix = "torchwave" if backend == "torch" else "jaxwave"
    if backend == "torch":
        import torch
        def tensor(value):
            return torch.tensor(value, dtype=torch.float64)
    else:
        import jax
        import jax.numpy as jnp
        jax.config.update("jax_enable_x64", True)
        tensor = jnp.asarray
    rhs = importlib.import_module(prefix + ".approximants._phenomt_spin_evolution").spin_derivatives
    x1, x2 = m1 / (m1 + m2), m2 / (m1 + m2)
    ms = (m1 + m2) * MSUN_SEC
    v = (np.pi * ms * fref)**(1 / 3)
    rates = [rhs(tensor(v), tensor([0., 0., 1.]),
                 tensor(sign * x1**2 * spin), tensor([0., 0., 0.]),
                 x1, x2, include_n3lo_ss=False) for sign in (1., -1.)]
    positive, negative = [np.asarray(row[0]) / ms for row in rates]
    even_spin = (np.asarray(rates[0][1]) + np.asarray(rates[1][1])) / (2 * ms * x1**2)
    return dict(even_rate_per_second=((positive + negative) / 2).tolist(),
                odd_rate_per_second=((positive - negative) / 2).tolist(),
                chi1_even_rate_per_second=even_spin.tolist())


def spin_taylor_control(m1, m2, spin, fref, rate, kappa):
    """Positive control: a public binary API with explicit quadrupole inputs.

    spinO=5 retains the leading quadrupole term and excludes v8. Its SO
    truncation and frequency driver differ from TP; only initial spin parity
    is compared. Component names are checked rather than assuming list order.
    """
    signed = []
    for sign in (1., -1.):
        result = ls.SimInspiralSpinTaylorPNEvolveOrbit(
            1 / rate, m1 * lal.MSUN_SI, m2 * lal.MSUN_SI, fref, 1.05 * fref,
            *(sign * spin), 0., 0., 0., 0., 0., 1., 1., 0., 0.,
            0., 0., kappa, kappa, 5, 0, 7, 0, ls.SpinTaylorT4,
        )
        series = {item.name: item for item in result}
        chi = np.stack([series[f"SPIN1_{axis}_COMPONENT"].data.data[:9]
                        for axis in ("X", "Y", "Z")], axis=-1)
        np.testing.assert_allclose(chi[0], sign * spin, rtol=0, atol=1e-12)
        high = endpoint_rate(chi, 1 / rate)
        low = endpoint_rate(chi[:5], 1 / rate)
        signed.append(dict(chi1_rate_per_second=high.tolist(),
                           stencil_difference_per_second=float(np.linalg.norm(high - low))))
    even = sum(np.array(row["chi1_rate_per_second"]) for row in signed) / 2
    return dict(kappa1=kappa, kappa2=kappa, signed_rates=signed,
                chi1_even_rate_per_second=even.tolist())


def published_odd_prediction(m1, m2, spin, fref, so_order, l_order):
    """Evaluate the published one-spin SO numerator and L denominator.

    This is deliberately an independent diagnostic transcription of Sturani
    T1500554v23 Eqs. (3), (5), (22), and (25), rather than a call into the native
    RHS.  The ``so_order`` choices retain terms through v^5, v^7, or v^9 and
    ``l_order`` selects the published nonspin 2PN or 3PN L factor.
    """
    total_mass = m1 + m2
    x1 = m1 / total_mass
    eta = x1 * (1 - x1)
    v = (math.pi * total_mass * MSUN_SEC * fref) ** (1 / 3)
    c1 = 1.5 - x1 - 0.5 * x1**2
    q1 = 9 / 8 - x1 / 2 + 7 * x1**2 / 12 - 7 * x1**3 / 6 - x1**4 / 24
    r1 = (27 / 16 - 51 * x1 / 8 + 181 * x1**2 / 16 - 23 * x1**3 / 6
          - 39 * x1**4 / 16 - 3 * x1**5 / 8 - x1**6 / 48)
    numerator = c1 * v**5
    if so_order >= 7:
        numerator += q1 * v**7
    if so_order >= 9:
        numerator += r1 * v**9
    l_factor = 1 + (1.5 + eta / 6) * v**2
    if l_order >= 2:
        l_factor += (27 / 8 - 19 * eta / 8 + eta**2 / 24) * v**4
    if l_order >= 3:
        l_factor += (135 / 16 + (-6889 / 144 + 41 * math.pi**2 / 24) * eta
                     + 31 * eta**2 / 24 + 7 * eta**3 / 1296) * v**6
    angular_momentum = eta / v * l_factor
    spin_vector = x1**2 * np.asarray(spin, dtype=float)
    cross = np.cross(np.array([0., 0., 1.]), spin_vector)
    rate = -numerator * cross / (angular_momentum * total_mass * MSUN_SEC)
    return rate


def odd_rate_sweep():
    """Return the bounded odd-rate publication-order sweep definition."""
    return (
        dict(label="equal_mass", m1=30., m2=30., spin=(.3, 0., .4), inc=1.0),
        dict(label="single_spin", m1=35., m2=20., spin=(.3, 0., .4), inc=1.1),
        dict(label="high_mass_ratio", m1=60., m2=6., spin=(.2, .1, .8), inc=1.2),
    )


def run_odd_rate_sweep(backends=("torch", "jax"), rate=16384):
    """Compare binary odd rates with unmodified published SO/L truncations."""
    frequencies = (20., 30., 40.)
    amplitudes = (.5, 1.)
    rows = []
    for case in odd_rate_sweep():
        for frequency in frequencies:
            for amplitude in amplitudes:
                spin = amplitude * np.asarray(case["spin"], dtype=float)
                signed = [binary_rate(case["m1"], case["m2"], sign * spin,
                                      case["inc"], 0., frequency, rate)
                          for sign in (1., -1.)]
                observed = (np.asarray(signed[0]["rate_per_second"])
                            - np.asarray(signed[1]["rate_per_second"])) / 2
                uncertainty = 0.5 * sum(
                    row["stencil_difference_per_second"] for row in signed)
                predictions = {}
                for so_order, so_label in ((5, "v5"), (7, "v5_v7"), (9, "v5_v7_v9")):
                    for l_order, l_label in ((2, "L2PN"), (3, "L3PN")):
                        vector = published_odd_prediction(
                            case["m1"], case["m2"], spin, frequency, so_order, l_order)
                        predictions[f"{so_label}_{l_label}"] = dict(
                            rate_per_second=vector.tolist(),
                            norm_per_second=float(np.linalg.norm(vector)),
                        )
                reference = np.asarray(predictions["v5_v7_v9_L3PN"]["rate_per_second"])
                angle = math.atan2(float(np.linalg.norm(np.cross(observed, reference))),
                                   float(np.dot(observed, reference)))
                native_comparison = {}
                for backend in backends:
                    native = np.asarray(native_rate(
                        backend, case["m1"], case["m2"], spin, frequency)["odd_rate_per_second"])
                    difference = native - reference
                    native_comparison[backend] = dict(
                        rate_per_second=native.tolist(),
                        norm_per_second=float(np.linalg.norm(native)),
                        max_abs_difference_per_second=float(np.max(np.abs(difference))),
                        relative_difference=float(np.linalg.norm(difference) / np.linalg.norm(reference)),
                    )
                rows.append(dict(
                    case=case["label"], mass1=case["m1"], mass2=case["m2"],
                    spin1=spin.tolist(), spin_amplitude_scale=amplitude,
                    spin2=[0., 0., 0.], inclination=case["inc"],
                    phic=0., distance_mpc=100., f_lower_hz=frequency,
                    f_ref_hz=frequency, sample_rate_hz=rate,
                    observed_odd_rate_per_second=observed.tolist(),
                    observed_odd_norm_per_second=float(np.linalg.norm(observed)),
                    direction_angle_to_v9_l3pn_rad=angle,
                    stencil_uncertainty_per_second=uncertainty,
                    signed_frequency_metrics=[dict(
                        sign=sign,
                        initial_carrier_frequency_hz=row["initial_carrier_frequency_hz"],
                        carrier_frequency_relative_error=row["carrier_frequency_relative_error"],
                        epoch_seconds=row["epoch_seconds"],
                    ) for sign, row in zip((1, -1), signed)],
                    native_full_formula_comparison=native_comparison,
                    predictions=predictions,
                ))
    by_amplitude = {(row["case"], row["f_ref_hz"], row["spin_amplitude_scale"]): row
                    for row in rows}
    for row in rows:
        half = by_amplitude[(row["case"], row["f_ref_hz"], .5)]
        full = by_amplitude[(row["case"], row["f_ref_hz"], 1.)]
        row["amplitude_double_half_relative_residual"] = (
            full["observed_odd_norm_per_second"]
            / (2 * half["observed_odd_norm_per_second"]) - 1)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=tuple(CASES), nargs="+",
                        default=["single_spin", "high_mass_ratio"])
    parser.add_argument("--sample-rate", type=int, nargs="+", default=[2048, 16384])
    parser.add_argument("--backend", choices=("torch", "jax"), nargs="+", default=["torch", "jax"])
    parser.add_argument("--phase", type=float, nargs="+", default=[0., .7])
    parser.add_argument("--odd-rate-sweep", action="store_true",
                        help="Run the bounded published SO numerator/L denominator sweep")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(rate <= 0 for rate in args.sample_rate):
        parser.error("Sample rates must be positive")
    results, positive_controls = [], []
    for case in args.case:
        m1, m2, s1, _, inc, _, fref = CASES[case]
        for control in (False, True):
            spin = np.array(s1)
            if control:
                spin[2] = 0.  # Self-quadrupole torque vanishes in the orbital plane.
            native = {backend: native_rate(backend, m1, m2, spin, fref)
                      for backend in args.backend}
            for rate in args.sample_rate:
                if not control:
                    positive_controls.append(dict(
                        case=case, sample_rate_hz=rate,
                        mass1=m1, mass2=m2, spin1=spin.tolist(), spin2=[0., 0., 0.],
                        f_start_hz=fref, f_end_hz=1.05 * fref, native=native,
                        binary=[spin_taylor_control(m1, m2, spin, fref, rate, kappa)
                                for kappa in (0., 1.)],
                    ))
                for phase in args.phase:
                    signed = [binary_rate(m1, m2, sign * spin, inc, phase, fref, rate)
                              for sign in (1., -1.)]
                    positive, negative = [np.array(row["rate_per_second"]) for row in signed]
                    even, odd = (positive + negative) / 2, (positive - negative) / 2
                    config = parameters(case) | dict(spin1=spin.tolist(), spin2=[0., 0., 0.],
                                                     f_lower_hz=fref, phic=phase)
                    results.append(dict(
                        case=case, in_plane_control=control, parameters=config,
                        sample_rate_hz=rate, signed_binary_rates=signed,
                        binary_even_rate_per_second=even.tolist(),
                        binary_odd_rate_per_second=odd.tolist(), native=native,
                        binary_even_to_odd_norm=float(np.linalg.norm(even) / np.linalg.norm(odd)),
                    ))
    receipt = dict(
        generated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        reproduction="python scripts/diagnose_phenomt_initial_rates.py --case " + " ".join(args.case)
                     + " --sample-rate " + " ".join(map(str, args.sample_rate))
                     + " --backend " + " ".join(args.backend)
                     + " --phase " + " ".join(map(str, args.phase))
                     + (" --odd-rate-sweep" if args.odd_rate_sweep else "")
                     + " --output receipt.json",
        versions=dict(python=platform.python_version(), numpy=np.__version__, lalsimulation=ls.__version__,
                      **{backend: importlib.import_module(backend).__version__ for backend in args.backend}),
        precession_version=300, final_spin_mod=4, dtype="float64", device="cpu",
        conventions=dict(
            derivative="Initial time derivative of the returned unit axis; 9-point one-sided "
                       "polynomial stencil, with a 5-point stencil sensitivity check.",
            spin_parity="Even=(rate(+S1)+rate(-S1))/2; odd=(rate(+S1)-rate(-S1))/2; "
                        "S2=0, fixed masses/f_ref/source triad. Each J-frame is rotated back "
                        "using its own initial beta and spin azimuth; initial Lhat=(0,0,1) and "
                        "the complete Rz(alpha0)Ry(beta0)Rz(gamma0) reference triad are checked.",
            physics="Sturani T1500554v23 Eq.21: native even rate is "
                    "(3/2)*kappa1*(1/x1-1)*v^6*(Lhat.S1)*(Lhat cross S1)/(L_NS*M_seconds), "
                    "where S1=x1^2*chi1, L_NS is the nonspin 3PN angular momentum and kappa1=1. "
                    "The v8 term is excluded as in production PV300.",
            scope="Binary diagnostic only. Near-zero spin-even behavior does not uniquely identify "
                  "the reference equations, internal quadrupole parameter, or denominator. "
                  "Finite differences, reference-frequency anchoring, spin-dependent denominators "
                  "and axis definitions can affect rates. No production equations are changed.",
            positive_control="SpinTaylorT4 public binary with explicit quadparam1/2=0 or 1, "
                             "spinO=5, tideO=0, phaseO=7, lscorr=0, lambda1/2=0. "
                             "Spin vectors are chi_i=S_i/m_i^2. This directly measures "
                             "spin parity without Euler-frame reconstruction; only its initial "
                             "spin-even rate is compared with native PV300, not its full dynamics.",
            odd_rate_sweep="Opt-in sweep uses fixed sample rate 16384 Hz, phase 0, and zero "
                           "companion spin. Rates and stencil uncertainties are in 1/s. It "
                           "records signed carrier-frequency anchoring metrics from binary_rate "
                           "and compares the selected native frontends' odd rates with the independent full "
                           "v5+v7+v9 over L3PN formula; no residual fitting is performed. "
                           "The SO numerator follows Sturani Eqs. (3), (22), and (25), with "
                           "nonspin L from Eq. (5) and the PhenomTP Eq. (13) constraint. "
                           "Amplitude residuals compare full/(2*half)-1 for each mass/frequency pair.",
        ), results=results, spin_taylor_positive_controls=positive_controls,
    )
    if args.odd_rate_sweep:
        receipt["odd_rate_sweep"] = run_odd_rate_sweep(args.backend)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()

"""Isolate PhenomTPHM mode rotation from spin evolution and carrier errors.

The public LAL binary supplies coprecessing modes, Euler-angle series, and
already-rotated J-frame modes.  This diagnostic feeds the first two through
the repository's ``project_modes`` helper and compares the result with a
direct spin-weighted-harmonic projection of the public J-frame modes.  It is
an isolated rotation/convention check; it makes no claim about native
precession accuracy.  No LAL implementation source is read.
"""

import argparse
import datetime
import importlib
import json
from pathlib import Path
import platform
import sys

import lal
import lalsimulation as ls
import numpy as np

try:
    # Repository-root imports make the helper reusable from pytest.
    from scripts.validate_phenomt_precession import CASES, parameters
except ModuleNotFoundError:
    # Direct execution puts this file's directory on sys.path instead.
    from validate_phenomt_precession import CASES, parameters


def _mode_keys(series, only22):
    max_l = 2 if only22 else 5
    return tuple(
        (ell, m) for ell in range(2, max_l + 1)
        for m in range(-ell, ell + 1)
        if ls.SphHarmTimeSeriesGetMode(series, ell, m)
    )


def _public_case(case, rate, only22):
    m1, m2, s1, s2, inc, phase, fref = CASES[case]
    options = lal.CreateDict()
    ls.SimInspiralWaveformParamsInsertPhenomXPrecVersion(options, 300)
    ls.SimInspiralWaveformParamsInsertPhenomXPFinalSpinMod(options, 4)
    args = (m1 * lal.MSUN_SI, m2 * lal.MSUN_SI, *s1, *s2,
            100e6 * lal.PC_SI, inc, 1 / rate, 20., fref, phase, options,
            only22)
    coprec, alpha, cosbeta, gamma, _ = ls.SimIMRPhenomTPHM_CoprecModes(*args)
    jmodes, ja, jb, jg, _ = ls.SimIMRPhenomTPHM_JModes(*args)
    angle_series = (alpha, cosbeta, gamma, ja, jb, jg)
    for series in angle_series:
        if series.data.length != alpha.data.length or not np.isclose(
            float(series.epoch), float(alpha.epoch), rtol=0., atol=1e-12
        ) or not np.isclose(series.deltaT, alpha.deltaT, rtol=0., atol=1e-18):
            raise RuntimeError(f"inconsistent binary angle grids for {case}")
    keys = _mode_keys(coprec, only22)
    if keys != _mode_keys(jmodes, only22):
        raise RuntimeError(f"coprec/J mode sets differ for {case}")
    l0modes = ls.SimIMRPhenomTPHM_L0Modes(*args) if np.isclose(fref, 20.0) else None
    if l0modes is not None and keys != _mode_keys(l0modes, only22):
        raise RuntimeError(f"J/L0 mode sets differ for {case}")
    for series in (coprec, jmodes, l0modes):
        if series is None:
            continue
        for key in keys:
            mode = ls.SphHarmTimeSeriesGetMode(series, *key)
            if mode.data.length != alpha.data.length or not np.isclose(
                float(mode.epoch), float(alpha.epoch), rtol=0., atol=1e-12
            ) or not np.isclose(mode.deltaT, alpha.deltaT, rtol=0., atol=1e-18):
                raise RuntimeError(f"inconsistent mode grid for {case} mode={key}")
    times = float(alpha.epoch) + alpha.deltaT * np.arange(alpha.data.length)
    return dict(
        args=args, keys=keys, times=times,
        coprec={key: ls.SphHarmTimeSeriesGetMode(coprec, *key).data.data for key in keys},
        jmode={key: ls.SphHarmTimeSeriesGetMode(jmodes, *key).data.data for key in keys},
        l0=({key: ls.SphHarmTimeSeriesGetMode(l0modes, *key).data.data for key in keys}
            if l0modes is not None else None),
        fref=fref,
        alpha=alpha.data.data, beta=np.arccos(np.clip(cosbeta.data.data, -1., 1.)),
        gamma=gamma.data.data,
    )


def _direct_j_projection(data, theta, zeta):
    """Project public J-frame modes with the public LAL harmonic callable."""
    strain = np.zeros_like(next(iter(data["jmode"].values())), dtype=np.complex128)
    for (ell, m), mode in data["jmode"].items():
        harmonic = lal.SpinWeightedSphericalHarmonic(theta, 0., -2, ell, m)
        strain += mode * harmonic * np.exp(2j * zeta)
    return strain


def measure(backend, case, rate, only22, views=((0.3, 0.2), (1.1, -0.4), (2.0, 1.0)), samples=0):
    data = _public_case(case, rate, only22)
    prefix = "torchwave" if backend == "torch" else "jaxwave"
    precession = importlib.import_module(prefix + ".approximants._phenomt_precession")
    if backend == "torch":
        import torch

        def tensor(x):
            return torch.as_tensor(x, dtype=torch.float64)

        def complex_tensor(x):
            return torch.as_tensor(x, dtype=torch.complex128)

        def array(x):
            return x.detach().cpu().numpy()
    else:
        import jax
        import jax.numpy as jnp
        jax.config.update("jax_enable_x64", True)

        def tensor(x):
            return jnp.asarray(x, dtype=jnp.float64)

        def complex_tensor(x):
            return jnp.asarray(x, dtype=jnp.complex128)

        def array(x):
            return np.asarray(x)
    if samples and samples < len(data["times"]):
        indices = np.linspace(0, len(data["times"]) - 1, samples, dtype=int)
    else:
        indices = np.arange(len(data["times"]))
    modes = {key: complex_tensor(value[indices]) for key, value in data["coprec"].items()}
    alpha, beta, gamma = (tensor(data[key][indices]) for key in ("alpha", "beta", "gamma"))
    input_mode_norm = float(np.linalg.norm(np.concatenate(
        [value[indices] for value in data["coprec"].values()]
    )))
    rows = []
    for theta, zeta in views:
        hp, hc = precession.project_modes(
            modes, alpha, beta, gamma, tensor(theta), tensor(zeta),
        )
        actual = array(hp - 1j * hc)
        expected = _direct_j_projection(data, theta, zeta)[indices]
        error = actual - expected
        rows.append(dict(
            theta_jn=theta, zeta=zeta,
            relative_l2=float(np.linalg.norm(error) / np.linalg.norm(expected)),
            max_absolute=float(np.max(np.abs(error))),
            reference_norm=float(np.linalg.norm(expected)),
            input_mode_norm=input_mode_norm,
            samples=len(indices),
        ))
    if data["l0"] is None:
        inverse_l0 = dict(
            status="skipped",
            reason="f_ref differs from f_min=20 Hz; the first sample is not the reference orientation.",
            views=[],
        )
    else:
        # Eq. 8 inverse rotation: (J -> L0) uses constants
        # (-gamma_ref, -beta_ref, -alpha_ref).  The reference is the first
        # sample here because f_ref=f_min=20 Hz.
        alpha_ref, beta_ref, gamma_ref = (data[key][0] for key in ("alpha", "beta", "gamma"))
        j_modes = {key: complex_tensor(value[indices]) for key, value in data["jmode"].items()}
        inverse_alpha = tensor(np.full(len(indices), -gamma_ref))
        inverse_beta = tensor(np.full(len(indices), -beta_ref))
        inverse_gamma = tensor(np.full(len(indices), -alpha_ref))
        inverse_views = []
        for theta, zeta in views:
            hp, hc = precession.project_modes(
                j_modes, inverse_alpha, inverse_beta, inverse_gamma,
                tensor(theta), tensor(zeta),
            )
            actual = array(hp - 1j * hc)
            expected = np.zeros(len(indices), dtype=np.complex128)
            for key, mode in data["l0"].items():
                ell, m = key
                expected += mode[indices] * lal.SpinWeightedSphericalHarmonic(
                    theta, 0., -2, ell, m,
                ) * np.exp(2j * zeta)
            error = actual - expected
            per_ell = []
            for ell in sorted({key[0] for key in data["jmode"]}):
                ell_modes = {
                    key: value for key, value in j_modes.items() if key[0] == ell
                }
                ell_hp, ell_hc = precession.project_modes(
                    ell_modes, inverse_alpha, inverse_beta, inverse_gamma,
                    tensor(theta), tensor(zeta),
                )
                ell_actual = array(ell_hp - 1j * ell_hc)
                ell_expected = np.zeros(len(indices), dtype=np.complex128)
                for key, mode in data["l0"].items():
                    if key[0] != ell:
                        continue
                    _, m = key
                    ell_expected += mode[indices] * lal.SpinWeightedSphericalHarmonic(
                        theta, 0., -2, ell, m,
                    ) * np.exp(2j * zeta)
                ell_error = ell_actual - ell_expected
                per_ell.append(dict(
                    ell=ell,
                    projected_relative_l2=float(
                        np.linalg.norm(ell_error) / np.linalg.norm(ell_expected)
                    ),
                    max_absolute=float(np.max(np.abs(ell_error))),
                    reference_norm=float(np.linalg.norm(ell_expected)),
                    input_j_mode_norm=float(np.linalg.norm(np.concatenate(
                        [value[indices] for key, value in data["jmode"].items()
                         if key[0] == ell]
                    ))),
                ))
            inverse_views.append(dict(
                theta_l0=theta, zeta=zeta,
                relative_l2=float(np.linalg.norm(error) / np.linalg.norm(expected)),
                max_absolute=float(np.max(np.abs(error))),
                reference_norm=float(np.linalg.norm(expected)),
                input_j_mode_norm=float(np.linalg.norm(np.concatenate(
                    [value[indices] for value in data["jmode"].values()]
                ))),
                per_ell=per_ell,
                samples=len(indices),
            ))
        inverse_l0 = dict(
            status="measured",
            reason="Full signed-mode inverse residual is reported for diagnosis; no full-mode accuracy gate is claimed.",
            views=inverse_views,
        )
    return dict(case=case, backend=backend, only22=only22, sample_rate_hz=rate,
                modes=[list(key) for key in data["keys"]], views=rows,
                inverse_l0=inverse_l0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("torch", "jax"), nargs="+", default=("torch", "jax"))
    parser.add_argument("--case", choices=tuple(CASES), nargs="+", default=tuple(CASES))
    parser.add_argument("--only22", type=int, choices=(0, 1), nargs="+", default=(1, 0),
                        help="LAL mode selection: 1 is 22-only; 0 requests all available modes.")
    parser.add_argument("--sample-rate", type=int, default=2048)
    parser.add_argument("--samples", type=int, default=0, help="Optional evenly spaced diagnostic samples.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.sample_rate <= 0 or args.samples < 0:
        parser.error("sample rate must be positive and samples nonnegative")
    versions = dict(python=platform.python_version(), numpy=np.__version__, lalsimulation=ls.__version__)
    for backend in args.backend:
        versions[backend] = importlib.import_module(backend).__version__
    results = []
    for case in args.case:
        for only22 in args.only22:
            for backend in args.backend:
                row = measure(backend, case, args.sample_rate, only22, samples=args.samples)
                results.append(row)
                worst = max(view["relative_l2"] for view in row["views"])
                print(f"{case}/only22={only22}/{backend}: rotation L2={worst:.3e}",
                      file=sys.stderr, flush=True)
                if row["inverse_l0"]["status"] == "measured":
                    inverse_worst = max(view["relative_l2"] for view in row["inverse_l0"]["views"])
                    print(f"{case}/only22={only22}/{backend}: inverse-L0 L2={inverse_worst:.3e}",
                          file=sys.stderr, flush=True)
    receipt = dict(
        generated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        reproduction="python scripts/diagnose_phenomt_rotation.py --backend "
                     + " ".join(args.backend) + " --case " + " ".join(args.case)
                     + " --only22 " + " ".join(map(str, args.only22))
                     + f" --sample-rate {args.sample_rate} --samples {args.samples} --output receipt.json",
        versions=versions, dtype="float64", device="cpu", precession_version=300,
        final_spin_mod=4, parameters={case: parameters(case) for case in args.case},
        scope="Isolated project_modes rotation/convention checks against public binary J-frame and L0 modes; "
              "does not validate native spin evolution, carrier, or physical waveform accuracy.",
        conventions={
            "angles": "Public alpha, acos(cosbeta), epsilon series are passed as alpha,beta,gamma.",
            "projection": "J-frame modes are projected at phi=0 with exp(2 i zeta), matching project_modes.",
            "modes": "only22=1 checks the complete l=2 set; only22=0 checks all available l=2..5 signed modes.",
            "relative_l2": "norm(project_modes output - direct J-frame projection)/norm(direct projection).",
            "inverse_l0": "For f_ref=f_min=20 Hz, Eq. 8 uses constants (-gamma_ref,-beta_ref,-alpha_ref) on all signed J modes and compares with public L0Modes.",
            "inverse_l0_status": "Full-mode inverse values are measured diagnostics; only22 inverse values are suitable for the strict convention check.",
        },
        results=results,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()

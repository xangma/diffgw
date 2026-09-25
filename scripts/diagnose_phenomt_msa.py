"""Measure MSA Euler-angle components against public LAL binary outputs.

The reference is used only through ``SimIMRPhenomTPHM_CoprecModes`` with
explicitly matched PV223xx/FinalSpinMod=3 options.  No binary implementation
source, coefficient fitting, time shift, or waveform alignment is used.
"""

from __future__ import annotations

import argparse
import datetime
import importlib
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

from diffgw_core.constants import MSUN_SEC
from scripts.validate_phenomt_precession import CASES, parameters


VERSIONS = (22300, 22310, 22311, 22321)


def _rotation_matrix(alpha, beta, gamma):
    """Active Rz(alpha) Ry(beta) Rz(gamma), TPHM Eqs. (4) and (8)."""
    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta), np.sin(beta)
    cg, sg = np.cos(gamma), np.sin(gamma)
    return np.stack((
        ca * cb * cg - sa * sg, -ca * cb * sg - sa * cg, ca * sb,
        sa * cb * cg + ca * sg, -sa * cb * sg + ca * cg, sa * sb,
        -sb * cg, sb * sg, cb,
    ), axis=-1).reshape(alpha.shape + (3, 3))


def _angle_metrics(actual, reference):
    residual = np.unwrap(actual) - np.unwrap(reference)
    residual -= residual[0]
    return dict(
        initial_offset_radians=float(np.angle(np.exp(1j * (actual[0] - reference[0])))),
        rms_anchored_radians=float(np.sqrt(np.mean(residual**2))),
        max_abs_anchored_radians=float(np.max(np.abs(residual))),
        final_anchored_radians=float(residual[-1]),
    )


def _beta_metrics(actual, reference):
    residual = actual - reference
    return dict(
        initial_radians=float(residual[0]),
        rms_radians=float(np.sqrt(np.mean(residual**2))),
        max_abs_radians=float(np.max(np.abs(residual))),
        final_radians=float(residual[-1]),
    )


def _rotation_metrics(actual, reference):
    # Remove each constant first-sample source/J-frame choice before comparing
    # the physical triad evolution.  This is not a fitted alignment.
    actual_relative = np.swapaxes(actual[0], -1, -2) @ actual
    reference_relative = np.swapaxes(reference[0], -1, -2) @ reference
    delta = np.swapaxes(reference_relative, -1, -2) @ actual_relative
    cosine = np.clip((np.trace(delta, axis1=-2, axis2=-1) - 1) / 2, -1, 1)
    angle = np.arccos(cosine)
    return dict(
        rms_radians=float(np.sqrt(np.mean(angle**2))),
        max_radians=float(np.max(angle)),
        final_radians=float(angle[-1]),
    )


def _segments(times, native, reference):
    result = {}
    for name, mask in (("all", np.ones(times.shape, dtype=bool)),
                       ("inspiral", times < 0), ("ringdown", times >= 0)):
        if not np.any(mask):
            continue
        result[name] = dict(
            samples=int(np.count_nonzero(mask)),
            alpha=_angle_metrics(native[0][mask], reference[0][mask]),
            beta=_beta_metrics(native[1][mask], reference[1][mask]),
            gamma=_angle_metrics(native[2][mask], reference[2][mask]),
            relative_rotation=_rotation_metrics(
                _rotation_matrix(*(value[mask] for value in native)),
                _rotation_matrix(*(value[mask] for value in reference)),
            ),
        )
    return result


def measure(backend, case, version, rate):
    prefix = "torchwave" if backend == "torch" else "jaxwave"
    model_module = importlib.import_module(prefix + ".approximants.imrphenomt")
    dispatch = importlib.import_module(prefix + ".approximants._phenomt_dispatch")
    msa = importlib.import_module(prefix + ".approximants._phenomt_msa")
    if backend == "torch":
        import torch

        def tensor(value):
            return torch.as_tensor(value, dtype=torch.float64)

        def array(value):
            return value.detach().numpy()
    else:
        import jax
        import jax.numpy as jnp
        jax.config.update("jax_enable_x64", True)

        def tensor(value):
            return jnp.asarray(value, dtype=jnp.float64)

        array = np.asarray

    m1, m2, s1, s2, inc, phase, fref = CASES[case]
    options = lal.CreateDict()
    ls.SimInspiralWaveformParamsInsertPhenomXPrecVersion(options, version)
    ls.SimInspiralWaveformParamsInsertPhenomXPFinalSpinMod(options, 3)
    args = (m1 * lal.MSUN_SI, m2 * lal.MSUN_SI, *s1, *s2,
            100e6 * lal.PC_SI, inc, 1 / rate, 20.0, fref, phase, options, 1)
    modes, alpha, cosbeta, gamma, _ = ls.SimIMRPhenomTPHM_CoprecModes(*args)
    mode22 = ls.SphHarmTimeSeriesGetMode(modes, 2, 2)
    times = float(mode22.epoch) + mode22.deltaT * np.arange(mode22.data.length)
    series = (alpha, cosbeta, gamma)
    for value in series:
        if (value.data.length != len(times)
                or not np.isclose(value.deltaT, mode22.deltaT, rtol=0, atol=1e-18)
                or not np.isclose(float(value.epoch), float(mode22.epoch), rtol=0, atol=1e-12)):
            raise RuntimeError(f"inconsistent public component grids for {case}/{version}")
    reference = (
        np.asarray(alpha.data.data),
        np.arccos(np.clip(cosbeta.data.data, -1, 1)),
        np.asarray(gamma.data.data),
    )

    model = model_module.IMRPhenomT()
    t = tensor(times)
    tm1, tm2 = tensor(m1), tensor(m2)
    tz1, tz2 = tensor(s1[2]), tensor(s2[2])
    state = model._coeff(tm1, tm2, tz1, tz2)
    total_seconds = (tm1 + tm2) * MSUN_SEC
    frequency = model._evaluate(t / total_seconds, state)[0] / (2 * math.pi * total_seconds)
    frequency_peak = state[13] / (2 * math.pi * total_seconds)
    final_spin = msa.msa_final_spin(tm1, tm2, *map(tensor, (*s1, *s2)), f_ref=tensor(fref))
    native = dispatch.precession_geometry(
        t, frequency, tm1, tm2, *map(tensor, (*s1, *s2)), tensor(inc), tensor(phase), tensor(fref),
        frequency_peak=frequency_peak, final_spin=final_spin,
        precession_method="msa", precession_version=version,
        frequency_at_time=lambda query: dispatch.carrier_frequency(model, state, tm1, tm2, query),
    )[:3]
    native = tuple(np.asarray(array(value)) for value in native)
    if not all(np.isfinite(value).all() for value in (*native, *reference)):
        raise RuntimeError(f"nonfinite MSA component for {backend}/{case}/{version}")
    return dict(
        backend=backend,
        case=case,
        sample_rate_hz=rate,
        precession_version=version,
        final_spin_mod=3,
        samples=len(times),
        epoch_seconds=float(mode22.epoch),
        native_final_spin=float(np.asarray(array(final_spin))),
        components=_segments(times, native, reference),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("torch", "jax"), nargs="+", default=["torch", "jax"])
    parser.add_argument("--case", choices=tuple(CASES), nargs="+", default=["baseline", "high_mass_ratio"])
    parser.add_argument("--version", choices=VERSIONS, type=int, nargs="+", default=list(VERSIONS))
    parser.add_argument("--sample-rate", type=int, default=2048)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.sample_rate <= 0:
        parser.error("sample rate must be positive")
    versions = dict(python=platform.python_version(), numpy=np.__version__, lalsimulation=ls.__version__)
    devices = {}
    for backend in args.backend:
        module = importlib.import_module(backend)
        versions[backend] = module.__version__
        devices[backend] = ["cpu"] if backend == "torch" else [str(device) for device in module.devices()]
    results = []
    for case in args.case:
        for version in args.version:
            for backend in args.backend:
                row = measure(backend, case, version, args.sample_rate)
                results.append(row)
                values = row["components"]
                print(f"{case}/{version}/{backend}: inspiral rotation max="
                      f"{values['inspiral']['relative_rotation']['max_radians']:.8g}, "
                      f"ringdown max={values['ringdown']['relative_rotation']['max_radians']:.8g}",
                      file=sys.stderr, flush=True)
    receipt = dict(
        generated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        reproduction="PYTHONPATH=src python scripts/diagnose_phenomt_msa.py --backend "
        + " ".join(args.backend) + " --case " + " ".join(args.case) + " --version "
        + " ".join(map(str, args.version)) + f" --sample-rate {args.sample_rate} --output receipt.json",
        versions=versions,
        dtype="float64",
        devices=devices,
        parameters={case: parameters(case) for case in args.case},
        conventions=dict(
            reference="Public SimIMRPhenomTPHM_CoprecModes binary output with explicitly matched "
                      "PhenomXPrecVersion and PhenomXPFinalSpinMod=3; no source inspection.",
            components="Alpha and gamma residuals are unwrapped and anchored only by subtracting "
                       "their first residual; beta is compared directly. No fitted alignment.",
            relative_rotation="Geodesic angle between R(first)^T R(t) triads. Each fixed first-sample "
                              "frame is removed by definition, with no optimized rotation or time shift.",
            segmentation="Physical time zero is the aligned-carrier amplitude peak; inspiral t<0 and "
                         "ringdown t>=0 are reported separately.",
        ),
        results=results,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()

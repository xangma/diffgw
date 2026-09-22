"""Separate numerical PhenomTP carrier, angle, and integration errors.

Uses only public LAL binary calls. The co-precessing mode has an independent
phase convention: its phase is anchored at the first sample, with no fit or
time/amplitude adjustment. Polarization errors remain completely unaligned.
No production coefficients or equations are changed by this diagnostic.
"""

import argparse
import datetime
from functools import partial
import importlib
import json
import math
from pathlib import Path
import platform
import sys
from unittest.mock import patch

import lal
import lalsimulation as ls
import numpy as np

from validate_phenomt_precession import CASES, metrics, parameters


def diagnose(backend, case, rate, backward_counts, kappa):
    prefix = "torchwave" if backend == "torch" else "jaxwave"
    frontend = importlib.import_module(prefix)
    numerical = importlib.import_module(prefix + ".approximants._phenomt_numerical")
    if backend == "torch":
        import torch
        tensor = torch.from_numpy

        def array(value):
            return value.detach().numpy()
    else:
        import jax
        import jax.numpy as jnp
        jax.config.update("jax_enable_x64", True)
        tensor, array = jnp.asarray, np.asarray

    m1, m2, s1, s2, inc, phase, fref = CASES[case]
    options = lal.CreateDict()
    ls.SimInspiralWaveformParamsInsertPhenomXPrecVersion(options, 300)
    ls.SimInspiralWaveformParamsInsertPhenomXPFinalSpinMod(options, 4)
    arguments = (m1 * lal.MSUN_SI, m2 * lal.MSUN_SI, *s1, *s2,
                 100e6 * lal.PC_SI, inc, 1 / rate, 20., fref, phase, options)
    coprec = ls.SimIMRPhenomTPHM_CoprecModes(*arguments, 1)
    mode = ls.SphHarmTimeSeriesGetMode(coprec[0], 2, 2)
    times = float(mode.epoch) + np.arange(mode.data.length) * mode.deltaT
    reference_mode = mode.data.data
    reference_beta = np.arccos(np.clip(coprec[2].data.data, -1, 1))
    for series in coprec[1:4]:
        if (series.data.length != len(times) or series.deltaT != mode.deltaT
                or series.epoch != mode.epoch):
            raise RuntimeError("Reference component grids do not match")
    # Use beta only: it is invariant under constant azimuthal frame rotations.
    selected = [0, np.argmin(abs(times + 1)), np.argmin(abs(times + .1)),
                np.argmin(abs(times))]
    hp, hc = ls.SimIMRPhenomTP(*arguments)
    strain_times = float(hp.epoch) + np.arange(hp.data.length) * hp.deltaT
    reference_strain = hp.data.data - 1j * hc.data.data
    results = []
    previous = None
    for nb in backward_counts:
        nf = nb // 2
        # Discrete equation sensitivity test, isolated to trajectory preparation.
        # kappa=0 suppresses the self-quadrupole term; it is not a BH waveform.
        # The reference always retains its ordinary PV300/FS4 options.
        rhs = partial(numerical.spin_derivatives,
                      quadrupole_kappa1=kappa, quadrupole_kappa2=kappa)
        with patch.object(numerical, "spin_derivatives", rhs):
            prepared = numerical.prepare_numerical_trajectory(
                m1, m2, *s1, *s2, fref, n_backward=nb, n_forward=nf,
            )
        final_spin = float(prepared["final_spin"])
        a, b = frontend.IMRPhenomT().time_domain(
            tensor(times), m1, m2, s1[2], s2[2], 100., 0., 0., 0., fref,
            _final_spin_override=prepared["final_spin"],
        )
        actual_mode = -array(a - 1j * b) / math.sqrt(5 / (4 * math.pi))
        phase_anchor = float(np.angle(reference_mode[0] / actual_mode[0]))
        actual_mode = actual_mode * np.exp(1j * phase_anchor)
        geometry = numerical.numerical_geometry(
            tensor(times[selected]), None, m1, m2, *s1, *s2, inc, phase, fref,
            prepared=prepared,
        )
        beta = array(geometry[1])
        a, b = frontend.IMRPhenomTP().time_domain(
            tensor(strain_times), m1, m2, chi1=s1[2], chi2=s2[2],
            spin1x=s1[0], spin1y=s1[1], spin2x=s2[0], spin2y=s2[1],
            distance=100., inclination=inc, phic=phase, f_ref=fref,
            precession_method="numerical", precession_version=300,
            _precession_state=prepared,
        )
        strain = array(a - 1j * b)
        if not all(np.isfinite(x).all() for x in (actual_mode, beta, strain)):
            raise RuntimeError(f"Nonfinite components for {case}/{backend}")
        states = array(prepared["states"])
        lhat, spin1, spin2, tangent = np.split(states, 4, axis=-1)
        norm_errors = {}
        for name, vector, target in (
            ("lhat", lhat, 1.), ("tangent", tangent, 1.),
            ("spin1", spin1, np.linalg.norm(array(prepared["initial"])[3:6])),
            ("spin2", spin2, np.linalg.norm(array(prepared["initial"])[6:9])),
        ):
            norm_errors[name] = float(np.max(abs(np.linalg.norm(vector, axis=-1) - target)))
        norm_errors["lhat_dot_tangent"] = float(np.max(abs(np.sum(lhat * tangent, axis=-1))))
        row = dict(
            case=case, backend=backend, sample_rate_hz=rate,
            backward_steps=nb, forward_steps=nf, native_final_spin=final_spin,
            native_quadrupole_kappa1=kappa, native_quadrupole_kappa2=kappa,
            samples=len(times), epoch_seconds=float(mode.epoch),
            mode_phase_anchor_radians=phase_anchor,
            coprec22={label: metrics(actual_mode[mask], reference_mode[mask])
                      for label, mask in (("all", np.ones(len(times), dtype=bool)),
                                          ("inspiral", times < 0), ("ringdown", times >= 0))},
            coprec22_amplitude_relative_l2=float(
                np.linalg.norm(abs(actual_mode) - abs(reference_mode)) / np.linalg.norm(reference_mode)),
            beta_sample_times_seconds=times[selected].tolist(),
            beta_difference_radians=(beta - reference_beta[selected]).tolist(),
            polarizations=metrics(strain, reference_strain),
            invariant_absolute_errors=norm_errors,
        )
        if previous is not None:
            row["relative_l2_from_previous_resolution"] = metrics(strain, previous)["relative_l2"]
        previous = strain
        results.append(row)
        print(f"{case}/{backend}/kappa={kappa}/{nb}+{nf} steps: carrier L2="
              f"{row['coprec22']['all']['relative_l2']:.8g}, strain L2="
              f"{row['polarizations']['relative_l2']:.8g}", file=sys.stderr, flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("torch", "jax"), nargs="+", default=["torch", "jax"])
    parser.add_argument("--case", choices=tuple(CASES), nargs="+", default=list(CASES))
    parser.add_argument("--sample-rate", type=int, default=2048)
    parser.add_argument("--backward-steps", type=int, nargs="+", default=[2048])
    parser.add_argument("--quadrupole-kappa", type=float, choices=(0., 1.), nargs="+", default=[1.],
                        help="Native equation diagnostic only: 1 is the BH value; 0 suppresses "
                             "self-quadrupole coupling. Reference settings remain unchanged.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.sample_rate <= 0 or any(n < 2 or n % 2 for n in args.backward_steps):
        parser.error("Sample rate must be positive; step counts must be even and at least two")
    counts = sorted(set(args.backward_steps))
    versions = dict(python=platform.python_version(), numpy=np.__version__, lalsimulation=ls.__version__)
    for backend in args.backend:
        versions[backend] = importlib.import_module(backend).__version__
    results = []
    for case in args.case:
        for backend in args.backend:
            for kappa in args.quadrupole_kappa:
                results.extend(diagnose(backend, case, args.sample_rate, counts, kappa))
    receipt = dict(
        generated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        reproduction="python scripts/diagnose_phenomt_precession.py --backend " + " ".join(args.backend)
                     + " --case " + " ".join(args.case) + f" --sample-rate {args.sample_rate}"
                     + " --backward-steps " + " ".join(map(str, counts))
                     + " --quadrupole-kappa " + " ".join(map(str, args.quadrupole_kappa))
                     + " --output receipt.json",
        versions=versions, dtype="float64", device="cpu", precession_version=300, final_spin_mod=4,
        parameters={case: parameters(case) for case in args.case},
        conventions=dict(
            coprec22="Native h22 uses the native evolved final spin. One phase rotation fixes the "
                     "first sample to the reference phase; no time shift, amplitude scaling, or fit.",
            polarizations="IMRPhenomTP hplus-i*hcross; no alignment or scaling.",
            relative_l2="norm(actual-reference)/norm(reference)",
            mismatch="1-abs(vdot(actual,reference))/(norm(actual)*norm(reference)); "
                     "unweighted, constant-phase maximization only, no time shift",
            beta="Native minus reference opening angle at the listed sample times.",
            resolution="Forward step count is half the backward count; both cover the full "
                       "intrinsic interval. Consecutive native waveforms use the same output grid.",
            quadrupole="Native kappa=1 is the production black-hole prescription. kappa=0 only "
                       "diagnoses sensitivity to the published self-quadrupole term; it is not "
                       "a supported production waveform or evidence of the reference equations.",
        ),
        results=results,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()

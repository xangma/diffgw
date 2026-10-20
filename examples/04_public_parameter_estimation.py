#!/usr/bin/env python3
"""Public PE API: IMRPhenomD mass recovery in a zero-noise H1/L1 example.

Use --backend torch --device cuda or --backend jax --device gpu for GPU PE.
This short example demonstrates execution, not posterior qualification.
"""

import argparse
from contextlib import nullcontext
from importlib import import_module
import json
from pathlib import Path

import numpy as np

import diffgw
from diffgw.inference import Posterior, UniformBoxPrior, WhittleLikelihood, run_nuts


def build_problem(backend="torch", device="cpu"):
    """Build a dense two-mass example; other waveform and detector inputs fixed."""
    prior = UniformBoxPrior({"mass1": (29.5, 30.5), "mass2": (24.5, 25.5)})
    # The callback is evaluated only after all device-resident setup completes.
    posterior = Posterior(lambda p: likelihood(templates(p)), prior, backend=backend, device=device)
    framework = import_module(backend)
    frontend = getattr(diffgw, backend)
    detectors = import_module(f"{frontend.__name__}.detectors")
    context = framework.default_device(posterior.device) if backend == "jax" else nullcontext()
    with context:
        frequencies = posterior.asarray(np.arange(20., 129., 2.))
        network = detectors.DetectorNetwork(["H1", "L1"])
        if backend == "torch":
            network = network.to(posterior.device)
        psd = detectors.aLIGOZeroDetHighPower(frequencies)

        def templates(p):
            hp, hc = diffgw.get_fd_waveform(
                "IMRPhenomD", backend=backend, sample_frequencies=frequencies,
                mass1=p["mass1"], mass2=p["mass2"], distance=1000.,
                chi1=0., chi2=0., inclination=0.4,
            )
            return network.project_fd_strain(
                f=frequencies, h_plus=hp, h_cross=hc,
                ra=1.5, dec=0.3, psi=0.8, t_gps=1126259462.,
            )

        truth = posterior.parameters(posterior.asarray([30., 25.]))
        data = {key: value.reshape(-1) for key, value in templates(truth).items()}
        likelihood = WhittleLikelihood(data, {key: psd for key in data}, df=2., backend=backend, device=device)
    return posterior


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["torch", "jax"], default="torch")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--draws", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=300)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path, help="Optional compressed samples and JSON metadata archive")
    args = parser.parse_args()
    problem = build_problem(args.backend, args.device)
    result = run_nuts(problem, [[29.9, 25.1], [30.1, 24.9]], num_samples=args.draws,
                      num_warmup=args.warmup, seed=args.seed, max_tree_depth=6)
    exported = result.to_numpy()
    print(json.dumps({"metadata": result.metadata,
                      "posterior_mean": dict(zip(result.names, exported["samples"].mean(axis=(0, 1)).tolist()))}, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output, samples=exported["samples"], names=np.asarray(result.names),
                            metadata=np.asarray(json.dumps(result.metadata)),
                            **{f"diagnostic_{key}": value for key, value in exported["diagnostics"].items()})


if __name__ == "__main__":
    main()

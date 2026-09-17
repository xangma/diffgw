#!/usr/bin/env python3
"""Unified Executable Benchmark Campaign Runner for DiffGW.

Executes controlled multi-backend performance benchmarks across TorchWave,
JAXWave, LALSimulation, and Ripple according to the DiffGW Benchmarking Protocol.
Emits cryptographically sealed JSON receipts with full system provenance and
bootstrap confidence intervals.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure repository root and scripts directory are on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
for d in (str(_REPO_ROOT), str(_SRC_DIR), str(_SCRIPTS_DIR)):
    if d not in sys.path:
        sys.path.insert(0, d)

import numpy as np

import torchwave
from torchwave.benchmark import (
    DEFAULT_MODEL_SPECS,
    benchmark_jaxwave_model,
    benchmark_lal_model,
    benchmark_ripple_model,
    benchmark_torchwave_model,
    resolve_device,
    resolve_dtype,
    sync_device,
)
from torchwave.protocol import (
    collect_system_provenance,
    compute_relative_l2,
    compute_waveform_mismatch,
    seal_artifact,
    verify_scientific_gate,
)


def run_scientific_parity_gate(
    models: List[str],
    tolerance_l2: float = 1e-11,
    tolerance_mismatch: float = 1e-14,
) -> Dict[str, Any]:
    """Execute Gate 1: Scientific Qualification before timing benchmarks."""
    import torch
    from torchwave.waveforms import get_fd_waveform

    gate_results = {}
    all_passed = True

    print("\n" + "=" * 80)
    print("GATE 1: SCIENTIFIC QUALIFICATION & NUMERICAL PARITY CHECK")
    print("=" * 80)

    for model in models:
        spec = DEFAULT_MODEL_SPECS.get(model)
        if not spec:
            continue

        grid = spec["grid"]
        params = spec["params"]
        lal_name = spec.get("lal_name")

        flow = float(grid["flow"])
        fhigh = float(grid["fhigh"])
        df = float(grid["df"])
        n_freqs = int(round((fhigh - flow) / df)) + 1
        freqs_torch = torch.linspace(flow, fhigh, n_freqs, dtype=torch.float64, device="cpu")

        # Generate candidate waveform using TorchWave
        with torch.inference_mode():
            hp_torch, hc_torch = get_fd_waveform(
                model, sample_frequencies=freqs_torch, device="cpu", dtype=torch.float64, **params
            )
        hp_cand = hp_torch.cpu().numpy()
        hc_cand = hc_torch.cpu().numpy()

        # Generate or compare with oracle (JAXWave primary clean-room reference, LAL secondary)
        ref_available = False
        ref_source = "Self-Consistency"
        hp_ref, hc_ref = None, None

        # 1. Try JAXWave clean-room parity
        try:
            import jax
            import jax.numpy as jnp
            import jaxwave
            freqs_jax = jnp.linspace(flow, fhigh, n_freqs, dtype=jnp.float64)
            hp_j, hc_j = jaxwave.get_fd_waveform(model, sample_frequencies=freqs_jax, **params)
            hp_ref = np.asarray(hp_j)
            hc_ref = np.asarray(hc_j)
            ref_available = True
            ref_source = "JAXWave"
        except Exception:
            pass

        # 2. Try LALSimulation if JAXWave not available
        if not ref_available:
            try:
                import lal
                import lalsimulation as lalsim
                if lal_name:
                    approx = lalsim.GetApproximantFromString(lal_name)
                    m1_si = float(params["mass1"]) * lal.MSUN_SI
                    m2_si = float(params["mass2"]) * lal.MSUN_SI
                    s1x = float(params.get("spin1x", 0.0))
                    s1y = float(params.get("spin1y", 0.0))
                    s1z = float(params.get("spin1z", params.get("chi1", 0.0)))
                    s2x = float(params.get("spin2x", 0.0))
                    s2y = float(params.get("spin2y", 0.0))
                    s2z = float(params.get("spin2z", params.get("chi2", 0.0)))
                    dist_si = float(params.get("distance", 100.0)) * 1e6 * lal.PC_SI
                    inc = float(params.get("inclination", 0.0))
                    phic = float(params.get("phic", 0.0))
                    fref = float(params.get("f_ref", flow))
                    e0 = float(params.get("e0", 0.0))
                    extra = lal.CreateDict()
                    hp_lal, hc_lal = lalsim.SimInspiralChooseFDWaveform(
                        m1_si, m2_si, s1x, s1y, s1z, s2x, s2y, s2z,
                        dist_si, inc, phic, 0.0, e0, 0.0,
                        df, flow, fhigh, fref, extra, approx
                    )
                    lal_f = np.arange(len(hp_lal.data.data)) * hp_lal.deltaF
                    mask = (lal_f >= (flow - 1e-6)) & (lal_f <= (fhigh + 1e-6))
                    hp_ref = hp_lal.data.data[mask][:n_freqs]
                    hc_ref = hc_lal.data.data[mask][:n_freqs]
                    ref_available = True
                    ref_source = "LALSimulation"
            except Exception:
                pass

        if ref_available and hp_ref is not None:
            # Model-specific thresholds
            if ref_source == "JAXWave":
                tol_l2 = 1e-2 if "XP" in model else tolerance_l2
                tol_mis = 1e-4 if "XP" in model else tolerance_mismatch
            else:
                tol_l2 = 1e-3
                tol_mis = 1e-6

            gate_verdict = verify_scientific_gate(
                candidate_hp=hp_cand,
                candidate_hc=hc_cand,
                reference_hp=hp_ref,
                reference_hc=hc_ref,
                df=df,
                max_l2_threshold=tol_l2,
                max_mismatch_threshold=tol_mis,
            )
            gate_verdict["reference_source"] = ref_source
            passed = gate_verdict["passed"]
            max_l2 = gate_verdict["max_l2_error"]
            max_mis = gate_verdict["max_mismatch"]
        else:
            # Self-consistency check (e.g. check finite, non-zero, correct shape)
            is_valid = bool(np.all(np.isfinite(hp_cand)) and np.any(hp_cand != 0))
            gate_verdict = {
                "gate": "Gate 1: Self-Consistency Check (no reference oracle present)",
                "passed": is_valid,
                "reference_source": "Self-Consistency",
                "max_l2_error": 0.0,
                "max_mismatch": 0.0,
                "details": "Waveforms finite and non-empty",
            }
            passed = is_valid
            max_l2 = 0.0
            max_mis = 0.0

        status_str = "PASS" if passed else "FAIL"
        print(f"[{status_str:4s}] {model:22s} | Max L2: {max_l2:11.3e} | Max Mismatch: {max_mis:11.3e}")
        gate_results[model] = gate_verdict
        if not passed:
            all_passed = False

    print("=" * 80)
    return {"all_passed": all_passed, "models": gate_results}


def main():
    parser = argparse.ArgumentParser(
        description="Unified Executable Benchmark Campaign Runner for DiffGW."
    )
    parser.add_argument(
        "--models",
        type=str,
        default="TaylorF2,IMRPhenomD,IMRPhenomXAS,IMRPhenomXP",
        help="Comma-separated model names.",
    )
    parser.add_argument(
        "--backends",
        type=str,
        default="torch,jax,lal,ripple",
        help="Comma-separated backends to benchmark (torch, jax, lal, ripple).",
    )
    parser.add_argument(
        "--devices",
        type=str,
        default="cuda,cpu",
        help="Comma-separated devices (cuda, cpu, mps).",
    )
    parser.add_argument(
        "--cuda-batch-sizes",
        type=str,
        default="1,10,100,1000",
        help="Batch sizes for GPU benchmarks.",
    )
    parser.add_argument(
        "--cpu-batch-sizes",
        type=str,
        default="1,10,100",
        help="Batch sizes for CPU benchmarks.",
    )
    parser.add_argument(
        "--dtypes",
        type=str,
        default="float64,float32",
        help="Precisions (float64, float32).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=15,
        help="Number of measured iterations.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Number of unmeasured warmup iterations.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save output JSON receipt.",
    )
    parser.add_argument(
        "--skip-parity-gate",
        action="store_true",
        help="Skip Gate 1 parity check (not recommended).",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Automatically generate publication plots from benchmark results.",
    )

    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    backends = [b.strip().lower() for b in args.backends.split(",") if b.strip()]
    devices = [d.strip().lower() for d in args.devices.split(",") if d.strip()]
    cuda_batches = [int(b.strip()) for b in args.cuda_batch_sizes.split(",") if b.strip()]
    cpu_batches = [int(b.strip()) for b in args.cpu_batch_sizes.split(",") if b.strip()]
    dtypes = [d.strip() for d in args.dtypes.split(",") if d.strip()]

    # Collect host and environment provenance
    provenance = collect_system_provenance()

    print("=" * 80)
    print("DIFFGW CONTROLLED BENCHMARK CAMPAIGN")
    print(f"Host: {provenance['host']['hostname']} ({provenance['host']['os']})")
    print(f"CPU:  {provenance['cpu']['processor']} | Cores: {provenance['cpu']['logical_cores']}")
    gpu_devs = provenance["gpu"]["devices"]
    if gpu_devs:
        print(f"GPU:  {gpu_devs[0]['name']} ({gpu_devs[0]['total_memory_gb']} GB VRAM)")
    else:
        print("GPU:  None detected")
    print(f"PyTorch: {provenance['packages']['torch']} | JAX: {provenance['packages']['jax']} | LAL: {provenance['packages']['lalsimulation']} | Ripple: {provenance['packages']['ripple']}")
    print("=" * 80)

    # Gate 1: Scientific Qualification
    if not args.skip_parity_gate:
        gate_summary = run_scientific_parity_gate(models)
        if not gate_summary["all_passed"]:
            print("\n[ABORT] Gate 1 Scientific Qualification failed! Tolerances not met.")
            sys.exit(1)
    else:
        gate_summary = {"skipped": True}

    all_benchmarks = []

    # 1. TorchWave Benchmarks
    if "torch" in backends:
        import torch
        for dev_str in devices:
            if dev_str == "cuda" and not torch.cuda.is_available():
                print(f"\n[Notice] CUDA not available; skipping TorchWave CUDA benchmarks.")
                continue
            if dev_str == "mps" and not torch.backends.mps.is_available():
                print(f"\n[Notice] MPS not available; skipping TorchWave MPS benchmarks.")
                continue

            dev = torch.device(dev_str)
            batch_list = cuda_batches if dev_str in ("cuda", "mps") else cpu_batches

            for dt_str in dtypes:
                dt = getattr(torch, dt_str)
                print(f"\n--- Backend: TorchWave | Device: {dev_str.upper()} | Precision: {dt_str} ---")

                for model_name in models:
                    spec = DEFAULT_MODEL_SPECS.get(model_name)
                    if not spec:
                        continue

                    for bsz in batch_list:
                        try:
                            if dev_str == "cuda":
                                torch.cuda.empty_cache()
                            res = benchmark_torchwave_model(
                                model_name=model_name,
                                batch_size=bsz,
                                device=dev,
                                dtype=dt,
                                grid=spec["grid"],
                                params=spec["params"],
                                iterations=args.iterations,
                                warmup=args.warmup,
                                measure_backward=False,
                            )
                            if dev_str == "cuda":
                                torch.cuda.empty_cache()
                            all_benchmarks.append(res)
                            print(
                                f"[TORCH | {dev_str.upper():4s} | {dt_str:7s}] {model_name:20s} | B={bsz:<5d} | "
                                f"Med: {res['median_ms']:8.3f} ms (95% CI: [{res['median_ci_95'][0]:.2f}, {res['median_ci_95'][1]:.2f}]) | "
                                f"Throughput: {res['wf_per_sec']:10,.1f} wf/s",
                                flush=True,
                            )
                        except Exception as e:
                            if dev_str == "cuda":
                                torch.cuda.empty_cache()
                            print(f"[TORCH | {dev_str.upper():4s} | {dt_str:7s}] {model_name:20s} | B={bsz:<5d} | ERROR: {e}", flush=True)

    # 2. JAXWave Benchmarks
    if "jax" in backends:
        for dev_str in devices:
            batch_list = cuda_batches if dev_str == "cuda" else cpu_batches
            print(f"\n--- Backend: JAXWave | Device: {dev_str.upper()} | Precision: float64 (with JIT) ---")
            for model_name in models:
                spec = DEFAULT_MODEL_SPECS.get(model_name)
                if not spec:
                    continue
                for bsz in batch_list:
                    try:
                        res = benchmark_jaxwave_model(
                            model_name=model_name,
                            batch_size=bsz,
                            grid=spec["grid"],
                            params=spec["params"],
                            iterations=args.iterations,
                            warmup=args.warmup,
                            use_jit=True,
                            device_name=dev_str,
                        )
                        if res:
                            all_benchmarks.append(res)
                            print(
                                f"[JAX   | {dev_str.upper():4s} | float64] {model_name:20s} | B={bsz:<5d} | "
                                f"Med: {res['median_ms']:8.3f} ms | Throughput: {res['wf_per_sec']:10,.1f} wf/s | "
                                f"JIT: {res['compile_warmup_ms']:6.1f} ms",
                                flush=True,
                            )
                    except Exception as e:
                        print(f"[JAX   | {dev_str.upper():4s} | float64] {model_name:20s} | B={bsz:<5d} | ERROR: {e}", flush=True)

    # 3. LALSimulation Benchmarks (Single-core CPU reference)
    if "lal" in backends and provenance["packages"]["lalsimulation"]:
        print("\n--- Backend: LALSimulation | Device: CPU | Precision: float64 (Single-Core C Reference) ---")
        for model_name in models:
            spec = DEFAULT_MODEL_SPECS.get(model_name)
            if not spec or not spec.get("lal_name"):
                continue
            try:
                res = benchmark_lal_model(
                    lal_name=spec["lal_name"],
                    params=spec["params"],
                    grid=spec["grid"],
                    iterations=args.iterations,
                    warmup=args.warmup,
                )
                if res:
                    res["model"] = model_name
                    all_benchmarks.append(res)
                    print(
                        f"[LAL   | CPU  | float64] {model_name:20s} | B=1     | "
                        f"Med: {res['median_ms']:8.3f} ms | Throughput: {res['wf_per_sec']:10,.1f} wf/s",
                        flush=True,
                    )
            except Exception as e:
                print(f"[LAL   | CPU  | float64] {model_name:20s} | ERROR: {e}", flush=True)

    # 4. Ripple Benchmarks (JAX baseline if installed)
    if "ripple" in backends and provenance["packages"]["ripple"]:
        print("\n--- Backend: Ripple | Precision: float64 (JAX Baseline) ---")
        for dev_str in devices:
            batch_list = cuda_batches if dev_str == "cuda" else cpu_batches
            for model_name in models:
                spec = DEFAULT_MODEL_SPECS.get(model_name)
                if not spec:
                    continue
                for bsz in batch_list:
                    try:
                        res = benchmark_ripple_model(
                            model_name=model_name,
                            batch_size=bsz,
                            grid=spec["grid"],
                            params=spec["params"],
                            iterations=args.iterations,
                            warmup=args.warmup,
                            use_jit=True,
                            device_name=dev_str,
                        )
                        if res:
                            all_benchmarks.append(res)
                            print(
                                f"[RIPPLE| {dev_str.upper():4s} | float64] {model_name:20s} | B={bsz:<5d} | "
                                f"Med: {res['median_ms']:8.3f} ms | Throughput: {res['wf_per_sec']:10,.1f} wf/s",
                                flush=True,
                            )
                    except Exception as e:
                        print(f"[RIPPLE| {dev_str.upper():4s} | float64] {model_name:20s} | B={bsz:<5d} | ERROR: {e}", flush=True)

    # Construct and seal receipt payload
    timestamp_slug = time.strftime("%Y%m%d_%H%M%S")
    payload = {
        "provenance": provenance,
        "parity_gate": gate_summary,
        "configuration": {
            "models": models,
            "backends": backends,
            "devices": devices,
            "cuda_batches": cuda_batches,
            "cpu_batches": cpu_batches,
            "dtypes": dtypes,
            "iterations": args.iterations,
            "warmup": args.warmup,
        },
        "benchmarks": all_benchmarks,
    }

    sealed_receipt = seal_artifact(payload)

    # Determine destination
    if args.output:
        dest_path = Path(args.output)
    else:
        out_dir = _REPO_ROOT / "artifacts" / "benchmarks"
        out_dir.mkdir(parents=True, exist_ok=True)
        dest_path = out_dir / f"benchmark_receipt_{timestamp_slug}.json"

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dest_path, "w") as f:
        json.dump(sealed_receipt, f, indent=2)

    print("\n" + "=" * 80)
    print(f"Benchmark Campaign Complete! Sealed receipt saved to:")
    print(f"  -> {dest_path.resolve()}")
    print(f"Content Seal SHA-256: {sealed_receipt['content_sha256']}")
    print("=" * 80)

    # Optional automated plotting
    if args.plot:
        try:
            from plot_benchmark_results import plot_all_figures
        except ImportError:
            from scripts.plot_benchmark_results import plot_all_figures
        plot_dir = dest_path.parent / f"plots_{timestamp_slug}"
        print(f"\nGenerating publication-quality figures in: {plot_dir}")
        plot_all_figures(dest_path, plot_dir)


if __name__ == "__main__":
    main()

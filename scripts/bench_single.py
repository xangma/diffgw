#!/usr/bin/env python3
"""Benchmark single-waveform (N=1) latency across LAL, TorchWave CPU (1 & 64 threads), and TorchWave GPU."""

import os
import sys
import time
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from torchwave.waveforms import get_fd_waveform

try:
    import lalsimulation as lalsim
    import lal
    HAS_LAL = True
except ImportError:
    HAS_LAL = False

def main():
    models = ["TaylorF2", "IMRPhenomD", "IMRPhenomXAS", "IMRPhenomXHM", "IMRPhenomXP", "IMRPhenomXPHM"]
    f = torch.arange(20.0, 1024.0, 0.25, dtype=torch.float64)
    has_cuda = torch.cuda.is_available()
    f_cuda = f.cuda() if has_cuda else None

    hdr_model = "Model"
    hdr_lal = "LAL (ms)"
    hdr_cpu1 = "TW CPU (1th)"
    hdr_cpu64 = "TW CPU (64th)"
    hdr_gpu = "TW GPU (ms)"

    print(f"{hdr_model:<16} | {hdr_lal:<10} | {hdr_cpu1:<13} | {hdr_cpu64:<14} | {hdr_gpu:<12}", flush=True)
    print("-" * 75, flush=True)

    for m in models:
        is_precessing = m in ("IMRPhenomXP", "IMRPhenomXPHM", "IMRPhenomPv2")
        s1x = 0.01 if is_precessing else 0.0
        s1y = 0.01 if is_precessing else 0.0
        s2x = -0.01 if is_precessing else 0.0
        s2y = -0.01 if is_precessing else 0.0

        # LAL
        t_lal = float("nan")
        if HAS_LAL:
            try:
                approx = lalsim.GetApproximantFromString(m)
                for _ in range(3):
                    lalsim.SimInspiralChooseFDWaveform(
                        35.0 * lal.MSUN_SI, 30.0 * lal.MSUN_SI,
                        s1x, s1y, 0.3,
                        s2x, s2y, -0.2,
                        100e6 * lal.PC_SI,
                        0.5, 0.2, 0.0, 0.0, 0.0,
                        0.25, 20.0, 1024.0, 20.0,
                        None, approx,
                    )
                t0 = time.perf_counter()
                n_lal = 50 if m not in ("IMRPhenomXHM", "IMRPhenomXPHM") else 15
                for _ in range(n_lal):
                    lalsim.SimInspiralChooseFDWaveform(
                        35.0 * lal.MSUN_SI, 30.0 * lal.MSUN_SI,
                        s1x, s1y, 0.3,
                        s2x, s2y, -0.2,
                        100e6 * lal.PC_SI,
                        0.5, 0.2, 0.0, 0.0, 0.0,
                        0.25, 20.0, 1024.0, 20.0,
                        None, approx,
                    )
                t_lal = (time.perf_counter() - t0) / n_lal * 1e3
            except Exception:
                t_lal = float("nan")

        kw = dict(
            sample_frequencies=f, mass1=35.0, mass2=30.0,
            spin1z=0.3, spin2z=-0.2,
            distance=100.0, inclination=0.5, phic=0.2, tc=0.0,
            device="cpu", dtype=torch.float64,
        )
        if is_precessing:
            kw.update(dict(spin1x=s1x, spin1y=s1y, spin2x=s2x, spin2y=s2y))

        n_runs = 30 if m not in ("IMRPhenomXHM", "IMRPhenomXP", "IMRPhenomXPHM") else 5

        # TW CPU 1 thread
        torch.set_num_threads(1)
        for _ in range(2):
            get_fd_waveform(approximant=m, **kw)
        t0 = time.perf_counter()
        for _ in range(n_runs):
            get_fd_waveform(approximant=m, **kw)
        t_cpu_1 = (time.perf_counter() - t0) / n_runs * 1e3

        # TW CPU 64 threads
        torch.set_num_threads(64)
        for _ in range(2):
            get_fd_waveform(approximant=m, **kw)
        t0 = time.perf_counter()
        for _ in range(n_runs):
            get_fd_waveform(approximant=m, **kw)
        t_cpu_64 = (time.perf_counter() - t0) / n_runs * 1e3

        # TW GPU
        t_gpu_str = "N/A"
        if has_cuda:
            kw_cuda = dict(kw, sample_frequencies=f_cuda, device="cuda")
            for _ in range(2):
                get_fd_waveform(approximant=m, **kw_cuda)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(n_runs):
                get_fd_waveform(approximant=m, **kw_cuda)
            torch.cuda.synchronize()
            t_gpu = (time.perf_counter() - t0) / n_runs * 1e3
            t_gpu_str = f"{t_gpu:8.3f}ms"

        lal_str = f"{t_lal:8.3f}ms" if not np.isnan(t_lal) else "     N/A"
        print(f"{m:<16} | {lal_str} | {t_cpu_1:10.3f}ms | {t_cpu_64:11.3f}ms | {t_gpu_str:>12}", flush=True)

if __name__ == "__main__":
    main()

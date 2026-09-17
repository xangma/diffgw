"""Controlled Executable Benchmark Protocol & Provenance Helper for DiffGW.

Implements cryptographically sealed benchmarking receipts, deterministic
bootstrap confidence intervals, environment provenance collection, and
scientific parity gatekeeping according to the DiffGW Benchmarking Protocol.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np

SCHEMA_VERSION = 1
SCHEMA_IDENTIFIER = f"diffgw-benchmark-receipt/v{SCHEMA_VERSION}"


def canonical_sha256(value: object) -> str:
    """Compute SHA-256 hash using canonical JSON serialization."""
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def artifact_content_sha256(payload: Mapping[str, Any]) -> str:
    """Hash an artifact excluding its self-referential content_sha256 seal."""
    material = dict(payload)
    material.pop("content_sha256", None)
    return canonical_sha256(material)


def seal_artifact(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Return an artifact dictionary carrying a cryptographic SHA-256 content seal."""
    sealed = dict(payload)
    sealed.pop("content_sha256", None)
    sealed["schema_version"] = SCHEMA_VERSION
    sealed["schema_identifier"] = SCHEMA_IDENTIFIER
    sealed["content_sha256"] = artifact_content_sha256(sealed)
    return sealed


def verify_seal(payload: Mapping[str, Any]) -> bool:
    """Verify whether the artifact's content_sha256 matches its canonical content."""
    seal = payload.get("content_sha256")
    if not seal:
        return False
    return seal == artifact_content_sha256(payload)


def percentile(values: Iterable[float], fraction: float) -> float:
    """Return the linear interpolated percentile of a sequence of floats."""
    ordered = sorted(float(v) for v in values)
    if not ordered:
        raise ValueError("Cannot calculate percentile of an empty sample")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("Fraction must be between 0.0 and 1.0")
    pos = (len(ordered) - 1) * fraction
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    weight = pos - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def bootstrap_median_ci(
    values: Iterable[float],
    confidence: float = 0.95,
    resamples: int = 2000,
    seed: int = 0,
) -> Tuple[float, float]:
    """Calculate a deterministic percentile-bootstrap confidence interval for the median."""
    samples = [float(v) for v in values]
    if not samples:
        raise ValueError("Cannot bootstrap an empty sample")
    if len(samples) == 1:
        return samples[0], samples[0]

    rng = random.Random(seed)
    n = len(samples)
    medians = [
        statistics.median(samples[rng.randrange(n)] for _ in range(n))
        for _ in range(resamples)
    ]
    tail = (1.0 - confidence) / 2.0
    point = float(statistics.median(samples))
    lower = percentile(medians, tail)
    upper = percentile(medians, 1.0 - tail)
    return max(min(samples), min(point, lower)), min(max(samples), max(point, upper))


def calculate_sample_statistics(
    timings_ms: List[float], batch_size: int
) -> Dict[str, Any]:
    """Compute comprehensive summary statistics and bootstrap confidence intervals."""
    if not timings_ms:
        raise ValueError("timings_ms list cannot be empty")

    n = len(timings_ms)
    mean_ms = float(statistics.mean(timings_ms))
    median_ms = float(statistics.median(timings_ms))
    stdev_ms = float(statistics.stdev(timings_ms)) if n > 1 else 0.0
    min_ms = float(min(timings_ms))
    max_ms = float(max(timings_ms))

    p05 = percentile(timings_ms, 0.05)
    p16 = percentile(timings_ms, 0.16)
    p50 = percentile(timings_ms, 0.50)
    p84 = percentile(timings_ms, 0.84)
    p95 = percentile(timings_ms, 0.95)
    iqr = p84 - p16

    med_ci_low, med_ci_high = bootstrap_median_ci(timings_ms, confidence=0.95)

    # Throughput (waveforms per second)
    wf_per_sec = (batch_size / (median_ms / 1000.0)) if median_ms > 0 else 0.0
    wf_per_sec_mean = (batch_size / (mean_ms / 1000.0)) if mean_ms > 0 else 0.0
    wf_ci_low = (batch_size / (med_ci_high / 1000.0)) if med_ci_high > 0 else 0.0
    wf_ci_high = (batch_size / (med_ci_low / 1000.0)) if med_ci_low > 0 else 0.0

    return {
        "n_samples": n,
        "batch_size": batch_size,
        "mean_ms": round(mean_ms, 4),
        "median_ms": round(median_ms, 4),
        "std_ms": round(stdev_ms, 4),
        "min_ms": round(min_ms, 4),
        "max_ms": round(max_ms, 4),
        "p05_ms": round(p05, 4),
        "p16_ms": round(p16, 4),
        "p50_ms": round(p50, 4),
        "p84_ms": round(p84, 4),
        "p95_ms": round(p95, 4),
        "iqr_ms": round(iqr, 4),
        "median_ci_95": [round(med_ci_low, 4), round(med_ci_high, 4)],
        "wf_per_sec": round(wf_per_sec, 2),
        "wf_per_sec_mean": round(wf_per_sec_mean, 2),
        "wf_per_sec_ci_95": [round(wf_ci_low, 2), round(wf_ci_high, 2)],
        "raw_timings_ms": [round(t, 4) for t in timings_ms],
    }


def get_git_info() -> Dict[str, Any]:
    """Query git commit hash, branch, and status if available."""
    info = {"commit": "unknown", "branch": "unknown", "dirty": False}
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        info["commit"] = commit
        info["branch"] = branch
        info["dirty"] = len(status) > 0
    except Exception:
        pass
    return info


def collect_system_provenance() -> Dict[str, Any]:
    """Capture comprehensive hardware, OS, and framework provenance."""
    # GPU detection via torch
    gpu_info = {"available": False, "count": 0, "devices": []}
    torch_version = None
    try:
        import torch
        torch_version = torch.__version__
        if torch.cuda.is_available():
            gpu_info["available"] = True
            gpu_info["count"] = torch.cuda.device_count()
            for i in range(gpu_info["count"]):
                props = torch.cuda.get_device_properties(i)
                gpu_info["devices"].append({
                    "index": i,
                    "name": props.name,
                    "total_memory_gb": round(props.total_memory / (1024 ** 3), 2),
                    "multi_processor_count": getattr(props, "multi_processor_count", None),
                })
            gpu_info["cuda_runtime_version"] = torch.version.cuda
    except ImportError:
        pass

    # JAX detection
    jax_version = None
    try:
        import jax
        jax_version = jax.__version__
    except ImportError:
        pass

    # LALSimulation detection
    lal_version = None
    try:
        import lalsimulation
        lal_version = getattr(lalsimulation, "__version__", "installed")
    except ImportError:
        pass

    # Ripple detection
    ripple_version = None
    try:
        import ripplegw
        ripple_version = getattr(ripplegw, "__version__", "installed")
    except ImportError:
        try:
            import ripple
            ripple_version = getattr(ripple, "__version__", "installed")
        except ImportError:
            pass

    # CPU information
    cpu_count_logical = os.cpu_count() or 1
    cpu_info = {
        "architecture": platform.machine(),
        "processor": platform.processor() or platform.machine(),
        "logical_cores": cpu_count_logical,
    }

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "host": {
            "hostname": platform.node(),
            "os": platform.system(),
            "os_release": platform.release(),
            "os_version": platform.version(),
        },
        "cpu": cpu_info,
        "gpu": gpu_info,
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
        },
        "packages": {
            "torch": torch_version,
            "jax": jax_version,
            "lalsimulation": lal_version,
            "ripple": ripple_version,
            "numpy": np.__version__,
        },
        "git": get_git_info(),
        "environment_limits": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "unset"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", "unset"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS", "unset"),
        },
    }


def compute_waveform_mismatch(
    h1: np.ndarray, h2: np.ndarray, df: float = 1.0
) -> float:
    """Compute standard normalized match mismatch: 1 - <h1|h2> / (||h1|| ||h2||)."""
    inner = np.real(np.sum(np.conj(h1) * h2)) * df
    norm1 = np.real(np.sum(np.conj(h1) * h1)) * df
    norm2 = np.real(np.sum(np.conj(h2) * h2)) * df
    if norm1 <= 0.0 or norm2 <= 0.0:
        return 1.0
    overlap = inner / np.sqrt(norm1 * norm2)
    return float(max(0.0, 1.0 - overlap))


def compute_relative_l2(h_cand: np.ndarray, h_ref: np.ndarray) -> float:
    """Compute relative L2 norm difference: ||h_cand - h_ref||_2 / ||h_ref||_2."""
    diff = np.linalg.norm(h_cand - h_ref)
    ref = np.linalg.norm(h_ref)
    if ref == 0.0:
        return float(diff)
    return float(diff / ref)


def verify_scientific_gate(
    candidate_hp: np.ndarray,
    candidate_hc: np.ndarray,
    reference_hp: np.ndarray,
    reference_hc: np.ndarray,
    df: float = 0.25,
    max_l2_threshold: float = 1e-11,
    max_mismatch_threshold: float = 1e-14,
) -> Dict[str, Any]:
    """Enforce Gate 1: Scientific Qualification before recording benchmark timing."""
    l2_hp = compute_relative_l2(candidate_hp, reference_hp)
    l2_hc = compute_relative_l2(candidate_hc, reference_hc)
    mis_hp = compute_waveform_mismatch(candidate_hp, reference_hp, df=df)
    mis_hc = compute_waveform_mismatch(candidate_hc, reference_hc, df=df)

    max_l2 = max(l2_hp, l2_hc)
    max_mis = max(mis_hp, mis_hc)
    passed = (max_l2 <= max_l2_threshold) and (max_mis <= max_mismatch_threshold)

    return {
        "gate": "Gate 1: Scientific Qualification",
        "passed": passed,
        "max_l2_error": float(max_l2),
        "max_mismatch": float(max_mis),
        "thresholds": {
            "max_l2_threshold": max_l2_threshold,
            "max_mismatch_threshold": max_mismatch_threshold,
        },
        "details": {
            "l2_plus": float(l2_hp),
            "l2_cross": float(l2_hc),
            "mismatch_plus": float(mis_hp),
            "mismatch_cross": float(mis_hc),
        },
    }

"""Reproducible numerical convergence diagnostic; no reference implementation.

Example: PYTHONPATH=src python scripts/verify_eob_published_convergence.py
  --output-dir docs/reviews/2026-09-12-clean-room/verification/pass-6/eob-convergence
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import time

import torch

from torchwave.approximants.eob.published_waveform import reconstruct_seobnrv4_22


def _comparison_stride(dt):
    ratio = 0.5 / dt
    stride = round(ratio)
    if stride < 1 or abs(ratio - stride) > 1e-10 * max(1.0, abs(ratio)):
        raise ValueError('each dt must divide the fixed 0.5 comparison spacing')
    return stride


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--tolerance', type=float, default=1e-7,
                        help='strict adjacent-grid relative complex L2 tolerance')
    parser.add_argument('--loose-tolerance', type=float, default=1e-6,
                        help='reported relaxed tolerance for diagnosis')
    parser.add_argument('--dt-values', type=float, nargs='+', default=[.5, .25, .125],
                        help='trajectory spacings to compare in descending order')
    parser.add_argument('--num-steps', type=int, default=6000)
    parser.add_argument('--ringdown-duration', type=float, default=150.)
    parser.add_argument('--no-richardson-extrapolation', dest='richardson_extrapolation',
                        action='store_false', default=True,
                        help='disable the certified Richardson trajectory refinement')
    args = parser.parse_args()
    if not (0 < args.tolerance <= args.loose_tolerance):
        parser.error('tolerance must be positive and no larger than loose-tolerance')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    paths = sorted((root/'src/torchwave/approximants/eob').glob('*.py'))
    paths += [root/'src/torchwave/approximants/kerr_qnm_generated.py', Path(__file__).resolve()]
    hashes = lambda: {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    before = hashes()
    choices = dict(spin_truncation='tabulated', spin_v6='bohe_total_energy',
                   include_pan_v8=False, include_positive_chi_540=False,
                   final_spin_fit_order=(1, 2))
    # Explicit diagnostic choices, fixed independently of any binary result.
    inputs = dict(omega_start=[.04, .04], mass1=[30., 30.], mass2=[20., 20.],
                  chi1=[0., .2], chi2=[0., -.1])
    tensors = {key: torch.tensor(value, dtype=torch.float64) for key, value in inputs.items()}
    report = dict(host=socket.gethostname(), cwd=os.getcwd(), pid=os.getpid(),
                  inputs=inputs, choices=choices, source_sha256_before=before, runs=[])
    print(json.dumps({key: report[key] for key in ('host', 'cwd', 'pid', 'choices')}), flush=True)
    data = []
    if len(args.dt_values) < 2 or any(dt <= 0 for dt in args.dt_values):
        parser.error('at least two positive dt-values are required')
    if any(a <= b for a, b in zip(args.dt_values, args.dt_values[1:])):
        parser.error('dt-values must be strictly descending')
    try:
        strides = [_comparison_stride(dt) for dt in args.dt_values]
    except ValueError as error:
        parser.error(str(error))
    (args.output_dir/'startup.json').write_text(json.dumps(report, indent=2)+'\n')
    for dt in args.dt_values:
        started = time.monotonic()
        with torch.no_grad():
            result = reconstruct_seobnrv4_22(**tensors, **choices, dt_geom=dt,
                                             num_steps=args.num_steps,
                                             ringdown_duration=args.ringdown_duration,
                                             richardson_extrapolation=args.richardson_extrapolation)
        filename = f'dt-{dt:g}.pt'
        torch.save(dict(time=result.time, mode=result.mode, trajectory_time=result.trajectory.time,
                        state=result.trajectory.state, omega=result.trajectory.frequency,
                        valid=result.trajectory.valid, match_time=result.attachment.match_time),
                   args.output_dir/filename)
        row = dict(dt=dt, seconds=time.monotonic()-started, artifact=filename,
                   peak_time=result.trajectory.peak.time.tolist(),
                   peak_frequency=result.trajectory.peak.frequency.tolist(),
                   match_time=result.attachment.match_time.tolist(),
                   final_mass_fraction=result.final_mass_fraction.tolist(),
                   final_spin=result.final_spin.tolist(),
                   amplitude_condition=result.attachment.calibration.amplitude_condition.tolist(),
                   phase_condition=result.attachment.calibration.phase_condition.tolist(),
                   samples=result.trajectory.valid.sum(0).tolist(),
                   finite=bool(torch.isfinite(result.mode).all()))
        report['runs'].append(row)
        (args.output_dir/'progress.json').write_text(json.dumps(report, indent=2)+'\n')
        data.append(result)
        print(json.dumps(row), flush=True)
    comparisons = []
    for pair_index, (coarse, fine) in enumerate(zip(data, data[1:])):
        # Common initial time/phase; no shift, rescaling or phase alignment.
        # Exclude the deliberately truncated end, keeping 100 M of ringdown.
        end = (torch.minimum(coarse.attachment.match_time, fine.attachment.match_time)+100).min()
        count = int(end/.5)+1
        a = coarse.mode[..., :count*strides[pair_index]:strides[pair_index]]
        b = fine.mode[..., :count*strides[pair_index+1]:strides[pair_index+1]]
        comparisons.append(dict(relative_complex_l2=(torch.linalg.vector_norm(a-b, dim=-1)
                            /torch.linalg.vector_norm(b, dim=-1)).tolist(),
                            peak_time_difference=(coarse.trajectory.peak.time-fine.trajectory.peak.time).abs().tolist(),
                            peak_frequency_difference=(coarse.trajectory.peak.frequency-fine.trajectory.peak.frequency).abs().tolist()))
    report['comparisons'] = comparisons
    report['source_sha256_after'] = hashes()
    report['changed_inputs'] = [key for key, value in before.items() if report['source_sha256_after'][key] != value]
    errors = [error for comparison in comparisons
              for error in comparison['relative_complex_l2']]
    finest_errors = comparisons[-1]['relative_complex_l2']
    report['tolerances'] = dict(strict=args.tolerance, loose=args.loose_tolerance,
                                max_relative_complex_l2=max(errors),
                                finest_relative_complex_l2=max(finest_errors),
                                strict_pass=bool(finest_errors) and max(finest_errors) <= args.tolerance,
                                loose_pass=bool(finest_errors) and max(finest_errors) <= args.loose_tolerance)
    monotonic = None
    if len(comparisons) > 1:
        monotonic = all(b < a for previous, following in zip(comparisons, comparisons[1:])
                        for a, b in zip(previous['relative_complex_l2'], following['relative_complex_l2']))
    report['monotonic'] = monotonic
    report['pass'] = (not report['changed_inputs'] and all(row['finite'] for row in report['runs'])
        and report['tolerances']['strict_pass'] and (monotonic is None or monotonic))
    (args.output_dir/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(comparisons=comparisons, changed_inputs=report['changed_inputs'], passed=report['pass'])), flush=True)
    return 0 if report['pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

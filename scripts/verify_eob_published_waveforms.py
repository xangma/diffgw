"""Unaligned public-LAL TD diagnostic of saved published EOB reconstructions.

Choices and trajectories must already be frozen by the convergence script.
No phase/time/amplitude fit is performed and no result selects model choices.
This diagnostic does not replace the canonical frequency-domain parity cases.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import lal
import lalsimulation as lalsim
import numpy as np
import torch

from torchwave.constants import C_SI, MPC_SI, MSUN_SEC


def _project_22(mode, inclination):
    ci = math.cos(inclination)
    return (math.sqrt(5/(16*math.pi))*(1+ci*ci)*mode.real,
            -math.sqrt(5/(4*math.pi))*ci*mode.imag)


def _invert_22(plus, cross, inclination):
    ci = math.cos(inclination)
    return plus/(math.sqrt(5/(16*math.pi))*(1+ci*ci)) + 1j*cross/(-math.sqrt(5/(4*math.pi))*ci)


def _shift_to_peak_origin(time, peak_time):
    """Transform a time array to the supplied equation-defined peak event."""
    return np.asarray(time) - float(peak_time)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--convergence-dir', type=Path, required=True)
    parser.add_argument('--json-out', type=Path, required=True)
    args = parser.parse_args()
    result_path = args.convergence_dir/'result.json'
    if result_path.exists():
        evidence = json.loads(result_path.read_text())
        provenance = 'convergence result produced by verifier'
    else:
        # Fine artifacts were recovered after a reporting-only crash.  Keep this
        # path explicit: the original startup hash was never persisted.
        recovery_path = args.convergence_dir/'recovered-result.json'
        evidence = json.loads(recovery_path.read_text())
        evidence['inputs'] = dict(omega_start=[0.04, 0.04], mass1=[30., 30.],
                                  mass2=[20., 20.], chi1=[0., .2], chi2=[0., -.1])
        evidence['choices'] = dict(spin_truncation='tabulated', spin_v6='bohe_total_energy',
                                   include_pan_v8=False, include_positive_chi_540=False,
                                   final_spin_fit_order=[1, 2])
        provenance = 'recovered artifacts; startup source hash unavailable'
    if 'runs' in evidence:
        run = evidence['runs'][-1]
    else:
        artifact_names = evidence['artifacts']
        artifact_name = min(artifact_names, key=lambda n: float(n.removeprefix('dt-').removesuffix('.pt')))
        run = dict(dt=float(artifact_name.removeprefix('dt-').removesuffix('.pt')), artifact=artifact_name)
    path = args.convergence_dir/run['artifact']
    native = torch.load(path, weights_only=True)
    results = []
    for row in range(len(evidence['inputs']['mass1'])):
        parameters = {name: value[row] for name, value in evidence['inputs'].items()}
        m1, m2, spin1, spin2 = (parameters[name] for name in ('mass1', 'mass2', 'chi1', 'chi2'))
        m_sec = (m1+m2)*MSUN_SEC
        dt = run['dt']*m_sec
        lower = parameters['omega_start']/(math.pi*m_sec)
        distance, inclination, phase = 100., .8, 0.
        hp, hc = lalsim.SimInspiralChooseTDWaveform(
            m1*lal.MSUN_SI, m2*lal.MSUN_SI, 0., 0., spin1, 0., 0., spin2,
            distance*1e6*lal.PC_SI, inclination, phase, 0., 0., 0., dt, lower,
            0., lal.CreateDict(), lalsim.GetApproximantFromString('SEOBNRv4'))
        ref_time = float(hp.epoch)+np.arange(len(hp.data.data))*float(hp.deltaT)
        native_time = (native['time'].numpy()-float(native['match_time'][row]))*m_sec
        mode = native['mode'][row].numpy()*m_sec*C_SI/(distance*MPC_SI)*np.exp(-2j*phase)
        plus, cross = _project_22(mode, inclination)
        # Invert the fixed inclination projection to obtain the reference (2,2)
        # complex mode.  This is a diagnostic convention, not an alignment.
        ref_mode = _invert_22(np.asarray(hp.data.data), np.asarray(hc.data.data), inclination)
        native_mode_interp = np.interp(ref_time, native_time, mode.real, left=0., right=0.) + 1j*np.interp(
            ref_time, native_time, mode.imag, left=0., right=0.)
        metrics = {}
        for name, signal, reference in (('plus', plus, hp.data.data), ('cross', cross, hc.data.data)):
            polarization_interp = np.interp(ref_time, native_time, signal, left=0., right=0.)
            metrics[name] = dict(relative_l2=float(np.linalg.norm(polarization_interp-reference)/np.linalg.norm(reference)),
                                 relative_norm_error=float(abs(np.linalg.norm(polarization_interp)/np.linalg.norm(reference)-1)),
                                 finite=bool(np.isfinite(polarization_interp).all() and np.isfinite(reference).all()))
        overlap = (ref_time >= native_time[0]) & (ref_time <= native_time[-1])
        # Zero-filled finite-support tails have undefined phase/frequency.  Keep
        # the raw polarization metrics above, but restrict mode phase diagnostics
        # to the common, nonzero signal support.
        active = overlap & (np.abs(ref_mode) > np.max(np.abs(ref_mode))*1e-4) & (np.abs(native_mode_interp) > np.max(np.abs(native_mode_interp))*1e-4)
        ref_mode_overlap, native_mode_overlap = ref_mode[active], native_mode_interp[active]
        ref_amp, native_amp = np.abs(ref_mode_overlap), np.abs(native_mode_overlap)
        ref_phase, native_phase = np.unwrap(np.angle(ref_mode_overlap)), np.unwrap(np.angle(native_mode_overlap))
        overlap_time = ref_time[active]
        reference_peak = float(overlap_time[np.argmax(ref_amp)])
        phase_residual = np.angle(np.exp(1j*(native_phase-ref_phase)))
        metrics['mode'] = dict(
            amplitude_relative_l2=float(np.linalg.norm(native_amp-ref_amp)/np.linalg.norm(ref_amp)),
            amplitude_norm_ratio=float(np.linalg.norm(native_amp)/np.linalg.norm(ref_amp)),
            phase_rms_rad=float(np.sqrt(np.mean(phase_residual**2))),
            phase_max_abs_rad=float(np.max(np.abs(phase_residual))),
            phase_at_common_start_rad=[float(native_phase[0]), float(ref_phase[0])],
            phase_at_reference_peak_rad=[float(native_phase[np.argmin(np.abs(overlap_time-reference_peak))]),
                                         float(ref_phase[np.argmin(np.abs(overlap_time-reference_peak))])],
            finite=bool(np.isfinite(native_amp).all() and np.isfinite(ref_amp).all()))
        native_phase_grid = np.unwrap(np.angle(mode))
        native_freq_grid = -np.gradient(native_phase_grid, native_time)
        native_freq = np.interp(overlap_time, native_time, native_freq_grid)
        ref_freq = -np.gradient(ref_phase, overlap_time)
        metrics['instantaneous_frequency'] = dict(
            native_mean=float(np.mean(native_freq)), reference_mean=float(np.mean(ref_freq)),
            relative_l2=float(np.linalg.norm(native_freq-ref_freq)/np.linalg.norm(ref_freq)),
            units='rad/s; finite differences of unwrapped (2,2) phase')
        # Report coordinate diagnostics independently of raw epoch residuals.
        native_peak = float(native_time[np.argmax(np.abs(mode))])
        peak_native_time = native_time-native_peak
        peak_reference_time = ref_time-reference_peak
        metrics['fixed_peak_time'] = dict(native_peak_s=native_peak, reference_peak_s=reference_peak,
            native_support_s=[float(peak_native_time[0]), float(peak_native_time[-1])],
            reference_support_s=[float(peak_reference_time[0]), float(peak_reference_time[-1])])
        recovered_rows = evidence.get('rows', {}).get(Path(run['artifact']).stem, {})
        if recovered_rows.get('peak_time'):
            trajectory_peak_s = (float(recovered_rows['peak_time'][row]) -
                                 float(recovered_rows['match_time'][row])) * m_sec
            native_peak_chart = _shift_to_peak_origin(native_time, trajectory_peak_s)
            native_mode_peak_chart = float(native_peak_chart[np.argmax(np.abs(mode))])
            metrics['trajectory_peak_origin'] = dict(
                native_trajectory_peak_s=trajectory_peak_s,
                native_mode_peak_in_trajectory_chart_s=native_mode_peak_chart,
                reference_amplitude_peak_s=reference_peak,
                peak_epoch_difference_s=float(reference_peak-native_mode_peak_chart),
                note='Equation-defined trajectory peak origin; raw NQC-match origin retained above; no fitted shift.')
            # Recompute residuals in the equation-defined trajectory-peak chart;
            # this is a coordinate transform, never an optimized alignment.
            peak_native_time = _shift_to_peak_origin(native_time, trajectory_peak_s)
            peak_interp = np.interp(ref_time, peak_native_time, mode.real, left=0., right=0.) + 1j*np.interp(
                ref_time, peak_native_time, mode.imag, left=0., right=0.)
            peak_active = ((ref_time >= peak_native_time[0]) & (ref_time <= peak_native_time[-1]) &
                           (np.abs(ref_mode) > np.max(np.abs(ref_mode))*1e-4) &
                           (np.abs(peak_interp) > np.max(np.abs(peak_interp))*1e-4))
            pa, ra = np.abs(peak_interp[peak_active]), np.abs(ref_mode[peak_active])
            pt = ref_time[peak_active]
            pp, rp = np.unwrap(np.angle(peak_interp[peak_active])), np.unwrap(np.angle(ref_mode[peak_active]))
            pf = np.interp(pt, peak_native_time, -np.gradient(np.unwrap(np.angle(mode)), peak_native_time))
            rf = -np.gradient(rp, pt)
            thirds = []
            for label, sl in (('early', slice(0, len(pt)//3)), ('mid', slice(len(pt)//3, 2*len(pt)//3)),
                              ('late', slice(2*len(pt)//3, None))):
                thirds.append(dict(window=label, time_s=[float(pt[sl][0]), float(pt[sl][-1])],
                                   amplitude_relative_l2=float(np.linalg.norm(pa[sl]-ra[sl])/np.linalg.norm(ra[sl])),
                                   frequency_relative_l2=float(np.linalg.norm(pf[sl]-rf[sl])/np.linalg.norm(rf[sl]))))
            metrics['trajectory_peak_chart_residuals'] = dict(
                common_support_s=[float(pt[0]), float(pt[-1])],
                amplitude_relative_l2=float(np.linalg.norm(pa-ra)/np.linalg.norm(ra)),
                frequency_relative_l2=float(np.linalg.norm(pf-rf)/np.linalg.norm(rf)),
                phase_at_common_start_rad=[float(pp[0]), float(rp[0])],
                phase_at_reference_peak_rad=[float(pp[np.argmin(np.abs(pt-reference_peak))]), float(rp[np.argmin(np.abs(pt-reference_peak))])],
                windows=thirds)
        metrics['orbital_phase'] = dict(native_at_peak_rad=float(np.angle(mode[np.argmax(np.abs(mode))])),
            reference_at_peak_rad=float(np.angle(ref_mode_overlap[np.argmax(ref_amp)])),
            note='reported separately; no phase shift or fit applied')
        results.append(dict(parameters=parameters, distance_Mpc=distance, inclination=inclination,
            phi_ref=phase, f_ref=0., f_lower=lower, delta_t=dt,
            reference_time_interval=[float(ref_time[0]), float(ref_time[-1])],
            native_time_interval=[float(native_time[0]), float(native_time[-1])],
            reference_samples=len(ref_time), metrics=metrics))
    output = dict(lal_version=lal.__version__, lalsimulation_version=lalsim.__version__,
        api='SimInspiralChooseTDWaveform', approximant='SEOBNRv4', choices=evidence['choices'],
        native_artifact=str(path), native_artifact_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        native_source_sha256=evidence.get('source_sha256_after'),
        input_sha256=hashlib.sha256(json.dumps(evidence['inputs'], sort_keys=True).encode()).hexdigest(),
        verifier_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        provenance=provenance,
        startup_source_sha256=evidence.get('startup_source_sha256'),
        convention='Native NQC amplitude peak is t=0; reference retains its returned epoch. Same numerical input phase zero, without establishing identical phase semantics. No optimization or fitted alignment.',
        scope='Separate TD diagnostic on reference time support; not a catalog pass or canonical FD acceptance test. Raw epoch residuals include linear native interpolation and finite start/end support. Peak-time and orbital-phase coordinates are diagnostics only.',
        results=results)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()

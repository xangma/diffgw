"""Process-local clean full-J frame assembly diagnostic (no production routing)."""
import copy, hashlib, json, sys
from pathlib import Path
import lal, lalsimulation, torch
from torchwave.approximants.imrphenomp import IMRPhenomP
import torchwave.approximants.precession_msa as msa
import torchwave.approximants.imrphenom_xp as xpmod
from torchwave.approximants.precession_nnlo_clean import coefficients
from torchwave.approximants.precession_beta_clean import beta as clean_beta
from torchwave.approximants.precession_source_frame_clean import source_frame_geometry
from torchwave.waveforms import get_fd_waveform
from parity_harness import get_canonical_catalog_cases, get_canonical_holdout_cases, direct_complex_relative_l2
from diagnose_precession_nnlo_paired import public_fd

ORIG = None
CALLS = 0

def clean_coeff(eta, delta, m1_norm, chip, chi_l):
    aa, ee = coefficients(eta, -delta, m1_norm, chi_l / m1_norm, chip)
    return tuple(aa.unbind(-1)) + tuple(ee.unbind(-1))

def frame_adapter(v, v0, **p):
    global CALLS
    CALLS += 1
    version = p.pop('prec_version', 102)
    convention = p.pop('convention', 1)
    p.pop('return_carrier_phase', None)
    phi_ref = p.pop('phi_ref', torch.zeros_like(v0))
    native = ORIG(v, v0, **p, phi_ref=phi_ref, prec_version=version,
                  convention=convention, return_carrier_phase=True)
    native_ref = ORIG(v0.reshape(1), v0, **p, phi_ref=phi_ref,
                      prec_version=version, convention=convention,
                      return_carrier_phase=True)
    M = p['m1'] + p['m2']
    f = v**3 / (torch.pi * M * lal.MTSUN_SI)
    f0 = v0**3 / (torch.pi * M * lal.MTSUN_SI)
    geom = source_frame_geometry(p['m1'], p['m2'], p['s1x'], p['s1y'], p['s1z'],
                                 p['s2x'], p['s2y'], p['s2z'], p['iota'],
                                 phi_ref, f0, compat_102=True)
    b = clean_beta(p['m1'], p['m2'], f, p['s1x'], p['s1y'], p['s1z'],
                   p['s2x'], p['s2y'], p['s2z'], compat_102=True)
    alpha = native[0] - native_ref[0].reshape(()) + geom[0]
    theta = geom[1]
    zeta = geom[2]
    gamma = native[2] - native_ref[2].reshape(())
    carrier = geom[3]
    return alpha, b, gamma, theta, zeta, carrier

def run_case(case, final_spin_mod, phic):
    global ORIG, CALLS
    c = copy.deepcopy(case)
    c['params']['phic'] = phic
    c['params']['convention'] = 0
    c['params']['final_spin_mod'] = final_spin_mod
    f, refp, refc = public_fd(c, 0)
    g = c['grid']; kw = dict(c['params'])
    kw.pop('f_lower', None); kw.pop('f_final', None)
    kw.update(prec_version=102, convention=0, final_spin_mod=final_spin_mod)
    old_coeff = IMRPhenomP.__dict__['nnlo_angle_coefficients']
    old_msa, old_xp = msa.evaluate_msa_precession, xpmod.evaluate_msa_precession
    ORIG = old_msa
    IMRPhenomP.nnlo_angle_coefficients = staticmethod(clean_coeff)
    try:
        native_hp, native_hc = get_fd_waveform('IMRPhenomXP', sample_frequencies=f,
            f_lower=float(g['flow']), f_final=float(g['fhigh']), dtype=torch.float64, **kw)
        msa.evaluate_msa_precession = frame_adapter
        xpmod.evaluate_msa_precession = frame_adapter
        clean_hp, clean_hc = get_fd_waveform('IMRPhenomXP', sample_frequencies=f,
            f_lower=float(g['flow']), f_final=float(g['fhigh']), dtype=torch.float64, **kw)
    finally:
        IMRPhenomP.nnlo_angle_coefficients = old_coeff
        msa.evaluate_msa_precession, xpmod.evaluate_msa_precession = old_msa, old_xp
    if clean_hp.numel() != f.numel():
        raise RuntimeError(f"clean output shape {tuple(clean_hp.shape)} for grid {f.numel()}")
    native_hp, native_hc = native_hp.reshape(-1), native_hc.reshape(-1)
    clean_hp, clean_hc = clean_hp.reshape(-1), clean_hc.reshape(-1)
    refp, refc = refp.reshape(-1), refc.reshape(-1)
    sel = (f >= float(g['flow'])) & (f < float(g['fhigh']))
    return {'native_plus': direct_complex_relative_l2(native_hp[sel], refp[sel]),
            'native_cross': direct_complex_relative_l2(native_hc[sel], refc[sel]),
            'clean_frame_plus': direct_complex_relative_l2(clean_hp[sel], refp[sel]),
            'clean_frame_cross': direct_complex_relative_l2(clean_hc[sel], refc[sel]),
            'n': int(sel.sum())}

if __name__ == '__main__':
    cases = [next(c for c in get_canonical_catalog_cases() if c['torch_name']=='IMRPhenomXP'),
             next(c for c in get_canonical_holdout_cases() if c['torch_name']=='IMRPhenomXP')]
    out = {'provenance':'PROCESS-LOCAL ANGLE/SOURCE-FRAME SUBSTITUTION (not a full Eq.45 mode-sum): clean coefficients + full-J source frame alpha0/thetaJN/zeta + compat102 beta; conv0 alpha=alpha_NNLO-alpha_NNLO(fref)+alpha0, gamma=gamma_NNLO-gamma_NNLO(fref), carrier=phi_J; paired XP102 convention0; public comparison uses same phic and final_spin_mod', 'python':sys.version, 'versions':{'lal':lal.__version__, 'lalsimulation':lalsimulation.__version__}, 'cases':[]}
    for c in cases:
        for fs in (0, 3):
            for phic in (0.0, float(c['params']['phic'])):
                out['cases'].append({'id':c['id'], 'final_spin_mod':fs, 'phic':phic, 'metrics':run_case(c,fs,phic), 'adapter_calls':CALLS})
    for p in ['scripts/diagnose_precession_clean_frame_assembly.py','src/torchwave/approximants/precession_source_frame_clean.py','src/torchwave/approximants/precession_beta_clean.py']:
        out.setdefault('sha256',{})[p] = hashlib.sha256(Path(p).read_bytes()).hexdigest()
    path=Path('docs/reviews/2026-09-13-parity-review/precession-clean-frame-assembly.json')
    path.write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))

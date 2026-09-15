"""Process-local XAS remnant-provider experiment; never changes production routing."""
import copy, hashlib, json
from pathlib import Path
import torch
from torchwave.approximants.imrphenom_xas import XASParameterFits
from diagnose_precession_carrier_inversion import one
from compare_precession_source_frame_public import cases


def run(label, patched=False):
    original = XASParameterFits.fit_final_spin
    calls = {'n': 0}
    if patched:
        def replacement(eta, chi1, chi2):
            calls['n'] += 1
            # fit_final_spin_nnlo calls the aligned fit internally; temporarily
            # expose the original to avoid recursion while preserving the hook.
            XASParameterFits.fit_final_spin = original
            try:
                return XASParameterFits.fit_final_spin_nnlo(
                    eta, chi1, chi2, torch.ones_like(eta))
            finally:
                XASParameterFits.fit_final_spin = replacement
        XASParameterFits.fit_final_spin = replacement
    try:
        cs = []
        for c in cases():
            c.setdefault('grid', {'flow':25., 'fhigh':160., 'df':.25})
            c.setdefault('lal_name', 'IMRPhenomXP')
            c['params'].update(prec_version=102, convention=0, final_spin_mod=0)
            cs.append(c)
        rows = [one(copy.deepcopy(c)) for c in cs]
    finally:
        XASParameterFits.fit_final_spin = original
    return {'label': label, 'hook_calls': calls['n'], 'rows': rows}


def override_probe():
    c = cases()[0]['params']
    from torchwave.approximants.imrphenom_xas import IMRPhenomXAS
    f = torch.tensor([30., 80., 140.], dtype=torch.float64)
    x = IMRPhenomXAS()
    kw = dict(f=f, mass1=float(c['mass1']), mass2=float(c['mass2']),
              chi1=float(c.get('spin1z', 0.)), chi2=float(c.get('spin2z', 0.)),
              inclination=float(c.get('inclination', 0.)), f_ref=float(c.get('f_ref', 20.)))
    a = x(_final_spin_override=torch.tensor(.1, dtype=torch.float64), **kw)
    b = x(_final_spin_override=torch.tensor(.9, dtype=torch.float64), **kw)
    def rel(x, y):
        return float(torch.linalg.vector_norm(x-y) / torch.clamp(torch.linalg.vector_norm(y), min=1e-300))
    return {'values': [0.1, 0.9], 'max_abs_hp_difference': float((a[0]-b[0]).abs().max()),
            'max_abs_hc_difference': float((a[1]-b[1]).abs().max()),
            'relative_l2_hp': rel(a[0], b[0]), 'relative_l2_hc': rel(a[1], b[1]),
            'relative_l2_combined': rel(torch.cat((a[0], a[1])), torch.cat((b[0], b[1])))}


def main():
    out = {
      'provenance': 'black-box XAS remnant-provider experiment; override keyword behavior and process-local fit_final_spin hook; no fitting',
      'python': __import__('sys').version,
      'override_probe': override_probe(),
      'baseline': run('baseline', False),
      'patched': run('fit_final_spin_to_nnlo_unit_chip', True),
      'source_sha256': {},
    }
    for p in ('scripts/diagnose_precession_carrier_remnant_override.py',
              'scripts/diagnose_precession_carrier_inversion.py',
              'src/torchwave/approximants/imrphenom_xas.py'):
        out['source_sha256'][p] = hashlib.sha256(Path(p).read_bytes()).hexdigest()
    # compact effect summary, retaining full inversion rows above
    base = {r['id']: r for r in out['baseline']['rows']}
    new = {r['id']: r for r in out['patched']['rows']}
    out['effect'] = [{'id': k,
      'printed_carrier_vs_xas_baseline': base[k]['variants'][1]['carrier_vs_xas_l2'],
      'printed_carrier_vs_xas_patched': new[k]['variants'][1]['carrier_vs_xas_l2']}
      for k in base]
    dst=Path('docs/reviews/2026-09-13-parity-review/precession-carrier-remnant-override.json')
    dst.write_text(json.dumps(out, indent=2)+'\n')
    print(json.dumps(out['effect'], indent=2))
    print('hook_calls', out['patched']['hook_calls'])

if __name__ == '__main__': main()

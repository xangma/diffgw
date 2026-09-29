"""Compare public LALSuite SpinTaylor angle API output with native angles.

Uses LAL binary outputs only; no implementation source.
"""

import json
import numpy as np
import torch
import lal
import lalsimulation as ls

from torchwave.approximants.precession_spintaylor import evaluate_spintaylor_precession

M1, M2 = 36.0, 29.0
SPINS = (0.3, 0.2, -0.1, 0.1, -0.2, 0.2)
DF, F_MIN, F_MAX, F_REF = 0.25, 20.0, 300.0, 37.3


def one(convention):
    p = lal.CreateDict()
    ls.SimInspiralWaveformParamsInsertPhenomXPrecVersion(p, 320)
    ls.SimInspiralWaveformParamsInsertPhenomXPConvention(p, convention)
    out = ls.SimIMRPhenomXPSpinTaylorAngles(
        M1*lal.MSUN_SI, M2*lal.MSUN_SI, *SPINS,
        F_MIN, F_MAX, DF, F_REF, 0.0, p,
    )
    ref = [np.array(v.data) for v in out]
    f = torch.arange(F_MIN, F_MAX+DF/2, DF, dtype=torch.float64)
    native = evaluate_spintaylor_precession(
        f=f, f_ref=F_REF, mass1=M1, mass2=M2,
        spin1x=SPINS[0], spin1y=SPINS[1], spin1z=SPINS[2],
        spin2x=SPINS[3], spin2y=SPINS[4], spin2z=SPINS[5],
        inclination=0.6, phi_ref=0.0, convention=convention,
    )
    pred = [native[0].detach().numpy(),
            torch.cos(native[1]).detach().numpy(),
            native[2].detach().numpy()]
    ref_index = round((37.25-F_MIN)/DF)
    insp = np.arange(len(f))*DF+F_MIN <= 60.0
    alpha_offset = pred[0][ref_index]-ref[0][ref_index]
    gamma_offset = pred[2][ref_index]-ref[2][ref_index]
    print(json.dumps({'convention':convention,'lengths':[len(x) for x in ref],
                      'native_lengths':[len(x) for x in pred],
                      'component_order':['alpha','cos_beta','gamma'],
                      'sample_frequencies':[20,37.25,60,100,150,300],
                      'lal_samples':[[float(x[round((hz-F_MIN)/DF)]) for hz in
                                      [20,37.25,60,100,150,300]] for x in ref],
                      'native_samples':[[float(x[round((hz-F_MIN)/DF)]) for hz in
                                         [20,37.25,60,100,150,300]] for x in pred],
                      'inspiral_alpha_error_after_ref_offset':float(np.max(np.abs(
                          pred[0][insp]-alpha_offset-ref[0][insp]))),
                      'inspiral_gamma_error_after_ref_offset':float(np.max(np.abs(
                          pred[2][insp]-gamma_offset-ref[2][insp]))),
                      'inspiral_beta_error_rad':float(np.max(np.abs(
                          np.arccos(np.clip(pred[1][insp],-1,1))-
                          np.arccos(np.clip(ref[1][insp],-1,1)))))}),flush=True)
    if all(len(x)==len(f) for x in ref):
        print(json.dumps({'convention':convention,'differences':[
            {'raw_l2':float(np.linalg.norm(x-y)/max(np.linalg.norm(x),1e-15)),
             'max_abs':float(np.max(np.abs(x-y)))}
            for x,y in zip(ref,pred)]}),flush=True)


for convention in [1,0]:
    one(convention)

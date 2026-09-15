"""Public binary-only option controls; no native waveforms or fitting."""
import datetime, hashlib, json, platform, sys
from pathlib import Path
import lal
import lalsimulation as ls
import numpy as np
import parity_harness as harness

out = Path(__file__).resolve().parent
options = [(None,None),(102,0),(223,3),(300,4),(320,2)]
records=[]
for case in harness.get_canonical_catalog_cases()+harness.get_canonical_holdout_cases():
    if case['torch_name'] not in ('IMRPhenomXP','IMRPhenomXPHM'): continue
    p,g=case['params'],case['grid']
    args=[p['mass1']*lal.MSUN_SI,p['mass2']*lal.MSUN_SI,
          *[p.get('spin'+str(i)+a,p.get('chi'+str(i),0.) if a=='z' else 0.) for i in (1,2) for a in ('x','y','z')],
          p.get('distance',100.)*1e6*lal.PC_SI,p.get('inclination',0.),p.get('phic',0.),0.,0.,0.,
          g['df'],g['flow'],g['fhigh'],p.get('f_ref',g['flow'])]
    reference=None
    for version, remnant in options:
        record=dict(case_id=case['id'],prec_version=version,final_spin_mod=remnant)
        extra=lal.CreateDict()
        if version is not None:
            ls.SimInspiralWaveformParamsInsertPhenomXPrecVersion(extra,version)
            ls.SimInspiralWaveformParamsInsertPhenomXPFinalSpinMod(extra,remnant)
        try:
            result=ls.SimInspiralChooseFDWaveform(*args,extra,ls.GetApproximantFromString(case['lal_name']))
            arrays=[x.data.data.copy() for x in result]
            if reference is None: reference=arrays
            record.update(relative_l2={pol:float(np.linalg.norm(a-b)/np.linalg.norm(b)) for pol,a,b in zip(('plus','cross'),arrays,reference)},
                          bitwise_equal=all(np.array_equal(a,b) for a,b in zip(arrays,reference)))
        except Exception as e: record['error']=str(e)
        records.append(record)
report=dict(scope='public_binary_only_default_option_controls',host=platform.node(),utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            versions=dict(lal=lal.__version__,lalsimulation=ls.__version__),
            verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            note='Finite-case output equivalence does not prove internal dispatch. No native implementation is selected, changed or fitted from these results. Frozen parity inventories unchanged.',
            option_basis={'102/0':'existing explicit NNLO control','223/3':'existing explicit MSA control','300/4':'public dictionary lookup defaults','320/2':'Colleoni et al. arXiv:2412.16721v1 Sec. II.2.1'},results=records)
(out/'public-defaults.json').write_text(json.dumps(report,indent=2)+'\n')
for r in records: print(r)

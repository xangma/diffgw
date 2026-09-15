"""Black-box native HM QNM inputs versus independent Kerr eigenfrequencies."""
import hashlib, inspect, json, pathlib
import torch
from torchwave.approximants.imrphenom_hm import IMRPhenomHM
from torchwave.approximants.imrphenomd import IMRPhenomD
from torchwave.approximants.kerr_qnm_generated import dimensionless_qnm, SUPPORTED_MODES
import lal, lalsimulation as lalsim

MODES=((2,2),(2,1),(3,3),(3,2),(4,4),(4,3))
CASES=((40.,15.,.3,-.2),(55.,11.,-.4,.2))

def run():
    hm=IMRPhenomHM(); rem=IMRPhenomD()
    native_path=pathlib.Path(inspect.getfile(hm.__class__))
    report={'provenance':'Native mode_qnm_frequencies is queried by signature only. Independent Kerr values are dimensionless QNM eigenvalues from kerr_qnm_generated.dimensionless_qnm(l,m,n,a_f), converted as f=(Re omega/2pi)*(M/Mf), with damping width=(Im omega/2pi)*(M/Mf). Remnant API docstring cites Husa et al. arXiv:1508.07250 Eqs. 16-17 and reports Mf [solar masses], af, and M*f quantities. The 4,3 independent clean table is unsupported and is reported as an API gap.', 'native_qnm_signature':str(inspect.signature(hm.mode_qnm_frequencies)), 'remnant_signature':str(inspect.signature(rem.remnant_properties)), 'native_source_sha256':hashlib.sha256(native_path.read_bytes()).hexdigest(), 'cases':[]}
    for mass1,mass2,chi1,chi2 in CASES:
        args=[torch.tensor(x,dtype=torch.float64) for x in (mass1,mass2,chi1,chi2)]
        mf,af,fr22,fd22=rem.remnant_properties(*args); ratio=(mass1+mass2)/float(mf); row={'inputs':dict(mass1=mass1,mass2=mass2,chi1=chi1,chi2=chi2),'remnant_blackbox':{'final_mass_solar':float(mf),'final_spin':float(af),'f_rd_M':float(fr22),'f_damp_M':float(fd22)},'modes':{}}
        for l,m in MODES:
            fn,fd=hm.mode_qnm_frequencies(l,m,*args); item={'native':{'f_M':float(fn),'damp_M':float(fd)}}
            key=(l,m,0)
            if key in SUPPORTED_MODES:
                wr,wi=dimensionless_qnm(l,m,0,float(af)); item['independent_kerr_same_af']={'f_M':float(wr)/(2*3.141592653589793)*ratio,'damp_M':float(wi)/(2*3.141592653589793)*ratio,'omega_dimensionless':float(wr),'im_dimensionless':float(wi)}
                item['native_over_independent']={'f':item['native']['f_M']/item['independent_kerr_same_af']['f_M'],'damp':item['native']['damp_M']/item['independent_kerr_same_af']['damp_M']}
            else: item['independent_kerr_same_af']={'status':'unsupported_mode_in_clean_table','mode':key}
            row['modes'][f'{l}{m}']=item
        report['cases'].append(row)
    p=pathlib.Path(__file__); report['script_sha256']=hashlib.sha256(p.read_bytes()).hexdigest(); pathlib.Path(__file__).with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__': run()

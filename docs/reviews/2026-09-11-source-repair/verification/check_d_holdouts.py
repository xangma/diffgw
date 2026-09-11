"""Independent frozen holdouts; uses public LAL binary APIs through the unchanged harness."""
import copy,json,sys
from pathlib import Path
import parity_harness as h
cases=[copy.deepcopy(h.get_canonical_model_cases()[0])]
for i,(m1,m2,c1,c2,ref,inc) in enumerate([(57.,13.,.83,-.41,27.,.9),(23.,21.,-.6,.2,35.,1.2),(80.,8.,-.7,.1,20.,.4),(42.,42.,.92,.92,23.,.7)]):
 c=copy.deepcopy(cases[0]);c['id']=f'IMRPhenomD_holdout_{i}'
 c['params'].update(mass1=m1,mass2=m2,chi1=c1,chi2=c2,f_ref=ref,inclination=inc,phic=.71,distance=680.)
 cases.append(c)
rows=[]
for c in cases:
 r=h.run_oracle_comparison(c,strict=True);rows.append(r)
 print(c['id'],r['status'],flush=True)
Path(sys.argv[1]).write_text(json.dumps({'purpose':'Additional diagnostic holdouts; canonical catalog unchanged','cases':cases,'results':rows},indent=2)+'\n')

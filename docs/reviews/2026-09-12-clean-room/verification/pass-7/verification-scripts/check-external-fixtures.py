import hashlib, json, pathlib, torch
from torchwave.approximants.kerr_qnm_generated import dimensionless_qnm
from torchwave.approximants.kerr_qnm_interp import evaluate_kerr_qnm_220
root=pathlib.Path('/Users/xangma/repos/torchwave')
fixture=root/'tests/data/kerr_qnm_author_holdout.json'
reference=json.loads(fixture.read_text())
report={'purpose':'Verification only after independent generation; no generator tuning from these residuals.', 'runtime_sha256':hashlib.sha256((root/'src/torchwave/approximants/kerr_qnm_generated.py').read_bytes()).hexdigest(), 'fixtures':{str(fixture.relative_to(root)):hashlib.sha256(fixture.read_bytes()).hexdigest()},'cook_modes':{}}
for label,entry in reference['modes'].items():
    rows=torch.tensor(entry['rows'],dtype=torch.float64)
    real,decay=dimensionless_qnm(*map(int,label.split(',')),rows[:,0])
    expected=rows[:,1:3]
    actual=torch.stack((real,-decay),dim=-1)
    error=abs(actual-expected)
    report['cook_modes'][label]={'count':len(rows),'max_abs':error.amax(dim=0).tolist(),'max_relative':(error/abs(expected)).amax(dim=0).tolist(),'passed_5e_10_absolute':bool(torch.all(error<=5e-10))}
fixture=root/'tests/data/kerr_qnm_220_leaver.json'
report['fixtures'][str(fixture.relative_to(root))]=hashlib.sha256(fixture.read_bytes()).hexdigest()
reference=json.loads(fixture.read_text())
report['historical_signed_220']={}
for label in ('holdouts','strict'):
    rows=reference[label]
    spin=torch.tensor([r['a'] for r in rows],dtype=torch.float64)
    expected=torch.tensor([[r['wr'],r['damping']] for r in rows],dtype=torch.float64)
    actual=torch.stack(evaluate_kerr_qnm_220(spin),dim=-1)
    error=abs(actual-expected)
    report['historical_signed_220'][label]={'count':len(rows),'max_abs':error.amax(dim=0).tolist(),'max_relative':(error/abs(expected)).amax(dim=0).tolist(),'passed_1p1e_10_relative':bool(torch.all(error/abs(expected)<=1.1e-10))}
out=root/'docs/reviews/2026-09-12-clean-room/verification/pass-7/qnm-external-check.json'
out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))

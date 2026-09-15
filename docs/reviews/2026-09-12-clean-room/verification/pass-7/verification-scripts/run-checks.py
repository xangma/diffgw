import hashlib, json, os, pathlib, socket, subprocess, sys, time
ROOT=pathlib.Path('/Users/xangma/repos/torchwave')
OUT=ROOT/'docs/reviews/2026-09-12-clean-room/verification/pass-7'
OUT.mkdir(parents=True,exist_ok=True)
PYTHON='/Users/xangma/miniconda3/envs/pycbcorig313/bin/python'
ENV=dict(os.environ,PYTHONPATH='src:tests',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',LAL_DATA_PATH='/tmp/torchwave-oracle-data',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
def hashes():
    paths=sorted((ROOT/'src').rglob('*.py'))+sorted((ROOT/'tests').rglob('*.py'))+sorted((ROOT/'scripts').rglob('*.py'))+[ROOT/'docs/model_manifest.json']
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
report=dict(host=socket.gethostname(),cwd=str(ROOT),pid=os.getpid(),source_sha256_before=hashes(),runs=[])
for label,args in [
    ('pytest',['-m','pytest','-q']),
    ('catalog',['tests/parity_harness.py','--strict','--scope=catalog','--json-out',str(OUT/'catalog.json')]),
    ('holdout',['tests/parity_harness.py','--strict','--scope=holdout','--json-out',str(OUT/'holdout.json')]),
    ('kerr-components',['scripts/verify_kerr_components.py','--json-out',str(OUT/'kerr-components.json')]),
    ('kerr-waveforms',['scripts/verify_kerr_waveforms.py','--json-out',str(OUT/'kerr-waveforms.json')]),
]:
    cmd=[PYTHON]+args
    print(json.dumps(dict(label=label,command=cmd,log=str(OUT/(label+'.log')))),flush=True)
    start=time.monotonic()
    with (OUT/(label+'.log')).open('w') as log:
        result=subprocess.run(cmd,cwd=ROOT,env=ENV,stdout=log,stderr=subprocess.STDOUT)
    row=dict(label=label,command=cmd,log=label+'.log',returncode=result.returncode,seconds=time.monotonic()-start)
    report['runs'].append(row)
    print(json.dumps(row),flush=True)
report['source_sha256_after']=hashes()
report['changed_inputs']=[key for key,value in report['source_sha256_before'].items() if report['source_sha256_after'].get(key)!=value]
(OUT/'verification-run.json').write_text(json.dumps(report,indent=2)+'\n')

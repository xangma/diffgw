"""Run the fixed component probe against the preserved pre-pass source snapshot."""
import datetime, hashlib, json, os, platform, subprocess, sys, time
from pathlib import Path
root = Path(__file__).resolve().parents[5]
out = Path(__file__).resolve().parent
snapshot = Path('/tmp/torchwave-pass12-before')
starting = json.loads((out/'starting-hashes.json').read_text())
files = starting.get('files', starting.get('sha256', starting))
print('snapshot inventory keys:', list(starting)[:8], flush=True)
if isinstance(files, dict):
    checked = {k: hashlib.sha256((snapshot/k).read_bytes()).hexdigest() == v for k,v in files.items() if isinstance(v,str) and len(v)==64 and (snapshot/k).is_file()}
    assert checked and all(checked.values()), 'Snapshot changed'
record = dict(host=platform.node(), cwd=str(root), driver_pid=os.getpid(),
    command=[sys.executable,str(out/'verify-components.py'),'--json-out',str(out/'components-before.json')],
    log=str(out/'components-before.log'), started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    snapshot=str(snapshot), snapshot_hashes_validated=True, validated_files=len(checked))
started = time.monotonic()
with open(record['log'],'w') as log:
    child = subprocess.Popen(record['command'], cwd=root, env=dict(os.environ,PYTHONPATH=f'{snapshot}/src:{snapshot}/tests',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',LAL_DATA_PATH='/tmp/torchwave-oracle-data'),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    record['pid']=child.pid
    (out/'components-before-run.json').write_text(json.dumps(record,indent=2)+'\n')
    record['returncode']=child.wait()
record.update(seconds=time.monotonic()-started,ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
(out/'components-before-run.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))

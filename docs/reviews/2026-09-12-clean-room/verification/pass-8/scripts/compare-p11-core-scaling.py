from pathlib import Path
import subprocess,time,datetime,json,hashlib,os
import h5py,numpy as np
b=Path('/Users/xangma/repos/torchwave/docs/reviews/2026-09-12-clean-room/verification/pass-8/provenance')
remote='/home/xangma/simulations/torchwave-nr/2026-09-12-pilot/independent-p11/BbhReductions.h5'
base=json.loads((b/'p11-16pe-partial-timings.json').read_text())['data']['GmresResiduals.dat']
started=time.monotonic(); print(json.dumps({'pid':os.getpid(),'scope':'read-only remote polling for matching first GMRES iterations; snapshots are diagnostic only','timeout_seconds':720}),flush=True)
while time.monotonic()-started < 720:
 stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
 snapshot=b/('p11-32pe-live-timings-'+stamp+'.h5')
 copied=subprocess.run(['scp','-q','len:'+remote,str(snapshot)],capture_output=True,text=True,timeout=45)
 if copied.returncode!=0:
  print(json.dumps({'utc':stamp,'read_error':copied.stderr}),flush=True);time.sleep(45);continue
 try:
  with h5py.File(snapshot,'r') as f:rows=np.asarray(f['GmresResiduals.dat'],dtype=float).tolist()
 except (OSError,KeyError) as error:
  print(json.dumps({'utc':stamp,'snapshot_not_readable':str(error)}),flush=True);time.sleep(45);continue
 if not rows or not np.isfinite(rows).all():raise RuntimeError('Nonfinite or missing solver timing data')
 print(json.dumps({'utc':stamp,'last_initial_gmres_rows':rows[:4],'snapshot':str(snapshot)}),flush=True)
 matches=[]
 for old in base[1:]:
  hits=[row for row in rows if row[0]==old[0]]
  if not hits:continue
  new=hits[0]
  assert np.isclose(old[2],new[2],rtol=1e-8,atol=1e-12),(old,new)
  matches.append({'iteration':int(old[0]),'walltime_16':old[1],'walltime_32':new[1],'speedup':old[1]/new[1],'residual_16':old[2],'residual_32':new[2]})
 if len(matches)>=2:
  record={'observed_utc':stamp,'same_input_sha256':'bf9b2e9ca5dd8b147dd5969c24e518c33ed71d440d9def352693c0b2981c0d84','num_procs':[16,32],'comparisons':matches,'snapshot':str(snapshot),'snapshot_sha256':hashlib.sha256(snapshot.read_bytes()).hexdigest(),'scope':'Early GMRES timings only, same input/image and matching residuals. 16-PE run intentionally interrupted; 32-PE file copied while running. Neither is a completed accepted P11 solution. No full-run speedup forecast.'}
  (b/'p11-core-scaling-comparison.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2),flush=True);break
 time.sleep(45)
else:raise SystemExit('No two comparable iterations within bounded observation window')

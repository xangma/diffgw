#!/usr/bin/env python3
"""Postprocess direct fine attachment capture without interpolation or fitting."""
import hashlib, json
from pathlib import Path
import numpy as np
from torchwave.constants import C_SI, MPC_SI, MSUN_SEC

ROOT=Path(__file__).resolve().parents[1]
CAP=ROOT/'docs/reviews/2026-09-13-parity-review/verification/eob-fine-stage-replay-clean/fine-attachment-intermediates.npz'
CACHE=ROOT/'docs/reviews/2026-09-13-parity-review/verification/eob-adaptive-full20-clean/native-full20-controller-clean.npz'
PUBLIC=Path('/tmp/torchwave-public-eob-dt_0p5M.npz')
OUT=ROOT/'docs/reviews/2026-09-13-parity-review/verification/eob-fine-stage-replay-clean'
def h(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 c=np.load(CAP); n=np.load(CACHE); pub=np.load(PUBLIC)
 restart=int(c['restart_index']); ref=int(c['refinement_factor']); attached=c['attached_mode']
 fine_idx=np.arange(0,attached.size,ref); native_idx=restart+fine_idx//ref
 keep=native_idx<n['mode_100mpc'].size; native=n['mode_100mpc'][native_idx[keep]]
 scale=50*float(MSUN_SEC)*float(C_SI)/(100*float(MPC_SI))
 got=attached[fine_idx[keep]]*scale
 def metric(x,y): return float(np.linalg.norm(x-y)/np.linalg.norm(y))
 public_mode=pub['mode22'][native_idx[keep]]
 report={'provenance':'Direct returned attachment.fine; exact native-grid indexing, no interpolation/alignment/fitting.',
  'inputs':{str(x):h(x) for x in (CAP,CACHE,PUBLIC)},'capture_sha256':h(ROOT/'scripts/capture_eob_fine_attachment_clean.py'),'script_sha256':h(Path(__file__)),
  'indexing':{'restart_index':restart,'refinement_factor':ref,'fine_indices':'arange(0,len(attached),refinement_factor)','native_indices':'restart_index+fine_indices//refinement_factor','samples_compared':int(keep.sum())},
  'metrics':{'attached_vs_cached_native_relative_l2':metric(got,native),'attached_vs_public_relative_l2':metric(got,public_mode),'cached_native_vs_public_relative_l2':metric(native,public_mode)},
  'scale_geometric_to_100mpc':scale}
 (OUT/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
if __name__=='__main__': main()

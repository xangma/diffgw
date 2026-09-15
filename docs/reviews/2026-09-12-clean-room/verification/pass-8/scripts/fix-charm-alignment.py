#!/usr/bin/env python3
"""Apply the recorded local Charm group fix and relink the pilot executables."""

import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import time


ROOT = Path('/campaign')
CHARM = Path('/work/charm_7_0_0')
BUILD = CHARM / 'multicore-linux-x86_64-gcc'
TARGETS = ['SolveXcts', 'EvolveGhBinaryBlackHole',
           'EvolveGhSingleBlackHole', 'CharacteristicExtract']


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def run(command, timeout=120):
    print(json.dumps({'command': command}), flush=True)
    subprocess.run(command, check=True, timeout=timeout)


source = CHARM / 'src/ck-core/ck.C'
patch = ROOT / 'diagnostics/charm7-alignment/group-allocation.patch'
expected = json.loads((patch.parent / 'patch-provenance.json').read_text())
record = {'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'patch_provenance': expected,
          'source_before': digest(source), 'patch_sha256': digest(patch),
          'libck_before': digest(BUILD / 'lib/libck.a'),
          'binaries_before': {name: digest(Path('/work/spectre/build/bin') / name)
                              for name in TARGETS}}
assert record['source_before'] == 'c8774371a9c3c768d2d9a9908f1a9b276fdc7c963875b9716f6ed74ce19177c1'
assert record['patch_sha256'] == 'dd96461cf701fad97f66f2b91f4b42ab91059007c0ab3c9e25ecc15223ceb232'
start = time.monotonic()
run(['patch', '--dry-run', '-p1', '-d', str(CHARM), '-i', str(patch)])
run(['patch', '-p1', '-d', str(CHARM), '-i', str(patch)])
record['source_after'] = digest(source)
assert record['source_after'] == 'a7c14ff611f576ee68020c60225b68b9b212edd02197eeefa27cc2bcfd56e93b'
run(['cmake', '--build', str(BUILD), '--target', 'ck', '-j4'], 1200)
record['libck_after'] = digest(BUILD / 'lib/libck.a')
assert record['libck_after'] != record['libck_before']
run(['cmake', '--build', '/work/spectre/build', '--target', *TARGETS, '-j4'], 1800)
record['binaries_after'] = {name: digest(Path('/work/spectre/build/bin') / name)
                            for name in TARGETS}
assert all(record['binaries_before'][name] != record['binaries_after'][name]
           for name in TARGETS), 'Every executable must relink the changed Charm library'
record['spectre_commit'] = subprocess.check_output(
    ['git', '-C', '/work/spectre', 'rev-parse', 'HEAD'], text=True).strip()
assert record['spectre_commit'] == 'ce221e8b4bf8f4ca1d0acad98453aed5b454f469'
record['charm_commit'] = subprocess.check_output(
    ['git', '-C', str(CHARM), 'rev-parse', 'HEAD'], text=True).strip()
record['charm_diff'] = subprocess.check_output(
    ['git', '-C', str(CHARM), 'diff', '--', 'src/ck-core/ck.C'], text=True)
record['wall_seconds'] = time.monotonic() - start
(ROOT / 'diagnostics/build-alignment-2/charm-alignment-build-result.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record), flush=True)

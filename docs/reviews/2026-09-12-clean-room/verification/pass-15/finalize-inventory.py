"""Hash the completed dirty worktree without replacing historical inventories."""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[4]
run = json.loads((OUT / 'verification-run.json').read_text())
assert run['source_stable'] and run['runs'][0]['returncode'] == 0
post = json.loads((OUT / 'post-verification-doc-update.json').read_text())
for name, expected in run['source_sha256_after'].items():
    if name == post['file']:
        assert expected == post['sha256_before']
        expected = post['sha256_after']
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
subprocess.run(['git', 'diff', '--check'], cwd=ROOT, check=True)
inventory = OUT.parent / 'worktree-sha256.json'
previous = json.loads(inventory.read_text())
names = set()
for command in (['git', 'diff', '--name-only', '-z'],
                ['git', 'diff', '--cached', '--name-only', '-z'],
                ['git', 'ls-files', '--others', '--exclude-standard', '-z']):
    names.update(subprocess.check_output(command, cwd=ROOT).decode().split('\0'))
hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
          for name in sorted(names) if name and (ROOT / name).is_file()
          and ROOT / name != inventory}
inventory.write_text(json.dumps(dict(
    starting_revision=previous['starting_revision'],
    recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    scope='All tracked changed and untracked non-ignored files after pass 15, excluding this inventory itself. Earlier verification inputs and pass-14 inventory are preserved separately. No commit created.',
    sha256=hashes), indent=2) + '\n')
print(f'Final input hashes match; git diff --check passed; inventoried {len(hashes)} files.')

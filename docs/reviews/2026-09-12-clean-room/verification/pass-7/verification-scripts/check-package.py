import hashlib,json,pathlib,tarfile,zipfile
root=pathlib.Path('/Users/xangma/repos/torchwave')
out=root/'docs/reviews/2026-09-12-clean-room/verification/pass-7'
dist=pathlib.Path('/tmp/torchwave-independent-dist')
sha=lambda b:hashlib.sha256(b).hexdigest()
report={'artifacts':{}}
for path in sorted(dist.iterdir()):
    if path.suffix=='.whl':
        with zipfile.ZipFile(path) as archive:
            names=archive.namelist(); contents={n:archive.read(n) for n in names if not n.endswith('/')}
    elif path.name.endswith('.tar.gz'):
        with tarfile.open(path,'r:gz') as archive:
            contents={m.name:archive.extractfile(m).read() for m in archive.getmembers() if m.isfile()}
    else:
        continue
    checked={}
    for name in ('LICENSE','THIRD_PARTY_NOTICES.md'):
        selected=[n for n in contents if n.rsplit('/',1)[-1]==name]
        assert len(selected)==1,(path,name,selected)
        assert contents[selected[0]]==(root/name).read_bytes()
        checked[name]={'path':selected[0],'sha256':sha(contents[selected[0]])}
    for name in ('kerr_qnm_generated.py','kerr_qnm_author.py','kerr_qnm_interp.py'):
        selected=[n for n in contents if n.endswith('/approximants/'+name)]
        assert len(selected)==1,(path,name,selected)
        assert contents[selected[0]]==(root/'src/torchwave/approximants'/name).read_bytes()
        checked[name]={'path':selected[0],'sha256':sha(contents[selected[0]])}
    author=next(b for n,b in contents.items() if n.endswith('/approximants/kerr_qnm_author.py'))
    assert b'_SPLINES' not in author and len(author)<2000
    interp=next(b for n,b in contents.items() if n.endswith('/approximants/kerr_qnm_interp.py'))
    assert len(interp)<2000
    report['artifacts'][path.name]={'sha256':sha(path.read_bytes()),'size_bytes':path.stat().st_size,'verified_files':checked,'removed_runtime_tables_absent':True}
assert len(report['artifacts'])==2
(out/'package-check.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))

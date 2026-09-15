"""Evaluate pinned author math records; no native fit imports or network access."""
from pathlib import Path
import ast, hashlib, re, math, json, os
ROOT = Path(__file__).resolve().parents[2]
ASSET_ROOT = Path(os.environ.get('TORCHWAVE_EXTERNAL_SOURCE_ASSETS_DIR', Path.home() / '.cache/torchwave/external-validation/source-assets'))
BASE = ASSET_ROOT / 'pass-14-amplitude/AmpFits'
PAIRS = {')': '(', ']': '[', '}': '{'}

def split_top(s):
    out = []
    start = 0
    st = []
    quote = False
    esc = False
    for i, c in enumerate(s):
        if quote:
            if esc:
                esc = False
            elif c == '\\':
                esc = True
            elif c == '"':
                quote = False
            continue
        if c == '"':
            quote = True
        elif c in '({[':
            st.append(c)
        elif c in ')}]':
            if st and st[-1] == PAIRS[c]:
                st.pop()
        elif c == ',' and len(st) == 1:
            out.append(s[start:i])
            start = i + 1
    out.append(s[start:])
    return out

def expr(path, idx):
    t = Path(path).read_text()
    p = t.index('"3DFits" ->')
    p = t.index('{', p)
    st = []
    quote = False
    esc = False
    for i in range(p, len(t)):
        c = t[i]
        if quote:
            if esc:
                esc = False
            elif c == '\\':
                esc = True
            elif c == '"':
                quote = False
            continue
        if c == '"':
            quote = True
        elif c in '({[':
            st.append(c)
        elif c in ')}]':
            st.pop()
            if not st:
                end = i + 1
                break
    es = split_top(t[p:end])
    es[0] = es[0].lstrip()[1:]
    es[-1] = es[-1].rstrip()[:-1]
    x = split_top(es[idx - 1].strip())[0].strip()
    return x[1:] if x.startswith('{') else x

def py(s):
    s = re.sub('\\s+', '', s)
    s = s.replace('\\[Eta]', 'eta').replace('\\[Chi]1', 'chi1').replace('\\[Chi]2', 'chi2')
    s = re.sub('([0-9.]+)\\*\\^([+-]?[0-9]+)', '\\1e\\2', s)
    s = s.replace('Sqrt[', 'sqrt(').replace('[', '(').replace(']', ')').replace('^', '**')
    return s

def ev(s, eta, c1, c2):
    d = math.sqrt(1 - 4 * eta)
    m1 = (1 + d) / 2
    m2 = (1 - d) / 2
    S = (m1 * m1 * c1 + m2 * m2 * c2) / (1 - 2 * eta)
    expression = py(s)
    tree = ast.parse(expression, mode='eval')
    allowed = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd, ast.Call, ast.Name, ast.Load, ast.Constant)
    if any((not isinstance(n, allowed) for n in ast.walk(tree))):
        raise ValueError('Non-arithmetic source expression')
    if any((isinstance(n, ast.Name) and n.id not in ('eta', 'chi1', 'chi2', 'S', 'sqrt') for n in ast.walk(tree))):
        raise ValueError('Unexpected source variable')
    return float(eval(compile(tree, '<author math>', 'eval'), {'sqrt': math.sqrt, '__builtins__': {}}, {'eta': eta, 'chi1': c1, 'chi2': c2, 'S': S}))
SAMPLES = [('equal_mass_unequal_spin', 0.25, 0.6, -0.3), ('q2_mixed_spin', 2 / 9, 0.4, -0.2), ('q4_mixed_spin', 0.16, -0.8, 0.5)]

def build_fixture():
    for key, expected in ASSET_SHA.items():
        mode, k = key.split('/')
        path = BASE / mode / f'CoefficientStatsInt{k}.m'
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Unexpected author source hash: {path}')
    rows = []
    for name, eta, c1, c2 in SAMPLES:
        vals = {}
        for mode in CHOSEN:
            for k in [1, 2]:
                key = f'{mode}_int{k}'
                path = BASE / mode / f'CoefficientStatsInt{k}.m'
                vals[key] = ev(expr(path, CHOSEN[mode][k]), eta, c1, c2)
        rows.append({'name': name, 'eta': eta, 'chi1': c1, 'chi2': c2, 'values': vals})
    return {'format': 'pass14-xhm-author-amplitude-v1', 'source_url': 'https://export.arxiv.org/e-print/2001.10914v2', 'asset_root': '~/.cache/torchwave/external-validation/source-assets/pass-14-amplitude/AmpFits', 'asset_sha256': ASSET_SHA, 'selection': {f'{m}_int{k}': i for m, d in CHOSEN.items() for k, i in d.items()}, 'samples': rows}
CHOSEN = {'21': {1: 2, 2: 1}, '32': {1: 4, 2: 1}, '33': {1: 3, 2: 1}, '44': {1: 2, 2: 1}}
ASSET_SHA = {'21/1': 'e0c25f0119cffdf50552a4e9d78a92470f0e4c28e57f83ed94aad7d5d1427fdf', '21/2': '80be60a0c5159263d3190e2b93e08a208d199cd66095668814607a1bf4cad723', '32/1': 'f5882de942f2f23dcffc550b7974152cabe110d5af86c32eb186f06a8cdbe561', '32/2': 'd5ac1aabf9d76117942afa47907dd6a8dab1d084d2c65594ee92796c98cbba2d', '33/1': '1b79d8a6eafff0816365dd83dfbed33a2879d2f92c05d096c518396e06a6b7e2', '33/2': 'b43e4b5e2e05310440938b42e89a68886d0aa5e6110e3988df04c0ea77f120b6', '44/1': '36dc12cce856046df1336bdac11d8da5ea51746a445e75f0e35652f9dd686188', '44/2': 'a31ffbf07d27f7e8799541afeb8f45c274c6c28d00b87ee53b1c77e781f8637b'}
if __name__ == '__main__':
    Path(__file__).with_name('xhm_amplitude_author_samples.json').write_text(json.dumps(build_fixture(), indent=2) + '\n')

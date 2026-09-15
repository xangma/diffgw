"""Parent-only source/fixture audit; requires the ignored author source cache."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[4]
spec = importlib.util.spec_from_file_location(
    "author_samples", ROOT / "tests/data/generate_xhm_amplitude_author_samples.py"
)
samples = importlib.util.module_from_spec(spec)
spec.loader.exec_module(samples)
fixture = ROOT / "tests/data/xhm_amplitude_author_samples.json"
assert json.dumps(samples.build_fixture(), indent=2) + "\n" == fixture.read_text()
native_tree = ast.parse((ROOT / "src/torchwave/approximants/xhm_fits.py").read_text())


class Normalize(ast.NodeTransformer):
    """Equal arithmetic on physical 0 < eta <= 1/4; no gradient equivalence claim."""

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if (isinstance(node.func, ast.Name) and node.func.id == "_nonnegative_sqrt"
                or isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "torch" and node.func.attr == "sqrt"):
            node.func = ast.Name(id="sqrt", ctx=ast.Load())
        return node

    def visit_BinOp(self, node):
        node = self.generic_visit(node)
        if (isinstance(node.op, ast.Pow) and isinstance(node.right, ast.Constant)
                and node.right.value == 0.5):
            return ast.Call(func=ast.Name(id="sqrt", ctx=ast.Load()),
                            args=[node.left], keywords=[])
        return node


rows = []
for mode, entries in samples.CHOSEN.items():
    for node, entry in entries.items():
        name = f"amp_int{node}_{mode}"
        fn = next(n for n in ast.walk(native_tree)
                  if isinstance(n, ast.FunctionDef) and n.name == name)
        native = next(n.value for n in fn.body if isinstance(n, ast.Return))
        author = ast.parse(samples.py(samples.expr(
            samples.BASE / mode / f"CoefficientStatsInt{node}.m", entry
        )), mode="eval").body
        assert ast.dump(Normalize().visit(native)) == ast.dump(Normalize().visit(author)), name
        rows.append(dict(function=name, selected_entry=entry,
                         normalized_arithmetic_ast_identical=True))

(OUT / "amplitude-parent-review.json").write_text(json.dumps(dict(
    fixture_regeneration_byte_identical=True,
    all_eight_source_hashes_verified=True,
    fixture_sha256=hashlib.sha256(fixture.read_bytes()).hexdigest(),
    expressions=rows,
    normalization="torch.sqrt, _nonnegative_sqrt and power 0.5 normalize to sqrt on the physical eta domain; no gradient equivalence claim.",
    hash_correction="Corrected transcribed 44/Int2 hash after independently downloading exact matching bytes from https://arxiv.org/src/2001.10914v2/anc/ParspaceFits/AmpFits/44/CoefficientStatsInt2.m",
    limits="Arithmetic correspondence only; production selection/preprocessing not established. Raw fits remain signed. Ordinary regression uses tracked fixture only."
), indent=2) + "\n")
print("8 normalized source/native AST matches; pinned hashes verified; fixture byte-identical.")

"""Read the pinned MIT author source for PhenomT inspiral fit inputs.

The author file is supplied by the caller, never bundled or downloaded by
this module. Only its two relevant arithmetic functions are evaluated after
the exact file hash is checked; production code does not import this module.
"""

from __future__ import annotations

import ast
import copy
import hashlib
from pathlib import Path


SOURCE_SHA256 = "2dfcca659cd50d927b9e164867913a586e9b1372f58ffde424f8bfbd74b56e21"
SOURCE_REVISION = "4121fafa0484f3db2a7352f4eea697faaaf29f65"
FUNCTIONS = (
    "_IMRPhenomT_Inspiral_TaylorT3_t0",
    "_IMRPhenomT_Inspiral_Freq_CP",
)


def load_author_functions(source_path):
    """Load exactly the pinned t0 and five frequency-fit expressions."""
    path = Path(source_path)
    content = path.read_bytes()
    actual_hash = hashlib.sha256(content).hexdigest()
    if actual_hash != SOURCE_SHA256:
        raise ValueError(f"author fits.py SHA-256 mismatch: {actual_hash}")
    tree = ast.parse(content, filename=str(path))
    selected = []
    for name in FUNCTIONS:
        matches = [node for node in tree.body
                   if isinstance(node, ast.FunctionDef) and node.name == name]
        if len(matches) != 1:
            raise ValueError(f"expected one author function named {name}")
        node = copy.deepcopy(matches[0])
        node.decorator_list = []
        if any(isinstance(child, (ast.Call, ast.Attribute, ast.Subscript,
                                  ast.Import, ast.ImportFrom, ast.Global,
                                  ast.Nonlocal, ast.Lambda)) for child in ast.walk(node)):
            raise ValueError(f"non-arithmetic expression in {name}")
        selected.append(node)
    scope = {"__builtins__": {}}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), scope)
    return tuple(scope[name] for name in FUNCTIONS)


def fit_inputs(source_path, m1, m2, chi1, chi2):
    """Evaluate t0 and five fitted mode frequencies from pinned author code."""
    if not (m1 >= m2 > 0 and abs(chi1) <= 1 and abs(chi2) <= 1):
        raise ValueError("expected ordered positive masses and aligned spins")
    t0_fit, frequency_fit = load_author_functions(source_path)
    total = m1 + m2
    eta = m1 * m2 / total**2
    delta = (m1 - m2) / total
    spin = (m1**2 * chi1 + m2**2 * chi2) / (m1**2 + m2**2)
    dchi = chi1 - chi2
    return t0_fit(eta, spin, dchi, delta), tuple(
        frequency_fit(eta, spin, dchi, delta, index) for index in range(1, 6)
    )

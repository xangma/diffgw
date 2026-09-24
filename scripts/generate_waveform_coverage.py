"""Render an honest full-surface inventory from the two public dispatch maps.

Only `versioned slice` rows belong to the executable mathematical-reference
catalog. This inventory records availability and work remaining, not accuracy.
"""

from __future__ import annotations

import argparse
import ast
from collections import defaultdict
from difflib import unified_diff
import json
from pathlib import Path

from diffgw_core.waveform_definitions import WAVEFORMS


ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / "docs/waveforms/coverage.json"
DOCUMENT = ROOT / "docs/waveforms/coverage.md"
START = "<!-- BEGIN GENERATED COVERAGE TABLE -->"
END = "<!-- END GENERATED COVERAGE TABLE -->"


def dispatch(backend: str) -> dict[str, str]:
    source = ROOT / "src" / backend / "waveforms.py"
    module = ast.parse(source.read_text())
    definitions = [node.value for node in module.body
                   if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "_APPROXIMANTS"
                           for target in node.targets)]
    if len(definitions) != 1 or not isinstance(definitions[0], ast.Dict):
        raise ValueError(f"cannot read {backend} dispatch registry")
    result = {}
    for key, value in zip(definitions[0].keys, definitions[0].values):
        if not isinstance(key, ast.Constant) or not isinstance(key.value, str) \
                or not isinstance(value, ast.Name):
            raise ValueError(f"unexpected {backend} dispatch entry")
        if key.value in result:
            raise ValueError(f"duplicate {backend} dispatch key {key.value}")
        result[key.value] = value.id
    return result


def _model_id(alias: str, class_name: str) -> str:
    if alias == "imrphenompv2":
        return "IMRPhenomPv2"
    if "spintaylor" in alias:
        return "IMRPhenomXPHM_SpinTaylor"
    if any(token in alias for token in ("xpnr", "xnpr")) or alias == "pnr":
        return "IMRPhenomXPNR"
    return class_name


def _expected_class(model_id: str, backend: str) -> str:
    if model_id == "IMRPhenomPv2" and backend == "torchwave":
        return "IMRPhenomP"
    if model_id in {"IMRPhenomXPHM_SpinTaylor", "IMRPhenomXPNR"}:
        return "IMRPhenomXPHM"
    return model_id


def _load_records() -> list[dict]:
    records = json.loads(RECORDS.read_text())
    ids = [record["id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate coverage record")
    for record in records:
        if not (ROOT / "docs" / record["guide"]).is_file():
            raise ValueError(f"missing guide for {record['id']}")
    versioned = {record["id"] for record in records if record["stage"] == "versioned slice"}
    cataloged = {definition.id for definition in WAVEFORMS}
    if versioned != cataloged:
        raise ValueError(f"versioned inventory differs from catalog: {versioned ^ cataloged}")
    return records


def _exports(backend: str) -> set[str]:
    path = ROOT / "src" / backend / "approximants/__init__.py"
    module = ast.parse(path.read_text())
    values = [node.value for node in module.body if isinstance(node, ast.Assign)
              and any(isinstance(target, ast.Name) and target.id == "__all__"
                      for target in node.targets)]
    if len(values) != 1 or not isinstance(values[0], (ast.List, ast.Tuple)):
        raise ValueError(f"cannot read {backend} approximant exports")
    return {node.value for node in values[0].elts if isinstance(node, ast.Constant)}


def render() -> str:
    records = _load_records()
    by_id = {record["id"]: record for record in records}
    aliases = defaultdict(lambda: defaultdict(list))
    discovered_classes = set()
    for backend in ("torchwave", "jaxwave"):
        for alias, class_name in dispatch(backend).items():
            model_id = _model_id(alias, class_name)
            if model_id not in by_id or class_name != _expected_class(model_id, backend):
                raise ValueError(f"unmapped {backend} dispatch: {alias} -> {class_name}")
            aliases[model_id][backend].append(alias)
            discovered_classes.add(class_name)
    expected_classes = {name for record in records for name in record["classes"]}
    if discovered_classes != expected_classes:
        raise ValueError(f"class coverage mismatch: missing={expected_classes - discovered_classes}, "
                         f"unexpected={discovered_classes - expected_classes}")
    for record in records:
        if record["kind"] == "direct waveform class":
            for backend in ("torchwave", "jaxwave"):
                if record["id"] not in _exports(backend):
                    raise ValueError(f"{record['id']} is not exported by {backend}.approximants")
    lines = ["| Model or variant | Form | Spec status | Torch dispatch | JAX dispatch |",
             "| --- | --- | --- | --- | --- |"]
    for record in records:
        model_id = record["id"]
        page = f"../{record['guide']}"
        stage = record["stage"]
        if stage == "versioned slice":
            stage = "[versioned slice](index.md)"
        def names(backend: str) -> str:
            found = aliases[model_id][backend]
            return ", ".join(f"`{name}`" for name in found) if found else "direct class only"
        lines.append(f"| [{model_id}]({page}) | {record['kind']} | {stage} | "
                     f"{names('torchwave')} | {names('jaxwave')} |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    current = DOCUMENT.read_text()
    if current.count(START) != 1 or current.count(END) != 1:
        raise ValueError("coverage table markers missing or duplicated")
    before, tail = current.split(START, 1)
    _, after = tail.split(END, 1)
    expected = f"{before}{START}\n\n{render()}\n\n{END}{after}"
    if expected == current:
        return 0
    if args.check:
        print("".join(unified_diff(current.splitlines(keepends=True),
                                   expected.splitlines(keepends=True),
                                   fromfile=str(DOCUMENT), tofile="generated")))
        return 1
    DOCUMENT.write_text(expected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

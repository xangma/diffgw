"""Keep waveform catalog views and canonical spec headers in sync."""

from __future__ import annotations

import argparse
from difflib import unified_diff
from pathlib import Path

from diffgw.waveforms import list_waveforms


ROOT = Path(__file__).resolve().parents[1]
DOCUMENT = ROOT / "docs" / "waveforms" / "index.md"
START = "<!-- BEGIN GENERATED WAVEFORM TABLE -->"
END = "<!-- END GENERATED WAVEFORM TABLE -->"
SPEC_START = "<!-- BEGIN GENERATED WAVEFORM DEFINITION -->"
SPEC_END = "<!-- END GENERATED WAVEFORM DEFINITION -->"
SPEC_SECTIONS = (
    "Scope and status", "Inputs, conventions, and options",
    "Mathematics and components", "Implementation mapping",
    "Evidence and limitations", "Reproduce the checks",
)


def _escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _format_choices(choices: tuple[str, ...]) -> str:
    return ", ".join(f"`{choice}`" for choice in choices) or "None"


def _between_markers(current: str, start: str, end: str, replacement: str) -> str:
    if current.count(start) != 1 or current.count(end) != 1:
        raise ValueError(f"expected one pair of markers: {start}, {end}")
    before, tail = current.split(start, 1)
    _, after = tail.split(end, 1)
    return f"{before}{start}\n\n{replacement}\n\n{end}{after}"


def _spec_header(model) -> str:
    backends = ", ".join(f"`{name}`" for name in ("torch", "jax")
                         if name in model.capabilities)
    modes = ", ".join(f"`({ell}, {m})`" for ell, m in model.modes)
    lines = [
        "| Definition field | Catalog value |",
        "| --- | --- |",
        f"| Model ID | `{model.id}` |",
        f"| Reference composition | `{model.reference_version}` |",
        f"| Scope | {_escape(model.scope)} |",
        f"| Domain and modes | `{model.domain}`; {modes} |",
        f"| Native backends | {backends or 'None'} |",
        f"| Scoped verification claims | {len(model.claims)} |",
        "",
        "| Option | Default | Native choices | Reference-only choices |",
        "| --- | --- | --- | --- |",
    ]
    for option in model.options:
        native = option.choices if option.native_choices is None else option.native_choices
        reference_only = tuple(choice for choice in option.choices if choice not in native)
        lines.append(f"| `{option.name}` | `{option.default}` | {_format_choices(native)} | "
                     f"{_format_choices(reference_only)} |")
    return "\n".join(lines)


def _validate_spec_outline(path: Path, current: str) -> None:
    headings = [line.removeprefix("## ") for line in current.splitlines()
                if line.startswith("## ")]
    if headings != list(SPEC_SECTIONS):
        raise ValueError(f"{path.relative_to(ROOT)}: expected level-two sections "
                         f"{SPEC_SECTIONS!r}; found {headings!r}")


def render_table() -> str:
    lines = [
        "| Model and specification | Definition version | Scope | Backends |",
        "| --- | --- | --- | --- |",
    ]
    for model in list_waveforms():
        page = model.spec_page
        if not (ROOT / "docs" / page).is_file():
            raise FileNotFoundError(f"missing canonical model page: {page}")
        scope = model.scope.replace("|", "\\|").replace("\n", " ")
        backends = ", ".join(f"`{name}`" for name in ("torch", "jax")
                             if name in model.capabilities)
        lines.append(
            f"| [{model.id}](../{page}) | `{model.reference_version}` | "
            f"{scope} | {backends} |"
        )
    return "\n".join(lines)


def updated_document(current: str) -> str:
    return _between_markers(current, START, END, render_table())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if the table is stale")
    args = parser.parse_args()
    documents = {DOCUMENT: updated_document(DOCUMENT.read_text())}
    for model in list_waveforms():
        path = ROOT / "docs" / model.spec_page
        current = path.read_text()
        documents[path] = _between_markers(current, SPEC_START, SPEC_END,
                                           _spec_header(model))
    for path in sorted((ROOT / "docs" / "math").glob("*.md")):
        if path.name in {"index.md", "units.md"} \
                or path.name.endswith("-coefficients.md"):
            continue
        _validate_spec_outline(path, path.read_text())
    stale = False
    for path, expected in documents.items():
        current = path.read_text()
        if current == expected:
            continue
        stale = True
        if args.check:
            print("".join(unified_diff(current.splitlines(keepends=True),
                                       expected.splitlines(keepends=True),
                                       fromfile=str(path), tofile="generated")))
        else:
            path.write_text(expected)
    return int(stale and args.check)


if __name__ == "__main__":
    raise SystemExit(main())

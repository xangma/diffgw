"""Exercise the required CI mode, including ways a suite can pass without running.

These subprocess checks use the real conftest, rather than duplicating its gate
logic. Optional local installations can skip this module; required CI cannot.
"""

from pathlib import Path

import pytest

for dependency in ("sympy", "mpmath", "torch", "jax"):
    pytest.importorskip(dependency)

pytest_plugins = ["pytester"]
CONFTEST = Path(__file__).with_name("conftest.py").read_text()


def prepare(pytester, body, extra_conftest=""):
    pytester.makeconftest(CONFTEST + "\n" + extra_conftest)
    directory = pytester.path / "tests"
    directory.mkdir()
    (directory / "test_probe_math_reference.py").write_text(body)


@pytest.mark.parametrize("body,args,summary", [
    ("def test_probe(): pass", (), "1/1 passed; 0 skipped/xfail; 0 deselected"),
    ("import pytest\ndef test_probe(): pytest.skip('missing evidence')", (),
     "0/1 passed; 1 skipped/xfail; 0 deselected"),
    ("import pytest\n@pytest.mark.xfail\ndef test_probe(): assert False", (),
     "0/1 passed; 1 skipped/xfail; 0 deselected"),
    ("import pytest\n@pytest.mark.xfail\ndef test_probe(): pass", (),
     "0/1 passed; 1 skipped/xfail; 0 deselected"),
    ("def test_probe(): pass", ("--collect-only",),
     "0/1 passed; 0 skipped/xfail; 0 deselected"),
    ("def test_probe(): pass\ndef test_other(): pass", ("-k", "not test_other"),
     "1/1 passed; 0 skipped/xfail; 1 deselected"),
])
def test_gate_requires_executed_checks(pytester, body, args, summary):
    prepare(pytester, body)
    result = pytester.runpytest_subprocess("--require-math-reference", *args, "-q")
    assert summary in result.stdout.str()
    assert (result.ret == 0) == (summary == "1/1 passed; 0 skipped/xfail; 0 deselected")


def test_gate_rejects_missing_reference_module(pytester):
    prepare(pytester, "def test_probe(): pass")
    (pytester.path / "tests" / "test_other_math_reference.py").write_text("def test_other(): pass")
    result = pytester.runpytest_subprocess(
        "tests/test_probe_math_reference.py", "--require-math-reference", "-q"
    )
    assert result.ret != 0
    assert "Required math reference modules were not collected" in result.stderr.str()


def test_gate_rejects_missing_dependency(pytester):
    prepare(pytester, "def test_probe(): pass", """
original_import = importlib.import_module
def missing_sympy(name, *args, **kwargs):
    if name == 'sympy':
        raise ImportError('simulated absent dependency')
    return original_import(name, *args, **kwargs)
importlib.import_module = missing_sympy
""")
    result = pytester.runpytest_subprocess("--require-math-reference", "-q")
    assert result.ret != 0
    assert "Required math reference dependency 'sympy' could not load" in result.stderr.str()

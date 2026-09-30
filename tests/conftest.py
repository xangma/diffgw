import os
import importlib
from importlib.metadata import version
from pathlib import Path

import pytest

# Prevent macOS crashes when conda-forge OpenBLAS and PyTorch bundle separate libomp runtimes
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def pytest_addoption(parser):
    group = parser.getgroup("mathematical reference")
    group.addoption("--require-math-reference", action="store_true",
                    help="Require all math reference tests, both backends, and no skips/xfails.")
    group.addoption("--reference-device", choices=("cpu", "cuda", "mps"), default="cpu",
                    help="Device explicitly qualified by the reference suite (default: cpu).")


@pytest.fixture
def reference_device(request):
    return request.config.getoption("--reference-device")


def pytest_configure(config):
    if config.getoption("--require-math-reference"):
        config.pluginmanager.register(MathReferenceGate(config), "math-reference-gate")


class MathReferenceGate:
    """Qualification must exercise the reference, not succeed through optional skips.

    Discover maintained reference modules by name so adding another waveform's
    reference automatically adds it to the required collection. Ordinary local
    pytest runs retain their existing optional-dependency behavior.
    """

    def __init__(self, config):
        self.config = config
        tests_path = Path(config.rootpath) / "tests"
        required_paths = (
            *tests_path.glob("test_*math_reference*.py"),
            *tests_path.glob("test_nsbh_*reference.py"),
            *tests_path.glob("test_emri_*reference.py"),
            *tests_path.glob("test_xp_msa_remnant_boundary_policy.py"),
        )
        self.files = {path.relative_to(config.rootpath).as_posix() for path in required_paths}
        self.selected = set()
        self.passed = set()
        self.disallowed = set()
        self.deselected = set()
        self.versions = []

    def is_reference(self, nodeid):
        return nodeid.split("::", 1)[0] in self.files

    def pytest_sessionstart(self, session):
        for name in ("sympy", "mpmath", "scipy", "torch", "jax"):
            try:
                importlib.import_module(name)
                self.versions.append(f"{name}={version(name)}")
            except Exception as exc:
                raise pytest.UsageError(
                    f"Required math reference dependency {name!r} could not load: {exc}. "
                    "Install '.[reference,torch,jax,dev]'."
                ) from exc

    def pytest_report_header(self):
        return ("Required math reference: " + ", ".join(self.versions)
                + f"; device={self.config.getoption('--reference-device')}")

    def pytest_deselected(self, items):
        self.deselected.update(item.nodeid for item in items if self.is_reference(item.nodeid))

    def pytest_collection_finish(self, session):
        self.selected = {item.nodeid for item in session.items if self.is_reference(item.nodeid)}
        collected_files = {nodeid.split("::", 1)[0] for nodeid in self.selected}
        missing = self.files - collected_files
        if not self.files or missing:
            raise pytest.UsageError(
                "Required math reference modules were not collected: "
                + (", ".join(sorted(missing)) if missing else "no reference modules found")
            )

    def pytest_runtest_logreport(self, report):
        if not self.is_reference(report.nodeid):
            return
        if report.skipped or hasattr(report, "wasxfail"):
            self.disallowed.add(report.nodeid)
        elif report.when == "call" and report.passed:
            self.passed.add(report.nodeid)

    def pytest_sessionfinish(self, session, exitstatus):
        incomplete = self.selected - self.passed
        if (self.disallowed or self.deselected or incomplete or not self.passed) and not exitstatus:
            session.exitstatus = pytest.ExitCode.TESTS_FAILED

    def pytest_terminal_summary(self, terminalreporter):
        terminalreporter.write_sep(
            "=", f"math reference: {len(self.passed)}/{len(self.selected)} passed; "
            f"{len(self.disallowed)} skipped/xfail; {len(self.deselected)} deselected"
        )


@pytest.fixture(scope="module")
def official_rom_path():
    """Use the supported explicit/environment/cache lookup for external tests."""
    pytest.importorskip("h5py")
    from torchwave.approximants.rom import resolve_official_rom_path

    try:
        return resolve_official_rom_path(os.environ.get("TORCHWAVE_EXTERNAL_ROM"))
    except FileNotFoundError:
        pytest.skip("official SEOBNRv4 ROM HDF5 is unavailable")

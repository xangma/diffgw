"""Inspect selected package boundaries in a wheel and rebuilt sdist."""

import ast
import configparser
from pathlib import Path
import subprocess
import sys
import tarfile
import zipfile


ROOT = Path(__file__).parents[1]
PROTOTYPE_SYMBOLS = {
    "SEOBNR_ODE", "SEOBNRv4Published", "SEOBNRv4PublishedNonspinning",
    "reconstruct_seobnrv4_22", "reconstruct_seobnrv4_22_adaptive",
    "conditioned_seobnrv4_fd", "generate_eob_training_grid",
    "train_and_export_surrogate", "TEOBResumS", "SpinPrecessionODE",
    "conditioned_teob_fd",
    "EOBHamiltonian", "EOBRadiationReaction",
}


def _inspect_member(name, content):
    parts = Path(name).parts
    assert "dev" not in parts, name
    assert "seobnr_surrogate" not in name, name
    # Tests and documentation can record retired model names. Shipped Python
    # implementations must not contain the actual prototype entry points.
    if name.endswith(".py") and {"torchwave", "jaxwave"}.intersection(parts):
        definitions = {
            node.name for node in ast.walk(ast.parse(content, filename=name))
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        }
        assert not definitions.intersection(PROTOTYPE_SYMBOLS), name


def test_release_archives_exclude_retired_prototypes_and_unchecked_pycbc_entries(tmp_path):
    # build is a required dev dependency: this gate must fail rather than skip
    # when the release validation environment is incomplete.
    output = tmp_path / "dist"
    result = subprocess.run(
        [sys.executable, "-m", "build", "--no-isolation", "--outdir", str(output), str(ROOT)],
        text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    wheels = list(output.glob("*.whl"))
    sdists = list(output.glob("*.tar.gz"))
    assert len(wheels) == len(sdists) == 1
    with zipfile.ZipFile(wheels[0]) as wheel:
        for name in wheel.namelist():
            _inspect_member(name, wheel.read(name))
        entry_point_files = [
            name for name in wheel.namelist()
            if name.endswith(".dist-info/entry_points.txt")
        ]
        assert len(entry_point_files) == 1
        entry_points = configparser.ConfigParser()
        entry_points.read_string(wheel.read(entry_point_files[0]).decode())
        assert set(entry_points["pycbc.waveform.fd"]) == {
            "torch_taylorf2", "torch_imrphenomd", "torch_imrphenomxas",
            "torch_imrphenomxp",
        }
        assert "torchwave/approximants/rom/official_wrapper.py" in wheel.namelist()
        assert "jaxwave/approximants/rom/official_wrapper.py" in wheel.namelist()
    with tarfile.open(sdists[0]) as sdist:
        for member in sdist.getmembers():
            if member.isfile():
                _inspect_member(member.name, sdist.extractfile(member).read())

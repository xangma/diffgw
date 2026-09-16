"""Public umbrella dispatch and independently optional backend contracts."""
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


def _isolated(code, blocked):
    root = Path(__file__).resolve().parents[1]
    setup = f'''
import importlib.abc
import sys
class BlockBackend(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {blocked!r}:
            raise ModuleNotFoundError(fullname, name=fullname)
sys.meta_path.insert(0, BlockBackend())
'''
    env = dict(os.environ, PYTHONPATH=str(root / "src"))
    subprocess.run([sys.executable, "-c", setup + code], env=env, check=True,
                   capture_output=True, text=True)


def test_core_import_needs_neither_backend():
    _isolated('''
import diffgw
assert diffgw.core.MSUN_SEC > 0
assert 'torch' not in sys.modules
assert 'jax' not in sys.modules
try:
    diffgw.get_fd_waveform(backend='torch')
except ModuleNotFoundError as e:
    assert 'diffgw[torch]' in str(e)
else:
    raise AssertionError('missing backend must be reported')
''', ["torch", "torchwave", "jax", "jaxwave"])


@pytest.mark.parametrize("backend,blocked", [
    ("torch", ["jax", "jaxwave"]),
    ("jax", ["torch", "torchwave"]),
])
def test_backend_can_generate_without_other_framework(backend, blocked):
    pytest.importorskip(backend)
    _isolated(f'''
import diffgw
hp, hc = diffgw.get_fd_waveform(
    'IMRPhenomD', mass1=30., mass2=25., f_lower=20., f_final=64.,
    delta_f=1., backend={backend!r})
assert hp.shape == hc.shape == (45,)
assert {blocked[0]!r} not in sys.modules
''', blocked)


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_umbrella_selects_native_array_and_generator(backend):
    framework = pytest.importorskip(backend)
    import diffgw
    frontend = getattr(diffgw, backend)
    params = dict(approximant="IMRPhenomD", mass1=30., mass2=25.,
                  f_lower=20., f_final=64., delta_f=1.)
    expected = frontend.get_fd_waveform(**params)
    actual = diffgw.get_fd_waveform(backend=backend, **params)
    array_type = framework.Tensor if backend == "torch" else framework.Array
    assert isinstance(actual[0], array_type)
    np.testing.assert_array_equal(actual[0], expected[0])
    generator = diffgw.WaveformGenerator(
        "IMRPhenomD", backend=backend, f_lower=20., f_final=64., delta_f=1.)
    assert isinstance(generator, frontend.WaveformGenerator)
    np.testing.assert_array_equal(generator(mass1=30., mass2=25.)[0], expected[0])


def test_unknown_backend_is_rejected():
    import diffgw
    with pytest.raises(ValueError, match="backend"):
        diffgw.get_fd_waveform(backend="numpy")

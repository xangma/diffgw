# Copyright (C) 2026
# This program is free software under the GNU General Public License,
# version 3 or (at your option) any later version.
"""TaylorF2 source identity, relocated from PyCBC's TorchWave adapter."""

import hashlib
from pathlib import Path


def taylorf2_source_identity():
    """Return the existing TaylorF2 source digest.

    This preserves the file list and byte ordering used by PyCBC's original
    provider identity. It identifies these sources, not model qualification,
    other approximants, runtime dependencies or the entire installation.
    Callers may cache this result for the lifetime of their loaded provider.
    """
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for relative in ('__init__.py', 'waveforms.py', 'constants.py',
                     'conversions.py', 'approximants/taylorf2.py'):
        digest.update(relative.encode())
        digest.update((root / relative).read_bytes())
    return digest.hexdigest()

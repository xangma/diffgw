"""Pure option routing for the IMRPhenomXPHM precession layer.

The numerical backends share this module so that a requested ``PrecVersion``
selects exactly one angle family and its published default remnant option.
"""

from numbers import Integral
from typing import Optional, Tuple


NNLO_VERSIONS = frozenset((101, 102, 103, 104))
MSA_VERSIONS = frozenset((220, 221, 222, 223, 224))
SPINTAYLOR_VERSIONS = frozenset((310, 311, 320, 321))
PNR_VERSIONS = frozenset((300, 330))
SUPPORTED_VERSIONS = NNLO_VERSIONS | MSA_VERSIONS | SPINTAYLOR_VERSIONS | PNR_VERSIONS

_METHOD_FAMILY = {
    "nnlo": "nnlo",
    "msa": "msa",
    "spintaylor": "spintaylor",
    "st": "spintaylor",
    "spintaylort4": "spintaylor",
    "pnr": "pnr",
    "xpnr": "pnr",
    "imrphenomxpnr": "pnr",
}
_DEFAULT_VERSION = {"nnlo": 102, "msa": 223, "spintaylor": 320, "pnr": 330}
_DEFAULT_FINAL_SPIN = {"nnlo": 0, "msa": 3, "spintaylor": 2, "pnr": 2}


def precession_family(version: int) -> str:
    """Return the unique angle family selected by a supported version."""
    if version in NNLO_VERSIONS:
        return "nnlo"
    if version in MSA_VERSIONS:
        return "msa"
    if version in SPINTAYLOR_VERSIONS:
        return "spintaylor"
    if version in PNR_VERSIONS:
        return "pnr"
    raise ValueError(f"Unsupported precession version: {version}.")


def resolve_xphm_options(
    method: str,
    version: Optional[int],
    final_spin_mod: Optional[int],
) -> Tuple[str, int, int]:
    """Resolve XPHM angle/remnant options without evaluating a waveform.

    ``version`` has precedence over the convenience ``method`` label.  The
    MSA and NNLO remnant defaults are Table 5 of Pratten et al. (2021); the
    SpinTaylor/PNR vectorial default is a documented DiffGW extension.
    """
    if not isinstance(method, str):
        raise ValueError("precession_method must be 'nnlo', 'msa', 'spintaylor', or 'pnr'.")
    method_l = method.lower()
    if method_l == "ode":
        raise ValueError("precession_method='ode' is an unvalidated prototype and is not available in releases.")
    if method_l not in _METHOD_FAMILY:
        raise ValueError("precession_method must be 'nnlo', 'msa', 'spintaylor', or 'pnr'.")
    if version is not None and (isinstance(version, bool) or not isinstance(version, Integral)):
        raise ValueError(f"Unsupported precession version: {version}.")
    resolved_version = _DEFAULT_VERSION[_METHOD_FAMILY[method_l]] if version is None else int(version)
    if resolved_version not in SUPPORTED_VERSIONS:
        raise ValueError(f"Unsupported precession version: {resolved_version}.")
    if resolved_version == 300:
        resolved_version = 330
    family = precession_family(resolved_version)

    if final_spin_mod is not None and (
        isinstance(final_spin_mod, bool) or not isinstance(final_spin_mod, Integral)
    ):
        raise ValueError("final_spin_mod must be None, 0 (NNLO), 2 (Vectorial), or 3 (MSA).")
    if final_spin_mod not in (None, 0, 2, 3):
        raise ValueError("final_spin_mod must be None, 0 (NNLO), 2 (Vectorial), or 3 (MSA).")
    resolved_final_spin = _DEFAULT_FINAL_SPIN[family] if final_spin_mod is None else int(final_spin_mod)
    return family, resolved_version, resolved_final_spin


__all__ = [
    "MSA_VERSIONS",
    "NNLO_VERSIONS",
    "PNR_VERSIONS",
    "SPINTAYLOR_VERSIONS",
    "precession_family",
    "resolve_xphm_options",
]

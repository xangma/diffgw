"""Backend-independent selection of published PhenomT precession options."""

from numbers import Integral


_VERSIONS = {
    "nnlo": (10200, 10210, 10211, 10221),
    "msa": (22300, 22310, 22311, 22321),
    "numerical": (300,),
}
_DEFAULTS = {"nnlo": 10210, "msa": 22310, "numerical": 300}


def resolve_precession(precession_method="nnlo", precession_version=None):
    """Validate static model options; keep NNLO as the compatibility default.

    Version aliases 102 and 223 select the corresponding ``10`` continuation,
    as in Appendix A of Estellés et al., arXiv:2105.05872.
    """
    method = str(precession_method).lower()
    if method not in _VERSIONS:
        raise ValueError("precession_method must be 'nnlo', 'msa', or 'numerical'")
    if precession_version is None:
        return method, _DEFAULTS[method]
    if isinstance(precession_version, bool) or not isinstance(precession_version, Integral):
        raise ValueError("precession_version must be an integer model option")
    version = {102: 10210, 223: 22310}.get(precession_version, int(precession_version))
    if version not in _VERSIONS[method]:
        raise ValueError(f"precession_version for {method!r} must be one of {_VERSIONS[method]}")
    return method, version

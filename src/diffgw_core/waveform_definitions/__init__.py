"""Backend-neutral waveform catalog schema and current scoped definitions.

Importing this package never loads SymPy, PyTorch, JAX, or reference evaluators.
"""

from .schema import (
    ComponentDefinition,
    ConventionDefinition,
    OptionDefinition,
    ParameterDefinition,
    SourceDefinition,
    SourceRef,
    UnitDefinition,
    VerificationClaim,
    WaveformDefinition,
    validate_catalog,
)
from .catalog import IMRPHENOMD, IMRPHENOMXAS, TAYLORF2, WAVEFORMS

__all__ = [
    "ComponentDefinition", "ConventionDefinition", "OptionDefinition",
    "ParameterDefinition", "SourceDefinition", "SourceRef", "UnitDefinition",
    "VerificationClaim", "WaveformDefinition", "validate_catalog",
    "TAYLORF2", "IMRPHENOMD", "IMRPHENOMXAS", "WAVEFORMS",
]

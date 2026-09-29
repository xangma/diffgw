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
from .catalog import (
    CATALOG,
    CONTINUOUS_WAVE_H1_DIRECT,
    EMRI_COMPONENTS,
    IMRPHENOMD,
    IMRPHENOMHM,
    IMRPHENOMXAS,
    KERR_RINGDOWN_220_DIRECT,
    MEMORY_22_DIRECT,
    NEWTONIAN_ECCENTRIC_EMRI,
    PARAMETERIZED_POST_EINSTEIN_INSPIRAL,
    PPE_COMPONENTS,
    TAYLORF2,
    TAYLORF2_ALIGNED_SPIN,
    TAYLORF2_ECCENTRIC,
    TAYLORF2_HM,
    TAYLORF2_SPIN_QUADRUPOLE,
    TAYLORF2_TIDAL,
    TF2_ALIGNED_SPIN_COMPONENTS,
    TF2_SPIN_QUAD_COMPONENTS,
    TF2_TIDAL_COMPONENTS,
    WAVEFORMS,
)

__all__ = [
    "ComponentDefinition", "ConventionDefinition", "OptionDefinition",
    "ParameterDefinition", "SourceDefinition", "SourceRef", "UnitDefinition",
    "VerificationClaim", "WaveformDefinition", "validate_catalog",
    "CATALOG", "WAVEFORMS",
    "TAYLORF2", "TAYLORF2_ECCENTRIC", "TAYLORF2_HM",
    "IMRPHENOMD", "IMRPHENOMHM", "IMRPHENOMXAS",
    "KERR_RINGDOWN_220_DIRECT", "CONTINUOUS_WAVE_H1_DIRECT", "MEMORY_22_DIRECT",
    "TAYLORF2_ALIGNED_SPIN", "TAYLORF2_TIDAL", "TAYLORF2_SPIN_QUADRUPOLE",
    "PARAMETERIZED_POST_EINSTEIN_INSPIRAL", "NEWTONIAN_ECCENTRIC_EMRI",
    "TF2_ALIGNED_SPIN_COMPONENTS", "TF2_TIDAL_COMPONENTS", "TF2_SPIN_QUAD_COMPONENTS",
    "PPE_COMPONENTS", "EMRI_COMPONENTS",
]

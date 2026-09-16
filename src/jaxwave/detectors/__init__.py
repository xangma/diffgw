"""Gravitational-wave detector geometry and response networks in JAX."""

from .network import (
    Detector,
    DetectorNetwork,
    greenwich_mean_sidereal_time,
    DETECTOR_SITE_DATA,
    OMEGA_EARTH,
)

from .psd import (
    aLIGOZeroDetHighPower,
    aLIGOEarlyHighPower,
    AdvancedVirgo,
    KAGRA,
    EinsteinTelescope,
    CosmicExplorer,
    get_psd,
)

from .lisa import (
    LISANetwork,
    lisa_psd,
    LISA_ARM_LENGTH,
    LISA_TRANSFER_FREQ,
)

from .pta import (
    PTANetwork,
    hellings_downs_correlation,
    stochastic_gwb_spectrum,
    pulsar_line_of_sight,
    angular_separation,
    SEC_PER_YEAR,
    F_YEAR,
)

__all__ = [
    "Detector",
    "DetectorNetwork",
    "greenwich_mean_sidereal_time",
    "DETECTOR_SITE_DATA",
    "OMEGA_EARTH",
    "aLIGOZeroDetHighPower",
    "aLIGOEarlyHighPower",
    "AdvancedVirgo",
    "KAGRA",
    "EinsteinTelescope",
    "CosmicExplorer",
    "get_psd",
    "LISANetwork",
    "lisa_psd",
    "LISA_ARM_LENGTH",
    "LISA_TRANSFER_FREQ",
    "PTANetwork",
    "hellings_downs_correlation",
    "stochastic_gwb_spectrum",
    "pulsar_line_of_sight",
    "angular_separation",
    "SEC_PER_YEAR",
    "F_YEAR",
]

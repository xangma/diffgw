"""Gravitational-wave detector geometry and response networks."""

from torchwave.detectors.network import (
    Detector,
    DetectorNetwork,
    greenwich_mean_sidereal_time,
    DETECTOR_SITE_DATA,
    OMEGA_EARTH,
)

from torchwave.detectors.psd import (
    aLIGOZeroDetHighPower,
    aLIGOEarlyHighPower,
    AdvancedVirgo,
    KAGRA,
    EinsteinTelescope,
    CosmicExplorer,
    get_psd,
)

from torchwave.detectors.lisa import (
    LISANetwork,
    lisa_psd,
    LISA_ARM_LENGTH,
    LISA_TRANSFER_FREQ,
)

from torchwave.detectors.pta import (
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


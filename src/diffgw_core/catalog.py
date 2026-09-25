"""Framework-independent inventory of implemented waveform approximants.

An entry means the frontend can evaluate the model; it is not a validation or
release-qualification claim. See docs/model_manifest.json for evidence scope.
"""

from typing import Dict, List, Optional, Union


CANONICAL_APPROXIMANTS: List[str] = [
    "TaylorF2",
    "TaylorF2_HM",
    "TaylorF2_Eccentric",
    "IMRPhenomD",
    "IMRPhenomT",
    "IMRPhenomTHM",
    "IMRPhenomTP",
    "IMRPhenomTPHM",
    "IMRPhenomD_NRTidal",
    "IMRPhenomD_NRTidalv2",
    "IMRPhenomHM",
    "IMRPhenomXAS",
    "IMRPhenomXHM",
    "IMRPhenomP",
    "IMRPhenomXP",
    "IMRPhenomXPHM",
    "IMRPhenomNSBH",
    "KerrRingdown",
    "SEOBNRv4_ROM",
]

APPROXIMANT_CATEGORIES: Dict[str, List[str]] = {
    "aligned_spin": ["IMRPhenomT", "IMRPhenomTHM", "TaylorF2", "IMRPhenomD", "IMRPhenomXAS", "SEOBNRv4_ROM"],
    "higher_modes": ["IMRPhenomTHM", "IMRPhenomTPHM", "TaylorF2_HM", "IMRPhenomHM", "IMRPhenomXHM", "IMRPhenomXPHM"],
    "precessing": ["IMRPhenomTP", "IMRPhenomTPHM", "IMRPhenomP", "IMRPhenomXP", "IMRPhenomXPHM"],
    "tidal": ["IMRPhenomD_NRTidal", "IMRPhenomD_NRTidalv2", "IMRPhenomNSBH"],
    "eccentric": ["TaylorF2_Eccentric"],
    "ringdown": ["KerrRingdown"],
    "eob": ["SEOBNRv4_ROM"],
    "frequency_domain": [
        "TaylorF2", "TaylorF2_HM", "TaylorF2_Eccentric", "IMRPhenomD",
        "IMRPhenomD_NRTidal", "IMRPhenomD_NRTidalv2", "IMRPhenomHM",
        "IMRPhenomXAS", "IMRPhenomXHM", "IMRPhenomP", "IMRPhenomXP",
        "IMRPhenomXPHM", "IMRPhenomNSBH", "KerrRingdown", "SEOBNRv4_ROM",
    ],
    "time_domain": ["IMRPhenomT", "IMRPhenomTHM", "IMRPhenomTP", "IMRPhenomTPHM"],
}


def list_approximants(
    category: Optional[str] = None,
    by_category: bool = False,
) -> Union[List[str], Dict[str, List[str]]]:
    """List implemented models by canonical name or physical category.

    This is a capability inventory, not a release-qualification certificate.
    """
    if by_category:
        return {name: list(models) for name, models in APPROXIMANT_CATEGORIES.items()}
    if category is not None:
        key = category.lower().replace("-", "_").replace(" ", "_")
        if key not in APPROXIMANT_CATEGORIES:
            raise ValueError(
                f"Unknown category '{category}'. Available: {sorted(APPROXIMANT_CATEGORIES)}"
            )
        return list(APPROXIMANT_CATEGORIES[key])
    return list(CANONICAL_APPROXIMANTS)

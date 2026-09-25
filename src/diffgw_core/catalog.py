"""Framework-independent waveform inventories with explicit evidence scope.

The default inventory filters the historical frozen reference snapshot, not
current release readiness or accuracy across a model's parameter domain.
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
    "IMRPhenomPv2",
    "IMRPhenomXP",
    "IMRPhenomXPHM",
    "IMRPhenomNSBH",
    "KerrRingdown",
    "SEOBNRv4_ROM",
]

# Frozen full-polarization reference cases in docs/model_manifest.json passed for
# these public names. This is scoped to the declared cases, not a domain-wide
# accuracy or derivative certificate. Pv2 is distinct from the legacy P route;
# the ROM name denotes only the official external-data evaluator.
HISTORICAL_REFERENCE_APPROXIMANTS: List[str] = [
    "TaylorF2", "TaylorF2_Eccentric", "IMRPhenomD", "IMRPhenomD_NRTidal",
    "IMRPhenomD_NRTidalv2", "IMRPhenomHM", "IMRPhenomXAS", "IMRPhenomPv2",
    "IMRPhenomXP", "IMRPhenomNSBH", "KerrRingdown", "SEOBNRv4_ROM",
]

APPROXIMANT_CATEGORIES: Dict[str, List[str]] = {
    "aligned_spin": ["IMRPhenomT", "IMRPhenomTHM", "TaylorF2", "IMRPhenomD", "IMRPhenomXAS", "SEOBNRv4_ROM"],
    "higher_modes": ["IMRPhenomTHM", "IMRPhenomTPHM", "TaylorF2_HM", "IMRPhenomHM", "IMRPhenomXHM", "IMRPhenomXPHM"],
    "precessing": ["IMRPhenomTP", "IMRPhenomTPHM", "IMRPhenomP", "IMRPhenomPv2", "IMRPhenomXP", "IMRPhenomXPHM"],
    "tidal": ["IMRPhenomD_NRTidal", "IMRPhenomD_NRTidalv2", "IMRPhenomNSBH"],
    "eccentric": ["TaylorF2_Eccentric"],
    "ringdown": ["KerrRingdown"],
    "eob": ["SEOBNRv4_ROM"],
    "frequency_domain": [
        "TaylorF2", "TaylorF2_HM", "TaylorF2_Eccentric", "IMRPhenomD",
        "IMRPhenomD_NRTidal", "IMRPhenomD_NRTidalv2", "IMRPhenomHM",
        "IMRPhenomXAS", "IMRPhenomXHM", "IMRPhenomP", "IMRPhenomPv2", "IMRPhenomXP",
        "IMRPhenomXPHM", "IMRPhenomNSBH", "KerrRingdown", "SEOBNRv4_ROM",
    ],
    "time_domain": ["IMRPhenomT", "IMRPhenomTHM", "IMRPhenomTP", "IMRPhenomTPHM"],
}


def list_approximants(
    category: Optional[str] = None,
    by_category: bool = False,
    inventory: str = "historical_reference",
) -> Union[List[str], Dict[str, List[str]]]:
    """List models by canonical name or physical category.

    The default includes models whose declared full-polarization cases passed
    in a historical frozen reference snapshot. ``inventory="callable"`` also
    lists research implementations. Neither view certifies current release
    readiness, full parameter domains, or optional methods.
    """
    if inventory not in ("historical_reference", "callable"):
        raise ValueError("inventory must be 'historical_reference' or 'callable'")
    allowed = set(HISTORICAL_REFERENCE_APPROXIMANTS) if inventory == "historical_reference" else None
    if by_category:
        return {
            name: [model for model in models if allowed is None or model in allowed]
            for name, models in APPROXIMANT_CATEGORIES.items()
        }
    if category is not None:
        key = category.lower().replace("-", "_").replace(" ", "_")
        if key not in APPROXIMANT_CATEGORIES:
            raise ValueError(
                f"Unknown category '{category}'. Available: {sorted(APPROXIMANT_CATEGORIES)}"
            )
        return [model for model in APPROXIMANT_CATEGORIES[key] if allowed is None or model in allowed]
    return [model for model in CANONICAL_APPROXIMANTS if allowed is None or model in allowed]

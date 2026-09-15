"""Public PhenomD phase baseline with an explicitly supplied remnant spin."""
import torch

from torchwave.approximants.imrphenomd import IMRPhenomD, evaluate_kerr_qnm_220


class NSBHPhaseBaseline(IMRPhenomD):
    """PhenomD phase baseline using the NSBH supplied final spin.

    The mass remains the documented Husa PhenomD remnant mass.  Kerr QNM
    outputs are converted from ``M_f omega`` to the PhenomD convention
    ``M_total f`` using the public component contracts.
    """

    def __init__(self, final_spin: torch.Tensor):
        super().__init__()
        self.final_spin = final_spin

    def __call__(self, *args, **kwargs):
        if "remnant_provider" in kwargs:
            raise TypeError("NSBHPhaseBaseline supplies remnant_provider per call")
        return IMRPhenomD.__call__(*args, remnant_provider=self.remnant_properties, **kwargs)

    def compute_slope_peak(self, *args, **kwargs):
        if "remnant_provider" in kwargs:
            raise TypeError("NSBHPhaseBaseline supplies remnant_provider per call")
        return IMRPhenomD.compute_slope_peak(
            *args, remnant_provider=self.remnant_properties, **kwargs)

    def remnant_properties(self, mass1, mass2, chi1, chi2):
        mass, _spin, _frd, _fdamp = IMRPhenomD.remnant_properties(mass1, mass2, chi1, chi2)
        qre, qim = evaluate_kerr_qnm_220(self.final_spin)
        factor = (mass1 + mass2) / mass / (2 * torch.pi)
        return mass, self.final_spin, factor * qre, factor * qim

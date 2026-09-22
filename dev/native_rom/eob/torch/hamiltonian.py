"""Unvalidated historical spin Hamiltonian; development branch only."""

import torch
from typing import Tuple
from torchwave.approximants.eob.hamiltonian import _HamiltonianBase, Nonspinning3PNHamiltonian


class EOBHamiltonian(_HamiltonianBase):
    """Prototype spin extensions retained for development comparisons."""

    metric_potentials = Nonspinning3PNHamiltonian.metric_potentials

    def gyro_factors(self, r: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return inherited prototype spin couplings, not calibrated v4 factors."""
        u = 1.0 / r
        nu = self.nu
        if r.dim() > nu.dim():
            nu = nu.unsqueeze(-1)

        # Leading PN expansions for gyro-gravitomagnetic couplings
        g_plus = 2.0 - (9.0 / 8.0) * nu * u
        g_minus = (5.0 / 4.0) * nu * u

        return g_plus, g_minus

    def effective_hamiltonian(
        self,
        r: torch.Tensor,
        pr_star: torch.Tensor,
        pphi: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the effective Hamiltonian H_eff / mu for equatorial aligned-spin orbits.

        Args:
            r: Dimensionless radial separation r / M
            pr_star: Conjugate momentum to tortoise coordinate r_* (p_{r_*} / mu)
            pphi: Orbital angular momentum (p_phi / (mu * M))

        Returns:
            Dimensionless effective Hamiltonian H_eff / mu
        """
        a_plus = self.a_plus
        a_minus = self.a_minus
        delta = self.delta
        nu = self.nu

        if r.dim() > a_plus.dim():
            a_plus = a_plus.unsqueeze(-1)
            a_minus = a_minus.unsqueeze(-1)
            delta = delta.unsqueeze(-1)
        if r.dim() > nu.dim():
            nu = nu.unsqueeze(-1)

        r2 = r * r
        r3 = r2 * r
        a_plus2 = a_plus * a_plus

        a_pot, _, _ = self.metric_potentials(r)
        g_plus, g_minus = self.gyro_factors(r)

        # 1. Odd-in-spin contribution (spin-orbit interaction)
        # H_odd = pphi * (g_a+ * a+ + g_a- * delta * a-) / (r^3 + a+^2 * (r + 2))
        so_num = pphi * (g_plus * a_plus + g_minus * delta * a_minus)
        so_den = r3 + a_plus2 * (r + 2.0)
        h_odd = so_num / so_den

        # Damour--Nagar Eq. (7): the quadratic tortoise momentum is OUTSIDE
        # A. The quartic term is 3PN accurate after their stated truncation of
        # the tortoise transformation. The spin-spin term remains a prototype.
        spin_spin_term = (a_plus2 / r2) * (1.0 + 2.0 / r) * (pphi * pphi)
        z3 = 2.0 * nu * (4.0 - 3.0 * nu)
        angular_term = 1.0 + (pphi * pphi) / r2 + spin_spin_term
        h_even = torch.sqrt(
            pr_star.square() + a_pot * (angular_term + z3 * pr_star.pow(4) / r2)
        )

        h_eff = h_odd + h_even
        return h_eff

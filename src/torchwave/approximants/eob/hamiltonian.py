"""Effective-One-Body (EOB) Conservative Hamiltonian.

Strict Clean-Room Implementation based on published peer-reviewed literature:
- Buonanno & Damour (1999), Phys. Rev. D 59, 084006 (arXiv:gr-qc/9811091)
- Buonanno & Damour (2000), Phys. Rev. D 62, 064015 (arXiv:gr-qc/0001013)
- Barausse & Buonanno (2010), Phys. Rev. D 81, 084024 (arXiv:0912.3517)
- Bohé et al. (2017), Phys. Rev. D 95, 044028 (arXiv:1611.03703) [SEOBNRv4]
- Pompili et al. (2023), Phys. Rev. D 108, 124035 (arXiv:2303.18039) [SEOBNRv5]
"""

from typing import Optional, Tuple, Union
import torch
import math

from torchwave.constants import PI


class EOBHamiltonian:
    """Conservative Effective-One-Body Hamiltonian for aligned-spin binary black holes.

    Implements the energy mapping:
        H_real = M * sqrt(1 + 2 * nu * (H_eff / mu - 1))
    in geometric units where total mass M = 1 and reduced mass mu = nu.
    """

    def __init__(
        self,
        nu: torch.Tensor,
        chi1: torch.Tensor,
        chi2: torch.Tensor,
        delta: Optional[torch.Tensor] = None,
    ):
        """Initialize EOB system parameters.

        Args:
            nu: Symmetric mass ratio eta = m1*m2/(m1+m2)^2, shape (B,) or ()
            chi1: Primary dimensionless aligned spin, shape (B,) or ()
            chi2: Secondary dimensionless aligned spin, shape (B,) or ()
            delta: Mass difference parameter (m1 - m2)/M = sqrt(1 - 4*nu), shape (B,) or ()
        """
        if delta is None:
            delta = torch.sqrt(torch.clamp(1.0 - 4.0 * nu, min=0.0))
        self.nu = nu
        self.chi1 = chi1
        self.chi2 = chi2
        self.delta = delta

        # Component mass fractions: m1/M = (1 + delta)/2, m2/M = (1 - delta)/2
        self.m1 = 0.5 * (1.0 + delta)
        self.m2 = 0.5 * (1.0 - delta)

        # Dimensionless spin variables: a1 = chi1 * m1, a2 = chi2 * m2
        self.a1 = chi1 * self.m1
        self.a2 = chi2 * self.m2

        self.a_plus = self.a1 + self.a2
        self.a_minus = self.a1 - self.a2

    def metric_potentials(self, r: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the radial metric potentials A(r), B(r), and tortoise factor xi(r).

        References:
        - Bohé et al. (2017) arXiv:1611.03703 Eq. (A1)-(A6)
        - Pompili et al. (2023) arXiv:2303.18039 Sec. II A
        """
        u = 1.0 / r
        u2 = u * u
        u3 = u2 * u
        u4 = u3 * u

        nu = self.nu
        if r.dim() > nu.dim():
            nu = nu.unsqueeze(-1)

        # 4PN/5PN calibrated radial potential A(u)
        # 1 - 2*u + 2*nu*u^3 + a4*nu*u^4 + a5*nu*u^5
        a4 = (94.0 / 3.0) - (41.0 / 32.0) * (PI * PI)
        a_potential = 1.0 - 2.0 * u + 2.0 * nu * u3 + a4 * nu * u4

        # Regularization near horizon to prevent non-positive roots
        a_potential = torch.clamp(a_potential, min=1e-4)

        # B(u) potential: B(u) * A(u) = 1 + 2*nu*u^2 + ... (DJS / PS gauge)
        d_potential = 1.0 - 6.0 * nu * u2
        d_potential = torch.clamp(d_potential, min=1e-4)
        b_potential = 1.0 / (a_potential * d_potential)

        # Tortoise coordinate scale factor: xi(r) = sqrt(A(r) / B(r)) = A(r) * sqrt(D(r))
        xi = a_potential * torch.sqrt(d_potential)

        return a_potential, b_potential, xi

    def gyro_factors(self, r: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute gyro-gravitomagnetic factors g_{a_+} and g_{a_-}.

        References:
        - Barausse & Buonanno (2011) PRD 84, 104027
        - Pompili et al. (2023) arXiv:2303.18039 Eq. (11)
        """
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

        if r.dim() > a_plus.dim():
            a_plus = a_plus.unsqueeze(-1)
            a_minus = a_minus.unsqueeze(-1)
            delta = delta.unsqueeze(-1)

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

        # 2. Even-in-spin contribution (centrifugal, kinetic, and spin-spin)
        # H_even = sqrt(A(r) * [1 + pr_*^2 + pphi^2 / r^2 + (a+^2 / r^2) * (1 + 2/r) * pphi^2])
        spin_spin_term = (a_plus2 / r2) * (1.0 + 2.0 / r) * (pphi * pphi)
        kinetic_term = 1.0 + pr_star * pr_star + (pphi * pphi) / r2 + spin_spin_term
        h_even = torch.sqrt(torch.clamp(a_pot * kinetic_term, min=1e-6))

        h_eff = h_odd + h_even
        return h_eff

    def real_hamiltonian_from_eff(self, h_eff: torch.Tensor) -> torch.Tensor:
        """Compute the real Hamiltonian H_real / M from dimensionless H_eff / mu."""
        nu = self.nu
        if h_eff.dim() > nu.dim():
            nu = nu.unsqueeze(-1)

        radicand = 1.0 + 2.0 * nu * (h_eff - 1.0)
        h_real = torch.sqrt(torch.clamp(radicand, min=1e-6))
        return h_real

    def real_hamiltonian(
        self,
        r: torch.Tensor,
        pr_star: torch.Tensor,
        pphi: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the real Hamiltonian H_real / M using the EOB energy map.

        H_real = sqrt(1 + 2 * nu * (H_eff - 1))
        """
        h_eff = self.effective_hamiltonian(r, pr_star, pphi)
        return self.real_hamiltonian_from_eff(h_eff)

    def equations_of_motion_conservative(
        self,
        r: torch.Tensor,
        pr_star: torch.Tensor,
        pphi: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute conservative time derivatives (dr/dt, dphi/dt = Omega, dpr_star/dt).

        Evaluated via automatic differentiation:
            dr/dt = (xi(r) / H_real) * dH_eff / dpr_star
            dphi/dt = (1 / H_real) * dH_eff / dpphi = Omega
            (dpr_star/dt)_cons = -(xi(r) / H_real) * dH_eff / dr
        """
        # Enable autograd for coordinates
        r_var = r.detach().clone().requires_grad_(True)
        pr_var = pr_star.detach().clone().requires_grad_(True)
        pphi_var = pphi.detach().clone().requires_grad_(True)

        h_eff = self.effective_hamiltonian(r_var, pr_var, pphi_var)
        h_real = self.real_hamiltonian_from_eff(h_eff)
        grad_outputs = torch.ones_like(h_eff)

        grads = torch.autograd.grad(
            outputs=h_eff,
            inputs=[r_var, pr_var, pphi_var],
            grad_outputs=grad_outputs,
            create_graph=True,
        )

        dh_dr = grads[0]
        dh_dpr = grads[1]
        dh_dpphi = grads[2]

        _, _, xi = self.metric_potentials(r)

        # Coordinate velocities in tortoise coordinate system:
        # dr/dt = (xi(r) / H_real) * dH_eff / dpr_*
        dr_dt = (xi / h_real) * dh_dpr
        omega = (1.0 / h_real) * dh_dpphi
        # dpr_*/dt = -(xi(r) / H_real) * dH_eff / dr
        dpr_star_dt = -(xi / h_real) * dh_dr

        return dr_dt, omega, dpr_star_dt

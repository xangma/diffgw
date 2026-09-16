"""Prototype Effective-One-Body conservative Hamiltonian.

The nonspinning sector implements the uncalibrated 3PN model of Damour and
Nagar, PRD 76, 044003 (2007), arXiv:0704.3550v2, Eqs. (2)-(11). The inherited
spin terms have not been established as the calibrated SEOBNRv4 Hamiltonian.
This module does not implement the full SEOBNRv4 or SEOBNRv5 conservative model.
"""

from typing import Optional, Tuple
import copy
import torch

from torchwave.constants import PI


def _ordinary_tensor_attributes(model):
    """Make a local copy when inference tensors would enter an autograd graph."""
    inference_names = [
        name for name, value in vars(model).items()
        if isinstance(value, torch.Tensor) and value.is_inference()
    ]
    if not inference_names:
        return model
    local_model = copy.copy(model)
    for name in inference_names:
        setattr(local_model, name, getattr(model, name).clone())
    return local_model


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
        """Compute the 3PN Padé potentials and xi = dr/dr_* = sqrt(A/B).

        Damour--Nagar arXiv:0704.3550v2 Eqs. (2)-(3), (6) and the following
        Padé prescription: A=P^1_3[A_3PN], D=P^0_3[D_3PN], B=D/A.
        The model is defined outside its effective horizon, where A > 0.
        """
        u = 1.0 / r
        u2 = u * u
        u3 = u2 * u

        nu = self.nu
        if r.dim() > nu.dim():
            nu = nu.unsqueeze(-1)

        # Solve the Padé coefficient equations for the published 3PN series
        # 1 - 2u + 2 nu u^3 + a4 nu u^4. No calibration constants enter.
        a4 = (94.0 / 3.0) - (41.0 / 32.0) * (PI * PI)
        d1 = nu * (a4 + 4.0) / (8.0 - 2.0 * nu)
        a_potential = (1.0 + (d1 - 2.0) * u) / (
            1.0 + d1 * u + 2.0 * d1 * u2 + (4.0 * d1 - 2.0 * nu) * u3
        )
        d_potential = 1.0 / (
            1.0 + 6.0 * nu * u2 + 2.0 * nu * (26.0 - 3.0 * nu) * u3
        )
        b_potential = d_potential / a_potential
        xi = a_potential / torch.sqrt(d_potential)

        return a_potential, b_potential, xi

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
        """Return (dr/dt, Omega, dp_r*/dt), preserving requested derivatives.

        Local Hamiltonian partials require autograd even during inference;
        retain their graph only when an input needs an outer derivative.
        """
        track_grad = torch.is_grad_enabled() and any(
            value.requires_grad
            for value in (r, pr_star, pphi, self.nu, self.chi1, self.chi2, self.delta)
        )
        result = self._conservative_partials(r, pr_star, pphi)
        if track_grad:
            return result
        return tuple(value.detach() for value in result)

    @torch.inference_mode(False)
    @torch.enable_grad()
    def _conservative_partials(
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
        self = _ordinary_tensor_attributes(self)
        r, pr_star, pphi = torch.broadcast_tensors(r, pr_star, pphi)
        with torch.no_grad():
            batch_shape = self.effective_hamiltonian(r, pr_star, pphi).shape
        # Separate coordinate nodes give partial derivatives even when the
        # inputs share a graph. Expand before cloning so each binary has its
        # own coordinates: differentiating a shared scalar would sum forces
        # across parameter batches. Incoming dependencies remain differentiable.
        r_var = r.expand(batch_shape).clone().requires_grad_(True)
        pr_var = pr_star.expand(batch_shape).clone().requires_grad_(True)
        pphi_var = pphi.expand(batch_shape).clone().requires_grad_(True)

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

        _, _, xi = self.metric_potentials(r_var)

        # Coordinate velocities in tortoise coordinate system:
        # dr/dt = (xi(r) / H_real) * dH_eff / dpr_*
        dr_dt = (xi / h_real) * dh_dpr
        omega = (1.0 / h_real) * dh_dpphi
        # dpr_*/dt = -(xi(r) / H_real) * dH_eff / dr
        dpr_star_dt = -(xi / h_real) * dh_dr

        return dr_dt, omega, dpr_star_dt

"""Pure PyTorch No-U-Turn Sampler (NUTS) and Hamiltonian Monte Carlo (HMC).

Strict Clean-Room Implementation based exclusively on published literature:
- Hoffman & Gelman (2011), "The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo",
  J. Mach. Learn. Res. 15, 1593 (arXiv:1111.4246)
- Betancourt (2017), "A Conceptual Introduction to Hamiltonian Monte Carlo",
  arXiv:1701.02434
- Gelman & Rubin (1992), "Inference from Iterative Simulation Using Multiple Sequences",
  Statist. Sci. 7, 457
"""

from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
import math
import torch
import torch.nn as nn

from torchwave.conversions import as_tensor


def gelman_rubin(chains: torch.Tensor) -> torch.Tensor:
    """Compute the Gelman-Rubin potential scale reduction factor R_hat for multiple MCMC chains.

    Literature: Gelman & Rubin (1992); Hoffman & Gelman (2011).

    Args:
        chains: Tensor of MCMC samples of shape (n_chains, n_samples, n_params).
            Must have n_chains >= 2 and n_samples >= 4.

    Returns:
        Tensor of shape (n_params,) containing R_hat for each parameter.
        Values close to 1.0 (typically R_hat < 1.05 or < 1.1) indicate convergence.
    """
    if chains.dim() != 3:
        raise ValueError(f"Expected chains tensor of shape (n_chains, n_samples, n_params), got {chains.shape}")

    m_chains, n_samples, n_params = chains.shape
    if m_chains < 2:
        raise ValueError(f"Need at least 2 chains for Gelman-Rubin diagnostic, got {m_chains}")
    if n_samples < 4:
        raise ValueError(f"Need at least 4 samples per chain, got {n_samples}")

    m = float(m_chains)
    n = float(n_samples)

    # Chain means: shape (m_chains, n_params)
    chain_means = torch.mean(chains, dim=1)
    # Overall mean: shape (n_params,)
    overall_mean = torch.mean(chain_means, dim=0)

    # Between-chain variance B / n: variance of the chain means
    b_over_n = torch.sum((chain_means - overall_mean) ** 2, dim=0) / (m - 1.0)
    b = b_over_n * n

    # Within-chain sample variances s_m^2: shape (m_chains, n_params)
    diffs = chains - chain_means.unsqueeze(1)
    s_m_sq = torch.sum(diffs ** 2, dim=1) / (n - 1.0)

    # Average within-chain variance W: shape (n_params,)
    w = torch.mean(s_m_sq, dim=0)
    w_safe = torch.clamp(w, min=1e-30)

    # Pooled variance estimate: var_plus = (n - 1) / n * W + 1 / n * B
    var_plus = ((n - 1.0) / n) * w + (1.0 / n) * b

    # Potential scale reduction factor R_hat = sqrt(var_plus / W)
    r_hat = torch.sqrt(torch.clamp(var_plus / w_safe, min=1.0))
    return r_hat


def leapfrog_step(
    theta: torch.Tensor,
    p: torch.Tensor,
    grad: torch.Tensor,
    epsilon: float,
    grad_fn: Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
    inv_mass: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single leapfrog integration step for Hamiltonian dynamics.

    p(t + eps/2) = p(t) - eps/2 * grad_V(theta(t))
    theta(t + eps) = theta(t) + eps * M^(-1) * p(t + eps/2)
    p(t + eps) = p(t + eps/2) - eps/2 * grad_V(theta(t + eps))

    Returns:
        Tuple of (theta_next, p_next, log_p_next, grad_next).
    """
    if inv_mass is None:
        p_half = p - 0.5 * epsilon * grad
        theta_next = theta + epsilon * p_half
    else:
        p_half = p - 0.5 * epsilon * grad
        theta_next = theta + epsilon * (inv_mass * p_half)

    log_p_next, grad_next = grad_fn(theta_next)

    p_next = p_half - 0.5 * epsilon * grad_next
    return theta_next, p_next, log_p_next, grad_next


class NUTS:
    """Pure PyTorch No-U-Turn Sampler (NUTS) with Dual-Averaging Step-Size Adaptation.

    Literature: Hoffman & Gelman (2011) Algorithms 3, 5, 6; Betancourt (2017).
    """

    def __init__(
        self,
        log_prob_fn: Callable[[torch.Tensor], torch.Tensor],
        target_accept: float = 0.65,
        max_depth: int = 10,
        inv_mass: Optional[torch.Tensor] = None,
    ):
        """Initialize NUTS sampler.

        Args:
            log_prob_fn: Function mapping parameter tensor theta of shape (D,) to scalar
                differentiable log-posterior ln p(theta) = ln L(theta) + ln pi(theta).
            target_accept: Desired Metropolis acceptance probability delta (default 0.65).
            max_depth: Maximum binary tree recursion depth (default 10).
            inv_mass: Diagonal inverse mass matrix elements M^(-1), shape (D,).
        """
        self.log_prob_fn = log_prob_fn
        self.target_accept = target_accept
        self.max_depth = max_depth
        self.inv_mass = inv_mass

    def value_and_grad(self, theta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute log-probability and its gradient with respect to theta using autograd."""
        theta_var = theta.detach().clone().requires_grad_(True)
        log_p = self.log_prob_fn(theta_var)
        if not log_p.requires_grad or log_p.grad_fn is None:
            grad_v = torch.zeros_like(theta)
        else:
            grad = torch.autograd.grad(log_p, theta_var, create_graph=False)[0]
            # Hamiltonian potential energy is V(theta) = -log_p(theta), so grad_V = -grad_log_p
            grad_v = -grad.detach()
            grad_v = torch.nan_to_num(grad_v, nan=0.0, posinf=1e10, neginf=-1e10)
        return log_p.detach(), grad_v

    def kinetic_energy(self, p: torch.Tensor) -> torch.Tensor:
        """Compute kinetic energy K(p) = 1/2 * p^T * M^(-1) * p."""
        if self.inv_mass is None:
            return 0.5 * torch.sum(p * p)
        return 0.5 * torch.sum(p * self.inv_mass * p)

    def find_reasonable_epsilon(
        self,
        theta0: torch.Tensor,
        log_p0: torch.Tensor,
        grad_v0: torch.Tensor,
    ) -> float:
        """Heuristic initialization of step size epsilon (Hoffman & Gelman 2011 Algorithm 4)."""
        eps = 1.0
        p0 = torch.randn_like(theta0)
        h0 = -log_p0 + self.kinetic_energy(p0)

        # Take one leapfrog step
        theta_p, p_p, log_p_p, _ = leapfrog_step(
            theta0, p0, grad_v0, eps, self.value_and_grad, self.inv_mass
        )
        h_p = -log_p_p + self.kinetic_energy(p_p)

        # Delta H = H(theta', p') - H(theta, p)
        delta_h = h_p - h0
        # Determine whether to increase or decrease epsilon
        direction = 1 if -delta_h > math.log(0.5) else -1

        count = 0
        while direction * (-delta_h) > direction * math.log(0.5) and count < 20:
            eps = eps * (2.0 ** direction)
            theta_p, p_p, log_p_p, _ = leapfrog_step(
                theta0, p0, grad_v0, eps, self.value_and_grad, self.inv_mass
            )
            h_p = -log_p_p + self.kinetic_energy(p_p)
            delta_h = h_p - h0
            count += 1

        return float(max(1e-6, min(10.0, eps)))

    def _build_tree(
        self,
        theta: torch.Tensor,
        p: torch.Tensor,
        grad: torch.Tensor,
        log_u: float,
        v: int,
        j: int,
        epsilon: float,
        h0: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, bool, float, int]:
        """Recursive binary tree expansion (Hoffman & Gelman 2011 Algorithm 6).

        Returns:
            theta_minus, p_minus, grad_minus, theta_plus, p_plus, grad_plus,
            theta_prime, grad_prime, n_prime, s_prime, alpha_prime, n_alpha_prime
        """
        if j == 0:
            # Base case: take a single leapfrog step in direction v
            theta_prime, p_prime, log_p_prime, grad_prime = leapfrog_step(
                theta, p, grad, v * epsilon, self.value_and_grad, self.inv_mass
            )
            h_prime = -log_p_prime + self.kinetic_energy(p_prime)
            h_val = float(h_prime.item())

            # Metropolis acceptance criteria
            n_prime = 1 if log_u <= -h_val else 0
            # U-turn termination and divergence check (delta H > 1000)
            divergent = (h_val - h0) > 1000.0 or math.isnan(h_val)
            s_prime = (not divergent) and (log_u < (1000.0 - h_val))

            alpha = min(1.0, math.exp(min(0.0, -h_val + h0)))
            return (
                theta_prime, p_prime, grad_prime,
                theta_prime, p_prime, grad_prime,
                theta_prime, grad_prime, n_prime, s_prime, alpha, 1
            )

        # Recursion: build the first half of the tree
        (
            theta_m, p_m, grad_m, theta_p, p_p, grad_p, theta_prime, grad_prime,
            n_prime, s_prime, alpha_prime, n_alpha_prime
        ) = self._build_tree(theta, p, grad, log_u, v, j - 1, epsilon, h0)

        if s_prime:
            if v == -1:
                (
                    theta_m, p_m, grad_m, _, _, _, theta_prime2, grad_prime2,
                    n_prime2, s_prime2, alpha_prime2, n_alpha_prime2
                ) = self._build_tree(theta_m, p_m, grad_m, log_u, v, j - 1, epsilon, h0)
            else:
                (
                    _, _, _, theta_p, p_p, grad_p, theta_prime2, grad_prime2,
                    n_prime2, s_prime2, alpha_prime2, n_alpha_prime2
                ) = self._build_tree(theta_p, p_p, grad_p, log_u, v, j - 1, epsilon, h0)

            # Progressive transition probability
            total_n = n_prime + n_prime2
            if total_n > 0 and (float(torch.rand(1).item()) < float(n_prime2) / float(total_n)):
                theta_prime = theta_prime2
                grad_prime = grad_prime2

            alpha_prime = alpha_prime + alpha_prime2
            n_alpha_prime = n_alpha_prime + n_alpha_prime2

            # Check U-turn termination on full span: (theta_plus - theta_minus) . p
            diff = theta_p - theta_m
            if self.inv_mass is not None:
                p_p_eff = self.inv_mass * p_p
                p_m_eff = self.inv_mass * p_m
            else:
                p_p_eff = p_p
                p_m_eff = p_m

            u_turn_forward = torch.sum(diff * p_p_eff).item() >= 0.0
            u_turn_backward = torch.sum(diff * p_m_eff).item() >= 0.0

            s_prime = s_prime2 and u_turn_forward and u_turn_backward
            n_prime = total_n

        return (
            theta_m, p_m, grad_m, theta_p, p_p, grad_p, theta_prime, grad_prime,
            n_prime, s_prime, alpha_prime, n_alpha_prime
        )

    def sample_chain(
        self,
        theta0: torch.Tensor,
        n_samples: int = 400,
        n_warmup: int = 200,
        epsilon: Optional[float] = None,
        adapt_step_size: bool = True,
    ) -> torch.Tensor:
        """Run a single NUTS chain with dual-averaging step-size adaptation.

        Args:
            theta0: Initial parameter tensor of shape (D,).
            n_samples: Number of post-warmup samples to return.
            n_warmup: Number of warmup/adaptation steps.
            epsilon: Optional initial step size. If None, chosen heuristically.
            adapt_step_size: If True, adapt step size during warmup.

        Returns:
            Tensor of shape (n_samples, D) containing post-warmup samples.
        """
        device = theta0.device
        dtype = theta0.dtype
        theta = theta0.clone()

        log_p, grad_v = self.value_and_grad(theta)

        if epsilon is None:
            eps = self.find_reasonable_epsilon(theta, log_p, grad_v)
        else:
            eps = float(epsilon)

        # Dual-averaging parameters (Hoffman & Gelman 2011 Algorithm 5)
        gamma_da = 0.05
        t0_da = 10.0
        kappa_da = 0.75
        mu_da = math.log(10.0 * eps)
        eps_bar = 1.0
        h_bar = 0.0

        total_steps = n_warmup + n_samples
        samples: List[torch.Tensor] = []

        for m in range(1, total_steps + 1):
            # Sample momentum: p ~ N(0, M)
            p0 = torch.randn_like(theta)
            h0 = float((-log_p + self.kinetic_energy(p0)).item())

            # Slice sampling threshold: u ~ Uniform(0, exp(-H0)) => log(u) = -H0 - Exp(1)
            exp_rnd = float(torch.empty(1, device=device).exponential_(1.0).item())
            log_u = -h0 - exp_rnd

            # Initialize tree endpoints
            theta_minus = theta.clone()
            theta_plus = theta.clone()
            p_minus = p0.clone()
            p_plus = p0.clone()
            grad_minus = grad_v.clone()
            grad_plus = grad_v.clone()
            theta_next = theta.clone()
            grad_next = grad_v.clone()

            j = 0
            n = 1
            s = True

            alpha_acc = 0.0
            n_alpha = 0

            while s and j < self.max_depth:
                # Direction v ~ Uniform({-1, 1})
                v = 1 if torch.rand(1, device=device).item() > 0.5 else -1

                if v == -1:
                    (
                        theta_minus, p_minus, grad_minus, _, _, _,
                        theta_prime, grad_prime, n_prime, s_prime,
                        alpha_step, n_alpha_step
                    ) = self._build_tree(
                        theta_minus, p_minus, grad_minus, log_u, v, j, eps, h0
                    )
                else:
                    (
                        _, _, _, theta_plus, p_plus, grad_plus,
                        theta_prime, grad_prime, n_prime, s_prime,
                        alpha_step, n_alpha_step
                    ) = self._build_tree(
                        theta_plus, p_plus, grad_plus, log_u, v, j, eps, h0
                    )

                if s_prime:
                    if n_prime > 0 and (float(torch.rand(1, device=device).item()) < float(n_prime) / float(n + n_prime)):
                        theta_next = theta_prime
                        grad_next = grad_prime

                n = n + n_prime
                alpha_acc = alpha_acc + alpha_step
                n_alpha = n_alpha + n_alpha_step

                # Check U-turn condition on outermost endpoints
                diff = theta_plus - theta_minus
                if self.inv_mass is not None:
                    p_plus_eff = self.inv_mass * p_plus
                    p_minus_eff = self.inv_mass * p_minus
                else:
                    p_plus_eff = p_plus
                    p_minus_eff = p_minus

                u_turn_fwd = torch.sum(diff * p_plus_eff).item() >= 0.0
                u_turn_bwd = torch.sum(diff * p_minus_eff).item() >= 0.0

                s = s_prime and u_turn_fwd and u_turn_bwd
                j = j + 1

            theta = theta_next
            grad_v = grad_next
            log_p = self.log_prob_fn(theta)

            # Dual averaging adaptation during warmup
            if m <= n_warmup and adapt_step_size:
                avg_alpha = alpha_acc / max(1, n_alpha)
                eta = 1.0 / (float(m) + t0_da)
                h_bar = (1.0 - eta) * h_bar + eta * (self.target_accept - avg_alpha)
                log_eps = mu_da - (math.sqrt(float(m)) / gamma_da) * h_bar
                eps = math.exp(log_eps)
                m_pow = float(m) ** (-kappa_da)
                eps_bar = math.exp(m_pow * log_eps + (1.0 - m_pow) * math.log(eps_bar))
            elif m == n_warmup and adapt_step_size:
                eps = eps_bar

            if m > n_warmup:
                samples.append(theta.detach().clone())

        return torch.stack(samples)

    def sample_parallel(
        self,
        initial_positions: torch.Tensor,
        n_samples: int = 400,
        n_warmup: int = 200,
        **chain_kwargs,
    ) -> torch.Tensor:
        """Run multiple NUTS chains starting from given initial positions.

        Args:
            initial_positions: Initial positions for each chain, shape (n_chains, D).
            n_samples: Number of post-warmup samples per chain.
            n_warmup: Number of warmup samples per chain.
            **chain_kwargs: Passed to sample_chain().

        Returns:
            Tensor of samples of shape (n_chains, n_samples, D).
        """
        n_chains = initial_positions.shape[0]
        chain_outputs = []

        for c in range(n_chains):
            theta0 = initial_positions[c]
            chain_samples = self.sample_chain(
                theta0,
                n_samples=n_samples,
                n_warmup=n_warmup,
                **chain_kwargs,
            )
            chain_outputs.append(chain_samples)

        return torch.stack(chain_outputs, dim=0)

    def __call__(self, *args, **kwargs):
        return self.sample_parallel(*args, **kwargs)

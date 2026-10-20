"""Pure PyTorch No-U-Turn Sampler (NUTS) and Hamiltonian Monte Carlo (HMC).

Method references:
- Hoffman & Gelman (2014), "The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo",
  J. Mach. Learn. Res. 15, 1593–1623 (2011 preprint: arXiv:1111.4246)
  https://jmlr.org/papers/volume15/hoffman14a/hoffman14a.pdf, Algorithm 6 and Section 3.2.4.
- Betancourt (2017), "A Conceptual Introduction to Hamiltonian Monte Carlo",
  arXiv:1701.02434
- Gelman & Rubin (1992), "Inference from Iterative Simulation Using Multiple Sequences",
  Statist. Sci. 7, 457
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import math
import torch


def gelman_rubin(chains: torch.Tensor) -> torch.Tensor:
    """Compute the Gelman-Rubin potential scale reduction factor R_hat for multiple MCMC chains.

    Literature: Gelman & Rubin (1992); Hoffman & Gelman (2014).

    Args:
        chains: Tensor of MCMC samples of shape (n_chains, n_samples, n_params).
            Must have n_chains >= 2 and n_samples >= 4.

    Returns:
        Tensor of shape (n_params,) containing R_hat for each parameter.
        Values near 1 are one diagnostic and do not establish convergence.
        This implementation uses the basic, unsplit variance-ratio statistic.
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

    Literature: Hoffman & Gelman (2014) Algorithms 3, 5, 6; Betancourt (2017).
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
        if not 0.0 < target_accept < 1.0:
            raise ValueError("target_accept must be between zero and one")
        if not isinstance(max_depth, int) or max_depth < 1:
            raise ValueError("max_depth must be a positive integer")
        self.target_accept = target_accept
        self.max_depth = max_depth
        if inv_mass is not None:
            inv_mass = torch.as_tensor(inv_mass).detach().clone()
            if inv_mass.ndim != 1 or not torch.isfinite(inv_mass).all() or not (inv_mass > 0).all():
                raise ValueError("inv_mass must be a finite positive diagonal vector")
        self.inv_mass = inv_mass

    def _metric_like(self, theta: torch.Tensor) -> Optional[torch.Tensor]:
        if self.inv_mass is None:
            return None
        if self.inv_mass.shape != theta.shape:
            raise ValueError("inv_mass must match the parameter vector shape")
        return self.inv_mass.to(device=theta.device, dtype=theta.dtype)

    def _sample_momentum(self, theta: torch.Tensor) -> torch.Tensor:
        """Draw p ~ N(0, M), consistently with K(p) = p^T M^-1 p / 2."""
        momentum = torch.randn_like(theta)
        metric = self._metric_like(theta)
        return momentum if metric is None else momentum / torch.sqrt(metric)

    @staticmethod
    def _finite_state(*values: torch.Tensor) -> bool:
        return all(bool(torch.isfinite(value).all()) for value in values)

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
        return log_p.detach(), grad_v

    def kinetic_energy(self, p: torch.Tensor) -> torch.Tensor:
        """Compute kinetic energy K(p) = 1/2 * p^T * M^(-1) * p."""
        metric = self._metric_like(p)
        if metric is None:
            return 0.5 * torch.sum(p * p)
        return 0.5 * torch.sum(p * metric * p)

    def find_reasonable_epsilon(
        self,
        theta0: torch.Tensor,
        log_p0: torch.Tensor,
        grad_v0: torch.Tensor,
    ) -> float:
        """Heuristic initialization of step size epsilon (Hoffman & Gelman 2011 Algorithm 4)."""
        eps = 1.0
        p0 = self._sample_momentum(theta0)
        h0 = -log_p0 + self.kinetic_energy(p0)
        metric = self._metric_like(theta0)

        def log_acceptance(step_size: float) -> float:
            theta_p, p_p, log_p_p, grad_p = leapfrog_step(
                theta0, p0, grad_v0, step_size, self.value_and_grad, metric
            )
            h_p = -log_p_p + self.kinetic_energy(p_p)
            if not self._finite_state(theta_p, p_p, log_p_p, grad_p, h_p):
                return -math.inf
            return float((h0 - h_p).item())

        log_accept = log_acceptance(eps)
        # Determine whether to increase or decrease epsilon
        direction = 1 if log_accept > math.log(0.5) else -1

        count = 0
        while direction * log_accept > direction * math.log(0.5) and count < 20:
            eps = eps * (2.0 ** direction)
            log_accept = log_acceptance(eps)
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
        diagnostics: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, bool, float, int]:
        """Recursive binary tree expansion (Hoffman & Gelman 2011 Algorithm 6).

        Returns:
            theta_minus, p_minus, grad_minus, theta_plus, p_plus, grad_plus,
            theta_prime, grad_prime, log_p_prime, n_prime, s_prime, alpha_prime, n_alpha_prime
        """
        if j == 0:
            # Base case: take a single leapfrog step in direction v
            theta_prime, p_prime, log_p_prime, grad_prime = leapfrog_step(
                theta, p, grad, v * epsilon, self.value_and_grad, self._metric_like(theta)
            )
            h_prime = -log_p_prime + self.kinetic_energy(p_prime)
            h_val = float(h_prime.item())

            # Metropolis acceptance criteria
            finite = math.isfinite(h_val) and self._finite_state(theta_prime, p_prime, log_p_prime, grad_prime)
            n_prime = int(finite and log_u <= -h_val)
            # U-turn termination and divergence check (delta H > 1000)
            divergent = not finite or (h_val - h0) > 1000.0
            s_prime = (not divergent) and (log_u < (1000.0 - h_val))
            if diagnostics is not None:
                diagnostics["divergent"] = diagnostics["divergent"] or divergent
            alpha = math.exp(min(0.0, -h_val + h0)) if finite else 0.0
            return (
                theta_prime, p_prime, grad_prime,
                theta_prime, p_prime, grad_prime,
                theta_prime, grad_prime, log_p_prime, n_prime, s_prime, alpha, 1
            )

        # Recursion: build the first half of the tree
        (
            theta_m, p_m, grad_m, theta_p, p_p, grad_p, theta_prime, grad_prime, log_p_prime,
            n_prime, s_prime, alpha_prime, n_alpha_prime
        ) = self._build_tree(theta, p, grad, log_u, v, j - 1, epsilon, h0, diagnostics)

        if s_prime:
            if v == -1:
                (
                    theta_m, p_m, grad_m, _, _, _, theta_prime2, grad_prime2, log_p_prime2,
                    n_prime2, s_prime2, alpha_prime2, n_alpha_prime2
                ) = self._build_tree(theta_m, p_m, grad_m, log_u, v, j - 1, epsilon, h0, diagnostics)
            else:
                (
                    _, _, _, theta_p, p_p, grad_p, theta_prime2, grad_prime2, log_p_prime2,
                    n_prime2, s_prime2, alpha_prime2, n_alpha_prime2
                ) = self._build_tree(theta_p, p_p, grad_p, log_u, v, j - 1, epsilon, h0, diagnostics)

            # Progressive transition probability
            total_n = n_prime + n_prime2
            if total_n > 0 and (float(torch.rand(1).item()) < float(n_prime2) / float(total_n)):
                theta_prime = theta_prime2
                grad_prime = grad_prime2
                log_p_prime = log_p_prime2

            alpha_prime = alpha_prime + alpha_prime2
            n_alpha_prime = n_alpha_prime + n_alpha_prime2

            # Check U-turn termination on full span: (theta_plus - theta_minus) . p
            diff = theta_p - theta_m
            metric = self._metric_like(theta)
            if metric is not None:
                p_p_eff = metric * p_p
                p_m_eff = metric * p_m
            else:
                p_p_eff = p_p
                p_m_eff = p_m

            u_turn_forward = torch.sum(diff * p_p_eff).item() >= 0.0
            u_turn_backward = torch.sum(diff * p_m_eff).item() >= 0.0

            s_prime = s_prime2 and u_turn_forward and u_turn_backward
            n_prime = total_n

        return (
            theta_m, p_m, grad_m, theta_p, p_p, grad_p, theta_prime, grad_prime, log_p_prime,
            n_prime, s_prime, alpha_prime, n_alpha_prime
        )

    def sample_chain(
        self,
        theta0: torch.Tensor,
        n_samples: int = 400,
        n_warmup: int = 200,
        epsilon: Optional[float] = None,
        adapt_step_size: bool = True,
        return_diagnostics: bool = False,
        progress_callback: Optional[Callable[[int, int, Dict[str, Any]], None]] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Run a single NUTS chain with dual-averaging step-size adaptation.

        Args:
            theta0: Initial parameter tensor of shape (D,).
            n_samples: Number of post-warmup samples to return.
            n_warmup: Number of warmup/adaptation steps.
            epsilon: Optional initial step size. If None, chosen heuristically.
            adapt_step_size: If True, adapt step size during warmup.
            return_diagnostics: Return (samples, diagnostics) when True. Each diagnostic
                tensor has n_warmup + n_samples entries, including warmup transitions.
            progress_callback: Optional callback(iteration, total_steps, diagnostics),
                called after every completed transition with one-based iteration.

        Returns:
            Tensor of shape (n_samples, D) containing post-warmup samples by default.
            Optional diagnostics: warmup, divergent, tree_depth, hit_max_depth,
            acceptance_stat (final doubling mean), step_size (used in transition),
            energy (initial Hamiltonian after refreshing momentum), and leapfrog_count.
            hit_max_depth records termination by the cap while expansion could continue.
        """
        device = theta0.device
        dtype = theta0.dtype
        if theta0.ndim != 1 or not theta0.is_floating_point():
            raise ValueError("theta0 must be a floating point parameter vector")
        if not isinstance(n_samples, int) or n_samples < 1 or not isinstance(n_warmup, int) or n_warmup < 0:
            raise ValueError("n_samples must be positive and n_warmup nonnegative integers")
        metric = self._metric_like(theta0)
        theta = theta0.detach().clone()

        log_p, grad_v = self.value_and_grad(theta)
        if not self._finite_state(theta, log_p, grad_v):
            raise ValueError("initial parameter, log probability and gradient must be finite")

        if epsilon is None:
            eps = self.find_reasonable_epsilon(theta, log_p, grad_v)
        else:
            eps = float(epsilon)
        if not math.isfinite(eps) or eps <= 0.0:
            raise ValueError("epsilon must be finite and positive")

        # Dual-averaging parameters (Hoffman & Gelman 2011 Algorithm 5)
        gamma_da = 0.05
        t0_da = 10.0
        kappa_da = 0.75
        mu_da = math.log(10.0 * eps)
        eps_bar = 1.0
        h_bar = 0.0

        total_steps = n_warmup + n_samples
        samples: List[torch.Tensor] = []
        diagnostic_rows: List[Dict[str, Any]] = []
        collect_diagnostics = return_diagnostics or progress_callback is not None

        for m in range(1, total_steps + 1):
            # Sample momentum: p ~ N(0, M)
            p0 = self._sample_momentum(theta)
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
            log_p_next = log_p

            j = 0
            n = 1
            s = True

            alpha_step = 0.0
            n_alpha_step = 0
            leapfrog_count = 0
            tree_diagnostics = {"divergent": False} if collect_diagnostics else None
            transition_epsilon = eps

            while s and j < self.max_depth:
                # Direction v ~ Uniform({-1, 1})
                v = 1 if torch.rand(1, device=device).item() > 0.5 else -1

                if v == -1:
                    (
                        theta_minus, p_minus, grad_minus, _, _, _,
                        theta_prime, grad_prime, log_p_prime, n_prime, s_prime,
                        alpha_step, n_alpha_step
                    ) = self._build_tree(
                        theta_minus, p_minus, grad_minus, log_u, v, j, eps, h0,
                        diagnostics=tree_diagnostics,
                    )
                else:
                    (
                        _, _, _, theta_plus, p_plus, grad_plus,
                        theta_prime, grad_prime, log_p_prime, n_prime, s_prime,
                        alpha_step, n_alpha_step
                    ) = self._build_tree(
                        theta_plus, p_plus, grad_plus, log_u, v, j, eps, h0,
                        diagnostics=tree_diagnostics,
                    )

                if s_prime:
                    # Algorithm 6's outer transition; recursive subtree selection uses
                    # n_prime2 / (n_prime + n_prime2), a distinct probability.
                    if n_prime > 0 and (float(torch.rand(1, device=device).item()) < min(1.0, float(n_prime) / float(n))):
                        theta_next = theta_prime
                        grad_next = grad_prime
                        log_p_next = log_p_prime

                n = n + n_prime
                leapfrog_count += n_alpha_step

                # Check U-turn condition on outermost endpoints
                diff = theta_plus - theta_minus
                if metric is not None:
                    p_plus_eff = metric * p_plus
                    p_minus_eff = metric * p_minus
                else:
                    p_plus_eff = p_plus
                    p_minus_eff = p_minus

                u_turn_fwd = torch.sum(diff * p_plus_eff).item() >= 0.0
                u_turn_bwd = torch.sum(diff * p_minus_eff).item() >= 0.0

                s = s_prime and u_turn_fwd and u_turn_bwd
                j = j + 1

            theta = theta_next
            grad_v = grad_next
            log_p = log_p_next
            # Section 3.2.4 averages the FINAL doubling, not all tree expansions.
            avg_alpha = alpha_step / max(1, n_alpha_step)

            if collect_diagnostics:
                row = {
                    "warmup": m <= n_warmup,
                    "divergent": tree_diagnostics["divergent"],
                    "tree_depth": j,
                    "hit_max_depth": bool(s and j == self.max_depth),
                    "acceptance_stat": avg_alpha,
                    "step_size": transition_epsilon,
                    "energy": h0,
                    "leapfrog_count": leapfrog_count,
                }
                if return_diagnostics:
                    diagnostic_rows.append(row.copy())
                if progress_callback is not None:
                    progress_callback(m, total_steps, row.copy())

            # Dual averaging adaptation during warmup
            if m <= n_warmup and adapt_step_size:
                eta = 1.0 / (float(m) + t0_da)
                h_bar = (1.0 - eta) * h_bar + eta * (self.target_accept - avg_alpha)
                log_eps = mu_da - (math.sqrt(float(m)) / gamma_da) * h_bar
                eps = math.exp(log_eps)
                m_pow = float(m) ** (-kappa_da)
                eps_bar = math.exp(m_pow * log_eps + (1.0 - m_pow) * math.log(eps_bar))
                if m == n_warmup:
                    eps = eps_bar

            if m > n_warmup:
                samples.append(theta.detach().clone())

        result = torch.stack(samples)
        if not return_diagnostics:
            return result
        diagnostic_types = {
            "warmup": torch.bool, "divergent": torch.bool, "tree_depth": torch.int64,
            "hit_max_depth": torch.bool, "acceptance_stat": dtype, "step_size": dtype,
            "energy": dtype, "leapfrog_count": torch.int64,
        }
        diagnostics = {
            key: torch.tensor([row[key] for row in diagnostic_rows], device=device, dtype=value_dtype)
            for key, value_dtype in diagnostic_types.items()
        }
        return result, diagnostics

    def sample_parallel(
        self,
        initial_positions: torch.Tensor,
        n_samples: int = 400,
        n_warmup: int = 200,
        **chain_kwargs: Any,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Run multiple NUTS chains sequentially from given initial positions.

        Args:
            initial_positions: Initial positions for each chain, shape (n_chains, D).
            n_samples: Number of post-warmup samples per chain.
            n_warmup: Number of warmup samples per chain.
            chain_kwargs: Arbitrary keyword arguments passed to sample_chain.

        Returns:
            Tensor of samples of shape (n_chains, n_samples, D) by default.
            With return_diagnostics=True, (samples, diagnostics), with each diagnostic
            tensor shaped (n_chains, n_warmup + n_samples).
        """
        n_chains = initial_positions.shape[0]
        chain_outputs = []
        chain_diagnostics = []

        for c in range(n_chains):
            theta0 = initial_positions[c]
            chain_samples = self.sample_chain(
                theta0,
                n_samples=n_samples,
                n_warmup=n_warmup,
                **chain_kwargs,
            )
            if chain_kwargs.get("return_diagnostics", False):
                samples, diagnostics = chain_samples
                chain_outputs.append(samples)
                chain_diagnostics.append(diagnostics)
            else:
                chain_outputs.append(chain_samples)

        result = torch.stack(chain_outputs, dim=0)
        if chain_diagnostics:
            return result, {
                key: torch.stack([diagnostics[key] for diagnostics in chain_diagnostics])
                for key in chain_diagnostics[0]
            }
        return result

    def __call__(self, *args, **kwargs):
        return self.sample_parallel(*args, **kwargs)

"""Differentiable Fisher Information Matrix and Gravitational-Wave Parameter Estimation Utilities in JAX.

Strict Clean-Room Implementation based exclusively on published literature:
- Cutler & Flanagan (1994), Phys. Rev. D 49, 2658 (arXiv:gr-qc/9402014)
- Vallisneri (2008), Phys. Rev. D 77, 042001 (arXiv:gr-qc/0703086)
- Barlow (1989), Statistics: A Guide to the Use of Statistical Methods
- Veitch et al. (2015), Phys. Rev. D 91, 042003 (arXiv:1409.7788)
"""

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import math
import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy.integrate as jsi

if hasattr(jsi, "trapezoid"):
    _trapezoid = jsi.trapezoid
elif hasattr(jnp, "trapezoid"):
    _trapezoid = jnp.trapezoid
else:
    _trapezoid = jnp.trapz

from jaxwave.conversions import as_array
from jaxwave.inference.likelihood import (
    noise_weighted_inner_product,
    optimal_snr,
    matched_filter_snr,
)


def fisher_information_matrix(
    parameters_dict: Dict[str, Union[float, jnp.ndarray]],
    waveform_fn: Callable[[Dict[str, jnp.ndarray]], Union[jnp.ndarray, Dict[str, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]]],
    psd_dict: Union[jnp.ndarray, Dict[str, jnp.ndarray]],
    detector_network: Optional[Any] = None,
    f_low: float = 20.0,
    f_high: float = 1024.0,
    df: Optional[float] = None,
    frequencies: Optional[jnp.ndarray] = None,
    parameter_order: Optional[Sequence[str]] = None,
    method: str = "autodiff",
    step_sizes: Optional[Dict[str, float]] = None,
) -> Tuple[jnp.ndarray, List[str]]:
    """Compute the differentiable Fisher Information Matrix Gamma_ij using JAX autodiff or parameter differences.

    Literature: Cutler & Flanagan (1994), Vallisneri (2008).

    Formula:
        Gamma_ij = sum_{k in network} 4 Re int_{f_low}^{f_high} (1 / Sn^(k)(f)) * (dh_k(f)/dtheta_i) * (dh_k(f)/dtheta_j)* df

    Args:
        parameters_dict: Dict mapping parameter names to parameter values/arrays
        waveform_fn: Function mapping parameter dictionary to frequency-domain strain(s)
        psd_dict: Real PSD array or dict of PSDs for each detector {det: psd_det}
        detector_network: Optional DetectorNetwork instance to project (h_plus, h_cross)
        f_low: Low-frequency cutoff in Hz
        f_high: High-frequency cutoff in Hz
        df: Frequency resolution in Hz
        frequencies: Optional frequency vector in Hz
        parameter_order: Optional explicit order of parameter keys
        method: "autodiff" (forward-mode JVP), "hessian" (autodiff Whittle Hessian), or "finite_diff"
        step_sizes: Optional dict of step sizes for finite differences

    Returns:
        (fisher_matrix, param_names): (P, P) symmetric Fisher matrix and list of parameter names.
        Degenerate parameters can make the matrix singular.
    """
    if parameter_order is None:
        param_names = list(parameters_dict.keys())
    else:
        param_names = list(parameter_order)

    base_params = {
        k: as_array(parameters_dict[k], dtype=jnp.float64)
        for k in param_names
    }
    num_params = len(param_names)

    def eval_strains(p_dict: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
        wf_out = waveform_fn(p_dict)
        if isinstance(wf_out, dict):
            return wf_out
        elif isinstance(wf_out, tuple) and len(wf_out) == 2:
            hp, hc = wf_out
            if detector_network is not None:
                ra = p_dict.get("ra", jnp.array(0.0, dtype=jnp.float64))
                dec = p_dict.get("dec", jnp.array(0.0, dtype=jnp.float64))
                psi = p_dict.get("psi", jnp.array(0.0, dtype=jnp.float64))
                tg = p_dict.get("t_gps", p_dict.get("tc", jnp.array(0.0, dtype=jnp.float64)))
                freq_vec = frequencies if frequencies is not None else jnp.array(0.0, dtype=jnp.float64)
                return detector_network.project_fd_strain(freq_vec, hp, hc, ra, dec, psi, tg)
            else:
                return {"det": hp}
        elif isinstance(wf_out, (jnp.ndarray, np.ndarray)):
            return {"det": wf_out}
        else:
            raise TypeError(f"Unsupported waveform_fn output type: {type(wf_out)}")

    if method == "hessian":
        # Compute Fisher matrix via the autodiff Hessian of the Whittle log-likelihood at fiducial theta
        def neg_log_lik_vec(theta_vec: jnp.ndarray) -> jnp.ndarray:
            p_dict = {name: theta_vec[idx] for idx, name in enumerate(param_names)}
            strains = eval_strains(p_dict)
            base_strains = eval_strains(base_params)
            total = jnp.array(0.0, dtype=jnp.float64)
            for det_k, h_k in strains.items():
                if isinstance(psd_dict, dict):
                    if det_k not in psd_dict:
                        if len(psd_dict) == 1:
                            psd_k = next(iter(psd_dict.values()))
                        else:
                            continue
                    else:
                        psd_k = psd_dict[det_k]
                else:
                    psd_k = psd_dict

                d_k = base_strains[det_k]
                term_dh = noise_weighted_inner_product(
                    d_k, h_k, psd_k, df=df, frequencies=frequencies, f_low=f_low, f_high=f_high
                )
                term_hh = noise_weighted_inner_product(
                    h_k, h_k, psd_k, df=df, frequencies=frequencies, f_low=f_low, f_high=f_high
                )
                total = total + (term_dh - 0.5 * term_hh)
            return -total

        theta_0 = jnp.array([base_params[k] for k in param_names], dtype=jnp.float64)
        fisher = jax.hessian(neg_log_lik_vec)(theta_0)
        fisher = 0.5 * (fisher + fisher.T)
        return fisher, param_names

    det_jacobians: Dict[str, List[jnp.ndarray]] = {}

    if method == "autodiff":
        for i, p_name in enumerate(param_names):
            tangent = {
                k: (jnp.array(1.0, dtype=jnp.float64) if k == p_name else jnp.array(0.0, dtype=jnp.float64))
                for k in param_names
            }
            _, dh_dict = jax.jvp(eval_strains, (base_params,), (tangent,))
            for det_k, dh_i in dh_dict.items():
                if det_k not in det_jacobians:
                    det_jacobians[det_k] = []
                det_jacobians[det_k].append(dh_i)
    elif method == "finite_diff":
        for i, p_name in enumerate(param_names):
            val = float(base_params[p_name])
            if step_sizes is not None and p_name in step_sizes:
                delta = float(step_sizes[p_name])
            else:
                delta = 1.0e-6 * max(abs(val), 1.0)

            p_plus = dict(base_params)
            p_plus[p_name] = base_params[p_name] + delta
            p_minus = dict(base_params)
            p_minus[p_name] = base_params[p_name] - delta

            s_plus = eval_strains(p_plus)
            s_minus = eval_strains(p_minus)

            for det_k in s_plus:
                dh_i = (s_plus[det_k] - s_minus[det_k]) / (2.0 * delta)
                if det_k not in det_jacobians:
                    det_jacobians[det_k] = []
                det_jacobians[det_k].append(dh_i)
    else:
        raise ValueError(f"Unknown method '{method}', supported: 'autodiff', 'hessian', 'finite_diff'")

    fisher = jnp.zeros((num_params, num_params), dtype=jnp.float64)

    for det_k, jac_list in det_jacobians.items():
        if isinstance(psd_dict, dict):
            if det_k not in psd_dict:
                if len(psd_dict) == 1:
                    psd_k = next(iter(psd_dict.values()))
                else:
                    continue
            else:
                psd_k = psd_dict[det_k]
        else:
            psd_k = psd_dict

        psd_k = as_array(psd_k, dtype=jnp.float64)
        psd_safe = jnp.maximum(psd_k, 1.0e-50)
        weight = 4.0 / psd_safe

        if frequencies is not None:
            freqs = as_array(frequencies, dtype=jnp.float64)
            mask = jnp.ones(freqs.shape, dtype=bool)
            if f_low is not None:
                mask = mask & (freqs >= f_low)
            if f_high is not None:
                mask = mask & (freqs <= f_high)
            weight = jnp.where(mask, weight, 0.0)
            integration_mode = "trapezoid"
        else:
            df_val = df if df is not None else 1.0
            weight = weight * df_val
            integration_mode = "sum"

        J = jnp.stack(jac_list, axis=-1)

        if integration_mode == "trapezoid":
            for i in range(num_params):
                for j in range(i, num_params):
                    integrand = weight * jnp.real(jac_list[i] * jnp.conj(jac_list[j]))
                    integral = _trapezoid(integrand, freqs, axis=-1)
                    fisher = fisher.at[i, j].add(integral)
                    if i != j:
                        fisher = fisher.at[j, i].add(integral)
        else:
            W = weight[:, None]
            W_J = W * J
            Gamma_det = jnp.real(jnp.matmul(jnp.conj(J).T, W_J))
            Gamma_det = 0.5 * (Gamma_det + Gamma_det.T)
            fisher = fisher + Gamma_det

    return fisher, param_names


def covariance_matrix(
    fisher_matrix: jnp.ndarray,
    eps: float = 1.0e-12,
) -> jnp.ndarray:
    """Compute a regularized inverse Fisher matrix for local uncertainty forecasts.

    Interpreting this as a posterior covariance requires a suitable local Gaussian
    approximation. Eigenvalue flooring changes the inverse in ill-conditioned
    directions; inspect conditioning and regularization before interpretation.

    Literature: Vallisneri (2008) Phys. Rev. D 77, 042001.

    Args:
        fisher_matrix: (P, P) symmetric Fisher information matrix Gamma
        eps: Minimum relative eigenvalue floor (regularization threshold)

    Returns:
        (P, P) symmetric regularized inverse Fisher matrix.
    """
    gamma = as_array(fisher_matrix, dtype=jnp.float64)
    gamma_sym = 0.5 * (gamma + jnp.swapaxes(gamma, -2, -1))
    eigenvalues, eigenvectors = jnp.linalg.eigh(gamma_sym)

    max_ev = jnp.max(jnp.abs(eigenvalues))
    threshold = jnp.maximum(max_ev * eps, 1.0e-30)
    ev_reg = jnp.maximum(eigenvalues, threshold)

    inv_ev = 1.0 / ev_reg
    cov = jnp.matmul(eigenvectors, jnp.matmul(jnp.diag(inv_ev), jnp.swapaxes(eigenvectors, -2, -1)))
    return 0.5 * (cov + jnp.swapaxes(cov, -2, -1))


def sky_localization_area(
    sigma_matrix: jnp.ndarray,
    dec_idx: int,
    ra_idx: int,
    dec: Optional[Union[float, jnp.ndarray]] = None,
) -> jnp.ndarray:
    """Compute 90% credible sky localization area Delta Omega_90% in steradians.

    Literature:
    - Cutler & Flanagan (1994), Phys. Rev. D 49, 2658 Eq. 3.22
    - Barlow (1989), Statistics: A Guide to the Use of Statistical Methods

    Formula:
        Delta Omega_90% = 2 * pi * ln(10) * sqrt(Sigma_aa * Sigma_dd - Sigma_ad^2) * cos(delta)

    Args:
        sigma_matrix: Covariance matrix Sigma
        dec_idx: Parameter index for declination delta
        ra_idx: Parameter index for right ascension alpha
        dec: Central declination in radians (if None, cos(delta) defaults to 1.0)

    Returns:
        Delta Omega_90% in steradians.
    """
    sigma = as_array(sigma_matrix, dtype=jnp.float64)
    sig_aa = sigma[ra_idx, ra_idx]
    sig_dd = sigma[dec_idx, dec_idx]
    sig_ad = sigma[ra_idx, dec_idx]

    det_2d = jnp.maximum(sig_aa * sig_dd - sig_ad * sig_ad, 0.0)
    cos_fac = 1.0
    if dec is not None:
        dec_arr = as_array(dec, dtype=sigma.dtype)
        cos_fac = jnp.abs(jnp.cos(dec_arr))

    factor = 2.0 * math.pi * math.log(10.0)
    return factor * jnp.sqrt(det_2d) * cos_fac


def optimize_map(
    data: Union[jnp.ndarray, Dict[str, jnp.ndarray]],
    psd: Union[jnp.ndarray, Dict[str, jnp.ndarray]],
    initial_params: Dict[str, Union[float, jnp.ndarray]],
    waveform_fn: Callable[[Dict[str, jnp.ndarray]], Union[jnp.ndarray, Dict[str, jnp.ndarray]]],
    bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    lr: float = 1.0e-2,
    steps: int = 100,
    df: Optional[float] = None,
    frequencies: Optional[jnp.ndarray] = None,
    f_low: Optional[float] = None,
    f_high: Optional[float] = None,
) -> Tuple[Dict[str, float], List[float]]:
    """Optimize the log likelihood with JAX Adam and optional box bounds.

    This objective is equivalent to MAP for a uniform prior within the bounds.
    No general prior is evaluated. Check convergence independently of the
    requested iteration count.

    Solves: argmax_theta ln L(theta)
    where: ln L(theta) = <d | h(theta)> - 0.5 * <h(theta) | h(theta)>

    Args:
        data: Frequency-domain strain data
        psd: Power spectral density
        initial_params: Starting guess dictionary {param_name: value}
        waveform_fn: Forward model function theta -> h(theta)
        bounds: Optional box bounds {param_name: (min_val, max_val)}
        lr: Adam learning rate
        steps: Number of optimization iterations
        df: Frequency spacing in Hz
        frequencies: Frequency grid in Hz
        f_low: Low-frequency cutoff in Hz
        f_high: High-frequency cutoff in Hz

    Returns:
        (best_params_dict, loss_history)
    """
    params = {k: as_array(v, dtype=jnp.float64) for k, v in initial_params.items()}

    def loss_fn(p: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        template = waveform_fn(p)
        if isinstance(template, dict):
            overlap_tot = None
            norm_tot = None
            for det_k, h_k in template.items():
                if det_k not in data or det_k not in psd:
                    continue
                d_k = data[det_k]
                psd_k = psd[det_k]
                term_d_h = noise_weighted_inner_product(
                    d_k, h_k, psd_k, df=df, frequencies=frequencies, f_low=f_low, f_high=f_high
                )
                term_h_h = noise_weighted_inner_product(
                    h_k, h_k, psd_k, df=df, frequencies=frequencies, f_low=f_low, f_high=f_high
                )
                overlap_tot = term_d_h if overlap_tot is None else overlap_tot + term_d_h
                norm_tot = term_h_h if norm_tot is None else norm_tot + term_h_h
            log_lik = overlap_tot - 0.5 * norm_tot
        else:
            psd_arr = psd if not isinstance(psd, dict) else next(iter(psd.values()))
            data_arr = data if not isinstance(data, dict) else next(iter(data.values()))
            term_d_h = noise_weighted_inner_product(
                data_arr, template, psd_arr, df=df, frequencies=frequencies, f_low=f_low, f_high=f_high
            )
            term_h_h = noise_weighted_inner_product(
                template, template, psd_arr, df=df, frequencies=frequencies, f_low=f_low, f_high=f_high
            )
            log_lik = term_d_h - 0.5 * term_h_h

        return -log_lik

    grad_fn = jax.value_and_grad(loss_fn)

    m = {k: jnp.zeros_like(v) for k, v in params.items()}
    v_opt = {k: jnp.zeros_like(v) for k, v in params.items()}
    beta1, beta2, eps_adam = 0.9, 0.999, 1.0e-8

    loss_history: List[float] = []
    best_loss = float("inf")
    best_state: Dict[str, float] = {k: float(val) for k, val in params.items()}

    for step in range(1, steps + 1):
        loss_val, grads = grad_fn(params)
        loss_f = float(loss_val)
        loss_history.append(loss_f)

        if loss_f < best_loss:
            best_loss = loss_f
            best_state = {k: float(val) for k, val in params.items()}

        for k in params:
            g = grads[k]
            m[k] = beta1 * m[k] + (1.0 - beta1) * g
            v_opt[k] = beta2 * v_opt[k] + (1.0 - beta2) * (g * g)
            m_hat = m[k] / (1.0 - beta1 ** step)
            v_hat = v_opt[k] / (1.0 - beta2 ** step)
            new_param = params[k] - lr * m_hat / (jnp.sqrt(v_hat) + eps_adam)

            if bounds is not None and k in bounds:
                low, high = bounds[k]
                new_param = jnp.clip(new_param, low, high)
            params[k] = new_param

    return best_state, loss_history


__all__ = [
    "optimal_snr",
    "matched_filter_snr",
    "fisher_information_matrix",
    "covariance_matrix",
    "sky_localization_area",
    "optimize_map",
]

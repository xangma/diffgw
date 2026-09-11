"""Differentiable Fisher Information Matrix and Gravitational-Wave Parameter Estimation Utilities.

Strict Clean-Room Implementation based exclusively on published literature:
- Cutler & Flanagan (1994), Phys. Rev. D 49, 2658 (arXiv:gr-qc/9402014)
- Vallisneri (2008), Phys. Rev. D 77, 042001 (arXiv:gr-qc/0703086)
- Barlow (1989), Statistics: A Guide to the Use of Statistical Methods
- Veitch et al. (2015), Phys. Rev. D 91, 042003 (arXiv:1409.7788)
"""

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import math
import torch
import torch.nn as nn
import torch.autograd.forward_ad as fwAD

from torchwave.conversions import as_tensor
from torchwave.inference.likelihood import noise_weighted_inner_product


def optimal_snr(
    waveform: Union[torch.Tensor, Dict[str, torch.Tensor]],
    psd: Union[torch.Tensor, Dict[str, torch.Tensor]],
    df: Optional[Union[float, torch.Tensor]] = None,
    frequencies: Optional[torch.Tensor] = None,
    f_low: Optional[float] = None,
    f_high: Optional[float] = None,
) -> torch.Tensor:
    """Compute optimal signal-to-noise ratio (SNR) rho = sqrt(<h | h>).

    Supports single-detector waveforms or multi-detector networks.
    For a network of detectors, returns rho_net = sqrt(sum_k <h_k | h_k>).

    Args:
        waveform: Complex frequency-domain tensor or dictionary {det: h_det}
        psd: Real PSD tensor or dictionary {det: psd_det}
        df: Frequency bin width in Hz
        frequencies: Frequency grid in Hz (if non-uniform)
        f_low: Low-frequency cutoff in Hz
        f_high: High-frequency cutoff in Hz

    Returns:
        Scalar or batched optimal SNR tensor.
    """
    if isinstance(waveform, dict):
        if not isinstance(psd, dict):
            raise TypeError("When waveform is a dict, psd must also be a dict.")
        net_snr_sq = None
        for det_name, h_k in waveform.items():
            if det_name not in psd:
                continue
            snr_sq_k = noise_weighted_inner_product(
                a=h_k,
                b=h_k,
                psd=psd[det_name],
                df=df,
                frequencies=frequencies,
                f_low=f_low,
                f_high=f_high,
            )
            net_snr_sq = snr_sq_k if net_snr_sq is None else net_snr_sq + snr_sq_k
        if net_snr_sq is None:
            raise ValueError("No matching detector keys between waveform and psd dictionaries.")
        return torch.sqrt(torch.clamp(net_snr_sq, min=0.0))

    # Single detector
    psd_tensor = psd
    if isinstance(psd, dict):
        psd_tensor = next(iter(psd.values()))

    snr_sq = noise_weighted_inner_product(
        a=waveform,
        b=waveform,
        psd=psd_tensor,
        df=df,
        frequencies=frequencies,
        f_low=f_low,
        f_high=f_high,
    )
    return torch.sqrt(torch.clamp(snr_sq, min=0.0))


def matched_filter_snr(
    data: Union[torch.Tensor, Dict[str, torch.Tensor]],
    template: Union[torch.Tensor, Dict[str, torch.Tensor]],
    psd: Union[torch.Tensor, Dict[str, torch.Tensor]],
    df: Optional[Union[float, torch.Tensor]] = None,
    frequencies: Optional[torch.Tensor] = None,
    f_low: Optional[float] = None,
    f_high: Optional[float] = None,
) -> torch.Tensor:
    """Compute matched-filter SNR rho_mf = Re<d | h> / sqrt(<h | h>).

    Args:
        data: Strain data tensor or dictionary {det: d_det}
        template: Template waveform tensor or dictionary {det: h_det}
        psd: PSD tensor or dictionary {det: psd_det}
        df: Frequency bin width in Hz
        frequencies: Frequency grid in Hz (if non-uniform)
        f_low: Low-frequency cutoff in Hz
        f_high: High-frequency cutoff in Hz

    Returns:
        Scalar or batched matched-filter SNR tensor.
    """
    if isinstance(template, dict):
        if not isinstance(data, dict) or not isinstance(psd, dict):
            raise TypeError("When template is a dict, data and psd must also be dicts.")
        net_overlap = None
        net_template_sq = None
        for det_name, h_k in template.items():
            if det_name not in data or det_name not in psd:
                continue
            d_k = data[det_name]
            psd_k = psd[det_name]
            overlap_k = noise_weighted_inner_product(
                a=d_k,
                b=h_k,
                psd=psd_k,
                df=df,
                frequencies=frequencies,
                f_low=f_low,
                f_high=f_high,
            )
            temp_sq_k = noise_weighted_inner_product(
                a=h_k,
                b=h_k,
                psd=psd_k,
                df=df,
                frequencies=frequencies,
                f_low=f_low,
                f_high=f_high,
            )
            net_overlap = overlap_k if net_overlap is None else net_overlap + overlap_k
            net_template_sq = temp_sq_k if net_template_sq is None else net_template_sq + temp_sq_k
        if net_overlap is None:
            raise ValueError("No matching detector keys between template, data, and psd.")
        denom = torch.sqrt(torch.clamp(net_template_sq, min=1.0e-40))
        return net_overlap / denom

    # Single detector
    psd_tensor = psd
    if isinstance(psd, dict):
        psd_tensor = next(iter(psd.values()))
    data_tensor = data
    if isinstance(data, dict):
        data_tensor = next(iter(data.values()))

    overlap = noise_weighted_inner_product(
        a=data_tensor,
        b=template,
        psd=psd_tensor,
        df=df,
        frequencies=frequencies,
        f_low=f_low,
        f_high=f_high,
    )
    temp_sq = noise_weighted_inner_product(
        a=template,
        b=template,
        psd=psd_tensor,
        df=df,
        frequencies=frequencies,
        f_low=f_low,
        f_high=f_high,
    )
    denom = torch.sqrt(torch.clamp(temp_sq, min=1.0e-40))
    return overlap / denom


def fisher_information_matrix(
    parameters_dict: Dict[str, Union[float, torch.Tensor]],
    waveform_fn: Callable[[Dict[str, torch.Tensor]], Union[torch.Tensor, Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]],
    psd_dict: Union[torch.Tensor, Dict[str, torch.Tensor]],
    detector_network: Optional[Any] = None,
    f_low: float = 20.0,
    f_high: float = 1024.0,
    df: Optional[float] = None,
    frequencies: Optional[torch.Tensor] = None,
    parameter_order: Optional[Sequence[str]] = None,
) -> Tuple[torch.Tensor, List[str]]:
    """Compute the differentiable Fisher Information Matrix Gamma_ij using PyTorch automatic differentiation.

    Literature: Cutler & Flanagan (1994), Vallisneri (2008).

    Formula:
        Gamma_ij = sum_{k in network} 4 Re int_{f_low}^{f_high} (1 / Sn^(k)(f)) * (dh_k(f)/dtheta_i) * (dh_k(f)/dtheta_j)* df

    Args:
        parameters_dict: Dict mapping parameter names to parameter values/tensors
        waveform_fn: Function mapping parameter dictionary to frequency-domain strain(s)
        psd_dict: Real PSD tensor or dict of PSDs for each detector {det: psd_det}
        detector_network: Optional DetectorNetwork instance to project (h_plus, h_cross)
        f_low: Low-frequency cutoff in Hz
        f_high: High-frequency cutoff in Hz
        df: Frequency resolution in Hz
        frequencies: Optional frequency vector in Hz
        parameter_order: Optional explicit order of parameter keys

    Returns:
        (fisher_matrix, param_names): (P, P) symmetric positive definite Fisher matrix and list of parameter names.
    """
    if parameter_order is None:
        param_names = list(parameters_dict.keys())
    else:
        param_names = list(parameter_order)

    device = None
    dtype = torch.float64
    for k in param_names:
        v = parameters_dict[k]
        if isinstance(v, torch.Tensor):
            device = v.device
            if v.is_floating_point():
                dtype = v.dtype
            break
    if device is None:
        device = torch.device("cpu")

    base_params = {}
    for k in param_names:
        val = parameters_dict[k]
        if not isinstance(val, torch.Tensor):
            base_params[k] = torch.tensor(val, dtype=dtype, device=device)
        else:
            base_params[k] = val.detach().to(device=device, dtype=dtype)

    num_params = len(param_names)

    def eval_strains(p_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        wf_out = waveform_fn(p_dict)
        if isinstance(wf_out, dict):
            return wf_out
        elif isinstance(wf_out, tuple) and len(wf_out) == 2:
            hp, hc = wf_out
            if detector_network is not None:
                ra = p_dict.get("ra", torch.tensor(0.0, device=device, dtype=dtype))
                dec = p_dict.get("dec", torch.tensor(0.0, device=device, dtype=dtype))
                psi = p_dict.get("psi", torch.tensor(0.0, device=device, dtype=dtype))
                tg = p_dict.get("t_gps", p_dict.get("tc", torch.tensor(0.0, device=device, dtype=dtype)))
                freq_vec = frequencies if frequencies is not None else torch.tensor(0.0, device=device, dtype=dtype)
                return detector_network.project_fd_strain(freq_vec, hp, hc, ra, dec, psi, tg)
            else:
                return {"det": hp}
        elif isinstance(wf_out, torch.Tensor):
            return {"det": wf_out}
        else:
            raise TypeError(f"Unsupported waveform_fn output type: {type(wf_out)}")

    det_jacobians: Dict[str, List[torch.Tensor]] = {}

    for i, p_name in enumerate(param_names):
        tangents = {k: torch.zeros_like(base_params[k]) for k in param_names}
        tangents[p_name] = torch.ones_like(base_params[p_name])

        with fwAD.dual_level():
            dual_params = {k: fwAD.make_dual(base_params[k], tangents[k]) for k in param_names}
            strains = eval_strains(dual_params)
            for det_k, strain_k in strains.items():
                unpacked = fwAD.unpack_dual(strain_k)
                dh_i = unpacked.tangent
                if dh_i is None:
                    dh_i = torch.zeros_like(unpacked.primal)
                if det_k not in det_jacobians:
                    det_jacobians[det_k] = []
                det_jacobians[det_k].append(dh_i)

    fisher = torch.zeros((num_params, num_params), dtype=dtype, device=device)

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

        psd_safe = torch.clamp(psd_k, min=1.0e-50)
        weight = 4.0 / psd_safe

        if frequencies is not None:
            mask = torch.ones_like(frequencies, dtype=torch.bool)
            if f_low is not None:
                mask = mask & (frequencies >= f_low)
            if f_high is not None:
                mask = mask & (frequencies <= f_high)
            weight = torch.where(mask, weight, torch.zeros_like(weight))
            integration_mode = "trapezoid"
        else:
            df_val = df if df is not None else 1.0
            weight = weight * df_val
            integration_mode = "sum"

        J = torch.stack(jac_list, dim=-1)

        if integration_mode == "trapezoid":
            for i in range(num_params):
                for j in range(i, num_params):
                    integrand = weight * torch.real(jac_list[i] * torch.conj(jac_list[j]))
                    integral = torch.trapezoid(integrand, frequencies, dim=-1)
                    fisher[i, j] = fisher[i, j] + integral
                    if i != j:
                        fisher[j, i] = fisher[i, j]
        else:
            W = weight.unsqueeze(-1)
            W_J = W * J
            Gamma_det = torch.real(torch.matmul(torch.conj(J).transpose(-2, -1), W_J))
            Gamma_det = 0.5 * (Gamma_det + Gamma_det.transpose(-2, -1))
            fisher = fisher + Gamma_det

    return fisher, param_names


def covariance_matrix(
    fisher_matrix: torch.Tensor,
    eps: float = 1.0e-12,
) -> torch.Tensor:
    """Invert Fisher information matrix with eigenvalue regularization to obtain covariance matrix Sigma = Gamma^(-1).

    Literature: Vallisneri (2008) Phys. Rev. D 77, 042001.

    Args:
        fisher_matrix: (P, P) symmetric Fisher information matrix Gamma
        eps: Minimum relative eigenvalue floor (regularization threshold)

    Returns:
        (P, P) regularized symmetric covariance matrix Sigma.
    """
    gamma_sym = 0.5 * (fisher_matrix + fisher_matrix.transpose(-2, -1))
    eigenvalues, eigenvectors = torch.linalg.eigh(gamma_sym)

    max_ev = torch.max(torch.abs(eigenvalues))
    threshold = torch.clamp(max_ev * eps, min=1.0e-30)
    ev_reg = torch.clamp(eigenvalues, min=threshold)

    inv_ev = 1.0 / ev_reg
    cov = torch.matmul(eigenvectors, torch.matmul(torch.diag_embed(inv_ev), eigenvectors.transpose(-2, -1)))
    return 0.5 * (cov + cov.transpose(-2, -1))


def sky_localization_area(
    sigma_matrix: torch.Tensor,
    dec_idx: int,
    ra_idx: int,
    dec: Optional[Union[float, torch.Tensor]] = None,
) -> torch.Tensor:
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
    sig_aa = sigma_matrix[ra_idx, ra_idx]
    sig_dd = sigma_matrix[dec_idx, dec_idx]
    sig_ad = sigma_matrix[ra_idx, dec_idx]

    det_2d = torch.clamp(sig_aa * sig_dd - sig_ad * sig_ad, min=0.0)
    cos_fac = 1.0
    if dec is not None:
        dec_t = as_tensor(dec, device=sigma_matrix.device, dtype=sigma_matrix.dtype)
        cos_fac = torch.abs(torch.cos(dec_t))

    factor = 2.0 * math.pi * math.log(10.0)
    return factor * torch.sqrt(det_2d) * cos_fac


def optimize_map(
    data: Union[torch.Tensor, Dict[str, torch.Tensor]],
    psd: Union[torch.Tensor, Dict[str, torch.Tensor]],
    initial_params: Dict[str, Union[float, torch.Tensor]],
    waveform_fn: Callable[[Dict[str, torch.Tensor]], Union[torch.Tensor, Dict[str, torch.Tensor]]],
    bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    lr: float = 1.0e-2,
    steps: int = 100,
    df: Optional[float] = None,
    frequencies: Optional[torch.Tensor] = None,
    f_low: Optional[float] = None,
    f_high: Optional[float] = None,
) -> Tuple[Dict[str, float], List[float]]:
    """Maximum A Posteriori (MAP) parameter optimizer driven directly by gradient of log-likelihood.

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
    device = None
    dtype = torch.float64
    for v in initial_params.values():
        if isinstance(v, torch.Tensor):
            device = v.device
            if v.is_floating_point():
                dtype = v.dtype
            break
    if device is None:
        device = torch.device("cpu")

    trainable_params: Dict[str, torch.nn.Parameter] = {}
    for k, v in initial_params.items():
        v_tensor = as_tensor(v, device=device, dtype=dtype)
        trainable_params[k] = torch.nn.Parameter(v_tensor.clone())

    optimizer = torch.optim.Adam(list(trainable_params.values()), lr=lr)

    loss_history: List[float] = []
    best_loss = float("inf")
    best_state: Dict[str, float] = {}

    for step in range(steps):
        optimizer.zero_grad()

        template = waveform_fn(trainable_params)

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
            psd_tensor = psd if not isinstance(psd, dict) else next(iter(psd.values()))
            data_tensor = data if not isinstance(data, dict) else next(iter(data.values()))
            term_d_h = noise_weighted_inner_product(
                data_tensor, template, psd_tensor, df=df, frequencies=frequencies, f_low=f_low, f_high=f_high
            )
            term_h_h = noise_weighted_inner_product(
                template, template, psd_tensor, df=df, frequencies=frequencies, f_low=f_low, f_high=f_high
            )
            log_lik = term_d_h - 0.5 * term_h_h

        loss = -log_lik

        loss.backward()
        optimizer.step()

        if bounds is not None:
            with torch.no_grad():
                for k, (low, high) in bounds.items():
                    if k in trainable_params:
                        trainable_params[k].clamp_(low, high)

        loss_val = loss.item()
        loss_history.append(loss_val)

        if loss_val < best_loss:
            best_loss = loss_val
            best_state = {k: v.detach().item() for k, v in trainable_params.items()}

    return best_state, loss_history

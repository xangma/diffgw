"""Development-only Torch/JAX evaluation of independently fitted ROM fields.

The caller must provide a qualified, preconditioned :class:`ReducedTensorInterpolant`.
This module contains no waveform table and is not imported by either release API.
"""

import numpy as np


def _nodal_derivative_matrix(nodes):
    weights = (-1.) ** np.arange(len(nodes))
    weights[[0, -1]] *= .5
    difference = nodes[:, None] - nodes[None, :]
    np.fill_diagonal(difference, 1.)
    matrix = weights[None, :] / (weights[:, None] * difference)
    np.fill_diagonal(matrix, 0.)
    np.fill_diagonal(matrix, -np.sum(matrix, axis=-1))
    return matrix


def _torch_cardinal(torch, nodes_np, query):
    nodes = torch.as_tensor(nodes_np, dtype=query.dtype, device=query.device)
    weights = torch.as_tensor((-1.) ** np.arange(len(nodes_np)),
                              dtype=query.dtype, device=query.device)
    weights = weights.clone()
    weights[[0, -1]] *= .5
    matrix = torch.as_tensor(_nodal_derivative_matrix(nodes_np),
                             dtype=query.dtype, device=query.device)
    diff = query[..., None] - nodes
    hit = torch.abs(diff) <= 8 * torch.finfo(query.dtype).eps * torch.maximum(
        torch.ones_like(nodes), torch.abs(nodes))
    ratio = weights / torch.where(hit, torch.ones_like(diff), diff)
    ordinary = ratio / ratio.sum(dim=-1, keepdim=True)
    index = hit.to(torch.int64).argmax(dim=-1)
    nodal = torch.nn.functional.one_hot(index, len(nodes_np)).to(query.dtype)
    nodal = nodal + (query - nodes[index])[..., None] * matrix[index]
    return torch.where(hit.any(dim=-1, keepdim=True), nodal, ordinary)


def _jax_cardinal(jnp, nodes_np, query):
    nodes = jnp.asarray(nodes_np, dtype=query.dtype)
    weights = jnp.asarray((-1.) ** np.arange(len(nodes_np)), dtype=query.dtype)
    weights = weights.at[jnp.array([0, len(nodes_np)-1])].multiply(.5)
    matrix = jnp.asarray(_nodal_derivative_matrix(nodes_np), dtype=query.dtype)
    diff = query[..., None] - nodes
    hit = jnp.abs(diff) <= 8 * jnp.finfo(query.dtype).eps * jnp.maximum(
        1., jnp.abs(nodes))
    ratio = weights / jnp.where(hit, 1., diff)
    ordinary = ratio / jnp.sum(ratio, axis=-1, keepdims=True)
    index = jnp.argmax(hit, axis=-1)
    nodal = jnp.eye(len(nodes_np), dtype=query.dtype)[index]
    nodal = nodal + (query - nodes[index])[..., None] * matrix[index]
    return jnp.where(jnp.any(hit, axis=-1, keepdims=True), nodal, ordinary)


def evaluate_torch(model, q, chi1, chi2):
    """Evaluate an in-memory reduced interpolant with Torch autograd."""
    import torch

    inputs = (q, chi1, chi2)
    device = next((x.device for x in inputs if isinstance(x, torch.Tensor)), torch.device("cpu"))
    queries = torch.broadcast_tensors(*(
        torch.as_tensor(x, dtype=torch.float64, device=device) for x in inputs))
    cardinal = [_torch_cardinal(torch, nodes, query)
                for nodes, query in zip(model.nodes, queries)]

    def field(part):
        coefficients = torch.as_tensor(part.coefficients, dtype=torch.float64, device=device)
        basis = torch.as_tensor(part.basis, dtype=torch.float64, device=device)
        mean = torch.as_tensor(part.mean, dtype=torch.float64, device=device)
        return mean + torch.einsum("...i,...j,...k,ijkl->...l", *cardinal,
                                    coefficients) @ basis

    amp, phase = field(model.log_amplitude), field(model.phase)
    waveform = torch.exp(amp + 1j * phase)
    outside = torch.zeros_like(queries[0], dtype=torch.bool)
    for nodes, query in zip(model.nodes, queries):
        outside = outside | (query < nodes[0]) | (query > nodes[-1])
    return torch.where(outside[..., None], torch.full_like(waveform, complex(float("nan"), 0)),
                       waveform)


def evaluate_jax(model, q, chi1, chi2):
    """Evaluate an in-memory reduced interpolant with JAX autodiff and JIT."""
    import jax
    import jax.numpy as jnp

    if not jax.config.jax_enable_x64:
        raise RuntimeError("development ROM evaluation requires JAX float64 enabled")
    queries = jnp.broadcast_arrays(*(
        jnp.asarray(x, dtype=jnp.float64) for x in (q, chi1, chi2)))
    cardinal = [_jax_cardinal(jnp, nodes, query)
                for nodes, query in zip(model.nodes, queries)]

    def field(part):
        coefficients = jnp.asarray(part.coefficients, dtype=jnp.float64)
        basis = jnp.asarray(part.basis, dtype=jnp.float64)
        mean = jnp.asarray(part.mean, dtype=jnp.float64)
        return mean + jnp.einsum("...i,...j,...k,ijkl->...l", *cardinal,
                                 coefficients) @ basis

    amp, phase = field(model.log_amplitude), field(model.phase)
    waveform = jnp.exp(amp + 1j * phase)
    outside = jnp.zeros_like(queries[0], dtype=bool)
    for nodes, query in zip(model.nodes, queries):
        outside = outside | (query < nodes[0]) | (query > nodes[-1])
    return jnp.where(outside[..., None], jnp.nan + 0j, waveform)

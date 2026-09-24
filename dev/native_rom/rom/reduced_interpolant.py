"""Reduced-basis tensor interpolation for a future, independently trained ROM.

This module accepts *preconditioned* log-amplitude and phase arrays on a common
frequency grid. Their gauge, physical generator, and holdout domain must be
specified and qualified separately. Nothing here generates waveforms, reads
the historical ROM tables, serializes an asset, or registers a runtime model.
"""

from dataclasses import dataclass

import numpy as np


def _chebyshev_nodes(nodes):
    nodes = np.asarray(nodes, dtype=np.float64)
    if (nodes.ndim != 1 or nodes.size < 2 or not np.isfinite(nodes).all()
            or not np.all(np.diff(nodes) > 0)):
        raise ValueError("parameter nodes must be finite, distinct, and increasing")
    expected = (nodes[0] + nodes[-1]) / 2 - (nodes[-1] - nodes[0]) / 2 * np.cos(
        np.pi * np.arange(nodes.size) / (nodes.size - 1))
    if not np.allclose(nodes, expected, rtol=0, atol=1e-12 * max(1., abs(nodes[-1]))):
        raise ValueError("parameter nodes must be Chebyshev-Lobatto nodes")
    return nodes.copy()


def _basis_and_derivative(nodes, query):
    """Barycentric cardinal weights and their physical-coordinate derivatives."""
    if np.any((query < nodes[0]) | (query > nodes[-1])):
        raise ValueError("ROM parameter outside its training domain")
    weights = (-1.) ** np.arange(nodes.size)
    weights[[0, -1]] *= .5
    difference = query[..., None] - nodes
    near_node = abs(difference) <= 8 * np.finfo(np.float64).eps * np.maximum(
        1., abs(nodes))
    safe = np.where(near_node, 1., difference)
    ratios = weights / safe
    cardinal = ratios / np.sum(ratios, axis=-1, keepdims=True)
    log_derivative = np.sum(cardinal / safe, axis=-1, keepdims=True)
    derivative = cardinal * (log_derivative - 1 / safe)

    # At a node, use the spectral differentiation matrix row instead of a
    # cancellation-prone limit of the barycentric formula.
    at_node = np.any(near_node, axis=-1)
    index = np.argmax(near_node, axis=-1)
    node_difference = nodes[index, None] - nodes
    self_node = np.arange(nodes.size) == index[..., None]
    node_safe = np.where(self_node, 1., node_difference)
    node_derivative = weights / (weights[index, None] * node_safe)
    node_derivative = np.where(self_node, 0., node_derivative)
    node_derivative = np.where(
        self_node, -np.sum(node_derivative, axis=-1, keepdims=True), node_derivative)
    node_cardinal = self_node.astype(np.float64) + (
        query - nodes[index])[..., None] * node_derivative
    cardinal = np.where(at_node[..., None], node_cardinal, cardinal)
    derivative = np.where(at_node[..., None], node_derivative, derivative)
    return cardinal, derivative


@dataclass(frozen=True)
class ReducedField:
    mean: np.ndarray
    basis: np.ndarray
    coefficients: np.ndarray
    training_relative_error: float


def _fit_field(data, rank):
    samples = data.reshape(-1, data.shape[-1])
    mean = np.mean(samples, axis=0)
    centered = samples - mean
    _, singular_values, basis = np.linalg.svd(centered, full_matrices=False)
    if not np.isfinite(singular_values).all():
        raise ValueError("SVD returned nonfinite singular values")
    basis = basis[:rank]
    coefficients = (centered @ basis.T).reshape(data.shape[:-1] + (rank,))
    reconstructed = coefficients.reshape(-1, rank) @ basis + mean
    scale = np.linalg.norm(samples)
    error = np.linalg.norm(reconstructed - samples) / max(scale, np.finfo(float).tiny)
    return ReducedField(mean, basis, coefficients, float(error))


@dataclass(frozen=True)
class ReducedTensorInterpolant:
    """In-memory three-parameter ROM for preconditioned log amplitude and phase."""

    nodes: tuple[np.ndarray, np.ndarray, np.ndarray]
    frequency: np.ndarray
    log_amplitude: ReducedField
    phase: ReducedField

    @classmethod
    def fit(cls, nodes, frequency, log_amplitude, phase, *, ranks):
        if len(nodes) != 3 or len(ranks) != 2:
            raise ValueError("expected three parameter grids and two ranks")
        grids = tuple(_chebyshev_nodes(n) for n in nodes)
        frequency = np.asarray(frequency, dtype=np.float64)
        if (frequency.ndim != 1 or frequency.size < 2
                or not np.isfinite(frequency).all()
                or not np.all(np.diff(frequency) > 0)):
            raise ValueError("frequency grid must be finite and increasing")
        shape = tuple(len(n) for n in grids) + (len(frequency),)
        fields = tuple(np.asarray(a, dtype=np.float64) for a in (log_amplitude, phase))
        if any(a.shape != shape or not np.isfinite(a).all() for a in fields):
            raise ValueError("training fields must be finite and match the tensor grid")
        if any(not isinstance(k, int) or k < 1 or k > min(np.prod(shape[:-1]), shape[-1])
               for k in ranks):
            raise ValueError("each rank must fit the sample and frequency dimensions")
        return cls(grids, frequency.copy(),
                   _fit_field(fields[0], ranks[0]), _fit_field(fields[1], ranks[1]))

    def evaluate(self, q, chi1, chi2, *, derivatives=False):
        """Return complex spectrum and, optionally, its three parameter derivatives."""
        queries = np.broadcast_arrays(q, chi1, chi2)
        if any(not np.isfinite(a).all() for a in queries):
            raise ValueError("ROM parameters must be finite")
        factors = [_basis_and_derivative(n, x) for n, x in zip(self.nodes, queries)]

        def field(part, axes):
            coefficients = np.einsum(
                "...i,...j,...k,ijkl->...l", *axes, part.coefficients, optimize=True)
            return part.mean + coefficients @ part.basis

        cardinal = [pair[0] for pair in factors]
        amp = field(self.log_amplitude, cardinal)
        phase = field(self.phase, cardinal)
        waveform = np.exp(amp + 1j * phase)
        if not derivatives:
            return waveform
        jacobian = []
        for axis in range(3):
            differentiated = [pair[1] if i == axis else pair[0]
                                for i, pair in enumerate(factors)]
            damp = field(self.log_amplitude, differentiated) - self.log_amplitude.mean
            dphase = field(self.phase, differentiated) - self.phase.mean
            jacobian.append(waveform * (damp + 1j * dphase))
        return waveform, np.stack(jacobian, axis=-2)

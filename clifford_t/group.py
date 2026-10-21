"""Exhaustive enumeration of the single-qubit Clifford+T group.

Every Clifford+T gate has a minimal T-count representative

    C_0 T C_1 T ... T C_t

with each C_i in the single-qubit Clifford group. T† does not need its own
generator: T† = S³ T, and S³ is Clifford, so it already appears at T-count 1.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from clifford_t.unitaries import H, I, S, T, X, Y, Z

_DECIMALS = 8
_PAULI = np.stack((X, Y, Z))


@dataclass(frozen=True)
class GateBank:
    """Distinct gates, each stored once, at its minimal T-count."""

    matrices: np.ndarray
    t_count: np.ndarray
    words: tuple[tuple[str, ...], ...]

    def at_most(self, t_count: int) -> GateBank:
        mask = self.t_count <= t_count
        return GateBank(
            self.matrices[mask],
            self.t_count[mask],
            tuple(word for word, keep in zip(self.words, mask) if keep),
        )


def _keys(matrices: np.ndarray) -> list[bytes]:
    """Bytes of the SO(3) action, so a global phase does not change the key.

    ``R_{ba} = (1/2) Tr(σ_b U σ_a U†)`` is the Bloch rotation implemented by
    ``U``. It is insensitive to ``U → e^{iα} U``.
    """
    matrices = np.asarray(matrices, dtype=np.complex128)
    if matrices.ndim == 2:
        matrices = matrices[None, ...]
    adjoint = np.conjugate(np.swapaxes(matrices, -1, -2))
    conjugated = np.einsum("nij,ajk,nkl->nail", matrices, _PAULI, adjoint, optimize=True)
    coeffs = np.real(np.einsum("bij,naji->nba", _PAULI, conjugated, optimize=True))
    packed = np.round(0.5 * coeffs, _DECIMALS)
    packed[packed == 0] = 0.0
    packed = packed.reshape(len(matrices), 9)
    raw = np.ascontiguousarray(packed).view(np.uint8).reshape(len(packed), -1)
    return [raw[i].tobytes() for i in range(len(raw))]


def _key(matrix: np.ndarray) -> bytes:
    return _keys(matrix[None, ...])[0]


def enumerate_cliffords() -> tuple[np.ndarray, tuple[tuple[str, ...], ...]]:
    """The 24 single-qubit Cliffords, each with a shortest H/S word."""
    generators = (("H", H), ("S", S))
    matrices = [I.copy()]
    words: list[tuple[str, ...]] = [()]
    seen = {_key(I)}
    queue = deque([0])
    while queue:
        index = queue.popleft()
        for name, generator in generators:
            product = generator @ matrices[index]
            key = _key(product)
            if key in seen:
                continue
            seen.add(key)
            matrices.append(product)
            words.append(words[index] + (name,))
            queue.append(len(matrices) - 1)
    return np.stack(matrices), tuple(words)


def build_group(max_t_count: int) -> GateBank:
    """All Clifford+T gates whose minimal T-count is at most ``max_t_count``."""
    if max_t_count < 0:
        raise ValueError("max_t_count must be non-negative")
    clifford_matrices, clifford_words = enumerate_cliffords()
    seen = set(_keys(clifford_matrices))
    shell_matrices = clifford_matrices
    shell_words = clifford_words
    stored = [clifford_matrices]
    stored_counts = [np.zeros(len(clifford_matrices), dtype=np.int64)]
    stored_words: list[tuple[str, ...]] = list(clifford_words)

    for t_count in range(max_t_count):
        followed_by_t = np.einsum("ij,njk->nik", T, shell_matrices)
        candidates = np.einsum("cij,njk->cnik", clifford_matrices, followed_by_t)
        candidates = candidates.reshape(-1, 2, 2)
        shell_size = len(shell_matrices)
        fresh_matrices = []
        fresh_words: list[tuple[str, ...]] = []
        for key, candidate_index in zip(_keys(candidates), range(len(candidates))):
            if key in seen:
                continue
            seen.add(key)
            clifford_index, previous_index = divmod(candidate_index, shell_size)
            fresh_matrices.append(candidates[candidate_index])
            fresh_words.append(shell_words[previous_index] + ("T",) + clifford_words[clifford_index])
        if not fresh_matrices:
            break
        shell_matrices = np.stack(fresh_matrices)
        shell_words = tuple(fresh_words)
        stored.append(shell_matrices)
        stored_counts.append(np.full(len(shell_matrices), t_count + 1, dtype=np.int64))
        stored_words.extend(fresh_words)

    return GateBank(
        matrices=np.concatenate(stored),
        t_count=np.concatenate(stored_counts),
        words=tuple(stored_words),
    )

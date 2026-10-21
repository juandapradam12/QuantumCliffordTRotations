"""Phase-invariant distance between single-qubit gates."""

from __future__ import annotations

import numpy as np


def process_fidelity(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Process fidelity of two unitary channels.

    Equals 1 exactly when the gates differ by a global phase. For batches,
    ``left`` has shape ``(a, 2, 2)`` and ``right`` has shape ``(b, 2, 2)``;
    the result has shape ``(a, b)``.
    """
    left = np.asarray(left, dtype=np.complex128)
    right = np.asarray(right, dtype=np.complex128)
    if left.ndim == 2 and right.ndim == 2:
        trace = np.trace(left.conj().T @ right)
        return np.abs(trace) ** 2 / 4
    if left.ndim == 2:
        left = left[None, ...]
    if right.ndim == 2:
        right = right[None, ...]
    overlap = np.einsum("aij,bij->ab", np.conjugate(left), right, optimize=True)
    fidelity = np.abs(overlap) ** 2 / 4
    if fidelity.shape == (1, 1):
        return fidelity[0, 0]
    return fidelity


def rotation_error(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Bloch-sphere rotation angle, in radians, taking one gate to the other.

    If ``F`` is the process fidelity, the angle is ``2 arccos(√F)``. It is the
    angle of the relative rotation ``left† right`` once the global phase has
    been removed, and it lies in ``[0, π]``.
    """
    fidelity = np.clip(np.real(process_fidelity(left, right)), 0.0, 1.0)
    return 2 * np.arccos(np.sqrt(fidelity))


def best_match(targets: np.ndarray, bank: np.ndarray, chunk: int = 20000) -> tuple[np.ndarray, np.ndarray]:
    """Highest process fidelity of each target against a bank of gates.

    Returns the fidelity of shape ``(n_targets,)`` and the bank index that
    attains it.
    """
    targets = np.asarray(targets, dtype=np.complex128)
    bank = np.asarray(bank, dtype=np.complex128)
    if targets.ndim == 2:
        targets = targets[None, ...]
    best_fidelity = np.zeros(len(targets), dtype=float)
    best_index = np.zeros(len(targets), dtype=np.int64)
    for start in range(0, len(bank), chunk):
        block = bank[start : start + chunk]
        overlap = np.einsum("aij,nij->an", np.conjugate(targets), block, optimize=True)
        fidelity = np.abs(overlap) ** 2 / 4
        local = np.argmax(fidelity, axis=1)
        values = fidelity[np.arange(len(targets)), local]
        improved = values > best_fidelity
        best_fidelity[improved] = values[improved]
        best_index[improved] = local[improved] + start
    return best_fidelity, best_index

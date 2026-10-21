"""Bloch vectors of pure single-qubit states."""

from __future__ import annotations

import numpy as np


def bloch_vector(state: np.ndarray) -> np.ndarray:
    """Expectation values ``(⟨X⟩, ⟨Y⟩, ⟨Z⟩)`` of one pure state."""
    state = np.asarray(state, dtype=np.complex128)
    state = state / np.linalg.norm(state)
    amplitude_0 = state[0]
    amplitude_1 = state[1]
    coherence = np.conjugate(amplitude_0) * amplitude_1
    return np.array(
        [
            2 * np.real(coherence),
            2 * np.imag(coherence),
            np.abs(amplitude_0) ** 2 - np.abs(amplitude_1) ** 2,
        ],
        dtype=float,
    )


def bloch_vectors(states: np.ndarray) -> np.ndarray:
    """Bloch vectors of a batch of pure states with shape ``(n, 2)``."""
    states = np.asarray(states, dtype=np.complex128)
    norms = np.linalg.norm(states, axis=1)
    states = states / norms[:, None]
    coherence = np.conjugate(states[:, 0]) * states[:, 1]
    return np.column_stack(
        [
            2 * np.real(coherence),
            2 * np.imag(coherence),
            np.abs(states[:, 0]) ** 2 - np.abs(states[:, 1]) ** 2,
        ]
    )


def angular_separation(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Angle, in radians, between Bloch vectors."""
    cosines = np.clip(np.sum(first * second, axis=-1), -1.0, 1.0)
    return np.arccos(cosines)

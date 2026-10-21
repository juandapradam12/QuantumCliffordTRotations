"""Matrix representations of single-qubit gates in the computational basis.

Angles are radians. A global phase is left as it falls out of the usual
definitions: T differs from R_z(π/4) by e^{iπ/8}, and the comparison between
gates is phase-invariant.
"""

from __future__ import annotations

import numpy as np

I = np.eye(2, dtype=np.complex128)
X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
S = np.array([[1, 0], [0, 1j]], dtype=np.complex128)
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=np.complex128)

ZERO = np.array([1, 0], dtype=np.complex128)
ONE = np.array([0, 1], dtype=np.complex128)
PLUS = np.array([1, 1], dtype=np.complex128) / np.sqrt(2)
MINUS = np.array([1, -1], dtype=np.complex128) / np.sqrt(2)
PLUS_I = np.array([1, 1j], dtype=np.complex128) / np.sqrt(2)
MINUS_I = np.array([1, -1j], dtype=np.complex128) / np.sqrt(2)

GATES = {"I": I, "H": H, "S": S, "T": T, "X": X, "Y": Y, "Z": Z}


def _angles(theta) -> np.ndarray:
    return np.asarray(theta, dtype=float)


def rx(theta) -> np.ndarray:
    """Rotation by ``theta`` about the Bloch x-axis."""
    theta = _angles(theta)
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    out = np.empty(np.shape(theta) + (2, 2), dtype=np.complex128)
    out[..., 0, 0] = c
    out[..., 0, 1] = -1j * s
    out[..., 1, 0] = -1j * s
    out[..., 1, 1] = c
    return out


def ry(theta) -> np.ndarray:
    """Rotation by ``theta`` about the Bloch y-axis."""
    theta = _angles(theta)
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    out = np.empty(np.shape(theta) + (2, 2), dtype=np.complex128)
    out[..., 0, 0] = c
    out[..., 0, 1] = -s
    out[..., 1, 0] = s
    out[..., 1, 1] = c
    return out


def rz(theta) -> np.ndarray:
    """Rotation by ``theta`` about the Bloch z-axis."""
    theta = _angles(theta)
    phase = np.exp(-1j * theta / 2)
    out = np.zeros(np.shape(theta) + (2, 2), dtype=np.complex128)
    out[..., 0, 0] = phase
    out[..., 1, 1] = np.conj(phase)
    return out


def u3(theta, phi, lam) -> np.ndarray:
    """Euler decomposition of a single-qubit unitary.

    .. math::

        U_3(\\theta, \\phi, \\lambda) =
        \\begin{pmatrix}
            \\cos(\\theta/2) & -e^{i\\lambda}\\sin(\\theta/2) \\\\
            e^{i\\phi}\\sin(\\theta/2) & e^{i(\\phi+\\lambda)}\\cos(\\theta/2)
        \\end{pmatrix}
    """
    theta = _angles(theta)
    phi = _angles(phi)
    lam = _angles(lam)
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    out = np.empty(np.broadcast_shapes(theta.shape, phi.shape, lam.shape) + (2, 2), dtype=np.complex128)
    out[..., 0, 0] = c
    out[..., 0, 1] = -np.exp(1j * lam) * s
    out[..., 1, 0] = np.exp(1j * phi) * s
    out[..., 1, 1] = np.exp(1j * (phi + lam)) * c
    return out


def product(word: tuple[str, ...]) -> np.ndarray:
    """Matrix of a word. The first name is applied to the state first."""
    unitary = I.copy()
    for name in word:
        unitary = GATES[name] @ unitary
    return unitary


def apply(unitary: np.ndarray, state: np.ndarray) -> np.ndarray:
    """Apply one gate, or a batch of gates, to a single state."""
    if unitary.ndim == 2:
        return unitary @ state
    return np.einsum("nij,j->ni", unitary, state)

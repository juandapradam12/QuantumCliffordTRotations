"""Phase-invariant identities, Bloch geometry, and the Clifford+T enumeration."""

import numpy as np

from clifford_t.bloch import bloch_vector
from clifford_t.group import build_group, enumerate_cliffords
from clifford_t.metrics import best_match, process_fidelity, rotation_error
from clifford_t.unitaries import (
    H,
    I,
    PLUS,
    PLUS_I,
    S,
    T,
    X,
    Y,
    Z,
    ZERO,
    product,
    rx,
    ry,
    rz,
    u3,
)


def test_named_gates_match_standard_matrices():
    assert np.allclose(H @ H, I)
    assert np.allclose(S @ S, Z)
    assert np.allclose(S @ S @ S @ S, I)
    assert np.allclose(T @ T, S)
    assert np.allclose(T, np.exp(1j * np.pi / 8) * rz(np.pi / 4))
    assert np.allclose(u3(np.pi / 2, 0, np.pi), H)
    assert np.allclose(u3(0, 0, np.pi / 2), S)
    assert np.allclose(u3(0, 0, np.pi / 4), T)
    assert np.allclose(u3(np.pi, 0, np.pi), X)


def test_hadamard_conjugates_z_rotations_to_x_rotations():
    angles = np.linspace(-2.3, 2.7, 9)
    for theta in angles:
        conjugated = H @ rz(theta) @ H
        assert process_fidelity(conjugated, rx(theta)) == np.float64(1) or np.isclose(
            process_fidelity(conjugated, rx(theta)), 1.0
        )
        assert np.allclose(conjugated, rx(theta))


def test_bloch_coordinates_of_the_pauli_eigenstates():
    assert np.allclose(bloch_vector(ZERO), [0, 0, 1])
    assert np.allclose(bloch_vector(PLUS), [1, 0, 0])
    assert np.allclose(bloch_vector(PLUS_I), [0, 1, 0])
    theta, phi = 0.7, -1.2
    state = np.array([np.cos(theta / 2), np.exp(1j * phi) * np.sin(theta / 2)])
    expected = [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
    assert np.allclose(bloch_vector(state), expected)
    assert np.isclose(np.linalg.norm(bloch_vector(state)), 1.0)


def test_lambda_is_a_right_multiplication_by_rz():
    theta, phi, lam = 0.4, -0.3, 0.8
    composed = np.exp(1j * lam / 2) * u3(theta, phi, 0) @ rz(lam)
    assert np.allclose(u3(theta, phi, lam), composed)


def test_third_euler_angle_drops_out_of_the_ground_state():
    prepared = u3(0.9, -0.4, np.array([0.2, 1.7, 2.8])) @ ZERO
    assert np.allclose(bloch_vector(prepared[0]), bloch_vector(prepared[1]))
    assert np.allclose(bloch_vector(prepared[0]), bloch_vector(prepared[2]))


def test_process_fidelity_ignores_global_phase():
    unitary = ry(0.8)
    phased = np.exp(-0.37j) * unitary
    assert np.isclose(process_fidelity(unitary, phased), 1.0)
    assert np.isclose(process_fidelity(unitary, -unitary), 1.0)
    assert process_fidelity(I, X) < 1e-12


def test_rotation_error_is_the_relative_angle():
    assert np.isclose(rotation_error(I, rx(0.4)), 0.4)
    assert np.isclose(rotation_error(I, rx(np.pi)), np.pi)
    assert np.isclose(rotation_error(rx(0.3), rx(0.5)), 0.2)
    assert np.isclose(rotation_error(I, rx(2 * np.pi)), 0.0)


def test_global_phase_does_not_split_a_gate():
    from clifford_t.group import _keys

    hadamard = H
    keys = _keys(np.stack([hadamard, -hadamard, 1j * hadamard, np.exp(0.4j) * hadamard, X]))
    assert keys[0] == keys[1] == keys[2] == keys[3]
    assert keys[0] != keys[4]


def test_clifford_group_has_order_24_and_permutes_the_paulis():
    matrices, words = enumerate_cliffords()
    assert len(matrices) == 24
    assert len(words) == 24
    for unitary, word in zip(matrices, words):
        assert np.allclose(product(word), unitary)
        assert np.allclose(unitary.conj().T @ unitary, I)
        for pauli in (X, Y, Z):
            conjugated = unitary @ pauli @ unitary.conj().T
            matches = [
                np.allclose(conjugated, sign * candidate)
                for sign in (1, -1)
                for candidate in (X, Y, Z)
            ]
            assert any(matches)


def test_minimal_t_count_of_the_quarter_rotations():
    bank = build_group(1)
    assert set(np.unique(bank.t_count)) == {0, 1}
    for matrix, word in zip(bank.matrices, bank.words):
        assert np.allclose(product(word), matrix, atol=1e-8)

    def best(target):
        fidelity = np.array([process_fidelity(target, gate) for gate in bank.matrices])
        index = int(np.argmax(fidelity))
        return fidelity[index], int(bank.t_count[index]), bank.words[index]

    fidelity, t_count, _ = best(rx(np.pi / 2))
    assert np.isclose(fidelity, 1.0)
    assert t_count == 0

    fidelity, t_count, word = best(rx(np.pi / 4))
    assert np.isclose(fidelity, 1.0)
    assert t_count == 1
    assert word.count("T") == 1

    coarse = bank.at_most(0)
    coarse_fidelity = max(process_fidelity(rx(np.pi / 4), gate) for gate in coarse.matrices)
    assert coarse_fidelity < 0.9


def _distance_to_ladder(theta: np.ndarray, step: float) -> np.ndarray:
    nearest = step * np.round(theta / step)
    delta = (theta - nearest + np.pi) % (2 * np.pi) - np.pi
    return np.abs(delta)


def test_population_doubles_with_each_new_t_gate():
    bank = build_group(4)
    fresh = [int(np.sum(bank.t_count == level)) for level in range(5)]
    assert fresh == [24, 72, 144, 288, 576]


def test_rx_follows_the_angle_ladder_through_t_count_3():
    bank = build_group(3)
    theta = np.linspace(0, 2 * np.pi, 180, endpoint=False)
    targets = rx(theta)
    ladders = {0: np.pi / 2, 1: np.pi / 4, 2: np.pi / 4, 3: np.pi / 4}
    for level, step in ladders.items():
        fidelity, _ = best_match(targets, bank.at_most(level).matrices)
        error = 2 * np.arccos(np.sqrt(np.clip(fidelity, 0.0, 1.0)))
        assert np.allclose(error, _distance_to_ladder(theta, step), atol=1e-6)
    eighth = rx(np.pi / 8)
    for level in (0, 1, 2, 3):
        fidelity, _ = best_match(eighth, bank.at_most(level).matrices)
        error = rotation_error(eighth, bank.at_most(level).matrices[int(_[0])])
        assert np.isclose(np.degrees(error), 22.5, atol=1e-6)

"""Enumerate Clifford+T and write the figures and the numeric summary."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np

from clifford_t.bloch import bloch_vector, bloch_vectors
from clifford_t.figures import (
    PROBE_PHI,
    PROBE_THETA,
    STATE_PHI,
    STATE_THETA,
    figure_bloch_state,
    figure_budget,
    figure_conjugation,
    figure_continuous_rotations,
    figure_coverage,
    figure_discrete_x_rotations,
    figure_eighth,
    figure_lambda_on_plus,
    figure_named_gates,
    figure_rotation_error,
    figure_u3_on_zero,
    use_style,
)
from clifford_t.group import build_group
from clifford_t.metrics import best_match
from clifford_t.unitaries import PLUS, ZERO, apply, rx

ROOT = Path(__file__).resolve().parents[1]
MAX_T_COUNT = 12
ANGLE_STEP_DEG = 0.25
HAAR_SAMPLES = 1024
HAAR_SEED = 0
CURVE_BUDGETS = (0, 1, 4, 7, 12)
ORBIT_BUDGETS = (0, 1, 2, 4)
EIGHTH_BUDGETS = (1, 12)


def haar_su2(count: int, seed: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    ginibre = generator.normal(size=(count, 2, 2)) + 1j * generator.normal(size=(count, 2, 2))
    unitary, triangular = np.linalg.qr(ginibre)
    diagonal = np.diagonal(triangular, axis1=-2, axis2=-1)
    unitary = unitary * np.conjugate(diagonal / np.abs(diagonal))[:, None, :]
    determinant = np.linalg.det(unitary)
    return unitary / np.sqrt(determinant)[:, None, None]


def _fidelity_to_degrees(fidelity: np.ndarray) -> np.ndarray:
    return np.degrees(2 * np.arccos(np.sqrt(np.clip(fidelity, 0.0, 1.0))))


def errors_by_budget(targets: np.ndarray, matrices: np.ndarray, t_count: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Best rotation error, in degrees, attainable at each T-count budget.

    The bank is ordered by increasing T-count. Each gate is scored once and
    then carried forward, so a later budget never loses an earlier gate.
    """
    if targets.ndim == 2:
        targets = targets[None, ...]
    n_levels = int(t_count.max()) + 1
    errors = np.zeros((n_levels, len(targets)))
    indices = np.zeros((n_levels, len(targets)), dtype=np.int64)
    carried = np.zeros(len(targets))
    carried_index = np.zeros(len(targets), dtype=np.int64)
    offset = 0
    for level in range(n_levels):
        shell = matrices[t_count == level]
        fidelity, local = best_match(targets, shell, chunk=4000)
        better = fidelity > carried + 1e-12
        carried = np.where(better, fidelity, carried)
        carried_index = np.where(better, local + offset, carried_index)
        errors[level] = _fidelity_to_degrees(carried)
        indices[level] = carried_index
        offset += len(shell)
    return errors, indices


def _orbit(matrices: np.ndarray) -> np.ndarray:
    return bloch_vectors(apply(matrices, PLUS))


def run() -> dict:
    use_style()
    figure_dir = ROOT / "figures"
    result_dir = ROOT / "results"
    figure_dir.mkdir(exist_ok=True)
    result_dir.mkdir(exist_ok=True)

    summary: dict = {
        "state": figure_bloch_state(figure_dir / "01_bloch_state.png"),
        "example_angles": {
            "theta": STATE_THETA,
            "phi": STATE_PHI,
            "probe_theta": PROBE_THETA,
            "probe_phi": PROBE_PHI,
        },
    }
    figure_continuous_rotations(figure_dir / "02_continuous_rotations.png")
    summary["u3_ground_state"] = figure_u3_on_zero(figure_dir / "03_u3_ground_state.png")
    summary["u3_plus"] = figure_lambda_on_plus(figure_dir / "04_u3_lambda.png")
    figure_named_gates(figure_dir / "05_named_gates.png")
    figure_conjugation(figure_dir / "06_conjugation.png")
    figure_discrete_x_rotations(figure_dir / "07_discrete_rx.png")

    bank = build_group(MAX_T_COUNT)
    theta = np.deg2rad(np.arange(0.0, 360.0, ANGLE_STEP_DEG))
    rx_errors, rx_index = errors_by_budget(rx(theta), bank.matrices, bank.t_count)
    eighth_errors, eighth_index = errors_by_budget(rx(np.pi / 8), bank.matrices, bank.t_count)
    haar_errors, _ = errors_by_budget(haar_su2(HAAR_SAMPLES, HAAR_SEED), bank.matrices, bank.t_count)

    plotted = {level: rx_errors[level] for level in CURVE_BUDGETS}
    figure_rotation_error(figure_dir / "09_rotation_error.png", theta, plotted)

    levels = list(range(MAX_T_COUNT + 1))
    worst = [float(np.max(rx_errors[level])) for level in levels]
    median = [float(np.median(rx_errors[level])) for level in levels]
    eighth = [float(eighth_errors[level, 0]) for level in levels]
    haar_median = [float(np.median(haar_errors[level])) for level in levels]
    figure_budget(figure_dir / "10_budget.png", levels, worst, median, eighth, haar_median)

    eighth_vectors = {
        level: bloch_vector(apply(bank.matrices[int(eighth_index[level, 0])], ZERO))
        for level in EIGHTH_BUDGETS
    }
    figure_eighth(
        figure_dir / "11_pi_over_8.png",
        eighth_vectors,
        {level: eighth[level] for level in EIGHTH_BUDGETS},
    )

    orbit_counts = {}
    coverage = []
    for level in levels:
        cloud = _orbit(bank.matrices[bank.t_count <= level])
        orbit_counts[str(level)] = int(len(np.unique(np.round(cloud, 6), axis=0)))
        if level in ORBIT_BUDGETS:
            coverage.append((level, cloud))
    figure_coverage(figure_dir / "08_coverage.png", coverage)

    per_level = {}
    for level in levels:
        slot = int(np.argmax(rx_errors[level]))
        word = bank.words[int(rx_index[level, slot])]
        eighth_word = bank.words[int(eighth_index[level, 0])]
        per_level[str(level)] = {
            "gates": int(np.sum(bank.t_count <= level)),
            "new_gates": int(np.sum(bank.t_count == level)),
            "plus_images": orbit_counts[str(level)],
            "max_error_deg": worst[level],
            "median_error_deg": median[level],
            "pi_over_8_error_deg": eighth[level],
            "haar_median_error_deg": haar_median[level],
            "haar_max_error_deg": float(np.max(haar_errors[level])),
            "worst_theta_over_pi": float(theta[slot] / np.pi),
            "worst_word": " ".join(word) if word else "I",
            "pi_over_8_word": " ".join(eighth_word) if eighth_word else "I",
            "pi_over_8_t_count": int(bank.t_count[int(eighth_index[level, 0])]),
        }

    summary.update(
        {
            "max_t_count": MAX_T_COUNT,
            "angle_step_deg": ANGLE_STEP_DEG,
            "haar_samples": HAAR_SAMPLES,
            "haar_seed": HAAR_SEED,
            "rx": per_level,
        }
    )
    (result_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    summary = run()
    brief = {
        level: {
            key: summary["rx"][level][key]
            for key in (
                "gates",
                "plus_images",
                "max_error_deg",
                "median_error_deg",
                "pi_over_8_error_deg",
                "haar_median_error_deg",
                "pi_over_8_t_count",
                "pi_over_8_word",
            )
        }
        for level in summary["rx"]
    }
    print(json.dumps({"u3_ground_state": summary["u3_ground_state"], "u3_plus": summary["u3_plus"], "rx": brief}, indent=2))


if __name__ == "__main__":
    main()

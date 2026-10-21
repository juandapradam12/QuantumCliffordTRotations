"""Figures for the Bloch geometry and the Clifford+T approximation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from clifford_t.bloch import angular_separation, bloch_vector, bloch_vectors
from clifford_t.unitaries import H, PLUS, S, T, Z, ZERO, apply, rx, rz, u3

RX = "#b4532a"
RY = "#2f6f4e"
RZ = "#1d4e89"
INK = "#1c1917"
MUTED = "#78716c"
SPHERE = "#e8eef3"
RING = "#c5d0da"
BUDGET_COLOR = {
    0: "#78716c",
    1: "#c4b5d4",
    2: "#8d6cab",
    4: "#6d4c93",
    7: "#5c3d86",
    8: "#4a3270",
    12: "#2a1845",
}

STATE_THETA = 2 * np.pi / 5
STATE_PHI = 2 * np.pi / 3
PROBE_THETA = 2 * np.pi / 3
PROBE_PHI = np.pi / 5


def use_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.edgecolor": "#d6d3d1",
        }
    )


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=170, bbox_inches="tight", pad_inches=0.18, facecolor="white")
    plt.close(fig)


def _dress(ax: Axes3D, guides: bool = True) -> None:
    phi = np.linspace(0, 2 * np.pi, 80)
    polar = np.linspace(0, np.pi, 40)
    mesh_phi, mesh_polar = np.meshgrid(phi, polar)
    x = np.cos(mesh_phi) * np.sin(mesh_polar)
    y = np.sin(mesh_phi) * np.sin(mesh_polar)
    z = np.cos(mesh_polar)
    ax.plot_surface(x, y, z, color=SPHERE, alpha=0.12, linewidth=0, antialiased=False, shade=False)
    if guides:
        ax.plot(np.cos(phi), np.sin(phi), np.zeros_like(phi), color=RING, lw=0.6)
        ax.plot(np.sin(phi), np.zeros_like(phi), np.cos(phi), color=RING, lw=0.4)
        ax.plot(np.zeros_like(phi), np.sin(phi), np.cos(phi), color=RING, lw=0.4)
    ax.plot([-1.25, 1.35], [0, 0], [0, 0], color="#a8a29e", lw=0.7)
    ax.plot([0, 0], [-1.25, 1.35], [0, 0], color="#a8a29e", lw=0.7)
    ax.plot([0, 0], [0, 0], [-1.25, 1.35], color="#a8a29e", lw=0.7)
    ax.text(1.48, 0, 0, r"$x$", fontsize=10)
    ax.text(0, 1.52, 0, r"$y$", fontsize=10)
    ax.text(0, 0, 1.48, r"$z$", fontsize=10)
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_zlim(-1.15, 1.15)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=22, azim=38)
    ax.set_proj_type("ortho")
    ax.set_axis_off()


def _vector(ax: Axes3D, vector: np.ndarray, color: str) -> None:
    tip = np.asarray(vector, dtype=float)
    ax.plot([0, tip[0]], [0, tip[1]], [0, tip[2]], color=color, lw=2.3, solid_capstyle="round")
    ax.scatter(*tip, color=color, s=32, depthshade=False)


def _label(ax: Axes3D, letter: str) -> None:
    ax.text2D(0.0, 0.98, letter, transform=ax.transAxes, fontsize=13, va="top", color=INK)


def figure_bloch_state(path: Path) -> dict:
    fig = plt.figure(figsize=(6.2, 5.6))
    ax = fig.add_subplot(111, projection="3d")
    _dress(ax)
    theta, phi = STATE_THETA, STATE_PHI
    radial = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
    azimuth = np.linspace(0, phi, 40)
    ax.plot(0.42 * np.cos(azimuth), 0.42 * np.sin(azimuth), np.zeros_like(azimuth), color=INK, lw=1.15)
    polar = np.linspace(0, theta, 50)
    ax.plot(
        0.62 * np.sin(polar) * np.cos(phi),
        0.62 * np.sin(polar) * np.sin(phi),
        0.62 * np.cos(polar),
        color=INK,
        lw=1.15,
    )
    ax.plot([radial[0], radial[0]], [radial[1], radial[1]], [0, radial[2]], color=MUTED, lw=0.7, ls=(0, (2, 2)))
    ax.plot([0, radial[0]], [0, radial[1]], [0, 0], color=MUTED, lw=0.7, ls=(0, (2, 2)))
    _vector(ax, radial, RX)
    ax.text(1.12 * radial[0], 1.12 * radial[1], 1.12 * radial[2], r"$|\psi\rangle$", color=RX, fontsize=12)
    ax.text(0.52 * np.cos(phi / 2), 0.52 * np.sin(phi / 2), 0.04, r"$\varphi$", fontsize=12)
    ax.text(
        0.74 * np.sin(theta * 0.45) * np.cos(phi),
        0.74 * np.sin(theta * 0.45) * np.sin(phi),
        0.74 * np.cos(theta * 0.45),
        r"$\theta$",
        fontsize=12,
    )
    ax.scatter(0, 0, 1, color=INK, s=14, depthshade=False)
    ax.scatter(0, 0, -1, color=INK, s=14, depthshade=False)
    ax.text(0.16, 0.0, 1.12, r"$|0\rangle$", fontsize=10, color=MUTED)
    ax.text(0.16, 0.0, -1.22, r"$|1\rangle$", fontsize=10, color=MUTED)
    _save(fig, path)
    return {"theta": theta, "phi": phi, "bloch": radial.tolist()}


def figure_continuous_rotations(path: Path) -> None:
    fig = plt.figure(figsize=(6.4, 5.6))
    ax = fig.add_subplot(111, projection="3d")
    _dress(ax, guides=False)
    angle = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.zeros_like(angle), -np.sin(angle), np.cos(angle), color=RX, lw=2.6)
    ax.plot(np.sin(angle), np.zeros_like(angle), np.cos(angle), color=RY, lw=2.6)
    ax.plot(np.cos(angle), np.sin(angle), np.zeros_like(angle), color=RZ, lw=2.6)
    ax.legend(
        handles=[
            Line2D([0], [0], color=RX, lw=2, label=r"$R_x$"),
            Line2D([0], [0], color=RY, lw=2, label=r"$R_y$"),
            Line2D([0], [0], color=RZ, lw=2, label=r"$R_z$"),
        ],
        loc="upper left",
        frameon=False,
        bbox_to_anchor=(0.0, 0.98),
    )
    _save(fig, path)


def figure_u3_on_zero(path: Path) -> dict:
    theta = np.linspace(0, np.pi, 28)
    phi = np.linspace(0, 2 * np.pi, 56, endpoint=False)
    mesh_theta, mesh_phi = np.meshgrid(theta, phi, indexing="ij")
    states = apply(u3(mesh_theta.ravel(), mesh_phi.ravel(), 0.0), ZERO)
    cloud = bloch_vectors(states)

    lam = np.linspace(0, 2 * np.pi, 120)
    probe = bloch_vectors(apply(u3(PROBE_THETA, PROBE_PHI, lam), ZERO))

    fig = plt.figure(figsize=(10.6, 5.2))
    ax = fig.add_subplot(121, projection="3d")
    _dress(ax)
    _label(ax, "a")
    points = ax.scatter(
        cloud[:, 0],
        cloud[:, 1],
        cloud[:, 2],
        c=mesh_phi.ravel(),
        cmap="cividis",
        s=8,
        depthshade=False,
        vmin=0,
        vmax=2 * np.pi,
    )
    colorbar = fig.colorbar(points, ax=ax, fraction=0.035, pad=0.02, shrink=0.72)
    colorbar.set_label(r"$\varphi$")
    colorbar.set_ticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    colorbar.set_ticklabels([r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
    ax.set_title(r"$U_3(\theta,\varphi,\lambda)\,|0\rangle$", pad=0)

    panel = fig.add_subplot(122)
    _label_2d(panel, "b")
    panel.plot(lam / np.pi, probe[:, 0], color=RX, lw=1.8, label=r"$\langle X\rangle$")
    panel.plot(lam / np.pi, probe[:, 1], color=RY, lw=1.8, label=r"$\langle Y\rangle$")
    panel.plot(lam / np.pi, probe[:, 2], color=RZ, lw=1.8, label=r"$\langle Z\rangle$")
    panel.set_xlim(0, 2)
    panel.set_ylim(-1.05, 1.05)
    panel.set_xlabel(r"$\lambda\,/\,\pi$")
    panel.set_ylabel("Bloch component")
    panel.set_title(rf"$\theta=2\pi/3,\ \varphi=\pi/5$, input $|0\rangle$")
    panel.legend(frameon=False, loc="lower right")
    panel.spines["top"].set_visible(False)
    panel.spines["right"].set_visible(False)
    fig.tight_layout()
    _save(fig, path)
    return {"max_component_drift": float(np.max(np.ptp(probe, axis=0)))}


def figure_lambda_on_plus(path: Path) -> dict:
    theta = np.linspace(0.15, np.pi - 0.15, 18)
    phi = np.linspace(0, 2 * np.pi, 36, endpoint=False)
    mesh_theta, mesh_phi = np.meshgrid(theta, phi, indexing="ij")
    flat_theta, flat_phi = mesh_theta.ravel(), mesh_phi.ravel()
    clouds = []
    for lam in (0.0, np.pi / 2):
        states = apply(u3(flat_theta, flat_phi, lam), PLUS)
        clouds.append(bloch_vectors(states))
    separation = angular_separation(clouds[0], clouds[1])

    fig = plt.figure(figsize=(10.4, 5.1))
    for index, (cloud, lam) in enumerate(zip(clouds, (0.0, np.pi / 2))):
        ax = fig.add_subplot(1, 2, index + 1, projection="3d")
        _dress(ax)
        _label(ax, "ab"[index])
        points = ax.scatter(
            cloud[:, 0],
            cloud[:, 1],
            cloud[:, 2],
            c=flat_phi,
            cmap="cividis",
            s=11,
            depthshade=False,
            vmin=0,
            vmax=2 * np.pi,
        )
        ax.set_title(rf"$\lambda={0 if lam == 0.0 else r'\pi/2'}$", pad=2)
        probe = bloch_vector(apply(u3(PROBE_THETA, PROBE_PHI, lam), PLUS))
        ax.scatter(*probe, color=RX, s=46, depthshade=False, edgecolors="white", linewidths=0.5)
    colorbar = fig.colorbar(points, ax=fig.axes, fraction=0.02, pad=0.04, shrink=0.7)
    colorbar.set_label(r"$\varphi$")
    colorbar.set_ticks([0, np.pi, 2 * np.pi])
    colorbar.set_ticklabels([r"$0$", r"$\pi$", r"$2\pi$"])
    _save(fig, path)
    return {
        "median_separation_deg": float(np.degrees(np.median(separation))),
        "max_separation_deg": float(np.degrees(np.max(separation))),
    }


def figure_named_gates(path: Path) -> None:
    images = {
        r"$T$": bloch_vector(apply(T, PLUS)),
        r"$S$": bloch_vector(apply(S, PLUS)),
        r"$Z$": bloch_vector(apply(Z, PLUS)),
        r"$H$": bloch_vector(apply(H, PLUS)),
    }
    colors = {r"$T$": "#a16207", r"$S$": RZ, r"$Z$": "#102a43", r"$H$": RY}

    fig = plt.figure(figsize=(10.6, 5.1))
    ax = fig.add_subplot(121, projection="3d")
    _dress(ax)
    _label(ax, "a")
    start = bloch_vector(PLUS)
    ax.scatter(*start, color=INK, s=28, depthshade=False)
    ax.text(1.02, 0.02, 0.16, r"$|+\rangle$", fontsize=11)
    for name, vector in images.items():
        _vector(ax, vector, colors[name])
        ax.text(*(1.14 * vector), name, color=colors[name], fontsize=11)
    ax.set_title(r"Images of $|+\rangle$", pad=0)

    panel = fig.add_subplot(122)
    _label_2d(panel, "b")
    circle = np.linspace(0, 2 * np.pi, 400)
    panel.plot(np.cos(circle), np.sin(circle), color=RING, lw=1.2)
    panel.axhline(0, color="#e7e5e4", lw=0.6)
    panel.axvline(0, color="#e7e5e4", lw=0.6)
    marks = [
        (0.0, r"$I$", INK),
        (np.pi / 4, r"$T$", "#a16207"),
        (np.pi / 2, r"$S$", RZ),
        (np.pi, r"$Z$", "#102a43"),
        (-np.pi / 4, r"$T^\dagger$", "#a16207"),
        (-np.pi / 2, r"$S^\dagger$", RZ),
    ]
    for angle, name, color in marks:
        panel.scatter([np.cos(angle)], [np.sin(angle)], color=color, s=36, zorder=3)
        panel.text(1.16 * np.cos(angle), 1.16 * np.sin(angle), name, color=color, fontsize=11, ha="center", va="center")
    panel.set_aspect("equal")
    panel.set_xlim(-1.45, 1.45)
    panel.set_ylim(-1.45, 1.45)
    panel.axis("off")
    panel.set_title(r"$R_z$ angles available from $T$", pad=8)
    fig.tight_layout()
    _save(fig, path)


def figure_conjugation(path: Path) -> None:
    theta = np.linspace(0, 2 * np.pi, 240, endpoint=False)
    equator = bloch_vectors(apply(rz(theta), PLUS))
    meridian = bloch_vectors(apply(H @ rz(theta) @ H, ZERO))

    fig = plt.figure(figsize=(10.4, 5.0))
    scatters = []
    for index, (cloud, title) in enumerate(
        (
            (equator, r"$R_z(\theta)\,|+\rangle$"),
            (meridian, r"$H R_z(\theta) H\,|0\rangle = R_x(\theta)\,|0\rangle$"),
        )
    ):
        ax = fig.add_subplot(1, 2, index + 1, projection="3d")
        _dress(ax)
        _label(ax, "ab"[index])
        ax.plot(cloud[:, 0], cloud[:, 1], cloud[:, 2], color="#d6d3d1", lw=1.0)
        scatters.append(
            ax.scatter(
                cloud[:, 0],
                cloud[:, 1],
                cloud[:, 2],
                c=theta,
                cmap="cividis",
                s=10,
                depthshade=False,
                vmin=0,
                vmax=2 * np.pi,
            )
        )
        marker = int(np.argmin(np.abs(theta - np.pi / 2)))
        ax.scatter(*cloud[marker], color=INK, s=40, depthshade=False, zorder=5)
        tip = cloud[marker]
        ax.text(tip[0] + 0.08, tip[1] + 0.14, tip[2] + 0.1, r"$\pi/2$", fontsize=10, color=INK)
        ax.set_title(title, pad=2)
    colorbar = fig.colorbar(scatters[1], ax=fig.axes, fraction=0.02, pad=0.04, shrink=0.7)
    colorbar.set_label(r"$\theta$")
    colorbar.set_ticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    colorbar.set_ticklabels([r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
    _save(fig, path)


def figure_discrete_x_rotations(path: Path) -> None:
    theta = np.linspace(0, 2 * np.pi, 400)
    curve = bloch_vectors(apply(rx(theta), ZERO))
    samples = np.arange(8) * np.pi / 4
    points = bloch_vectors(apply(rx(samples), ZERO))

    fig = plt.figure(figsize=(10.6, 5.0))
    ax = fig.add_subplot(121, projection="3d")
    _dress(ax)
    _label(ax, "a")
    ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], color="#e7cfc3", lw=2.0)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color=RX, s=36, depthshade=False, zorder=5)
    ax.set_title(r"$R_x(k\pi/4)\,|0\rangle$", pad=0)

    panel = fig.add_subplot(122)
    _label_2d(panel, "b")
    panel.axhline(0, color=RING, lw=1)
    panel.plot(theta / np.pi, np.zeros_like(theta), color="#e7cfc3", lw=6, solid_capstyle="butt")
    panel.scatter(samples / np.pi, np.zeros_like(samples), color=RX, s=42, zorder=3)
    panel.plot([0, 0], [0.04, 0.16], color=INK, lw=0.8)
    panel.plot([0.25, 0.25], [0.04, 0.16], color=INK, lw=0.8)
    panel.annotate(
        "",
        xy=(0.25, 0.16),
        xytext=(0.0, 0.16),
        arrowprops={"arrowstyle": "<->", "color": INK, "lw": 1.0},
    )
    panel.text(0.125, 0.26, r"$\pi/4$", ha="center", fontsize=12)
    panel.set_xlim(-0.05, 2.05)
    panel.set_ylim(-0.55, 0.7)
    panel.set_xlabel(r"$\theta\,/\,\pi$")
    panel.set_yticks([])
    panel.set_title(r"Exact samples of $R_x$", pad=8)
    panel.spines["top"].set_visible(False)
    panel.spines["right"].set_visible(False)
    panel.spines["left"].set_visible(False)
    fig.tight_layout()
    _save(fig, path)


def figure_coverage(path: Path, orbits: list[tuple[int, np.ndarray]]) -> dict[int, int]:
    fig = plt.figure(figsize=(12.2, 3.9))
    counts = {}
    for index, (t_count, cloud) in enumerate(orbits):
        ax = fig.add_subplot(1, len(orbits), index + 1, projection="3d")
        _dress(ax)
        unique = np.unique(np.round(cloud, 6), axis=0)
        counts[t_count] = int(len(unique))
        ax.scatter(
            unique[:, 0],
            unique[:, 1],
            unique[:, 2],
            color=BUDGET_COLOR[t_count],
            s=18 if len(unique) < 30 else 8,
            alpha=0.95 if len(unique) < 80 else 0.75,
            depthshade=False,
            edgecolors="none",
        )
        ax.set_title(rf"$T$-count $\leq {t_count}$" + "\n" + f"{len(unique)} images", pad=-4, fontsize=11)
    fig.subplots_adjust(wspace=-0.02)
    _save(fig, path)
    return counts


def figure_rotation_error(path: Path, theta: np.ndarray, errors_deg: dict[int, np.ndarray]) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    for t_count, series in errors_deg.items():
        ax.plot(
            theta / np.pi,
            series,
            color=BUDGET_COLOR[t_count],
            lw=1.7,
            label=rf"$\leq {t_count}$",
        )
    ax.axvline(0.125, color=RX, lw=0.7, ls=(0, (2, 2)))
    ax.set_xlim(0, 2)
    ax.set_ylim(-1, 50)
    ax.set_xlabel(r"$\theta\,/\,\pi$")
    ax.set_ylabel(r"rotation error $\delta$ (degrees)")
    ax.legend(
        frameon=False,
        ncol=5,
        title=r"$T$-count",
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save(fig, path)


def figure_budget(
    path: Path,
    levels: list[int],
    worst: list[float],
    median: list[float],
    eighth: list[float],
    haar_median: list[float],
) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.plot(levels, worst, color=RX, marker="o", ms=5, lw=1.7, label=r"worst $R_x(\theta)$")
    ax.plot(levels, eighth, color="#a16207", marker="o", ms=5, lw=1.7, label=r"$\theta=\pi/8$")
    ax.plot(levels, median, color="#57534e", marker="o", ms=5, lw=1.7, label=r"median $R_x(\theta)$")
    ax.plot(
        levels,
        haar_median,
        color=RZ,
        marker="o",
        ms=5,
        lw=1.4,
        ls=(0, (3, 1.5)),
        mfc="white",
        label="median, Haar sample",
    )
    ax.set_xlim(-0.3, levels[-1] + 0.3)
    ax.set_xticks(levels)
    ax.set_ylim(-1, 52)
    ax.set_xlabel(r"$T$-count budget")
    ax.set_ylabel(r"rotation error $\delta$ (degrees)")
    ax.legend(frameon=False, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save(fig, path)


def figure_eighth(path: Path, vectors: dict[int, np.ndarray], errors_deg: dict[int, float]) -> None:
    """Target state of ``R_x(π/8)`` on ``|+⟩``, and the best gates at two budgets."""
    target = bloch_vector(apply(rx(np.pi / 8), ZERO))
    fig = plt.figure(figsize=(10.4, 5.0))
    ax = fig.add_subplot(121, projection="3d")
    _dress(ax, guides=False)
    _label(ax, "a")
    _vector(ax, target, RX)
    ax.text(*(1.15 * target), "target", color=RX, fontsize=10)
    for t_count, vector in vectors.items():
        _vector(ax, vector, BUDGET_COLOR[t_count])
    handles = [Line2D([0], [0], color=RX, lw=1.8, label=r"$R_x(\pi/8)|0\rangle$")]
    for t_count in vectors:
        handles.append(Line2D([0], [0], color=BUDGET_COLOR[t_count], lw=1.8, label=rf"$T$-count $\leq {t_count}$"))
    ax.legend(handles=handles, frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.98))
    ax.set_title(r"Where $|0\rangle$ is sent", pad=0)

    panel = fig.add_subplot(122)
    _label_2d(panel, "b")
    window = np.linspace(-0.55, 0.55, 300)
    panel.plot(np.cos(window), np.sin(window), color=RING, lw=1.4)
    panel.scatter([1], [0], color=RX, s=46, zorder=4)
    panel.text(0.90, -0.02, "target", color=RX, fontsize=10, ha="right", va="top")
    for t_count, sign, radius in ((1, 1.0, 0.78), (12, -1.0, 0.90)):
        delta = sign * np.radians(errors_deg[t_count])
        panel.scatter([np.cos(delta)], [np.sin(delta)], color=BUDGET_COLOR[t_count], s=46, zorder=4)
        arc = np.linspace(0, delta, 80)
        panel.plot(radius * np.cos(arc), radius * np.sin(arc), color=BUDGET_COLOR[t_count], lw=1.6)
        panel.text(
            0.52,
            0.22 if delta > 0 else -0.22,
            rf"$\leq {t_count}$: {errors_deg[t_count]:.2f}$^\circ$",
            color=BUDGET_COLOR[t_count],
            fontsize=11,
            ha="center",
            va="center",
        )
    panel.set_aspect("equal")
    panel.set_xlim(0.35, 1.08)
    panel.set_ylim(-0.48, 0.52)
    panel.axis("off")
    panel.set_title(r"Gate error $\delta$", pad=8)
    fig.tight_layout()
    _save(fig, path)


def _label_2d(ax, letter: str) -> None:
    ax.text(-0.06, 1.04, letter, transform=ax.transAxes, fontsize=13, va="bottom", color=INK)

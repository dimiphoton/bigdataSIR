"""Courbes des diapositives, en grande police, avec au plus trois séries."""

from collections.abc import Mapping
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

from epidemic.config import load_game_config, load_reporting_config
from epidemic.inference.validation import load_course_series
from epidemic.model.priors import (
    BETA_MU,
    BETA_SIGMA,
    HOSP_MU_LOC,
    INCUBATION_MU,
    INCUBATION_SIGMA,
    SENSITIVITY_ALPHA,
    SENSITIVITY_BETA,
    SYMPTOM_ALPHA,
    SYMPTOM_BETA,
    SYMPTOMATIC_MU,
    SYMPTOMATIC_SIGMA,
)

CONFIRMED = "#1f4e79"
HOSPITAL = "#1b7f6e"
ICU = "#9b2226"
ACCENT = "#1f4e79"
TEXT = 20

_COST_NAMES = {
    "mask": "Mask",
    "case_isolation": "Case isolation",
    "social_distancing": "Distancing",
    "household_quarantine": "Household quarantine",
    "lockdown": "Lockdown",
}


def save_slide_figures(folder: Path, context: Mapping[str, object]) -> None:
    """Écrit les PNG référencés par les deux decks."""
    folder.mkdir(parents=True, exist_ok=True)
    course = load_course_series()
    day = np.arange(1, course.n_days + 1)
    _tested_and_confirmed(
        folder / "tested.png", day, course.num_tested, course.num_positive
    )
    _one_curve(
        folder / "confirmed.png", day, course.num_positive, "Confirmed", CONFIRMED
    )
    _beds(
        folder / "beds.png",
        day,
        course.num_hospitalised,
        course.num_critical,
        _as_int(context, "cap_hospital"),
        _as_int(context, "cap_icu"),
    )
    _two_bars(
        folder / "coverage.png",
        ["Announced", "Realised"],
        [_as_int(context, "high_pct"), _as_int(context, "course_coverage_pct")],
        [ACCENT, "#c45c26"],
        "Share inside the range (%)",
    )
    raw_levels = context["level_pcts"]
    if not isinstance(raw_levels, list):
        raise TypeError("level_pcts doit être une liste.")
    levels = [int(value) for value in raw_levels]
    _level_bars(
        folder / "levels.png",
        levels,
        _as_int(context, "high_pct"),
        _as_int(context, "worst_level"),
    )
    _costs(folder / "costs.png")
    _published(
        folder / "published.png",
        day,
        course.num_positive,
        course.num_hospitalised,
        course.num_critical,
    )
    _flow(folder / "flow.png")
    _compartments(folder / "compartments.png")
    _prior_table(folder / "priors.png")


def _as_int(context: Mapping[str, object], key: str) -> int:
    value = context[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{key} doit être un nombre.")
    return int(value)


def _tested_and_confirmed(
    path: Path, day: np.ndarray, tested: np.ndarray, confirmed: np.ndarray
) -> None:
    figure, axis = _axes()
    axis.plot(day, tested, color="#7f8c9b", lw=2.4)
    axis.plot(day, confirmed, color=CONFIRMED, lw=2.6)
    _end_label(axis, day, tested, "Tested", "#7f8c9b")
    _end_label(axis, day, confirmed, "Confirmed", CONFIRMED)
    axis.set_ylabel("People per day")
    _finish(figure, axis, path)


def _one_curve(
    path: Path, day: np.ndarray, values: np.ndarray, label: str, color: str
) -> None:
    figure, axis = _axes()
    axis.plot(day, values, color=color, lw=2.6)
    _end_label(axis, day, values, label, color)
    axis.set_ylabel("People per day")
    _finish(figure, axis, path)


def _beds(
    path: Path,
    day: np.ndarray,
    ward: np.ndarray,
    critical: np.ndarray,
    cap_ward: int,
    cap_icu: int,
) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(7.2, 5.0), sharex=True)
    _panel(axes[0], day, ward, "Ward", HOSPITAL, cap_ward)
    _panel(axes[1], day, critical, "ICU", ICU, cap_icu)
    axes[1].set_xlabel("Day", fontsize=TEXT)
    figure.tight_layout()
    _save(figure, path)


def _panel(
    axis: plt.Axes,
    day: np.ndarray,
    values: np.ndarray,
    label: str,
    color: str,
    cap: int,
) -> None:
    axis.plot(day, values, color=color, lw=2.6)
    _end_label(axis, day, values, label, color)
    peak = float(np.max(values)) if values.size else 0.0
    if peak > 0 and cap <= peak * 5:
        axis.axhline(cap, color=color, ls="--", lw=1.6)
        axis.text(day[0], cap, f"cap {cap}", color=color, fontsize=TEXT, va="bottom")
    axis.tick_params(labelsize=16)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    _room(axis, day)


def _two_bars(
    path: Path, labels: list[str], values: list[int], colors: list[str], ylabel: str
) -> None:
    figure, axis = _axes()
    axis.bar(labels, values, color=colors, width=0.62)
    for index, value in enumerate(values):
        axis.text(
            index,
            value,
            str(value),
            ha="center",
            va="bottom",
            fontsize=TEXT,
            color="#17202a",
        )
    axis.set_ylabel(ylabel)
    axis.set_ylim(0, max(values + [100]) * 1.18)
    _finish(figure, axis, path)


def _level_bars(path: Path, values: list[int], nominal: int, worst_level: int) -> None:
    figure, axis = _axes()
    labels = [str(level) for level in range(len(values))]
    colors = [
        "#9b2226" if level == worst_level else ACCENT for level in range(len(values))
    ]
    axis.bar(labels, values, color=colors, width=0.66)
    for index, value in enumerate(values):
        axis.text(index, value, str(value), ha="center", va="bottom", fontsize=TEXT)
    axis.axhline(nominal, color="#17202a", ls="--", lw=1.5)
    axis.text(0, nominal, f"  {nominal}%", fontsize=TEXT, va="bottom")
    axis.set_xlabel("Stochasticity level")
    axis.set_ylabel("Coverage of the 90% range (%)")
    axis.set_ylim(0, 110)
    _finish(figure, axis, path)


def _costs(path: Path) -> None:
    game = load_game_config()
    pairs = sorted(game.constraint_costs.items(), key=lambda item: item[1])
    labels = [_COST_NAMES.get(name, name) for name, _cost in pairs]
    values = [float(cost) for _name, cost in pairs]
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    axis.barh(labels, values, color=ACCENT, height=0.62)
    for index, value in enumerate(values):
        axis.text(
            value, index, f"  {value:g}", va="center", fontsize=TEXT, color=ACCENT
        )
    axis.set_xlabel("Constraint cost per day", fontsize=TEXT)
    axis.tick_params(labelsize=18)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    figure.tight_layout()
    _save(figure, path)


def _published(
    path: Path,
    day: np.ndarray,
    confirmed: np.ndarray,
    ward: np.ndarray,
    critical: np.ndarray,
) -> None:
    figure, axes = plt.subplots(3, 1, figsize=(7.2, 5.4), sharex=True)
    series = (
        (confirmed, "Confirmed", CONFIRMED),
        (ward, "Ward", HOSPITAL),
        (critical, "ICU", ICU),
    )
    for axis, (values, label, color) in zip(axes, series, strict=True):
        axis.plot(day, values, color=color, lw=2.4)
        _end_label(axis, day, values, label, color)
        axis.tick_params(labelsize=14)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        _room(axis, day)
    axes[-1].set_xlabel("Day", fontsize=TEXT)
    figure.tight_layout()
    _save(figure, path)


def _flow(path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 3.6))
    axis.set_xlim(0, 12)
    axis.set_ylim(0, 4)
    axis.axis("off")
    boxes = ((2, "Hidden\nstate"), (6, "Test"), (10, "Published\ncounts"))
    for x_pos, label in boxes:
        patch = FancyBboxPatch(
            (x_pos - 1.5, 1.2),
            3.0,
            1.6,
            boxstyle="round,pad=0.04,rounding_size=0.15",
            facecolor="#f3f7fb",
            edgecolor=ACCENT,
            linewidth=2,
        )
        axis.add_patch(patch)
        axis.text(
            x_pos, 2.0, label, ha="center", va="center", fontsize=TEXT, color="#17202a"
        )
    for start in (3.5, 7.5):
        axis.annotate(
            "",
            xy=(start + 1.0, 2.0),
            xytext=(start, 2.0),
            arrowprops={"arrowstyle": "->", "color": ACCENT, "lw": 2},
        )
    _save(figure, path)


def _compartments(path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8.2, 3.6))
    axis.set_xlim(0, 16)
    axis.set_ylim(0, 5)
    axis.axis("off")
    row = (("S", 1.1), ("E", 3.5), ("I", 5.9), ("H", 8.3), ("ICU", 11.0))
    for name, x_pos in row:
        _node(axis, x_pos, 2.5, name)
    xs = [x_pos for _name, x_pos in row]
    for left, right in zip(xs, xs[1:], strict=False):
        axis.annotate(
            "",
            xy=(right - 0.75, 2.5),
            xytext=(left + 0.75, 2.5),
            arrowprops={"arrowstyle": "->", "color": ACCENT, "lw": 1.8},
        )
    _node(axis, 14.2, 3.8, "R")
    _node(axis, 14.2, 1.2, "D")
    axis.annotate(
        "",
        xy=(13.45, 3.8),
        xytext=(11.7, 2.85),
        arrowprops={"arrowstyle": "->", "color": ACCENT, "lw": 1.8},
    )
    axis.annotate(
        "",
        xy=(13.45, 1.2),
        xytext=(11.7, 2.15),
        arrowprops={"arrowstyle": "->", "color": ACCENT, "lw": 1.8},
    )
    _save(figure, path)


def _node(axis: plt.Axes, x_pos: float, y_pos: float, name: str) -> None:
    patch = FancyBboxPatch(
        (x_pos - 0.7, y_pos - 0.42),
        1.4,
        0.84,
        boxstyle="round,pad=0.03,rounding_size=0.12",
        facecolor="#f3f7fb",
        edgecolor=ACCENT,
        linewidth=1.8,
    )
    axis.add_patch(patch)
    axis.text(
        x_pos, y_pos, name, ha="center", va="center", fontsize=18, color="#17202a"
    )


def _prior_table(path: Path) -> None:
    rows = (
        (
            "Incubation",
            "log-normal",
            f"μ = {INCUBATION_MU:.2f}, σ = {INCUBATION_SIGMA:.2f}",
        ),
        (
            "Symptoms",
            "log-normal",
            f"μ = {SYMPTOMATIC_MU:.2f}, σ = {SYMPTOMATIC_SIGMA:.2f}",
        ),
        (
            "Ill share",
            "beta",
            f"α = {SYMPTOM_ALPHA:.1f}, β = {SYMPTOM_BETA:.1f}",
        ),
        (
            "Sensitivity",
            "beta",
            f"α = {SENSITIVITY_ALPHA:.1f}, β = {SENSITIVITY_BETA:.1f}",
        ),
        ("Transmission", "log-normal", f"μ = {BETA_MU:.2f}, σ = {BETA_SIGMA:.2f}"),
        ("Hosp. risk", "logit-normal", f"mean {HOSP_MU_LOC:.2f}"),
    )
    figure, axis = plt.subplots(figsize=(9.0, 4.6))
    axis.axis("off")
    header = ("Parameter", "Law", "Setting")
    columns = (0.02, 0.40, 0.64)
    for x_pos, label in zip(columns, header, strict=True):
        axis.text(
            x_pos,
            0.94,
            label,
            transform=axis.transAxes,
            fontsize=18,
            color=ACCENT,
            fontweight="bold",
        )
    for index, row in enumerate(rows):
        y_pos = 0.78 - index * 0.12
        for x_pos, cell in zip(columns, row, strict=True):
            axis.text(
                x_pos,
                y_pos,
                cell,
                transform=axis.transAxes,
                fontsize=18,
                color="#17202a",
            )
    _save(figure, path)


def _axes() -> tuple[plt.Figure, plt.Axes]:
    return plt.subplots(figsize=(7.2, 4.6))


def _finish(figure: plt.Figure, axis: plt.Axes, path: Path) -> None:
    axis.tick_params(labelsize=16)
    axis.xaxis.label.set_fontsize(TEXT)
    axis.yaxis.label.set_fontsize(TEXT)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    figure.tight_layout()
    _save(figure, path)


def _save(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dpi = load_reporting_config().figure_dpi
    figure.savefig(path, dpi=dpi, facecolor="white", bbox_inches="tight")
    plt.close(figure)


def _end_label(
    axis: plt.Axes, day: np.ndarray, values: np.ndarray, label: str, color: str
) -> None:
    axis.annotate(
        label,
        xy=(day[-1], values[-1]),
        xytext=(8, 0),
        textcoords="offset points",
        color=color,
        fontsize=TEXT,
        va="center",
    )
    _room(axis, day)


def _room(axis: plt.Axes, day: np.ndarray) -> None:
    span = float(day[-1] - day[0]) or 1.0
    axis.set_xlim(day[0], day[-1] + span * 0.28)

"""Graphes de suivi. Le code couleur ne change pas d'une figure à l'autre."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from epidemic.forecast.bands import Fan  # noqa: E402

CONFIRMED = "#1f4e79"
INFECTIONS = "#c45c26"
HOSPITAL = "#1b7f6e"
ICU = "#9b2226"
RT_COLOR = "#6c3483"
# Bande claire, puis bande plus étroite.
BAND = 0.18


def log_slope(series: np.ndarray, window: int = 14) -> float:
    """Pente de log(moyenne glissante), sur les derniers jours."""
    values = np.asarray(series, dtype=float)
    if values.size < 3:
        return 0.0
    if values.size >= 7:
        kernel = np.ones(7) / 7.0
        smooth = np.convolve(values, kernel, mode="valid")
    else:
        smooth = values
    log_values = np.log(np.clip(smooth, 1e-6, None))
    axis = np.arange(log_values.size, dtype=float)
    if log_values.size > window:
        axis = axis[-window:]
        log_values = log_values[-window:]
    slope = float(np.polyfit(axis, log_values, 1)[0])
    return slope


def doubling_sentence(slope: float) -> str:
    """Temps de doublement si la pente est positive, de division par deux sinon."""
    if abs(slope) < 1e-4:
        return "croissance quasi nulle"
    days = float(np.log(2.0) / slope)
    if slope > 0:
        return f"temps de doublement {days:.1f} jours"
    return f"temps de division par deux {abs(days):.1f} jours"


def growth_sentence(samples: np.ndarray) -> str:
    """Temps de doublement médian, avec l'intervalle des pentes à 90 %."""
    rows = np.asarray(samples, dtype=float)
    if rows.ndim != 2 or rows.shape[0] < 1 or rows.shape[1] < 3:
        return "série trop courte pour un temps de doublement"
    slopes = np.array([log_slope(row) for row in rows])
    low, median, high = np.quantile(slopes, [0.05, 0.5, 0.95])
    center = doubling_sentence(float(median))
    if low * high <= 0 or abs(low) < 1e-4 or abs(high) < 1e-4:
        return center
    left = abs(float(np.log(2.0) / low))
    right = abs(float(np.log(2.0) / high))
    lo, hi = sorted((left, right))
    return f"{center} (intervalle {lo:.1f} à {hi:.1f} jours)"


def plot_incidence(
    observed: np.ndarray,
    observed_day: np.ndarray,
    infections: Fan,
    *,
    bulletin_day: int | None = None,
    spans: list[tuple[int, int]] | None = None,
    note: str | None = None,
) -> plt.Figure:
    """Cas confirmés et infections estimées, en échelle semi-log."""
    figure, axis = plt.subplots(figsize=(8, 4.2))
    _shade(axis, spans)
    _mark_bulletin(axis, bulletin_day)
    rolling = _rolling_mean(observed, 7)
    axis.plot(
        observed_day,
        np.clip(observed, 0.5, None),
        color=CONFIRMED,
        label="Cas confirmés",
    )
    axis.plot(
        observed_day[-rolling.size :],
        rolling.clip(0.5),
        color=CONFIRMED,
        ls="--",
        label="Moyenne 7 j",
    )
    _fan(axis, infections, INFECTIONS, "Infections estimées")
    axis.set_yscale("log")
    axis.set_title(note or doubling_sentence(log_slope(infections.median)))
    axis.set_ylabel("Nouveaux cas par jour")
    axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    return figure


def plot_attack_rate(
    beta: Fan,
    rt: Fan,
    *,
    bulletin_day: int | None = None,
    spans: list[tuple[int, int]] | None = None,
) -> plt.Figure:
    """β effectif, et Rt sur un second axe."""
    figure, axis = plt.subplots(figsize=(8, 4.2))
    _shade(axis, spans)
    _mark_bulletin(axis, bulletin_day)
    _fan(axis, beta, INFECTIONS, "β effectif")
    axis.set_yscale("log")
    axis.set_ylabel("Nouvelles infections / (S I / N)")
    twin = axis.twinx()
    twin.plot(rt.day, rt.median, color=RT_COLOR, label="Rt")
    twin.axhline(1.0, color=RT_COLOR, lw=0.6, alpha=0.5)
    twin.set_ylabel("Rt")
    axis.set_title("Taux d'attaque par personne exposée")
    figure.tight_layout()
    return figure


def plot_loglog(cumulative: np.ndarray, weekly: np.ndarray) -> plt.Figure:
    """Nouveaux cas de la semaine en fonction des cas cumulés, sans le temps."""
    figure, axis = plt.subplots(figsize=(5.2, 5))
    axis.plot(
        np.clip(cumulative, 1.0, None), np.clip(weekly, 1.0, None), color=CONFIRMED
    )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Cas cumulés")
    axis.set_ylabel("Cas des 7 derniers jours")
    axis.set_title("Trajectoire log-log")
    figure.tight_layout()
    return figure


def plot_rt(
    rt: Fan,
    weekly: np.ndarray | None = None,
    *,
    bulletin_day: int | None = None,
    spans: list[tuple[int, int]] | None = None,
) -> plt.Figure:
    """Rt a posteriori, et P(Rt < 1) calculée sur les tirages de chaque semaine."""
    figure, axis = plt.subplots(figsize=(8, 4.2))
    _shade(axis, spans)
    _mark_bulletin(axis, bulletin_day)
    _fan(axis, rt, RT_COLOR, "Rt")
    axis.axhline(1.0, color="black", lw=0.8)
    axis.set_ylabel("Rt")
    if weekly is not None and weekly.size:
        twin = axis.twinx()
        twin.plot(
            rt.day[6::7][: weekly.size],
            weekly,
            "o",
            color="black",
            ms=4,
            label="P(Rt < 1)",
        )
        twin.set_ylim(0, 1)
        twin.set_ylabel("P(Rt < 1)")
    axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    return figure


def plot_capacity(
    observed_h: np.ndarray,
    observed_c: np.ndarray,
    observed_day: np.ndarray,
    hospital: Fan,
    critical: Fan,
    *,
    max_hospitalized: float,
    max_critical: float,
    saturation: np.ndarray | None = None,
    bulletin_day: int | None = None,
    spans: list[tuple[int, int]] | None = None,
) -> plt.Figure:
    """Lits observés, projection, et probabilité cumulée de saturation."""
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(8, 6.2),
        sharex=True,
        gridspec_kw={"height_ratios": [2.4, 1]},
    )
    _shade(axes[0], spans)
    _mark_bulletin(axes[0], bulletin_day)
    axes[0].plot(
        observed_day, observed_h, color=HOSPITAL, label="Hospitalisés observés"
    )
    axes[0].plot(observed_day, observed_c, color=ICU, label="Soins intensifs observés")
    _fan(axes[0], hospital, HOSPITAL, "Hospitalisés projetés")
    _fan(axes[0], critical, ICU, "Soins intensifs projetés")
    axes[0].axhline(max_hospitalized, color=HOSPITAL, ls=":", lw=0.8)
    axes[0].axhline(max_critical, color=ICU, ls=":", lw=0.8)
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("Lits")
    if saturation is not None:
        axes[1].plot(hospital.day, saturation, color="black")
    axes[1].set_ylabel("P(saturation avant t)")
    axes[1].set_ylim(0, 1)
    figure.tight_layout()
    return figure


def plot_underreporting(
    ratio: Fan,
    positivity: np.ndarray,
    positivity_day: np.ndarray,
    *,
    bulletin_day: int | None = None,
    spans: list[tuple[int, int]] | None = None,
) -> plt.Figure:
    """Infections estimées / cas confirmés, et taux de positivité observé."""
    figure, axis = plt.subplots(figsize=(8, 4.2))
    _shade(axis, spans)
    _mark_bulletin(axis, bulletin_day)
    _fan(axis, ratio, INFECTIONS, "Sous-déclaration")
    axis.set_ylabel("Infections / cas confirmés")
    twin = axis.twinx()
    twin.plot(positivity_day, positivity, color=CONFIRMED, label="Positivité")
    twin.set_ylabel("num_positive / num_tested")
    axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    return figure


def plot_pareto(
    labels: list[str],
    costs: np.ndarray,
    risks: np.ndarray,
    on_front: np.ndarray,
    kinds: list[str],
) -> plt.Figure:
    """Coût cumulé contre P(saturation). La frontière est en trait plein."""
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    for kind, marker in (("fixe", "o"), ("conditionnelle", "s")):
        mask = np.array([item == kind for item in kinds])
        if not np.any(mask):
            continue
        axis.scatter(costs[mask], risks[mask], marker=marker, label=kind)
    front = np.asarray(on_front, dtype=bool)
    if np.any(front):
        order = np.argsort(costs[front])
        axis.plot(
            costs[front][order], risks[front][order], color="black", label="Pareto"
        )
    for label, cost, risk, kept in zip(labels, costs, risks, front, strict=False):
        if kept:
            axis.annotate(label, (cost, risk), fontsize=7)
    axis.set_xlabel("Coût cumulé des contraintes")
    axis.set_ylabel("P(saturation)")
    axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    return figure


def plot_scenario_fans(curves: list[tuple[str, Fan]], *, ylabel: str) -> plt.Figure:
    """Médiane de chaque scénario sur le même axe."""
    figure, axis = plt.subplots(figsize=(8, 4.2))
    for label, fan in curves:
        axis.plot(fan.day, fan.median, label=label)
    axis.set_ylabel(ylabel)
    axis.legend(frameon=False, fontsize=7)
    figure.tight_layout()
    return figure


def plot_prior_posterior(
    prior: np.ndarray, posterior: np.ndarray, name: str
) -> plt.Figure:
    """Densités a priori et a posteriori d'un paramètre."""
    figure, axis = plt.subplots(figsize=(5.4, 3.4))
    axis.hist(
        np.asarray(prior).ravel(), bins=30, density=True, alpha=0.45, label="a priori"
    )
    axis.hist(
        np.asarray(posterior).ravel(),
        bins=30,
        density=True,
        alpha=0.45,
        label="a posteriori",
    )
    axis.set_title(name)
    axis.legend(frameon=False)
    figure.tight_layout()
    return figure


def save_figure(figure: plt.Figure, path: Path) -> None:
    """Écrit une figure et ferme la fenêtre."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def _fan(axis: plt.Axes, fan: Fan, color: str, label: str) -> None:
    axis.fill_between(
        fan.day, fan.low90, fan.high90, color=color, alpha=BAND, linewidth=0
    )
    axis.fill_between(
        fan.day, fan.low50, fan.high50, color=color, alpha=BAND, linewidth=0
    )
    axis.plot(fan.day, fan.median, color=color, label=label)


def _shade(axis: plt.Axes, spans: list[tuple[int, int]] | None) -> None:
    if not spans:
        return
    for start, end in spans:
        axis.axvspan(start, end, color="#f4e4c4", zorder=0)


def _mark_bulletin(axis: plt.Axes, day: int | None) -> None:
    if day is not None:
        axis.axvline(day, color="black", lw=0.8)


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if values.size < window:
        return np.asarray(values, dtype=float)
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")

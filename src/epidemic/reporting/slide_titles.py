"""Titres des deux decks. Les chiffres viennent du contexte, pas du texte."""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

_BANNED_WORDS = {
    "leverage",
    "robust",
    "cutting-edge",
    "seamless",
    "delve",
    "unlock",
    "harness",
    "powerful",
    "comprehensive",
    "state-of-the-art",
    "journey",
    "game-changer",
}
_BANNED_PHRASES = (
    "in this presentation",
    "let's dive in",
    "key takeaways",
)
_EXPECTED = {
    "recruiter_titles.txt": 7,
    "technical_titles.txt": 12,
}


@dataclass(frozen=True)
class CoverageExtent:
    """Jusqu'où les couvertures restent proches du niveau nominal."""

    holds: bool
    partial: bool
    max_ok_level: int
    worst_level: int
    worst_coverage: float


def city_phrase(population: int) -> str:
    """Écrit la population pour un titre, en anglais."""
    if population <= 0:
        raise ValueError("La population doit être strictement positive.")
    if population % 1_000_000 == 0:
        return f"{population // 1_000_000} million"
    return f"{population:,}"


def coverage_extent(
    levels: list[tuple[int, float]],
    nominal: float,
    tolerance: float,
) -> CoverageExtent:
    """Niveau le plus haut encore calé, et le plus mauvais écart."""
    if not levels:
        raise ValueError("Aucune couverture par niveau.")
    if tolerance <= 0:
        raise ValueError("La tolérance est strictement positive.")
    ordered = sorted(levels, key=lambda item: item[0])
    max_ok: int | None = None
    for level, coverage in ordered:
        if abs(coverage - nominal) > tolerance:
            break
        max_ok = level
    worst_level, worst_coverage = min(ordered, key=lambda item: (item[1], -item[0]))
    holds = max_ok is not None and max_ok == ordered[-1][0]
    return CoverageExtent(
        holds=holds,
        partial=max_ok is not None and not holds,
        max_ok_level=0 if max_ok is None else max_ok,
        worst_level=worst_level,
        worst_coverage=worst_coverage,
    )


def title_groups(context: Mapping[str, object]) -> dict[str, list[str]]:
    """Rend les titres et refuse un titre trop court ou un mot interdit."""
    from epidemic.reporting.render import render_named

    groups: dict[str, list[str]] = {}
    for name, expected in _EXPECTED.items():
        text = render_named(f"slides/{name}.j2", context)
        groups[name] = _checked_titles(text, expected, name)
    return groups


def emit_titles(docs_dir: Path, context: Mapping[str, object]) -> list[Path]:
    """Écrit les deux fichiers de titres, une ligne par diapositive."""
    slides = docs_dir / "slides"
    slides.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, titles in title_groups(context).items():
        path = slides / name
        path.write_text("\n".join(titles) + "\n", encoding="utf-8")
        written.append(path)
    return written


def check_slide_text(markdown: str, limit: int, name: str) -> None:
    """Compte les mots hors titre, image et notes. Lève si le plafond saute."""
    text = re.sub(r"<!--.*?-->", " ", markdown, flags=re.DOTALL)
    if text.startswith("---"):
        end = text.find("\n---", 3)
        text = text[end + 4 :]
    for index, slide in enumerate(re.split(r"\n---\n", text), start=1):
        kept: list[str] = []
        skipped_title = False
        for line in slide.splitlines():
            if not skipped_title and line.startswith("# "):
                skipped_title = True
                continue
            if line.startswith("!["):
                continue
            kept.append(line)
        body = re.sub(r"<[^>]+>", " ", " ".join(kept))
        count = len(body.split())
        if count > limit:
            raise ValueError(
                f"{name}, diapositive {index} : {count} mots, plafond {limit}."
            )


def render_slide_titles(destination: Path | None = None) -> list[Path]:
    """Recalcule le contexte et écrit les titres, sans exporter de figures."""
    from epidemic.config import load_reporting_config
    from epidemic.reporting.context import build_context, output_base

    base = output_base(destination)
    docs_dir = base / load_reporting_config().docs_dir
    docs_dir.mkdir(parents=True, exist_ok=True)
    return emit_titles(docs_dir, build_context())


def _checked_titles(text: str, expected: int, name: str) -> list[str]:
    titles = [line.strip() for line in text.splitlines() if line.strip()]
    if len(titles) != expected:
        raise ValueError(f"{name} contient {len(titles)} titres, pas {expected}.")
    for title in titles:
        words = title.split()
        if len(words) < 5:
            raise ValueError(f"Titre de moins de 5 mots : {title}")
        tokens = {word.strip(".,:;").lower() for word in words}
        banned = tokens & _BANNED_WORDS
        if banned:
            raise ValueError(f"Mot interdit dans « {title} » : {sorted(banned)}")
        folded = title.lower()
        for phrase in _BANNED_PHRASES:
            if phrase in folded:
                raise ValueError(f"Formule interdite dans « {title} ».")
    return titles

"""Mise en forme française des nombres produits par le rendu."""

from datetime import date

_MONTHS = (
    "janvier",
    "février",
    "mars",
    "avril",
    "mai",
    "juin",
    "juillet",
    "août",
    "septembre",
    "octobre",
    "novembre",
    "décembre",
)


def format_fr(value: float, digits: int = 0) -> str:
    """Écrit un nombre avec une espace entre les milliers et une virgule décimale."""
    text = f"{value:,.{digits}f}"
    return text.replace(",", " ").replace(".", ",")


def format_percent(value: float, digits: int = 0) -> str:
    """Écrit une part entre 0 et 1 comme un pourcentage."""
    return format_fr(100.0 * value, digits) + " %"


def french_date(value: date) -> str:
    """Écrit une date sans dépendre de la locale du système."""
    return f"{value.day} {_MONTHS[value.month - 1]} {value.year}"


def format_auto(value: float) -> str:
    """Choisit les décimales selon l'ordre de grandeur."""
    magnitude = abs(value)
    if magnitude >= 100:
        return format_fr(value, 0)
    if magnitude >= 10:
        return format_fr(value, 1)
    return format_fr(value, 2)

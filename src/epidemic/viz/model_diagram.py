"""Graphes de compartiments, générés depuis les flux du code.

Le simulateur a un stade présymptomatique. Le modèle d'inférence n'en a pas :
la liste des nœuds a la même longueur que ``N_COMPARTMENTS``.
"""

from epidemic.model.dynamics import N_COMPARTMENTS

# (source, cible, étiquette, style). Le style "dashed" est la partie observation.
INFERENCE_EDGES: tuple[tuple[str, str, str, str], ...] = (
    ("S", "E", "β", "solid"),
    ("E", "Ia", "σ (1 − p_sym)", "solid"),
    ("E", "Is", "σ p_sym", "solid"),
    ("Ia", "R", "γ", "solid"),
    ("Is", "H", "γ p_hosp", "solid"),
    ("Is", "R", "γ (1 − p_hosp)", "solid"),
    ("H", "ICU", "γ_h p_icu", "solid"),
    ("H", "R", "guérison", "solid"),
    ("H", "D", "décès service", "solid"),
    ("ICU", "R", "guérison", "solid"),
    ("ICU", "D", "décès SI", "solid"),
    ("Is", "Test", "p_test", "dashed"),
    ("Test", "Positif", "sensibilité", "dashed"),
    ("Test", "FauxNegatif", "1 − sensibilité", "dashed"),
)

SIMULATOR_EDGES: tuple[tuple[str, str, str, str], ...] = (
    ("S", "E", "β", "solid"),
    ("E", "Ia", "1 − p_sym", "solid"),
    ("E", "Ip", "p_sym", "solid"),
    ("Ip", "Is", "présymptomatique", "solid"),
    ("Ia", "R", "γ", "solid"),
    ("Is", "H", "p_hosp", "solid"),
    ("Is", "R", "1 − p_hosp", "solid"),
    ("H", "ICU", "p_icu", "solid"),
    ("H", "R", "guérison", "solid"),
    ("H", "D", "décès service", "solid"),
    ("ICU", "R", "guérison", "solid"),
    ("ICU", "D", "décès SI", "solid"),
    ("Is", "Test", "p_test", "dashed"),
    ("Test", "Positif", "sensibilité", "dashed"),
    ("Test", "FauxNegatif", "1 − sensibilité", "dashed"),
)

INFERENCE_NODES = ("S", "E", "Ia", "Is", "H", "ICU", "R", "D")


def inference_node_count() -> int:
    """Le schéma d'inférence a un nœud par compartiment du scan."""
    return len(INFERENCE_NODES)


def compartment_mermaid(kind: str) -> str:
    """Diagramme Mermaid. ``kind`` vaut ``simulator`` ou ``inference``."""
    edges = _edges(kind)
    lines = ["flowchart LR"]
    for source, target, label, style in edges:
        arrow = "-.->" if style == "dashed" else "-->"
        lines.append(f"    {source} {arrow}|{label}| {target}")
    return "\n".join(lines)


def compartment_dot(kind: str) -> str:
    """Même graphe au format Graphviz, sans exiger le binaire ``dot``."""
    edges = _edges(kind)
    lines = ["digraph compartments {", "  rankdir=LR;"]
    for source, target, label, style in edges:
        extra = ', style="dashed"' if style == "dashed" else ""
        lines.append(f'  {source} -> {target} [label="{label}"{extra}];')
    lines.append("}")
    return "\n".join(lines)


def _edges(kind: str) -> tuple[tuple[str, str, str, str], ...]:
    if kind == "inference":
        if len(INFERENCE_NODES) != N_COMPARTMENTS:
            raise ValueError(
                "Le schéma d'inférence n'a pas le bon nombre de compartiments."
            )
        return INFERENCE_EDGES
    if kind == "simulator":
        return SIMULATOR_EDGES
    raise ValueError("Le schéma vaut simulator ou inference.")

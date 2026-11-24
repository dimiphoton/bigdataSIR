"""Graines séparées, pour que chaque source d'aléa soit reproductible."""

from typing import Any

import numpy as np

STREAM_NAMES = ("parameters", "demographic", "environmental", "individual")


def make_streams(seed: int) -> dict[str, np.random.Generator]:
    """Quatre générateurs indépendants, toujours dans le même ordre."""
    sequence = np.random.SeedSequence(seed)
    children = sequence.spawn(len(STREAM_NAMES))
    return {
        name: np.random.default_rng(child)
        for name, child in zip(STREAM_NAMES, children, strict=True)
    }


def _plain(value: Any) -> Any:
    """Convertit les entiers numpy pour que l'état tienne dans un JSON."""
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def dump_rng(rng: np.random.Generator) -> dict[str, Any]:
    """Sérialise l'état du générateur."""
    return _plain(rng.bit_generator.state)


def load_rng(payload: dict[str, Any]) -> np.random.Generator:
    """Recharge un générateur PCG64."""
    bit_generator = np.random.PCG64()
    bit_generator.state = payload  # type: ignore[assignment]
    return np.random.Generator(bit_generator)

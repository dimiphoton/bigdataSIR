"""Mode Jouer."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jouer import render  # noqa: E402

render()

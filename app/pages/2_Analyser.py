"""Mode Analyser : séries publiées, score naïf, ajustement SVI."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analyser import render  # noqa: E402

render()

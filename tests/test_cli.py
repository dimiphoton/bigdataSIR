"""Commandes simulate et play."""

from pathlib import Path

from typer.testing import CliRunner

from epidemic.cli import app
from epidemic.simulator.dynamics import PUBLISHED_COLUMNS

RUNNER = CliRunner()


def test_simulate_writes_public_and_secret_files(tmp_path: Path) -> None:
    result = RUNNER.invoke(
        app, ["simulate", "--days", "2", "--seed", "1", "--output-dir", str(tmp_path)]
    )
    assert result.exit_code == 0
    observations = (tmp_path / "observations.csv").read_text(encoding="utf-8")
    assert observations.splitlines()[0] == ",".join(PUBLISHED_COLUMNS)
    assert (tmp_path / "truth_daily.csv").is_file()
    assert (tmp_path / ".truth.json").is_file()
    assert "Paramètres secrets" in result.stdout


def test_play_can_advance_two_bulletins_without_measures(tmp_path: Path) -> None:
    result = RUNNER.invoke(
        app,
        [
            "play",
            "--end-day",
            "14",
            "--seed",
            "1",
            "--level",
            "0",
            "--output-dir",
            str(tmp_path),
        ],
        input="fin\nfin\n",
    )
    assert result.exit_code == 0, result.stdout
    assert "Bulletin du jour 7" in result.stdout
    assert "Bulletin du jour 14" in result.stdout
    assert "Fin de partie" in result.stdout

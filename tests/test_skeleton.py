"""Vérifie que le squelette charge les configs et trouve les données."""

from datetime import date
from pathlib import Path

from typer.testing import CliRunner

from epidemic.cli import app
from epidemic.config import (
    load_priors_config,
    load_scenarios_config,
    load_simulator_config,
    load_thresholds_config,
    project_root,
)

RUNNER = CliRunner()


def test_configs_match_the_brief() -> None:
    simulator = load_simulator_config()
    priors = load_priors_config()
    scenarios = load_scenarios_config()
    thresholds = load_thresholds_config()

    assert simulator.population == 1_000_000
    assert simulator.stochasticity_level == 1
    assert simulator.day1_date == date(2020, 10, 2)
    assert priors.symptomatic_fraction.low == 0.50
    assert priors.test_sensitivity.high == 0.85
    assert priors.min_fraction_symptomatic_tested == 0.50
    assert scenarios.compliance == 0.9
    assert scenarios.mask_multiplies is True
    assert ("case_isolation", "household_quarantine") in scenarios.incompatible_pairs
    assert thresholds.max_hospitalized == 1500
    assert thresholds.max_critical == 300
    assert thresholds.horizon_end == date(2021, 1, 31)


def test_external_course_data_is_present() -> None:
    external = project_root() / "data" / "external"
    assert (external / "data.csv").is_file()
    assert (external / "SOURCE.md").is_file()
    stats = external / "stats"
    for name in (
        "age.json",
        "households.json",
        "schools.json",
        "workplaces.json",
        "communities.json",
    ):
        assert (stats / name).is_file()


def test_report_command_is_available() -> None:
    result = RUNNER.invoke(app, ["report", "--help"])
    assert result.exit_code == 0
    assert "PDF" in result.stdout or "pdf" in result.stdout


def test_fit_help_is_available() -> None:
    result = RUNNER.invoke(app, ["fit", "--help"])
    assert result.exit_code == 0
    assert "svi" in result.stdout


def test_generated_data_dir_exists() -> None:
    generated = project_root() / "data" / "generated"
    assert generated.is_dir()
    assert Path(generated / ".gitkeep").is_file()

"""Calendrier, paliers et décisions."""

from pathlib import Path

import pytest

from epidemic.config import load_game_config
from epidemic.simulator.game import Decision, Game, apply_decision, implementation_day


def _reach(game: Game, day: int) -> None:
    while game.day < day:
        if game.next_bulletin_day() is not None and game.next_bulletin_day() <= day:
            game.next_bulletin()
            if game.day < day and game.day in game.bulletin_day_list():
                game.decide(_keep(game))
        else:
            game.advance(day - game.day)


def _keep(game: Game) -> list[Decision]:
    """Répète les mesures ouvertes pour ne pas les lever trop tôt."""
    latest: dict[str, Decision] = {}
    for decisions in game.decision_log.values():
        for decision in decisions:
            latest[decision.name] = decision
    return list(latest.values())


def test_calendar_matches_the_brief() -> None:
    config = load_game_config()
    assert config.end_day == 122
    assert config.bulletin_period_days == 7
    assert implementation_day(21, config) == 24
    assert implementation_day(70, config) == 75
    assert config.tiers[1].from_day == 21
    assert config.tiers[2].from_day == 42
    assert config.tiers[3].from_day == 63
    assert config.reveals[0].day == 4
    assert config.reveals[1].day == 25
    assert config.reveals[2].day == 60


def test_tier_zero_blocks_every_measure() -> None:
    game = Game.start(seed=1, level=0, write_secret=False)
    bulletin = game.next_bulletin()
    assert bulletin.day == 7
    assert bulletin.tier == 0
    assert bulletin.allowed == []
    with pytest.raises(ValueError):
        game.decide([Decision("mask")])
    game.submit_risk(0.9, 0.9)
    assert game.tier_level() == 0


def test_mask_starts_three_days_after_the_decision() -> None:
    game = Game.start(seed=1, level=0, write_secret=False)
    _reach(game, 21)
    game.decide([Decision("mask")])
    assert "mask" not in game.active_names(23)
    assert "mask" in game.active_names(24)


def test_a_measure_cannot_be_lifted_before_fourteen_days() -> None:
    game = Game.start(seed=1, level=0, write_secret=False)
    _reach(game, 21)
    game.decide([Decision("mask")])
    _reach(game, 28)
    with pytest.raises(ValueError):
        game.decide([])
    _reach(game, 42)
    game.decide([])
    assert "mask" in game.active_names(44)
    assert "mask" not in game.active_names(45)


def test_incompatible_measures_and_early_unlock() -> None:
    game = Game.start(seed=1, level=0, write_secret=False)
    _reach(game, 21)
    game.submit_risk(0.25, None)
    assert "household_quarantine" in game.allowed_interventions()
    assert "lockdown" not in game.allowed_interventions()
    with pytest.raises(ValueError):
        game.decide(
            [
                Decision("case_isolation", duration_days=10),
                Decision("household_quarantine", duration_days=10),
            ]
        )
    game.submit_risk(0.25, 0.3)
    assert "lockdown" in game.allowed_interventions()


def test_final_decision_is_applied_on_day_75() -> None:
    policies = apply_decision(
        [],
        [Decision("mask")],
        decision_day=70,
        effect_day=implementation_day(70, load_game_config()),
        min_policy_days=14,
    )
    assert policies[0].effect_start == 75
    assert policies[0].min_end == 88


def test_save_and_load_continue_the_same_epidemic(tmp_path: Path) -> None:
    game = Game.start(seed=2, level=1, write_secret=False)
    game.next_bulletin()
    game.decide([])
    game.advance(3)
    path = tmp_path / "game.json"
    game.save(path)
    restored = Game.load(path)
    game.advance(4)
    restored.advance(4)
    assert game.observations().to_dict(
        orient="list"
    ) == restored.observations().to_dict(orient="list")


def test_score_counts_constraint_days() -> None:
    game = Game.start(seed=1, level=0, write_secret=False)
    _reach(game, 21)
    game.decide([Decision("mask")])
    game.advance(4)
    result = game.score()
    # Jours 24 et 25 : le masque a commencé, coût 1 par jour.
    assert result.constraint_cost == pytest.approx(2.0)

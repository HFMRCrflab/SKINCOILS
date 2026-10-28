"""Sweeps and search over protocol settings."""
from __future__ import annotations

import numpy as np
import pytest

from skinsim import Acquisition, Protocol, Vessel, model, optimise, tissues


@pytest.fixture
def draws():
    return tissues.sample(["blood", "fat"], draws=24, seed=99)


@pytest.fixture
def base():
    return Protocol(tr_ms=30.0, te_ms=5.0, flip_deg=25.0, slab_mm=10.0)


@pytest.fixture
def acquisition():
    return Acquisition(in_plane_mm=0.3, partition_mm=0.3, phase_fov_mm=40.0, acceleration=2.0)


# --- sweeps ---------------------------------------------------------------

def test_a_sweep_fills_a_grid_of_the_right_shape(base, draws):
    score = optimise.contrast_score(draws["blood"], draws["fat"], Vessel(8.0))
    result = optimise.sweep(base, {"flip_deg": np.linspace(5, 40, 7),
                                   "tr_ms": np.linspace(15, 60, 5)}, score)
    assert result.values.shape == (7, 5)
    assert result.names == ("flip_deg", "tr_ms")
    assert np.all(np.isfinite(result.values))


def test_a_sweep_reports_its_best_point_and_can_rebuild_that_protocol(base, draws):
    score = optimise.contrast_score(draws["blood"], draws["fat"], Vessel(8.0))
    angles = np.linspace(5, 40, 8)
    result = optimise.sweep(base, {"flip_deg": angles}, score)
    settings, value = result.best()
    assert settings["flip_deg"] in set(angles)
    assert value == pytest.approx(np.nanmax(result.values))
    assert result.best_protocol().flip_deg == pytest.approx(settings["flip_deg"])
    assert "best" in result.summary()


def test_a_sweep_marks_impossible_corners_rather_than_failing(base, draws):
    """A TR below the echo time cannot exist; the sweep should not care."""
    score = optimise.contrast_score(draws["blood"], draws["fat"], Vessel(8.0))
    result = optimise.sweep(base, {"tr_ms": np.array([2.0, 4.0, 20.0, 50.0])}, score)
    assert np.isnan(result.values[0]) and np.isnan(result.values[1])
    assert np.all(np.isfinite(result.values[2:]))


def test_a_sweep_needs_an_axis(base, draws):
    score = optimise.contrast_score(draws["blood"], draws["fat"], Vessel(8.0))
    with pytest.raises(ValueError, match="at least one axis"):
        optimise.sweep(base, {}, score)


# --- scores ---------------------------------------------------------------

def test_a_score_is_any_callable_and_finds_a_known_answer(base):
    """Maximising a stationary tissue's own signal must land on the Ernst angle."""
    tissue = tissues.LIBRARY["dermis"].nominal()
    ernst = np.degrees(np.arccos(np.exp(-base.tr_ms / tissue.t1_ms)))

    def score(protocol):
        return float(model.stationary_signal(tissue, protocol))

    found = optimise.optimise(base, {"flip_deg": (1.0, 60.0)}, score, seed=1, iterations=40)
    assert found.protocol.flip_deg == pytest.approx(ernst, abs=0.3)


def test_the_statistic_chooses_which_end_of_the_spread_is_optimised(base, draws):
    vessel = Vessel(8.0)
    typical = optimise.contrast_score(draws["blood"], draws["fat"], vessel)
    cautious = optimise.contrast_score(draws["blood"], draws["fat"], vessel,
                                       statistic=lambda v: np.percentile(v, 5))
    assert cautious(base) < typical(base)


def test_efficiency_and_contrast_disagree_about_repetition_time(base, draws, acquisition):
    """The point of the efficiency metric: contrast alone ignores the clock.

    Suppressed fat, so the contrast is positive and maximising it means what it
    sounds like. Contrast then rises all the way to the end of the range while
    efficiency turns over partway, which is the whole reason it exists.
    """
    vessel = Vessel(8.0)
    suppressed = base.with_values(fat_sat=True)
    axis = {"tr_ms": np.linspace(12, 300, 25)}
    by_contrast = optimise.sweep(
        suppressed, axis, optimise.contrast_score(draws["blood"], draws["fat"], vessel))
    by_efficiency = optimise.sweep(
        suppressed, axis,
        optimise.efficiency_score(draws["blood"], draws["fat"], vessel, acquisition))
    assert by_contrast.best()[0]["tr_ms"] == pytest.approx(300.0)
    assert by_efficiency.best()[0]["tr_ms"] < 200.0


def test_a_negative_contrast_traps_a_signed_search(base, draws, acquisition):
    """Without fat suppression blood is the darker tissue, and the sign bites.

    Maximising a negative efficiency chases it towards zero, which is the least
    visible vessel. Asking for the magnitude instead gives the right answer.
    """
    vessel = Vessel(8.0)
    axis = {"tr_ms": np.linspace(12, 300, 25)}
    signed = optimise.sweep(
        base, axis, optimise.efficiency_score(draws["blood"], draws["fat"], vessel,
                                              acquisition))
    magnitude = optimise.sweep(
        base, axis, optimise.efficiency_score(draws["blood"], draws["fat"], vessel,
                                              acquisition, absolute=True))
    assert signed.best()[1] < 0                      # the best signed value is still negative
    assert magnitude.best()[1] > 0
    assert magnitude.best()[0]["tr_ms"] < signed.best()[0]["tr_ms"]


# --- search ---------------------------------------------------------------

def test_search_improves_on_the_starting_protocol(base, draws, acquisition):
    score = optimise.efficiency_score(draws["blood"], draws["fat"], Vessel(8.0), acquisition)
    found = optimise.optimise(base, {"flip_deg": (3.0, 50.0), "tr_ms": (12.0, 80.0)},
                              score, seed=2, iterations=25)
    assert found.score >= score(base)
    assert found.evaluations > 0
    assert "score" in found.summary()


def test_search_over_switches_tries_every_combination(base, draws, acquisition):
    score = optimise.efficiency_score(draws["blood"], draws["fat"], Vessel(8.0), acquisition)
    found = optimise.optimise(base, {"flip_deg": (3.0, 50.0)}, score,
                              switches=["fat_sat"], seed=3, iterations=15)
    assert "fat_sat" in found.settings
    assert found.protocol.fat_sat == found.settings["fat_sat"]


def test_suppressing_fat_wins_when_fat_is_the_background(base, draws, acquisition):
    score = optimise.efficiency_score(draws["blood"], draws["fat"], Vessel(8.0), acquisition)
    found = optimise.optimise(base, {"flip_deg": (3.0, 50.0), "tr_ms": (12.0, 80.0)},
                              score, switches=["fat_sat"], seed=4, iterations=25)
    assert found.protocol.fat_sat is True


def test_a_constraint_keeps_the_search_inside_a_time_budget(base, draws, acquisition):
    score = optimise.efficiency_score(draws["blood"], draws["fat"], Vessel(8.0), acquisition)
    budget = 120.0
    found = optimise.optimise(
        base, {"tr_ms": (12.0, 80.0), "slab_mm": (4.0, 30.0)}, score,
        constraint=lambda p: acquisition.scan_time_s(p) <= budget, seed=5, iterations=25)
    assert acquisition.scan_time_s(found.protocol) <= budget


def test_an_impossible_constraint_says_so(base, draws, acquisition):
    score = optimise.efficiency_score(draws["blood"], draws["fat"], Vessel(8.0), acquisition)
    with pytest.raises(ValueError, match="no valid protocol"):
        optimise.optimise(base, {"tr_ms": (12.0, 80.0)}, score,
                          constraint=lambda p: False, seed=6, iterations=5)


def test_search_needs_something_to_search(base, draws):
    score = optimise.contrast_score(draws["blood"], draws["fat"], Vessel(8.0))
    with pytest.raises(ValueError, match="at least one bounded"):
        optimise.optimise(base, {}, score)


def test_search_is_deterministic_for_a_seed(base, draws, acquisition):
    score = optimise.efficiency_score(draws["blood"], draws["fat"], Vessel(8.0), acquisition)
    kwargs = dict(bounds={"flip_deg": (3.0, 50.0)}, score=score, seed=11, iterations=15)
    first = optimise.optimise(base, **kwargs)
    again = optimise.optimise(base, **kwargs)
    assert first.score == pytest.approx(again.score)
    assert first.protocol.flip_deg == pytest.approx(again.protocol.flip_deg)

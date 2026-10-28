"""The engine: affine maps on longitudinal magnetisation."""
from __future__ import annotations

import numpy as np
import pytest

from skinsim.magnetisation import (Step, excitation, identity, readout, relaxation,
                                   saturation)


def test_a_step_applies_its_gain_and_offset():
    step = Step(0.5, 0.25)
    assert step(1.0) == pytest.approx(0.75)
    assert np.allclose(step(np.array([0.0, 2.0])), [0.25, 1.25])


def test_composition_is_the_same_as_applying_in_turn():
    first, second = Step(0.6, 0.2), Step(0.3, 0.5)
    combined = first.then(second)
    for m in (-1.0, 0.0, 0.37, 2.5):
        assert combined(m) == pytest.approx(second(first(m)))


def test_composition_is_associative():
    a, b, c = Step(0.6, 0.2), Step(0.3, 0.5), Step(0.9, -0.1)
    left = a.then(b).then(c)
    right = a.then(b.then(c))
    assert left.gain == pytest.approx(right.gain)
    assert left.offset == pytest.approx(right.offset)


def test_identity_changes_nothing():
    step = Step(0.4, 0.3)
    assert step.then(identity()).gain == pytest.approx(step.gain)
    assert identity().then(step).offset == pytest.approx(step.offset)


def test_repeating_matches_applying_one_at_a_time():
    step = Step(0.7, 0.2)
    m = 1.0
    for _ in range(9):
        m = step(m)
    assert step.repeated(9)(1.0) == pytest.approx(m)


def test_repeating_a_step_that_only_shifts_adds_up():
    """Gain of exactly one is where the closed form would divide by zero."""
    step = Step(1.0, 0.3)
    assert step.repeated(5)(0.0) == pytest.approx(1.5)
    assert np.isfinite(step.repeated(5).offset).all()


def test_repeating_forever_lands_on_the_fixed_point():
    step = Step(0.8, 0.3)
    assert step.repeated(np.inf)(1.0) == pytest.approx(step.settled)
    assert step.repeated(2000)(1.0) == pytest.approx(step.settled)


def test_the_fixed_point_is_left_unchanged():
    step = Step(0.65, 0.42)
    assert step(step.settled) == pytest.approx(step.settled)


def test_a_step_that_changes_nothing_has_no_steady_state():
    with pytest.raises(ValueError, match="no steady state"):
        _ = Step(1.0, 0.1).settled


def test_relaxation_recovers_towards_equilibrium():
    step = relaxation(duration_ms=1000.0, t1_ms=1000.0, equilibrium=1.0)
    assert step(0.0) == pytest.approx(1.0 - np.exp(-1.0))
    assert step(1.0) == pytest.approx(1.0)                 # already there, stays there
    assert step.settled == pytest.approx(1.0)


def test_relaxation_over_no_time_does_nothing():
    step = relaxation(0.0, 800.0, 1.0)
    assert step(0.3) == pytest.approx(0.3)


def test_excitation_keeps_the_cosine_of_what_it_tips():
    assert excitation(0.0)(1.0) == pytest.approx(1.0)
    assert excitation(90.0)(1.0) == pytest.approx(0.0, abs=1e-12)
    assert excitation(60.0)(1.0) == pytest.approx(0.5)


def test_saturation_scales_and_a_perfect_pulse_empties():
    assert saturation(0.1)(1.0) == pytest.approx(0.1)
    assert saturation(0.0)(5.0) == pytest.approx(0.0)


def test_readout_tips_and_decays():
    value = readout(1.0, flip_deg=90.0, te_ms=10.0, t2star_ms=10.0)
    assert value == pytest.approx(np.exp(-1.0))
    assert readout(1.0, 0.0, 5.0, 50.0) == pytest.approx(0.0, abs=1e-12)


def test_everything_broadcasts_over_an_ensemble():
    t1 = np.array([500.0, 1000.0, 1500.0])
    step = relaxation(30.0, t1, equilibrium=np.array([0.8, 0.9, 1.0]))
    assert step(np.zeros(3)).shape == (3,)
    assert step.repeated(40)(np.zeros(3)).shape == (3,)
    assert step.settled.shape == (3,)

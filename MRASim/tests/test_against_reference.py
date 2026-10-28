"""The package checked against independent implementations in ``reference.py``.

Those are literal translations of the textbook expressions and of a spin marched
one excitation at a time. They import nothing from ``skinsim`` and ``skinsim``
imports nothing from them, which is what makes this a real check rather than
the fast path agreeing with itself.
"""
from __future__ import annotations

import numpy as np
import pytest

import reference
from skinsim import Protocol, Vessel, magnetisation, model, tissues


@pytest.fixture
def tissue():
    return tissues.TissueValues(name="test", t1_ms=1100.0, t2star_ms=38.0,
                                spin_density=0.72, fat_sat_surviving=0.18,
                                mtc_surviving=0.55)


def test_stationary_signal_matches_the_textbook_expression(tissue):
    protocol = Protocol(tr_ms=27.0, te_ms=4.6, flip_deg=22.0, slab_mm=10.0)
    expected = reference.spoiled_gre_signal(protocol.tr_ms, protocol.te_ms, tissue.t1_ms,
                                            tissue.t2star_ms, protocol.flip_deg,
                                            tissue.spin_density)
    assert float(model.stationary_signal(tissue, protocol)) == pytest.approx(expected)


@pytest.mark.parametrize("tr_ms", [12.0, 30.0, 75.0])
def test_the_optimum_flip_matches_the_ernst_angle(tissue, tr_ms):
    protocol = Protocol(tr_ms=tr_ms, te_ms=4.0, flip_deg=10.0, slab_mm=10.0)
    expected = reference.ernst_angle_deg(tr_ms, tissue.t1_ms)
    angles = np.linspace(0.5, 80.0, 4000)
    signal = [float(model.stationary_signal(tissue, protocol.with_values(flip_deg=a)))
              for a in angles]
    assert angles[int(np.argmax(signal))] == pytest.approx(expected, abs=0.05)


def test_a_prepared_steady_state_matches_running_the_cycle_to_convergence(tissue):
    """The fixed point should equal what brute force converges to."""
    protocol = Protocol(tr_ms=40.0, te_ms=5.0, flip_deg=20.0, slab_mm=10.0,
                        fat_sat=True, prep_delay_ms=8.0)
    settled = reference.saturated_steady_state(
        protocol.tr_ms, tissue.t1_ms, protocol.flip_deg, tissue.spin_density,
        residual=tissue.fat_sat_surviving, delay_ms=protocol.prep_delay_ms)
    expected = (settled * np.sin(np.radians(protocol.flip_deg))
                * np.exp(-protocol.te_ms / tissue.t2star_ms))
    assert float(model.stationary_signal(tissue, protocol)) == pytest.approx(expected, rel=1e-9)


def test_the_closed_form_power_matches_a_spin_marched_pulse_by_pulse(tissue):
    """The whole speed argument of the package rests on this agreeing."""
    protocol = Protocol(tr_ms=25.0, te_ms=5.0, flip_deg=24.0, slab_mm=10.0)
    marched = reference.marched_blood_signal(protocol.tr_ms, protocol.te_ms, tissue.t1_ms,
                                             tissue.t2star_ms, protocol.flip_deg,
                                             tissue.spin_density, n_excitations=25)
    cycle = model.repetition(tissue, protocol, protocol.flip_deg)
    for pulses, expected in enumerate(marched):
        settled = cycle.repeated(pulses)(tissue.spin_density)
        got = magnetisation.readout(settled, protocol.flip_deg, protocol.te_ms,
                                    tissue.t2star_ms)
        assert float(got) == pytest.approx(expected, rel=1e-10), f"at pulse {pulses}"


def test_the_blood_profile_matches_the_marched_spin_at_matching_positions(tissue):
    """Chosen so one step along the path is exactly one excitation."""
    protocol = Protocol(tr_ms=25.0, te_ms=5.0, flip_deg=24.0, slab_mm=10.0)
    vessel = Vessel(velocity_mm_s=20.0, diameter_mm=0.2)       # dwell 500 ms = 20 TR
    steps = 20
    positions, signal = model.blood_profile(tissue, protocol, vessel, shells=1,
                                            samples=steps + 1)
    marched = reference.marched_blood_signal(protocol.tr_ms, protocol.te_ms, tissue.t1_ms,
                                             tissue.t2star_ms, protocol.flip_deg,
                                             tissue.spin_density, n_excitations=steps + 1)
    assert np.allclose(np.ravel(signal), marched, rtol=1e-10)


def test_fractional_repetitions_compose_into_whole_ones(tissue):
    """Walking the path in pieces must give the same answer as walking it in one."""
    protocol = Protocol(tr_ms=30.0, te_ms=5.0, flip_deg=18.0, slab_mm=10.0)
    cycle = model.repetition(tissue, protocol, protocol.flip_deg)
    start = tissue.spin_density
    in_one = cycle.repeated(12.0)(start)
    in_pieces = start
    for _ in range(8):
        in_pieces = cycle.repeated(1.5)(in_pieces)
    assert float(in_pieces) == pytest.approx(float(in_one), rel=1e-12)


def test_splitting_the_path_more_finely_changes_no_value_on_it(tissue):
    """With a uniform flip the walk is exact, so sample count cannot move a value.

    The path *average* does move slightly, because averaging more points of a
    curve is a better quadrature - but the signal at any given position along
    the path must come out identical however finely the walk was split.
    """
    protocol = Protocol(tr_ms=30.0, te_ms=5.0, flip_deg=18.0, slab_mm=10.0)
    vessel = Vessel(velocity_mm_s=7.0)
    _, coarse = model.blood_profile(tissue, protocol, vessel, shells=4, samples=9)
    _, fine = model.blood_profile(tissue, protocol, vessel, shells=4, samples=257)
    assert float(coarse[0]) == pytest.approx(float(fine[0]), rel=1e-12)    # entry
    assert float(coarse[-1]) == pytest.approx(float(fine[-1]), rel=1e-12)  # exit
    assert float(coarse[4]) == pytest.approx(float(fine[128]), rel=1e-12)  # halfway


def test_the_path_average_converges_as_the_walk_is_refined(tissue):
    protocol = Protocol(tr_ms=30.0, te_ms=5.0, flip_deg=18.0, slab_mm=10.0)
    vessel = Vessel(velocity_mm_s=7.0)
    values = [float(model.blood_signal(tissue, protocol, vessel, shells=4, samples=n))
              for n in (17, 65, 257, 1025)]
    gaps = [abs(b - a) for a, b in zip(values, values[1:])]
    assert all(b < a for a, b in zip(gaps, gaps[1:]))
    assert gaps[-1] < 1e-4

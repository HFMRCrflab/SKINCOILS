"""Vessel geometry, dwell time and the laminar profile."""
from __future__ import annotations

import numpy as np
import pytest

from skinsim.geometry import (DEFAULT_MAX_PATH_MM, SKIN_LAYERS, Vessel,
                              laminar_shells)


def test_a_vessel_crossing_head_on_travels_the_slab_thickness():
    assert Vessel(velocity_mm_s=5.0).path_mm(slab_mm=10.0) == pytest.approx(10.0)


def test_an_angled_vessel_travels_further_by_one_over_the_cosine():
    vessel = Vessel(velocity_mm_s=5.0, angle_deg=60.0)
    assert vessel.path_mm(slab_mm=10.0) == pytest.approx(20.0)


def test_the_path_is_capped_so_a_near_parallel_vessel_stays_finite():
    vessel = Vessel(velocity_mm_s=5.0, angle_deg=89.9)
    assert vessel.path_mm(slab_mm=10.0) == pytest.approx(DEFAULT_MAX_PATH_MM)
    assert vessel.path_mm(slab_mm=10.0, max_path_mm=25.0) == pytest.approx(25.0)


def test_dwell_time_is_path_over_speed():
    vessel = Vessel(velocity_mm_s=10.0)
    assert vessel.dwell_ms(slab_mm=10.0) == pytest.approx(1000.0)
    assert vessel.dwell_ms(slab_mm=10.0, speed_mm_s=5.0) == pytest.approx(2000.0)


def test_blood_that_does_not_move_never_leaves():
    assert Vessel(velocity_mm_s=0.0).dwell_ms(slab_mm=10.0) == np.inf


def test_an_angle_at_or_beyond_the_slab_plane_is_refused():
    with pytest.raises(ValueError, match="angle"):
        Vessel(velocity_mm_s=5.0, angle_deg=90.0)
    with pytest.raises(ValueError, match="angle"):
        Vessel(velocity_mm_s=5.0, angle_deg=-5.0)


def test_a_negative_velocity_or_zero_diameter_is_refused():
    with pytest.raises(ValueError, match="velocity"):
        Vessel(velocity_mm_s=-1.0)
    with pytest.raises(ValueError, match="diameter"):
        Vessel(velocity_mm_s=1.0, diameter_mm=0.0)


@pytest.mark.parametrize("count", [1, 2, 3, 8, 32])
def test_shell_weights_sum_to_one_and_the_mean_speed_is_the_mean_speed(count):
    speed, weight = laminar_shells(count)
    assert weight.sum() == pytest.approx(1.0)
    assert float(speed @ weight) == pytest.approx(1.0)


def test_one_shell_is_plug_flow():
    speed, weight = laminar_shells(1)
    assert speed[0] == pytest.approx(1.0)
    assert weight[0] == pytest.approx(1.0)


def test_shells_run_from_fast_at_the_centre_to_nearly_still_at_the_wall():
    speed, _ = laminar_shells(16)
    assert np.all(np.diff(speed) < 0)
    assert speed[0] < 2.0                       # never faster than twice the mean
    assert speed[-1] == pytest.approx(1 / 16)   # the wall shell barely moves
    assert speed[-1] > 0


def test_more_shells_resolve_the_profile_without_changing_its_mean():
    for count in (4, 16, 64):
        speed, weight = laminar_shells(count)
        assert float(speed @ weight) == pytest.approx(1.0)


def test_at_least_one_shell_is_required():
    with pytest.raises(ValueError, match="at least one"):
        laminar_shells(0)


def test_the_skin_layers_run_outermost_first_and_deepen_monotonically():
    depths = [depth for _, depth in SKIN_LAYERS]
    assert depths == sorted(depths)
    assert len(set(depths)) == len(depths)
    assert depths[0] > 0.0


def test_the_skin_layers_span_the_depths_this_work_images():
    names = [name for name, _ in SKIN_LAYERS]
    assert names == ["epidermis", "dermis", "hypodermis"]
    assert SKIN_LAYERS[-1][1] >= 10.0        # the deepest vessel this work targets

"""The signal model, checked against the closed forms it should reproduce."""
from __future__ import annotations

import numpy as np
import pytest

from skinsim import Protocol, Vessel, model, tissues


def ernst_signal(spin_density, t1_ms, t2star_ms, tr_ms, te_ms, flip_deg):
    """The textbook spoiled gradient-echo steady state, written out independently."""
    e1 = np.exp(-tr_ms / t1_ms)
    flip = np.radians(flip_deg)
    return (spin_density * (1 - e1) * np.sin(flip) / (1 - e1 * np.cos(flip))
            * np.exp(-te_ms / t2star_ms))


@pytest.fixture
def draws():
    return tissues.sample(["blood", "dermis", "fat"], draws=64, seed=20260101)


@pytest.fixture
def plain():
    return Protocol(tr_ms=30.0, te_ms=5.0, flip_deg=25.0, slab_mm=10.0)


def test_stationary_tissue_reproduces_the_ernst_steady_state(draws, plain):
    for name in ("blood", "dermis", "fat"):
        tissue = draws[name]
        expected = ernst_signal(tissue.spin_density, tissue.t1_ms, tissue.t2star_ms,
                                plain.tr_ms, plain.te_ms, plain.flip_deg)
        assert np.allclose(model.stationary_signal(tissue, plain), expected)


def test_the_best_flip_for_stationary_tissue_is_the_ernst_angle(plain):
    tissue = tissues.LIBRARY["dermis"].nominal()
    ernst = np.degrees(np.arccos(np.exp(-plain.tr_ms / tissue.t1_ms)))
    angles = np.linspace(1.0, 60.0, 1200)
    signal = [model.stationary_signal(tissue, plain.with_values(flip_deg=a)) for a in angles]
    assert angles[int(np.argmax(signal))] == pytest.approx(ernst, abs=0.2)


def test_stationary_mean_equals_the_single_value_without_a_ramp(draws, plain):
    assert np.allclose(model.stationary_mean(draws["fat"], plain),
                       model.stationary_signal(draws["fat"], plain))


def test_a_tone_ramp_makes_stationary_tissue_depth_dependent(draws, plain):
    """The ramp is usually discussed as something done to blood. Tissue sees it too."""
    ramped = plain.with_values(tone_percent=70.0)
    entry = model.stationary_signal(draws["dermis"], ramped, position=0.0)
    exit_ = model.stationary_signal(draws["dermis"], ramped, position=1.0)
    assert not np.allclose(entry, exit_)
    # Dermis has a long T1, so its Ernst angle is small and the ramp's larger
    # flip at the exit face overshoots it: the background darkens with depth.
    assert np.all(exit_ < entry)


def test_a_tone_ramp_moves_the_slab_averaged_background(draws, plain):
    ramped = plain.with_values(tone_percent=70.0)
    assert not np.allclose(model.stationary_mean(draws["dermis"], ramped),
                           model.stationary_signal(draws["dermis"], plain))


def test_stationary_blood_is_the_same_whether_it_flows_at_zero_or_sits(draws, plain):
    still = Vessel(velocity_mm_s=0.0, diameter_mm=0.2)
    assert np.allclose(model.blood_signal(draws["blood"], plain, still),
                       model.stationary_signal(draws["blood"], plain), rtol=1e-6)


def test_blood_crossing_instantly_keeps_all_its_magnetisation(draws, plain):
    blood = draws["blood"]
    racing = Vessel(velocity_mm_s=1e6, diameter_mm=0.2)
    unsaturated = blood.spin_density * np.sin(np.radians(plain.flip_deg)) \
        * np.exp(-plain.te_ms / blood.t2star_ms)
    assert np.allclose(model.blood_signal(blood, plain, racing), unsaturated, rtol=1e-4)


def test_blood_signal_rises_with_velocity(draws, plain):
    speeds = [0.5, 1, 2, 5, 10, 20, 50, 100, 400]
    values = [float(np.median(model.blood_signal(draws["blood"], plain,
                                                 Vessel(velocity_mm_s=v))))
              for v in speeds]
    assert all(b > a for a, b in zip(values, values[1:]))


def test_blood_fades_along_its_path_through_the_slab(draws, plain):
    positions, signal = model.blood_profile(draws["blood"], plain,
                                            Vessel(velocity_mm_s=8.0))
    median = np.median(signal, axis=-1)
    assert median[0] > median[-1]
    assert np.all(np.diff(median) <= 1e-12)


def test_a_thicker_slab_saturates_blood_more(draws, plain):
    vessel = Vessel(velocity_mm_s=8.0)
    thin = model.blood_signal(draws["blood"], plain.with_values(slab_mm=5.0), vessel)
    thick = model.blood_signal(draws["blood"], plain.with_values(slab_mm=25.0), vessel)
    assert np.median(thin) > np.median(thick)


def test_a_vessel_at_an_angle_is_saturated_more_than_one_crossing_head_on(draws, plain):
    head_on = Vessel(velocity_mm_s=8.0, angle_deg=0.0)
    oblique = Vessel(velocity_mm_s=8.0, angle_deg=75.0)
    assert np.median(model.blood_signal(draws["blood"], plain, head_on)) > \
        np.median(model.blood_signal(draws["blood"], plain, oblique))


def test_plug_flow_overstates_the_signal_compared_with_laminar(draws, plain):
    vessel = Vessel(velocity_mm_s=6.0)
    plug = model.blood_signal(draws["blood"], plain, vessel, shells=1)
    laminar = model.blood_signal(draws["blood"], plain, vessel, shells=16)
    assert np.median(plug) > np.median(laminar)


def test_fat_saturation_takes_far_more_from_fat_than_from_blood(draws, plain):
    suppressed = plain.with_values(fat_sat=True)
    fat_loss = (np.median(model.stationary_mean(draws["fat"], plain))
                / np.median(model.stationary_mean(draws["fat"], suppressed)))
    blood_loss = (np.median(model.blood_signal(draws["blood"], plain, Vessel(8.0)))
                  / np.median(model.blood_signal(draws["blood"], suppressed, Vessel(8.0))))
    assert fat_loss > 2.0
    assert blood_loss < 1.3


def test_magnetisation_transfer_takes_more_from_dermis_than_from_fat(draws, plain):
    prepared = plain.with_values(mtc=True)
    dermis = (np.median(model.stationary_mean(draws["dermis"], prepared))
              / np.median(model.stationary_mean(draws["dermis"], plain)))
    fat = (np.median(model.stationary_mean(draws["fat"], prepared))
           / np.median(model.stationary_mean(draws["fat"], plain)))
    assert dermis < fat


def test_blood_is_darker_than_unsuppressed_fat_at_cutaneous_speeds(draws, plain):
    """Worth locking in: the sign of the contrast is not a given here.

    Fat is bright and blood at a few mm/s is heavily saturated, so without fat
    suppression the vessel is the darker of the two. Any search that maximises
    a signed contrast has to reckon with that.
    """
    signed = model.contrast(draws["blood"], draws["fat"], plain, Vessel(velocity_mm_s=8.0))
    assert np.median(signed) < 0
    suppressed = model.contrast(draws["blood"], draws["fat"],
                                plain.with_values(fat_sat=True), Vessel(velocity_mm_s=8.0))
    assert np.median(suppressed) > 0


def test_partial_volume_dilutes_the_contrast_a_voxel_reports(draws, plain):
    from skinsim import Acquisition

    vessel = Vessel(velocity_mm_s=8.0, diameter_mm=0.2)
    acquisition = Acquisition(in_plane_mm=0.5, partition_mm=0.5, phase_fov_mm=40.0)
    fraction = model.blood_fraction(vessel, acquisition)
    assert 0 < fraction < 1
    true = model.contrast(draws["blood"], draws["fat"], plain, vessel)
    seen = model.apparent_contrast(draws["blood"], draws["fat"], plain, vessel, acquisition)
    assert np.allclose(seen, fraction * true)
    assert np.all(np.abs(seen) < np.abs(true))


def test_a_vessel_wider_than_the_voxel_fills_it(draws):
    from skinsim import Acquisition

    wide = Vessel(velocity_mm_s=8.0, diameter_mm=2.0)
    assert model.blood_fraction(wide, Acquisition(in_plane_mm=0.3)) == 1.0


def test_contrast_is_blood_minus_background(draws, plain):
    vessel = Vessel(velocity_mm_s=8.0)
    expected = (model.blood_signal(draws["blood"], plain, vessel)
                - model.stationary_mean(draws["fat"], plain))
    assert np.allclose(model.contrast(draws["blood"], draws["fat"], plain, vessel), expected)


def test_efficiency_prices_a_longer_repetition(draws, plain):
    """A longer repetition buys contrast. Efficiency charges for the time it took."""
    from skinsim import Acquisition

    acquisition = Acquisition(in_plane_mm=0.25, partition_mm=0.25, phase_fov_mm=50.0)
    vessel = Vessel(velocity_mm_s=8.0)
    suppressed = plain.with_values(fat_sat=True)
    quick, slow = suppressed.with_values(tr_ms=20.0), suppressed.with_values(tr_ms=60.0)
    assert acquisition.scan_time_s(slow) > acquisition.scan_time_s(quick)

    def gain(metric, **extra):
        return (np.median(metric(draws["blood"], draws["fat"], slow, vessel, **extra))
                / np.median(metric(draws["blood"], draws["fat"], quick, vessel, **extra)))

    by_contrast = gain(model.contrast)
    by_efficiency = gain(model.efficiency, acquisition=acquisition)
    assert by_contrast > 1.0            # the longer repetition really does gain contrast
    assert by_efficiency < by_contrast  # but not enough to pay for itself in full


def test_the_ensemble_shape_survives_the_whole_calculation(draws, plain):
    out = model.contrast(draws["blood"], draws["fat"], plain, Vessel(8.0))
    assert out.shape == (64,)
    assert np.all(np.isfinite(out))

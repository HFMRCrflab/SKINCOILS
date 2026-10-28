"""Protocol and acquisition settings, and the tissue parameter table."""
from __future__ import annotations

import numpy as np
import pytest

from skinsim import Acquisition, Protocol
from skinsim import tissues as T


# --- protocol -------------------------------------------------------------

def test_a_protocol_describes_itself():
    text = str(Protocol(tr_ms=30, te_ms=5, flip_deg=25, slab_mm=10, fat_sat=True))
    assert "TR 30 ms" in text and "fat sat" in text


def test_an_echo_outside_the_repetition_is_refused():
    with pytest.raises(ValueError, match="TE"):
        Protocol(tr_ms=5.0, te_ms=5.0)


def test_an_impossible_flip_or_slab_is_refused():
    with pytest.raises(ValueError, match="flip"):
        Protocol(flip_deg=0.0)
    with pytest.raises(ValueError, match="slab"):
        Protocol(slab_mm=0.0)


def test_a_preparation_that_does_not_fit_in_the_repetition_is_refused():
    with pytest.raises(ValueError, match="preparation delay"):
        Protocol(tr_ms=6.0, te_ms=2.0, fat_sat=True, prep_delay_ms=8.0)


def test_preparations_are_reported_as_the_tissue_parameters_they_use():
    assert Protocol().preparations == ()
    assert Protocol(fat_sat=True).preparations == ("fat_sat_surviving",)
    assert Protocol(fat_sat=True, mtc=True).preparations == (
        "fat_sat_surviving", "mtc_surviving")


def test_without_a_ramp_the_flip_is_the_same_everywhere():
    protocol = Protocol(flip_deg=25.0)
    assert protocol.flip_at(0.0) == pytest.approx(25.0)
    assert protocol.flip_at(1.0) == pytest.approx(25.0)


def test_a_ramp_runs_from_weak_where_blood_enters_to_strong_where_it_leaves():
    protocol = Protocol(flip_deg=20.0, tone_percent=70.0)
    assert protocol.flip_at(0.0) == pytest.approx(20.0 * 0.65)
    assert protocol.flip_at(0.5) == pytest.approx(20.0)
    assert protocol.flip_at(1.0) == pytest.approx(20.0 * 1.35)


def test_a_ramp_never_produces_a_non_positive_flip():
    assert Protocol(flip_deg=10.0, tone_percent=400.0).flip_at(0.0) > 0


def test_copies_replace_only_what_is_asked():
    base = Protocol(tr_ms=30.0, flip_deg=25.0)
    changed = base.with_values(flip_deg=12.0)
    assert changed.flip_deg == 12.0 and changed.tr_ms == base.tr_ms
    assert base.flip_deg == 25.0


# --- acquisition ----------------------------------------------------------

def test_readout_duration_is_the_reciprocal_bandwidth():
    assert Acquisition(bandwidth_hz_per_px=200.0).readout_ms == pytest.approx(5.0)


def test_counts_follow_from_resolution_and_field_of_view():
    acquisition = Acquisition(in_plane_mm=0.25, partition_mm=0.25, phase_fov_mm=50.0)
    protocol = Protocol(slab_mm=10.0)
    assert acquisition.lines(protocol) == 200
    assert acquisition.partitions(protocol) == 40


def test_acceleration_reduces_the_lines_and_the_time():
    protocol = Protocol(slab_mm=10.0)
    full = Acquisition(phase_fov_mm=50.0, in_plane_mm=0.25)
    fast = full.with_values(acceleration=2.0)
    assert fast.lines(protocol) * 2 == pytest.approx(full.lines(protocol), abs=1)
    assert fast.scan_time_s(protocol) < full.scan_time_s(protocol)


def test_scan_time_is_the_repetition_times_the_number_of_repetitions():
    acquisition = Acquisition(in_plane_mm=0.25, partition_mm=0.25, phase_fov_mm=50.0)
    protocol = Protocol(tr_ms=30.0, slab_mm=10.0)
    expected = 0.030 * 200 * 40
    assert acquisition.scan_time_s(protocol) == pytest.approx(expected)


def test_signal_to_noise_grows_with_voxel_volume():
    protocol = Protocol(slab_mm=10.0)
    coarse = Acquisition(in_plane_mm=0.50, partition_mm=0.50, phase_fov_mm=50.0)
    fine = Acquisition(in_plane_mm=0.25, partition_mm=0.25, phase_fov_mm=50.0)
    assert coarse.relative_snr(protocol) > fine.relative_snr(protocol)


def test_a_g_factor_below_one_or_fourier_outside_range_is_refused():
    with pytest.raises(ValueError):
        Acquisition(g_factor=0.5)
    with pytest.raises(ValueError):
        Acquisition(partial_fourier=0.2)


# --- tissues --------------------------------------------------------------

def test_every_library_entry_has_every_parameter_with_a_source():
    for name, tissue in T.LIBRARY.items():
        for parameter, span in tissue.ranges().items():
            assert span.low <= span.high, f"{name}.{parameter}"
            assert span.source.strip(), f"{name}.{parameter} has no source"
            assert span.confidence in ("high", "medium", "low")


def test_an_inverted_range_or_bad_confidence_is_refused():
    with pytest.raises(ValueError, match="inverted"):
        T.Range(5.0, 1.0, "low", "x")
    with pytest.raises(ValueError, match="confidence"):
        T.Range(1.0, 5.0, "certain", "x")


def test_sampling_is_reproducible_and_lands_inside_the_intervals():
    first = T.sample(["blood", "fat"], draws=128, seed=7)
    again = T.sample(["blood", "fat"], draws=128, seed=7)
    different = T.sample(["blood", "fat"], draws=128, seed=8)
    for name in ("blood", "fat"):
        for parameter in T.PARAMETERS:
            drawn = getattr(first[name], parameter)
            span = getattr(T.LIBRARY[name], parameter)
            assert drawn.shape == (128,)
            assert np.all(drawn >= span.low) and np.all(drawn <= span.high)
            assert np.allclose(drawn, getattr(again[name], parameter))
            if span.span > 0:
                assert not np.allclose(drawn, getattr(different[name], parameter))


def test_a_latin_hypercube_covers_each_interval_evenly():
    """Every parameter should spread across its range, not cluster in the middle."""
    drawn = T.sample(["dermis"], draws=400, seed=3)["dermis"].t1_ms
    span = T.LIBRARY["dermis"].t1_ms
    fraction = (drawn - span.low) / span.span
    counts, _ = np.histogram(fraction, bins=10, range=(0, 1))
    assert counts.min() >= 30


def test_an_unknown_tissue_is_refused():
    with pytest.raises(KeyError, match="unknown tissue"):
        T.sample(["bone"], draws=4, seed=1)


def test_nominal_values_are_interval_midpoints():
    nominal = T.LIBRARY["fat"].nominal()
    assert nominal.t1_ms == pytest.approx(T.LIBRARY["fat"].t1_ms.nominal)


def test_the_report_names_the_weak_parameters_and_their_sources():
    text = T.report(("low",))
    assert "dermis.t1_ms" in text
    assert "no consensus" in text.lower()
    assert "fat.spin_density" not in text          # that one is medium confidence

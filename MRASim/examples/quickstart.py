"""The shortest useful thing you can do: one protocol, one vessel, one answer.

Run: python3 examples/quickstart.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from skinsim import Acquisition, Protocol, Vessel, model, tissues  # noqa: E402

# Tissue properties are intervals, not numbers, so we draw a spread of them.
draws = tissues.sample(["blood", "dermis", "fat"], draws=256, seed=1)

protocol = Protocol(tr_ms=30.0, te_ms=5.0, flip_deg=25.0, slab_mm=10.0, fat_sat=True)
vessel = Vessel(velocity_mm_s=8.0, diameter_mm=0.2)
acquisition = Acquisition(in_plane_mm=0.25, partition_mm=0.25, phase_fov_mm=50.0,
                          acceleration=2.0, g_factor=1.3)

print(f"protocol:    {protocol}")
print(f"vessel:      {vessel}")
print(f"scan time:   {acquisition.scan_time_s(protocol) / 60:.1f} min "
      f"({acquisition.lines(protocol)} lines x {acquisition.partitions(protocol)} partitions)")


def spread(label, values):
    """Print the median and the 5-95 % range of a spread of draws."""
    low, mid, high = np.percentile(values, [5, 50, 95])
    print(f"  {label:28s} {mid:+.4f}  [{low:+.4f} to {high:+.4f}]")


print("\nsignals, in units where a fully relaxed unit spin density gives 1.0")
spread("blood, flowing", model.blood_signal(draws["blood"], protocol, vessel))
spread("blood, stagnant", model.stationary_mean(draws["blood"], protocol))
spread("dermis", model.stationary_mean(draws["dermis"], protocol))
spread("fat, suppressed", model.stationary_mean(draws["fat"], protocol))

print("\ncontrast against each background")
for name in ("dermis", "fat"):
    spread(f"vs {name}, true", model.contrast(draws["blood"], draws[name], protocol, vessel))
    spread(f"vs {name}, in a voxel",
           model.apparent_contrast(draws["blood"], draws[name], protocol, vessel, acquisition))

fraction = model.blood_fraction(vessel, acquisition)
print(f"\nThe vessel fills {fraction:.1%} of a voxel, so that is what the contrast "
      f"is multiplied by\nbefore anything can be detected.")

print("\nhow the vessel fades along its path through the slab")
positions, profile = model.blood_profile(draws["blood"], protocol, vessel)
for at in (0.0, 0.25, 0.5, 0.75, 1.0):
    index = int(at * (len(positions) - 1))
    print(f"  {at:4.0%} of the way across   {np.median(profile[index]):.4f}")

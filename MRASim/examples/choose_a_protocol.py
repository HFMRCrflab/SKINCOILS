"""Choosing a protocol rather than checking one, and the traps along the way.

Four searches, each making a point:

1. a signed score in a regime where blood is the darker tissue, which goes wrong;
2. the same search asking for the magnitude instead, which does not;
3. an unconstrained search, which drives the slab to nothing;
4. the search you actually want, with coverage and scan time pinned down.

Run: python3 examples/choose_a_protocol.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from skinsim import Acquisition, Protocol, Vessel, optimise, tissues  # noqa: E402

SEED = 4
draws = tissues.sample(["blood", "fat"], draws=96, seed=SEED)
blood, fat = draws["blood"], draws["fat"]

start = Protocol(tr_ms=30.0, te_ms=5.0, flip_deg=25.0, slab_mm=10.0)
vessel = Vessel(velocity_mm_s=6.0, diameter_mm=0.2)
acquisition = Acquisition(in_plane_mm=0.25, partition_mm=0.25, phase_fov_mm=50.0,
                          acceleration=2.0, g_factor=1.3)
FAST = dict(shells=4, samples=33)
BOUNDS = {"flip_deg": (3.0, 45.0), "tr_ms": (12.0, 200.0)}


def show(title, best):
    """Print one search result with what it costs on the scanner."""
    minutes = acquisition.scan_time_s(best.protocol) / 60.0
    print(f"\n{title}")
    print(f"  {best.protocol}")
    print(f"  score {best.score:+.3e}   scan time {minutes:.1f} min")


# 1. Without fat suppression blood is darker than fat, so the contrast is
#    negative and maximising it chases the least visible vessel.
signed = optimise.efficiency_score(blood, fat, vessel, acquisition, **FAST)
show("1. signed score, fat not suppressed - the trap",
     optimise.optimise(start, BOUNDS, signed, seed=SEED, iterations=30))

# 2. Asking for the magnitude finds the dark-blood optimum honestly.
magnitude = optimise.efficiency_score(blood, fat, vessel, acquisition, absolute=True, **FAST)
show("2. same thing, asking for magnitude",
     optimise.optimise(start, BOUNDS, magnitude, seed=SEED, iterations=30))

# 3. With fat suppressed the contrast is positive, but nothing in the score
#    says the slab has to cover anything, so it collapses to the lower bound.
suppressed = start.with_values(fat_sat=True)
score = optimise.efficiency_score(blood, fat, vessel, acquisition, **FAST)
show("3. fat suppressed, slab free to shrink",
     optimise.optimise(suppressed, {**BOUNDS, "slab_mm": (4.0, 30.0)}, score,
                       seed=SEED, iterations=30))

# 4. The search worth trusting: coverage required, time budgeted, preparations
#    left to the optimiser, and the cautious end of the parameter ranges.
cautious = optimise.efficiency_score(blood, fat, vessel, acquisition,
                                     statistic=lambda v: np.percentile(v, 5), **FAST)
budget_s = 300.0
show(f"4. slab fixed at 12 mm, under {budget_s / 60:.0f} min, 5th percentile of the draws",
     optimise.optimise(
         suppressed.with_values(slab_mm=12.0),
         {**BOUNDS, "tone_percent": (0.0, 90.0)},
         cautious,
         switches=["fat_sat", "mtc"],
         constraint=lambda p: acquisition.scan_time_s(p) <= budget_s,
         seed=SEED, iterations=30))

# How much does the answer move with the flow you are targeting?
print("\n\nthe optimum moves with the velocity you design for")
for speed in (1.0, 3.0, 10.0, 30.0):
    target = Vessel(velocity_mm_s=speed, diameter_mm=0.2)
    best = optimise.optimise(
        suppressed.with_values(slab_mm=12.0),
        BOUNDS, optimise.efficiency_score(blood, fat, target, acquisition, **FAST),
        seed=SEED, iterations=25)
    print(f"  {speed:5.1f} mm/s   flip {best.protocol.flip_deg:4.1f}°   "
          f"TR {best.protocol.tr_ms:5.1f} ms   "
          f"{acquisition.scan_time_s(best.protocol) / 60:4.1f} min")

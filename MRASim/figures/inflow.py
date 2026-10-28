"""What inflow buys, and which background it has to beat.

One panel. Blood signal against velocity, with the four levels it is measured
against drawn as flat references:

    blood, stagnant       the spoiled-gradient-echo fixed point, a vessel with
                          no flow. The gap between it and the curve IS the
                          inflow enhancement, so the figure shows inflow rather
                          than asking the reader to assume it.
    subcutaneous fat      the background at the depth these vessels sit at.
    fat, suppressed       the same background with a fat-saturation pulse.
    dermis                the background for the shallowest vessels.

Two things follow, and they are different claims:

  - inflow still works at cutaneous speeds, several-fold over the stagnant
    floor, so it is not exhausted;
  - but it is not what makes the vessel visible. Stagnant blood already beats
    dermis and suppressed fat. Inflow is load-bearing only against
    UNSUPPRESSED fat, and there it fails: below the crossing velocity the
    vessel is the darker of the two.

The velocity regimes sit on a rail below the axes rather than as shaded bands
inside it, so the plot area carries curves and nothing else.

Run: python3 figures/inflow.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import brentq  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from skinsim import Protocol, Vessel, model, tissues  # noqa: E402
from skinsim.geometry import TARGET_VELOCITY_MM_S  # noqa: E402
from skinsim.plotting import GRID, INK, MUTED, SERIES  # noqa: E402

BLOOD_C, FAT_C = SERIES[1], SERIES[0]
DOTTED = (0, (1.2, 1.8))

SEED, DRAWS = 20260422, 256

# The readout this describes is a spoiled-gradient-echo TOF acquisition at its
# shortest repetition with no preparation pulse played. Fat suppression at the
# same TR is then the counterfactual the figure needs, rather than a comparison
# that also moves the timing. TONE and any travelling saturation band are left
# off deliberately: both would fold into the same curve and could not be read
# apart from inflow itself.
PROTOCOL = Protocol(tr_ms=35.06, te_ms=5.86, flip_deg=25.0, slab_mm=11.44)
DIAMETER_MM = 0.2

VELOCITIES = np.geomspace(2.0, 300.0, 64)
TICKS = (2, 3, 5, 10, 20, 30, 50, 100, 200, 300)
CONVENTIONAL_TOF = (200.0, 600.0)   # intracranial and run-off TOF, for scale


def blood_at(blood, protocol, velocity):
    """Return the path-averaged blood signal at one velocity, over all draws.

    The vessel crosses the slab head-on, so the whole velocity counts towards
    the crossing and the path is as short as the geometry allows. That is the
    BEST case for inflow - any other angle lengthens the path and saturates the
    blood further - so an inversion reported here is the conservative end.
    """
    return model.blood_signal(blood, protocol,
                              Vessel(velocity_mm_s=float(velocity), diameter_mm=DIAMETER_MM,
                                     angle_deg=0.0))


def crossing_velocities(blood, protocol, level):
    """Return, per draw, the velocity at which blood overtakes its own background.

    Solved by bisection in log velocity rather than read off the plotting grid,
    so the bounds do not move when the grid is refined.
    """
    found = []
    for index in range(np.size(level)):
        one = blood.with_values(**{field: np.atleast_1d(getattr(blood, field))[index]
                                   for field in tissues.PARAMETERS})
        target = float(np.atleast_1d(level)[index])

        def excess(log_velocity, sample=one, against=target):
            return float(blood_at(sample, protocol, np.exp(log_velocity))) - against

        low, high = np.log(VELOCITIES[0]), np.log(VELOCITIES[-1])
        if excess(low) * excess(high) < 0:
            found.append(float(np.exp(brentq(excess, low, high, xtol=1e-4))))
    return np.array(found)


def reference(axis, values, colour, style, label, labels, band=True):
    """Draw a velocity-independent level and queue its right-margin label.

    The label is queued rather than placed, because two levels can land on top
    of each other - and in this parameter table dermis and stagnant blood do.
    """
    low, middle, high = np.percentile(np.asarray(values, dtype=float), [5, 50, 95])
    if band:
        axis.axhspan(low, high, color=colour, alpha=0.13, lw=0, zorder=1)
    axis.axhline(middle, color=colour, lw=1.7, ls=style, zorder=3)
    labels.append((float(middle), colour, label, False))
    return float(middle)


def place_labels(axis, labels, ceiling):
    """Write the queued right-margin labels, nudging apart any that would collide."""
    gap = 0.05 * ceiling
    placed = []
    for value, colour, text, bold in sorted(labels, key=lambda item: item[0]):
        at = value if not placed or value - placed[-1] >= gap else placed[-1] + gap
        placed.append(at)
        axis.annotate(text, (1.0, at), xycoords=("axes fraction", "data"), xytext=(7, 0),
                      textcoords="offset points", va="center", fontsize=8.5, color=colour,
                      weight="bold" if bold else "normal", annotation_clip=False)


def regime_rail(axis):
    """Draw the velocity regimes on a rail below the axes, keeping the plot area for curves."""
    transform = axis.get_xaxis_transform()
    for (low, high), colour, text in ((TARGET_VELOCITY_MM_S, INK, "targeted vessels"),
                                      (CONVENTIONAL_TOF, MUTED, "conventional TOF")):
        drawn = (max(low, VELOCITIES[0]), min(high, VELOCITIES[-1]))
        axis.plot(drawn, [-0.09, -0.09], transform=transform, color=colour, lw=3.2, alpha=0.5,
                  solid_capstyle="butt", clip_on=False, zorder=5)
        axis.annotate(f"{text}\n{low:g}-{high:g} mm/s", (np.sqrt(drawn[0] * drawn[1]), -0.125),
                      xycoords=transform, ha="center", va="top", fontsize=9, color=INK,
                      clip_on=False)


def main() -> Path:
    drawn = tissues.sample(["blood", "dermis", "fat"], draws=DRAWS, seed=SEED)
    blood, dermis, fat = drawn["blood"], drawn["dermis"], drawn["fat"]
    suppressed = PROTOCOL.with_values(fat_sat=True)

    figure, axis = plt.subplots(figsize=(7.6, 4.9))
    figure.subplots_adjust(left=0.135, right=0.775, top=0.90, bottom=0.26)

    flowing = np.array([blood_at(blood, PROTOCOL, v) for v in VELOCITIES])
    low, middle, high = np.percentile(flowing, [5, 50, 95], axis=-1)

    labels: list[tuple[float, str, str, bool]] = []
    fat_level = reference(axis, model.stationary_mean(fat, PROTOCOL), FAT_C, "-",
                          "subcutaneous fat", labels)
    fat_suppressed = reference(axis, model.stationary_mean(fat, suppressed), FAT_C, "--",
                               "fat, suppressed", labels)
    dermis_values = model.stationary_mean(dermis, PROTOCOL)
    stagnant_values = model.stationary_mean(blood, PROTOCOL)
    dermis_level = reference(axis, dermis_values, MUTED, DOTTED, "dermis", labels)
    stagnant = reference(axis, stagnant_values, BLOOD_C, DOTTED, "blood, stagnant", labels)

    # The inversion as an area, not a crossing the reader has to decode.
    axis.fill_between(VELOCITIES, middle, fat_level, where=middle < fat_level, color=BLOOD_C,
                      alpha=0.26, lw=0, zorder=2, interpolate=True)
    axis.fill_between(VELOCITIES, low, high, color=BLOOD_C, alpha=0.32, lw=0, zorder=4)
    axis.plot(VELOCITIES, middle, color=BLOOD_C, lw=2.5, zorder=5)
    labels.append((float(middle[-1]), BLOOD_C, "blood, inflowing", True))

    crossings = crossing_velocities(blood, PROTOCOL, model.stationary_mean(fat, PROTOCOL))
    c5, c50, c95 = np.percentile(crossings, [5, 50, 95])
    axis.plot([c50], [fat_level], "o", ms=8, mfc="white", mec=BLOOD_C, mew=2.1, zorder=7)

    ceiling = float(np.ceil(high.max() / 0.05) * 0.05)
    place_labels(axis, labels, ceiling)
    axis.set(xscale="log", xlim=(VELOCITIES[0], VELOCITIES[-1]), ylim=(0, ceiling), xticks=TICKS)
    axis.set_xticklabels([str(t) for t in TICKS])
    axis.xaxis.set_minor_locator(plt.NullLocator())
    axis.set_xlabel("mean blood velocity (mm/s)", color=INK, fontsize=11, labelpad=40)
    axis.set_ylabel("simulated signal\n(equilibrium of unit spin density = 1)", color=INK,
                    fontsize=11)
    axis.text(0, 1.02, f"spoiled-GRE TOF, TR {PROTOCOL.tr_ms:g} ms, {PROTOCOL.flip_deg:g}°, "
                       f"{PROTOCOL.slab_mm:g} mm slab",
              transform=axis.transAxes, fontsize=9.5, color=INK)
    axis.grid(color=GRID, lw=0.6, zorder=0)
    axis.set_axisbelow(True)
    axis.tick_params(colors=INK, labelsize=9.5)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    regime_rail(axis)

    # --- every number the caption needs ---------------------------------
    slow, fast = TARGET_VELOCITY_MM_S
    i_slow, i_fast = (int(np.argmin(np.abs(VELOCITIES - v))) for v in (slow, fast))
    print(f"Levels: stagnant blood {stagnant:.3f} | dermis {dermis_level:.3f} | "
          f"fat {fat_level:.3f} | suppressed fat {fat_suppressed:.3f}")
    print(f"Stagnant blood already beats dermis {stagnant / dermis_level:.2f}x and suppressed "
          f"fat {stagnant / fat_suppressed:.1f}x, but unsuppressed fat only "
          f"{stagnant / fat_level:.2f}x")
    print(f"Inflow over stagnant: {middle[i_slow] / stagnant:.2f}x at {VELOCITIES[i_slow]:.0f} "
          f"mm/s, {middle[i_fast] / stagnant:.2f}x at {VELOCITIES[i_fast]:.0f} mm/s")
    print(f"Inversion vs unsuppressed fat: {len(crossings)}/{DRAWS} draws invert, at "
          f"{c50:.0f} mm/s [{c5:.0f}-{c95:.0f}] (transit {PROTOCOL.slab_mm / c50:.2f} s)")
    print(f"Against suppressed fat the median never inverts: minimum "
          f"{np.min(middle) / fat_suppressed:.1f}x")
    for name, level in (("dermis", dermis_values),
                        ("suppressed fat", model.stationary_mean(fat, suppressed))):
        difference = stagnant_values - level
        print(f"  paired, stagnant blood - {name}: {np.median(difference):+.3f} "
              f"[{np.percentile(difference, 5):+.3f}, {np.percentile(difference, 95):+.3f}], "
              f"brighter in {np.mean(difference > 0):.0%} of draws")

    out = Path(__file__).resolve().parent / "inflow.png"
    figure.savefig(out, dpi=200)
    plt.close(figure)
    return out


if __name__ == "__main__":
    print(f"\nwrote {main()}")

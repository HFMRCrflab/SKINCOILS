"""Where the best protocol sits, and how far it moves when the flow changes.

Contrast-to-noise per square root of scan time, over flip angle and repetition
time, at three blood velocities. The point is not any single optimum but how
much it travels: a protocol tuned for one flow regime is not the protocol for
another, and the difference is larger than the usual rules of thumb suggest.

Run: python3 figures/optimisation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from skinsim import Acquisition, Protocol, Vessel, optimise, tissues  # noqa: E402
from skinsim.plotting import INK, MUTED, style  # noqa: E402

SEED, DRAWS = 20260422, 96
FLIPS = np.linspace(3.0, 45.0, 36)
REPETITIONS = np.linspace(12.0, 260.0, 36)
VELOCITIES = (2.0, 8.0, 30.0)
BASE = Protocol(tr_ms=30.0, te_ms=5.0, flip_deg=25.0, slab_mm=10.0, fat_sat=True)
ACQUISITION = Acquisition(in_plane_mm=0.25, partition_mm=0.25, phase_fov_mm=50.0,
                          bandwidth_hz_per_px=200.0, acceleration=2.0, g_factor=1.3)
RESOLVED = dict(shells=4, samples=33)       # enough for a map; the tests use more


def main() -> Path:
    drawn = tissues.sample(["blood", "fat"], draws=DRAWS, seed=SEED)
    blood, fat = drawn["blood"], drawn["fat"]

    maps, optima = [], []
    for speed in VELOCITIES:
        vessel = Vessel(velocity_mm_s=speed, diameter_mm=0.2)
        score = optimise.efficiency_score(blood, fat, vessel, ACQUISITION, **RESOLVED)
        result = optimise.sweep(BASE, {"flip_deg": FLIPS, "tr_ms": REPETITIONS}, score)
        maps.append(result.values)
        optima.append(result.best())
        print(f"{speed:5.1f} mm/s   {result.summary()}")

    highest = max(float(np.nanmax(m)) for m in maps)
    figure, axes = plt.subplots(1, len(VELOCITIES), figsize=(12.4, 4.3), sharey=True,
                                constrained_layout=True)

    for axis, speed, values, (settings, best) in zip(axes, VELOCITIES, maps, optima):
        mesh = axis.pcolormesh(FLIPS, REPETITIONS, values.T / highest, cmap="viridis",
                               vmin=0.0, vmax=1.0, shading="nearest", rasterized=True)
        axis.contour(FLIPS, REPETITIONS, values.T / highest,
                     levels=[0.5, 0.7, 0.85, 0.95], colors="white", linewidths=0.6,
                     alpha=0.55)
        axis.plot(settings["flip_deg"], settings["tr_ms"], marker="o", ms=7,
                  mfc="none", mec="white", mew=1.8)
        high = settings["tr_ms"] > REPETITIONS[0] + 0.85 * (REPETITIONS[-1] - REPETITIONS[0])
        right = settings["flip_deg"] > FLIPS[0] + 0.7 * (FLIPS[-1] - FLIPS[0])
        axis.annotate(f"{settings['flip_deg']:.0f}°, TR {settings['tr_ms']:.0f} ms",
                      xy=(settings["flip_deg"], settings["tr_ms"]),
                      xytext=(-8 if right else 8, -14 if high else 8),
                      ha="right" if right else "left",
                      textcoords="offset points", fontsize=8.5, color="white")
        axis.set_xlabel("flip angle (degrees)", color=MUTED)
        axis.set_title(f"blood at {speed:g} mm/s", fontsize=11, color=INK, loc="left")
        style(axis)
        axis.grid(False)

    axes[0].set_ylabel("repetition time (ms)", color=MUTED)
    bar = figure.colorbar(mesh, ax=axes, fraction=0.025, pad=0.012)
    bar.set_label("efficiency, relative to the best point shown", color=MUTED, fontsize=9)
    bar.ax.tick_params(colors=MUTED, labelsize=8)

    figure.suptitle("Contrast-to-noise per root scan time, against fat suppressed",
                    fontsize=12.5, x=0.006, ha="left", color=INK)
    figure.text(0.006, -0.015,
                f"Literature parameter intervals only. Latin hypercube, seed {SEED}, "
                f"{DRAWS} draws, median over draws. Slab 10 mm, 0.25 mm in-plane, "
                f"GRAPPA 2 at g = 1.3. Partial volume included for a 200 um vessel.",
                fontsize=7.5, color=MUTED, ha="left", va="top")

    out = Path(__file__).resolve().parent / "optimisation.png"
    figure.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return out


def search() -> None:
    """Search all four knobs at once, rather than the two the map can show."""
    drawn = tissues.sample(["blood", "fat"], draws=DRAWS, seed=SEED)
    print("\nsearching flip, TR, slab and TONE together, with and without each preparation")
    for speed in VELOCITIES:
        vessel = Vessel(velocity_mm_s=speed, diameter_mm=0.2)
        score = optimise.efficiency_score(drawn["blood"], drawn["fat"], vessel,
                                          ACQUISITION, **RESOLVED)
        best = optimise.optimise(
            BASE,
            bounds={"flip_deg": (3.0, 45.0), "tr_ms": (12.0, 160.0),
                    "slab_mm": (4.0, 30.0), "tone_percent": (0.0, 90.0)},
            switches=["fat_sat", "mtc"], score=score, seed=SEED, iterations=30)
        minutes = ACQUISITION.scan_time_s(best.protocol) / 60.0
        print(f"  {speed:5.1f} mm/s  {best.protocol}  ->  {minutes:.1f} min")


if __name__ == "__main__":
    print(f"wrote {main()}")
    search()

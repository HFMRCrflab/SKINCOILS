"""A small shared style, so every figure in the project looks like the others.

Nothing here is required to use the toolbox. It exists so that figures do not
each invent their own colours, and so that the spread over tissue parameter
draws is always drawn the same way: a line at the median, a band between the
5th and 95th percentiles.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

__all__ = ["SERIES", "INK", "MUTED", "GRID", "percentiles", "band", "style", "annotate_level"]

#: A categorical palette that stays distinguishable in greyscale and to the
#: most common colour vision deficiencies.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")

INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"


def percentiles(values: np.ndarray, low: float = 5.0, high: float = 95.0) -> np.ndarray:
    """Return the low, median and high percentiles over the last axis.

    Non-finite draws are dropped rather than poisoning the result, which
    matters where some corner of the parameter ranges gives no answer at all.
    """
    finite = np.where(np.isfinite(values), values, np.nan)
    with np.errstate(invalid="ignore"):
        return np.nanpercentile(finite, [low, 50.0, high], axis=-1)


def band(ax, x: Sequence[float], values: np.ndarray, colour: str, label: str | None = None,
         **line_options):
    """Draw the median over draws as a line and the 5-95 % range as a band."""
    low, middle, high = percentiles(np.asarray(values))
    ax.fill_between(x, low, high, color=colour, alpha=0.26, linewidth=0)
    ax.plot(x, middle, color=colour, lw=2.0, label=label, **line_options)
    return middle


def style(ax) -> None:
    """Apply the recessive grid and axes used throughout."""
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK, labelsize=9.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def annotate_level(ax, values: np.ndarray, colour: str, text: str, where: float = 0.02,
                   above: bool = True) -> float:
    """Draw a horizontal line at the median of ``values`` with its band, and label it.

    Used for stationary tissue, whose signal does not depend on velocity and so
    appears as a level rather than a curve.
    """
    low, middle, high = percentiles(np.asarray(values))
    ax.axhspan(float(low), float(high), color=colour, alpha=0.16, lw=0)
    ax.axhline(float(middle), color=colour, lw=1.6)
    ax.annotate(text, xy=(where, float(middle)), xycoords=("axes fraction", "data"),
                xytext=(0, 4 if above else -11), textcoords="offset points",
                fontsize=8.5, color=colour)
    return float(middle)

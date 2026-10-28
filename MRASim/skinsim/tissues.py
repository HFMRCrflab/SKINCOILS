"""Tissue properties at 3 T, as intervals rather than single numbers.

Nothing here is a measurement made with this toolbox. Every entry is a span of
values reported in the literature or, where the literature does not settle the
question, a deliberately wide interval with its confidence marked ``low``. The
uncertainty is the point: a prediction that only holds for one corner of these
intervals is not a prediction worth quoting, and the only way to find that out
is to carry the whole span through the calculation.

Signals are in units where a tissue of spin density 1.0 at full relaxation
gives 1.0. There is no reference tissue, so a signal can be compared with any
other signal in the toolbox without asking what it was divided by.

Read the parameters you are about to rely on::

    from skinsim.tissues import LIBRARY, report
    print(report())
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Iterable, Sequence

import numpy as np
from scipy.stats import qmc

__all__ = ["Range", "Tissue", "TissueValues", "LIBRARY", "PARAMETERS", "sample", "report"]

#: The parameter names carried by every tissue, in a fixed order.
PARAMETERS = ("t1_ms", "t2star_ms", "spin_density", "fat_sat_surviving", "mtc_surviving")


@dataclass(frozen=True)
class Range:
    """An interval a parameter is believed to lie in, with where that belief came from.

    ``confidence`` is one of ``"high"``, ``"medium"``, ``"low"``. It describes
    how well the interval is pinned down, not how wide it is.
    """

    low: float
    high: float
    confidence: str
    source: str

    def __post_init__(self) -> None:
        if self.low > self.high:
            raise ValueError(f"range is inverted: {self.low} > {self.high}")
        if self.confidence not in ("high", "medium", "low"):
            raise ValueError(f"confidence must be high, medium or low, not {self.confidence!r}")

    @property
    def nominal(self) -> float:
        """Return the midpoint, for a quick single-number answer."""
        return 0.5 * (self.low + self.high)

    @property
    def span(self) -> float:
        """Return the width of the interval."""
        return self.high - self.low


@dataclass(frozen=True)
class Tissue:
    """One tissue, every parameter an interval."""

    name: str
    t1_ms: Range
    t2star_ms: Range
    spin_density: Range
    fat_sat_surviving: Range
    mtc_surviving: Range

    def nominal(self) -> "TissueValues":
        """Return this tissue with every parameter at its interval midpoint."""
        return TissueValues(self.name,
                            **{p: getattr(self, p).nominal for p in PARAMETERS})

    def ranges(self) -> dict[str, Range]:
        """Return the parameter intervals keyed by name."""
        return {p: getattr(self, p) for p in PARAMETERS}


@dataclass(frozen=True)
class TissueValues:
    """One tissue with concrete numbers: scalars, or arrays over a set of draws."""

    name: str
    t1_ms: float | np.ndarray
    t2star_ms: float | np.ndarray
    spin_density: float | np.ndarray
    fat_sat_surviving: float | np.ndarray
    mtc_surviving: float | np.ndarray

    @property
    def size(self) -> int:
        """Return how many draws this carries, or 1 for scalars."""
        return int(np.size(self.t1_ms))

    def with_values(self, **changes) -> "TissueValues":
        """Return a copy with some parameters replaced."""
        return replace(self, **changes)


# --------------------------------------------------------------------------
# The library.
#
# T2* rather than T2 is what a gradient-echo readout sees, and it is the least
# well determined quantity here for every tissue: it depends on voxel size,
# shim and local susceptibility as much as on the tissue. Those entries are
# marked low confidence and given wide intervals on purpose.
# --------------------------------------------------------------------------

_BLOOD = Tissue(
    name="arterial blood",
    t1_ms=Range(1550, 1950, "medium",
                "Reported 3 T whole-blood T1 clusters near 1650-1930 ms "
                "(Lu et al., Magn Reson Med 2004;52:679; Stanisz et al., Magn Reson Med "
                "2005;54:507). The interval spans the reported values and haematocrit "
                "dependence."),
    t2star_ms=Range(30, 70, "low",
                    "Blood T2 at 3 T is long (165-275 ms, Stanisz et al. 2005) but T2* is "
                    "far shorter and depends on oxygenation and on the vessel's angle to B0. "
                    "No consensus value exists for vessels of this calibre."),
    spin_density=Range(0.82, 0.90, "medium",
                       "Water fraction of whole blood at normal haematocrit."),
    fat_sat_surviving=Range(0.92, 1.00, "low",
                            "A chemically selective fat pulse is off-resonance for water, so "
                            "it should barely touch blood; the shortfall from 1.0 allows for "
                            "direct saturation and imperfect selectivity."),
    mtc_surviving=Range(0.88, 0.98, "low",
                        "Blood carries little macromolecular content, so magnetisation "
                        "transfer should be weak, but the off-resonance pulse still causes "
                        "some direct saturation."),
)

_DERMIS = Tissue(
    name="dermis",
    t1_ms=Range(900, 1400, "low",
                "There is no consensus 3 T measurement of dermis T1. Values near 850 ms "
                "reported at 1.5 T rise at 3 T; the interval is deliberately wide and this "
                "is the parameter most worth measuring."),
    t2star_ms=Range(8, 15, "medium",
                    "Measured directly, in the dorsal calf, at 1.5 T: 12.0 +- 1.4 ms fitted "
                    "over TE 2.8-20 ms, resolving into 9.8 ms (91 %) and 42.7 ms (9 %) once "
                    "the echo range is extended to 60 ms (Song et al., Magn Reson Med "
                    "1997;37:185). Barral et al., Magn Reson Med 2010;63:790 take 10 ms as "
                    "the typical dermis value. True T2 at 3 T is 27.6 ms [24.0-33.0] in 33 "
                    "subjects (Mesrar et al., Skin Res Technol 2017;23:289), which bounds "
                    "T2* from above; dephasing grows with field, so the 1.5 T measurement is "
                    "itself an upper bound at 3 T. This interval covers the dominant "
                    "component only, which carries 91 % of the signal."),
    spin_density=Range(0.60, 0.75, "low",
                       "Water content of dermis is roughly two thirds by volume."),
    fat_sat_surviving=Range(0.90, 1.00, "low",
                            "Off-resonance for water, but collagen's broad line means a fat "
                            "pulse removes some dermis magnetisation by transfer."),
    mtc_surviving=Range(0.40, 0.70, "low",
                        "Collagen is strongly magnetisation-transfer active, so dermis is "
                        "expected to lose a large fraction. The size of that loss is the "
                        "least certain number in this table."),
)

_FAT = Tissue(
    name="subcutaneous fat",
    t1_ms=Range(320, 400, "medium",
                "3 T fat T1 is commonly reported near 365 ms across several studies."),
    t2star_ms=Range(25, 60, "low",
                    "Fat T2 is long (near 130 ms) but the multiple spectral peaks dephase "
                    "against each other, so the effective T2* in a gradient echo is much "
                    "shorter and depends on TE."),
    spin_density=Range(0.85, 1.00, "medium",
                       "Lipid protons are dense, so fat gives a strong signal per unit "
                       "volume even before its short T1 is taken into account."),
    fat_sat_surviving=Range(0.05, 0.30, "low",
                            "How much fat survives a chemically selective pulse depends on "
                            "pulse design, B1 uniformity and shim. The interval spans a good "
                            "and a mediocre suppression."),
    mtc_surviving=Range(0.90, 1.00, "medium",
                        "Lipid protons show little magnetisation transfer, which is why "
                        "an MT pulse suppresses dermis and muscle far more than fat."),
)

_MUSCLE = Tissue(
    name="muscle",
    t1_ms=Range(1300, 1500, "medium",
                "Stanisz et al., Magn Reson Med 2005;54:507 report about 1412 ms at 3 T."),
    t2star_ms=Range(20, 35, "medium",
                    "Song et al., Magn Reson Med 1997;37:185 measured 29.8 ms in vivo at "
                    "1.5 T; T2* is shorter again at 3 T and remains shim-dependent."),
    spin_density=Range(0.70, 0.85, "medium",
                       "Water content of skeletal muscle."),
    fat_sat_surviving=Range(0.92, 1.00, "low",
                            "Off-resonance for water; the shortfall allows for transfer "
                            "and direct saturation."),
    mtc_surviving=Range(0.45, 0.65, "medium",
                        "Muscle shows substantial magnetisation transfer (Stanisz et al. "
                        "2005), comparable to other collagen- and protein-rich tissue."),
)

#: Every tissue the toolbox knows, keyed by a short name.
LIBRARY: dict[str, Tissue] = {
    "blood": _BLOOD,
    "dermis": _DERMIS,
    "fat": _FAT,
    "muscle": _MUSCLE,
}


def sample(names: Sequence[str], draws: int, seed: int) -> dict[str, TissueValues]:
    """Return a Latin hypercube draw of the named tissues' parameters.

    Every parameter of every named tissue gets its own dimension of the
    hypercube, so the draws explore the joint space rather than varying one
    parameter at a time. The result maps each name to a
    :class:`TissueValues` whose fields are arrays of length ``draws``.

    ``seed`` is required, not optional: a figure that cannot be regenerated
    exactly is not a result.
    """
    unknown = [n for n in names if n not in LIBRARY]
    if unknown:
        raise KeyError(f"unknown tissue(s) {unknown}; known names are {sorted(LIBRARY)}")
    if draws < 1:
        raise ValueError("need at least one draw")

    dimensions = len(names) * len(PARAMETERS)
    unit = qmc.LatinHypercube(d=dimensions, seed=seed).random(draws)

    out: dict[str, TissueValues] = {}
    column = 0
    for name in names:
        tissue = LIBRARY[name]
        values: dict[str, np.ndarray] = {}
        for parameter in PARAMETERS:
            span = getattr(tissue, parameter)
            values[parameter] = span.low + unit[:, column] * (span.high - span.low)
            column += 1
        out[name] = TissueValues(tissue.name, **values)
    return out


def report(confidence: Iterable[str] = ("low",)) -> str:
    """Return a readable list of parameters at the given confidence levels.

    Call this before quoting any number the toolbox produces. By default it
    lists everything resting on a weak footing.
    """
    wanted = set(confidence)
    lines: list[str] = []
    for key, tissue in LIBRARY.items():
        for parameter, span in tissue.ranges().items():
            if span.confidence in wanted:
                lines.append(f"{key}.{parameter}: {span.low:g} to {span.high:g} "
                             f"[{span.confidence}]\n    {span.source}")
    if not lines:
        return f"no parameters at confidence {sorted(wanted)}"
    header = f"{len(lines)} parameter(s) at confidence {sorted(wanted)}:"
    return "\n".join([header, *lines])

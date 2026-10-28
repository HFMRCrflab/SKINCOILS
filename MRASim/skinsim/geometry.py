"""Where the blood goes, and how long it stays in the slab.

Inflow contrast comes down to one number: how long a parcel of blood spends
inside the excited slab, because that fixes how many excitations it receives
before it leaves. That dwell time is the path length through the slab divided
by the speed along it, and the path length follows from the slab thickness and
the angle the vessel makes with the slab normal::

    path = thickness / cos(angle)

A vessel crossing the slab head-on (angle 0) takes the shortest path and keeps
the most magnetisation. A vessel lying in the plane of the slab (angle 90)
never leaves, and is saturated like stationary tissue.

Flow is laminar, not plug: in a vessel of this calibre the profile is
parabolic, so a single lumen contains a whole range of dwell times at once.
Blood at the wall barely moves and is fully saturated no matter how fast the
centre travels, which is why a plug-flow model overstates what a vessel will
look like.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

__all__ = ["Vessel", "laminar_shells", "SKIN_LAYERS", "TARGET_VELOCITY_MM_S",
           "CUTANEOUS_DIAMETER_MM", "DEFAULT_MAX_PATH_MM"]

#: Depth below the skin surface of the bottom of each layer, in mm, outermost
#: first. The simulation itself is one-compartment and never consults this: it
#: is here so that figures and protocol choices quote depths from one place
#: rather than from literals. The spans are nominal for the lower leg and vary
#: with site, age and sex; hypodermis in particular has no fixed floor, and the
#: value given is the depth this work images to rather than an anatomical limit.
SKIN_LAYERS = (("epidermis", 0.1), ("dermis", 2.0), ("hypodermis", 10.0))

#: Speeds, in mm/s, of the vessels this work sets out to image: the arterioles
#: and small arteries and veins of the dermal and subcutaneous plexus, at the
#: calibre given below. It is deliberately not the whole cutaneous range, which
#: reaches down to well under 1 mm/s in the capillary bed; those vessels are
#: below the resolution targeted here, and including them would understate the
#: inflow available to the vessels that are actually resolvable.
#:
#: Flow in skin is under thermoregulatory control and varies several-fold with
#: temperature and vasomotion, so this is a working span rather than a
#: physiological constant, and the upper end is the less certain of the two.
TARGET_VELOCITY_MM_S = (3.0, 80.0)

#: Calibres of the subcutaneous vessels this toolbox was written for, in mm.
CUTANEOUS_DIAMETER_MM = (0.1, 0.5)

#: The longest straight run through the slab that is still modelled, in mm.
#: A vessel lying almost in the plane of the slab would otherwise have an
#: unbounded path. This is a modelling cap standing in for the extent of the
#: imaged region, not a physical quantity, so it is stated rather than hidden.
DEFAULT_MAX_PATH_MM = 60.0


@dataclass(frozen=True)
class Vessel:
    """A straight vessel crossing the slab at an angle.

    ``angle_deg`` is measured between the vessel axis and the slab normal: 0
    crosses the slab head-on, 90 lies in its plane. ``velocity_mm_s`` is the
    mean speed over the lumen, so the centreline moves at twice this.
    """

    velocity_mm_s: float
    diameter_mm: float = 0.2
    angle_deg: float = 0.0

    def __post_init__(self) -> None:
        if self.velocity_mm_s < 0:
            raise ValueError("velocity cannot be negative")
        if self.diameter_mm <= 0:
            raise ValueError("diameter must be positive")
        if not 0 <= self.angle_deg < 90:
            raise ValueError("angle to the slab normal must be in [0, 90) degrees")

    def path_mm(self, slab_mm: float, max_path_mm: float = DEFAULT_MAX_PATH_MM) -> float:
        """Return the distance travelled inside a slab of this thickness."""
        straight = slab_mm / np.cos(np.radians(self.angle_deg))
        return float(min(straight, max_path_mm))

    def dwell_ms(self, slab_mm: float, speed_mm_s: float | None = None,
                 max_path_mm: float = DEFAULT_MAX_PATH_MM) -> float:
        """Return how long blood at ``speed_mm_s`` stays inside the slab.

        Defaults to the vessel's mean speed. Pass a shell speed to get that
        shell's dwell time. Stationary blood never leaves, so this is infinite.
        """
        speed = self.velocity_mm_s if speed_mm_s is None else speed_mm_s
        if speed <= 0:
            return float("inf")
        return 1000.0 * self.path_mm(slab_mm, max_path_mm) / speed

    def with_values(self, **changes) -> "Vessel":
        """Return a copy with some properties replaced."""
        return replace(self, **changes)

    def __str__(self) -> str:
        return (f"{self.diameter_mm * 1000:g} um vessel at {self.velocity_mm_s:g} mm/s, "
                f"{self.angle_deg:g}° to the slab normal")


def laminar_shells(count: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the speeds and weights of equal-area shells across a laminar lumen.

    The returned speeds are multiples of the vessel's mean speed, running from
    fast at the centre to nearly stationary at the wall. The weights are the
    fraction of the cross-section each shell occupies, and sum to one.

    Because the shells are chosen to have equal area, every weight is ``1 /
    count`` and the speed of shell ``k`` has the closed form ``2 - (2k + 1) /
    count``. One shell reproduces plug flow at the mean speed, which is useful
    for showing what the laminar profile costs.
    """
    if count < 1:
        raise ValueError("need at least one shell")
    index = np.arange(count, dtype=float)
    speed_factor = 2.0 - (2.0 * index + 1.0) / count
    weight = np.full(count, 1.0 / count)
    return speed_factor, weight

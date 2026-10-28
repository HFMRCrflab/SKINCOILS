"""What you set on the scanner: the sequence, and how k-space is covered.

Two objects. :class:`Protocol` is the spoiled gradient-echo time-of-flight
sequence itself - the knobs that change the signal. :class:`Acquisition` is how
much k-space you collect and how fast, which changes the noise and the clock
but not the signal. They are kept apart because the first decides contrast and
the second decides whether that contrast is worth the time.

Both are frozen. To vary one, make a copy::

    faster = protocol.with_values(tr_ms=20.0)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

__all__ = ["Protocol", "Acquisition"]


@dataclass(frozen=True)
class Protocol:
    """A spoiled gradient-echo time-of-flight sequence.

    ``tone_percent`` ramps the flip angle across the slab: the pulse is weaker
    where blood enters and stronger where it is about to leave, so freshly
    arrived blood is preserved and blood on its way out is harvested. At 70 the
    flip runs from 0.65 to 1.35 of ``flip_deg``. Zero switches the ramp off.

    ``fat_sat`` and ``mtc`` each add a non-selective preparation pulse once per
    repetition, ``prep_delay_ms`` before the excitation. How much of a given
    tissue survives them comes from that tissue's parameters, not from here.
    """

    tr_ms: float = 30.0
    te_ms: float = 5.0
    flip_deg: float = 25.0
    slab_mm: float = 10.0
    tone_percent: float = 0.0
    fat_sat: bool = False
    mtc: bool = False
    prep_delay_ms: float = 8.0

    def __post_init__(self) -> None:
        if self.tr_ms <= 0 or self.te_ms <= 0:
            raise ValueError("TR and TE must be positive")
        if self.te_ms >= self.tr_ms:
            raise ValueError(f"TE ({self.te_ms} ms) must fall inside TR ({self.tr_ms} ms)")
        if not 0 < self.flip_deg < 180:
            raise ValueError(f"flip angle {self.flip_deg} is outside (0, 180) degrees")
        if self.slab_mm <= 0:
            raise ValueError("slab thickness must be positive")
        if self.prep_delay_ms < 0:
            raise ValueError("preparation delay cannot be negative")
        if self.preparations and self.prep_delay_ms >= self.tr_ms:
            raise ValueError("the preparation delay must fall inside TR")

    @property
    def preparations(self) -> tuple[str, ...]:
        """Return which preparation pulses are played, as tissue parameter names."""
        played = []
        if self.fat_sat:
            played.append("fat_sat_surviving")
        if self.mtc:
            played.append("mtc_surviving")
        return tuple(played)

    def flip_at(self, position: float | np.ndarray) -> float | np.ndarray:
        """Return the flip angle at a fractional position through the slab.

        ``position`` runs 0 at the face where blood enters to 1 at the face
        where it leaves. Without a TONE ramp the answer is the same everywhere.
        """
        offset = self.tone_percent / 100.0 * (np.asarray(position, dtype=float) - 0.5)
        return np.maximum(self.flip_deg * (1.0 + offset), 1e-6)

    def with_values(self, **changes) -> "Protocol":
        """Return a copy with some settings replaced."""
        return replace(self, **changes)

    def __str__(self) -> str:
        bits = [f"TR {self.tr_ms:g} ms", f"TE {self.te_ms:g} ms", f"{self.flip_deg:g}°",
                f"slab {self.slab_mm:g} mm"]
        if self.tone_percent:
            bits.append(f"TONE {self.tone_percent:g} %")
        if self.fat_sat:
            bits.append("fat sat")
        if self.mtc:
            bits.append("MTC")
        return ", ".join(bits)


@dataclass(frozen=True)
class Acquisition:
    """How k-space is covered: resolution, field of view, speed.

    This decides the noise and the clock. The signal model never looks at it.

    ``bandwidth_hz_per_px`` fixes how long one line of k-space takes to read,
    which is the only part of the repetition that collects signal. Everything
    else in TR is dead time as far as signal-to-noise is concerned, which is
    why a long TR is penalised by :meth:`relative_snr` divided by scan time.
    """

    in_plane_mm: float = 0.25
    partition_mm: float = 0.25
    phase_fov_mm: float = 60.0
    bandwidth_hz_per_px: float = 200.0
    acceleration: float = 1.0
    g_factor: float = 1.0
    partial_fourier: float = 1.0

    def __post_init__(self) -> None:
        if min(self.in_plane_mm, self.partition_mm, self.phase_fov_mm) <= 0:
            raise ValueError("resolutions and field of view must be positive")
        if self.bandwidth_hz_per_px <= 0:
            raise ValueError("bandwidth must be positive")
        if self.acceleration < 1 or self.g_factor < 1:
            raise ValueError("acceleration and g-factor are at least 1")
        if not 0.5 <= self.partial_fourier <= 1.0:
            raise ValueError("partial Fourier fraction must be between 0.5 and 1")

    @property
    def readout_ms(self) -> float:
        """Return how long one line of k-space takes to read."""
        return 1000.0 / self.bandwidth_hz_per_px

    @property
    def voxel_mm3(self) -> float:
        """Return the voxel volume."""
        return self.in_plane_mm**2 * self.partition_mm

    def partitions(self, protocol: Protocol) -> int:
        """Return how many partitions cover the slab."""
        return max(1, math.ceil(protocol.slab_mm / self.partition_mm))

    def lines(self, protocol: Protocol) -> int:
        """Return how many phase-encoding lines are actually acquired per partition."""
        full = self.phase_fov_mm / self.in_plane_mm
        return max(1, math.ceil(full * self.partial_fourier / self.acceleration))

    def scan_time_s(self, protocol: Protocol) -> float:
        """Return the time the acquisition takes."""
        return (protocol.tr_ms * 1e-3) * self.lines(protocol) * self.partitions(protocol)

    def relative_snr(self, protocol: Protocol) -> float:
        """Return signal-to-noise per unit signal, in arbitrary but consistent units.

        Proportional to the voxel volume and to the square root of the total
        time actually spent reading k-space, and reduced by the g-factor of the
        parallel-imaging reconstruction. It is a ratio between protocols, never
        an absolute number: that would need the coil's sensitivity, which this
        toolbox does not model.
        """
        samples = self.lines(protocol) * self.partitions(protocol)
        return self.voxel_mm3 * math.sqrt(samples * self.readout_ms * 1e-3) / self.g_factor

    def with_values(self, **changes) -> "Acquisition":
        """Return a copy with some settings replaced."""
        return replace(self, **changes)

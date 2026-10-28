"""How longitudinal magnetisation moves. No MR sequence knowledge lives here.

Under ideal spoiling the transverse magnetisation is destroyed at the end of
every repetition, so the only thing carried from one repetition to the next is
the longitudinal component ``m``. Every event that acts on it - relaxing for a
while, being tipped by an RF pulse, being partly destroyed by a preparation
pulse - changes it by a scaling and a shift::

    m  ->  gain * m + offset

That is the whole content of this module. Two consequences follow from it and
are the reason the rest of the toolbox is fast and short:

* A chain of events is itself one such map, so a whole repetition collapses to
  a single ``Step`` (:meth:`Step.then`).
* Repeating that map forever lands on its fixed point (:attr:`Step.settled`),
  which is what stationary tissue does, and repeating it ``n`` times has a
  closed form (:meth:`Step.repeated`), which is what blood that entered the
  slab ``n`` pulses ago has done.

Every quantity may be a plain float or a NumPy array. Arrays broadcast, so a
whole ensemble of tissue parameter draws is propagated in one go.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Step", "relaxation", "excitation", "saturation", "identity", "readout"]

_UNIT_GAIN = 1e-12  # below this, a gain of 1 is treated as exactly 1


@dataclass(frozen=True)
class Step:
    """One event acting on longitudinal magnetisation, as ``m -> gain * m + offset``.

    Parameters may be scalars or arrays of any shape that broadcast together.
    """

    gain: float | np.ndarray
    offset: float | np.ndarray

    def __call__(self, m: float | np.ndarray) -> float | np.ndarray:
        """Apply the step to magnetisation ``m``."""
        return self.gain * m + self.offset

    def then(self, following: "Step") -> "Step":
        """Return the single step equivalent to this one followed by ``following``."""
        return Step(following.gain * self.gain, following.gain * self.offset + following.offset)

    @property
    def settled(self) -> float | np.ndarray:
        """Return the magnetisation left unchanged by this step, its fixed point.

        Stationary tissue reaches this after enough repetitions. A step with a
        gain of one has no unique fixed point and raises instead.
        """
        shortfall = 1.0 - np.asarray(self.gain, dtype=float)
        if np.any(np.abs(shortfall) < _UNIT_GAIN):
            raise ValueError(
                "this step leaves magnetisation unchanged (gain = 1), so it has no steady state"
            )
        return self.offset / shortfall

    def repeated(self, n: int | np.ndarray) -> "Step":
        """Return the single step equivalent to applying this one ``n`` times.

        Uses the closed form of the geometric sum, so the cost does not grow
        with ``n``. A gain of one, where that form divides by zero, is handled
        separately: the offsets simply add up.
        """
        gain = np.asarray(self.gain, dtype=float)
        offset = np.asarray(self.offset, dtype=float)
        count = np.asarray(n, dtype=float)
        shortfall = 1.0 - gain
        unit = np.abs(shortfall) < _UNIT_GAIN
        compounded = gain**count
        with np.errstate(divide="ignore", invalid="ignore"):
            accumulated = np.where(unit, offset * count,
                                   offset * (1.0 - compounded) / np.where(unit, 1.0, shortfall))
        return Step(compounded, accumulated)


def relaxation(duration_ms: float | np.ndarray, t1_ms: float | np.ndarray,
               equilibrium: float | np.ndarray = 1.0) -> Step:
    """Return the step for longitudinal recovery towards ``equilibrium`` for a duration."""
    retained = np.exp(-np.asarray(duration_ms, dtype=float) / np.asarray(t1_ms, dtype=float))
    return Step(retained, equilibrium * (1.0 - retained))


def excitation(flip_deg: float | np.ndarray) -> Step:
    """Return the step for an RF pulse that tips magnetisation by ``flip_deg``."""
    return Step(np.cos(np.radians(flip_deg)), 0.0)


def saturation(surviving: float | np.ndarray) -> Step:
    """Return the step for a preparation pulse leaving ``surviving`` of the magnetisation.

    ``surviving`` is a fraction: 1.0 for a pulse that does nothing to this
    tissue, 0.0 for one that destroys its longitudinal magnetisation outright.
    """
    return Step(surviving, 0.0)


def identity() -> Step:
    """Return the step that changes nothing."""
    return Step(1.0, 0.0)


def readout(m: float | np.ndarray, flip_deg: float | np.ndarray,
            te_ms: float | np.ndarray, t2star_ms: float | np.ndarray) -> float | np.ndarray:
    """Return the signal observed when magnetisation ``m`` is tipped and read at ``te_ms``.

    The tipped fraction is ``sin(flip)`` and what survives to the echo is
    ``exp(-TE / T2*)``.
    """
    tipped = np.sin(np.radians(flip_deg))
    return m * tipped * np.exp(-np.asarray(te_ms, dtype=float)
                               / np.asarray(t2star_ms, dtype=float))

"""Independent implementations of what the package computes, written from scratch.

Nothing in ``skinsim`` imports this, and this imports nothing from ``skinsim``.
That is the point: a fast path checked against itself is not checked at all, so
these are deliberately slow, literal translations of the textbook expressions
and of a spin marched one excitation at a time.

If a test fails between here and the package, work out which one is wrong before
changing either.
"""
from __future__ import annotations

import math

__all__ = ["ernst_steady_state", "spoiled_gre_signal", "ernst_angle_deg",
           "marched_blood_signal", "saturated_steady_state"]


def ernst_steady_state(tr_ms: float, t1_ms: float, flip_deg: float, m0: float = 1.0) -> float:
    """Return steady-state Mz of spoiled gradient echo, from the textbook expression."""
    e1 = math.exp(-tr_ms / t1_ms)
    return m0 * (1.0 - e1) / (1.0 - e1 * math.cos(math.radians(flip_deg)))


def spoiled_gre_signal(tr_ms: float, te_ms: float, t1_ms: float, t2star_ms: float,
                       flip_deg: float, m0: float = 1.0) -> float:
    """Return the measured steady-state signal of spoiled gradient echo."""
    return (ernst_steady_state(tr_ms, t1_ms, flip_deg, m0)
            * math.sin(math.radians(flip_deg)) * math.exp(-te_ms / t2star_ms))


def ernst_angle_deg(tr_ms: float, t1_ms: float) -> float:
    """Return the flip angle that maximises steady-state signal."""
    return math.degrees(math.acos(math.exp(-tr_ms / t1_ms)))


def saturated_steady_state(tr_ms: float, t1_ms: float, flip_deg: float, m0: float,
                           residual: float, delay_ms: float, n_cycles: int = 4000) -> float:
    """Return the steady state with one saturation pulse per repetition, by brute force.

    No closed form here on purpose: this just runs the cycle until it stops
    changing, which is the definition the package replaces with a fixed point.
    """
    cos_a = math.cos(math.radians(flip_deg))
    e_rest = math.exp(-(tr_ms - delay_ms) / t1_ms)
    e_delay = math.exp(-delay_ms / t1_ms)
    mz = m0
    for _ in range(n_cycles):
        mz = mz * cos_a                                  # excite
        mz = m0 * (1.0 - e_rest) + e_rest * mz           # relax for the rest of TR
        mz = mz * residual                               # saturation pulse
        mz = m0 * (1.0 - e_delay) + e_delay * mz         # relax over the delay
    return mz


def marched_blood_signal(tr_ms: float, te_ms: float, t1_ms: float, t2star_ms: float,
                         flip_deg: float, m0: float, n_excitations: int) -> list[float]:
    """Return the signal of a fresh blood parcel at each of its first excitations.

    Entry ``k`` is what the parcel gives at its ``k``-th excitation, counting the
    first one at entry as ``k = 0``. A plain loop, no algebra.
    """
    cos_a, sin_a = math.cos(math.radians(flip_deg)), math.sin(math.radians(flip_deg))
    decay = math.exp(-te_ms / t2star_ms)
    e1 = math.exp(-tr_ms / t1_ms)
    out, mz = [], m0
    for _ in range(n_excitations):
        out.append(mz * sin_a * decay)
        mz = m0 * (1.0 - e1) + e1 * cos_a * mz
    return out

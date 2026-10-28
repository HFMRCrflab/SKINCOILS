"""SkinSim: what a time-of-flight sequence can see in small, slow-flowing vessels.

Time-of-flight angiography works because blood arriving in the imaged slab has
not yet been saturated by the sequence, while the tissue around it has. In the
brain that is easy: blood moves fast, crosses the slab in a fraction of a
second and leaves before the pulses catch up. In skin the vessels are a few
hundred micrometres across and the blood moves at millimetres per second, so it
may sit in the slab long enough to be saturated like everything else - and the
surrounding fat is bright to begin with.

This toolbox answers whether, and under what settings, anything is left to see.

A first look::

    from skinsim import Protocol, Vessel, tissues, model

    draws = tissues.sample(["blood", "fat"], draws=256, seed=1)
    protocol = Protocol(tr_ms=30, te_ms=5, flip_deg=25, slab_mm=10, fat_sat=True)
    vessel = Vessel(velocity_mm_s=8.0, diameter_mm=0.2)

    print(model.contrast(draws["blood"], draws["fat"], protocol, vessel).mean())

Choosing settings rather than checking one::

    from skinsim import Acquisition, optimise

    acquisition = Acquisition(in_plane_mm=0.25, partition_mm=0.25, phase_fov_mm=60)
    score = optimise.efficiency_score(draws["blood"], draws["fat"], vessel, acquisition)
    best = optimise.optimise(protocol,
                             bounds={"flip_deg": (3, 50), "tr_ms": (12, 80),
                                     "slab_mm": (4, 30), "tone_percent": (0, 90)},
                             switches=["fat_sat", "mtc"], score=score)
    print(best.summary())

Where things live:

``magnetisation``
    How longitudinal magnetisation moves. No MR sequence knowledge.
``tissues``
    Tissue properties at 3 T as intervals, with sources, and the sampler.
``protocol``
    What you set on the scanner: the sequence, and how k-space is covered.
``geometry``
    The vessel's path through the slab, dwell time, laminar shells.
``model``
    Signals and the contrast between them.
``optimise``
    Grid sweeps and numerical search over protocol settings.
``plotting``
    A small shared style, so figures look like each other.

Every result is a spread, not a number, because every tissue parameter is an
interval rather than a value. Reduce it however the question demands - the
median for a typical answer, a low percentile for a cautious one - but reduce
it deliberately.
"""
from __future__ import annotations

from . import geometry, magnetisation, model, optimise, plotting, protocol, tissues
from .geometry import Vessel, laminar_shells
from .model import (blood_profile, blood_signal, cnr, contrast, efficiency, repetition,
                    stationary_mean, stationary_signal)
from .protocol import Acquisition, Protocol
from .tissues import LIBRARY, Range, Tissue, TissueValues

__version__ = "2.0.0"

__all__ = [
    "Acquisition", "LIBRARY", "Protocol", "Range", "Tissue", "TissueValues", "Vessel",
    "blood_profile", "blood_signal", "cnr", "contrast", "efficiency", "laminar_shells",
    "repetition", "stationary_mean", "stationary_signal",
    "geometry", "magnetisation", "model", "optimise", "plotting", "protocol", "tissues",
    "__version__",
]

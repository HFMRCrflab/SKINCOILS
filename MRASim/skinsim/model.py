"""Signals, and the contrast between them. This is where the pieces meet.

A repetition of the sequence is built once, as a single map on longitudinal
magnetisation (:func:`repetition`). Everything else follows from applying it:

* Stationary tissue has had that map applied forever, so it sits at the map's
  fixed point (:func:`stationary_signal`).
* Blood that entered the slab arrives fully relaxed and has the map applied
  once per repetition until it leaves, so its signal depends on where along
  its path you look (:func:`blood_profile`).

The walk along the vessel is done in fractional steps, which is exact when the
flip angle is uniform and a fine approximation when a TONE ramp makes it vary.
Stationary blood costs nothing special: its dwell time is infinite, the map is
applied infinitely often, and the closed form lands exactly on the fixed point.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from .geometry import DEFAULT_MAX_PATH_MM, Vessel, laminar_shells
from .magnetisation import Step, excitation, readout, relaxation, saturation
from .protocol import Acquisition, Protocol
from .tissues import TissueValues

__all__ = ["repetition", "stationary_signal", "stationary_mean", "blood_profile",
           "blood_signal", "contrast", "blood_fraction", "apparent_contrast", "cnr",
           "efficiency", "detectable", "ROSE_CRITERION"]

DEFAULT_SHELLS = 8
DEFAULT_SAMPLES = 65

#: The classical threshold for picking a feature out of noise.
ROSE_CRITERION = 4.0


def repetition(tissue: TissueValues, protocol: Protocol,
               flip_deg: float | np.ndarray) -> Step:
    """Return the single map describing one repetition of the sequence for a tissue.

    The repetition starts just before the excitation and ends just before the
    next one, so the magnetisation it carries is always the magnetisation about
    to be tipped. The order of events is: excite, relax for the rest of TR
    minus the preparation delay, play any preparation pulses, relax for the
    delay.
    """
    delay = protocol.prep_delay_ms if protocol.preparations else 0.0
    step = excitation(flip_deg)
    step = step.then(relaxation(protocol.tr_ms - delay, tissue.t1_ms, tissue.spin_density))
    for parameter in protocol.preparations:
        step = step.then(saturation(getattr(tissue, parameter)))
    if delay:
        step = step.then(relaxation(delay, tissue.t1_ms, tissue.spin_density))
    return step


def stationary_signal(tissue: TissueValues, protocol: Protocol,
                      position: float | np.ndarray = 0.5) -> np.ndarray:
    """Return the steady-state signal of tissue that does not move.

    ``position`` only matters under a TONE ramp, which makes the flip angle and
    therefore the background depend on depth through the slab.
    """
    flip = protocol.flip_at(position)
    settled = repetition(tissue, protocol, flip).settled
    return readout(settled, flip, protocol.te_ms, tissue.t2star_ms)


def stationary_mean(tissue: TissueValues, protocol: Protocol,
                    samples: int = DEFAULT_SAMPLES) -> np.ndarray:
    """Return the stationary signal averaged across the slab.

    Without a TONE ramp this is the same as :func:`stationary_signal`. With one
    it is not, and the difference is easy to forget: the ramp is usually
    discussed as something done to blood, but stationary tissue sees it too.
    """
    if protocol.tone_percent == 0:
        return stationary_signal(tissue, protocol)
    across = np.linspace(0.0, 1.0, samples)[:, None]
    return np.mean(stationary_signal(tissue, protocol, across), axis=0)


def blood_profile(blood: TissueValues, protocol: Protocol, vessel: Vessel,
                  shells: int = DEFAULT_SHELLS, samples: int = DEFAULT_SAMPLES,
                  max_path_mm: float = DEFAULT_MAX_PATH_MM) -> tuple[np.ndarray, np.ndarray]:
    """Return the blood signal at each fractional position along the path through the slab.

    The returned positions run 0 where blood enters to 1 where it leaves. The
    returned signal has shape ``(samples, draws)``, already averaged over the
    laminar shells by cross-sectional area.

    Set ``shells=1`` to get plug flow at the mean speed, which is the right way
    to see what the laminar profile costs.
    """
    positions = np.linspace(0.0, 1.0, samples)
    speed_factor, weight = laminar_shells(shells)
    total: np.ndarray | None = None

    for factor, share in zip(speed_factor, weight):
        dwell_ms = vessel.dwell_ms(protocol.slab_mm, vessel.velocity_mm_s * factor, max_path_mm)
        # Blood that never leaves has been here forever, so it sits at the steady
        # state everywhere rather than arriving fresh at the entry face.
        stagnant = not np.isfinite(dwell_ms)
        magnetisation = np.asarray(blood.spin_density, dtype=float)
        trace = []
        for index, position in enumerate(positions):
            flip = protocol.flip_at(position)
            if stagnant:
                magnetisation = repetition(blood, protocol, flip).settled
            trace.append(readout(magnetisation, flip, protocol.te_ms, blood.t2star_ms))
            if index + 1 < samples:
                step_fraction = positions[index + 1] - position
                midpoint = position + 0.5 * step_fraction
                pulses = step_fraction * dwell_ms / protocol.tr_ms
                cycle = repetition(blood, protocol, protocol.flip_at(midpoint))
                magnetisation = cycle.repeated(pulses)(magnetisation)
        contribution = share * np.asarray(trace, dtype=float)
        total = contribution if total is None else total + contribution

    return positions, total


def blood_signal(blood: TissueValues, protocol: Protocol, vessel: Vessel,
                 reduce: Callable[[np.ndarray], np.ndarray] | None = None,
                 **profile_options) -> np.ndarray:
    """Return the blood signal averaged along its path through the slab.

    Pass ``reduce=lambda s: s.max(axis=0)`` for the brightest point instead, or
    ``reduce=lambda s: s[0]`` for the signal where blood enters.
    """
    _, signal = blood_profile(blood, protocol, vessel, **profile_options)
    if reduce is None:
        return np.mean(signal, axis=0)
    return reduce(signal)


def contrast(blood: TissueValues, background: TissueValues, protocol: Protocol,
             vessel: Vessel, **profile_options) -> np.ndarray:
    """Return blood signal minus background signal.

    A difference, never a ratio. Once a background is well suppressed a ratio
    against it says more about how close the denominator got to zero than about
    whether anything is visible.
    """
    return (blood_signal(blood, protocol, vessel, **profile_options)
            - stationary_mean(background, protocol))


def blood_fraction(vessel: Vessel, acquisition: Acquisition) -> float:
    """Return the fraction of a voxel that the vessel's cross-section fills.

    The vessel is taken to run along one voxel axis, so what matters is its
    circular cross-section against the voxel's in-plane face. A vessel wider
    than the voxel fills it and the fraction stops at 1.
    """
    filled = (np.pi / 4.0) * vessel.diameter_mm**2 / acquisition.in_plane_mm**2
    return float(min(1.0, filled))


def apparent_contrast(blood: TissueValues, background: TissueValues, protocol: Protocol,
                      vessel: Vessel, acquisition: Acquisition,
                      **profile_options) -> np.ndarray:
    """Return the contrast a voxel actually reports, diluted by partial volume.

    A 200 um vessel in a 250 um voxel fills under a third of it, and the voxel
    reports a blend of vessel and surroundings. The contrast that reaches the
    image is the true contrast times that fraction, which is why resolution is
    worth more than its noise cost until the voxel is comparable to the vessel.
    """
    return (blood_fraction(vessel, acquisition)
            * contrast(blood, background, protocol, vessel, **profile_options))


def cnr(blood: TissueValues, background: TissueValues, protocol: Protocol,
        vessel: Vessel, acquisition: Acquisition, partial_volume: bool = True,
        sensitivity: float = 1.0, **profile_options) -> np.ndarray:
    """Return the contrast-to-noise ratio of the whole acquisition.

    Partial volume is included by default, because what a voxel reports is what
    can be detected. Pass ``partial_volume=False`` to compare the tissues
    themselves rather than their appearance in a voxel.

    ``sensitivity`` is the one number no Bloch simulation can supply: the
    signal-to-noise a unit signal would have in a 1 mm3 voxel after one second,
    which depends on the receive coil. It defaults to 1.0, so by default every
    number out of here is a ratio between protocols rather than an absolute. If
    you set it from a measurement, say so when you quote the result.
    """
    difference = (apparent_contrast(blood, background, protocol, vessel, acquisition,
                                    **profile_options) if partial_volume
                  else contrast(blood, background, protocol, vessel, **profile_options))
    return difference * acquisition.relative_snr(protocol) * sensitivity


def efficiency(blood: TissueValues, background: TissueValues, protocol: Protocol,
               vessel: Vessel, acquisition: Acquisition, **cnr_options) -> np.ndarray:
    """Return contrast-to-noise per square root of scan time.

    **This is the quantity to optimise.** Raw contrast rewards any change that
    widens the gap regardless of cost, and a longer repetition almost always
    does: relaxation recovery is pure gain until you pay for the time it takes.
    Dividing by the square root of the scan time prices that time, so a
    protocol that wins only by running longer does not win here.
    """
    seconds = acquisition.scan_time_s(protocol)
    return cnr(blood, background, protocol, vessel, acquisition, **cnr_options) \
        / np.sqrt(seconds)


def detectable(blood: TissueValues, background: TissueValues, protocol: Protocol,
               vessel: Vessel, acquisition: Acquisition, sensitivity: float,
               threshold: float = ROSE_CRITERION, **cnr_options) -> np.ndarray:
    """Return whether each draw clears the Rose criterion for being seen.

    ``sensitivity`` is required rather than defaulting, because a detection
    verdict is meaningless without an absolute noise scale. Supply it from a
    measurement of the coil you intend to use.
    """
    return cnr(blood, background, protocol, vessel, acquisition,
               sensitivity=sensitivity, **cnr_options) >= threshold

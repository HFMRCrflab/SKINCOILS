"""Search the protocol space: grid sweeps and numerical optimisation.

This is what the toolbox is for. Everything else exists so that a protocol can
be scored, and this module turns scoring into choosing.

Two ways in. :func:`sweep` evaluates a grid and hands back the whole surface,
which is what you want when you need to see the shape of the problem and put it
in a figure. :func:`optimise` searches continuously and hands back one answer,
which is what you want when there are more than two knobs to turn.

Both take a *score*: any callable mapping a :class:`~skinsim.protocol.Protocol`
to a single number. The factories here build the usual ones, and because the
tissue parameters are drawn as a spread rather than fixed, every score has to
say how it reduces that spread to one number. The default is the median. Pass
``statistic=lambda values: np.percentile(values, 5)`` to optimise for the
unlucky end of the parameter ranges instead, which is a different and often
wiser question.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import differential_evolution

from .geometry import Vessel
from .model import contrast, efficiency
from .protocol import Acquisition, Protocol
from .tissues import TissueValues

__all__ = ["Score", "Sweep", "Best", "contrast_score", "efficiency_score", "sweep", "optimise"]

#: A score is any callable turning a protocol into a single number.
Score = Callable[[Protocol], float]

Statistic = Callable[[np.ndarray], float]


def _reduce(values: np.ndarray, statistic: Statistic) -> float:
    """Reduce a spread of draws to one number, passing through anything non-finite."""
    values = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(values)):
        values = values[np.isfinite(values)]
        if values.size == 0:
            return float("nan")
    return float(statistic(values))


def contrast_score(blood: TissueValues, background: TissueValues, vessel: Vessel,
                   statistic: Statistic = np.median, absolute: bool = False,
                   **profile_options) -> Score:
    """Return a score giving the blood-to-background contrast of a protocol.

    Use this when scan time is fixed or irrelevant. If it is neither, use
    :func:`efficiency_score`: maximising raw contrast will happily spend an
    hour to gain a percent.

    **On the sign.** Contrast is a signed difference, and blood is not always
    the brighter of the two - against unsuppressed fat it is usually darker.
    Maximising a signed quantity that is negative chases the value closest to
    zero, which is the least visible vessel rather than the most. Set
    ``absolute=True`` to maximise the magnitude instead, which is the right
    question whenever either polarity would do.
    """

    def score(protocol: Protocol) -> float:
        value = _reduce(contrast(blood, background, protocol, vessel, **profile_options),
                        statistic)
        return abs(value) if absolute else value

    return score


def efficiency_score(blood: TissueValues, background: TissueValues, vessel: Vessel,
                     acquisition: Acquisition, statistic: Statistic = np.median,
                     absolute: bool = False, **profile_options) -> Score:
    """Return a score giving contrast-to-noise per square root of scan time.

    The same warning about the sign applies as in :func:`contrast_score`, and
    bites harder: dividing a negative contrast by the square root of a growing
    scan time makes it less negative, so an unguarded search in a regime where
    blood is the darker tissue will run the repetition time to its upper bound
    and report that as the answer.
    """

    def score(protocol: Protocol) -> float:
        value = _reduce(
            efficiency(blood, background, protocol, vessel, acquisition, **profile_options),
            statistic)
        return abs(value) if absolute else value

    return score


@dataclass(frozen=True)
class Sweep:
    """The result of evaluating a score over a grid of protocol settings."""

    base: Protocol
    axes: Mapping[str, np.ndarray]
    values: np.ndarray

    @property
    def names(self) -> tuple[str, ...]:
        """Return the swept setting names, in the order of the value array's axes."""
        return tuple(self.axes)

    def best(self) -> tuple[dict[str, float], float]:
        """Return the settings giving the highest score, and that score."""
        if np.all(np.isnan(self.values)):
            raise ValueError("every point in the sweep was invalid")
        flat = int(np.nanargmax(self.values))
        where = np.unravel_index(flat, self.values.shape)
        settings = {name: float(self.axes[name][index])
                    for name, index in zip(self.names, where)}
        return settings, float(self.values[where])

    def best_protocol(self) -> Protocol:
        """Return the base protocol updated with the best settings found."""
        settings, _ = self.best()
        return self.base.with_values(**settings)

    def summary(self) -> str:
        """Return a one-line description of the best point."""
        settings, value = self.best()
        knobs = ", ".join(f"{k} {v:g}" for k, v in settings.items())
        valid = int(np.count_nonzero(~np.isnan(self.values)))
        return (f"best {value:.4g} at {knobs} "
                f"({valid}/{self.values.size} points valid)")


def sweep(base: Protocol, axes: Mapping[str, Sequence[float]], score: Score) -> Sweep:
    """Evaluate ``score`` over the cartesian product of the given settings.

    ``axes`` maps protocol field names to the values to try. Combinations that
    do not make a valid protocol - a TR shorter than the echo time, say - are
    scored as ``nan`` rather than raising, so a sweep across a sensible range
    is never derailed by its own corners.
    """
    if not axes:
        raise ValueError("a sweep needs at least one axis")
    grids = {name: np.asarray(values, dtype=float) for name, values in axes.items()}
    shape = tuple(len(values) for values in grids.values())
    out = np.full(shape, np.nan)

    for where in itertools.product(*(range(n) for n in shape)):
        settings = {name: float(values[index])
                    for (name, values), index in zip(grids.items(), where)}
        try:
            candidate = base.with_values(**settings)
        except ValueError:
            continue
        out[where] = score(candidate)
    return Sweep(base, grids, out)


@dataclass(frozen=True)
class Best:
    """The outcome of an optimisation."""

    protocol: Protocol
    score: float
    settings: dict[str, float | bool]
    evaluations: int

    def summary(self) -> str:
        """Return a one-line description."""
        knobs = ", ".join(f"{k} {v:g}" if isinstance(v, (int, float)) and not isinstance(v, bool)
                          else f"{k} {v}" for k, v in self.settings.items())
        return f"score {self.score:.4g} at {knobs} ({self.evaluations} evaluations)"


def optimise(base: Protocol, bounds: Mapping[str, tuple[float, float]], score: Score,
             switches: Iterable[str] = (), constraint: Callable[[Protocol], bool] | None = None,
             seed: int = 0, iterations: int = 60, population: int = 16) -> Best:
    """Search for the protocol settings that score highest.

    ``bounds`` maps protocol field names to ``(low, high)`` limits, searched
    continuously. ``switches`` names boolean fields - ``fat_sat``, ``mtc`` -
    which cannot be searched continuously, so every combination of them gets
    its own search and the best is returned. That makes "which preparation,
    and with what settings" a single call.

    ``constraint`` rejects protocols that are otherwise valid: pass
    ``lambda p: acquisition.scan_time_s(p) <= 300`` to stay inside five
    minutes. Rejected and invalid protocols score as minus infinity, so the
    search walks away from them.

    The search is deterministic for a given ``seed``.
    """
    if not bounds:
        raise ValueError("optimisation needs at least one bounded setting")
    names = list(bounds)
    limits = [tuple(float(v) for v in bounds[name]) for name in names]
    switch_names = list(switches)
    calls = 0

    def penalty(settings: dict) -> float:
        nonlocal calls
        calls += 1
        try:
            candidate = base.with_values(**settings)
        except ValueError:
            return np.inf
        if constraint is not None and not constraint(candidate):
            return np.inf
        value = score(candidate)
        return np.inf if not np.isfinite(value) else -float(value)

    best: Best | None = None
    for combination in itertools.product(*([(False, True)] * len(switch_names))):
        fixed = dict(zip(switch_names, combination))

        def cost(vector: np.ndarray, fixed=fixed) -> float:
            return penalty({**fixed, **dict(zip(names, (float(v) for v in vector)))})

        result = differential_evolution(cost, limits, seed=seed, maxiter=iterations,
                                        popsize=population, polish=True, tol=1e-8)
        if not np.isfinite(result.fun):
            continue
        settings = {**fixed, **dict(zip(names, (float(v) for v in result.x)))}
        candidate = Best(base.with_values(**settings), -float(result.fun), settings, calls)
        if best is None or candidate.score > best.score:
            best = candidate

    if best is None:
        raise ValueError("no valid protocol was found inside the given bounds and constraint")
    return Best(best.protocol, best.score, best.settings, calls)

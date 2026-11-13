"""How deep a method still sees a target: the depth limit, solved for instead of assumed.

``evaluate`` answers "can this method see the target at this depth?". The inverse question is the one that shapes a
survey plan: *how deep can the target be before the method loses it?*

    >>> from detectability import depth_limit
    >>> limit = depth_limit({"shape": "sphere", "size": 20, "material": "air", "host": "limestone"},
    ...                     {"latitude": 39.75, "longitude": 37.02, "date": "2026-07-01"}, method="gravity")
    >>> print(limit)

``depth_top`` on the target is ignored, and may be left out entirely: it is what the search solves for. If the shape
also needs ``depth_bottom``, the body keeps its thickness as it moves down.

The search makes no assumption that signal-to-noise falls smoothly with depth, because it does not always: the GPR
antenna frequency, when left to the default, is chosen from the target depth, so the response can step. A coarse
geometric scan brackets the **shallowest** crossing of the threshold and a bisection refines it. The shallowest
crossing is the honest answer — beyond it the method can no longer be relied on, whatever happens deeper.

Each limit costs ``steps`` plus about twenty forward models. That is milliseconds for gravity, magnetics and GPR.
For ERT, where a single forward model inverts a mesh and takes tens of seconds, a limit takes tens of minutes; pass a
narrow ``search`` range and a small ``steps`` if you ask for one at all.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from . import detect
from .survey import NEEDS_BOTTOM, Survey, Target, _as, evaluate

__all__ = ["Limit", "depth_limit"]

SEARCH = (0.5, 2000.0)      # m below ground: shallow enough for utilities, deep enough for an ore body
STEPS = 24                  # coarse scan points, geometric over the search range
REFINE = 20                 # bisections; takes a 24-point scan of 0.5-2000 m below 0.1 % in depth


@dataclass
class Limit:
    """The greatest depth at which ``method`` still reaches ``threshold`` for a given target."""

    method: str
    threshold: float
    depth: float | None             # m to the top of the target; None when the method never reaches the threshold
    snr_shallowest: float | None    # signal-to-noise at the shallowest depth searched, for context
    search: tuple[float, float]
    beyond_search: bool = False     # still detected at the deepest depth searched: the limit is at least that
    available: bool = True
    reason: str = ""

    def __str__(self) -> str:
        if not self.available:
            return f"{self.method}: not assessed ({self.reason})"
        if self.depth is None:
            return (f"{self.method}: never reaches S/N {self.threshold:g}; "
                    f"S/N {self.snr_shallowest:.3g} even at {self.search[0]:g} m")
        at_least = "at least " if self.beyond_search else ""
        return f"{self.method}: detectable (S/N >= {self.threshold:g}) down to {at_least}{self.depth:.3g} m"


def _at(target: Target, depth: float) -> Target:
    """The same body, moved so its top sits at ``depth``; thickness kept where the shape needs a bottom."""
    moved = replace(target, depth_top=depth)
    if target.shape in NEEDS_BOTTOM:
        thickness = (target.depth_bottom - target.depth_top) if target.depth_bottom else max(target.size or 1.0, 1.0)
        moved.depth_bottom = depth + thickness
    return moved


def depth_limit(target, survey=None, method: str = "gravity", threshold: float = detect.CLEAR,
                search: tuple[float, float] = SEARCH, steps: int = STEPS, refine: int = REFINE) -> Limit:
    """Greatest depth at which ``method`` still reaches ``threshold`` for ``target`` under ``survey``.

    ``threshold`` defaults to the clearly-detectable signal-to-noise (5); pass ``detect.MARGINAL`` for the lower one.
    The result's ``depth`` is None when the method never reaches the threshold, even at the shallowest depth searched.
    """
    shallow, deep = float(search[0]), float(search[1])
    if not 0 < shallow < deep:
        raise ValueError(f"search must be (shallow, deep) with 0 < shallow < deep; got {search!r}")
    if steps < 2:
        raise ValueError("steps must be at least 2")

    if isinstance(target, dict) and "depth_top" not in target:
        target = {**target, "depth_top": shallow}   # it is what the search solves for; any value will do
    t, s = _as(Target, target), _as(Survey, survey)
    t.validate()

    def snr(depth: float) -> tuple[float, bool, str]:
        result = evaluate(_at(t, depth), s, methods=[method]).results[0]
        if not result.available:
            return 0.0, False, result.reason
        return result.assessment.snr, True, ""

    first, ok, why = snr(shallow)
    if not ok:
        return Limit(method, threshold, None, None, (shallow, deep), available=False, reason=why)
    if first < threshold:
        return Limit(method, threshold, None, first, (shallow, deep))

    # Coarse scan for the shallowest crossing, then bisect inside that bracket.
    ratio = (deep / shallow) ** (1 / (steps - 1))
    lo, lo_depth = shallow, shallow
    for i in range(1, steps):
        probe = shallow * ratio ** i
        value, ok, _ = snr(probe)
        if not ok or value < threshold:
            hi = probe
            break
        lo_depth = probe
    else:
        return Limit(method, threshold, deep, first, (shallow, deep), beyond_search=True)

    lo = lo_depth
    for _ in range(refine):
        mid = (lo * hi) ** 0.5
        value, ok, _ = snr(mid)
        if ok and value >= threshold:
            lo = mid
        else:
            hi = mid
    return Limit(method, threshold, (lo * hi) ** 0.5, first, (shallow, deep))

"""Online concurrency control for a single GPU shared by policy replicas."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from math import isfinite
from typing import Protocol


@dataclass(frozen=True)
class Config:
    k: int = 12
    g: int = 3

    def __post_init__(self) -> None:
        if type(self.k) is not int or type(self.g) is not int or not 1 <= self.g <= self.k:
            raise ValueError("worker and replica counts must be integers with 1 <= g <= k")


@dataclass(frozen=True)
class Thresholds:
    wait_low: float = 0.25
    wait_high: float = 0.60
    sm_low: float = 0.60
    sm_high: float = 0.85
    slowdown_low: float = 1.10
    slowdown_high: float = 1.30

    def __post_init__(self) -> None:
        if not 0 <= self.wait_low < self.wait_high <= 1:
            raise ValueError("worker wait thresholds must be ordered fractions")
        if not 0 <= self.sm_low < self.sm_high <= 1:
            raise ValueError("SM activity thresholds must be ordered fractions")
        if not 1 <= self.slowdown_low < self.slowdown_high or not isfinite(self.slowdown_high):
            raise ValueError("contention thresholds must be finite, ordered and at least one")


def _ratio(total: float, exposure: float, *, bounded: bool) -> float | None:
    if not all(isfinite(x) and x >= 0 for x in (total, exposure)):
        raise ValueError("window counters must be finite and nonnegative")
    if bounded and total > exposure:
        raise ValueError("window fraction exceeds one")
    if exposure == 0:
        if total:
            raise ValueError("a positive total needs a positive exposure")
        return None
    return total / exposure


@dataclass(frozen=True)
class Window:
    """Deltas for one settled configuration. Zero exposure means unavailable."""

    worker_wait_s: float
    worker_live_s: float
    sm_active_s: float
    gpu_sampled_s: float
    matched_service_s: float = 0.0
    matched_solo_s: float = 0.0

    @classmethod
    def from_samples(
        cls,
        *,
        workers: Iterable[tuple[float, float]],
        gpu: Iterable[tuple[float, float]],
        batches: Iterable[tuple[str, float]] = (),
        solo_by_shape: Mapping[str, float] | None = None,
    ) -> "Window":
        """Aggregate (wait, live), (SM fraction, seconds), and (shape, service seconds)."""
        wait = live = sm_active = sampled = service = solo = 0.0
        for wait_s, live_s in workers:
            _ratio(wait_s, live_s, bounded=True)
            wait += wait_s
            live += live_s
        for activity, seconds in gpu:
            if not 0 <= activity <= 1 or not isfinite(seconds) or seconds <= 0:
                raise ValueError("GPU samples require a fraction and positive duration")
            sm_active += activity * seconds
            sampled += seconds
        for shape, seconds in batches:
            if not isfinite(seconds) or seconds < 0:
                raise ValueError("service time must be finite and nonnegative")
            if solo_by_shape is None or shape not in solo_by_shape:
                continue
            reference = solo_by_shape[shape]
            if not isfinite(reference) or reference <= 0:
                raise ValueError("solo service time must be finite and positive")
            service += seconds
            solo += reference
        return cls(wait, live, sm_active, sampled, service, solo)


@dataclass(frozen=True)
class Decision:
    target: Config
    reason: str
    wait: float | None
    sm: float | None
    slowdown: float | None


def decide(
    current: Config,
    window: Window,
    *,
    limits: Config = Config(32, 4),
    thresholds: Thresholds = Thresholds(),
) -> Decision:
    """Return one bounded, single-axis change, or hold the current configuration."""
    if current.k > limits.k or current.g > limits.g:
        raise ValueError("current configuration exceeds the supplied limits")
    wait = _ratio(window.worker_wait_s, window.worker_live_s, bounded=True)
    sm = _ratio(window.sm_active_s, window.gpu_sampled_s, bounded=True)
    slowdown = _ratio(window.matched_service_s, window.matched_solo_s, bounded=False)
    k, g = current.k, current.g
    target, reason = current, "deadband"
    if current.g > 1 and slowdown is not None and slowdown > thresholds.slowdown_high:
        target, reason = Config(k, g - 1), "replica_contention"
    elif wait is None or sm is None:
        reason = "missing_wait_or_sm"
    elif wait > thresholds.wait_high and sm > thresholds.sm_high:
        if k > g:
            target, reason = Config(k - 1, g), "saturated_with_waiters"
        else:
            reason = "worker_floor"
    elif wait < thresholds.wait_low and sm < thresholds.sm_low:
        if k < limits.k:
            target, reason = Config(k + 1, g), "underfed_gpu"
        else:
            reason = "worker_limit"
    elif wait > thresholds.wait_high and sm < thresholds.sm_low:
        if g >= min(k, limits.g):
            reason = "replica_limit"
        elif g == 1 or (slowdown is not None and slowdown < thresholds.slowdown_low):
            target, reason = Config(k, g + 1), "inference_wait_with_gpu_headroom"
        elif slowdown is None:
            reason = "missing_solo_reference"
    return Decision(target, reason, wait, sm, slowdown)


class Runtime(Protocol):
    def configure(self, target: Config) -> bool:
        """Apply atomically at a safe boundary. False means the old config is intact."""
        ...

    def observe(self, current: Config) -> Window | None:
        """Collect the next settled work window, or return None once the run is over."""
        ...


def run(
    runtime: Runtime,
    *,
    limits: Config = Config(32, 4),
    thresholds: Thresholds = Thresholds(),
) -> Config:
    """Drive a runtime until evaluation completes. Return its last applied config."""
    current = Config()
    if current.k > limits.k or current.g > limits.g:
        raise ValueError("the (12, 3) prior exceeds the supplied limits")
    if not runtime.configure(current):
        raise RuntimeError("the runtime rejected the initial (12, 3) configuration")
    while (window := runtime.observe(current)) is not None:
        decision = decide(current, window, limits=limits, thresholds=thresholds)
        if decision.target != current and runtime.configure(decision.target):
            current = decision.target
    return current

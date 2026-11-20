"""Online concurrency control for a single GPU shared by policy replicas.

This module intentionally uses only the Python standard library so hosts can
embed the threshold controller without pulling in NumPy or ML frameworks.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from math import isfinite
from typing import Protocol


def _is_nonneg_real(value: float) -> bool:
    """Return True for finite, nonnegative int/float values while rejecting bool."""
    return not isinstance(value, bool) and isfinite(value) and value >= 0


@dataclass(frozen=True)
class Config:
    """Active rollout topology: ``k`` environment workers sharing ``g`` policy replicas."""

    k: int = 12
    g: int = 3

    def __post_init__(self) -> None:
        if type(self.k) is not int or type(self.g) is not int or not 1 <= self.g <= self.k:
            raise ValueError("worker and replica counts must be integers with 1 <= g <= k")


@dataclass(frozen=True)
class Thresholds:
    """Deadband cutoffs for the single-step rule controller (``baseline/threshold-v1``)."""

    wait_low: float = 0.25
    wait_high: float = 0.60
    sm_low: float = 0.60
    sm_high: float = 0.85
    slowdown_low: float = 1.10
    slowdown_high: float = 1.30

    def __post_init__(self) -> None:
        if any(
            isinstance(x, bool)
            for x in (
                self.wait_low,
                self.wait_high,
                self.sm_low,
                self.sm_high,
                self.slowdown_low,
                self.slowdown_high,
            )
        ):
            raise ValueError("thresholds must be numeric values, not booleans")
        if not 0 <= self.wait_low < self.wait_high <= 1:
            raise ValueError("worker wait thresholds must be ordered fractions")
        if not 0 <= self.sm_low < self.sm_high <= 1:
            raise ValueError("SM activity thresholds must be ordered fractions")
        if not 1 <= self.slowdown_low < self.slowdown_high or not isfinite(self.slowdown_high):
            raise ValueError("contention thresholds must be finite, ordered and at least one")


def _ratio(total: float, exposure: float, *, bounded: bool) -> float | None:
    """Compute ``total / exposure``, returning ``None`` when ``exposure == 0``."""
    if not (_is_nonneg_real(total) and _is_nonneg_real(exposure)):
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
    """Accumulated counters for one settled configuration window.

    Zero exposure (``worker_live_s == 0``, ``gpu_sampled_s == 0``, or
    ``matched_solo_s == 0``) marks the corresponding ratio signal as unavailable.
    """

    worker_wait_s: float
    worker_live_s: float
    sm_active_s: float
    gpu_sampled_s: float
    matched_service_s: float = 0.0
    matched_solo_s: float = 0.0

    def __post_init__(self) -> None:
        for value in (
            self.worker_wait_s,
            self.worker_live_s,
            self.sm_active_s,
            self.gpu_sampled_s,
            self.matched_service_s,
            self.matched_solo_s,
        ):
            if not _is_nonneg_real(value):
                raise ValueError("window counters must be finite and nonnegative")

    @classmethod
    def from_samples(
        cls,
        *,
        workers: Iterable[tuple[float, float]],
        gpu: Iterable[tuple[float, float]],
        batches: Iterable[tuple[str, float]] = (),
        solo_by_shape: Mapping[str, float] | None = None,
    ) -> "Window":
        """Aggregate raw ``(wait_s, live_s)``, ``(sm_fraction, duration_s)``, and batch timings."""
        wait = live = sm_active = sampled = service = solo = 0.0
        for wait_s, live_s in workers:
            _ratio(wait_s, live_s, bounded=True)
            wait += wait_s
            live += live_s
        for activity, seconds in gpu:
            if (
                not _is_nonneg_real(activity)
                or activity > 1
                or not _is_nonneg_real(seconds)
                or seconds <= 0
            ):
                raise ValueError("GPU samples require a fraction and positive duration")
            sm_active += activity * seconds
            sampled += seconds
        for shape, seconds in batches:
            if not _is_nonneg_real(seconds):
                raise ValueError("service time must be finite and nonnegative")
            if solo_by_shape is None or shape not in solo_by_shape:
                continue
            reference = solo_by_shape[shape]
            if not _is_nonneg_real(reference) or reference <= 0:
                raise ValueError("solo service time must be finite and positive")
            service += seconds
            solo += reference
        return cls(wait, live, sm_active, sampled, service, solo)


@dataclass(frozen=True)
class Decision:
    """Controller output for one window along with computed telemetry ratios and reason tag."""

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
    """Return one bounded single-axis change, or hold ``current``.

    Rule priority:
    1. Back off replica count ``g - 1`` on severe multi-replica service contention
       (checked before ``wait``/``sm`` so contention relief still works without DCGM).
    2. Hold if ``wait`` or ``sm`` is unavailable.
    3. Back off worker count ``k - 1`` when both worker wait and GPU SM are saturated.
    4. Scale up worker count ``k + 1`` when both worker wait and GPU SM are low.
    5. Scale up replica count ``g + 1`` when workers wait on inference while GPU SM
       still has headroom and existing replicas are not contending.
    """
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
    """Host interface for blocking window-by-window concurrency control."""

    def configure(self, target: Config) -> bool:
        """Apply ``target`` atomically at a safe boundary; ``False`` leaves old config intact."""
        ...

    def observe(self, current: Config) -> Window | None:
        """Collect the next settled work window under ``current``, or ``None`` when finished."""
        ...


def run(
    runtime: Runtime,
    *,
    limits: Config = Config(32, 4),
    thresholds: Thresholds = Thresholds(),
) -> Config:
    """Drive ``runtime`` from ``Config(12, 3)`` until evaluation completes."""
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

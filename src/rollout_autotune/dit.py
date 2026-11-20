"""CPU-only step-fused DIT policy with non-blocking hot-scaling support.

This module depends only on NumPy and the Python standard library so the online
controller executes purely on CPU with zero GPU VRAM or SM interference.
"""

import argparse
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor, wait as wait_for_futures
from dataclasses import asdict
import json
import math
import subprocess
import sys
import time

import numpy as np

from . import Config


# 15 discrete (dk, dg) actions: index 0 MUST remain HOLD (0, 0).
ACTIONS = (
    (0, 0),
    (1, 0),
    (-1, 0),
    (2, 0),
    (-2, 0),
    (4, 0),
    (-4, 0),
    (8, 0),
    (-8, 0),
    (0, 1),
    (0, -1),
    (0, 2),
    (0, -2),
    (4, 1),
    (-4, -1),
)
NUM_ACTIONS = len(ACTIONS)
REPLICA_UP_MASK = np.array([dg > 0 for _, dg in ACTIONS], dtype=bool)
REPLICA_CHANGE_MASK = np.array([dg != 0 for _, dg in ACTIONS], dtype=bool)
SHRINK_MASK = np.array(
    [dk <= 0 and dg <= 0 and (dk < 0 or dg < 0) for dk, dg in ACTIONS], dtype=bool
)
_MIB = 1024 * 1024

# Causal Transformer architecture (must match training.model.Transformer).
LAYERS, WIDTH, HEADS, INNER, CONTEXT = 6, 256, 8, 704, 64
HEAD_DIM = WIDTH // HEADS

# Step-fused token layout:
#   [0..12]  normalized state + availability/reset flags (STATE_DIM = 13)
#   [13..27] one-hot previous action (NUM_ACTIONS = 15)
#   [28..29] (switch_cost_ratio, work_normalized_reward)
STATE_DIM = 13
INPUT = STATE_DIM + NUM_ACTIONS + 2
FORMAT = 3
NO_SM_FORMAT = 4
# Full token indices 3 and 9 encode the SM fraction and its availability bit.
NO_SM_COLUMNS = tuple(i for i in range(INPUT) if i not in (3, 9))
REQUIRED_OBSERVE_KEYS = ("wait", "slowdown", "rate", "work_left", "elapsed_s")


def model_inputs(tokens, use_sm):
    """Project full 30-dim step tokens to 28 dims when using an SM-free checkpoint."""
    return tokens if use_sm else tokens[..., NO_SM_COLUMNS]


def features(
    k,
    g,
    wait,
    sm,
    slowdown,
    rate,
    work_left,
    reset,
    limits=Config(32, 4),
    load_s=None,
):
    """Encode normalized state and availability flags (13 dims, bit-aligned with simulator)."""
    values = [
        k / limits.k,  # 0: worker fraction
        g / limits.g,  # 1: replica fraction
        0.0 if wait is None else wait,  # 2: inference wait ratio in [0, 1]
        0.0 if sm is None else sm,  # 3: GPU SM active ratio in [0, 1] (dropped in NO_SM_FORMAT)
        0.0 if slowdown is None else min(slowdown, 8.0) / 4.0,  # 4: clipped service slowdown
        math.log1p(rate) / 6.0,  # 5: log throughput (steps/s)
        math.log1p(work_left) / 12.0,  # 6: log remaining steps
        0.0 if load_s is None else min(math.log1p(load_s) / 4.0, 2.0),  # 7: per-replica load prior
        wait is not None,  # 8: wait available
        sm is not None,  # 9: sm available (dropped in NO_SM_FORMAT)
        slowdown is not None,  # 10: slowdown available
        load_s is not None,  # 11: load_s available
        bool(reset),  # 12: initial window of trajectory
    ]
    return np.asarray(values, dtype=np.float32)


def step_metrics(base_rate, rate, elapsed_s, switch_s):
    """Compute dimensionless ``(switch_cost_ratio, work_normalized_reward)``.

    Reward ``-(base_rate / rate) * (1 + switch_s / elapsed_s)`` measures normalized
    wall time per unit work, unifying fixed-work and fixed-duration windows.
    """
    denom_s = max(float(elapsed_s), 1.0)
    cost_ratio = min(float(switch_s) / denom_s, 4.0)
    ref_rate = max(float(base_rate), 1e-3)
    cur_rate = max(float(rate), 1e-3)
    reward = -min((ref_rate / cur_rate) * (1.0 + float(switch_s) / denom_s), 10.0)
    return cost_ratio, reward


def action_mask(config, limits, rejected):
    """Return boolean legality mask over ``ACTIONS``; ``HOLD`` (index 0) is always legal."""
    mask = np.array(
        [
            1 <= config.k + dk <= limits.k
            and 1 <= config.g + dg <= min(config.k + dk, limits.g)
            and (config.k + dk, config.g + dg) not in rejected
            for dk, dg in ACTIONS
        ],
        dtype=bool,
    )
    mask[0] = True
    return mask


def token(state, previous, cost, reward):
    """Pack ``state``, one-hot ``previous`` action index, ``cost``, and ``reward`` into 1D token."""
    x = np.zeros(INPUT, dtype=np.float32)
    x[:STATE_DIM] = state
    if previous is not None:
        x[STATE_DIM + previous] = 1.0
    x[-2:] = (cost, reward)
    return x


def norm(x, gain):
    """Pre-RMSNorm with eps=1e-6 matching ``torch.nn.RMSNorm``."""
    return x * (1.0 / np.sqrt(np.mean(x * x) + 1e-6)) * gain


class CPUModel:
    """Pure-NumPy causal Transformer actor with preallocated KV cache and sliding-window rebuild."""

    def __init__(self, path):
        with np.load(path, allow_pickle=False) as checkpoint:
            version = int(checkpoint["format_version"])
            if version not in (FORMAT, NO_SM_FORMAT):
                raise ValueError("unsupported policy format")
            self.uses_sm = version == FORMAT
            self.w = {
                k: checkpoint[k].astype(np.float32, copy=False)
                for k in checkpoint.files
                if k != "format_version"
            }
        expected = INPUT if self.uses_sm else len(NO_SM_COLUMNS)
        if self.w["embed.weight"].shape != (WIDTH, expected) or self.w["head.weight"].shape != (
            NUM_ACTIONS,
            WIDTH,
        ):
            raise ValueError("policy feature schema and network shapes disagree")

        self.k_cache = np.empty((LAYERS, HEADS, CONTEXT, HEAD_DIM), dtype=np.float32)
        self.v_cache = np.empty((LAYERS, HEADS, CONTEXT, HEAD_DIM), dtype=np.float32)
        self.pos = 0
        self.history = deque(maxlen=CONTEXT)

        freq = np.float32(10000) ** (-np.arange(0, HEAD_DIM, 2, dtype=np.float32) / HEAD_DIM)
        angles = np.arange(CONTEXT, dtype=np.float32)[:, None] * freq
        self.cos = np.cos(angles)
        self.sin = np.sin(angles)
        self.inv_sqrt_d = np.float32(1.0) / np.sqrt(np.float32(HEAD_DIM))

    def _step(self, x):
        w = self.w
        pos = self.pos
        cos, sin = self.cos[pos], self.sin[pos]

        def rotate(a):
            p, q = a[:, ::2], a[:, 1::2]
            return np.stack((p * cos - q * sin, p * sin + q * cos), axis=-1).reshape(a.shape)

        x = w["embed.weight"] @ x + w["embed.bias"]
        for i in range(LAYERS):
            prefix = f"blocks.{i}."

            def param(name):
                return w[prefix + name + ".weight"]

            q, k, v = (param("qkv") @ norm(x, param("n1"))).reshape(3, HEADS, HEAD_DIM)
            self.k_cache[i, :, pos] = rotate(k)
            self.v_cache[i, :, pos] = v
            keys = self.k_cache[i, :, : pos + 1]
            values = self.v_cache[i, :, : pos + 1]
            scores = np.einsum("hd,htd->ht", rotate(q), keys) * self.inv_sqrt_d
            weights = np.exp(scores - scores.max(axis=1, keepdims=True))
            weights /= weights.sum(axis=1, keepdims=True)
            x = x + param("proj") @ np.einsum("ht,htd->hd", weights, values).ravel()
            gate, up = np.split(param("up") @ norm(x, param("n2")), 2)
            x = x + param("down") @ ((gate / (1 + np.exp(-np.clip(gate, -70, 70)))) * up)
        self.pos = pos + 1
        return w["head.weight"] @ norm(x, w["final.weight"]) + w["head.bias"]

    def logits(self, x):
        """Append one full 30-dim step token and return unmasked 15-action logits."""
        x = model_inputs(x, self.uses_sm)
        self.history.append(x.copy())
        if self.pos == CONTEXT:
            # Recompute retained tokens from pos=0 so evicted context cannot leak via RoPE/KV states.
            self.pos = 0
            for old in list(self.history)[:-1]:
                self._step(old)
        return self._step(x)


def _validate_load_s(load_s):
    if load_s is not None and (
        isinstance(load_s, bool) or not math.isfinite(load_s) or load_s < 0
    ):
        raise ValueError("load_s must be finite and nonnegative or null")


def plan_suite_windows(
    total_episodes: int,
    *,
    window_episodes: int = 50,
    k: int = 12,
    num_tasks: int | None = None,
) -> int:
    """Choose a nearby window count for an episode-major round-robin partition.

    In benchmarks where items are ordered ``episode * num_tasks + task_id`` and partitioned
    by ``ordinal % count == index``, choosing ``gcd(count, num_tasks) == 1`` ensures every
    window samples all tasks evenly instead of aliasing onto a subset of tasks.

    Candidates near ``ceil(total_episodes / window_episodes)`` must also be representable
    as ``ceil(total_episodes / shard_episodes) == count`` so explicit shard counts and
    per-shard episode caps agree. Remaining ties minimize suite-wide tail-wave idle worker
    slots under ``k``, then stay closest to the requested window count and window size.
    """
    if (
        type(total_episodes) is not int
        or type(window_episodes) is not int
        or type(k) is not int
        or total_episodes < 1
        or window_episodes < 1
        or k < 1
    ):
        raise ValueError("total_episodes, window_episodes, and k must be positive integers")
    if num_tasks is not None and (type(num_tasks) is not int or num_tasks < 1):
        raise ValueError("num_tasks must be a positive integer or null")

    if total_episodes <= window_episodes:
        return 1

    base = math.ceil(total_episodes / window_episodes)

    def shard_cost(count: int) -> tuple[int, int, int, int]:
        shorter_size, longer_count = divmod(total_episodes, count)
        shorter_count = count - longer_count
        shard_size = shorter_size + (1 if longer_count else 0)
        unfilled = shorter_count * (-shorter_size % k) + longer_count * (-(shorter_size + 1) % k)
        task_cycle = math.gcd(count, num_tasks) if num_tasks is not None else 1
        return task_cycle, unfilled, abs(count - base), abs(shard_size - window_episodes)

    candidates = range(max(2, base - 2), min(total_episodes, base + 2) + 1)
    valid = (
        count
        for count in candidates
        if math.ceil(total_episodes / math.ceil(total_episodes / count)) == count
    )
    return min(valid, key=shard_cost, default=base)


class Session:
    """Stateful online DIT controller tracking causal history, rejections, and hot-scaling."""

    def __init__(
        self,
        path,
        limits=Config(32, 4),
        seed=7,
        load_s=None,
        initial=Config(12, 3),
    ):
        _validate_load_s(load_s)
        self.model = CPUModel(path)
        self.current = initial
        if limits.k < self.current.k or limits.g < self.current.g:
            raise ValueError("initial configuration exceeds limits")
        self.limits = limits
        self.rng = np.random.default_rng(seed)
        self.rejected = set()
        self.pending = None
        self.previous = None
        self.base_rate = None
        self.load_s = None if load_s is None else float(load_s)
        self.last_accepted_dg = 0
        self.warming_g = None
        self.ready_warmup = None
        self.windows = 0
        self.hold_streak = 0
        self._pending_settled_hold = False

    @property
    def settled(self) -> bool:
        """True once at least two consecutive settled windows held the active configuration."""
        return self.hold_streak >= 2 and self.warming_g is None and self.pending is None

    def notify_warmup(self, accepted: bool, *, load_elapsed_s: float = 0.0) -> None:
        """Report completion of an asynchronous background replica warmup before ``propose()``."""
        if type(accepted) is not bool or self.warming_g is None:
            raise ValueError("expected an active background warmup and boolean status")
        if isinstance(load_elapsed_s, bool) or not math.isfinite(load_elapsed_s) or load_elapsed_s < 0:
            raise ValueError("load_elapsed_s must be finite and nonnegative")
        self.ready_warmup = (accepted, float(load_elapsed_s))

    def _finish_warmup(self) -> bool:
        """Commit a reported background warmup and return whether ``current.g`` increased."""
        if self.ready_warmup is None:
            return False
        ok, bg_s = self.ready_warmup
        target_g = self.warming_g
        observed_cfg = self.current
        self.ready_warmup = None
        self.warming_g = None
        if ok:
            dg_done = max(target_g - observed_cfg.g, 1)
            if bg_s > 0:
                per_rep = bg_s / dg_done
                self.load_s = per_rep if self.load_s is None else 0.5 * self.load_s + 0.5 * per_rep
            self.current = Config(max(observed_cfg.k, target_g), target_g)
            self.hold_streak = 0
        else:
            # Mask all configurations at or above the OOMing (k, target_g) envelope.
            for rk in range(observed_cfg.k, self.limits.k + 1):
                for rg in range(target_g, self.limits.g + 1):
                    if rg <= rk:
                        self.rejected.add((rk, rg))
        return ok

    def propose(
        self,
        *,
        wait,
        sm=None,
        slowdown,
        rate,
        work_left,
        elapsed_s,
        switch_s=0.0,
        load_s=None,
        vram_mask=None,
        vram_pressure: bool = False,
    ):
        """Record a completed window and return the next proposed ``Config``.

        ``elapsed_s`` covers the settled observation; ``switch_s`` is disjoint blocking
        transition time before the window. ``vram_mask`` optionally masks scale-up moves
        predicted to exceed safe GPU VRAM, while ``vram_pressure=True`` marks the active
        envelope infeasible and forces an immediate step-down when feasible.
        Must be followed by :meth:`acknowledge`.
        """
        if self.pending is not None:
            raise RuntimeError("acknowledge the previous proposal first")
        if type(vram_pressure) is not bool:
            raise ValueError("vram_pressure must be a boolean")
        if not self.model.uses_sm:
            sm = None
        for v in (wait, sm):
            if v is not None and (isinstance(v, bool) or not 0 <= v <= 1 or not math.isfinite(v)):
                raise ValueError("expected fractional telemetry or null")
        for v in (rate, work_left, elapsed_s, switch_s, 0.0 if slowdown is None else slowdown):
            if isinstance(v, bool) or not math.isfinite(v) or v < 0:
                raise ValueError("invalid telemetry")
        _validate_load_s(load_s)
        vram_arr = None
        if vram_mask is not None:
            vram_arr = np.asarray(vram_mask, dtype=bool)
            if vram_arr.shape != (NUM_ACTIONS,):
                raise ValueError(f"vram_mask must have shape ({NUM_ACTIONS},)")

        if load_s is not None:
            self.load_s = float(load_s)
        elif self.windows > 0 and self.last_accepted_dg > 0 and switch_s > 0:
            inferred = float(switch_s) / max(self.last_accepted_dg, 1)
            self.load_s = inferred if self.load_s is None else 0.5 * self.load_s + 0.5 * inferred

        if self.base_rate is None and rate > 0:
            self.base_rate = float(rate)

        # Causal attribution: encode the exact (k, g) that served the completed window
        # BEFORE promoting any newly attached background replicas.
        observed_cfg = self.current
        state = features(
            observed_cfg.k,
            observed_cfg.g,
            wait,
            sm,
            slowdown,
            rate,
            work_left,
            self.windows == 0,
            limits=self.limits,
            load_s=self.load_s,
        )
        if self.windows == 0:
            cost_feat, reward_feat = 0.0, 0.0
        else:
            ref = self.base_rate if self.base_rate is not None else max(float(rate), 1.0)
            cost_feat, reward_feat = step_metrics(ref, rate, elapsed_s, switch_s)
        x = token(state, self.previous, cost_feat, reward_feat)
        logits = self.model.logits(x)

        attached = self._finish_warmup()
        base_cfg = self.current
        if vram_pressure:
            eff_g = max(base_cfg.g, self.warming_g or 0)
            for rk in range(base_cfg.k, self.limits.k + 1):
                for rg in range(eff_g, self.limits.g + 1):
                    if rg <= rk:
                        self.rejected.add((rk, rg))

        # Lock dg != 0 while warming in background AND for one settled window right after attach.
        lock_replicas = attached or self.warming_g is not None

        mask = action_mask(base_cfg, self.limits, self.rejected)
        has_telemetry = wait is not None and (not self.model.uses_sm or sm is not None)
        if lock_replicas:
            mask[REPLICA_CHANGE_MASK] = False
            if self.warming_g is not None:
                for i, (dk, _) in enumerate(ACTIONS):
                    if base_cfg.k + dk < self.warming_g:
                        mask[i] = False
                    elif dk > 0 and (base_cfg.k + dk, self.warming_g) in self.rejected:
                        mask[i] = False
        if vram_arr is not None and not vram_pressure:
            mask[~SHRINK_MASK] &= vram_arr[~SHRINK_MASK]
            mask[0] = True

        if not has_telemetry:
            mask[1:] = False
        elif vram_pressure:
            shrink = mask & SHRINK_MASK
            if vram_arr is not None and (shrink & vram_arr).any():
                mask = shrink & vram_arr
            elif shrink.any():
                mask = shrink
            elif attached and self.warming_g is None:
                # At k == g > 1 right after attach, allow replica step-down rather than staying over VRAM.
                emergency = action_mask(base_cfg, self.limits, self.rejected) & SHRINK_MASK
                if emergency.any():
                    mask = emergency
                else:
                    mask[:] = False
                    mask[0] = True
            else:
                mask[:] = False
                mask[0] = True
        elif self.windows == 0 and base_cfg.g > 1 and slowdown is None:
            # Avoid speculative replica increases on cold start without a solo reference.
            mask[REPLICA_UP_MASK] = False

        scores = np.where(mask, logits, -np.inf)
        action = int(np.argmax(scores))
        dk, dg = ACTIONS[action]
        self._pending_settled_hold = (
            action == 0
            and not attached
            and not vram_pressure
            and self.warming_g is None
            and has_telemetry
        )
        self.pending = action, Config(base_cfg.k + dk, base_cfg.g + dg)
        self.windows += 1
        return self.pending[1]

    def acknowledge(self, accepted, *, warming: bool = False):
        """Commit host acceptance/rejection of the pending proposal.

        Pass ``warming=True`` when ``target.g > current.g`` was accepted for asynchronous
        background warmup (so ``k`` updates immediately while ``current.g`` stays unchanged
        until :meth:`notify_warmup` reports completion).
        """
        if type(accepted) is not bool or type(warming) is not bool or self.pending is None:
            raise ValueError("expected a pending proposal and boolean acknowledgement flags")
        action, target = self.pending
        settled_hold = self._pending_settled_hold
        self._pending_settled_hold = False
        if accepted:
            if warming and target.g > self.current.g:
                if self.warming_g is not None:
                    raise RuntimeError("background replica warmup already in progress")
                self.warming_g = target.g
                self.last_accepted_dg = 0
                self.current = Config(max(target.k, self.current.g), self.current.g)
                self.hold_streak = 0
            else:
                self.last_accepted_dg = target.g - self.current.g
                self.current = target
                self.hold_streak = self.hold_streak + 1 if settled_hold else 0
        else:
            self.last_accepted_dg = 0
            self.hold_streak = 0
            if action:
                self.rejected.add((target.k, target.g))
        self.previous = action
        self.pending = None


class HotPoolScaler:
    """Non-blocking background replica scaler with two-layer GPU VRAM guardrails.

    1. Pre-flight total VRAM budget guard: models both resident policy replicas and incoming
       environment workers ``baseline + g_eff * replica_vram + k * worker_vram <= budget_ceiling``
       so between-window ``memory.free`` readings (taken while workers are stopped) cannot mask
       worker + EGL framebuffer OOMs on the next window.
    2. Post-window high-water & fault guard: when observed window peak VRAM crosses
       ``high_water_bytes`` (or a worker suffers a recoverable OOM/EGL fault), marks the
       ``(>=k, >=g)`` envelope rejected and triggers immediate concurrency step-down.
    """

    def __init__(
        self,
        pool,
        session: Session,
        *,
        vram_budget_fraction: float = 0.92,
        vram_min_headroom_bytes: int = 1536 * _MIB,
        vram_high_water_fraction: float = 0.94,
        vram_high_water_headroom_bytes: int = 1024 * _MIB,
        memory_probe=None,
    ):
        self.pool = pool
        self.session = session
        self.vram_budget_fraction = float(vram_budget_fraction)
        self.vram_min_headroom_bytes = int(vram_min_headroom_bytes)
        self.vram_high_water_fraction = float(vram_high_water_fraction)
        self.vram_high_water_headroom_bytes = int(vram_high_water_headroom_bytes)
        self._memory_probe = memory_probe
        self.total_vram_bytes: int | None = None
        self._baseline_used_bytes: int = 0
        self.replica_vram_bytes: int = 0
        self.worker_vram_bytes: int = 0
        if getattr(pool, "adapter", None) is not None and getattr(
            pool.adapter, "load_timeout_s", 90.0
        ) < 90.0:
            pool.adapter.load_timeout_s = 90.0
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._future: Future | None = None
        self._staging = None

    def _num_gpus(self) -> int:
        devices = getattr(self.pool, "gpu_devices", ()) or ()
        return max(len(devices), 1)

    def _query_gpu_memory(self) -> tuple[int, int, int] | None:
        """Return ``(min_total_bytes, max_used_bytes, min_free_bytes)`` across pool GPUs."""
        if self._memory_probe is not None:
            return self._memory_probe()
        if getattr(self.pool, "adapter", None) is None:
            return None
        devices = getattr(self.pool, "gpu_devices", ())
        if not devices:
            return None
        totals, useds, frees = [], [], []
        for dev in devices:
            try:
                res = subprocess.run(
                    [
                        "nvidia-smi",
                        f"--id={dev}",
                        "--query-gpu=memory.total,memory.used,memory.free",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=5,
                )
                if res.returncode == 0:
                    parts = [float(p.strip()) for p in res.stdout.strip().splitlines()[0].split(",")]
                    if len(parts) == 3:
                        totals.append(int(parts[0] * _MIB))
                        useds.append(int(parts[1] * _MIB))
                        frees.append(int(parts[2] * _MIB))
            except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
                continue
        if not totals:
            return None
        return min(totals), max(useds), min(frees)

    def _effective_replica_vram_bytes(self) -> int:
        if self.replica_vram_bytes > 0:
            return self.replica_vram_bytes
        vram_fn = getattr(getattr(self.pool, "adapter", None), "vram_bytes_per_replica", None)
        if callable(vram_fn):
            per_rep = vram_fn()
            if per_rep and per_rep > 0:
                return int(per_rep)
        return 0

    def refresh_replica_vram(self) -> None:
        """Sample between-window idle GPU memory (when 0 workers and 0 staging loads are active)."""
        if self._future is not None or self.pool.size <= 0:
            return
        mem = self._query_gpu_memory()
        if mem is None:
            return
        total_b, used_b, _ = mem
        if total_b > 0:
            self.total_vram_bytes = total_b
        n_gpus = self._num_gpus()
        per_gpu_live = max((self.pool.size + n_gpus - 1) // n_gpus, 1)
        delta = used_b - self._baseline_used_bytes
        if delta > 0:
            self.replica_vram_bytes = max(self.replica_vram_bytes, delta // per_gpu_live)

    def budget_ceiling_bytes(self) -> int | None:
        """Maximum estimated peak VRAM allowed when proposing scale-up actions."""
        if self.total_vram_bytes is None or self.total_vram_bytes <= 0:
            return None
        total = self.total_vram_bytes
        return max(
            int(total * 0.80),
            min(int(total * self.vram_budget_fraction), total - self.vram_min_headroom_bytes),
        )

    def high_water_bytes(self) -> int | None:
        """Observed window peak VRAM threshold above which active configurations step down."""
        if self.total_vram_bytes is None or self.total_vram_bytes <= 0:
            return None
        total = self.total_vram_bytes
        return max(
            int(total * 0.85),
            min(
                int(total * self.vram_high_water_fraction),
                total - self.vram_high_water_headroom_bytes,
            ),
        )

    def estimate_vram_bytes(self, k: int, g: int) -> int | None:
        """Estimate per-GPU peak VRAM bytes for ``(k workers, g replicas)``."""
        rep_b = self._effective_replica_vram_bytes()
        if rep_b <= 0 and self.worker_vram_bytes <= 0:
            return None
        n_gpus = self._num_gpus()
        per_gpu_g = (g + n_gpus - 1) // n_gpus
        per_gpu_k = (k + n_gpus - 1) // n_gpus
        return self._baseline_used_bytes + per_gpu_g * rep_b + per_gpu_k * self.worker_vram_bytes

    def fits_vram_budget(self, k: int, g: int) -> bool:
        """Return False when ``(k, g)`` is predicted to exceed ``budget_ceiling_bytes()``."""
        ceiling = self.budget_ceiling_bytes()
        est = self.estimate_vram_bytes(k, g)
        if ceiling is None or est is None:
            return True
        return est <= ceiling

    def _reject_envelope(self, min_k: int, min_g: int) -> None:
        for rk in range(min_k, self.session.limits.k + 1):
            for rg in range(min_g, self.session.limits.g + 1):
                if rg <= rk:
                    self.session.rejected.add((rk, rg))

    def record_window(
        self,
        config: Config,
        *,
        active_workers: int | None = None,
        peak_vram_bytes: int | None = None,
        warming_g: int | None = None,
        had_fault: bool = False,
        total_vram_bytes: int | None = None,
    ) -> bool:
        """Update per-replica/per-worker VRAM estimates from a completed window and return pressure flag."""
        self.refresh_replica_vram()
        if total_vram_bytes is not None and total_vram_bytes > 0:
            self.total_vram_bytes = int(total_vram_bytes)

        g_res = max(config.g, warming_g or 0)
        k_act = max(int(active_workers if active_workers is not None else config.k), 1)
        rep_b = self._effective_replica_vram_bytes()
        if peak_vram_bytes is not None and peak_vram_bytes > 0 and rep_b > 0:
            n_gpus = self._num_gpus()
            per_gpu_g = (g_res + n_gpus - 1) // n_gpus
            per_gpu_k = max((k_act + n_gpus - 1) // n_gpus, 1)
            worker_total = int(peak_vram_bytes) - (self._baseline_used_bytes + per_gpu_g * rep_b)
            if worker_total > 0:
                per_worker = worker_total // per_gpu_k
                self.worker_vram_bytes = (
                    per_worker
                    if self.worker_vram_bytes <= 0
                    else max(int(0.7 * self.worker_vram_bytes + 0.3 * per_worker), per_worker)
                )

        high_water = self.high_water_bytes()
        over_high_water = (
            peak_vram_bytes is not None
            and high_water is not None
            and int(peak_vram_bytes) > high_water
        )
        pressure = bool(had_fault or over_high_water)
        if pressure:
            self._reject_envelope(config.k, g_res)
        return pressure

    def vram_action_mask(self) -> np.ndarray | None:
        """Build a 15-action boolean VRAM feasibility mask for the upcoming ``Session.propose`` call."""
        if self.budget_ceiling_bytes() is None or (
            self._effective_replica_vram_bytes() <= 0 and self.worker_vram_bytes <= 0
        ):
            return None
        cur = self.session.current
        if self.session.ready_warmup is not None:
            ok, _ = self.session.ready_warmup
            target_g = self.session.warming_g
            base_cfg = Config(max(cur.k, target_g), target_g) if ok and target_g else cur
            in_flight_g = 0
        else:
            base_cfg = cur
            in_flight_g = self.session.warming_g or 0

        mask = np.zeros(NUM_ACTIONS, dtype=bool)
        for i, (dk, dg) in enumerate(ACTIONS):
            ck, cg = base_cfg.k + dk, base_cfg.g + dg
            if 1 <= cg <= ck <= self.session.limits.k and cg <= self.session.limits.g:
                mask[i] = self.fits_vram_budget(ck, max(cg, in_flight_g))
        return mask

    def _has_vram_headroom(self, target: Config, added_replicas: int) -> bool:
        """Check both modeled full-window VRAM budget and live free GPU memory."""
        if not self.fits_vram_budget(target.k, target.g):
            return False
        per_rep = self._effective_replica_vram_bytes()
        if per_rep <= 0:
            return True
        mem = self._query_gpu_memory()
        if mem is None:
            return True
        _, _, free_bytes = mem
        n_gpus = self._num_gpus()
        per_gpu_added = (added_replicas + n_gpus - 1) // n_gpus
        per_gpu_k = (target.k + n_gpus - 1) // n_gpus
        required_bytes = int(per_gpu_added * per_rep * 1.05) + per_gpu_k * self.worker_vram_bytes
        return free_bytes >= required_bytes

    def start_initial(self) -> float:
        """Start the initial replica count synchronously and seed ``session.load_s`` if unset."""
        g0 = self.session.current.g
        existing = self.pool.size
        if existing == 0:
            before = self._query_gpu_memory()
            if before is not None:
                self.total_vram_bytes, self._baseline_used_bytes, _ = before
        t0 = time.monotonic()
        self.pool.scale(g0)
        elapsed = time.monotonic() - t0
        added = max(g0 - existing, 0)
        if added > 0 and elapsed > 0 and self.session.load_s is None:
            self.session.load_s = elapsed / added
        self.refresh_replica_vram()
        return elapsed

    def _spawn_staging(self, replica_ids: tuple[int, ...]) -> tuple[object, float]:
        t0 = time.monotonic()
        staging = type(self.pool)(
            adapter=self.pool.adapter,
            gpu_devices=self.pool.gpu_devices,
            base_port=self.pool.base_port,
            output_dir=self.pool.output_dir,
            ready_timeout_s=self.pool.ready_timeout_s,
            cuda_mps=getattr(self.pool, "cuda_mps", None),
        )
        self._staging = staging
        staging._start_replicas(replica_ids)
        return staging, time.monotonic() - t0

    def poll(self) -> float:
        """Attach completed background replicas without blocking; returns cutover seconds."""
        if self._future is None or not self._future.done():
            return 0.0
        t0 = time.monotonic()
        fut = self._future
        self._future = None
        try:
            staging, load_elapsed_s = fut.result()
            self.pool.contexts.update(staging.contexts)
            self.pool.processes.update(staging.processes)
            staging.contexts.clear()
            staging.processes.clear()
            self._staging = None
            self.session.notify_warmup(True, load_elapsed_s=load_elapsed_s)
        except Exception:
            if self._staging is not None:
                try:
                    self._staging.stop()
                except Exception:
                    pass
                self._staging = None
            self.session.notify_warmup(False, load_elapsed_s=0.0)
        return time.monotonic() - t0

    def finish_warmup(self) -> float:
        """Block until any in-flight background warmup settles and commit it to ``session``.

        Call only at a suite/run boundary when no workers are running and no proposal is pending.
        """
        if self.session.pending is not None:
            raise RuntimeError("acknowledge the previous proposal before finishing warmup")
        if self._future is None and self.session.ready_warmup is None:
            return 0.0
        start = time.monotonic()
        if self._future is not None:
            wait_for_futures([self._future])
            self.poll()
        self.session._finish_warmup()
        return time.monotonic() - start

    def apply(self, target: Config) -> float:
        """Apply a proposed ``Config``; scale-ups launch asynchronously with zero blocking pause."""
        cur = self.session.current
        max_w = self.pool.maximum_workers_per_replica()
        if max_w is not None and target.k > target.g * max_w:
            self.session.acknowledge(False)
            return 0.0

        if target.g > cur.g:
            if self._future is not None or self.session.warming_g is not None:
                self.session.acknowledge(False)
                return 0.0
            new_ids = tuple(r for r in range(target.g) if r not in self.pool.processes)
            if not new_ids:
                self.session.acknowledge(True, warming=False)
                return 0.0
            if not self._has_vram_headroom(target, len(new_ids)):
                min_k = cur.k if not self.fits_vram_budget(cur.k, target.g) else target.k
                self._reject_envelope(min_k, target.g)
                self.session.acknowledge(False)
                return 0.0
            self._future = self._executor.submit(self._spawn_staging, new_ids)
            self.session.acknowledge(True, warming=True)
            return 0.0

        if target.g < cur.g:
            if self._future is not None or self.session.warming_g is not None:
                self.session.acknowledge(False)
                return 0.0
            t0 = time.monotonic()
            self.pool.scale(target.g)
            dt = time.monotonic() - t0
            self.session.acknowledge(True, warming=False)
            return dt

        eff_g = max(target.g, self.session.warming_g or 0)
        if target.k > cur.k and not self.fits_vram_budget(target.k, eff_g):
            self._reject_envelope(target.k, eff_g)
            self.session.acknowledge(False)
            return 0.0

        self.session.acknowledge(True, warming=False)
        return 0.0

    def close(self) -> None:
        """Cleanly cancel or stop any in-flight staging replicas and shut down the executor."""
        if self._future is not None:
            try:
                staging, _ = self._future.result(timeout=5.0)
                staging.stop()
            except Exception:
                if self._staging is not None:
                    try:
                        self._staging.stop()
                    except Exception:
                        pass
            self._future = None
        self._executor.shutdown(wait=False)


def main():
    """Run the DIT controller over line-delimited JSON stdio (supporting optional hot-scaling)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-workers", type=int, default=32)
    parser.add_argument("--max-replicas", type=int, default=4)
    parser.add_argument("--initial-workers", type=int, default=4)
    parser.add_argument("--initial-replicas", type=int, default=1)
    parser.add_argument("--load-s", type=float, default=None)
    args = parser.parse_args()
    session = Session(
        args.weights,
        Config(args.max_workers, args.max_replicas),
        args.seed,
        load_s=args.load_s,
        initial=Config(args.initial_workers, args.initial_replicas),
    )

    def exchange(op, c):
        print(json.dumps(dict(op=op, **asdict(c))), flush=True)
        line = sys.stdin.readline()
        if not line:
            raise EOFError(f"runtime disconnected during {op}")
        return json.loads(line)

    initial_ack = exchange("configure", session.current)
    if type(initial_ack) is not bool:
        raise ValueError("configure requires a JSON boolean acknowledgement")
    if not initial_ack:
        raise RuntimeError(f"initial ({session.current.k},{session.current.g}) refused")
    while (window := exchange("observe", session.current)) is not None:
        if not isinstance(window, dict):
            raise ValueError("observe requires a JSON object or null")
        missing = [k for k in REQUIRED_OBSERVE_KEYS if k not in window]
        if missing:
            raise ValueError(f"observe missing required keys: {', '.join(missing)}")
        if "warmup_accepted" in window and window["warmup_accepted"] is not None:
            session.notify_warmup(
                window["warmup_accepted"],
                load_elapsed_s=window.get("warmup_s", 0.0),
            )
        target = session.propose(
            wait=window["wait"],
            sm=window.get("sm"),
            slowdown=window["slowdown"],
            rate=window["rate"],
            work_left=window["work_left"],
            elapsed_s=window["elapsed_s"],
            switch_s=window.get("switch_s", 0.0),
            load_s=window.get("load_s"),
            vram_pressure=bool(window.get("vram_pressure", False)),
        )
        if target == session.current:
            session.acknowledge(True)
        else:
            ack = exchange("configure", target)
            if ack == "warming" or (isinstance(ack, dict) and ack.get("warming") is True):
                session.acknowledge(True, warming=True)
            elif type(ack) is bool:
                session.acknowledge(ack)
            else:
                raise ValueError("configure requires a JSON boolean or {'warming': true}")


if __name__ == "__main__":
    main()

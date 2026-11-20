"""Vectorized closed-loop throughput surrogate with non-blocking hot-scaling.

State normalization, 15-action ordering, legality masks, and background warmup
handoff rules must stay bit-aligned with ``rollout_autotune.dit``.
"""

import numpy as np

from rollout_autotune.dit import (
    ACTIONS,
    INPUT,
    NUM_ACTIONS,
    REPLICA_CHANGE_MASK,
    REPLICA_UP_MASK,
    STATE_DIM,
)


DELTA = np.asarray(ACTIONS)
# Maps each action index in ACTIONS to its exact opposite (-dk, -dg) move.
REVERSE_ACTION = np.array([0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13])
LENGTH = 32


class Worlds:
    """Parameterized batch of synthetic hardware + policy + environment worlds."""

    def __init__(self, n, rng, *, fixed_limits=False):
        u = lambda a, b: rng.uniform(a, b, n)
        # Span lightweight MLP/ACT policies, medium policies, and heavy 7B+ VLA models.
        tier = rng.random(n)
        light = tier < 0.30
        medium = (tier >= 0.30) & (tier < 0.60)

        self.worker_rate = u(14, 68)
        self.cpu_rate = u(110, 480)
        self.replica_rate = np.where(
            light, u(220, 580), np.where(medium, u(110, 240), u(55, 175))
        )
        self.replica_mem = np.where(
            light, u(0.8, 2.5), np.where(medium, u(2.5, 5.5), u(5.5, 11.0))
        )
        self.load = np.where(
            light, u(0.6, 4.0), np.where(medium, u(4.0, 14.0), u(14.0, 40.0))
        )
        # Span constrained 16-24GB cards, 32-48GB cards, and 80GB data-center GPUs.
        gpu_tier = rng.random(n)
        self.mem_cap = np.where(
            gpu_tier < 0.30, u(16, 26), np.where(gpu_tier < 0.75, u(30, 50), u(58, 82))
        )
        self.gpu_cap = np.where(
            gpu_tier < 0.30, u(110, 280), np.where(gpu_tier < 0.75, u(160, 440), u(280, 720))
        )

        # Background device contention per worker (e.g., GPU EGL offscreen rendering / PCIe).
        has_egl = rng.random(n) < 0.65
        self.egl_penalty = np.where(has_egl, u(0.005, 0.030), 0.0)
        # Inter-replica contention: moderate sharing preserves g / S(g) parallel gains;
        # severe contention (22% of worlds) causes multi-replica throughput inversion.
        heavy_contend = rng.random(n) < 0.22
        self.contention = np.where(heavy_contend, u(0.55, 1.15), u(0.08, 0.40))
        self.contention_exp = np.where(heavy_contend, u(1.25, 1.60), u(0.95, 1.22))

        self.cpu_penalty = u(0.012, 0.075)
        self.cpu_knee = rng.integers(6, 27, n)
        self.worker_mem = u(0.18, 0.60)
        self.block = u(1500, 8500)
        self.drain = u(0.5, 5.0)
        self.cpu_switch = u(0.2, 3.0)
        self.bg_copy_penalty = u(0.04, 0.11)
        self.shard_sigma = u(0.02, 0.15)
        self.timed = rng.random(n) < 0.5
        self.window_s = u(25, 65)
        if fixed_limits:
            self.max_k = np.full(n, 32, dtype=int)
            self.max_g = np.full(n, 4, dtype=int)
        else:
            custom = rng.random(n) < 0.25
            self.max_k = np.where(custom, rng.choice([16, 24, 32, 48, 64], size=n), 32)
            self.max_g = np.where(custom, rng.choice([3, 4, 6, 8], size=n), 4)

    def observe(self, k, g, phase, noise, shard_factor=1.0):
        """Evaluate closed-loop steady-state ``(rate, wait, sm, slowdown)`` at ``(k, g)``."""
        k_eff = np.maximum(k, 1)
        g_eff = np.maximum(g, 1)
        over_knee = np.maximum(k_eff - self.cpu_knee, 0)
        g_extra = np.maximum(g_eff - 1, 0)

        env_per_worker = (
            self.worker_rate
            * shard_factor
            * (1.0 - 0.10 * phase)
            / (1.0 + self.cpu_penalty * over_knee)
        )
        cpu_cap_eff = self.cpu_rate / (1.0 + 0.5 * self.cpu_penalty * over_knee)
        t_env_eff = np.maximum(
            1.0 / np.maximum(env_per_worker, 1e-6), k_eff / np.maximum(cpu_cap_eff, 1e-6)
        )

        # Parallel GPU throughput identity: g replicas at full-load slowdown S(k, g) deliver g * R_1 / S(k, g).
        bg_slow = 1.0 + self.egl_penalty * k_eff
        raw_full_slow = bg_slow * (1.0 + self.contention * (g_extra ** self.contention_exp))
        gpu_compute = np.minimum(self.gpu_cap, g_eff * self.replica_rate / raw_full_slow)
        eff_full_slow = np.maximum(
            raw_full_slow, g_eff * self.replica_rate / np.maximum(gpu_compute, 1e-6)
        )

        # Serial inference latency per worker cycle (even without queueing, a worker awaits its own forward pass).
        t_svc_base = bg_slow / self.replica_rate
        worker_offered = k_eff / np.maximum(t_env_eff + t_svc_base, 1e-6)

        # Smooth L_p bottleneck blend between CPU/worker supply and GPU replica service capacity.
        p = 3.5
        mean_rate = (worker_offered ** (-p) + gpu_compute ** (-p)) ** (-1.0 / p)
        rate = mean_rate * noise
        sm = np.clip(rate / self.gpu_cap, 0.0, 1.0)
        gpu_load = np.clip(mean_rate / np.maximum(gpu_compute, 1e-6), 0.0, 1.0)
        slow = np.maximum(
            1.0,
            (1.0 + (eff_full_slow - 1.0) * (0.45 + 0.55 * gpu_load)) * noise,
        )
        # Worker wait includes both serial policy service time and multi-worker queueing wait.
        wait = np.clip(1.0 - (mean_rate * t_env_eff) / k_eff, 0.0, 1.0)
        return rate, wait, sm, slow


def _window_step(worlds, work_rem, active_rate, *, drain=False):
    """Advance one fixed-work shard or fixed-time window (or drain remaining steps at horizon)."""
    if drain:
        step_work = work_rem
        elapsed_s = np.where(work_rem > 0, work_rem / np.maximum(active_rate, 1e-6), 0.0)
        return step_work, elapsed_s
    work_block = np.minimum(work_rem, worlds.block)
    elapsed_work = work_block / np.maximum(active_rate, 1e-6)
    elapsed_timed = np.minimum(worlds.window_s, work_rem / np.maximum(active_rate, 1e-6))
    elapsed_s = np.where(worlds.timed, elapsed_timed, elapsed_work)
    step_work = np.where(worlds.timed, np.minimum(work_rem, active_rate * elapsed_s), work_block)
    return step_work, elapsed_s


def _feasible_initial(worlds, k, g):
    """Clamp initial ``(k, g)`` so it fits within VRAM and grid bounds."""
    max_fit_g = np.maximum(
        ((worlds.mem_cap - 2.0 * worlds.worker_mem) / worlds.replica_mem).astype(int), 1
    )
    g = np.clip(g, 1, np.minimum(worlds.max_g, max_fit_g))
    max_mem_k = np.maximum(
        ((worlds.mem_cap - g * worlds.replica_mem) / worlds.worker_mem).astype(int), g
    )
    k = np.clip(np.minimum(k, max_mem_k), g, worlds.max_k)
    return k, g


def _reject_above_g(refused, world_indices, min_k, min_g):
    """Mark all configurations ``(k >= min_k, g >= min_g)`` infeasible after a replica OOM."""
    for wi, mk, mg in zip(world_indices, min_k, min_g):
        refused[wi, int(mk) :, int(mg) :] = True


def make_data(worlds, rng, prior=None, *, use_sm=True, hot_scale=True, return_wall_s=False):
    """Generate two independent rollouts per world and pack into context buffers.

    When ``prior is None``, mixes three suboptimal behavioral explorers:
      - style 0: randomized multi-scale threshold controller with epsilon exploration;
      - style 1: sticky uniform random explorer;
      - style 2: empirical hill-climber that reverses degrading moves and repeats gains.
    When ``prior == "threshold"`` or a callable policy, evaluates that controller deterministically.
    """
    n = len(worlds.block)
    slots = 2 * LENGTH
    x = np.zeros((n, slots, INPUT), np.float32)
    masks = np.zeros((n, slots, NUM_ACTIONS), bool)
    actions = np.zeros((n, slots), np.int64)
    rewards = np.zeros((n, slots), np.float32)
    terminal = np.zeros((n, slots), bool)
    wall_s = np.zeros((2, n), np.float64)
    units = rng.integers(8, 24, (2, n))
    ix = np.arange(n)
    seg0_used = np.zeros(n, dtype=int)

    for segment in range(2):
        offset = seg0_used * segment
        total_work = units[segment] * worlds.block
        if prior is None:
            fixed = rng.random(n) < 0.45
        else:
            fixed = rng.random(n) < 0.50
        rand_k = (rng.uniform(0.12, 0.75, n) * worlds.max_k).astype(int).clip(min=2)
        rand_g = np.minimum(rand_k, rng.integers(1, 5, n))
        k = np.where(fixed, 12, np.minimum(rand_k, worlds.max_k))
        g = np.where(fixed, 3, np.minimum(rand_g, worlds.max_g))
        k, g = _feasible_initial(worlds, k, g)

        old = np.full(n, -1)
        prev_rate = np.zeros(n, np.float64)
        cost_feat = np.zeros(n, np.float32)
        reward_feat = np.zeros(n, np.float32)
        refused = np.zeros((n, 65, 9), bool)

        # Hot-scaling state: background replica warmup overlaps with active evaluation.
        warming_g = np.zeros(n, dtype=int)
        warmup_rem_s = np.zeros(n, np.float64)
        warmup_total_s = np.zeros(n, np.float64)
        warming_will_oom = np.zeros(n, bool)
        ready_g = np.zeros(n, dtype=int)
        ready_oom = np.zeros(n, bool)

        has_load = rng.random(n) < 0.65
        obs_load_s = np.where(has_load, worlds.load * rng.uniform(0.85, 1.15, n), 0.0)

        style_draw = rng.random(n)
        style = np.where(style_draw < 0.50, 0, np.where(style_draw < 0.80, 1, 2))
        perturb = (rng.random(n) < 0.85) & (prior is None)
        wl = np.where(perturb, rng.uniform(0.15, 0.35, n), 0.25)
        wh = np.where(perturb, rng.uniform(0.45, 0.78, n), 0.60)
        sl = np.where(perturb, rng.uniform(0.45, 0.70, n), 0.60)
        sh = np.where(perturb, rng.uniform(0.75, 0.95, n), 0.85)
        slow_hi = np.where(perturb, rng.uniform(1.25, 3.20, n), 1.30)
        slow_lo = np.where(perturb, np.minimum(slow_hi - 0.15, rng.uniform(1.05, 1.90, n)), 1.10)
        world_has_slow = np.ones(n, bool) if prior is not None else (rng.random(n) < 0.75)

        shard0 = np.clip(np.exp(rng.normal(0.0, worlds.shard_sigma)), 0.75, 1.30)
        active_rate, wait, sm, slow = worlds.observe(
            k, g, 0.0, rng.uniform(0.97, 1.03, n), shard_factor=shard0
        )
        w0, elapsed_0 = _window_step(worlds, total_work, active_rate)
        work_rem = np.maximum(total_work - w0, 0.0)
        wall_s[segment] += elapsed_0
        base_rate = np.maximum(active_rate, 1e-3)
        prev_rate[:] = active_rate
        finished = work_rem <= 1e-3

        for t in range(LENGTH):
            has_wait = np.ones(n, bool) if prior is not None else (rng.random(n) < 0.97)
            has_sm = np.ones(n, bool) if prior is not None else (rng.random(n) < 0.90)
            has_slow = world_has_slow & (
                np.ones(n, bool) if (prior is not None or t > 0) else (rng.random(n) < 0.85)
            )
            step_noise = rng.uniform(0.97, 1.03, n)
            shard_t = np.clip(np.exp(rng.normal(0.0, worlds.shard_sigma)), 0.75, 1.30)
            explore_u = rng.random((n, NUM_ACTIONS)) if prior is None else None
            explore_pick = rng.random(n) if prior is None else None
            step_jump_u = rng.random(n) if prior is None else None

            alive = ~finished
            if not alive.any():
                continue
            idx = ix[alive]
            slot = offset[idx] + t

            # Build token from the exact (k, g) under which the completed window ran (matching Session.propose).
            wait_obs = np.where(has_wait, wait, 0.0)
            sm_obs = np.where(has_sm, sm, 0.0)
            slow_obs = np.where(has_slow, np.minimum(slow, 8.0) / 4.0, 0.0)
            load_obs = np.where(has_load, np.minimum(np.log1p(obs_load_s) / 4.0, 2.0), 0.0)
            state_cols = np.column_stack(
                (
                    k / worlds.max_k,
                    g / worlds.max_g,
                    wait_obs,
                    sm_obs,
                    slow_obs,
                    np.log1p(active_rate) / 6.0,
                    np.log1p(work_rem) / 12.0,
                    load_obs,
                    has_wait,
                    has_sm,
                    has_slow,
                    has_load,
                    np.full(n, t == 0),
                )
            )
            assert state_cols.shape[1] == STATE_DIM
            tok = np.column_stack(
                (
                    state_cols,
                    old[:, None] == np.arange(NUM_ACTIONS),
                    cost_feat,
                    reward_feat,
                )
            ).astype(np.float32)

            # Promote any finished background replica warmups right before computing legal actions
            # (matching Session.notify_warmup + Session.propose handoff).
            lock_replicas = np.zeros(n, bool)
            if hot_scale:
                succeeded_ready = alive & (ready_g > 0) & (~ready_oom)
                failed_ready = alive & (ready_g > 0) & ready_oom
                if failed_ready.any():
                    _reject_above_g(
                        refused,
                        ix[failed_ready],
                        k[failed_ready].clip(0, 64),
                        ready_g[failed_ready].clip(0, 8),
                    )
                if succeeded_ready.any():
                    dg_done = np.maximum(ready_g - g, 1)
                    per_rep = warmup_total_s / dg_done
                    obs_load_s = np.where(
                        succeeded_ready,
                        np.where(has_load, 0.5 * obs_load_s + 0.5 * per_rep, per_rep),
                        obs_load_s,
                    )
                    has_load |= succeeded_ready
                    g = np.where(succeeded_ready, ready_g, g)
                    k = np.where(succeeded_ready, np.maximum(k, g), k)
                    lock_replicas |= succeeded_ready
                ready_g = np.where(alive, 0, ready_g)
                ready_oom = np.where(alive, False, ready_oom)
                lock_replicas |= alive & (warming_g > 0)

            ck = k[:, None] + DELTA[:, 0]
            cg = g[:, None] + DELTA[:, 1]
            mask = (
                (ck >= 1)
                & (ck <= worlds.max_k[:, None])
                & (cg >= 1)
                & (cg <= worlds.max_g[:, None])
                & (cg <= ck)
            )
            mask &= ~refused[ix[:, None], ck.clip(0, 64), cg.clip(0, 8)]
            if lock_replicas.any():
                mask[np.ix_(lock_replicas, REPLICA_CHANGE_MASK)] = False
                in_flight = alive & (warming_g > 0)
                if in_flight.any():
                    wg = warming_g[:, None].clip(0, 8)
                    conflict = in_flight[:, None] & (DELTA[None, :, 0] > 0) & refused[
                        ix[:, None], ck.clip(0, 64), np.broadcast_to(wg, ck.shape)
                    ]
                    mask &= ~conflict
            mask[:, 0] = True
            hold_only = (~has_wait) | (use_sm & (~has_sm))
            mask[hold_only, 1:] = False
            cold_no_ref = (t == 0) & (g > 1) & (~has_slow) & (~hold_only)
            mask[np.ix_(cold_no_ref, REPLICA_UP_MASK)] = False

            if prior == "threshold":
                a = np.select(
                    [
                        has_slow & (g > 1) & (slow > 1.30),
                        has_wait & has_sm & (wait > 0.60) & (sm > 0.85),
                        has_wait & has_sm & (wait < 0.25) & (sm < 0.60),
                        has_wait
                        & has_sm
                        & (wait > 0.60)
                        & (sm < 0.60)
                        & ((g == 1) | (has_slow & (slow < 1.10))),
                    ],
                    [10, 2, 1, 9],
                    default=0,
                )
                a = np.where(mask[ix, a], a, 0)
            elif prior is None:
                k_up = np.where(
                    step_jump_u < 0.30,
                    1,
                    np.where(step_jump_u < 0.55, 3, np.where(step_jump_u < 0.82, 5, 7)),
                )
                k_dn = np.where(
                    step_jump_u < 0.30,
                    2,
                    np.where(step_jump_u < 0.55, 4, np.where(step_jump_u < 0.82, 6, 8)),
                )
                g_up = np.where(step_jump_u < 0.55, 9, np.where(step_jump_u < 0.80, 11, 13))
                g_dn = np.where(step_jump_u < 0.55, 10, np.where(step_jump_u < 0.80, 12, 14))
                thresh_a = np.select(
                    [
                        has_slow & (g > 1) & (slow > slow_hi),
                        has_wait & has_sm & (wait > wh) & (sm > sh),
                        has_wait & has_sm & (wait < wl) & (sm < sl),
                        has_wait
                        & has_sm
                        & (wait > wh)
                        & (sm < sl)
                        & ((g == 1) | (has_slow & (slow < slow_lo))),
                    ],
                    [g_dn, k_dn, k_up, g_up],
                    default=0,
                )
                thresh_a = np.where(mask[ix, thresh_a], thresh_a, 0)
                rand_a = np.where(mask, explore_u, -1.0).argmax(axis=1)
                moved = old > 0
                degraded = moved & (active_rate < 0.97 * prev_rate)
                improved = moved & (active_rate > 1.04 * prev_rate)
                rev_a = REVERSE_ACTION[np.maximum(old, 0)]
                rev_a = np.where(mask[ix, rev_a], rev_a, 0)
                cont_a = np.where(mask[ix, np.maximum(old, 0)], np.maximum(old, 0), 0)
                hill_a = np.where(
                    degraded,
                    rev_a,
                    np.where(
                        improved & (explore_pick < 0.55),
                        cont_a,
                        np.where(explore_pick < 0.20, rand_a, 0),
                    ),
                )
                a_style0 = np.where(explore_pick < 0.28, rand_a, thresh_a)
                a_style1 = np.where(explore_pick < 0.50, rand_a, 0)
                a = np.where(style == 0, a_style0, np.where(style == 1, a_style1, hill_a))
                a = np.where(mask[ix, a], a, 0)
            else:
                x[idx, slot] = tok[idx]
                a = np.zeros(n, dtype=int)
                proposed = np.asarray(prior(x, idx, slot, mask[idx], t), dtype=int)
                proposed = np.clip(proposed, 0, NUM_ACTIONS - 1)
                a[idx] = np.where(mask[idx, proposed], proposed, 0)

            nk, ng = ck[ix, a], cg[ix, a]
            eff_g_mem = np.maximum(ng, warming_g) if hot_scale else ng
            mem_exceeded = (
                nk * worlds.worker_mem + eff_g_mem * worlds.replica_mem > worlds.mem_cap
            ) & (a != 0)
            dk_abs = np.abs(nk - k)
            dg_up = np.maximum(ng - g, 0)
            dg_down = np.maximum(g - ng, 0)

            if hot_scale:
                # Non-blocking hot-scale: replica scale-up runs in background without pausing workers.
                immediate_reject = mem_exceeded & (dg_up == 0)
                refused[
                    ix[immediate_reject & alive],
                    nk[immediate_reject & alive],
                    ng[immediate_reject & alive],
                ] = True

                start_bg = alive & (dg_up > 0)
                # Check if target k alone fits on current g during background warmup.
                k_warm = np.where(
                    nk * worlds.worker_mem + g * worlds.replica_mem <= worlds.mem_cap, nk, k
                )
                k_warm = np.maximum(k_warm, g)
                warming_g = np.where(start_bg, ng, warming_g)
                # Serial same-GPU replica launch duration in background (InferencePool._start_replicas).
                bg_dur = worlds.load * dg_up
                warmup_rem_s = np.where(start_bg, np.where(mem_exceeded, 2.0, bg_dur), warmup_rem_s)
                warmup_total_s = np.where(start_bg, bg_dur, warmup_total_s)
                warming_will_oom = np.where(start_bg, mem_exceeded, warming_will_oom)

                rep_down_s = np.where(dg_down > 0, 0.35 * worlds.drain, 0.0)
                worker_sw_s = np.where(
                    dk_abs > 0, 0.30 * worlds.cpu_switch * np.sqrt(dk_abs), 0.0
                )
                switch_s = np.where(
                    immediate_reject,
                    0.2,
                    np.where(dg_up > 0, worker_sw_s, rep_down_s + worker_sw_s),
                )
                k = np.where(
                    alive & (~immediate_reject), np.where(dg_up > 0, k_warm, nk), k
                )
                g = np.where(alive & (~immediate_reject) & (dg_up == 0), ng, g)
            else:
                rejected = mem_exceeded
                refused[ix[rejected & alive], nk[rejected & alive], ng[rejected & alive]] = True
                rep_up_s = np.where(dg_up > 0, worlds.load * dg_up, 0.0)
                rep_down_s = np.where(
                    dg_down > 0, worlds.drain * (1.0 + 0.25 * (dg_down - 1)), 0.0
                )
                worker_sw_s = np.where(
                    dk_abs > 0, worlds.cpu_switch * (0.6 + 0.4 * np.sqrt(dk_abs)), 0.0
                )
                switch_s = np.where(rejected, 0.5, rep_up_s + rep_down_s + worker_sw_s)
                accepted_rep_up = alive & (~rejected) & (dg_up > 0)
                inferred_load = switch_s / np.maximum(dg_up, 1)
                obs_load_s = np.where(
                    accepted_rep_up,
                    np.where(has_load, 0.5 * obs_load_s + 0.5 * inferred_load, inferred_load),
                    obs_load_s,
                )
                has_load |= accepted_rep_up
                k = np.where(alive & ~rejected, nk, k)
                g = np.where(alive & ~rejected, ng, g)

            phase = 1.0 - work_rem / np.maximum(total_work, 1.0)
            next_rate, next_wait, next_sm, next_slow = worlds.observe(
                k, g, phase, step_noise, shard_factor=shard_t
            )

            # During active background warmup, PCIe/HBM weight copy imposes mild transient contention.
            if hot_scale:
                is_warming = alive & (warming_g > 0)
                _, nominal_elapsed = _window_step(
                    worlds, work_rem, next_rate, drain=(t + 1 == LENGTH)
                )
                overlap_frac = np.where(
                    is_warming,
                    np.clip(warmup_rem_s / np.maximum(nominal_elapsed, 1.0), 0.0, 1.0),
                    0.0,
                )
                copy_dip = 1.0 - worlds.bg_copy_penalty * overlap_frac
                next_rate = next_rate * copy_dip
                next_slow = next_slow * (1.0 + 0.5 * worlds.bg_copy_penalty * overlap_frac)

            step_work, elapsed_s = _window_step(
                worlds, work_rem, next_rate, drain=(t + 1 == LENGTH)
            )

            if hot_scale:
                was_warming = alive & (warming_g > 0)
                warmup_rem_s = np.where(
                    was_warming, np.maximum(warmup_rem_s - elapsed_s, 0.0), warmup_rem_s
                )
                just_finished = was_warming & (warmup_rem_s <= 1e-6)
                ready_g = np.where(just_finished, warming_g, ready_g)
                ready_oom = np.where(just_finished, warming_will_oom, ready_oom)
                warming_g = np.where(just_finished, 0, warming_g)
                warming_will_oom = np.where(just_finished, False, warming_will_oom)

            denom_s = np.maximum(elapsed_s, 1.0)
            next_cost_feat = np.minimum(switch_s / denom_s, 4.0)
            next_reward_feat = -np.minimum(
                (base_rate / np.maximum(next_rate, 1e-3)) * (1.0 + switch_s / denom_s),
                10.0,
            )

            x[idx, slot] = tok[idx]
            masks[idx, slot] = mask[idx]
            actions[idx, slot] = a[idx]
            rewards[idx, slot] = next_reward_feat[idx]

            work_rem = np.where(alive, np.maximum(work_rem - step_work, 0.0), work_rem)
            wall_s[segment, idx] += switch_s[idx] + elapsed_s[idx]
            done_now = alive & ((work_rem <= 1e-3) | (t + 1 == LENGTH))
            terminal[idx, slot] = done_now[idx]
            finished |= done_now

            if segment == 0:
                seg0_used[idx] = t + 1

            prev_rate = np.where(alive, active_rate, prev_rate)
            active_rate = np.where(alive, next_rate, active_rate)
            wait = np.where(alive, next_wait, wait)
            sm = np.where(alive, next_sm, sm)
            slow = np.where(alive, next_slow, slow)
            cost_feat = np.where(alive, next_cost_feat, cost_feat).astype(np.float32)
            reward_feat = np.where(alive, next_reward_feat, reward_feat).astype(np.float32)
            old = np.where(alive, a, old)

    if return_wall_s:
        return x, masks, actions, rewards, terminal, wall_s
    return x, masks, actions, rewards, terminal

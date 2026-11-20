# Host Protocol and Hot-Scaling

The host owns worker processes, policy replica processes, and telemetry collection. `rollout-autotune` communicates with the host through Python APIs or newline-delimited JSON over `stdio`.

## 1. Threshold `Runtime` protocol (`rollout_autotune`)

Python hosts implement `configure(target: Config) -> bool` and `observe(current: Config) -> Window | None`, then call `rollout_autotune.run(runtime)`.

Over `stdio` (`python -m rollout_autotune`):

- Controller sends `{"op": "configure", "k": 12, "g": 3}`; host replies `true` or `false`.
- Controller sends `{"op": "observe", "k": 12, "g": 3}`; host replies with `Window` counter fields (`worker_wait_s`, `worker_live_s`, `sm_active_s`, `gpu_sampled_s`, `matched_service_s`, `matched_solo_s`) or `null` when complete.

## 2. DIT `Session` and `stdio` protocol (`rollout_autotune.dit`)

`Session(weights, limits=Config(32, 4), seed=7, load_s=None, initial=Config(12, 3))` consumes per-window ratios and rates directly:

- `wait` (`float | None`): Active worker inference wait fraction in `[0, 1]`.
- `sm` (`float | None`): GPU SM-active fraction in `[0, 1]` (ignored by `--no-sm` checkpoints).
- `slowdown` (`float | None`): Matched service time divided by solo reference time (`>= 0`).
- `rate` (`float`): Effective environment `steps / s` completed during the settled window (`steps / formal_s`), without coupling to framework-internal steady-state estimators.
- `work_left` (`float`): Estimated remaining environment steps in the evaluation (spanning remaining suites when a single `Session` is shared across suites).
- `elapsed_s` (`float`): Settled window wall seconds (excluding disjoint blocking switch time).
- `switch_s` (`float`, default `0.0`): Blocking reconfiguration seconds preceding the window.
- `load_s` (`float | None`, optional): Per-replica cold startup seconds. If omitted, `Session` infers `load_s` online once a replica scale-up completes.

`Session` also tracks `hold_streak` (consecutive settled windows holding the active configuration with valid telemetry) and exposes `session.settled` (`hold_streak >= 2` with no pending proposal or in-flight warmup). For round-robin sharded benchmarks, call `plan_suite_windows(total_episodes, window_episodes=50, k=session.current.k, num_tasks=...)` at each suite start to pick a shard count coprime with `num_tasks` while minimizing tail-wave worker slot waste.

## 3. Non-blocking hot-scaling (`HotPoolScaler`)

Synchronous `pool.scale(g)` blocks all workers and idle-waits already-loaded replicas for 15–30s while a new large model checkpoint loads. `HotPoolScaler(pool, session)` removes this barrier:

1. **Two-layer VRAM guardrail**:
   - *Pre-flight total budget*: Queries `nvidia-smi` between windows (when 0 workers are active) to measure exact per-replica resident VRAM `M_rep` (including PyTorch caching allocator pools) and decomposes window `peak_vram_bytes` into per-worker VRAM `M_worker` (accounting for EGL offscreen framebuffers). Candidate configurations `(k', g')` whose predicted peak `baseline + g_eff * M_rep + k' * M_worker` exceeds `min(0.92 * M_total, M_total - 1.5 GiB)` are masked out before spawning.
   - *Post-window high-water & fault backoff*: If a completed window's observed peak exceeds `min(0.94 * M_total, M_total - 1.0 GiB)` or triggers a recoverable worker/EGL fault, `scaler.record_window(...)` marks the `(>=k, >=g)` envelope in `session.rejected` and `Session.propose(..., vram_pressure=True)` immediately steps down to a safe configuration without aborting the run.
2. **Isolated staging warmup**: Added replicas start in a separate staging pool instance on a background worker thread, avoiding concurrent mutation of `pool.processes` during active healthchecks. Meanwhile, `K` updates immediately and evaluation proceeds at `(target.k, current.g)`.
3. **Causal telemetry attribution & handoff**: Completed warmup attaches at `scaler.poll()` in `< 1 ms`. `Session.propose()` encodes the completed window under the `(k, g)` that actually served it, promotes `current.g` to the newly warmed replica count, and locks `dg` for one settled window so clean post-attach telemetry is observed before any further replica resize. Across multi-suite runs, background warmups may continue seamlessly across intermediate suite boundaries, calling `scaler.finish_warmup()` only after the final suite completes.

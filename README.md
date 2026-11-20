# Rollout Autotune

Online concurrency control `(K workers, g replicas)` for single-GPU closed-loop policy evaluation. The host owns the environment workers, GPU telemetry and policy processes; the controller runs independently of the evaluation framework.

Two controllers are included:

- **Threshold baseline (`rollout_autotune`)**: Zero-dependency Python standard library rule controller (`baseline/threshold-v1`), starting from `(12, 3)` and adjusting one axis by `±1` per settled window.
- **Step-fused DIT (`rollout_autotune.dit`)**: Opt-in ~4.83M-parameter causal Decision Importance Transformer served purely on CPU in NumPy, supporting 15 multi-scale `(ΔK, Δg)` jumps, replica startup cost priors (`load_s`), task-interleaved modulo window planning (`plan_suite_windows`), cross-suite warm starts, and non-blocking **hot-scaling** (`HotPoolScaler`).

## Threshold controller

Install with `python -m pip install .` (Python 3.11+). Embed `rollout_autotune.run(runtime)` or launch `python -m rollout_autotune` as a subprocess. Python callers can override grid limits with `limits=Config(32, 4)` and cutoffs with `thresholds=Thresholds(...)`. The CLI accepts `--max-workers` and `--max-replicas`.

The Python `Runtime` protocol implements two blocking methods:

- `configure(target: Config) -> bool` applies the requested `(k, g)` counts. Acknowledge after the target is active, preserving in-flight episodes and completed work. Return `False` with the previous configuration intact on rejection.
- `observe(current: Config) -> Window | None` returns deltas measured entirely under `current` (e.g., a 30-second settled window or a fixed-episode shard), excluding initialization and reconfiguration intervals. Return `None` once the suite finishes.

Over stdio the controller writes one JSON line and reads one JSON line:

```json
{"op":"configure","k":12,"g":3}
true
{"op":"observe","k":12,"g":3}
{"worker_wait_s":252,"worker_live_s":360,"sm_active_s":15,"gpu_sampled_s":30,"matched_service_s":10.4,"matched_solo_s":10}
```

Reply `null` to an `observe` request when complete. Extra diagnostic keys in `observe` JSON objects are ignored. Call `decide(current, window)` directly in Python to inspect the three computed ratios, target config and decision reason.

### Window measurements

| Signal | How the host measures it |
| --- | --- |
| Worker inference wait | Sum wall time active workers spent awaiting policy replies divided by summed active worker wall time, clipped to the window. Exclude idle workers with no assigned work. |
| GPU saturation proxy | Accumulate `SM-active fraction × sample duration` into `sm_active_s`, divided by `gpu_sampled_s` (e.g., NVIDIA DCGM field 1002 SM Activity). |
| Replica contention proxy | Divide summed inference service time (`matched_service_s`, dispatch to device completion excluding queue wait) by summed warmed isolated service times (`matched_solo_s`) for matched batch shapes. Other device activity (such as GPU EGL rendering) also raises this ratio. |

Use `Window.from_samples(workers=[(wait_s, live_s), ...], gpu=[(sm_fraction, duration_s), ...], batches=[(shape_key, service_s), ...], solo_by_shape=references)` to aggregate raw host samples.

### Initial decision rules

Apply the first matching rule per settled window (`1 ≤ g ≤ K`, default `K ≤ 32, g ≤ 4`):

| Condition | Action |
| --- | --- |
| Multiple replicas and matched slowdown > 1.30 | Remove one replica (`g − 1`) |
| Wait > 0.60 and SM activity > 0.85 | Remove one worker (`K − 1`) |
| Wait < 0.25 and SM activity < 0.60 | Add one worker (`K + 1`) |
| Wait > 0.60 and SM activity < 0.60, with one replica or matched slowdown < 1.10 | Add one replica (`g + 1`) |
| Within deadband, required observations unavailable, or action hits a limit | Hold |

## Step-fused DIT

### Train

Install training dependencies with `python -m pip install '.[training]'` and train the offline critic + policy networks:

```sh
python training/train.py --out runs/initial --worlds 30000 --epochs 4 --seed 82
```

Key training options:

- `--no-sm`: Train a 28-input SM-free variant (omitting SM fraction and its availability bit from the 30-input token) for hosts without DCGM profiling permissions (`ERR_NVGPUCTRPERM`).
- `--cold-scale`: Model synchronous stop-the-world replica launches instead of the default non-blocking background hot-scaling physics.
- `--real-windows windows.jsonl --real-changes changes.jsonl [--real-weight 20]`: Mix recorded hardware trajectories with synthetic tasks. Rejected proposals are tracked and masked automatically, and logs exceeding 64 transitions are split into overlapping causal chunks with lookahead bootstrap.

#### Architecture and objective

- **Backbone**: Independent causal Transformers for critic and actor (`6` pre-RMSNorm blocks, `d_model = 256`, `8` RoPE heads, `704`-wide SwiGLU, context `64`, **~4.83M actor parameters**). Only `policy.npz` is exported for CPU deployment.
- **Action space (15 actions)**: `HOLD (0, 0)`, worker steps `(±1, 0), (±2, 0), (±4, 0), (±8, 0)`, replica steps `(0, ±1), (0, ±2)`, and coupled diagonal steps `(+4, +1), (−4, −1)`.
- **Step token**: Fuses normalized `(k, g)`, telemetry (`wait`, `sm`, `slowdown`, `log1p(rate)`, `log1p(work_left)`), optional per-replica cold load prior `log1p(load_s)`, availability and episode-reset flags, previous requested action one-hot, blocking `switch_s / elapsed_s`, and work-normalized step reward `-(base_rate / rate) * (1 + switch_s / elapsed_s)`.
- **Risk-neutral critic + debiased WMLE**: Phase 1 fits MSE action-values `Q(H_t, a)` (averaging over latent switch costs without Jensen optimism bias), an in-sample `τ = 0.62` expectile state-value `V(H_t)`, and a behavior propensity head `μ(a | H_t)`. Phase 2 freezes the critic and minimizes masked cross-entropy weighted by `μ(a_t | H_t)^(-0.55) * exp(scaled_expected_advantage / 0.4)`.

### Deploy

Install inference dependencies with `python -m pip install '.[dit]'`. Run over stdio (optional `--initial-workers`, `--initial-replicas`, `--max-workers`, `--max-replicas`, `--load-s`):

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m rollout_autotune.dit runs/initial/policy.npz --load-s 15.2
```

Observation records supply rates and settled window durations (use `null` for unavailable `wait`, `sm` or `slowdown`; `sm`, `switch_s` and `load_s` are optional):

```json
{"wait":0.54,"sm":0.80,"slowdown":1.4,"rate":160,"work_left":75000,"elapsed_s":30,"switch_s":0.0,"load_s":15.2}
```

For non-blocking **hot-scaling** and **task-interleaved windowing** in Python hosts, wrap an `InferencePool`-compatible server pool with `HotPoolScaler` and compute coprime shard counts with `plan_suite_windows` (see `experiments/libero_pro/run.py` for a complete multi-suite runner):

```python
from rollout_autotune import Config
from rollout_autotune.dit import HotPoolScaler, Session, plan_suite_windows

session = Session("runs/initial/policy.npz", seed=7, initial=Config(4, 1))
scaler = HotPoolScaler(pool, session)
try:
    scaler.start_initial()
    n_windows = plan_suite_windows(500, window_episodes=50, k=session.current.k, num_tasks=10)
    switch_s = 0.0
    for window_idx in range(n_windows):
        cur = session.current
        # ... run evaluation window under concurrency=cur.k, replicas=cur.g ...
        if window_idx + 1 < n_windows:
            cutover_s = scaler.poll()
            target = session.propose(
                wait=wait, slowdown=slowdown, rate=rate,
                work_left=work_left, elapsed_s=elapsed_s, switch_s=switch_s + cutover_s,
            )
            switch_s = scaler.apply(target)
finally:
    scaler.close()
    pool.stop()
```

Over stdio, hosts using asynchronous warmup can reply `{"warming": true}` to `configure` and attach `"warmup_accepted": true, "warmup_s": 16.0` on the subsequent `observe` payload once ready.

## Testing

Run the unit and end-to-end test suite with:

```sh
pytest -v
```

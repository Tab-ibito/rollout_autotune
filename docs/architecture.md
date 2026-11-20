# Architecture

`rollout-autotune` separates online concurrency control from both policy model internals and simulator internals. It ships two runtime controllers, an offline training suite, and a reference multi-suite benchmark runner.

## Module layout

- `src/rollout_autotune/__init__.py`: Core dataclasses (`Config`, `Thresholds`, `Window`, `Decision`), the `Runtime` protocol, and the stdlib-only threshold rule controller.
- `src/rollout_autotune/__main__.py`: Newline-delimited JSON (`stdio`) subprocess wrapper for the threshold controller.
- `src/rollout_autotune/dit.py`: CPU-only NumPy causal Transformer actor (`CPUModel` with preallocated KV cache), stateful online controller (`Session`), adaptive suite-window planner (`plan_suite_windows`), non-blocking staging pool coordinator (`HotPoolScaler`), and `stdio` CLI.
- `training/model.py`: PyTorch pre-RMSNorm causal Transformer (`6` layers, width `256`, `8` RoPE heads, `704` SwiGLU, `~4.83M` actor parameters).
- `training/simulator.py`: Vectorized closed-loop queueing surrogate modeling CPU/render stepping, GPU multi-replica throughput `g * R_1 / S(k, g)`, VRAM limits, missing telemetry, and background hot-scaling warmup contention.
- `training/train.py`: Two-stage offline Decision Importance Transformer (DIT) trainer, historical JSONL log importer (`real_history`), and CPU/PyTorch parity + paired unseen-world validator.
- `experiments/libero_pro/run.py`: 16-suite LIBERO-Pro + StarVLA evaluation runner driving a persistent `InferencePool` via `Session`, `HotPoolScaler`, and `plan_suite_windows` with start/end solo drift calibration.

## Design principles

1. **Zero GPU interference**: Policy replicas and EGL rendering workers already contend for GPU SMs and VRAM. The online controller executes exclusively on CPU (`< 1 ms` per decision in NumPy with preallocated KV buffers).
2. **Train dual networks, deploy actor only**: Offline training uses a 31-head critic (`V`, `Q`, and behavior propensity `μ`) to compute debiased expected advantages, then exports only the 15-action actor weights (`policy.npz`).
3. **Multi-scale actions**: The 15 `(ΔK, Δg)` deltas span `K ∈ {±1, ±2, ±4, ±8}`, `g ∈ {±1, ±2}`, diagonal `(±4, ±1)`, and `HOLD (0, 0)`, reaching distant operating regimes in 2–3 windows while preserving fine local adjustments.
4. **Non-blocking hot-scaling**: `HotPoolScaler` launches added replicas in an isolated staging pool on a background thread while existing workers continue evaluating, eliminating 15–30s stop-the-world replica load pauses.
5. **Task-interleaved windows**: At each suite boundary, `plan_suite_windows` looks near the requested window size, preferring shard counts coprime with the task count so round-robin sharding samples all tasks evenly in every window, then minimizing unfilled worker slots in the last wave using the active worker count. Keeping multi-window partitions across all suites preserves online adaptability whenever workload characteristics shift between suites.

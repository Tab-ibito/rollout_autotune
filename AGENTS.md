# AGENTS.md

Repository guide for AI coding agents and contributors working on `rollout-autotune`.

## Dependency boundaries (strict)

1. **`src/rollout_autotune/__init__.py` and `__main__.py` (Threshold baseline)**:
   - **Standard library only** (`python >= 3.11`). Never import `numpy`, `torch`, or evaluation-framework packages here.
   - Preserve backward compatibility with the `baseline/threshold-v1` `Config`, `Thresholds`, `Window`, `Decision`, `decide`, `Runtime`, and `run` contracts.
2. **`src/rollout_autotune/dit.py` (CPU DIT inference, window planning & hot-scaling)**:
   - **NumPy + Python standard library only** (`pip install '.[dit]'`). Never import `torch` at runtime; the controller must execute on CPU with zero GPU VRAM or SM footprint.
   - Keep `CPUModel` numerically equivalent (`max_abs_err < 1e-4`) to `training.model.Transformer` across arbitrary sequence lengths, including sliding-window cache rebuilds past `CONTEXT = 64`.
3. **`training/` (Offline simulation & PyTorch training)**:
   - Uses `numpy` and `torch` (`pip install '.[training]'`). Not packaged into the runtime wheel (`packages = ['rollout_autotune']` under `src/`).
   - Feature normalization, action ordering (`ACTIONS`), availability masks, and hot-scaling state transitions in `training/simulator.py` and `training/train.py` must stay bit-aligned with `rollout_autotune.dit`.
4. **`experiments/libero_pro/run.py` (LIBERO-Pro host runner)**:
   - Must keep the controller process free of `torch` (enforced via `"torch" in sys.modules`) and decoupled from framework-private steady-state heuristics (`rate = window_steps / formal`, `elapsed_s = wall`).

## Commands

- Run tests: `pytest -v`
- Verify stdlib-only import: `PYTHONPATH=src python3 -c "import rollout_autotune"`
- Train SM-free hot-scaling policy:
  `python training/train.py --out runs/initial --worlds 30000 --epochs 4 --seed 82 --no-sm`
- Run threshold controller over stdio: `python -m rollout_autotune`
- Run DIT controller over stdio:
  `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m rollout_autotune.dit runs/initial/policy.npz`

## Control-loop invariants

- Feasibility always enforces `1 <= g <= k <= limits.k` and `1 <= g <= limits.g`.
- Currently active `(k, g)` (`HOLD`, action index `0`) must always remain legal in `action_mask`.
- Missing `wait` (and missing `sm` on full 30-input checkpoints) must force `HOLD`.
- During asynchronous hot-scaling (`HotPoolScaler` / `warming=True`), telemetry from a window must be recorded under the exact `(k, g)` that actively served that window, and further replica changes (`dg != 0`) remain locked until the warmed replicas attach and complete at least one settled window.
- For episode-major round-robin sharding, `plan_suite_windows` must return counts representable as `ceil(total_episodes / ceil(total_episodes / count)) == count`, preferring counts coprime with `num_tasks` before minimizing tail-wave idle worker slots.

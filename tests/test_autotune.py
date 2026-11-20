import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from rollout_autotune import Config, Thresholds, Window, decide
from rollout_autotune import dit
from training.model import Transformer
from training.simulator import Worlds, make_data
from training.train import real_history, td_targets


def _save_dummy_policy(path: Path, *, use_sm: bool = True, seed: int = 0) -> Transformer:
    torch.manual_seed(seed)
    input_dim = dit.INPUT if use_sm else len(dit.NO_SM_COLUMNS)
    net = Transformer(dit.NUM_ACTIONS, input_dim=input_dim).eval()
    np.savez(
        path,
        format_version=dit.FORMAT if use_sm else dit.NO_SM_FORMAT,
        **{k: v.cpu().numpy() for k, v in net.state_dict().items()},
    )
    return net


def test_model_scale_and_multi_step_action_space():
    net = Transformer(dit.NUM_ACTIONS, input_dim=dit.INPUT)
    params = sum(p.numel() for p in net.parameters())
    # 6 blocks, width 256, 8 heads, 704 SwiGLU -> ~4.83M parameters.
    assert 4_500_000 <= params <= 5_200_000
    assert dit.NUM_ACTIONS == 15
    assert dit.ACTIONS[0] == (0, 0)
    assert (8, 0) in dit.ACTIONS and (-8, 0) in dit.ACTIONS
    assert (0, 2) in dit.ACTIONS and (4, 1) in dit.ACTIONS and (-4, -1) in dit.ACTIONS


def test_threshold_controller_and_window_samples():
    w = Window.from_samples(
        workers=[(12.0, 30.0), (15.0, 30.0)],
        gpu=[(0.5, 30.0)],
        batches=[("b4", 2.6), ("unknown", 9.9)],
        solo_by_shape={"b4": 2.0},
    )
    dec = decide(Config(12, 3), w, thresholds=Thresholds())
    assert dec.target == Config(12, 3)
    assert dec.reason == "deadband"

    # High slowdown removes replica even if wait/sm missing.
    w_contend = Window(0.0, 0.0, 0.0, 0.0, 14.0, 10.0)
    assert decide(Config(12, 3), w_contend).target == Config(12, 2)

    # Booleans must be rejected where numeric counters or thresholds are expected.
    with pytest.raises(ValueError):
        Window(True, 1.0, 0.0, 0.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        Thresholds(wait_low=False)  # type: ignore[arg-type]


def test_step_metrics_unifies_fixed_work_and_fixed_time_windows():
    _, r_fixed_work = dit.step_metrics(base_rate=100.0, rate=200.0, elapsed_s=15.0, switch_s=0.0)
    _, r_fixed_time = dit.step_metrics(base_rate=100.0, rate=200.0, elapsed_s=30.0, switch_s=0.0)
    assert math.isclose(r_fixed_work, -0.5, rel_tol=1e-6)
    assert math.isclose(r_fixed_time, -0.5, rel_tol=1e-6)

    cost_ratio, r_switched = dit.step_metrics(
        base_rate=100.0, rate=200.0, elapsed_s=30.0, switch_s=3.0
    )
    assert math.isclose(cost_ratio, 0.1, rel_tol=1e-6)
    assert -1.0 < r_switched < -0.5


def test_simulator_closed_loop_physics_causal_direction():
    rng = np.random.default_rng(42)
    w = Worlds(1, rng, fixed_limits=True)
    w.worker_rate[:] = 30.0
    w.cpu_rate[:] = 180.0
    w.cpu_knee[:] = 8
    w.cpu_penalty[:] = 0.08
    w.egl_penalty[:] = 0.0
    w.replica_rate[:] = 500.0
    w.gpu_cap[:] = 1000.0
    w.contention[:] = 0.10
    w.contention_exp[:] = 1.1

    _, wait_low_k, sm_low_k, slow_g1 = w.observe(
        np.array([8]), np.array([1]), 0.0, np.array([1.0])
    )
    _, wait_high_k, sm_high_k, _ = w.observe(
        np.array([28]), np.array([1]), 0.0, np.array([1.0])
    )
    assert wait_high_k[0] <= wait_low_k[0] < 0.20
    assert sm_high_k[0] < 0.25
    assert math.isclose(slow_g1[0], 1.0, rel_tol=1e-6)

    w.cpu_rate[:] = 300.0
    w.replica_rate[:] = 60.0
    w.gpu_cap[:] = 300.0
    _, wait_rep, sm_rep, _ = w.observe(np.array([16]), np.array([1]), 0.0, np.array([1.0]))
    assert wait_rep[0] > 0.65
    assert sm_rep[0] < 0.30

    # StarVLA-like EGL background load + moderate replica contention:
    # slowdown exceeds 1.5 at g=3, yet aggregate throughput increases monotonically g=1 -> 2 -> 3.
    w.worker_rate[:] = 28.0
    w.cpu_rate[:] = 320.0
    w.cpu_knee[:] = 16
    w.egl_penalty[:] = 0.020
    w.replica_rate[:] = 140.0
    w.gpu_cap[:] = 280.0
    w.contention[:] = 0.22
    w.contention_exp[:] = 1.10
    r1, _, _, s1 = w.observe(np.array([12]), np.array([1]), 0.0, np.array([1.0]))
    r2, _, _, s2 = w.observe(np.array([12]), np.array([2]), 0.0, np.array([1.0]))
    r3, _, _, s3 = w.observe(np.array([12]), np.array([3]), 0.0, np.array([1.0]))
    assert s1[0] > 1.15 and s3[0] > 1.55
    assert r1[0] < r2[0] < r3[0]


def test_simulator_generates_missing_telemetry_load_prior_and_multi_step_actions():
    rng = np.random.default_rng(123)
    worlds = Worlds(256, rng)
    x, masks, actions, rewards, terminal = make_data(worlds, rng, use_sm=True)
    valid = masks.any(-1)
    assert valid.any()
    # Availability columns (8: wait, 9: sm, 10: slowdown, 11: load_s) include both present and missing.
    avail = x[valid][:, 8:12]
    for col in range(4):
        assert (avail[:, col] == 0).any() and (avail[:, col] == 1).any()
    # Multi-step actions beyond single-step +-1 must be exercised.
    assert (actions[valid] >= 3).any()
    chosen_ok = masks[valid][np.arange(int(valid.sum())), actions[valid]]
    assert chosen_ok.all()
    assert np.isfinite(rewards[valid]).all()


def test_real_history_handles_rejections_multi_step_zero_episodes_and_long_runs(tmp_path: Path):
    windows_file = tmp_path / "windows.jsonl"
    changes_file = tmp_path / "changes.jsonl"

    rows = [
        {
            "index": 0,
            "episodes": 0,
            "steps": 4000,
            "formal_s": 25.0,
            "seconds": 25.0,
            "config": {"k": 12, "g": 3},
            "next": {"k": 8, "g": 2},  # Multi-step diagonal (-4, -1)
            "wait": 0.55,
            "sm": None,
            "slowdown": 1.85,
        },
        {
            "index": 1,
            "episodes": 0,
            "steps": 4500,
            "formal_s": 25.0,
            "seconds": 26.2,
            "config": {"k": 8, "g": 2},
            "next": {"k": 8, "g": 3},  # Rejected by host!
            "wait": 0.65,
            "sm": 0.45,
            "slowdown": 1.05,
        },
    ]
    for idx in range(2, 71):
        rows.append(
            {
                "index": idx,
                "episodes": 5,
                "steps": 5000,
                "formal_s": 25.0,
                "seconds": 25.0,
                "config": {"k": 8, "g": 2},
                "next": {"k": 8, "g": 2},
                "wait": 0.40,
                "sm": 0.70,
                "slowdown": 1.15,
            }
        )
    rows.append(
        {
            "index": 71,
            "episodes": 5,
            "steps": 5000,
            "formal_s": 25.0,
            "seconds": 25.0,
            "config": {"k": 8, "g": 2},
            "next": None,
            "wait": 0.40,
            "sm": 0.70,
            "slowdown": 1.15,
        }
    )

    changes = [
        {"after_episodes": 0, "seconds": 1.2},
        {"after_episodes": 0, "seconds": 0.6},
    ]
    windows_file.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    changes_file.write_text("\n".join(json.dumps(c) for c in changes) + "\n")

    x, masks, actions, rewards, done = real_history(windows_file, changes_file, load_s=15.0)
    valid = masks.any(-1)
    assert int(valid.sum()) == 71
    assert int(done.sum()) == 1
    g_plus_1_idx = dit.ACTIONS.index((0, 1))
    assert masks[0, 1, g_plus_1_idx]
    assert not masks[0, 2, g_plus_1_idx]
    assert actions[0, 0] == dit.ACTIONS.index((-4, -1))
    assert masks[valid][np.arange(71), actions[valid]].all()
    assert np.isfinite(x).all() and np.isfinite(rewards).all()

    v = torch.full((len(x), dit.CONTEXT), -5.0)
    targets = td_targets(
        torch.from_numpy(rewards),
        torch.from_numpy(done),
        torch.from_numpy(valid),
        v,
        0.9,
        0.9,
    )
    assert torch.isfinite(targets).all()
    expected_62 = rewards[0, 62] + 0.9 * (-5.0)
    assert math.isclose(float(targets[0, 62]), float(expected_62), rel_tol=1e-5)


@pytest.mark.parametrize("use_sm", [True, False])
def test_cpu_model_parity_and_load_s_inference(tmp_path: Path, use_sm: bool):
    weights = tmp_path / "policy.npz"
    net = _save_dummy_policy(weights, use_sm=use_sm, seed=11)
    cpu = dit.CPUModel(weights)

    rng = np.random.default_rng(9)
    probe = rng.normal(size=(70, dit.INPUT)).astype(np.float32)
    with torch.no_grad():
        for i, step_tok in enumerate(probe):
            got = cpu.logits(step_tok)
            window_in = dit.model_inputs(probe[max(0, i - 63) : i + 1], use_sm)
            ref = net(torch.from_numpy(window_in)[None])[0, -1].numpy()
            assert np.max(np.abs(got - ref)) < 1e-4

    # Test missing wait guard and online load_s inference from accepted replica scale-up.
    session = dit.Session(weights, limits=Config(64, 8), seed=3)
    assert session.load_s is None
    target0 = session.propose(
        wait=None, sm=0.5, slowdown=None, rate=120.0, work_left=50000.0, elapsed_s=30.0
    )
    assert target0 == Config(12, 3)
    session.acknowledge(True)

    # Simulate an accepted +2 serial replica jump taking 40s -> inferred load_s = 40 / 2 = 20.0s.
    session.pending = (dit.ACTIONS.index((0, 2)), Config(12, 5))
    session.acknowledge(True)
    session.propose(
        wait=0.4, sm=0.5, slowdown=1.4, rate=180.0, work_left=40000.0, elapsed_s=30.0, switch_s=40.0
    )
    assert session.load_s is not None and math.isclose(session.load_s, 20.0, rel_tol=1e-5)
    session.acknowledge(True)


def test_dit_cli_jsonl_protocol_and_eof(tmp_path: Path):
    weights = tmp_path / "no_sm_policy.npz"
    _save_dummy_policy(weights, use_sm=False, seed=21)

    p = subprocess.Popen(
        [sys.executable, "-m", "rollout_autotune.dit", str(weights), "--load-s", "14.5"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert p.stdin is not None and p.stdout is not None
    req0 = json.loads(p.stdout.readline())
    assert req0 == {"op": "configure", "k": 4, "g": 1}
    p.stdin.write("true\n")
    p.stdin.flush()

    req1 = json.loads(p.stdout.readline())
    assert req1["op"] == "observe"
    p.stdin.write(
        json.dumps(
            {
                "extra_field": "ignored",
                "wait": 0.52,
                "slowdown": None,
                "rate": 150.0,
                "work_left": 30000.0,
                "elapsed_s": 30.0,
                "load_s": 16.0,
            }
        )
        + "\n"
    )
    p.stdin.flush()

    nxt = json.loads(p.stdout.readline())
    if nxt["op"] == "configure":
        p.stdin.write("false\n")
        p.stdin.flush()
        nxt = json.loads(p.stdout.readline())
    assert nxt["op"] == "observe"
    p.stdin.write("null\n")
    p.stdin.flush()
    assert p.wait(timeout=5) == 0

    p_eof = subprocess.run(
        [sys.executable, "-m", "rollout_autotune.dit", str(weights)],
        input="",
        text=True,
        capture_output=True,
        check=False,
    )
    assert p_eof.returncode != 0
    assert "EOFError" in p_eof.stderr


def test_end_to_end_train_with_real_logs_and_no_sm(tmp_path: Path):
    windows_file = tmp_path / "w.jsonl"
    changes_file = tmp_path / "c.jsonl"
    windows_file.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "episodes": 10,
                        "steps": 3000,
                        "formal_s": 20.0,
                        "seconds": 20.0,
                        "config": {"k": 12, "g": 3},
                        "next": {"k": 12, "g": 2},
                        "wait": 0.5,
                        "sm": None,
                        "slowdown": 1.6,
                    }
                ),
                json.dumps(
                    {
                        "episodes": 10,
                        "steps": 3600,
                        "formal_s": 20.0,
                        "seconds": 21.0,
                        "config": {"k": 12, "g": 2},
                        "next": None,
                        "wait": 0.4,
                        "sm": None,
                        "slowdown": 1.1,
                    }
                ),
            ]
        )
        + "\n"
    )
    changes_file.write_text(json.dumps({"after_episodes": 10, "seconds": 1.0}) + "\n")

    out_dir = tmp_path / "run_no_sm"
    subprocess.run(
        [
            sys.executable,
            "training/train.py",
            "--out",
            str(out_dir),
            "--worlds",
            "64",
            "--epochs",
            "1",
            "--seed",
            "7",
            "--no-sm",
            "--real-windows",
            str(windows_file),
            "--real-changes",
            str(changes_file),
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    report = json.loads((out_dir / "report.json").read_text())
    assert report["uses_sm"] is False
    assert report["input_dim"] == 28
    assert report["num_actions"] == 15
    assert report["actor_parameters"] > 4_500_000
    assert report["hot_scale"] is True
    assert report["real_transitions"] == 1
    assert report["cpu_logit_max_error"] < 1e-4
    assert (out_dir / "policy.npz").exists()


def test_hot_pool_scaler_background_staging_and_oom_isolation(tmp_path: Path):
    import time

    weights = tmp_path / "hot_policy.npz"
    _save_dummy_policy(weights, use_sm=False, seed=5)

    class FakePool:
        fail_above_g = 3

        def __init__(
            self,
            *,
            adapter=None,
            gpu_devices=("0",),
            base_port=16694,
            output_dir=tmp_path,
            ready_timeout_s=60.0,
            cuda_mps=None,
        ):
            self.adapter = adapter
            self.gpu_devices = gpu_devices
            self.base_port = base_port
            self.output_dir = output_dir
            self.ready_timeout_s = ready_timeout_s
            self.cuda_mps = cuda_mps
            self.processes = {}
            self.contexts = {}

        @property
        def size(self):
            return len(self.processes)

        def maximum_workers_per_replica(self):
            return None

        def _start_replicas(self, replica_ids):
            for rid in replica_ids:
                time.sleep(0.05)
                if rid >= self.fail_above_g:
                    raise RuntimeError("CUDA out of memory")
                self.contexts[rid] = f"ctx-{rid}"
                self.processes[rid] = f"proc-{rid}"

        def scale(self, target: int):
            for rid in [r for r in self.processes if r >= target]:
                self.processes.pop(rid, None)
                self.contexts.pop(rid, None)
            self._start_replicas([r for r in range(target) if r not in self.processes])

        def stop(self):
            self.processes.clear()
            self.contexts.clear()

    pool = FakePool()
    session = dit.Session(weights, seed=7, initial=Config(6, 1))
    scaler = dit.HotPoolScaler(pool, session)
    try:
        scaler.start_initial()
        assert pool.size == 1
        assert session.load_s is not None

        # Simulate a proposal jumping (6, 1) -> (10, 3): k updates immediately to 10,
        # while replicas 1 and 2 warm asynchronously in staging without touching pool.processes!
        session.propose(
            wait=0.65, slowdown=1.2, rate=100.0, work_left=50000.0, elapsed_s=30.0, switch_s=0.0
        )
        session.pending = (dit.ACTIONS.index((4, 1)), Config(10, 2))
        sw_apply = scaler.apply(Config(10, 2))
        assert sw_apply == 0.0
        # Active window immediately sees k=10, g=1 while replica #1 warms in background.
        assert session.current == Config(10, 1)
        assert session.warming_g == 2
        assert set(pool.processes) == {0}

        # Wait for background staging to finish (simulating evaluation window running concurrently).
        time.sleep(0.12)
        sw_poll = scaler.poll()
        assert sw_poll < 0.05
        assert set(pool.processes) == {0, 1}

        # Next propose() records window under (10, 1) and hands off cleanly to g=2.
        tgt = session.propose(
            wait=0.60,
            slowdown=1.25,
            rate=110.0,
            work_left=40000.0,
            elapsed_s=30.0,
            switch_s=sw_poll,
        )
        assert session.current.g == 2 and tgt.g == 2
        scaler.apply(tgt)

        # Now attempt scaling to g=4 where rid=3 fails with background OOM:
        # active pool {0, 1} must remain 100% intact and (_, 4) must be masked in rejected!
        session.propose(
            wait=0.60, slowdown=1.4, rate=150.0, work_left=30000.0, elapsed_s=30.0, switch_s=0.0
        )
        session.pending = (dit.ACTIONS.index((0, 2)), Config(session.current.k, 4))
        scaler.apply(Config(session.current.k, 4))
        time.sleep(0.18)
        scaler.poll()
        assert set(pool.processes) == {0, 1}
        tgt_after_oom = session.propose(
            wait=0.60, slowdown=1.4, rate=150.0, work_left=20000.0, elapsed_s=30.0, switch_s=0.0
        )
        assert (session.current.k, 4) in session.rejected
        assert tgt_after_oom.g < 4
        scaler.apply(tgt_after_oom)

        # A suite can end before the new replica has finished loading. Settle it
        # before creating the next session so its first window uses the live pool.
        session.propose(
            wait=0.60, slowdown=1.4, rate=150.0, work_left=20000.0, elapsed_s=30.0
        )
        target = Config(session.current.k, 3)
        session.pending = (dit.ACTIONS.index((0, 1)), target)
        scaler.apply(target)
        assert scaler.finish_warmup() > 0.0
        next_session = dit.Session(weights, initial=session.current, load_s=session.load_s)
        next_session.rejected.update(session.rejected)
        assert next_session.current == target
        assert next_session.current.g == pool.size == 3
        assert next_session.load_s == session.load_s
        assert next_session.rejected == session.rejected
        assert next_session.base_rate is None and next_session.windows == 0
        assert session.warming_g is None and session.ready_warmup is None
        assert scaler.finish_warmup() == 0.0
    finally:
        scaler.close()
        pool.stop()


def test_session_hold_streak(tmp_path: Path):
    weights = tmp_path / "policy.npz"
    _save_dummy_policy(weights, use_sm=False, seed=13)
    session = dit.Session(weights, seed=7, initial=Config(4, 1))
    assert session.hold_streak == 0 and not session.settled

    # Missing telemetry forces HOLD, which must NOT count as a settled hold.
    t0 = session.propose(
        wait=None, slowdown=1.1, rate=90.0, work_left=500000.0, elapsed_s=30.0
    )
    assert t0 == Config(4, 1)
    session.acknowledge(True)
    assert session.hold_streak == 0 and not session.settled

    # Force two genuine settled HOLDs -> hold_streak reaches 2 and session.settled becomes True.
    for expected_streak in (1, 2):
        session.propose(
            wait=0.45, slowdown=1.2, rate=100.0, work_left=500000.0, elapsed_s=30.0
        )
        session.pending = (0, session.current)
        session._pending_settled_hold = True
        session.acknowledge(True)
        assert session.hold_streak == expected_streak
    assert session.settled

    # Starting a background warmup immediately clears hold_streak and settled;
    # the attachment step (observed under old g) also keeps hold_streak == 0!
    session.propose(
        wait=0.60, slowdown=1.2, rate=100.0, work_left=500000.0, elapsed_s=30.0
    )
    session.pending = (dit.ACTIONS.index((8, 0)), Config(12, 1))
    session._pending_settled_hold = False
    session.acknowledge(True)
    assert session.hold_streak == 0 and not session.settled

    session.propose(
        wait=0.62, slowdown=1.3, rate=120.0, work_left=480000.0, elapsed_s=30.0
    )
    session.pending = (dit.ACTIONS.index((0, 1)), Config(12, 2))
    session._pending_settled_hold = False
    session.acknowledge(True, warming=True)
    assert session.warming_g == 2 and session.hold_streak == 0 and not session.settled

    # Warmup finishes; next propose() attaches g=2 -> even if action==0, attached==True so hold_streak stays 0!
    session.notify_warmup(True, load_elapsed_s=17.0)
    session.propose(
        wait=0.62, slowdown=1.3, rate=120.0, work_left=450000.0, elapsed_s=30.0
    )
    session.pending = (0, session.current)
    session.acknowledge(True)
    assert session.current == Config(12, 2)
    assert session.hold_streak == 0 and not session.settled


def test_modulo_suite_window_planning():
    for workers, expected in ((4, 9), (8, 9), (12, 11), (16, 11)):
        assert dit.plan_suite_windows(500, window_episodes=50, k=workers, num_tasks=10) == expected
    assert dit.plan_suite_windows(50, window_episodes=50, k=12, num_tasks=10) == 1
    # Tie-breaker prefers shard size closer to window_episodes=100 (7 -> 72 vs 3 -> 167).
    assert dit.plan_suite_windows(500, window_episodes=100, k=12, num_tasks=10) == 7


def test_libero_pro_runner_keeps_modulo_windows_after_holds(tmp_path: Path):
    weights = tmp_path / "hold_policy.npz"
    net = _save_dummy_policy(weights, use_sm=False, seed=31)
    # A policy that always holds still gets observations throughout each suite.
    state = {k: v.cpu().numpy() for k, v in net.state_dict().items()}
    state["head.bias"] = np.full(dit.NUM_ACTIONS, -10.0, dtype=np.float32)
    state["head.bias"][0] = 10.0
    np.savez(weights, format_version=dit.NO_SM_FORMAT, **state)

    solo_inputs = tmp_path / "obs.npz"
    np.savez(
        solo_inputs,
        primary=np.zeros((20, 4, 4, 3), dtype=np.uint8),
        wrist=np.zeros((20, 4, 4, 3), dtype=np.uint8),
        proprioception=np.zeros((20, 8), dtype=np.float32),
        instructions=np.array(["pick"] * 20),
    )
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text("{}\n")
    baseline_file = tmp_path / "baseline.json"
    baseline_file.write_text(
        json.dumps(
            {
                "source": "mock-baseline",
                "solo_median_s": 0.055,
                "end_to_end_s": 400.0,
                "suites": {
                    s: {"episodes": 200, "steps": 20000, "formal_s": 200.0}
                    for s in ("libero_goal_object", "libero_goal_swap", "libero_goal_lan")
                },
            }
        )
        + "\n"
    )
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    out_dir = tmp_path / "out"

    # Create a minimal mock roboscope_kit package in tmp_path/mock_kit to exercise run.py end-to-end
    # inside a clean torch-free Python subprocess (testing the 'torch not in sys.modules' guard too!).
    kit_root = tmp_path / "mock_kit" / "roboscope_kit"
    (kit_root / "runtime" / "benchmark").mkdir(parents=True)
    (kit_root / "__init__.py").write_text("")
    (kit_root / "runtime" / "__init__.py").write_text("")
    (kit_root / "runtime" / "benchmark" / "__init__.py").write_text("")
    (kit_root / "policy.py").write_text(
        "class EpisodeContext:\n"
        "    def __init__(self, episode_id, task_id): pass\n"
        "class Observation:\n"
        "    def __init__(self, episode_id, step, features): pass\n"
    )
    (kit_root / "client.py").write_text(
        "from pathlib import Path\n"
        "class PolicyClient:\n"
        "    def __init__(self, socket_path, timeout_s=120, encoding='msgpack'):\n"
        "        self.trace = Path(socket_path)\n"
        "    def __enter__(self): return self\n"
        "    def __exit__(self, *a): pass\n"
        "    def reset(self, ctx): pass\n"
        "    def act(self, obs):\n"
        "        with self.trace.open('a') as f: f.write('0.055\\n')\n"
        "    def end_episode(self, ep): pass\n"
    )
    (kit_root / "runtime" / "benchmark" / "processes.py").write_text(
        "from types import SimpleNamespace\n"
        "class InferencePool:\n"
        "    def __init__(self, *, adapter, gpu_devices, base_port, output_dir, ready_timeout_s, cuda_mps=None):\n"
        "        self.adapter = adapter; self.gpu_devices = gpu_devices; self.base_port = base_port\n"
        "        self.output_dir = output_dir; self.ready_timeout_s = ready_timeout_s; self.cuda_mps = cuda_mps\n"
        "        self.processes = {}; self.contexts = {}\n"
        "    @property\n"
        "    def size(self): return len(self.processes)\n"
        "    def maximum_workers_per_replica(self): return None\n"
        "    def _start_replicas(self, ids):\n"
        "        self.output_dir.mkdir(parents=True, exist_ok=True)\n"
        "        trace = self.output_dir.parent / 'service' / 'service-1.jsonl'\n"
        "        for r in ids:\n"
        "            self.processes[r] = f'p-{r}'\n"
        "            self.contexts[r] = SimpleNamespace(socket_path=str(trace))\n"
        "    def scale(self, g):\n"
        "        for r in [k for k in self.processes if k >= g]:\n"
        "            self.processes.pop(r, None); self.contexts.pop(r, None)\n"
        "        self._start_replicas([r for r in range(g) if r not in self.processes])\n"
        "    def stop(self): self.processes.clear(); self.contexts.clear()\n"
    )
    (kit_root / "runtime" / "benchmark" / "shard_merge.py").write_text(
        "import json\n"
        "def merge_shards(*, benchmark, suite, episodes, manifest_path, shard_dirs):\n"
        "    assert len(shard_dirs) >= 2, 'merge_shards should only be called for multi-shard suites'\n"
        "    rids, succ, steps = [], 0, 0\n"
        "    tasks = {t: 0 for t in range(10)}\n"
        "    for d in shard_dirs:\n"
        "        rj = json.loads((d / 'result.json').read_text())\n"
        "        steps += rj['result']['total_steps']\n"
        "        succ += rj['result']['successes']\n"
        "        for p in (d / 'queue/completed').glob('*.json'):\n"
        "            item = json.loads(p.read_text())\n"
        "            for rec in item['records']:\n"
        "                rids.append(rec['record_id'])\n"
        "                tasks[rec['metadata']['task_id']] += 1\n"
        "    return {\n"
        "        'result': {'total_steps': steps, 'successes': succ, 'episodes': len(rids)},\n"
        "        'official_result': {'tasks': [{'task_id': t, 'episodes': tasks[t]} for t in range(10)]},\n"
        "        'validation': {'total_records': len(rids), 'unique_record_ids': len(set(rids)) == len(rids), 'coverage': 'complete'},\n"
        "    }\n"
    )
    (kit_root / "runtime" / "benchmark" / "cli.py").write_text(
        "from dataclasses import dataclass\n"
        "from enum import Enum\n"
        "import json\n"
        "from types import SimpleNamespace\n"
        "class BenchmarkName(Enum):\n"
        "    LIBERO_PRO = 'libero_pro'\n"
        "@dataclass(frozen=True)\n"
        "class Spec:\n"
        "    suite: str\n"
        "@dataclass(frozen=True)\n"
        "class BenchmarkRunRequest:\n"
        "    bench: object; suite: str; checkpoint: str; output_dir: object; episodes: int; probe: bool; extra: object\n"
        "def atomic_json(path, obj):\n"
        "    path.parent.mkdir(parents=True, exist_ok=True)\n"
        "    path.write_text(json.dumps(obj))\n"
        "def load_benchmark_manifest(path):\n"
        "    suites = [f'libero_{g}_{v}' for g in ('goal', 'spatial', '10', 'object') for v in ('object', 'swap', 'lan', 'task')]\n"
        "    return SimpleNamespace(supported_suites={BenchmarkName.LIBERO_PRO: suites}, defaults={})\n"
        "def benchmark_factory(bench):\n"
        "    return SimpleNamespace(validate_suite=lambda s: None)\n"
        "def parse_args(argv):\n"
        "    return SimpleNamespace(checkpoint_source='ckpt')\n"
        "def _preflight_evaluation(cli_args, bench, manifest, factory):\n"
        "    return SimpleNamespace(benchmark_spec=Spec(suite='libero_goal_object'), policy='p', gpu_devices=('0',))\n"
        "def _build_benchmark_spec(bench, suite, extra): return Spec(suite=suite)\n"
        "def shard_build(manifest): return {'kit_version': '0.8.0b66', 'manifest_sha256': 'abc'}\n"
        "def _build_inference(policy, req, manifest, spec): return SimpleNamespace(load_timeout_s=90.0)\n"
        "def _run_with_pool(*, factory, request, manifest, pool, gpu_devices, concurrency, replicas, output_dir, probe, shard_kwargs=None):\n"
        "    comp_dir = output_dir / 'queue' / 'completed'\n"
        "    comp_dir.mkdir(parents=True, exist_ok=True)\n"
        "    total = request.episodes * 10\n"
        "    if shard_kwargs is None:\n"
        "        ordinals = list(range(total))\n"
        "    else:\n"
        "        c = shard_kwargs['shard_count']; idx = shard_kwargs['shard_index']\n"
        "        ordinals = list(range(idx, total, c))\n"
        "    trace = output_dir.parent.parent / 'service' / 'service-1.jsonl'\n"
        "    with trace.open('a') as tf:\n"
        "        for _ in ordinals: tf.write('0.080\\n')\n"
        "    tasks = {t: 0 for t in range(10)}\n"
        "    for o in ordinals:\n"
        "        tid, eid = o % 10, o // 10\n"
        "        tasks[tid] += 1\n"
        "        rec_id = f'task-{tid:04d}-episode-{eid:04d}'\n"
        "        (comp_dir / f'{rec_id}.json').write_text(json.dumps({\n"
        "            'item_id': rec_id,\n"
        "            'records': [{'record_id': rec_id, 'steps': 100, 'metadata': {'task_id': tid, 'episode_index': eid}}],\n"
        "            'metadata': {'inference_wait_s': 4.0, 'elapsed_s': 10.0},\n"
        "        }))\n"
        "    steps = 100 * len(ordinals)\n"
        "    data = {\n"
        "        'phase_timings': {'evaluation_s': max(len(ordinals) * 0.5, 1.0)},\n"
        "        'result': {'episodes': len(ordinals), 'total_steps': steps, 'successes': len(ordinals) // 2, 'steady_steps_per_s': 200.0},\n"
        "        'official_result': {'tasks': [{'task_id': t, 'episodes': tasks[t]} for t in range(10)]},\n"
        "        'recovery': {'worker_failures': [], 'oom_events': 0, 'initial_workers_per_gpu': {'0': min(concurrency, len(ordinals))}},\n"
        "    }\n"
        "    atomic_json(output_dir / 'result.json', data)\n"
        "    return data\n"
    )

    env = {
        **dict(__import__("os").environ),
        "PYTHONPATH": f"{tmp_path / 'mock_kit'}:{Path('src').resolve()}",
    }
    proc = subprocess.run(
        [
            sys.executable,
            "experiments/libero_pro/run.py",
            "--manifest",
            str(manifest_file),
            "--checkpoint",
            str(ckpt_dir),
            "--weights",
            str(weights),
            "--model-python",
            sys.executable,
            "--model-cwd",
            str(tmp_path),
            "--solo-inputs",
            str(solo_inputs),
            "--baseline",
            str(baseline_file),
            "--output-dir",
            str(out_dir),
            "--suites",
            "libero_goal_object",
            "libero_goal_swap",
            "libero_goal_lan",
            "--episodes",
            "20",
            "--window-episodes",
            "50",
        ],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"
    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["adaptive_windows"] is True
    assert summary["window_strategy"] == "modulo"
    assert summary["episodes"] == 600
    s1 = summary["suites"]["libero_goal_object"]
    s2 = summary["suites"]["libero_goal_swap"]
    s3 = summary["suites"]["libero_goal_lan"]
    assert s2["hold_streak_start"] >= 2
    assert s1["windows_per_suite"] == s2["windows_per_suite"] == s3["windows_per_suite"] == 3
    assert s2["windows"][0]["target"] == {"k": 4, "g": 1}
    assert s3["windows"][0]["target"] == {"k": 4, "g": 1}
    first_window = s1["windows"][0]
    assert first_window["steps_per_s"] == pytest.approx(
        first_window["steps"] / first_window["formal_s"]
    )
    assert "steady_steps_per_s" not in first_window
    assert "control_steps_per_s" not in first_window
    assert s2["validation"] == {
        "total_records": 200,
        "unique_record_ids": True,
        "coverage": "complete",
    }
    # Work left in Suite 1's final window accounted for Suites 2 & 3 (> 0)!
    assert s1["windows"][-1]["work_left"] > 30000.0
    # Baseline ran at 100 steps/s (20000 / 200s); mock run executes at ~200 steps/s -> ~2.0x speedup!
    paired_s1 = summary["baseline_comparison"]["suites"]["libero_goal_object"]
    assert paired_s1["formal_speedup"] > 1.5
    assert paired_s1["steps_per_s_speedup"] == pytest.approx(paired_s1["formal_speedup"])
    assert summary["calibration"]["within_five_percent"] is True


def test_vram_guardrail_blocks_24gib_overcommit_and_backs_off_high_water(tmp_path: Path):
    mib = 1024 * 1024
    weights = tmp_path / "vram_policy.npz"
    net = _save_dummy_policy(weights, use_sm=False, seed=41)
    # Bias logits: #1 (+4, +1), #2 (0, +1), #3 HOLD, #4 (-4, 0), #5 (-2, 0).
    state = {k: v.cpu().numpy() for k, v in net.state_dict().items()}
    state["head.weight"] = np.zeros_like(state["head.weight"])
    bias = np.full(dit.NUM_ACTIONS, -20.0, dtype=np.float32)
    bias[dit.ACTIONS.index((4, 1))] = 12.0
    bias[dit.ACTIONS.index((0, 1))] = 9.0
    bias[0] = 8.0
    bias[dit.ACTIONS.index((-4, 0))] = 6.0
    bias[dit.ACTIONS.index((-2, 0))] = 4.0
    state["head.bias"] = bias
    np.savez(weights, format_version=dit.NO_SM_FORMAT, **state)

    class ProbePool:
        def __init__(
            self,
            *,
            adapter=None,
            gpu_devices=("0",),
            base_port=16694,
            output_dir=tmp_path,
            ready_timeout_s=60.0,
            cuda_mps=None,
        ):
            self.adapter = adapter
            self.gpu_devices = gpu_devices
            self.base_port = base_port
            self.output_dir = output_dir
            self.ready_timeout_s = ready_timeout_s
            self.cuda_mps = cuda_mps
            self.processes = {}
            self.contexts = {}

        @property
        def size(self):
            return len(self.processes)

        def maximum_workers_per_replica(self):
            return None

        def _start_replicas(self, ids):
            for rid in ids:
                self.processes[rid] = f"p-{rid}"
                self.contexts[rid] = f"c-{rid}"

        def scale(self, target: int):
            for rid in [r for r in self.processes if r >= target]:
                self.processes.pop(rid, None)
                self.contexts.pop(rid, None)
            self._start_replicas([r for r in range(target) if r not in self.processes])

        def stop(self):
            self.processes.clear()
            self.contexts.clear()

    total_24g = 24576 * mib
    gpu_state = {"used": 10 * mib}

    def probe():
        u = gpu_state["used"]
        return total_24g, u, total_24g - u

    pool = ProbePool()
    session = dit.Session(weights, seed=7, initial=Config(8, 1))
    scaler = dit.HotPoolScaler(pool, session, memory_probe=probe)
    try:
        scaler.start_initial()
        # Simulate post-forward PyTorch caching allocator footprint (~8950 MiB/replica).
        gpu_state["used"] = 8960 * mib
        scaler.refresh_replica_vram()

        # Window at (8, 1) peaks at 12,880 MiB (~490 MiB/worker).
        # Candidate (12, 2) would require ~23,790 MiB > 22,609 MiB safe budget on 24 GiB!
        pressure = scaler.record_window(
            Config(8, 1), active_workers=8, peak_vram_bytes=12880 * mib
        )
        assert pressure is False
        vmask = scaler.vram_action_mask()
        assert vmask is not None
        assert not vmask[dit.ACTIONS.index((4, 1))]  # (12, 2) blocked!
        assert vmask[dit.ACTIONS.index((0, 1))]  # (8, 2) (~21.8 GiB) allowed!
        assert vmask[dit.ACTIONS.index((4, 0))]  # (12, 1) (~14.8 GiB) allowed!

        target = session.propose(
            wait=0.60,
            slowdown=1.15,
            rate=92.0,
            work_left=400000.0,
            elapsed_s=85.0,
            vram_mask=vmask,
            vram_pressure=pressure,
        )
        assert target == Config(8, 2)
        scaler.apply(target)

        # Wait for staging replica #1 to attach -> active becomes (8, 2).
        import time

        for _ in range(20):
            if scaler.poll() > 0 or session.ready_warmup is not None:
                break
            time.sleep(0.01)
        gpu_state["used"] = 17910 * mib

        # Simulate a window that crossed the 94% high-water mark (23,613 MiB > 23,101 MiB):
        # controller must reject (>=8, >=2) and actively step down to (4, 2) via (-4, 0)!
        pressure_hw = scaler.record_window(
            Config(8, 1),
            active_workers=8,
            peak_vram_bytes=23613 * mib,
            warming_g=2,
        )
        assert pressure_hw is True
        assert (8, 2) in session.rejected
        target_backoff = session.propose(
            wait=0.55,
            slowdown=1.40,
            rate=130.0,
            work_left=350000.0,
            elapsed_s=70.0,
            vram_mask=scaler.vram_action_mask(),
            vram_pressure=pressure_hw,
        )
        assert target_backoff == Config(4, 2)
        scaler.apply(target_backoff)
        assert session.current == Config(4, 2)
    finally:
        scaler.close()
        pool.stop()


def test_libero_pro_runner_recovers_from_egl_worker_fault_without_abort(tmp_path: Path):
    weights = tmp_path / "scale_policy.npz"
    net = _save_dummy_policy(weights, use_sm=False, seed=51)
    # Bias (+8, 0) higher than HOLD so Window 1 jumps (4, 1) -> (12, 1), where mock kit
    # triggers a recovered 0x8CDD EGL worker fault, and (-4, 0) steps back down to (8, 1).
    state = {k: v.cpu().numpy() for k, v in net.state_dict().items()}
    state["head.weight"] = np.zeros_like(state["head.weight"])
    bias = np.full(dit.NUM_ACTIONS, -20.0, dtype=np.float32)
    bias[dit.ACTIONS.index((8, 0))] = 10.0
    bias[0] = 6.0
    bias[dit.ACTIONS.index((-4, 0))] = 4.0
    state["head.bias"] = bias
    np.savez(weights, format_version=dit.NO_SM_FORMAT, **state)

    solo_inputs = tmp_path / "obs.npz"
    np.savez(
        solo_inputs,
        primary=np.zeros((20, 4, 4, 3), dtype=np.uint8),
        wrist=np.zeros((20, 4, 4, 3), dtype=np.uint8),
        proprioception=np.zeros((20, 8), dtype=np.float32),
        instructions=np.array(["pick"] * 20),
    )
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text("{}\n")
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    out_dir = tmp_path / "out_fault"

    kit_root = tmp_path / "mock_kit_fault" / "roboscope_kit"
    (kit_root / "runtime" / "benchmark").mkdir(parents=True)
    (kit_root / "__init__.py").write_text("")
    (kit_root / "runtime" / "__init__.py").write_text("")
    (kit_root / "runtime" / "benchmark" / "__init__.py").write_text("")
    (kit_root / "policy.py").write_text(
        "class EpisodeContext:\n"
        "    def __init__(self, episode_id, task_id): pass\n"
        "class Observation:\n"
        "    def __init__(self, episode_id, step, features): pass\n"
    )
    (kit_root / "client.py").write_text(
        "from pathlib import Path\n"
        "class PolicyClient:\n"
        "    def __init__(self, socket_path, timeout_s=120, encoding='msgpack'):\n"
        "        self.trace = Path(socket_path)\n"
        "    def __enter__(self): return self\n"
        "    def __exit__(self, *a): pass\n"
        "    def reset(self, ctx): pass\n"
        "    def act(self, obs):\n"
        "        with self.trace.open('a') as f: f.write('0.055\\n')\n"
        "    def end_episode(self, ep): pass\n"
    )
    (kit_root / "runtime" / "benchmark" / "processes.py").write_text(
        "from types import SimpleNamespace\n"
        "class InferencePool:\n"
        "    def __init__(self, *, adapter, gpu_devices, base_port, output_dir, ready_timeout_s, cuda_mps=None):\n"
        "        self.adapter = adapter; self.gpu_devices = gpu_devices; self.base_port = base_port\n"
        "        self.output_dir = output_dir; self.ready_timeout_s = ready_timeout_s; self.cuda_mps = cuda_mps\n"
        "        self.processes = {}; self.contexts = {}\n"
        "    @property\n"
        "    def size(self): return len(self.processes)\n"
        "    def maximum_workers_per_replica(self): return None\n"
        "    def _start_replicas(self, ids):\n"
        "        self.output_dir.mkdir(parents=True, exist_ok=True)\n"
        "        trace = self.output_dir.parent / 'service' / 'service-1.jsonl'\n"
        "        for r in ids:\n"
        "            self.processes[r] = f'p-{r}'\n"
        "            self.contexts[r] = SimpleNamespace(socket_path=str(trace))\n"
        "    def scale(self, g):\n"
        "        for r in [k for k in self.processes if k >= g]:\n"
        "            self.processes.pop(r, None); self.contexts.pop(r, None)\n"
        "        self._start_replicas([r for r in range(g) if r not in self.processes])\n"
        "    def stop(self): self.processes.clear(); self.contexts.clear()\n"
    )
    (kit_root / "runtime" / "benchmark" / "shard_merge.py").write_text(
        "import json\n"
        "def merge_shards(*, benchmark, suite, episodes, manifest_path, shard_dirs):\n"
        "    rids, succ, steps = [], 0, 0\n"
        "    tasks = {t: 0 for t in range(10)}\n"
        "    for d in shard_dirs:\n"
        "        rj = json.loads((d / 'result.json').read_text())\n"
        "        steps += rj['result']['total_steps']\n"
        "        succ += rj['result']['successes']\n"
        "        for p in (d / 'queue/completed').glob('*.json'):\n"
        "            item = json.loads(p.read_text())\n"
        "            for rec in item['records']:\n"
        "                rids.append(rec['record_id'])\n"
        "                tasks[rec['metadata']['task_id']] += 1\n"
        "    return {\n"
        "        'result': {'total_steps': steps, 'successes': succ, 'episodes': len(rids)},\n"
        "        'official_result': {'tasks': [{'task_id': t, 'episodes': tasks[t]} for t in range(10)]},\n"
        "        'validation': {'total_records': len(rids), 'unique_record_ids': len(set(rids)) == len(rids), 'coverage': 'complete'},\n"
        "    }\n"
    )
    (kit_root / "runtime" / "benchmark" / "cli.py").write_text(
        "from dataclasses import dataclass\n"
        "from enum import Enum\n"
        "import json\n"
        "from types import SimpleNamespace\n"
        "class BenchmarkName(Enum):\n"
        "    LIBERO_PRO = 'libero_pro'\n"
        "@dataclass(frozen=True)\n"
        "class Spec:\n"
        "    suite: str\n"
        "@dataclass(frozen=True)\n"
        "class BenchmarkRunRequest:\n"
        "    bench: object; suite: str; checkpoint: str; output_dir: object; episodes: int; probe: bool; extra: object\n"
        "def atomic_json(path, obj):\n"
        "    path.parent.mkdir(parents=True, exist_ok=True)\n"
        "    path.write_text(json.dumps(obj))\n"
        "def load_benchmark_manifest(path):\n"
        "    suites = [f'libero_{g}_{v}' for g in ('goal', 'spatial', '10', 'object') for v in ('object', 'swap', 'lan', 'task')]\n"
        "    return SimpleNamespace(supported_suites={BenchmarkName.LIBERO_PRO: suites}, defaults={})\n"
        "def benchmark_factory(bench):\n"
        "    return SimpleNamespace(validate_suite=lambda s: None)\n"
        "def parse_args(argv):\n"
        "    return SimpleNamespace(checkpoint_source='ckpt')\n"
        "def _preflight_evaluation(cli_args, bench, manifest, factory):\n"
        "    return SimpleNamespace(benchmark_spec=Spec(suite='libero_goal_object'), policy='p', gpu_devices=('0',))\n"
        "def _build_benchmark_spec(bench, suite, extra): return Spec(suite=suite)\n"
        "def shard_build(manifest): return {'kit_version': '0.8.0b66', 'manifest_sha256': 'abc'}\n"
        "def _build_inference(policy, req, manifest, spec): return SimpleNamespace(load_timeout_s=90.0)\n"
        "def _run_with_pool(*, factory, request, manifest, pool, gpu_devices, concurrency, replicas, output_dir, probe, shard_kwargs=None):\n"
        "    comp_dir = output_dir / 'queue' / 'completed'\n"
        "    comp_dir.mkdir(parents=True, exist_ok=True)\n"
        "    total = request.episodes * 10\n"
        "    c = shard_kwargs['shard_count']; idx = shard_kwargs['shard_index']\n"
        "    ordinals = list(range(idx, total, c))\n"
        "    trace = output_dir.parent.parent / 'service' / 'service-1.jsonl'\n"
        "    with trace.open('a') as tf:\n"
        "        for _ in ordinals: tf.write('0.080\\n')\n"
        "    tasks = {t: 0 for t in range(10)}\n"
        "    for o in ordinals:\n"
        "        tid, eid = o % 10, o // 10\n"
        "        tasks[tid] += 1\n"
        "        rec_id = f'task-{tid:04d}-episode-{eid:04d}'\n"
        "        (comp_dir / f'{rec_id}.json').write_text(json.dumps({\n"
        "            'item_id': rec_id,\n"
        "            'records': [{'record_id': rec_id, 'steps': 100, 'metadata': {'task_id': tid, 'episode_index': eid}}],\n"
        "            'metadata': {'inference_wait_s': 4.0, 'elapsed_s': 10.0},\n"
        "        }))\n"
        "    failures = [{'worker_id': 4, 'category': 'OTHER', 'message': '0x8CDD GL_FRAMEBUFFER_UNSUPPORTED'}] if concurrency >= 12 else []\n"
        "    peak_mib = 24561.0 if concurrency >= 12 else 14800.0\n"
        "    steps = 100 * len(ordinals)\n"
        "    data = {\n"
        "        'phase_timings': {'evaluation_s': max(len(ordinals) * 0.5, 1.0)},\n"
        "        'result': {'episodes': len(ordinals), 'total_steps': steps, 'successes': len(ordinals) // 2},\n"
        "        'official_result': {'tasks': [{'task_id': t, 'episodes': tasks[t]} for t in range(10)]},\n"
        "        'resources': {'gpu': {'gpu_mem_mib_max': {'0': peak_mib}, 'total_memory_bytes': 24576 * 1024**2}},\n"
        "        'recovery': {'worker_failures': failures, 'oom_events': 0, 'initial_workers_per_gpu': {'0': min(concurrency, len(ordinals))}},\n"
        "    }\n"
        "    atomic_json(output_dir / 'result.json', data)\n"
        "    return data\n"
    )

    env = {
        **dict(__import__("os").environ),
        "PYTHONPATH": f"{tmp_path / 'mock_kit_fault'}:{Path('src').resolve()}",
    }
    proc = subprocess.run(
        [
            sys.executable,
            "experiments/libero_pro/run.py",
            "--manifest",
            str(manifest_file),
            "--checkpoint",
            str(ckpt_dir),
            "--weights",
            str(weights),
            "--model-python",
            sys.executable,
            "--model-cwd",
            str(tmp_path),
            "--solo-inputs",
            str(solo_inputs),
            "--output-dir",
            str(out_dir),
            "--suites",
            "libero_goal_object",
            "--episodes",
            "20",
            "--window-episodes",
            "50",
        ],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"
    summary = json.loads((out_dir / "summary.json").read_text())
    suite_res = summary["suites"]["libero_goal_object"]
    assert suite_res["recovered_worker_failures"] == 1
    assert suite_res["vram_guard_events"] == 1
    # Window 1: (4, 1) -> proposes (12, 1); Window 2: (12, 1) hits 0x8CDD -> steps down to (8, 1) for Window 3!
    assert [w["k"] for w in suite_res["windows"]] == [4, 12, 8]
    assert [12, 1] in suite_res["rejected"]

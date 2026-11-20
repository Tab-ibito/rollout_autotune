"""Evaluate the frozen DIT across LIBERO-Pro with a single StarVLA inference pool.

Use RoboScope Kit a7f756c on PYTHONPATH, its StarVLA example in the model
environment, and the benchmark manifest and checkpoint from the Autotune run.
"""

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path


def serve(output):
    """Run a StarVLA replica process while appending per-forward service seconds to ``output``."""
    from starvla_policy import StarVLAPolicy, run

    class TimedPolicy(StarVLAPolicy):
        def __init__(self):
            super().__init__()
            self.trace = (output / f"service-{os.getpid()}.jsonl").open("x", buffering=1)

        def act(self, observation):
            start = time.monotonic_ns()
            result = super().act(observation)
            self.trace.write(f"{(time.monotonic_ns() - start) * 1e-9}\n")
            return result

    run(TimedPolicy())


def read(path):
    return json.loads(path.read_text())


class ServiceSamples:
    """Incrementally tail ``service-*.jsonl`` latency traces across active replica PIDs."""

    def __init__(self, directory):
        self.directory = directory
        self.offsets = {}

    def read_new(self):
        samples = []
        for path in sorted(self.directory.glob("service-*.jsonl")):
            with path.open() as trace:
                trace.seek(self.offsets.get(path, 0))
                samples.extend(float(line) for line in trace)
                self.offsets[path] = trace.tell()
        return samples


def replay_solo(pool, samples, inputs, phase):
    """Measure 10 isolated single-replica forward passes at run start/end for slowdown & drift."""
    import numpy as np
    from roboscope_kit.client import PolicyClient
    from roboscope_kit.policy import EpisodeContext, Observation

    samples.read_new()
    with np.load(inputs, allow_pickle=False) as archive:
        primary, wrist = archive["primary"], archive["wrist"]
        state, instructions = archive["proprioception"], archive["instructions"]
    with PolicyClient(pool.contexts[0].socket_path, timeout_s=120, encoding="msgpack") as client:
        episode = f"solo-{phase}"
        client.reset(EpisodeContext(episode_id=episode, task_id="0"))
        measured = list(range(0, 20, 2))
        # Warm CUDA/Triton kernels on cold start before recording the 10 reference timings.
        indices = [*measured[:3], *measured] if phase == "start" else measured
        for step, index in enumerate(indices):
            features = {
                "images.primary": primary[index],
                "images.wrist": wrist[index],
                "state.proprioception": state[index],
                "language.instruction": str(instructions[index]),
            }
            client.act(Observation(episode_id=episode, step=step, features=features))
        client.end_episode(episode)
    timings = samples.read_new()
    if phase == "start":
        timings = timings[3:]
    if len(timings) != 10 or any(not math.isfinite(s) or s <= 0 for s in timings):
        raise RuntimeError(f"expected 10 valid solo timings, got {timings}")
    return timings


def calibration_report(start, end=None, baseline=None):
    """Summarize solo inference medians and relative drift between start, end, and baseline."""
    median_start = statistics.median(start)
    median_end = statistics.median(end) if end else None
    historical = baseline.get("solo_median_s") if baseline is not None else None
    drift = abs(median_end / median_start - 1) if median_end is not None else None
    start_delta = abs(median_start / historical - 1) if historical else None
    end_delta = abs(median_end / historical - 1) if historical and median_end is not None else None
    deltas = [d for d in (drift, start_delta, end_delta) if d is not None]
    within_five_percent = (
        max(deltas) < 0.05
        if drift is not None and (historical is None or end_delta is not None)
        else None
    )
    return dict(
        solo_median_s=median_start,
        start_samples_s=start,
        end_samples_s=end,
        end_solo_median_s=median_end,
        baseline_solo_median_s=historical,
        baseline_solo_source=baseline.get("solo_source") if baseline else None,
        start_end_relative_delta=drift,
        start_baseline_relative_delta=start_delta,
        end_baseline_relative_delta=end_delta,
        within_five_percent=within_five_percent,
    )


def window_peak_vram_bytes(data):
    """Extract peak GPU memory usage in bytes from window ``resources.gpu`` if recorded."""
    gpu_res = data.get("resources", {}).get("gpu", {})
    mib_by_gpu = gpu_res.get("gpu_mem_mib_max") or {}
    if mib_by_gpu:
        return int(max(float(v) for v in mib_by_gpu.values()) * 1024**2)
    peak_bytes = gpu_res.get("peak_memory_bytes")
    if peak_bytes:
        return int(float(peak_bytes))
    return None


def window_total_vram_bytes(data):
    """Extract total GPU memory capacity in bytes from window ``resources.gpu`` when present."""
    gpu_res = data.get("resources", {}).get("gpu", {})
    total_bytes = gpu_res.get("total_memory_bytes")
    if total_bytes:
        return int(float(total_bytes))
    return None


def checked_window(data, output, expected, config, pool, *, warming_g=None):
    """Verify exact ``(k, g)`` & episode count; return True on recovered capacity/render faults."""
    recovery = data["recovery"]
    active_k = sum(recovery["initial_workers_per_gpu"].values())
    if (
        data["result"]["episodes"] != expected
        or active_k != min(config.k, expected)
        or pool.size != config.g
    ):
        raise RuntimeError(f"unexpected workload or active configuration in {output}")
    had_fault = bool(recovery["worker_failures"] or recovery["oom_events"])
    if had_fault and config.k == 1 and config.g == 1 and warming_g is None:
        raise RuntimeError(f"unrecoverable worker failure at (1, 1) in {output}: {recovery}")
    return had_fault


def compare(suites, baseline, end_to_end):
    """Compute per-suite and aggregate speedups (> 1.0 means current is faster than baseline)."""
    if baseline is None:
        return None
    paired = {}
    for name, current in suites.items():
        previous = baseline["suites"].get(name)
        if previous is None or previous["episodes"] != current["episodes"]:
            continue
        baseline_steps_per_s = previous["steps"] / previous["formal_s"]
        paired[name] = dict(
            baseline=previous,
            formal_speedup=previous["formal_s"] / current["formal_s"],
            formal_wall_speedup=(
                previous["formal_wall_s"] / current["formal_wall_s"]
                if previous.get("formal_wall_s")
                else None
            ),
            steps_per_s_speedup=current["steps_per_s"] / baseline_steps_per_s,
        )
    complete = len(paired) == len(suites) == len(baseline["suites"])
    return dict(
        source=baseline["source"],
        suites=paired,
        formal_speedup=(
            sum(r["baseline"]["formal_s"] for r in paired.values())
            / sum(r["formal_s"] for r in suites.values())
            if complete
            else None
        ),
        end_to_end_speedup=(
            baseline["end_to_end_s"] / end_to_end
            if complete and baseline.get("end_to_end_s")
            else None
        ),
    )


def run_experiment(args):
    """Execute all requested LIBERO-Pro suites on one persistent hot-scaled inference pool."""
    from roboscope_kit.runtime.benchmark import cli
    from roboscope_kit.runtime.benchmark.processes import InferencePool
    from roboscope_kit.runtime.benchmark.shard_merge import merge_shards
    from rollout_autotune import Config, _ratio
    from rollout_autotune.dit import HotPoolScaler, Session, plan_suite_windows

    started = time.monotonic()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    trace_dir = out / "service"
    trace_dir.mkdir()
    baseline = read(args.baseline) if args.baseline else None
    manifest = cli.load_benchmark_manifest(args.manifest)
    bench = cli.BenchmarkName.LIBERO_PRO
    factory = cli.benchmark_factory(bench)
    suites = manifest.supported_suites[bench]
    expected_suites = {
        f"libero_{group}_{variant}"
        for group in ("goal", "spatial", "10", "object")
        for variant in ("object", "swap", "lan", "task")
    }
    if set(suites) != expected_suites:
        raise ValueError("the manifest must list all 16 LIBERO-Pro suites")
    if args.suites is not None:
        suites = tuple(args.suites)
        if len(set(suites)) != len(suites) or not set(suites) <= expected_suites:
            raise ValueError("expected unique LIBERO-Pro suite names")

    policy_config = out / "policy.json"
    cli.atomic_json(
        policy_config,
        {
            "schema_version": 2,
            "cwd": str(args.model_cwd.resolve()),
            "command": [
                str(args.model_python.absolute()),
                str(Path(__file__).resolve()),
                "serve",
                str(trace_dir),
            ],
        },
    )
    session = Session(args.weights, seed=7, initial=Config(4, 1))
    if session.model.uses_sm:
        raise ValueError("the checkpoint must use the SM-free feature schema")

    first = suites[0]
    cli_args = cli.parse_args(
        [
            "--benchmark",
            bench.value,
            "--suite",
            first,
            "--output-dir",
            str(out),
            "--checkpoint-source",
            str(args.checkpoint.resolve()),
            "--concurrency",
            "4",
            "--replicas",
            "1",
            "--policy-config",
            str(policy_config),
            "--benchmark-manifest",
            str(args.manifest),
            "--run-cache",
            "off",
        ]
    )
    plan = cli._preflight_evaluation(cli_args, bench, manifest, factory)
    for suite in suites:
        factory.validate_suite(suite)
        if cli._build_benchmark_spec(bench, suite, None) != replace(
            plan.benchmark_spec, suite=suite
        ):
            raise ValueError(f"StarVLA input or action contract changed for {suite}")
    shard_build = cli.shard_build(manifest)
    first_request = cli.BenchmarkRunRequest(
        bench, first, cli_args.checkpoint_source, out, args.episodes, False, None
    )
    pool = InferencePool(
        adapter=cli._build_inference(plan.policy, first_request, manifest, plan.benchmark_spec),
        gpu_devices=plan.gpu_devices,
        base_port=int(manifest.defaults.get("base_port", 16694)),
        output_dir=out / "pool",
        ready_timeout_s=float(manifest.defaults.get("ready_timeout_s", 300)),
    )
    scaler = HotPoolScaler(pool, session)
    samples = ServiceSamples(trace_dir)
    results = {}
    suite_episodes = args.episodes * 10
    fixed_windows_per_suite = math.ceil(suite_episodes / args.window_episodes)
    weights_sha256 = hashlib.sha256(args.weights.read_bytes()).hexdigest()

    def evaluate(request, config, output, index, count, size):
        kwargs = (
            dict(
                shard_kwargs=dict(
                    shard_index=index,
                    shard_count=count,
                    shard_episodes=size,
                    shard_build=shard_build,
                )
            )
            if count > 1
            else {}
        )
        return cli._run_with_pool(
            factory=factory,
            request=request,
            manifest=manifest,
            pool=pool,
            gpu_devices=plan.gpu_devices,
            concurrency=config.k,
            replicas=config.g,
            output_dir=output,
            probe=False,
            **kwargs,
        )

    def summary(end=None, end_to_end=None):
        return dict(
            policy_sha256=weights_sha256,
            policy_device="cpu",
            mps=False,
            hot_scale=True,
            adaptive_windows=not args.fixed_windows,
            window_strategy="fixed" if args.fixed_windows else "modulo",
            initial_config=[4, 1],
            suite_order=list(suites),
            suites=results,
            episodes=sum(r["episodes"] for r in results.values()),
            initial_replica_load_s=initial_replica_load_s,
            kit_version=shard_build["kit_version"],
            manifest_sha256=shard_build["manifest_sha256"],
            checkpoint=str(cli_args.checkpoint_source),
            solo_inputs=str(args.solo_inputs.resolve()),
            end_to_end_s=end_to_end,
            formal_s=sum(r["formal_s"] for r in results.values()),
            formal_wall_s=sum(r["formal_wall_s"] for r in results.values()),
            calibration=calibration_report(solo_start, end, baseline),
            baseline_comparison=(
                compare(results, baseline, end_to_end) if end_to_end is not None else None
            ),
        )

    try:
        initial_replica_load_s = scaler.start_initial()
        solo_start = replay_solo(pool, samples, args.solo_inputs, "start")
        scaler.refresh_replica_vram()
        references = calibration_report(solo_start, baseline=baseline)
        cli.atomic_json(out / "references.json", references)
        cli.atomic_json(out / "summary.json", summary())
        print(json.dumps({"event": "calibrated", **references}), flush=True)

        global_steps = 0
        global_episodes = 0
        switch_s = 0.0
        for index, suite in enumerate(suites):
            initial = session.current
            hold_streak_start = session.hold_streak
            windows_per_suite = (
                fixed_windows_per_suite
                if args.fixed_windows
                else plan_suite_windows(
                    suite_episodes,
                    window_episodes=args.window_episodes,
                    k=initial.k,
                    num_tasks=10,
                )
            )
            shard_episodes = math.ceil(suite_episodes / windows_per_suite)
            folder = out / suite
            request = replace(first_request, suite=suite, output_dir=folder)
            print(
                json.dumps(
                    dict(
                        event="suite_started",
                        suite=suite,
                        initial=asdict(initial),
                        windows_per_suite=windows_per_suite,
                        hold_streak=hold_streak_start,
                        warming_g=session.warming_g,
                    )
                ),
                flush=True,
            )
            windows, directories = [], []
            steps = episodes = 0
            last_data = None
            last_completed_items = ()
            for i in range(windows_per_suite):
                config = session.current
                window_dir = folder / f"window-{i:02}"
                begin = time.monotonic()
                data = evaluate(request, config, window_dir, i, windows_per_suite, shard_episodes)
                wall = time.monotonic() - begin
                count = data["result"]["episodes"]
                expected = (suite_episodes + windows_per_suite - i - 1) // windows_per_suite
                warming_during = session.warming_g
                had_fault = checked_window(
                    data, window_dir, expected, config, pool, warming_g=warming_during
                )
                peak_vram_bytes = window_peak_vram_bytes(data)
                total_vram_bytes = window_total_vram_bytes(data)
                formal = data["phase_timings"]["evaluation_s"]
                window_steps = data["result"]["total_steps"]
                steps += window_steps
                episodes += count
                global_steps += window_steps
                global_episodes += count
                completed_items = [read(p) for p in (window_dir / "queue/completed").glob("*.json")]
                metadata = [d["metadata"] for d in completed_items]
                wait = _ratio(
                    sum(m["inference_wait_s"] for m in metadata),
                    sum(m["elapsed_s"] for m in metadata),
                    bounded=True,
                )
                service = samples.read_new()
                slowdown = _ratio(
                    sum(service), len(service) * references["solo_median_s"], bounded=False
                )
                # Keep throughput telemetry decoupled from framework-internal steady-state estimators.
                rate = window_steps / formal
                suite_rem_ep = suite_episodes - episodes
                future_suites_ep = (len(suites) - 1 - index) * suite_episodes
                work_left = (
                    suite_rem_ep * (steps / episodes)
                    + future_suites_ep * (global_steps / global_episodes)
                )
                recovery = data["recovery"]
                row = dict(
                    suite=suite,
                    window=i + 1,
                    windows_per_suite=windows_per_suite,
                    **asdict(config),
                    steps=window_steps,
                    episodes=count,
                    formal_s=formal,
                    formal_wall_s=wall,
                    steps_per_s=rate,
                    wait=wait,
                    slowdown=slowdown,
                    work_left=work_left,
                    service_samples=len(service),
                    warming_during_window_g=warming_during,
                    peak_vram_mib=(
                        round(peak_vram_bytes / 1024**2, 1)
                        if peak_vram_bytes is not None
                        else None
                    ),
                    worker_failures=len(recovery["worker_failures"]),
                    oom_events=recovery["oom_events"],
                )
                is_final_global_window = (index + 1 == len(suites)) and (
                    i + 1 == windows_per_suite
                )
                if not is_final_global_window:
                    cutover = scaler.poll()
                    ready = session.ready_warmup
                    vram_pressure = scaler.record_window(
                        config,
                        active_workers=min(config.k, count),
                        peak_vram_bytes=peak_vram_bytes,
                        warming_g=warming_during,
                        had_fault=had_fault,
                        total_vram_bytes=total_vram_bytes,
                    )
                    target = session.propose(
                        wait=wait,
                        slowdown=slowdown,
                        rate=rate,
                        elapsed_s=wall,
                        work_left=work_left,
                        switch_s=switch_s + cutover,
                        vram_mask=scaler.vram_action_mask(),
                        vram_pressure=vram_pressure,
                    )
                    if pool.size != session.current.g:
                        raise RuntimeError("pool and controller disagree after the cutover")
                    decision_base = session.current
                    switch_s = scaler.apply(target)
                    row.update(
                        target=asdict(target),
                        decision_base=asdict(decision_base),
                        next_active=asdict(session.current),
                        warming_next_g=session.warming_g,
                        warmup_ready=ready,
                        cutover_s=cutover,
                        switch_s=switch_s,
                        hold_streak=session.hold_streak,
                        vram_pressure=vram_pressure,
                    )
                else:
                    vram_pressure = scaler.record_window(
                        config,
                        active_workers=min(config.k, count),
                        peak_vram_bytes=peak_vram_bytes,
                        warming_g=warming_during,
                        had_fault=had_fault,
                        total_vram_bytes=total_vram_bytes,
                    )
                    row.update(vram_pressure=vram_pressure)
                directories.append(window_dir)
                windows.append(row)
                last_data = data
                last_completed_items = completed_items
                print(json.dumps(dict(event="window_complete", **row)), flush=True)

            # Background warmups may span intermediate suite boundaries; only drain on the final suite.
            warmup_target = session.warming_g
            handoff_s = scaler.finish_warmup() if index + 1 == len(suites) else 0.0
            if pool.size != session.current.g:
                raise RuntimeError("pool and controller disagree at the suite boundary")
            if windows_per_suite > 1:
                merged = merge_shards(
                    benchmark=bench,
                    suite=suite,
                    episodes=args.episodes,
                    manifest_path=args.manifest,
                    shard_dirs=directories,
                )
                validation = merged["validation"]
            else:
                record_ids = [
                    rec["record_id"]
                    for d in last_completed_items
                    for rec in d.get("records", ())
                ]
                unique_ids = len(set(record_ids)) == len(record_ids)
                complete = unique_ids and len(record_ids) == suite_episodes
                validation = dict(
                    total_records=len(record_ids),
                    unique_record_ids=unique_ids,
                    coverage="complete" if complete else "incomplete",
                )
                merged = {**last_data, "merged": False, "validation": validation}
            tasks = merged["official_result"]["tasks"]
            if (
                validation["coverage"] != "complete"
                or not validation["unique_record_ids"]
                or episodes != suite_episodes
                or validation["total_records"] != episodes
                or len(tasks) != 10
                or any(t["episodes"] != args.episodes for t in tasks)
                or steps != merged["result"]["total_steps"]
            ):
                raise RuntimeError(f"full coverage check failed for {suite}: {validation}")
            cli.atomic_json(folder / "result.json", merged)
            formal_s = sum(w["formal_s"] for w in windows)
            results[suite] = dict(
                initial=asdict(initial),
                final=asdict(session.current),
                windows_per_suite=windows_per_suite,
                hold_streak_start=hold_streak_start,
                hold_streak_end=session.hold_streak,
                load_s=session.load_s,
                rejected=sorted(session.rejected),
                handoff_s=handoff_s,
                handoff_warmup_target_g=warmup_target,
                episodes=episodes,
                successes=merged["result"]["successes"],
                steps=steps,
                formal_s=formal_s,
                formal_wall_s=sum(w["formal_wall_s"] for w in windows),
                steps_per_s=steps / formal_s,
                recovered_worker_failures=sum(w["worker_failures"] for w in windows),
                vram_guard_events=sum(bool(w.get("vram_pressure")) for w in windows),
                windows=windows,
                validation=validation,
            )
            cli.atomic_json(out / "summary.json", summary())
            print(
                json.dumps(
                    dict(
                        event="suite_complete",
                        suite=suite,
                        steps=steps,
                        formal_s=formal_s,
                        windows_per_suite=windows_per_suite,
                        final=asdict(session.current),
                    )
                ),
                flush=True,
            )

        pool.scale(1)
        solo_end = replay_solo(pool, samples, args.solo_inputs, "end")
    finally:
        try:
            scaler.close()
        finally:
            pool.stop()
    if "torch" in sys.modules:
        raise RuntimeError("the DIT controller process imported torch")
    end_to_end = time.monotonic() - started
    cli.atomic_json(out / "references.json", calibration_report(solo_start, solo_end, baseline))
    cli.atomic_json(out / "summary.json", summary(solo_end, end_to_end))
    print(json.dumps(dict(event="complete", end_to_end_s=end_to_end)), flush=True)


if __name__ == "__main__":
    if sys.argv[1:2] == ["serve"]:
        trace_dir = Path(sys.argv.pop(2))
        sys.argv.pop(1)
        serve(trace_dir)
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        for name in (
            "manifest",
            "checkpoint",
            "weights",
            "model-python",
            "model-cwd",
            "solo-inputs",
            "output-dir",
        ):
            parser.add_argument(f"--{name}", type=Path, required=True)
        parser.add_argument("--baseline", type=Path)
        parser.add_argument("--suites", nargs="+")
        parser.add_argument("--episodes", type=int, default=50, help="episodes per task")
        parser.add_argument("--window-episodes", type=int, default=50)
        parser.add_argument(
            "--fixed-windows",
            action="store_true",
            help="use the requested window size without adjusting it to the worker and task counts",
        )
        args = parser.parse_args()
        if args.episodes < 1 or args.window_episodes < 1:
            parser.error("episode counts must be positive")
        run_experiment(args)

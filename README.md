# Rollout Autotune

A single-GPU, online threshold controller for closed-loop policy evaluation. It starts with 12 environment workers and 3 inference replicas. A decision changes one count by one. The host owns the workers, GPU telemetry and model processes. The library uses the Python standard library and runs independently of the evaluation framework.

## Running the controller

Install with `python -m pip install .` (Python 3.11+). Embed `rollout_autotune.run(runtime)` or launch `python -m rollout_autotune` as a subprocess. Both use the same control loop. Python callers can override the default grid limits with `limits=Config(32, 4)` and the cutoffs with `thresholds=Thresholds(...)`. The prior must fit the host. The CLI accepts `--max-workers` and `--max-replicas`.

The Python runtime implements two blocking methods.

- `configure(target: Config) -> bool` applies the requested counts. Acknowledge after the target is active. Check feasible worker/replica combinations before applying, drain or preserve in-flight episodes and per-episode model state, and retain completed work. On rejection return false with the previous configuration intact. Propagate an error if the previous configuration cannot be restored.
- `observe(current: Config) -> Window | None` returns deltas measured entirely under that configuration. A 30-second settled window is a starting point. Exclude initialization and reconfiguration intervals. End the loop with None when the suite finishes, including when it finishes partway through a window.

Over stdio the controller writes a JSON line and the host replies with one JSON line. For example, the controller first sends `{"op":"configure","k":12,"g":3}`; the host replies `true`. On `{"op":"observe","k":12,"g":3}` the host might return:

```json
{"worker_wait_s":252,"worker_live_s":360,"sm_active_s":15,"gpu_sampled_s":30,"matched_service_s":10.4,"matched_solo_s":10}
```

The controller then requests `{"op":"configure","k":12,"g":4}`. If the host replies `false`, the next observation is requested at (12, 3). To finish, reply `null` to an observation. To log or replay individual decisions from Python, call `decide(current, window)`. The result contains the three ratios, target and decision reason.

## Window measurements

| Signal | How the host measures it |
| --- | --- |
| Worker inference wait | Sum the wall time each active worker spent awaiting policy replies and divide by summed active worker wall time. Clip both to the window. Exclude idle workers with no assigned work. |
| GPU saturation proxy | Accumulate SM-active fraction × sample duration into `sm_active_s`, then divide by covered sample time. Use a single device shared by the replicas. A suitable source is DCGM SM Activity (field 1002). This measures active SM cycles, including memory stalls. Treat device-wide kernel-busy percentage as a different metric. |
| Replica contention proxy | Divide summed inference service time by the sum of expected solo service times for the same requests or batches. Measure from dispatch to device completion, excluding queue wait. Use warmed references from isolated execution on the same device, with matching model, precision, serving path and input/batch shape. Warmup observations may supply these references. Include a request in both totals only if its reference exists. A ratio of 1.30 means service took 30% longer. Other device activity can also cause slowdown. |

For raw host observations, use `Window.from_samples(workers=[(wait_s, live_s), ...], gpu=[(sm_fraction, duration_s), ...], batches=[(shape_key, service_s), ...], solo_by_shape=references)`. Use disjoint GPU intervals; report each completed batch once. The reference map stores warmed isolated service seconds per batch key. Unmatched batches contribute to neither total.

A missing metric has zero numerator and zero exposure. Missing saturation or wait disables rules that require it. A missing solo reference prevents speculative replica increases when multiple replicas already share the GPU. With one replica, there is no inter-replica competition yet. GPU measurement details are in the [NVIDIA DCGM profiling guide](https://docs.nvidia.com/datacenter/dcgm/latest/learn/modules/profiling.html).

## Initial decision rules

Apply the first matching rule per settled window. Limits default to K ≤ 32, g ≤ 4 and g ≤ K. The host can impose stricter pair-specific feasibility checks.

| Condition | Action |
| --- | --- |
| Multiple replicas and matched slowdown > 1.30 | Remove one replica |
| Wait > 0.60 and SM activity > 0.85 | Remove one worker |
| Wait < 0.25 and SM activity < 0.60 | Add one worker |
| Wait > 0.60 and SM activity < 0.60, with one replica or matched slowdown < 1.10 | Add one replica |
| Within the deadband, required observations unavailable, or the action hits a limit | Hold |

The thresholds are initial hypotheses for live evaluation. The fixed throughput grid motivated the prior and search range. It does not contain the matched per-window counters required to validate these rules.

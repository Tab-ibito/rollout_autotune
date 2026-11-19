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

## Step-fused DIT

The learned controller is opt-in. The threshold baseline remains available via the original Python API and CLI; its tagged revision is `baseline/threshold-v1`. The two controllers have distinct missing-telemetry behavior.

### Train

From the checkout, install with `python -m pip install '.[training]'`. The following command trains two separate networks and exports CPU weights:

```sh
python training/train.py --out runs/initial --worlds 25000 --epochs 3 --seed 82
```

To include recorded transitions, append `--real-windows windows.jsonl --real-changes changes.jsonl` once per run. These files use the threshold experiment's JSONL schema. Each completed window records config, next config (null on completion), episodes, steps, formal_s, total window seconds, wait, sm and slowdown. Each acknowledged change records after_episodes and seconds. The action chosen at the end of one window receives the elapsed time of the following window plus its disjoint transition time. The initial fixed placement contributes no policy action label. Supply contiguous complete runs of at most 65 windows to this historical importer.

Each synthetic task draws CPU and policy service capacities, replica scaling, switch costs, a memory limit and per-window work. Two independently sampled episodes on that same task share one causal history of at most 64 tokens; each episode starts with a reset indicator. TD targets and previous actions/rewards reset there. A rejected resize consumes time and the host keeps serving with its previous configuration. Behavior consists of the threshold policy and randomized threshold policies with 15% uniform feasible-action exploration. Each step completes a block of useful work. Negative elapsed time in units of 60 seconds is the reward; known switch time is an additional input. The simulator is an analytical prior.

The value and policy transformers have independent weights. Each has four pre-RMSNorm blocks, width 128, four attention heads, RoPE and a 352-wide SwiGLU. One step fuses the current measurements, availability and reset bits, previous requested action, previous transition time and previous reward. Training uses MSE on TD(λ) with γ = λ = 0.90, then freezes the value net. Policy loss is masked cross-entropy weighted by exp(standard-deviation-scaled one-step advantage / 0.4), clipped to [0.05, 20]. Both stages use AdamW, 3e-4 initial learning rate, 0.01 weight decay, 0.1 residual dropout, batch size 64 and cosine decay. Discounting favors near-term elapsed time. Evaluate changes on undiscounted suite wall time. The report includes a paired comparison on 128 unseen synthetic worlds, independent of the training seed stream.

### Deploy

Install `python -m pip install '.[dit]'` on the host. Deploy only `policy.npz`; the value network and PyTorch checkpoints serve offline training. For example:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m rollout_autotune.dit runs/initial/policy.npz
```

The JSONL protocol starts by requesting (12,3). Reply to configure with a boolean and to observe with a completed-window record such as:

```json
{"wait":0.54,"sm":0.80,"slowdown":1.4,"rate":160,"work_left":75000,"elapsed_s":30,"switch_s":0.5}
```

Use null for missing wait, SM activity or slowdown. Rate is effective environment steps/s; work_left is a decision-time estimate in the same step units. Supply total window elapsed seconds excluding the separately measured switch time; any completed rollout work counts toward the suite. Reply null instead of a record once the suite is complete. Python hosts can create a `rollout_autotune.dit.Session` for each suite and call `propose` then `acknowledge` for every decision, including holds.

The actor runs exclusively in NumPy with a per-session KV cache. It applies bounds and rejected-target masks before sampling the first three windows (temperature 0.5) and then selects the highest allowed logit. Missing wait or SM activity forces HOLD, including when measured slowdown is high. A rejected destination stays masked until a new session. The action order is HOLD, K+1, K−1, g+1, g−1. After the 64-token limit the predictor rebuilds the retained suffix for equivalence with a bounded full forward.

### SM-free policy

Use `--no-sm` during training to fit both networks with 16 input features. This removes
the SM fraction and its presence indicator. Training and inference select the same
columns. The policy artifact carries its feature schema. The standard DIT CLI can
load either checkpoint format. An SM-free session accepts observations without the
`sm` key and chooses among feasible actions when worker wait is available. Missing
worker wait still forces HOLD. Original 18-input policies keep their SM requirement.

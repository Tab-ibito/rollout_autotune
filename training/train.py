"""Train the two causal networks and export the policy-only inference artifact.

Stage 1 fits a 31-head joint critic ``[V(H_t), Q(H_t, a), behavior_logits(H_t, a)]``:
  - ``Q`` uses standard MSE for risk-neutral expected returns over latent startup costs;
  - ``V`` uses asymmetric expectile loss (tau=0.62) against detached ``Q`` for in-sample stitching;
  - ``behavior_logits`` estimates action propensities ``mu(a | H_t)``.
Stage 2 freezes the critic, computes debiased advantages ``Q(H_t, a_t) - V(H_t)``,
and trains the 15-action actor via inverse-propensity-weighted cross-entropy before
verifying numerical parity with ``rollout_autotune.dit.CPUModel``.
"""

import argparse
from collections import Counter, defaultdict, deque
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

from rollout_autotune import Config
from rollout_autotune.dit import (
    ACTIONS,
    CONTEXT,
    FORMAT,
    INPUT,
    NO_SM_COLUMNS,
    NO_SM_FORMAT,
    NUM_ACTIONS,
    CPUModel,
    action_mask,
    features,
    model_inputs,
    step_metrics,
    token,
)

if __package__:
    from .model import Transformer
    from .simulator import Worlds, make_data
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from training.model import Transformer
    from training.simulator import Worlds, make_data


def _match_action(delta):
    """Map a recorded ``(dk, dg)`` configuration jump to the closest discrete action in ``ACTIONS``."""
    if delta in ACTIONS:
        return ACTIONS.index(delta)
    dk, dg = delta
    # Snap arbitrary historical deltas to the closest discrete multi-step action in the same direction.
    candidates = [
        i
        for i, (ak, ag) in enumerate(ACTIONS)
        if (ak == 0 or (ak > 0) == (dk > 0))
        and (ag == 0 or (ag > 0) == (dg > 0))
        and (ak != 0 or ag != 0)
    ]
    if not candidates:
        return 0
    return min(candidates, key=lambda i: abs(ACTIONS[i][0] - dk) + 3 * abs(ACTIONS[i][1] - dg))


def real_history(windows_path, changes_path, limits=Config(32, 4), load_s=None):
    """Convert recorded hardware ``(windows.jsonl, changes.jsonl)`` logs into CONTEXT-sized chunks."""
    rows = [json.loads(line) for line in Path(windows_path).read_text().splitlines() if line.strip()]
    change_rows = [
        json.loads(line) for line in Path(changes_path).read_text().splitlines() if line.strip()
    ]
    if len(rows) < 2:
        raise ValueError("real window log must contain at least two windows")

    changes_by_idx = {}
    changes_by_ep = defaultdict(deque)
    for rec in change_rows:
        sec = float(rec["seconds"])
        if "index" in rec and rec["index"] is not None:
            changes_by_idx[int(rec["index"])] = sec
        elif "after_episodes" in rec and rec["after_episodes"] is not None:
            changes_by_ep[int(rec["after_episodes"])].append(sec)

    total_episodes = sum(int(r.get("episodes", 0)) for r in rows)
    total_steps = sum(float(r.get("steps", 0.0)) for r in rows)
    used_episodes = 0
    used_steps = 0.0
    rejected = set()
    base_rate = None
    cur_load_s = float(load_s) if load_s is not None else None
    prev_action = None
    cost_feat = 0.0
    reward_feat = 0.0

    seq_x = []
    seq_masks = []
    seq_actions = []
    seq_rewards = []

    for t, (r, nxt) in enumerate(zip(rows, rows[1:])):
        if r.get("next") is None:
            break
        used_episodes += int(r.get("episodes", 0))
        used_steps += float(r.get("steps", 0.0))
        c = Config(**r["config"])
        target = Config(**r["next"])
        actual_next = Config(**nxt["config"])
        delta = (target.k - c.k, target.g - c.g)
        if actual_next != target and not (target != c and actual_next == c):
            raise ValueError(
                f"discontinuous configuration log at window {t}: target={target}, next={actual_next}"
            )
        if r.get("load_s") is not None:
            cur_load_s = float(r["load_s"])

        formal_s = float(r.get("formal_s") or r.get("seconds") or 0.0)
        rate = float(r.get("steps", 0.0)) / formal_s if formal_s > 0 else 0.0
        if base_rate is None and rate > 0:
            base_rate = rate

        if r.get("work_left") is not None:
            work_left = max(float(r["work_left"]), 0.0)
        elif used_episodes > 0 and total_episodes >= used_episodes:
            work_left = max((total_episodes - used_episodes) * (used_steps / used_episodes), 0.0)
        else:
            work_left = max(total_steps - used_steps, 0.0)

        state = features(
            c.k,
            c.g,
            r.get("wait"),
            r.get("sm"),
            r.get("slowdown"),
            rate,
            work_left,
            t == 0,
            limits=limits,
            load_s=cur_load_s,
        )
        tok = token(state, prev_action, cost_feat, reward_feat)
        act = _match_action(delta)
        mask_t = action_mask(c, limits, rejected)
        mask_t[act] = True

        if t in changes_by_idx:
            switch_s = changes_by_idx[t]
        elif target != c and changes_by_ep[used_episodes]:
            switch_s = changes_by_ep[used_episodes].popleft()
        elif target != c:
            nxt_total = float(nxt.get("seconds") or nxt.get("formal_wall_s") or 0.0)
            nxt_formal = float(nxt.get("formal_s") or nxt_total)
            switch_s = max(nxt_total - nxt_formal, 0.0)
        else:
            switch_s = 0.0

        dg_accepted = actual_next.g - c.g
        if dg_accepted > 0 and switch_s > 0:
            inferred = switch_s / dg_accepted
            cur_load_s = inferred if cur_load_s is None else 0.5 * cur_load_s + 0.5 * inferred

        nxt_formal_s = float(nxt.get("formal_s") or nxt.get("seconds") or 1.0)
        nxt_wall_s = float(nxt.get("formal_wall_s") or nxt.get("seconds") or nxt_formal_s)
        nxt_rate = float(nxt.get("steps", 0.0)) / nxt_formal_s if nxt_formal_s > 0 else 0.0
        ref_rate = base_rate if base_rate is not None else max(nxt_rate, 1.0)
        next_cost_feat, next_reward_feat = step_metrics(ref_rate, nxt_rate, nxt_wall_s, switch_s)

        if target != c and actual_next == c:
            rejected.add((target.k, target.g))

        seq_x.append(tok)
        seq_masks.append(mask_t)
        seq_actions.append(act)
        seq_rewards.append(next_reward_feat)
        prev_action = act
        cost_feat = next_cost_feat
        reward_feat = next_reward_feat

    m = len(seq_x)
    if m == 0:
        raise ValueError("real window log contains no actionable transitions")
    completed_run = rows[-1].get("next") is None
    if not completed_run and m < 2:
        raise ValueError("incomplete real window log needs at least two transitions for bootstrap")

    # Pack into bounded CONTEXT windows; long runs use 50%-overlapping causal context without double-counting losses.
    starts = [0]
    while starts[-1] + CONTEXT < m:
        starts.append(starts[-1] + CONTEXT // 2)

    num_chunks = len(starts)
    x = np.zeros((num_chunks, CONTEXT, INPUT), np.float32)
    masks = np.zeros((num_chunks, CONTEXT, NUM_ACTIONS), bool)
    actions = np.zeros((num_chunks, CONTEXT), np.int64)
    rewards = np.zeros((num_chunks, CONTEXT), np.float32)
    done = np.zeros((num_chunks, CONTEXT), bool)
    covered = 0

    for ci, start in enumerate(starts):
        end = min(start + CONTEXT, m)
        length = end - start
        x[ci, :length] = np.asarray(seq_x[start:end], dtype=np.float32)
        actions[ci, :length] = np.asarray(seq_actions[start:end], dtype=np.int64)
        rewards[ci, :length] = np.asarray(seq_rewards[start:end], dtype=np.float32)
        # Reserve the last slot as lookahead bootstrap when truncated or when the log ended mid-suite.
        is_terminal_chunk = (end == m) and completed_run
        active_end = end if is_terminal_chunk else end - 1
        for global_t in range(max(covered, start), active_end):
            local_t = global_t - start
            masks[ci, local_t] = seq_masks[global_t]
        if is_terminal_chunk:
            done[ci, length - 1] = True
        covered = max(covered, active_end)

    return x, masks, actions, rewards, done


def td_targets(r, terminal, valid, v, gamma, lam):
    """Compute TD(lambda) return targets with lookahead bootstrap at chunk truncation boundaries."""
    next_values = F.pad(v[:, 1:], (0, 1))
    following = torch.zeros_like(v[:, -1])
    out = torch.zeros_like(v)
    for t in reversed(range(v.shape[1])):
        boot = (~terminal[:, t]) * ((1.0 - lam) * next_values[:, t] + lam * following)
        # When valid[t+1] is False only because a long chunk reserved t+1 as lookahead, use v[:, t+1].
        if t + 1 < v.shape[1]:
            use_v_direct = valid[:, t] & (~valid[:, t + 1]) & (~terminal[:, t])
            boot = torch.where(use_v_direct, next_values[:, t], boot)
        step_ret = r[:, t] + gamma * boot
        following = torch.where(valid[:, t], step_ret, 0.0)
        out[:, t] = following
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--worlds", type=int, default=30000)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--seed", type=int, default=82)
    p.add_argument("--real-windows", type=Path, action="append", default=[])
    p.add_argument("--real-changes", type=Path, action="append", default=[])
    p.add_argument(
        "--real-weight",
        type=float,
        default=20.0,
        help="Relative loss weight applied to recorded real transitions",
    )
    p.add_argument(
        "--no-sm",
        action="store_true",
        help="Exclude SM activity and its availability bit from both networks",
    )
    p.add_argument(
        "--cold-scale",
        action="store_true",
        help="Train with blocking stop-the-world replica launches instead of background hot-scaling",
    )
    args = p.parse_args()
    use_sm = not args.no_sm
    hot_scale = not args.cold_scale
    input_dim = INPUT if use_sm else len(NO_SM_COLUMNS)
    if len(args.real_windows) != len(args.real_changes):
        p.error("each real window log requires its matching configuration-change log")
    if args.real_weight <= 0 or not math.isfinite(args.real_weight):
        p.error("--real-weight must be finite and positive")

    args.out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda" and torch.cuda.is_bf16_supported()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    # Critic outputs: [V(H_t), Q(H_t, 0..A-1), behavior_logits(H_t, 0..A-1)].
    critic = Transformer(1 + 2 * NUM_ACTIONS, input_dim=input_dim).to(device)
    actor = Transformer(NUM_ACTIONS, input_dim=input_dim).to(device)

    data = make_data(Worlds(args.worlds, rng), rng, use_sm=use_sm, hot_scale=hot_scale)
    sample_weights = np.ones(len(data[0]), dtype=np.float32)
    is_real_row = np.zeros(len(data[0]), dtype=bool)
    nr = 0
    for windows, changes in zip(args.real_windows, args.real_changes):
        real = real_history(windows, changes)
        nr += int(real[1].any(-1).sum())
        data = tuple(np.concatenate((a, b)) for a, b in zip(data, real))
        sample_weights = np.concatenate(
            (sample_weights, np.full(len(real[0]), args.real_weight, dtype=np.float32))
        )
        is_real_row = np.concatenate((is_real_row, np.ones(len(real[0]), dtype=bool)))

    counts = Counter(data[2][data[1].any(-1)].tolist())
    print(
        "dataset",
        args.worlds * 2,
        "synthetic trajectories,",
        nr,
        "real decisions",
        counts,
        flush=True,
    )
    data = (np.ascontiguousarray(model_inputs(data[0], use_sm)), *data[1:])
    tensors = [torch.from_numpy(d).to(device) for d in data]
    row_weights = torch.from_numpy(sample_weights).to(device)[:, None].expand(-1, CONTEXT)
    real_flags = torch.from_numpy(is_real_row).to(device)[:, None].expand(-1, CONTEXT)
    del data
    x, masks, actions, rewards, done = tensors
    valid = masks.any(-1)
    gamma, lam, beta = 0.90, 0.80, 0.4
    expectile_tau = 0.62
    losses = {}
    batches = 128 if device == "cuda" else 64

    for phase, net in [("value", critic), ("policy", actor)]:
        optimizer = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-2)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, args.epochs * ((len(x) + batches - 1) // batches)
        )
        if phase == "policy":
            critic.eval().requires_grad_(False)
            with torch.no_grad():
                adv_chunks, prop_chunks = [], []
                for idx_slice in torch.arange(len(x), device=device).split(batches):
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                        out = critic(x[idx_slice]).float()
                    v_val = out[..., 0]
                    q_all = out[..., 1 : 1 + NUM_ACTIONS]
                    q_taken = q_all.gather(-1, actions[idx_slice].unsqueeze(-1)).squeeze(-1)
                    # Risk-neutral expected advantage Q(H_t, a_t) - V(H_t) eliminates Jensen optimism
                    # over latent switch costs; blend empirical TD residual on sparse real rows.
                    q_adv = q_taken - v_val
                    td_adv = (
                        rewards[idx_slice]
                        + gamma * (~done[idx_slice]) * F.pad(v_val[:, 1:], (0, 1))
                        - v_val
                    )
                    adv_chunks.append(
                        torch.where(real_flags[idx_slice], 0.5 * q_adv + 0.5 * td_adv, q_adv)
                    )
                    bc_logits = out[..., 1 + NUM_ACTIONS :].masked_fill(
                        ~masks[idx_slice], -torch.inf
                    )
                    bc_probs = torch.softmax(bc_logits, dim=-1)
                    taken_p = bc_probs.gather(-1, actions[idx_slice].unsqueeze(-1)).squeeze(-1)
                    prop_chunks.append(torch.where(valid[idx_slice], taken_p, 1.0))
                advantage = torch.cat(adv_chunks)
                propensities = torch.cat(prop_chunks)

        for epoch in range(args.epochs):
            if phase == "value":
                critic.eval()
                with torch.no_grad():
                    v_parts = []
                    for a in x.split(batches):
                        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                            v_parts.append(critic(a)[..., 0].float())
                    v = torch.cat(v_parts)
                    # Epoch 0 uses multi-step TD(lam) to seed horizon returns rapidly; later epochs
                    # transition toward 1-step Bellman backups onto the expectile value function.
                    lam_epoch = lam if epoch == 0 else 0.25
                    q_targets = torch.cat(
                        [
                            td_targets(
                                rewards[i : i + 512],
                                done[i : i + 512],
                                valid[i : i + 512],
                                v[i : i + 512],
                                gamma,
                                lam_epoch,
                            )
                            for i in range(0, len(x), 512)
                        ]
                    )
            net.train()
            epoch_loss = []
            for ids in torch.randperm(len(x), device=device).split(batches):
                ix = valid[ids]
                if not ix.any():
                    continue
                rw = row_weights[ids][ix]
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    h = net(x[ids]).float()
                if phase == "value":
                    v_pred = h[..., 0][ix]
                    q_all = h[..., 1 : 1 + NUM_ACTIONS][ix]
                    act_ix = actions[ids][ix]
                    q_taken = q_all.gather(-1, act_ix.unsqueeze(-1)).squeeze(-1)
                    # Standard MSE on Q guarantees risk-neutral expectation over unobserved switch costs.
                    q_loss = (rw * (q_taken - q_targets[ids][ix]).square()).sum() / rw.sum()
                    # Expectile loss on V against detached Q stitches multi-step actions in-sample.
                    diff = q_taken.detach() - v_pred
                    exp_w = torch.where(diff >= 0, expectile_tau, 1.0 - expectile_tau)
                    v_loss = (rw * exp_w * diff.square()).sum() / rw.sum()
                    bc_logits = h[..., 1 + NUM_ACTIONS :][ix].masked_fill(
                        ~masks[ids][ix], -torch.inf
                    )
                    bc_loss = F.cross_entropy(bc_logits, act_ix)
                    loss = q_loss + v_loss + 0.25 * bc_loss
                else:
                    logits = h[ix].masked_fill(~masks[ids][ix], -torch.inf)
                    adv = advantage[ids][ix]
                    scale = adv.std(unbiased=False).clamp(min=0.05)
                    norm_adv = adv / scale
                    logw = (norm_adv / beta).clamp(math.log(0.02), math.log(20.0))
                    inv_prop = (1.0 / propensities[ids][ix].clamp(min=0.14)).pow(0.55)
                    # Keep all negative-advantage synthetic switches uncensored so costly load_s losses
                    # push down risky moves; only prevent real_weight amplification on negative real moves.
                    eff_rw = torch.where(real_flags[ids][ix] & (adv <= 0), 1.0, rw)
                    weight = eff_rw * inv_prop * logw.exp()
                    ce = F.cross_entropy(logits, actions[ids][ix], reduction="none")
                    loss = (weight * ce).sum() / weight.sum().clamp(min=1e-6)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                optimizer.step()
                schedule.step()
                epoch_loss.append(loss.detach())
            val = torch.stack(epoch_loss).mean().item()
            losses.setdefault(phase, []).append(val)
            print(phase, epoch + 1, val, flush=True)
        torch.save(net.state_dict(), args.out / (phase + ".pt"))

    actor.eval()
    path = args.out / "policy.npz"
    np.savez(
        path,
        format_version=FORMAT if use_sm else NO_SM_FORMAT,
        **{k: v.cpu().numpy() for k, v in actor.state_dict().items()},
    )
    # Compare every token including bounded-context eviction to an uncached full forward.
    cpu = CPUModel(path)
    cpu_actor = Transformer(NUM_ACTIONS, input_dim=input_dim).eval()
    cpu_actor.load_state_dict(actor.state_dict())
    probe = rng.normal(size=(72, INPUT)).astype(np.float32)
    with torch.no_grad():
        err = max(
            float(
                np.max(
                    np.abs(
                        cpu.logits(t)
                        - cpu_actor(
                            torch.from_numpy(model_inputs(probe[max(0, i - 63) : i + 1], use_sm))[
                                None
                            ]
                        )[0, -1].numpy()
                    )
                )
            )
            for i, t in enumerate(probe)
        )
    assert err < 1e-4, err

    def validation(policy):
        vr = np.random.default_rng(args.seed + 9001)
        worlds = Worlds(128, vr, fixed_limits=True)
        *_, wall_s = make_data(
            worlds,
            vr,
            prior=policy,
            use_sm=use_sm,
            hot_scale=hot_scale,
            return_wall_s=True,
        )
        return wall_s[0]

    @torch.no_grad()
    def choose(x_buf, idx, slot, mask, t):
        history = torch.from_numpy(model_inputs(x_buf[idx, : max(slot) + 1], use_sm)).to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            out = actor(history).float()
        logits = (
            out[torch.arange(len(idx), device=device), torch.as_tensor(slot, device=device)]
            .cpu()
            .numpy()
        )
        logits = np.where(mask, logits, -np.inf)
        return logits.argmax(1)

    base = validation("threshold")
    new = validation(choose)
    report = dict(
        uses_sm=use_sm,
        hot_scale=hot_scale,
        input_dim=input_dim,
        num_actions=NUM_ACTIONS,
        seed=args.seed,
        synthetic_worlds=args.worlds,
        synthetic_trajectories=2 * args.worlds,
        real_transitions=nr,
        train_decisions=int(valid.sum()),
        actions=dict(counts),
        gamma=gamma,
        td_lambda=lam,
        expectile_tau=expectile_tau,
        beta=beta,
        loss=losses,
        actor_parameters=sum(p.numel() for p in actor.parameters()),
        cpu_logit_max_error=err,
        device=device,
        elapsed_s=time.monotonic() - start,
        validation=dict(
            held_out_synthetic_worlds=len(base),
            threshold_mean_s=float(base.mean()),
            policy_mean_s=float(new.mean()),
            fraction_faster=float((new < base).mean()),
            wins=int((new < base).sum()),
            ties=int((new == base).sum()),
            losses=int((new > base).sum()),
        ),
    )
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()

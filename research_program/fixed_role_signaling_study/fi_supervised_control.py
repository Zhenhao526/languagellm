"""Supervised FI action control for the fixed-role neural architecture."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import design, environment, model


SUPERVISED_UPDATES = 1000
SUPERVISED_BATCH = 256
SUPERVISED_LR = design.LEARNING_RATE


def labelled_batch(seed: int, task: str, count: int, *, evaluation: bool, update: int):
    ep = design.episode_stream(seed, "scarce", task, count, evaluation=evaluation, update=update)
    site = np.asarray(ep["site_type"], dtype=np.int8)
    goal = np.asarray(ep["goal"], dtype=np.int8)[:, 0]
    remain = np.stack([ep["capacity"], ep["capacity"]], axis=1).astype(np.int16)
    xs = np.zeros((design.HORIZON, count, design.OBS_DIM), dtype=np.float64)
    labels = np.zeros((design.HORIZON, count), dtype=np.int64)
    recv = np.zeros((count, design.RECV_DIM), dtype=np.float64)
    recv[:, -1] = 1.0
    for t in range(design.HORIZON):
        for i in range(count):
            target = int(goal[i, t])
            candidates = [s for s in (0, 1)
                          if remain[i, s] > 0 and int(site[i, s]) == target]
            action = candidates[0] + 1 if t >= design.ACTION_START and candidates else 0
            labels[t, i] = action
            xs[t, i] = design.encode_observation(
                own_goal=0, local_type=int(site[i, 1]),
                local_inventory=int(remain[i, 1]), last_result=0, time=t,
                partner_goal=target, remote_type=int(site[i, 0]),
                remote_inventory=int(remain[i, 0]), information="FI",
                sender_role=False)
            if action:
                remain[i, action - 1] -= 1
    return ep, xs, labels, recv


def supervised_step(net, xs, labels, recv):
    B = xs.shape[1]
    hs = []
    probs = []
    caches = []
    hprev = np.zeros((B, design.HIDDEN), dtype=np.float64)
    for t in range(design.HORIZON):
        h, ml, al, cache = model.forward(net, xs[t], recv, hprev, "recurrent")
        hs.append(h); probs.append(model.softmax(al)); caches.append(cache)
        hprev = h
    grads = model.zero_grads([net])[0]
    carry = np.zeros((B, design.HIDDEN), dtype=np.float64)
    loss = 0.0
    for t in range(design.HORIZON - 1, -1, -1):
        p = probs[t]
        y = labels[t]
        active = t >= design.ACTION_START
        one = np.zeros_like(p)
        one[np.arange(B), y] = 1.0
        dal = (p - one) / B
        if not active:
            dal[:] = 0.0
        loss += float(-np.log(np.maximum(p[np.arange(B), y], 1e-300)).mean()) if active else 0.0
        x0, recv_t, hprev, pre, h = caches[t]
        dh = np.einsum("bk,hk->bh", dal, net["W_act"]) + carry
        dpre = dh * (1.0 - h * h)
        grads["W_act"] += np.einsum("bi,bj->ij", h, dal)
        grads["b_act"] += dal.sum(0)
        grads["W_obs"] += np.einsum("bi,bh->ih", x0, dpre)
        grads["W_recv"] += np.einsum("bi,bh->ih", recv_t, dpre)
        grads["W_h"] += np.einsum("bi,bh->ih", hprev, dpre)
        grads["b_h"] += dpre.sum(0)
        carry = np.einsum("bi,hi->bh", dpre, net["W_h"])
    return grads, loss / max(design.HORIZON - design.ACTION_START, 1)


def evaluate(net, seed: int, task: str, evaluation: bool):
    ep = design.episode_stream(seed, "scarce", task, 4096, evaluation=evaluation)
    scout = model.make_networks(seed, "recurrent")[0]
    nets = [scout, {k: v.copy() for k, v in net.items()}]
    tr = environment.run_episode_batch(nets, ep, "recurrent", "FI", "silent", sample=False, collect=False)
    oracle = environment.oracle_team_return(ep)
    return {
        "split": "heldout" if evaluation else "training_support",
        "team_return_mean": float(tr["team_return"].mean()),
        "oracle_team_return_mean": float(oracle.mean()),
        "oracle_regret_mean": float((oracle - tr["team_return"]).mean()),
        "action_histogram": np.bincount(tr["actions"].ravel(), minlength=design.ACTION_COUNT).tolist(),
    }


def run(seeds=(68101, 68102, 68103, 68104), updates: int = SUPERVISED_UPDATES):
    rows = []
    for seed in seeds:
        net = model.make_network(seed, 1, "recurrent")
        opt = {k: {"m": np.zeros_like(v), "v": np.zeros_like(v)} for k, v in net.items()}
        losses = []
        for update in range(1, updates + 1):
            _, xs, labels, recv = labelled_batch(seed, "persistent", SUPERVISED_BATCH,
                                                  evaluation=False, update=update)
            grads, loss = supervised_step(net, xs, labels, recv)
            model.adam_step([net], [grads], [opt], update)
            losses.append(loss)
        support = evaluate(net, seed, "persistent", False)
        heldout = evaluate(net, seed, "persistent", True)
        rows.append({"seed": seed, "updates": updates,
                     "final_supervised_loss": float(losses[-1]),
                     "mean_last_100_loss": float(np.mean(losses[-100:])),
                     "training_support": support, "heldout": heldout})
    return {"schema": "fixed_role_fi_supervised_control_v1", "status": "passed",
            "updates": updates, "batch_size": SUPERVISED_BATCH, "learning_rate": SUPERVISED_LR,
            "rows": rows,
            "source_note": "Worker is trained on explicit FI oracle action labels; no messages or self-play credit assignment."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--updates", type=int, default=SUPERVISED_UPDATES)
    args = parser.parse_args()
    answer = run(updates=args.updates)
    Path(args.out).write_text(json.dumps(answer, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print(json.dumps(answer, ensure_ascii=False, sort_keys=True))

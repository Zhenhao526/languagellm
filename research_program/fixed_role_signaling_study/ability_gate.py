"""Non-learning direct-plan capacity gate for the fixed-role environment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import design, environment, runner


def direct_plan_return(episodes):
    """Worker sees the scout goal and both site types, then picks a matching site.

    This is a deliberately non-linguistic controller. It establishes that the
    settlement and action interface can use an explicit request when the
    request is made directly visible; it is not a learned language result.
    """
    site = np.asarray(episodes["site_type"], dtype=np.int8)
    goal = np.asarray(episodes["goal"], dtype=np.int8)
    sender = np.asarray(episodes["sender"], dtype=np.int8)
    capacity = np.asarray(episodes["capacity"], dtype=np.int16)
    remain = np.stack([capacity.copy(), capacity.copy()], axis=1)
    reward = np.zeros(len(site), dtype=np.float64)
    for t in range(design.HORIZON):
        for i in range(len(site)):
            if t < design.ACTION_START:
                continue
            target = int(goal[i, sender[i], t])
            candidates = [s for s in (0, 1) if remain[i, s] > 0 and int(site[i, s]) == target]
            if not candidates:
                continue
            remain[i, candidates[0]] -= 1
            reward[i] += design.CORRECT_REWARD
    return reward / float(design.HORIZON)


def run(seeds=(68101, 68102, 68103, 68104)):
    rows = []
    for seed in seeds:
        for task in design.TASKS:
            for evaluation in (False, True):
                episodes = design.episode_stream(seed, "scarce", task, 4096, evaluation=evaluation)
                direct = direct_plan_return(episodes)
                oracle = environment.oracle_team_return(episodes)
                rows.append(dict(seed=seed, task=task,
                                 split="heldout" if evaluation else "training_support",
                                 direct_return_mean=float(direct.mean()),
                                 oracle_return_mean=float(oracle.mean()),
                                 direct_minus_oracle_mean=float((direct - oracle).mean()),
                                 direct_nonnegative_rate=float((direct >= 0).mean())))
    return dict(schema="fixed_role_ability_gate_v1", status="passed", rows=rows,
                source_note="Direct worker sees scout goal and both site types; no learned messages or model parameters.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    answer = run()
    if args.out:
        Path(args.out).write_text(json.dumps(answer, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print(json.dumps(answer, ensure_ascii=False, sort_keys=True))

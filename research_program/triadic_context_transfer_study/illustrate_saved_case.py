"""Read all live policies for the one statically preselected example; no inference."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from research_program.triadic_task import environment as env

HERE = Path(__file__).resolve().parent
ALPHABET = ("@", "#", "$", "%", "&", "*", "+", "~")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def packet_text(packet):
    return ["".join(ALPHABET[int(x)] for x in window) for window in packet]


def execute(run, out):
    run, out = Path(run).resolve(), Path(out).resolve()
    result_path = run / "execution/results.json"
    results = read(result_path)
    assert results["status"] == "completed"
    selected_path = HERE / "static_illustration_001.json"
    selected = read(selected_path)
    assert selected["model_outcomes_read"] == 0
    row, shift, part = (selected[k] for k in ("dataset_row", "shift_index", "partition"))
    sender, listener = (selected["case"][k] for k in ("sender", "listener"))
    lookup = {(r["seed"], r["condition"], r["partition"], r["mode"], r["shift_index"], r["direction"]): r
              for r in results["records"]}
    sources = {str(result_path): sha(result_path), str(selected_path): sha(selected_path),
               str(Path(__file__).resolve()): sha(__file__), str(Path(env.__file__).resolve()): sha(env.__file__)}
    records = []
    for seed in (51101, 51102, 51103, 51104):
        for condition in ("PL_live", "LL_live"):
            for direction in (0, 1):
                for mode in ("remote_same_both", "remote_opposite_both"):
                    source = lookup[seed, condition, part, mode, shift, direction]
                    path = Path(source["path"])
                    assert not source["is_silent_alias"] and sha(path) == source["data_sha256"]
                    sources[str(path)] = source["data_sha256"]
                    with np.load(path, allow_pickle=False) as data:
                        assert int(data["dataset_rows"][row]) == row
                        recipient = int(data["recipient_indices"][row])
                        donor = int(data["donor_indices"][row])
                        assert recipient == selected["recipient_indices"][direction]
                        endpoint = direction if mode == "remote_same_both" else 1 - direction
                        assert donor == selected["donor_indices"][endpoint]
                        actions = data["action_indices"][row].tolist()
                        target = selected["recipient_correct_actions"][1 - direction][listener]
                        current = selected["recipient_correct_actions"][direction][listener]
                        records.append(dict(seed=seed, condition=condition, direction=direction, mode=mode,
                            recipient_index=recipient, donor_index=donor,
                            donor_packets=data["donor_packets"][row].tolist(),
                            donor_packet_symbols=packet_text(data["donor_packets"][row]),
                            self_generated_messages=data["messages"][row].tolist(),
                            action_indices=actions,
                            listener_action=env.all_actions(env.AGENTS[listener])[actions[listener]],
                            current_correct_action=current, target_correct_action=target,
                            current_apt=actions[listener] == current, target_apt=actions[listener] == target,
                            native_reward=float(data["greedy_reward"][row]),
                            counterfactual_reward=float(data["counterfactual_reward"][row]),
                            input_equals_donor_fixed_endpoints=data["action_input_equals_donor"][row, listener].tolist(),
                            source_path=str(path), source_sha256=source["data_sha256"]))
    assert len(records) == 32
    out.mkdir(exist_ok=False)
    payload = dict(status="completed_read_only_illustration", at=datetime.now(timezone.utc).isoformat(),
        selection=selected, records=records, source_sha256=sources,
        neural_forwards=0, parameter_loads=0, training_updates=0,
        scope="One case selected by static row order before outcomes; all eight live policies and both directions, no success selection. Not a new population estimate or a symbol meaning assignment.")
    (out / "illustration.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    lines = ["# 预先选定案例的实际消息与行动", "",
        "案例在查看干预结果前，按静态第一行与首个合格供体选定。列出四个初始化、两种信息条件、两个方向的全部结果；不按成功挑选。这里的符号是两窗实际供体出站包，不能据此为某个字符指定词义。", "",
        "唯一正确搭档为A与C。方向0检验木材→纤维，C的当前／反事实正确动作编号为1／9；方向1反过来为9／1。编号对应完整伙伴、位置与目的地动作。", "",
        "| 初始化 | 条件 | 方向 | 同需求包 W1 / W2 | 相反需求包 W1 / W2 | C动作编号 同 / 反 | 反事实命中 同 / 反 |", "|---|---|---|---|---|---|---|"]
    for i in range(0, len(records), 2):
        same, opposite = records[i:i + 2]
        strings = [" / ".join("`" + x + "`" for x in r["donor_packet_symbols"]) for r in (same, opposite)]
        lines.append(f"| {same['seed']} | {same['condition']} | {same['direction']} | {strings[0]} | {strings[1]} | {same['action_indices'][listener]} / {opposite['action_indices'][listener]} | {int(same['target_apt'])} / {int(opposite['target_apt'])} |")
    lines += ["", "逐项动作、全部自生消息、原生／反事实收益、完整输入相等标记及文件哈希见[原始提取记录](illustration.json)。本表仅帮助理解干预，不替代全域加权结果。", ""]
    (out / "实际案例.md").write_text("\n".join(lines))
    return dict(status=payload["status"], records=len(records), neural_forwards=0,
                output=str(out), illustration_sha256=sha(out / "illustration.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    print(json.dumps(execute(args.run, args.out), ensure_ascii=False))

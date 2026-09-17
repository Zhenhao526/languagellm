"""Prepare/execute a post-hoc v0.6 greedy receiver coverage trajectory.

Import and prepare use only the standard library. Execute alone imports Torch,
loads the small saved interfaces with weights_only=True, and evaluates receive.
No visual backbone, feature bank, sender inference, optimizer or training is used.
"""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
import hashlib
import importlib
import importlib.metadata
from itertools import permutations, product
import json
from pathlib import Path
import platform
import shutil
import sys
import traceback
from zoneinfo import ZoneInfo


WORK = Path(__file__).resolve().parents[1]
SOURCE = WORK / "redesign_v0.6/results/recombination_001/protocol_analysis.json"
DEFAULT_OUTPUT = WORK / "research_program/receiver_coverage_trajectory_v06_20260915"
CHECKPOINTS = (0, 100, 300, 600, 1200, 1800, 2100, 2400)
SEEDS = (25101, 25102, 25103, 25104)
CONDITIONS = tuple(f"split{s}_{kind}" for s in (1, 2, 3)
                   for kind in ("course", "mixed", "direct", "atomic_direct")) + ("full_direct", "full_blocked")
CODE_FILES = (
    "research_program/receiver_coverage_trajectory.py",
    "redesign_v0.6/camp.py", "redesign_v0.6/run_experiment.py",
    "redesign_v0.6/analyze_protocols.py", "redesign_v0.6/analysis_core.py",
    "redesign_v0.4/agents.py", "redesign_v0.4/run_pilot.py", "redesign_v0.4/resource_env.py",
)
CATEGORIES = ("same_code_both_correct", "both_marginals_but_no_joint_code",
              "exactly_one_marginal_reachable", "neither_marginal_reachable")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_bytes())


def write_json_new(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def proportion(n, d):
    return {"numerator": int(n), "denominator": int(d), "rate": int(n) / int(d) if d else None}


def runtime_versions():
    packages = {}
    for name in ("numpy", "torch"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": packages}


def menu_gather_source_check(path):
    """Check the reviewed implementation uses menu only in the final gather."""
    tree = ast.parse(Path(path).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "CampAgent")
    receive = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "receive")
    references = [n for n in ast.walk(receive) if isinstance(n, ast.Name) and n.id == "menu"]
    assignment = next(n for n in receive.body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "logits" for t in n.targets))
    value = assignment.value
    require(len(references) == 1 and isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute) and value.func.attr == "gather"
            and len(value.args) == 2 and isinstance(value.args[0], ast.Constant)
            and value.args[0].value == 1 and isinstance(value.args[1], ast.Name)
            and value.args[1].id == "menu", "Menu is not exclusively the reviewed final location-logit gather")
    require(isinstance(receive.body[-1], ast.Return)
            and isinstance(receive.body[-1].value, ast.Tuple)
            and isinstance(receive.body[-1].value.elts[0], ast.Name)
            and receive.body[-1].value.elts[0].id == "logits", "Returned action logits changed")
    return {"menu_name_uses": 1, "only_menu_use": "location_logits[rows, goal].gather(1, menu)",
            "receive_ast_sha256": hashlib.sha256(ast.dump(receive, include_attributes=False).encode()).hexdigest(),
            "unique_max_implication": "Permuting entries of a finite vector with a unique maximum preserves its maximizing physical site.",
            "tie_rule": "An exact maximum tie triggers the full 49-code x 2-goal x 720-menu receive enumeration."}


def plan_markdown(plan):
    return "\n".join([
        "# DINO v0.6：接收覆盖轨迹冻结计划", "",
        "状态：已准备、尚未执行。此探针在v0.6训练完成及终点审计后提出，属于事后探索设计。prepare只读取JSON、源码和文件字节计算哈希，没有反序列化策略权重或调用模型。", "",
        "覆盖全部56个已完成运行、4个训练种子、14个条件；每运行固定检查点0、100、300、600、1200、1800、2100、2400，共448个策略文件、896个接收者检查点。全部条件及失败结果保留，不按终点表现筛选。", "",
        "execute在CPU单线程下通过原remake_agents恢复两个独立小接口；prepared与checkpoint均使用torch.load(weights_only=True, map_location='cpu')，严格匹配state_dict。不会构造ImageBank、调用project/observe/send、加载DINO视觉骨干或照片特征，也不调用优化器。", "",
        "每个接收者固定空inventory和history，完整六地点动作菜单；枚举全部49码和两个私有目标。固定身份菜单先得到49×2×6地点logits，保存其最大值间隔和并列情况。CampAgent.receive中菜单仅执行gather的源码结构与哈希已检查。若每个最大值都唯一，720排列在物理argmax上数学等价；若任何一条并列，完整枚举49×2×720实际接口，保留菜单轴而不暗中折叠。并列按源Torch argmax的首位置规则处理，不加噪声或改温度。", "",
        "主要读数按训练地图和留出地图分别记录：同码双目标可达、两边际分别可达但没有共同码、仅一边际可达、两边际均不可达；固定贪心接收者的完整码上界为max_m[(食物正确+水正确)/2]。full_blocked另保留仅零码允许的通道上界，完整49码上界不称为该封锁通道的可实现效果。", "",
        "如果出现菜单相关并列，先在每个接收者内对其全部菜单和地图计数；跨方向、划分先在种子内等权平均，最后保留四个独立种子值。720菜单、30地图或多个检查点都不是独立训练样本。完整地图控制只有一个split0，不复制成三个划分。", "",
        "检查点2400的两个state_dict必须与各自final.pt逐张量相同；其完整物理接收表再与已有protocol_analysis.json的112个终点表逐消息、逐目标、动作频数和完整表哈希核对。任一不匹配终止并保留失败记录，不换检查点或重试。", "",
        "本轮只测接收映射，既不生成自然消息，也不读取旧curve的随机照片回报作同照片对照。不存在本轮自然回报样本N；覆盖分母为地图×菜单，均匀目标上界分母再乘2。该轨迹显示所存检查点上的变化，不能定位相邻存盘点之间的首次形成时刻、证明单调学习或确证生态原因。贪心不可达不表示随机策略成功概率为0，也不表示物理或信道容量不足。", "",
        "全部源checkpoint、prepared、final锚点、运行配置、训练日程与依赖源码哈希在plan.json中冻结，源码另存code_snapshot。execute启动前重新核对；execution目录采用排他创建，不覆盖已有运行。发生异常只写失败状态，不自动重跑。", "",
        f"最少接收接口批调用为{plan['expected_receiver_checkpoint_count']}次、87808个(code,goal)输入；若每个接收者都有并列，最多再评63221760个完整菜单输入。无训练、无发送推断。", "",
        "命令：", "", "```sh",
        f"python research_program/receiver_coverage_trajectory.py execute '{plan['prepared_dir']}'",
        "```", "",
        "本计划与Qwen历史续跑分别记录，不把二者当作同组实验。", ""])


def prepare(output_dir=DEFAULT_OUTPUT, source_protocol=None):
    """Freeze all sources without importing Torch/camp or loading any weights."""
    require("torch" not in sys.modules and "camp" not in sys.modules,
            "Prepare must run in a clean process without imported model/runtime modules")
    output = Path(output_dir).resolve()
    require(not output.exists(), "Preparation directory already exists; never overwrite")
    source = Path(source_protocol).resolve() if source_protocol is not None else SOURCE
    data = read_json(source)
    require(data["status"] == "complete" and data["expected_run_count"] == 56
            and not data["missing_runs"] and data["expected_seeds"] == list(SEEDS), "Source batch is incomplete")
    indexed = {(r["seed"], r["condition"]): r for r in data["runs"]}
    require(len(indexed) == len(data["runs"]) == 56
            and set(indexed) == set(product(SEEDS, CONDITIONS)), "Unexpected source seed/condition grid")
    hashes = {}

    def freeze(path):
        path = Path(path).resolve()
        require(path.is_file(), f"Missing source file: {path}")
        if str(path) not in hashes:
            hashes[str(path)] = sha(path)
        return str(path)

    freeze(source)
    for relative in CODE_FILES:
        freeze(WORK / relative)
    require(hashes[str((WORK / "redesign_v0.6/camp.py").resolve())] == data["fingerprints"]["camp"]
            and hashes[str((WORK / "redesign_v0.6/analyze_protocols.py").resolve())] == data["fingerprints"]["analysis"],
            "Current environment/protocol source differs from the final anchor")
    gather_check = menu_gather_source_check(WORK / "redesign_v0.6/camp.py")
    runs = []
    for seed, condition in product(SEEDS, CONDITIONS):
        saved = indexed[seed, condition]
        directory = Path(saved["final_checkpoint"]).resolve().parent
        config_path = freeze(directory / "config.json")
        config = read_json(config_path)
        require(config["seed"] == seed and config["condition"] == condition
                and config["plan"] == saved["plan"] and config["checkpoints"] == list(CHECKPOINTS)
                and config["updates"] == 2400 and config["sites"] == 6 and config["history_dim"] == 18
                and config["map_table"] == [list(m) for m in permutations(range(6), 2)]
                and config["train_map_ids"] == saved["train_map_ids"]
                and config["heldout_map_ids"] == saved["heldout_map_ids"], "Source config/anchor mismatch")
        for filename, expected in config["source_hashes"].items():
            filename = freeze(WORK / "redesign_v0.6" / filename)
            require(hashes[filename] == expected, "Training source no longer matches its saved configuration")
        prepared = freeze(saved["prepared_checkpoint"])
        final = freeze(saved["final_checkpoint"])
        require(hashes[prepared] == saved["prepared_sha256"] == config["prepared_source"]["sha256"]
                and hashes[final] == saved["final_sha256"], "Final/prepared anchor hash mismatch")
        checkpoints = {str(update): freeze(directory / f"checkpoint_{update:04d}.pt") for update in CHECKPOINTS}
        schedule_path = freeze(directory / "training_schedule.json")
        schedule = read_json(schedule_path)
        require(len(schedule["levels"]) == len(schedule["batch_identities"]) == 2400,
                "Training schedule has unexpected length")
        freeze(directory / "result.json")
        runs.append({"seed": seed, "condition": condition, "plan": deepcopy(saved["plan"]),
            "train_map_ids": saved["train_map_ids"], "heldout_map_ids": saved["heldout_map_ids"],
            "config_file": config_path, "training_schedule_file": schedule_path,
            "prepared_file": prepared, "final_file": final, "checkpoint_files": checkpoints})
    plan = {"schema_version": 1, "status_at_freeze": "prepared_not_executed",
        "design": "post_hoc_receiver_coverage_trajectory", "created": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "prepared_dir": str(output), "source_protocol": str(source), "checkpoints": list(CHECKPOINTS),
        "seeds": list(SEEDS), "conditions": list(CONDITIONS), "runs": runs,
        "expected_run_count": 56, "expected_checkpoint_file_count": 448, "expected_receiver_checkpoint_count": 896,
        "context": {"inventory": [0, 0], "history_length": 18, "history_value": 0,
                    "goal_vectors": [[1, 0], [0, 1]], "physical_menu": list(range(6))},
        "receiver_policy": "frozen greedy logits.argmax", "device": "cpu", "torch_num_threads": 1,
        "full_menu_chunk_size": 4096, "menu_rule": gather_check, "source_files_sha256": hashes,
        "code_files": list(CODE_FILES), "prepare_runtime": runtime_versions(),
        "prepare_weights_loaded": 0, "prepare_forward_calls": 0,
        "prohibitions": ["no training", "no DINO backbone or cached image features", "no sender/visual inference",
                         "no random-photo curve reward comparison", "no outcome-based source selection", "no automatic retries"]}
    output.mkdir(parents=True, exist_ok=False)
    for relative in CODE_FILES:
        target = output / "code_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(WORK / relative, target)
    write_json_new(output / "plan.json", plan)
    (output / "冻结计划.md").write_text(plan_markdown(plan), encoding="utf-8")
    write_json_new(output / "freeze.json", {"plan_sha256": sha(output / "plan.json"),
        "plan_markdown_sha256": sha(output / "冻结计划.md"), "status": "prepared_not_executed"})
    return plan


def coverage_metrics(actions, train_ids, heldout_ids, *, blocked=False):
    """NumPy-only 49 x 2 x menu physical-action lookup -> coverage metrics.

    A tie-dependent lookup retains all 720 menus; each map is scored separately
    under each menu before aggregation. No menus are treated as training seeds.
    """
    import numpy as np

    actions = np.asarray(actions)
    require(actions.shape[:2] == (49, 2) and actions.ndim == 3 and actions.shape[2] in (1, 720)
            and np.issubdtype(actions.dtype, np.integer) and ((actions >= 0) & (actions < 6)).all(),
            "Invalid physical action table")
    nmenus = actions.shape[2]
    require(not set(train_ids) & set(heldout_ids) and set(train_ids) | set(heldout_ids) == set(range(30)),
            "Invalid map partition")
    maps = []
    for map_id, target in enumerate(permutations(range(6), 2)):
        correct = actions == np.asarray(target)[None, :, None]
        food, water = correct[:, 0].any(0), correct[:, 1].any(0)
        joint = correct.all(1).any(0)
        classes = {"same_code_both_correct": joint,
            "both_marginals_but_no_joint_code": food & water & ~joint,
            "exactly_one_marginal_reachable": food ^ water,
            "neither_marginal_reachable": ~food & ~water}
        best = correct.sum(1).max(0)
        require(np.array_equal(best, np.where(joint, 2, np.where(food | water, 1, 0))),
                "Uniform-goal ceiling and coverage classes disagree")
        maps.append({"map_id": map_id, "food_location": target[0], "water_location": target[1],
            "subset": "heldout" if map_id in heldout_ids else "train", "menus": nmenus,
            "food_marginal_reachable": proportion(food.sum(), nmenus),
            "water_marginal_reachable": proportion(water.sum(), nmenus),
            "coverage_classes": {name: proportion(value.sum(), nmenus) for name, value in classes.items()},
            "full_code_uniform_goal_ceiling": proportion(best.sum(), 2 * nmenus),
            "channel_admissible_uniform_goal_ceiling": proportion(correct[0].sum() if blocked else best.sum(), 2 * nmenus),
            "identity_menu_category": next(name for name, value in classes.items() if value[0]),
            "identity_menu_uniform_goal_ceiling": int(best[0]) / 2})
    summaries = {}
    for name, ids in (("all", range(30)), ("train", train_ids), ("heldout", heldout_ids)):
        selected = [m for m in maps if m["map_id"] in ids]
        denominator = len(selected) * nmenus
        summary = {"map_count": len(selected), "menus_per_map": nmenus,
            "coverage_classes": {c: proportion(sum(m["coverage_classes"][c]["numerator"] for m in selected), denominator)
                                 for c in CATEGORIES}}
        for field in ("food_marginal_reachable", "water_marginal_reachable", "full_code_uniform_goal_ceiling",
                      "channel_admissible_uniform_goal_ceiling"):
            summary[field] = proportion(sum(m[field]["numerator"] for m in selected),
                                       sum(m[field]["denominator"] for m in selected))
        summaries[name] = summary
    return {"maps": maps, "subsets": summaries}


def receiver_lookup(agent, torch, *, history_size=18, chunk_size=4096):
    """Evaluate only receive; exact ties trigger actual all-menu enumeration."""
    import numpy as np

    messages = np.asarray(list(product(range(agent.vocab), repeat=agent.length)), dtype=np.int64)
    require(len(messages) == 49, "Receiver does not expose exactly 49 codes")
    codes = np.repeat(np.arange(49), 2)
    goals = np.tile(np.arange(2), 49)
    identity = np.tile(np.arange(6, dtype=np.int64), (98, 1))

    def receive(m, g, menus):
        logits, _ = agent.receive(torch.from_numpy(messages[m]),
            torch.from_numpy(np.eye(2, dtype=np.float32)[g]), torch.zeros(len(m), 2),
            torch.zeros(len(m), history_size), torch.from_numpy(menus))
        result = logits.detach().cpu().numpy()
        require(result.shape == (len(m), 6) and np.isfinite(result).all(), "Nonfinite or malformed receiver logits")
        return result

    with torch.inference_mode():
        logits = receive(codes, goals, identity).reshape(49, 2, 6)
        maxima = logits.max(-1, keepdims=True)
        tied = (logits == maxima).sum(-1)
        sorted_values = np.sort(logits, axis=-1)
        margins = sorted_values[:, :, -1] - sorted_values[:, :, -2]
        has_tie = bool((tied > 1).any())
        if has_tie:
            menus = np.asarray(list(permutations(range(6))), dtype=np.int64)
            indices = np.asarray(list(product(range(49), range(2), range(720))), dtype=np.int64)
            physical = np.empty(len(indices), dtype=np.int64)
            for start in range(0, len(indices), chunk_size):
                m, g, order = indices[start:start+chunk_size].T
                scores = receive(m, g, menus[order])
                physical[start:start+len(m)] = menus[order, scores.argmax(-1)]
            full = physical.reshape(49, 2, 720)
            require(np.array_equal(full[:, :, 0], logits.argmax(-1)),
                    "Identity-menu argmax changed across enumeration batches")
            scored = full  # Retain the menu axis even if observed outputs happen to match.
            forward_calls = 1 + (len(indices) + chunk_size - 1) // chunk_size
        else:
            scored = logits.argmax(-1).astype(np.int64)[:, :, None]
            full = np.repeat(scored, 720, axis=2)
            forward_calls = 1
    counts = [[np.bincount(full[c, g], minlength=6).tolist() for g in range(2)] for c in range(49)]
    table = [{"message": messages[c].tolist(), "actions_by_goal": full[c, :, 0].tolist(),
              "menu_action_counts_by_goal": counts[c]} for c in range(49)]
    invariant = bool(np.all(full == full[:, :, :1]))
    if not invariant:
        for c, row in enumerate(table):
            row["physical_actions_by_goal_and_menu"] = full[c].tolist()
    return scored, {"receiver_decoder_table": table, "identity_menu_logits": logits.tolist(),
        "maximum_multiplicity": tied.tolist(), "top_two_margin": margins.tolist(),
        "minimum_top_two_margin": float(margins.min()),
        "tie_code_goal_count": int((tied > 1).sum()), "complete_menu_enumeration_triggered": has_tie,
        "menus_evaluated_per_code_goal": 720 if has_tie else 1,
        "menus_retained_for_metrics": scored.shape[2], "physical_menu_invariant": invariant,
        "unique_argmax_equivalence_proof_used": not has_tie,
        "physical_action_sha256": hashlib.sha256(full.tobytes()).hexdigest(),
        "receiver_forward_calls": forward_calls,
        "evaluated_code_goal_menu_rows": 98 + (49 * 2 * 720 if has_tie else 0)}


def endpoint_check(lookup, source_direction):
    require(lookup["receiver_decoder_table"] == source_direction["receiver_decoder_table"],
            "Checkpoint 2400 decoder table differs from the saved final table")
    audit = source_direction["menu_audit"]
    require(audit["enumerated_messages"] == 49 and audit["goals"] == 2 and audit["menus"] == 720
            and audit["cases"] == 70560 and lookup["physical_action_sha256"] == audit["physical_action_sha256"]
            and lookup["physical_menu_invariant"] == audit["all_menu_permutations_physically_equivalent"],
            "Checkpoint 2400 full-menu anchor audit differs")
    return {"status": "exact_match", "code_rows": 49, "goal_rows": 98,
            "all_menu_physical_cases": 70560, "source_physical_action_sha256": audit["physical_action_sha256"]}


def trajectory_summaries(records):
    """Equal weight over directions/splits within each of the four seeds."""
    import numpy as np

    grouped = defaultdict(list)
    for row in records:
        condition = row["condition"]
        family = condition.split("_", 1)[1] if condition.startswith("split") else condition
        for subset in ("train", "heldout"):
            summary = row["coverage"]["subsets"][subset]
            if summary["map_count"]:
                grouped[family, row["seed"], row["update"], subset].append(summary)
    seed_rows = []
    for (family, seed, update, subset), rows in sorted(grouped.items()):
        means = {category: float(np.mean([r["coverage_classes"][category]["rate"] for r in rows])) for category in CATEGORIES}
        for field in ("food_marginal_reachable", "water_marginal_reachable", "full_code_uniform_goal_ceiling",
                      "channel_admissible_uniform_goal_ceiling"):
            means[field] = float(np.mean([r[field]["rate"] for r in rows]))
        seed_rows.append({"family": family, "seed": seed, "update": update, "subset": subset,
                          "receiver_direction_count": len(rows), "equal_weight_means": means})
    families = []
    for family, update, subset in sorted({(r["family"], r["update"], r["subset"]) for r in seed_rows}):
        rows = [r for r in seed_rows if (r["family"], r["update"], r["subset"]) == (family, update, subset)]
        require(len(rows) == 4 and {r["seed"] for r in rows} == set(SEEDS), "Incomplete seed trajectory summary")
        families.append({"family": family, "update": update, "subset": subset,
            "training_seeds": [r["seed"] for r in rows],
            "seed_values": {key: [r["equal_weight_means"][key] for r in rows] for key in rows[0]["equal_weight_means"]},
            "means": {key: float(np.mean([r["equal_weight_means"][key] for r in rows])) for key in rows[0]["equal_weight_means"]}})
    return {"by_seed": seed_rows, "by_family": families}


def verify_plan(directory):
    freeze = read_json(directory / "freeze.json")
    require(sha(directory / "plan.json") == freeze["plan_sha256"]
            and sha(directory / "冻结计划.md") == freeze["plan_markdown_sha256"], "Frozen plan changed")
    plan = read_json(directory / "plan.json")
    require(plan["status_at_freeze"] == "prepared_not_executed"
            and plan["checkpoints"] == list(CHECKPOINTS), "Unexpected prepared plan")
    for filename, expected in plan["source_files_sha256"].items():
        require(sha(filename) == expected, f"Frozen source changed: {filename}")
    for relative in plan["code_files"]:
        require(sha(directory / "code_snapshot" / relative)
                == plan["source_files_sha256"][str((WORK / relative).resolve())], "Frozen source snapshot changed")
    require(menu_gather_source_check(WORK / "redesign_v0.6/camp.py") == plan["menu_rule"], "Reviewed menu semantics changed")
    return plan


def execute(prepared_dir):
    """Run the separately authorized frozen probe once; never overwrite/retry."""
    directory = Path(prepared_dir).resolve()
    require(not (directory / "execution").exists(), "Execution directory already exists; no overwrites or automatic retries")
    plan = verify_plan(directory)  # Check every source before importing Torch/loading tensors.
    execution = directory / "execution"
    execution.mkdir(exist_ok=False)
    write_json_new(execution / "started.json", {"status": "running", "plan_sha256": sha(directory / "plan.json"),
        "started": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "runtime": runtime_versions()})
    records = []
    try:
        require(all(name not in sys.modules for name in ("camp", "agents", "run_pilot", "resource_env")),
                "Execute requires a clean process to avoid ambiguous legacy module imports")
        import torch
        torch.set_num_threads(plan["torch_num_threads"])
        sys.path.insert(0, str(WORK / "redesign_v0.6"))
        camp = importlib.import_module("camp")
        for module, relative in (("camp", "redesign_v0.6/camp.py"), ("agents", "redesign_v0.4/agents.py"),
                                 ("run_pilot", "redesign_v0.4/run_pilot.py"), ("resource_env", "redesign_v0.4/resource_env.py")):
            require(Path(sys.modules[module].__file__).resolve() == (WORK / relative).resolve(), "Unexpected legacy import source")
        require(camp.SITES == 6 and camp.HISTORY == 18, "Restored interface dimensions differ")
        source = read_json(plan["source_protocol"])
        source_index = {(r["seed"], r["condition"]): r for r in source["runs"]}
        prepared_cache = {}
        with (execution / "records.jsonl").open("x", encoding="utf-8") as log:
            for run in plan["runs"]:
                if run["prepared_file"] not in prepared_cache:
                    prepared_cache[run["prepared_file"]] = torch.load(run["prepared_file"], map_location="cpu", weights_only=True)
                agents = camp.remake_agents(run["seed"], prepared_cache[run["prepared_file"]],
                                            run["plan"]["vocab"], run["plan"]["length"])
                require(len(agents) == 2, "Expected two independent interface agents")
                final = torch.load(run["final_file"], map_location="cpu", weights_only=True)
                for update in CHECKPOINTS:
                    states = torch.load(run["checkpoint_files"][str(update)], map_location="cpu", weights_only=True)
                    require(isinstance(states, list) and len(states) == 2, "Malformed checkpoint state list")
                    if update == 2400:
                        require(isinstance(final, list) and len(final) == 2
                                and all(set(s) == set(f) and all(torch.equal(s[k], f[k]) for k in s)
                                        for s, f in zip(states, final)), "Checkpoint 2400 tensors differ from final.pt")
                    for receiver, (agent, state) in enumerate(zip(agents, states)):
                        agent.load_state_dict(state, strict=True)
                        agent.eval(); agent.requires_grad_(False)
                        physical, lookup = receiver_lookup(agent, torch, history_size=camp.HISTORY,
                                                           chunk_size=plan["full_menu_chunk_size"])
                        row = {"seed": run["seed"], "condition": run["condition"], "update": update,
                            "receiver": receiver, "scout": 1-receiver, "collector": receiver,
                            "checkpoint_file": run["checkpoint_files"][str(update)],
                            "checkpoint_sha256": plan["source_files_sha256"][run["checkpoint_files"][str(update)]],
                            "lookup": lookup, "coverage": coverage_metrics(physical, run["train_map_ids"],
                                run["heldout_map_ids"], blocked=run["plan"]["blocked"]), "endpoint_anchor": None}
                        if update == 2400:
                            anchor = next(d for d in source_index[run["seed"], run["condition"]]["directions"]
                                          if d["collector"] == receiver and d["scout"] == 1-receiver)
                            row["endpoint_anchor"] = endpoint_check(lookup, anchor)
                            row["endpoint_anchor"]["checkpoint_final_state_dict_exact"] = True
                        log.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"); log.flush()
                        records.append(row)
                print(json.dumps({"completed_run": f"s{run['seed']}_{run['condition']}",
                                  "receiver_checkpoints": len(records)}), flush=True)
        require(len(records) == 896 and sum(r["endpoint_anchor"] is not None for r in records) == 112,
                "Evaluation did not cover the complete frozen grid")
        result = {"status": "completed", "analysis": "post_hoc_frozen_greedy_receiver_coverage_trajectory",
            "plan_sha256": sha(directory / "plan.json"), "receiver_checkpoint_count": len(records),
            "endpoint_anchors_exact": 112, "tie_receiver_checkpoint_count": sum(r["lookup"]["complete_menu_enumeration_triggered"] for r in records),
            "receiver_forward_calls": sum(r["lookup"]["receiver_forward_calls"] for r in records),
            "evaluated_code_goal_menu_rows": sum(r["lookup"]["evaluated_code_goal_menu_rows"] for r in records),
            "visual_backbones_loaded": 0, "feature_banks_loaded": 0, "sender_forward_calls": 0, "training_updates": 0,
            "natural_reward_sample_count": 0, "records_file": "records.jsonl", "records_sha256": sha(execution / "records.jsonl"),
            "summaries": trajectory_summaries(records), "runtime": runtime_versions(),
            "limits": ["greedy argmax coverage, not stochastic policy support", "four seeds; repeated checkpoints/maps/directions",
                       "post-hoc endpoint-informed probe", "no same-photo natural-message reward was measured",
                       "saved checkpoints do not identify exact formation time or ecological causation"]}
        write_json_new(execution / "results.json", result)
        write_json_new(execution / "status.json", {"status": "completed", "receiver_checkpoint_count": len(records), "endpoint_anchors_exact": 112})
        return result
    except BaseException as error:
        write_json_new(execution / "failure.json", {"status": "failed", "error": str(error), "traceback": traceback.format_exc(),
            "receiver_checkpoint_records_written": len(records), "automatic_retry": False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    prep.add_argument("--source", type=Path, default=SOURCE)
    run = commands.add_parser("execute")
    run.add_argument("prepared_dir", type=Path)
    args = parser.parse_args()
    result = prepare(args.output, args.source) if args.command == "prepare" else execute(args.prepared_dir)
    print(json.dumps({"command": args.command, "status": result.get("status", result.get("status_at_freeze")),
                      "prepared_dir": result.get("prepared_dir", str(args.prepared_dir) if args.command == "execute" else None),
                      "new_forward_calls": result.get("receiver_forward_calls", 0)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

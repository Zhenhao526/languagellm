"""Run the frozen single-slot aligned/placebo compositional transfer probe."""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import platform
import shutil
import time
from pathlib import Path

import numpy as np

from research_program.triadic_action_dependency_study import dataset, environment as action_environment
from research_program.triadic_factorized_neutral_altpartner_direction_probe import intervention
from research_program.triadic_factorized_neutral_altpartner_semantic_transfer_probe import probe as transfer_kernel
from research_program.triadic_factorized_neutral_altpartner_study import runner as source_runner
from research_program.triadic_message_study import runner as core
from research_program.triadic_reciprocal_execution_study import environment as execution_environment

from . import design

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MATCHED_AUDIT = ROOT / "research_program/triadic_compositional_matched_control_semantic_transfer_probe/audit_matched_control_005"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf8")


def code_files():
    paths = [HERE / name for name in ("__init__.py", "design.py", "probe.py", "audit.py", "aggregate.py", "plot_results.py", "plan.md", "tests/test_design.py")]
    paths += [
        Path(design.matched_design.__file__),
        Path(transfer_kernel.__file__), Path(transfer_kernel.intervention.__file__),
        Path(source_runner.__file__), Path(core.__file__), Path(core.base.__file__),
        Path(dataset.__file__), Path(action_environment.__file__),
    ]
    require(all(path.is_file() for path in paths), "Missing slot-transfer source")
    return {str(path.resolve().relative_to(ROOT)): sha(path) for path in paths}


def frozen_inputs():
    prepared_path = design.MATCHED_RUN / "prepared.json"
    plan_path = design.MATCHED_RUN / "plan.json"
    freeze_path = design.MATCHED_RUN / "freeze.json"
    result_path = design.MATCHED_RUN / "execution/results.json"
    audit_path = MATCHED_AUDIT / "verification.json"
    require(all(path.is_file() for path in (prepared_path, plan_path, freeze_path, result_path, audit_path)), "Missing matched authority file")
    matched = json.loads(prepared_path.read_text(encoding="utf8"))
    require(matched["schema"] == "triadic_compositional_matched_control_semantic_transfer_prepared_v1", "Unexpected matched schema")
    audit = json.loads(audit_path.read_text(encoding="utf8"))
    require(audit["status"] == "passed" and float(audit["max_abs_error"]) == 0.0, "Matched authority audit failed")
    sources = design.make_prepared()["sources"]
    checkpoints = {}
    for key, source in sources.items():
        result = Path(source["result"]); checkpoint = Path(source["checkpoint"])
        require(result.is_file() and checkpoint.is_file(), f"Missing source checkpoint: {key}")
        result_json = json.loads(result.read_text(encoding="utf8"))
        digest = sha(checkpoint)
        require(digest == source["checkpoint_sha256"] == result_json["checkpoint_sha256" if "checkpoint_sha256" in result_json else "final_checkpoint_sha256"], f"Checkpoint binding mismatch: {key}")
        checkpoints[key] = dict(source, result_sha256=sha(result))
    return dict(
        authority=dict(
            run=str(design.MATCHED_RUN), prepared_sha256=sha(prepared_path), plan_sha256=sha(plan_path),
            freeze_sha256=sha(freeze_path), result_sha256=sha(result_path), audit_sha256=sha(audit_path),
        ),
        checkpoints=checkpoints,
    )


def prepared():
    static = design.make_prepared()
    static["matched_prepared_sha256"] = sha(design.MATCHED_RUN / "prepared.json")
    static["source_sha256"] = code_files()
    static["runtime"] = dict(python=platform.python_version(), numpy=np.__version__)
    policies = len(design.SEEDS) * len(design.CONDITIONS)
    cases = int(static["cases"]["case_count"])
    masks = len(design.MASKS)
    worlds = int(static["heldout_spec"]["world_count"])
    live_policies = policies // 2
    natural = policies * worlds * 9
    interventions = live_policies * (cases * masks * 2 + 3 * design.SHAM_ROWS_PER_SENDER * masks) * 6
    static["budget"] = dict(policy_blocks=policies, masks=masks, cases_per_policy=cases, worlds_per_policy=worlds,
                             natural_module_samples=natural, intervention_module_samples=interventions,
                             total_module_samples=natural + interventions, optimizer_updates=0)
    return static


def prepare(out):
    out = Path(out).resolve(); require(not out.exists(), "Never overwrite preparation")
    static = prepared(); inputs = frozen_inputs(); out.mkdir(parents=True)
    for relative in static["source_sha256"]:
        source = ROOT / relative; target = out / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(source, target)
    write(out / "prepared.json", static); write(out / "inputs.json", inputs)
    write(out / "plan.json", dict(status="prepared_without_probe_forward", created_at=core.base.now(),
                                   source_sha256=static["source_sha256"], prepared_sha256=sha(out / "prepared.json"),
                                   inputs_sha256=sha(out / "inputs.json"), policy_blocks=64, masks=5,
                                   case_rows=static["cases"]["case_count"], no_training=True))
    write(out / "freeze.json", dict(plan_sha256=sha(out / "plan.json"), prepared_sha256=sha(out / "prepared.json"),
                                         inputs_sha256=sha(out / "inputs.json")))
    verify(out)
    return dict(status="prepared_without_probe_forward", output=str(out), case_rows=static["cases"]["case_count"], masks=5)


def verify(out):
    out = Path(out).resolve()
    static = json.loads((out / "prepared.json").read_text(encoding="utf8")); plan = json.loads((out / "plan.json").read_text(encoding="utf8")); freeze = json.loads((out / "freeze.json").read_text(encoding="utf8")); inputs = json.loads((out / "inputs.json").read_text(encoding="utf8"))
    require(sha(out / "prepared.json") == freeze["prepared_sha256"] == plan["prepared_sha256"], "Prepared hash mismatch")
    require(sha(out / "plan.json") == freeze["plan_sha256"], "Plan hash mismatch")
    require(sha(out / "inputs.json") == freeze["inputs_sha256"] == plan["inputs_sha256"], "Inputs hash mismatch")
    require(static == prepared(), "Static preparation changed")
    require(inputs == frozen_inputs(), "Frozen authority changed")
    for relative, digest in plan["source_sha256"].items():
        require(sha(out / "source_snapshot" / relative) == digest, "Source snapshot changed: " + relative)
    return plan, static, inputs


def masked_packets(receiver, donor, slots):
    receiver = np.asarray(receiver, dtype=np.int8); donor = np.asarray(donor, dtype=np.int8)
    require(receiver.shape == donor.shape and receiver.ndim == 2 and receiver.shape[1] == 4, "Invalid packet arrays")
    output = receiver.copy()
    output[:, list(slots)] = donor[:, list(slots)]
    return output


def mean(values):
    return float(np.mean(values)) if len(values) else 0.0


def partner_probability(probabilities, sender, recipient):
    return probabilities[:, recipient, execution_environment.PROPOSAL_ROLES[recipient] == sender].sum(axis=1)


def physical_metrics(states, probabilities, legal):
    actions = probabilities.argmax(-1).astype(np.int16)
    settled = execution_environment.settle(states, actions, "strict")
    physical = settled["actual_pair_index"] >= 0
    q = np.zeros(len(states), dtype=bool)
    for index, action_set in enumerate(legal):
        if physical[index]:
            q[index] = tuple(actions[index].tolist()) in action_set
    return dict(physical=float(physical.mean()), q=float(q.mean()),
                conditional_q=float(q[physical].mean()) if physical.any() else 0.0,
                actions=actions)


def run_group(networks, arrays, bank, case_indices, cases, sender, legal_cache, pair_cache, live, mask_name, slots):
    aligned_transfer = []; placebo_transfer = []; aligned_partner = []; placebo_partner = []
    natural_physical = []; aligned_physical = []; placebo_physical = []
    natural_q = []; aligned_q = []; placebo_q = []
    natural_cq = []; aligned_cq = []; placebo_cq = []; aligned_change = []; placebo_change = []
    receiver_ids = np.asarray(cases["receiver_state_indices"], dtype=np.int64)[case_indices]
    donor_ids = np.asarray(cases["donor_state_indices"], dtype=np.int64)[case_indices]
    placebo_case_ids = np.asarray(cases["placebo_donor_case_indices"], dtype=np.int64)[case_indices]
    placebo_ids = np.asarray(cases["donor_state_indices"], dtype=np.int64)[placebo_case_ids]
    for start in range(0, len(receiver_ids), design.CHUNK_SIZE):
        stop = min(start + design.CHUNK_SIZE, len(receiver_ids)); ri = receiver_ids[start:stop]; di = donor_ids[start:stop]; pi = placebo_ids[start:stop]
        natural = bank["action_probabilities"][ri]; states = bank["states"][ri]
        receiver_packet = bank["messages"][ri, 0, sender]
        aligned_packet = masked_packets(receiver_packet, bank["messages"][di, 0, sender], slots)
        placebo_packet = masked_packets(receiver_packet, bank["messages"][pi, 0, sender], slots)
        if live:
            aligned = intervention.intervene(networks, arrays["x_PL"][ri], bank["messages"][ri], np.full(len(ri), sender), aligned_packet)["action_probabilities"]
            placebo = intervention.intervene(networks, arrays["x_PL"][ri], bank["messages"][ri], np.full(len(ri), sender), placebo_packet)["action_probabilities"]
        else:
            aligned = placebo = natural
        for row in range(len(ri)):
            target_legal = legal_cache[int(ri[row])]; source_legal = legal_cache[int(di[row])]
            donor_only = source_legal - target_legal; target_only = target_legal - source_legal
            baseline = transfer_kernel.mass(natural[row:row + 1], donor_only)[0] - transfer_kernel.mass(natural[row:row + 1], target_only)[0]
            aligned_transfer.append(transfer_kernel.mass(aligned[row:row + 1], donor_only)[0] - transfer_kernel.mass(aligned[row:row + 1], target_only)[0] - baseline)
            placebo_transfer.append(transfer_kernel.mass(placebo[row:row + 1], donor_only)[0] - transfer_kernel.mass(placebo[row:row + 1], target_only)[0] - baseline)
            target_pairs = pair_cache[int(ri[row])]; source_pairs = pair_cache[int(di[row])]; effects_a = []; effects_p = []
            for recipient in range(3):
                if recipient == sender:
                    continue
                desired = int(tuple(sorted((sender, recipient))) in source_pairs) - int(tuple(sorted((sender, recipient))) in target_pairs)
                if desired:
                    effects_a.append(desired * (partner_probability(aligned[row:row + 1], sender, recipient)[0] - partner_probability(natural[row:row + 1], sender, recipient)[0]))
                    effects_p.append(desired * (partner_probability(placebo[row:row + 1], sender, recipient)[0] - partner_probability(natural[row:row + 1], sender, recipient)[0]))
            aligned_partner.append(mean(effects_a)); placebo_partner.append(mean(effects_p))
        nm = physical_metrics(states, natural, [legal_cache[int(i)] for i in ri]); am = physical_metrics(states, aligned, [legal_cache[int(i)] for i in ri]); pm = physical_metrics(states, placebo, [legal_cache[int(i)] for i in ri])
        natural_physical.append(nm["physical"]); aligned_physical.append(am["physical"]); placebo_physical.append(pm["physical"])
        natural_q.append(nm["q"]); aligned_q.append(am["q"]); placebo_q.append(pm["q"])
        natural_cq.append(nm["conditional_q"]); aligned_cq.append(am["conditional_q"]); placebo_cq.append(pm["conditional_q"])
        aligned_change.append(float(np.any(am["actions"] != nm["actions"], axis=1).mean())); placebo_change.append(float(np.any(pm["actions"] != nm["actions"], axis=1).mean()))
    return dict(mask=mask_name, slots=list(slots), rows=len(receiver_ids),
                aligned_plan_transfer=mean(aligned_transfer), placebo_plan_transfer=mean(placebo_transfer),
                aligned_minus_placebo_plan_transfer=mean(np.asarray(aligned_transfer) - np.asarray(placebo_transfer)),
                aligned_partner_transfer=mean(aligned_partner), placebo_partner_transfer=mean(placebo_partner),
                aligned_minus_placebo_partner_transfer=mean(np.asarray(aligned_partner) - np.asarray(placebo_partner)),
                natural_physical=mean(natural_physical), aligned_physical=mean(aligned_physical), placebo_physical=mean(placebo_physical),
                natural_q=mean(natural_q), aligned_q=mean(aligned_q), placebo_q=mean(placebo_q),
                natural_conditional_q=mean(natural_cq), aligned_conditional_q=mean(aligned_cq), placebo_conditional_q=mean(placebo_cq),
                aligned_action_change=mean(aligned_change), placebo_action_change=mean(placebo_change))


def legal_sets(states):
    output = []
    for row in states:
        plans = action_environment.full_success_plans(tuple(map(int, row[:3])), tuple(map(int, row[3:7])))
        require(len(plans) == 2, "State is not a two-plan world")
        output.append({tuple(dataset.plan_action_indices(plan)) for plan in plans})
    return output


def pair_sets(states):
    return [{tuple(plan[:2]) for plan in action_environment.full_success_plans(tuple(map(int, row[:3])), tuple(map(int, row[3:7])))} for row in states]


def run_policy(networks, arrays, cases, live):
    bank = transfer_kernel.native_bank(networks, arrays, live)
    legal_cache = legal_sets(bank["states"]); pair_cache = pair_sets(bank["states"])
    sender = np.asarray(cases["sender"], dtype=np.int8); axis = np.asarray(cases["axis"], dtype=object); groups = {}
    for who in range(3):
        for axis_name in design.AXES:
            ids = np.flatnonzero((sender == who) & (axis == axis_name))
            for mask_name, slots in design.MASKS:
                groups[f"{axis_name}/{who}/{mask_name}"] = run_group(networks, arrays, bank, ids, cases, who, legal_cache, pair_cache, live, mask_name, slots)
    sham = []
    for who in range(3):
        ids = np.flatnonzero(sender == who)[:design.SHAM_ROWS_PER_SENDER]
        for mask_name, slots in design.MASKS:
            if live:
                ri = np.asarray(cases["receiver_state_indices"], dtype=np.int64)[ids]
                receiver = bank["messages"][ri, 0, who]
                same_packet = masked_packets(receiver, receiver, slots)
                same = intervention.intervene(networks, arrays["x_PL"][ri], bank["messages"][ri], np.full(len(ri), who), same_packet)
                natural = bank["action_probabilities"][ri]
                sham.append(dict(sender=who, mask=mask_name, rows=len(ids), action_equal=bool(np.array_equal(same["action_indices"], natural.argmax(-1))), message_equal=bool(np.array_equal(same["generated_messages"][:, 0], bank["messages"][ri, 0])), max_probability_error=float(np.max(np.abs(same["action_probabilities"] - natural)))))
            else:
                sham.append(dict(sender=who, mask=mask_name, rows=len(ids), action_equal=True, message_equal=True, max_probability_error=0.0))
    case_count = len(cases["receiver_state_indices"]); world_count = len(bank["states"])
    intervention_samples = (case_count * len(design.MASKS) * 2 + 3 * design.SHAM_ROWS_PER_SENDER * len(design.MASKS)) * 12 if live else 0
    return dict(worlds=world_count, natural_module_samples=9 * world_count, intervention_module_samples=intervention_samples, groups=groups, sham=sham)


def worker(payload):
    seed, static, inputs, execution = payload; execution = Path(execution); out = execution / f"seed_{seed}"; out.mkdir(parents=True, exist_ok=False)
    arrays = transfer_kernel.make_arrays(static["heldout_spec"]); rows = []
    for arm in design.ARMS:
        for schedule in design.SCHEDULES:
            for live in design.LIVES:
                condition = design.matched_design.condition(arm, schedule, live); key = f"{seed}:{arm}:{condition}"; meta = inputs["checkpoints"][key]
                checkpoint = Path(meta["checkpoint"]); require(sha(checkpoint) == meta["checkpoint_sha256"], "Checkpoint binding mismatch")
                networks = source_runner.load_networks(checkpoint); started = time.perf_counter(); summary = run_policy(networks, arrays, static["cases"], bool(live))
                row = dict(seed=seed, arm=arm, condition=condition, schedule=schedule, live=bool(live), checkpoint_sha256=meta["checkpoint_sha256"], source_result_sha256=meta["result_sha256"], parameter_sha256=source_runner.parameter_hash(networks), elapsed_seconds=time.perf_counter() - started, **summary)
                write(out / f"{arm}_{condition}.json", row); rows.append(row)
    return rows


def execute(out, workers=4):
    out = Path(out).resolve(); verify(out); static = json.loads((out / "prepared.json").read_text(encoding="utf8")); inputs = json.loads((out / "inputs.json").read_text(encoding="utf8")); execution = out / "execution"; require(not execution.exists(), "Never overwrite execution"); execution.mkdir(); started = time.perf_counter(); write(execution / "started.json", dict(started_at=core.base.now(), plan_sha256=sha(out / "plan.json")))
    try:
        with multiprocessing.get_context("spawn").Pool(workers) as pool:
            groups = pool.map(worker, [(seed, static, inputs, str(execution)) for seed in design.SEEDS])
        rows = [row for group in groups for row in group]; require(len(rows) == 64, "Incomplete policy grid")
        result = dict(status="completed_compositional_slot_transfer_probe", created_at=core.base.now(), elapsed_seconds=time.perf_counter() - started, plan_sha256=sha(out / "plan.json"), rows=rows, policy_blocks=64, masks=len(design.MASKS), case_count=static["cases"]["case_count"], worlds=static["heldout_spec"]["world_count"], model_forward_samples=sum(row["natural_module_samples"] + row["intervention_module_samples"] for row in rows), optimizer_updates=0, no_training_updates=True, posthoc=True)
        write(execution / "results.json", result); result_sha = sha(execution / "results.json"); write(execution / "status.json", dict(status="completed", results_sha256=result_sha)); write(execution / "receipt.json", dict(status="passed", results_sha256=result_sha, policy_blocks=64, masks=len(design.MASKS), case_count=static["cases"]["case_count"], optimizer_updates=0)); return result
    except BaseException as error:
        write(execution / "failure.json", dict(status="failed", error=repr(error), elapsed_seconds=time.perf_counter() - started)); raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("command", choices=("prepare", "verify", "execute")); parser.add_argument("--out", required=True); parser.add_argument("--workers", type=int, default=4); args = parser.parse_args()
    answer = prepare(args.out) if args.command == "prepare" else execute(args.out, args.workers) if args.command == "execute" else verify(args.out)[0]
    print(json.dumps(answer, ensure_ascii=False))

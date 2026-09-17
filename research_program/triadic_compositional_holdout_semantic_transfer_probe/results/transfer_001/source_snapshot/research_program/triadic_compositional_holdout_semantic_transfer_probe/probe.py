"""Run the heldout joint-combination aligned/placebo probe."""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import platform
import shutil
import time
from pathlib import Path

from research_program.triadic_factorized_neutral_altpartner_semantic_transfer_probe import probe as transfer_kernel
from research_program.triadic_factorized_neutral_altpartner_study import runner as source_runner
from research_program.triadic_message_study import runner as core

from . import design

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf8")


def code_files():
    paths = [HERE / name for name in ("__init__.py", "design.py", "probe.py", "audit.py", "aggregate.py", "plan.md", "tests/test_design.py")]
    paths += [Path(transfer_kernel.__file__), Path(transfer_kernel.intervention.__file__), Path(source_runner.__file__), Path(core.__file__), Path(core.base.__file__)]
    require(all(path.is_file() for path in paths), "Missing probe source")
    return {str(path.resolve().relative_to(ROOT)): sha(path) for path in paths}


def frozen_inputs():
    source = design.SOURCE_RUN; source_prepared = json.loads((source / "prepared.json").read_text(encoding="utf8")); checkpoints = {}
    for seed in design.SEEDS:
        for condition in design.CONDITIONS:
            directory = source / "execution" / f"seed_{seed}_{condition}"; result_path = directory / "result.json"; checkpoint = directory / "checkpoint_6000.npz"
            require(result_path.is_file() and checkpoint.is_file(), f"Missing source policy {seed}:{condition}")
            result = json.loads(result_path.read_text(encoding="utf8")); digest = sha(checkpoint); require(result["final_checkpoint_sha256"] == digest, "Checkpoint hash mismatch")
            checkpoints[f"{seed}:{condition}"] = dict(path=str(checkpoint.resolve()), sha256=digest, result_sha256=sha(result_path), seed=seed, condition=condition)
    return dict(source_run=str(source), source_prepared_sha256=sha(source / "prepared.json"), source_plan_sha256=sha(source / "plan.json"), source_freeze_sha256=sha(source / "freeze.json"), checkpoints=checkpoints, source_schema=source_prepared["schema"])


def prepared():
    static = design.make_prepared(); static["source_sha256"] = code_files(); static["runtime"] = dict(python=platform.python_version(), numpy=transfer_kernel.np.__version__)
    static["budget"] = dict(policy_blocks=32, case_rows=static["cases"]["case_count"] * 32, worlds_per_policy=static["heldout_spec"]["world_count"], optimizer_updates=0)
    return static


def prepare(out):
    out=Path(out).resolve(); require(not out.exists(),"Never overwrite preparation"); static=prepared(); inputs=frozen_inputs(); out.mkdir(parents=True)
    for relative in static["source_sha256"]:
        source=ROOT/relative; target=out/"source_snapshot"/relative; target.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(source,target)
    write(out/"prepared.json",static); write(out/"inputs.json",inputs); write(out/"plan.json",dict(status="prepared_without_probe_forward",created_at=core.base.now(),source_sha256=static["source_sha256"],prepared_sha256=sha(out/"prepared.json"),inputs_sha256=sha(out/"inputs.json"),policy_blocks=32,case_rows=static["cases"]["case_count"],no_training=True)); write(out/"freeze.json",dict(plan_sha256=sha(out/"plan.json"),prepared_sha256=sha(out/"prepared.json"),inputs_sha256=sha(out/"inputs.json"))); verify(out); return dict(status="prepared_without_probe_forward",output=str(out),case_rows=static["cases"]["case_count"])


def verify(out):
    out=Path(out).resolve(); static=json.loads((out/"prepared.json").read_text()); plan=json.loads((out/"plan.json").read_text()); freeze=json.loads((out/"freeze.json").read_text()); inputs=json.loads((out/"inputs.json").read_text())
    require(sha(out/"prepared.json")==freeze["prepared_sha256"]==plan["prepared_sha256"],"Prepared hash mismatch"); require(sha(out/"plan.json")==freeze["plan_sha256"],"Plan hash mismatch"); require(sha(out/"inputs.json")==freeze["inputs_sha256"]==plan["inputs_sha256"],"Inputs hash mismatch"); require(static==prepared(),"Static preparation changed"); require(inputs==frozen_inputs(),"Frozen inputs changed")
    for relative,digest in plan["source_sha256"].items(): require(sha(out/"source_snapshot"/relative)==digest,"Source snapshot changed: "+relative)
    return plan,static,inputs


def worker(payload):
    seed,static,inputs,execution=payload; execution=Path(execution); out=execution/f"seed_{seed}"; out.mkdir(parents=True,exist_ok=False); arrays=transfer_kernel.make_arrays(static["heldout_spec"]); cases=static["cases"]; rows=[]
    for condition in design.CONDITIONS:
        meta=inputs["checkpoints"][f"{seed}:{condition}"]; checkpoint=Path(meta["path"]); require(sha(checkpoint)==meta["sha256"],"Checkpoint binding mismatch"); networks=source_runner.load_networks(checkpoint); started=time.perf_counter(); summary=transfer_kernel.run_policy(networks,arrays,cases,condition.endswith("_live")); row=dict(seed=seed,condition=condition,schedule=condition.split("_",1)[0],live=condition.endswith("_live"),checkpoint_sha256=meta["sha256"],source_result_sha256=meta["result_sha256"],parameter_sha256=source_runner.parameter_hash(networks),elapsed_seconds=time.perf_counter()-started,**summary); write(out/f"{condition}.json",row); rows.append(row)
    return rows


def execute(out,workers=4):
    out=Path(out).resolve(); verify(out); static=json.loads((out/"prepared.json").read_text()); inputs=json.loads((out/"inputs.json").read_text()); execution=out/"execution"; require(not execution.exists(),"Never overwrite execution"); execution.mkdir(); started=time.perf_counter(); write(execution/"started.json",dict(started_at=core.base.now(),plan_sha256=sha(out/"plan.json")))
    try:
        with multiprocessing.get_context("spawn").Pool(workers) as pool: groups=pool.map(worker,[(seed,static,inputs,str(execution)) for seed in design.SEEDS])
        rows=[row for group in groups for row in group]; require(len(rows)==32,"Incomplete policy grid"); result=dict(status="completed_compositional_holdout_semantic_transfer_probe",created_at=core.base.now(),elapsed_seconds=time.perf_counter()-started,plan_sha256=sha(out/"plan.json"),rows=rows,policy_blocks=32,case_count=static["cases"]["case_count"],worlds=static["heldout_spec"]["world_count"],model_forward_samples=sum(row["natural_module_samples"]+row["intervention_module_samples"] for row in rows),optimizer_updates=0,no_training_updates=True,posthoc=True); write(execution/"results.json",result); result_sha=sha(execution/"results.json"); write(execution/"status.json",dict(status="completed",results_sha256=result_sha)); write(execution/"receipt.json",dict(status="passed",results_sha256=result_sha,policy_blocks=32,case_count=static["cases"]["case_count"],optimizer_updates=0)); return result
    except BaseException as error:
        write(execution/"failure.json",dict(status="failed",error=repr(error),elapsed_seconds=time.perf_counter()-started)); raise


if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("command",choices=("prepare","verify","execute")); parser.add_argument("--out",required=True); parser.add_argument("--workers",type=int,default=4); args=parser.parse_args(); answer=prepare(args.out) if args.command=="prepare" else execute(args.out,args.workers) if args.command=="execute" else verify(args.out)[0]; print(json.dumps(answer,ensure_ascii=False))

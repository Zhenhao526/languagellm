"""Final-only paired objective report; no policies or weights are loaded."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import tempfile

from research_program.triadic_learning_baseline import summarize_results as helpers

SEEDS = (45101, 45102, 45103, 45104)
OBJECTIVES = ("mean_J", "mean_log_J")
PARTITIONS = helpers.PARTITIONS
NAMES = helpers.NAMES
METRICS = helpers.METRICS
COUNTS = dict(train=82836, new_needs=24732, new_layouts=27612, new_needs_and_layouts=8244)
ROOT = Path(__file__).resolve().parents[2]
sha, read, require, near = helpers.sha, helpers.read, helpers.require, helpers.near


def summarize_runs(runs):
    require([(r["seed"],r["objective"]) for r in runs] == [(s,o) for s in SEEDS for o in OBJECTIVES],
            "Eight fixed runs required, in their fixed order")
    rows, behavior, candidate_cells = [], [], []
    for run in runs:
        require(run["updates"] == 6000 and run["training_world_samples"] == 1536000, "Wrong endpoint or budget")
        require(set(run["final"]) == set(PARTITIONS), "Incomplete final matrix")
        require([m["update"] for m in run["monitor"]] == [0,100,500,1500,3000,6000], "Fixed checkpoints missing")
        domain = run["full_domain_actions"]
        require(domain["worlds"] == 143424 and sum(c["worlds"] for c in domain["joint_argmax_action_counts"]) == 143424,
                "Incomplete domain action inventory")
        require(domain["distinct_joint_argmax_actions"] == len(domain["joint_argmax_action_counts"])
                == len({tuple(c["action_indices"]) for c in domain["joint_argmax_action_counts"]}), "Action inventory duplicates")
        pair_counts = dict(AB=0, AC=0, BC=0)
        executed = 0; full = 0; reward_sum = 0.; exceeds = []
        run_candidates = []
        for name in PARTITIONS:
            c = run["final"][name]
            helpers.check_cell(c, COUNTS[name])
            require(sum(p["worlds"] for p in c["joint_argmax_action_counts"]) == COUNTS[name]
                    and c["distinct_joint_argmax_actions"] == len(c["joint_argmax_action_counts"]), "Cell action inventory mismatch")
            num, den = (613,767) if name in ("train","new_layouts") else (191,229)
            above = c["greedy_full_successes"]*den > COUNTS[name]*num
            used = [p for p,n in c["greedy_executed_pair_worlds"].items() if n]
            require(not above or len(used) >= 2, "Above single-pair bound but only one pair executed")
            rows.append({"seed":run["seed"], "objective":run["objective"], "partition":name, "label":NAMES[name],
                **deepcopy(c), "fixed_single_pair_full_success_bound":{"numerator":num,"denominator":den,"rate":num/den},
                "above_fixed_single_pair_full_success_bound":above})
            for pair,n in c["greedy_executed_pair_worlds"].items(): pair_counts[pair] += n
            executed += c["greedy_executed_transports"]
            full += c["greedy_full_successes"]; reward_sum += c["greedy_reward_sum"]
            if above: exceeds.append(name)
            if name != "train":
                passed = c["greedy_full_success_rate"] >= .99
                run_candidates.append(passed)
                candidate_cells.append({"seed":run["seed"],"objective":run["objective"],"partition":name,"passes":passed})
        require(run["candidate_threshold_met"] is all(run_candidates), "Per-run candidate flag differs")
        used = [p for p,n in pair_counts.items() if n]
        constant = domain["distinct_joint_argmax_actions"] == 1
        description = ("constant_joint_action" if constant else "multiple_joint_actions_single_executed_pair" if len(used)==1
                       else "multiple_joint_actions_and_executed_pairs" if len(used)>1 else "multiple_joint_actions_no_execution")
        behavior.append({"seed":run["seed"],"objective":run["objective"], "worlds":143424,
            "distinct_joint_argmax_actions":domain["distinct_joint_argmax_actions"],
            "joint_argmax_action_counts":deepcopy(domain["joint_argmax_action_counts"]),
            "executed_pair_world_counts":pair_counts,"executed_transport_actions":executed,
            "all_worlds_physically_matched":executed==2*143424,
            "full_successes":full,"full_success_rate":full/143424,"native_reward_mean":reward_sum/143424,
            "partitions_above_fixed_single_pair_bound":exceeds,"description":description,
            "scope":"Action variation alone is not evidence of correct use of input or capability improvement."})
    index = {(r["seed"],r["objective"],r["partition"]):r for r in rows}
    paired, means = [], []
    for name in PARTITIONS:
        for seed in SEEDS:
            left,right=[index[(seed,o,name)] for o in OBJECTIVES]
            paired.append({"seed":seed,"partition":name,"worlds_per_arm":COUNTS[name],
                "full_successes_mean_J":left["greedy_full_successes"],"full_successes_mean_log_J":right["greedy_full_successes"],
                "difference_log_minus_mean":{m:right[m]-left[m] for m in METRICS}})
        for objective in OBJECTIVES:
            vals=[index[(s,objective,name)] for s in SEEDS]
            means.append({"objective":objective,"partition":name,"label":NAMES[name],"metrics":{
                m:{"seed_order":list(SEEDS),"values":[v[m] for v in vals],
                    "mean":statistics.fmean(v[m] for v in vals),"min":min(v[m] for v in vals),"max":max(v[m] for v in vals)} for m in METRICS}})
    primary=[r for r in paired if r["partition"]=="new_needs_and_layouts"]
    return {"rows":rows,"paired_by_partition":paired,"means_and_ranges":means,"behavior":behavior,
        "primary":{"partition":"new_needs_and_layouts","metric":"greedy_full_success_rate","endpoint_update":6000,
            "seed_order":list(SEEDS),"paired_differences":[r["difference_log_minus_mean"]["greedy_full_success_rate"] for r in primary],
            "mean_paired_difference":statistics.fmean(r["difference_log_minus_mean"]["greedy_full_success_rate"] for r in primary),
            "full_count_pairs":[[r["full_successes_mean_J"],r["full_successes_mean_log_J"]] for r in primary],
            "worlds_per_arm_per_seed":8244,"expected_direction":"positive"},
        "candidate_cells":candidate_cells,
        "candidate_cells_passed_by_objective":{o:sum(c["passes"] for c in candidate_cells if c["objective"]==o) for o in OBJECTIVES},
        "all_candidates_by_objective":{o:all(c["passes"] for c in candidate_cells if c["objective"]==o) for o in OBJECTIVES},
        "statistical_tests":0,"symbolic_phase_unlocked":False}


def analyze(run):
    run=Path(run).resolve()
    require((run/'execution/status.json').is_file(), "No completed status; do not read partial scores")
    status=read(run/'execution/status.json')
    require(status.get('status')=='completed' and status.get('completed_runs')==8, "All eight runs must finish")
    inputs={}
    def tracked(name):
        p=run/name;inputs[str(p)]=sha(p);return read(p)
    status=tracked('execution/status.json')
    result,plan,prepared,freeze=[tracked(p) for p in ('execution/results.json','plan.json','prepared.json','freeze.json')]
    require(result['status']=='completed' and result['completed_run_count']==8, 'Incomplete result')
    require(inputs[str(run/'plan.json')]==freeze['plan_sha256']==result['plan_sha256'], 'Plan anchor differs')
    require(inputs[str(run/'prepared.json')]==freeze['prepared_sha256']==plan['prepared_sha256'], 'Prepared anchor differs')
    require(plan['config']==prepared['config'] and plan['config']['seeds']==list(SEEDS)
            and plan['config']['objectives']==list(OBJECTIVES), 'Configuration differs')
    require(result['updates_total']==48000 and result['training_world_samples_total']==12288000
            and result['weighted_structural_action_contributions']==294912000, 'Budget differs')
    require(result['symbolic_phase_unlocked'] is False,'Unexpected symbolic gate')
    for relative,digest in plan['sources'].items():
        for p in (ROOT/relative,run/'source_snapshot'/relative):
            inputs[str(p)]=sha(p);require(inputs[str(p)]==digest,'Frozen source differs')
    expected=[f'seed_{s}_{o}' for s in SEEDS for o in OBJECTIVES]
    require(sorted(p.name for p in (run/'execution').glob('seed_*'))==sorted(expected),'Missing/extra run directory')
    runs=[]
    for i,name in enumerate(expected):
        r=tracked(f'execution/{name}/result.json')
        require(r==result['runs'][i],'Individual and aggregate run differ')
        files={f'execution/{name}/training.jsonl':r['training_log_sha256'],
               f'execution/{name}/checkpoint_6000.npz':r['final_checkpoint_sha256']}
        files.update({f'execution/{name}/final_{p}.npz':r['final'][p]['data_sha256'] for p in PARTITIONS})
        for relative,digest in files.items():
            p=run/relative;inputs[str(p)]=sha(p);require(inputs[str(p)]==digest,'Final data/log anchor differs')
        runs.append(r)
    for seed in SEEDS:
        pair=[r for r in runs if r['seed']==seed]
        require(pair[0]['initial_parameter_sha256']==pair[1]['initial_parameter_sha256'],'Different paired initialization')
    summary=summarize_runs(runs)
    stored=result['primary_comparison']
    require(stored['partition']=='new_needs_and_layouts' and stored['endpoint_update']==6000
            and [p['seed'] for p in stored['seed_pairs']]==list(SEEDS),'Stored primary scope differs')
    for actual,expected_delta in zip(stored['seed_pairs'],summary['primary']['paired_differences']):
        near(actual['paired_difference_log_minus_mean'],expected_delta,'primary paired difference')
    near(stored['equal_weight_mean_paired_difference'],summary['primary']['mean_paired_difference'],'primary mean')
    require(summary['all_candidates_by_objective']==result['all_seed_candidates_by_objective'],'Candidate flags differ')
    summary.update(status='complete_final_paired_summary',completed_at=datetime.now(timezone.utc).isoformat(),
        run_directory=str(run),input_sha256=inputs,summary_source_sha256=sha(__file__),
        summary_helper_source_sha256=sha(helpers.__file__),elapsed_seconds=result['elapsed_seconds'],
        paired_team_initializations=4,fixed_data_splits=1,endpoint_update=6000,
        weight_deserializations=0,neural_forward_calls=0,training_updates=0,
        scope='Final counts/hash summary; not the separate full paired execution/gradient/physical audit.')
    return summary


def markdown(s):
    p=s['primary']
    out=['# 平均期望与平均对数期望：终点报告草稿','',
        '本报告使用全部8运行更新6000的正式终点。新主体是随机初始化MLP，不是本轮重新部署或训练Qwen。', '',
        f"预先固定的双留出主读数：logJ−meanJ满分率平均配对差为{100*p['mean_paired_difference']:+.4f}个百分点；全部4个差值如下。正向或反向结果均保留。",'',
        '| 种子 | meanJ满分数/8244 | logJ满分数/8244 | 配对差（百分点） |','| --- | ---: | ---: | ---: |']
    for seed,counts,delta in zip(SEEDS,p['full_count_pairs'],p['paired_differences']):out.append(f'| {seed} | {counts[0]} | {counts[1]} | {100*delta:+.4f} |')
    out+=['','4个种子是4个配对团队初始化，所有运行共用一套需求/布局拆分；未进行显著性检验。', '',
        '## 全部32格','', '| 种子 | 目标 | 格 | 世界 | R0 | R0.5 | R1 | 未满分 | 满分率 | 原R均值 | 随机E[R] | 随机满分概率 |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in s['rows']:
        c=r['greedy_reward_counts'];out.append(f"| {r['seed']} | {r['objective']} | {r['label']} | {r['worlds']} | {c['0.0']} | {c['0.5']} | {c['1.0']} | {r['greedy_non_full_success_worlds']} | {100*r['greedy_full_success_rate']:.3f}% | {r['greedy_reward_mean']:.6f} | {r['exact_stochastic_expected_reward_mean']:.6f} | {100*r['exact_stochastic_full_success_probability_mean']:.3f}% |")
    out+=['','随机量是现有三策略独立采样分布的精确期望，不是额外真实行动次数。显示值有舍入，JSON保留全部精度。', '',
        '## 每格4种子的均值与范围','', '| 目标 | 格 | 满分率：均值（最小–最大） | 原R均值：均值（最小–最大） |','| --- | --- | --- | --- |']
    for r in s['means_and_ranges']:
        m=r['metrics'];out.append(f"| {r['objective']} | {r['label']} | {helpers.span(m['greedy_full_success_rate'],percent=True)} | {helpers.span(m['greedy_reward_mean'])} |")
    out+=['','## 全域动作与配对','', '| 种子 | 目标 | 联合argmax种类数 | 全域满分率 | 所有世界都物理匹配 | 实际配对及世界数 | 超单对界的格 |',
          '| --- | --- | ---: | ---: | --- | --- | --- |']
    for r in s['behavior']:
        pairs=', '.join(f'{k}:{v}' for k,v in r['executed_pair_world_counts'].items() if v)
        out.append(f"| {r['seed']} | {r['objective']} | {r['distinct_joint_argmax_actions']} | {100*r['full_success_rate']:.3f}% | {r['all_worlds_physically_matched']} | {pairs or '无'} | {', '.join(r['partitions_above_fixed_single_pair_bound']) or '无'} |")
    out+=['','动作种类变多不等于正确利用输入。固定单对界为训练需求格613/767、新需求格191/229，允许同一对随状态完美选择物资/目的地；超过该界至少需要多对实际参与，但不要求三对都出现。', '',
        '## 能力筛查与范围','',f"两个目标各12个种子×留出格达到99%的数目为：{s['candidate_cells_passed_by_objective']}；是否全通过为：{s['all_candidates_by_objective']}。符号阶段仍未自动解锁。",'',
        f"共48000更新、12288000训练状态抽样、294912000个24项加权贡献；整批记录耗时{s['elapsed_seconds']:.2f}秒。相同原生R、输入、动作、初始化和批次，不表示训练目标或梯度尺度保持相同。logJ目标同时改变状态权重、整体梯度及固定熵/clip的相对作用，不能把结果唯一归因于其中一项。",'',
        '本批不追加训练、不换种子、不补第三臂。无论干预效果如何，它都是完整信息能力控制，不是新约定或语言形成证据，也不是新的MARL方法贡献。完整配对/执行/物理审计请由根任务补充对应链接；本汇总只校验终点数字与字节来源，不替代那些审计。','']
    return '\n'.join(out)


def execute(run,out):
    out=Path(out).resolve();require(not out.exists(),'Refuse to overwrite summary')
    s=analyze(run);out.mkdir(parents=True,exist_ok=False)
    with (out/'summary.json').open('x') as f:json.dump(s,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    with (out/'结果与下一步_草稿.md').open('x') as f:f.write(markdown(s))
    return {'status':s['status'],'output':str(out),'summary_sha256':sha(out/'summary.json')}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',required=True,type=Path);parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args();print(json.dumps(execute(args.run,args.out),ensure_ascii=False))

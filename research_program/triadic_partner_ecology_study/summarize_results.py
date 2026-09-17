"""Completed-batch descriptive aggregation, with no model or learner imports.

All32 runs and every saved evaluation are required. NPZ bytes are hashed; scores
are read from the frozen producer JSON, not independently recomputed here.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path

SEEDS=(49101,49102,49103,49104)
ECOLOGIES=('unique','multiple')
CONDITIONS=('FI_silent','FI_live','PI_silent','PI_live')
PARTITIONS=('train','heldout_layouts')
CHECKPOINTS=(0,100,500,1500,3000,6000)
METRICS=('compatible_role_rate','topology_consistency_rate','physical_match_rate','full_success_rate','reward_mean')
COUNTS={'unique':{'train':34992,'heldout_layouts':11664},'multiple':{'train':107568,'heldout_layouts':35856}}


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):return json.loads(Path(path).read_text())

def write_new(path,obj):
    with Path(path).open('x') as stream:json.dump(obj,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')

def now():return datetime.now(timezone.utc).isoformat()

def key(row):return (row['seed'],row['ecology'],row['condition'],row['partition'],row['stage'],row['update'],row['mode'])


def proposal_inventory(distribution,worlds):
    """All proposed actions, including wait, overload, nonmutual and failed ones."""
    agents=[]
    require(sum(row['worlds'] for row in distribution)==worlds,'Incomplete full-domain action counts')
    require(len({tuple(row['action_indices']) for row in distribution})==len(distribution),'Repeated joint action count')
    for a,name in enumerate('ABC'):
        action_counts={};partners={other:0 for other in 'ABC' if other!=name};sites={str(i):0 for i in range(4)};destinations={'L':0,'R':0}
        for row in distribution:
            action=row['action_indices'][a];count=row['worlds']
            require(isinstance(action,int) and 0<=action<17 and isinstance(count,int) and count>0,'Invalid counted action')
            action_counts[str(action)]=action_counts.get(str(action),0)+count
            if action:
                packed=action-1;others=[n for n in 'ABC' if n!=name]
                partners[others[packed%2]]+=count;sites[str(packed//4)]+=count;destinations['LR'[(packed%4)//2]]+=count
        support=[p for p,c in partners.items() if c]
        agents.append(dict(agent=name,raw_worlds=worlds,wait_worlds=action_counts.get('0',0),
            transport_proposal_worlds=sum(partners.values()),always_wait=action_counts.get('0',0)==worlds,
            always_proposes_transport=action_counts.get('0',0)==0,partner_support=support,
            transport_partner_is_constant=len(support)==1,proposed_partner_worlds=partners,
            proposed_site_worlds=sites,proposed_destination_worlds=destinations,action_counts=action_counts))
    waiting=[x['agent'] for x in agents if x['always_wait']]
    active=[x for x in agents if x['always_proposes_transport']]
    fixed_pair=(len(waiting)==1 and len(active)==2 and all(x['transport_partner_is_constant'] for x in active)
        and active[0]['partner_support']==[active[1]['agent']] and active[1]['partner_support']==[active[0]['agent']])
    return dict(raw_worlds=worlds,distinct_joint_actions=len(distribution),agents=agents,
        globally_always_waiting_agents=waiting,globally_fixed_mutual_active_pair=''.join(x['agent'] for x in active) if fixed_pair else None,
        interpretation='Proposal support over all enumerated states, not only physically executed/successful cases; raw counts do not replace weighted scores.')


def paired_contrasts(index):
    rows=[]
    for s in SEEDS:
        for p in PARTITIONS:
            for metric in METRICS:
                cells={e:{c:index[(s,e,c,p,'final',6000,'natural')]['weighted'][metric] for c in CONDITIONS} for e in ECOLOGIES}
                pi={e:cells[e]['PI_live']-cells[e]['PI_silent'] for e in ECOLOGIES}
                fi={e:cells[e]['FI_live']-cells[e]['FI_silent'] for e in ECOLOGIES}
                rows.append(dict(seed=s,partition=p,metric=metric,cells=cells,PI_gains=pi,FI_gains=fi,
                    PI_DiD_unique_minus_multiple=pi['unique']-pi['multiple'],FI_DiD_unique_minus_multiple=fi['unique']-fi['multiple']))
    means=[]
    for p in PARTITIONS:
        for m in METRICS:
            group=[x for x in rows if x['partition']==p and x['metric']==m]
            means.append(dict(partition=p,metric=m,PI_DiD_unique_minus_multiple=sum(x['PI_DiD_unique_minus_multiple'] for x in group)/4,
                FI_DiD_unique_minus_multiple=sum(x['FI_DiD_unique_minus_multiple'] for x in group)/4))
    return dict(seed_values=rows,equal_seed_means=means,independent_paired_initializations=4,significance_test=None)


def aggregate_curves_and_cells(index):
    rows=[]
    for e in ECOLOGIES:
        for c in CONDITIONS:
            for p in PARTITIONS:
                for stage,updates in (('final',(6000,)),('monitor',CHECKPOINTS)):
                    for u in updates:
                        for mode in ('natural','closed'):
                            for metric in METRICS:
                                vals=[index[(s,e,c,p,stage,u,mode)]['weighted'][metric] for s in SEEDS]
                                rows.append(dict(ecology=e,condition=c,partition=p,stage=stage,update=u,mode=mode,metric=metric,
                                    seed_values=dict(zip(map(str,SEEDS),vals)),mean=sum(vals)/4,min=min(vals),max=max(vals)))
    return rows


def analyze(run,out):
    run=Path(run).resolve();out=Path(out).resolve();execution=run/'execution'
    require(not out.exists(),'Refuse to overwrite summary output')
    status=read(execution/'status.json')
    require(status['status']=='completed' and status['completed_runs']==32,'Wait for all32completed before analysis')
    source=execution/'results.json';result=read(source)
    require(result['status']=='completed' and result['completed_run_count']==32,'Incomplete batch result')
    prepared=read(run/'prepared.json');plan=read(run/'plan.json');freeze=read(run/'freeze.json')
    require(sha(run/'plan.json')==freeze['plan_sha256']==result['plan_sha256'],'Plan binding mismatch')
    require(sha(run/'prepared.json')==freeze['prepared_sha256']==plan['prepared_sha256'],'Prepared binding mismatch')
    expected=[(s,e,c) for s in SEEDS for e in ECOLOGIES for c in CONDITIONS]
    require([(x['seed'],x['ecology'],x['condition']) for x in result['runs']]==expected,'All32 runs in fixed order required')
    out.mkdir(parents=True,exist_ok=False);inputs={};records=[];aliases=[];inventories=[]
    def bind(path,digest=None):
        path=Path(path);value=sha(path)
        require(digest is None or digest==value,'File SHA mismatch: '+str(path))
        inputs[str(path)]=value;return value
    try:
        for filename in ('plan.json','prepared.json','freeze.json'):bind(run/filename)
        bind(source);bind(execution/'status.json')
        for relative,digest in plan['sources'].items():bind(run/'source_snapshot'/relative,digest)
        for row in result['runs']:
            s,e,c=row['seed'],row['ecology'],row['condition'];folder=execution/f'seed_{s}_{e}_{c}'
            bind(folder/'result.json');require(read(folder/'result.json')==row,'Run aggregate mismatch')
            require(row['updates']==6000 and len(row['monitor'])==6 and [x['update'] for x in row['monitor']]==list(CHECKPOINTS),'Wrong fixed checkpoints')
            bind(folder/'training.jsonl',row['training_log_sha256'])
            for checkpoint in row['monitor']:bind(folder/f"checkpoint_{checkpoint['update']:04d}.npz",checkpoint['checkpoint_sha256'])
            bind(folder/'checkpoint_6000.npz',row['final_checkpoint_sha256'])
            counts=row['full_domain_actions']['joint_argmax_action_counts']
            inv=proposal_inventory(counts,sum(COUNTS[e].values()));inv.update(seed=s,ecology=e,condition=c);inventories.append(inv)
            for stage,checkpoints in (('monitor',[(m['update'],m['monitor']) for m in row['monitor']]),('final',[(6000,row['final'])])):
                for u,parts in checkpoints:
                    require(set(parts)==set(PARTITIONS),'Missing evaluation partition')
                    for p in PARTITIONS:
                        require(set(parts[p])=={'natural','closed'},'Missing route control')
                        for mode in ('natural','closed'):
                            value=parts[p][mode];identity=dict(seed=s,ecology=e,condition=c,partition=p,stage=stage,update=u,mode=mode)
                            n=672 if stage=='monitor' else COUNTS[e][p]
                            require(value['raw']['worlds']==value['uniform']['worlds']==n,'Wrong evaluation denominator')
                            for metric in METRICS:
                                v=value['weighted'][metric];require(math.isfinite(v) and -1e-12<=v<=1+1e-12,'Invalid weighted rate')
                            require(abs(value['raw']['weights_sum']-1)<1e-12,'Unnormalized target weights')
                            expected_reuse=(mode=='closed' and c.endswith('_silent'))
                            require(value['reused_natural']==expected_reuse,'Incorrect closed-route reuse')
                            expected_condition=c.replace('_live','_silent') if mode=='closed' else c
                            require(value['actual_rollout_condition']==expected_condition,'Wrong closed information/route condition')
                            if expected_reuse:
                                nat=parts[p]['natural']
                                require(all(value[k]==nat[k] for k in ('data_file','data_sha256','weights_sha256','weighted','raw','uniform')),'Silent alias differs from natural')
                                aliases.append(dict(**identity,source_mode='natural',data_file=value['data_file']));continue
                            bind(execution/value['data_file'],value['data_sha256'])
                            bind(execution/value['weights_file'],value['weights_file_sha256'])
                            records.append(dict(**identity,worlds=n,weighted=value['weighted'],
                                uniform_reward_counts=value['uniform']['greedy_reward_counts'],
                                uniform_failure_categories=value['uniform']['greedy_failure_categories'],
                                uniform_full_success_rate=value['uniform']['greedy_full_success_rate'],
                                raw_compatible_role_worlds=value['raw']['compatible_role_worlds'],
                                data_file=value['data_file'],data_sha256=value['data_sha256'],
                                weights_file=value['weights_file'],weights_sha256=value['weights_sha256'],
                                weight_array=value['weight_array'],population_weighting=value['population_weighting']))
        require(len(records)==672 and len(aliases)==224,'Expected672 actual evaluations and224 explicit aliases')
        require(len({key(x) for x in records})==672 and len({x['data_file'] for x in records})==672,'Duplicate actual evaluation')
        require(sum(x['stage']=='final' for x in records)==96 and sum(x['stage']=='monitor' for x in records)==576,'Wrong actual stage counts')
        index={key(x):x for x in records}
        for alias in aliases:index[key(alias)]=index[(*key(alias)[:-1],'natural')]
        require(len(index)==896,'Incomplete logical condition grid')
        contrasts=paired_contrasts(index);aggregates=aggregate_curves_and_cells(index)
        primary=result['primary_comparison'];require([x['seed'] for x in primary['seed_pairs']]==list(SEEDS),'Primary seed order')
        for i,s in enumerate(SEEDS):
            role=next(x for x in contrasts['seed_values'] if x['seed']==s and x['partition']=='heldout_layouts' and x['metric']=='compatible_role_rate')
            full=next(x for x in contrasts['seed_values'] if x['seed']==s and x['partition']=='heldout_layouts' and x['metric']=='full_success_rate')
            for name,value in (('primary_DiD_unique_minus_multiple',role['PI_DiD_unique_minus_multiple']),
                ('FI_DiD_unique_minus_multiple',role['FI_DiD_unique_minus_multiple']),('full_success_DiD_unique_minus_multiple',full['PI_DiD_unique_minus_multiple'])):
                require(abs(primary['seed_pairs'][i][name]-value)<1e-14,'Stored primary contrast inconsistent')
        channel=[]
        for s in SEEDS:
            for e in ECOLOGIES:
                for c in CONDITIONS:
                    for p in PARTITIONS:
                        for stage,updates in (('final',(6000,)),('monitor',CHECKPOINTS)):
                            for u in updates:
                                vals={m:index[(s,e,c,p,stage,u,m)] for m in ('natural','closed')}
                                channel.append(dict(seed=s,ecology=e,condition=c,partition=p,stage=stage,update=u,
                                    natural_minus_closed={m:vals['natural']['weighted'][m]-vals['closed']['weighted'][m] for m in METRICS},
                                    closed_is_natural_reference=c.endswith('_silent')))
        summary=dict(status='completed',created_at=now(),run=str(run),contract=dict(seeds=list(SEEDS),ecologies=list(ECOLOGIES),
            conditions=list(CONDITIONS),partitions=list(PARTITIONS),checkpoints=list(CHECKPOINTS),actual_monitor_records=576,actual_final_records=96,
            natural_monitor_records=384,extra_live_closed_monitor_records=192,natural_final_records=64,extra_live_closed_final_records=32,
            aliases=224,monitor_worlds=672,target_weights='equal destination strata; conditional uniform needs/layouts/owners'),
            execution_elapsed_seconds=result['elapsed_seconds'],evaluations=records,closed_aliases=aliases,
            paired_contrasts=contrasts,aggregates=aggregates,channel_differences=channel,full_domain_proposals=inventories,
            source_primary_comparison=primary,input_sha256=inputs,
            scope=dict(weights_read=False,model_parameters_loaded=False,neural_forwards=0,training_updates=0,
                actual_npz_bytes_verified=True,numeric_scores_from_frozen_producer_json=True,
                independent_physical_recomputation=False,primary_changes_after_results=False,
                note='Behavioral role/action descriptions, not semantic or compositional language evidence; four team initializations and one development layout split.'))
        write_new(out/'summary.json',summary)
        fields=('seed','ecology','condition','partition','mode','worlds',*METRICS,'best_fixed_pair_gamma','compatible_role_excess_best_fixed_pair','data_file')
        with (out/'complete_endpoints.csv').open('x',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
            for row in records:
                if row['stage']!='final':continue
                writer.writerow({k:row[k] if k in row else row['weighted'][k] for k in fields})
        lines=['# 完整加权终点与配对比较','', '32组全部完成后汇总。64份自然＋32份额外闭通道终点；静默closed引用自然文件，不重复计数。比例表以百分数表示，差以百分点表示。监测672样本另存，不替代这些全量目标分布结果。','',
            '| 种子 | PI角色主DiD | FI角色DiD | PI满分DiD | FI满分DiD |','|---|---:|---:|---:|---:|']
        for s in SEEDS:
            rs={x['metric']:x for x in contrasts['seed_values'] if x['seed']==s and x['partition']=='heldout_layouts'}
            lines.append('| '+str(s)+' | '+' | '.join(f'{100*rs[m][k]:+.6f}' for m,k in [('compatible_role_rate','PI_DiD_unique_minus_multiple'),('compatible_role_rate','FI_DiD_unique_minus_multiple'),('full_success_rate','PI_DiD_unique_minus_multiple'),('full_success_rate','FI_DiD_unique_minus_multiple')])+' |')
        lines+=['','DiD=unique的开放−静默，再减multiple的开放−静默。全部四种子等权；不做显著性检验。','',
            '| 种子 | 生态 | 条件 | 分区 | 路由 | 兼容角色% | 物理匹配% | 满分% | 原R |','|---|---|---|---|---|---:|---:|---:|---:|']
        for row in records:
            if row['stage']=='final':
                w=row['weighted'];lines.append(f"| {row['seed']} | {row['ecology']} | {row['condition']} | {row['partition']} | {row['mode']} | {100*w['compatible_role_rate']:.6f} | {100*w['physical_match_rate']:.6f} | {100*w['full_success_rate']:.6f} | {w['reward_mean']:.8f} |")
        lines+=['','## 全域提议行为','', '下表由全部自然终点联合动作的原始计数得到，包含超载、不互选和失败；不只统计成功执行者。伙伴支持指发生过的运输提议，频数不代表词义。','',
            '| 种子 | 生态 | 条件 | 联合动作种类 | 全域恒等待者 | 全域固定互选活跃对 | A/B/C的伙伴提议支持 |','|---|---|---|---:|---|---|---|']
        for row in inventories:
            supports='；'.join(x['agent']+':'+('/'.join(x['partner_support']) or '无运输') for x in row['agents'])
            lines.append(f"| {row['seed']} | {row['ecology']} | {row['condition']} | {row['distinct_joint_actions']} | {','.join(row['globally_always_waiting_agents']) or '无'} | {row['globally_fixed_mutual_active_pair'] or '无'} | {supports} |")
        lines+=['','[JSON](summary.json)保留全部五项指标、每种子、每检查点、两路由及权重来源；[完整终点CSV](complete_endpoints.csv)保留96行真实评价。','',
            '关闭通道也改变冻结政策的输入分布；角色变化、通信依赖、物理执行和任务满分分列，均不自动证明形成了词义或语法。']
        (out/'完整结果表.md').write_text('\n'.join(lines)+'\n')
        require(all(sha(path)==digest for path,digest in inputs.items()),'Inputs changed during summary')
        receipt=dict(status='completed',created_at=now(),script_sha256=sha(__file__),input_files=len(inputs),
            outputs={p.name:sha(p) for p in out.iterdir() if p.is_file()},neural_forwards=0,training_updates=0)
        write_new(out/'receipt.json',receipt)
        return dict(status='completed',out=str(out),summary_sha256=sha(out/'summary.json'),actual_evaluations=672)
    except BaseException as error:
        write_new(out/'failure.json',dict(status='failed',error_type=type(error).__name__,error=str(error)));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',required=True,type=Path);parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args();print(json.dumps(analyze(args.run,args.out),ensure_ascii=False))

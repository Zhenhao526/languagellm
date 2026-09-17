"""Complete matched-vs-shuffled baseline analysis; no new model inference."""
from __future__ import annotations
import argparse,hashlib,itertools,json,sys,tempfile
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'redesign_v0.16'))
import analyze_branches as prior
previous=prior.previous;v13=prior.v13
import numpy as np
import torch
SEEDS=prior.SEEDS;PARTITIONS=prior.PARTITIONS;TIMES=prior.TIMES;MODES=prior.MODES;GROUPS=prior.GROUPS;PREFIXES=prior.PREFIXES
ARMS=('matched','shuffled')
CONTRASTS={'state_correspondence':{'matched':1,'shuffled':-1}}
LABELS={'matched':'Matched baseline','shuffled':'Shuffled baseline'}
COLORS={'matched':'#3f9274','shuffled':'#bd7444'}
SOURCE=prior.SOURCE;RESET_SOURCE=prior.RESET_SOURCE;SCALED_SOURCE=prior.SCALED_SOURCE;CALIBRATION=prior.CALIBRATION
PHASES=('all','entropy_on','entropy_off')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def array_sha(v):return hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()

def load_social(path,source,seed,p,label,times,formal):
    cfg,curve,result=[read(path/name) for name in ('config.json','curve.json','result.json')]
    actual_arm={'matched':'reset_scaled','shuffled':'both_sender_baseline_shuffle'}[label]
    for obj in (cfg,result):assert (obj['seed'],obj['partition'],obj['arm'],obj['updates'])==(seed,p,actual_arm,times[-1])
    assert cfg['checkpoints']==list(times)==[r['update'] for r in curve]
    assert cfg['training_pool']==v13.private.partition_maps(p)['old'].tolist()
    assert cfg['sealed_never_trained'] and result['sealed_never_trained'] and result['frozen_modules_verified']
    assert (cfg['rng_world_namespace'],cfg['rng_policy_namespace'],cfg['training_policy_stream'],cfg['evaluation_policy_stream'])==(13014,11011,11,91)
    assert (cfg['learning_rate'],cfg['entropy_coefficient'],cfg['entropy_off_after'],cfg['role_loss_weights'])==(.0007,.02,2100,{'sender':.5,'receiver':.5})
    if formal:assert cfg['batch']==512 and cfg['eval_n']==9600
    for name,digest in cfg['source_hashes'].items():assert sha(name)==digest,name
    assert cfg['source_checkpoint']['sha256']==sha(cfg['source_checkpoint']['path'])
    pcfg=read(source/f'private_s{seed}_p{p}_control/config.json')
    final={};normal=None
    for mode in MODES:
        raw=v13.load(path/f'final_{mode}.npz');final[mode]=v13.recompute_social(raw,p)
        v13.nested_equal(final[mode],result['scores'][mode],path.name+'.'+mode)
        if mode=='normal':
            normal=raw;assert np.array_equal(raw['sent'],raw['delivered'])
        elif mode=='blank':assert not raw['delivered'].any()
        for key in ('positions','photo_ids','goals','menu','inventory','history'):assert np.array_equal(raw[key],normal[key])
    v13.nested_equal(curve[-1]['scores']['normal'],final['normal'])
    assert len({r['scores']['normal']['world_sha256'] for r in curve})==1
    for row in curve:
        row['protocol']=v13.load_protocol(path,row['update'],p,pcfg['test_photo_ids'])
        for d in (0,1):
            for group in GROUPS:
                assert row['scores']['normal']['direction_groups'][d][group]['n']==cfg['eval_n']*len(v13.private.partition_maps(p)[group])//60
    if label=='shuffled':
        assert cfg['sender_baseline_shuffle'] and cfg['baseline_rng_namespace']==17017
        assert cfg['factors']==[1,1]
    sequence=hashlib.sha256()
    with (path/'training.jsonl').open() as stream:
        for update,line in enumerate(stream,1):
            row=json.loads(line);assert row['update']==update
            assert row['stats']['n']==cfg['batch'] and row['stats']['map_groups']['added']['n']==row['stats']['map_groups']['sealed']['n']==0
            assert row['entropy_weight']==(.02 if update<=2100 else 0.)
            sequence.update((row['stats']['world_sha256']+'\n').encode())
    assert update==times[-1]
    return dict(seed=seed,partition=p,arm=label,source_arm=actual_arm,path=str(path),curve=curve,final=final,
        world_sequence_sha256=sequence.hexdigest(),parameter_partition=cfg['parameter_partition'],
        initial_sha256=cfg['initial_sha256'],source_hashes=cfg['source_hashes'],
        file_hashes={n:sha(path/n) for n in ('config.json','curve.json','result.json','initial.pt','final.pt','training.jsonl')})


def summarize(runs,seeds,partitions,times):
    cells=[]
    for seed,arm in itertools.product(seeds,ARMS):
        selected=[r for r in runs if r['seed']==seed and r['arm']==arm]
        assert {r['partition'] for r in selected}==set(partitions)
        curve=[]
        for t in times:
            points=[v13.point_metrics(next(c for c in r['curve'] if c['update']==t)) for r in selected]
            curve.append(dict(update=t,metrics={k:float(np.mean([x[k] for x in points])) for k in points[0]}))
        metrics=dict(curve[-1]['metrics'])
        for g in GROUPS:metrics[f'auc.{g}.N']=v13.auc(times,[x['metrics'][f'normal.{g}.N'] for x in curve])
        for mode in MODES:
            for group in (*GROUPS,'all'):
                rows=[r['final'][mode] if group=='all' else r['final'][mode]['map_groups'][group] for r in selected]
                for k,k2 in [('joint','both_accuracy'),('reward','mean_reward')]:metrics[f'intervention.{mode}.{group}.{k}']=float(np.mean([z[k2] for z in rows]))
        cells.append(dict(seed=seed,arm=arm,partitions_averaged=len(partitions),directions_per_partition=2,metrics=metrics,curve=curve))
    aggregate={}
    for arm in ARMS:
        chosen=[c for c in cells if c['arm']==arm]
        aggregate[arm]=dict(metrics={k:float(np.mean([c['metrics'][k] for c in chosen])) for k in cells[0]['metrics']},
            curve=[dict(update=t,metrics={k:float(np.mean([next(x for x in c['curve'] if x['update']==t)['metrics'][k] for c in chosen])) for k in cells[0]['curve'][0]['metrics']}) for t in times])
    contrasts={};lookup={(c['seed'],c['arm']):c for c in cells}
    for name,weights in CONTRASTS.items():
        differences={k:[sum(w*lookup[s,arm]['metrics'][k] for arm,w in weights.items()) for s in seeds] for k in cells[0]['metrics']}
        contrasts[name]=dict(weights=weights,seed_differences=differences,
            mean_differences={k:float(np.mean(v)) for k,v in differences.items()},difference_ranges={k:[min(v),max(v)] for k,v in differences.items()})
    return dict(seed_cells=cells,aggregate=aggregate,contrasts=contrasts)


def figures(r,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
    files=[];s=r['summary'];tt=r['times']
    def save(fig,name):
        fig.canvas.draw()
        for ax in fig.axes:
            ax.get_xticklabels();ax.get_yticklabels()
        fig.canvas.draw()
        fig.savefig(out/f'{name}.png');fig.savefig(out/f'{name}.pdf');plt.close(fig)
        files.append(dict(png=str(out/f'{name}.png'),pdf=str(out/f'{name}.pdf')))
    fig,axes=plt.subplots(1,3,figsize=(12.5,4.3),sharey=True,constrained_layout=True)
    for ax,g,title in zip(axes,GROUPS,('Old 18: trained','Added 6: never trained','Sealed 6: primary never-trained set')):
        for arm in ARMS:
            for c in s['seed_cells']:
                if c['arm']==arm:ax.plot(tt,[100*x['metrics'][f'normal.{g}.N'] for x in c['curve']],color=COLORS[arm],alpha=.19,lw=.8)
            ax.plot(tt,[100*x['metrics'][f'normal.{g}.N'] for x in s['aggregate'][arm]['curve']],color=COLORS[arm],lw=2,label=LABELS[arm])
        ax.set(title=title,xlabel='Social learning updates',ylim=(-2,102));ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Greedy sender + greedy receiver: N (%)');axes[1].legend(frameon=False,fontsize=8)
    fig.suptitle('Four inherited source seeds; only shuffled runs are new; thin lines average partitions and directions within seed',fontsize=9)
    save(fig,'01_four_cell_learning')
    fig,axes=plt.subplots(1,3,figsize=(12.5,4.3),constrained_layout=True)
    for ax,(key,other,title,ylabel,scale) in zip(axes,[('native_joint_oracle','G','Native sender: oracle and G','Common 30-map joint success (%)',100),
            ('N',None,'Greedy sender + greedy receiver','Common 30-map natural N (%)',100),('joint_entropy',None,'Native sender ambiguity','H(food, water | message), nats',1)]):
        for arm in ARMS:
            for k in (key,other) if other else (key,):
                for c in s['seed_cells']:
                    if c['arm']==arm:ax.plot(tt,[scale*x['metrics'][f'protocol.common30.{k}'] for x in c['curve']],color=COLORS[arm],ls='--' if k=='native_joint_oracle' else '-',alpha=.12,lw=.7)
                ax.plot(tt,[scale*x['metrics'][f'protocol.common30.{k}'] for x in s['aggregate'][arm]['curve']],color=COLORS[arm],ls='--' if k=='native_joint_oracle' else '-',lw=2,
                    label=LABELS[arm]+(': oracle' if k=='native_joint_oracle' else ': G' if k=='G' else ''))
        ax.set(title=title,ylabel=ylabel,xlabel='Social learning updates');ax.grid(axis='y',alpha=.2);ax.tick_params(axis='y',labelleft=True)
        if scale==100:ax.set_ylim(-2,102);ax.set_yticks(np.arange(0,101,20))
    axes[0].legend(frameon=False,fontsize=6.5);axes[1].legend(frameon=False,fontsize=8)
    fig.suptitle('Fixed 16 photo pairs; exact 49-code distributions; each native oracle uses its own sender policy',fontsize=9)
    save(fig,'02_common_support_and_natural_use')
    return files


def inventory(batch,source,reset_source,scaled_source,calibration,seeds,parts,times):
    groups=[];missing=[]
    names=['config.json','result.json','curve.json','training.jsonl','initial.pt','final.pt',*(f'final_{m}.npz' for m in MODES),*(f'protocol_{t:04d}.json' for t in times),*(f'protocol_{t:04d}_d{d}.npz' for t in times for d in (0,1))]
    for seed,p in itertools.product(seeds,parts):
        folders=dict(matched=scaled_source/f'social_s{seed}_p{p}_reset_scaled',shuffled=batch/f'social_s{seed}_p{p}_both_sender_baseline_shuffle')
        groups.append((seed,p,folders))
        for folder in folders.values():
            absent=[n for n in names if not (folder/n).is_file()]
            if absent:missing.append(dict(path=str(folder),files=absent))
        paths=[reset_source/f'social_s{seed}_p{p}_reset/initial.pt',*(calibration/f's{seed}_p{p}'/name for name in ('calibration.json','person0.npz','person1.npz')),
               folders['shuffled']/'train_0001.npz']
        if times[-1]>=2101:paths.append(folders['shuffled']/'train_2101.npz')
        for path in paths:
            if not path.is_file():missing.append(dict(path=str(path)))
    for folder,name in ((batch,'training_complete.json'),(source,'training_complete.json'),(reset_source,'training_complete.json'),(scaled_source,'training_complete.json'),(batch,'source_receipt.json'),(batch,'calibration_receipt.json'),(calibration,'calibration_complete.json')):
        if not (folder/name).is_file():missing.append(dict(path=str(folder/name)))
    return groups,missing


def verify_pair(seed,p,cells,calibration,reset_source,times):
    base=torch.load(reset_source/f'social_s{seed}_p{p}_reset/initial.pt',weights_only=True)
    matched=torch.load(Path(cells['matched']['path'])/'initial.pt',weights_only=True)
    folder=Path(cells['shuffled']['path']);initial=torch.load(folder/'initial.pt',weights_only=True);final=torch.load(folder/'final.pt',weights_only=True)
    cfg=read(folder/'config.json');alphas=[person['alpha_float32'] for person in calibration['persons']]
    assert cfg['calibration_scales_float32']==alphas
    assert cfg['parameter_partition']==cells['matched']['parameter_partition']
    for key in ('reset_initial','retained_social_initial','calibration','scaled_initial'):
        ref=cfg['source_inputs'][key];assert sha(ref['path'])==ref['sha256']
    for who in (0,1):
        assert set(matched[who])==set(base[who])|{'visual_scale'}
        assert set(initial[who])==set(base[who])|{'send_context.h_scale','send_value.h_scale'}==set(final[who])
        assert float(matched[who]['visual_scale'])==alphas[who]
        assert all(torch.equal(value,matched[who][key]) and torch.equal(value,initial[who][key]) for key,value in base[who].items())
        assert cfg['branch_scales_float32'][who]==dict(policy=alphas[who],value=alphas[who])
        for key in ('send_context.h_scale','send_value.h_scale'):
            assert initial[who][key].ndim==0 and initial[who][key].dtype==torch.float32 and float(initial[who][key])==alphas[who]
            assert torch.equal(initial[who][key],final[who][key])
        for key,value in initial[who].items():
            if key.startswith(PREFIXES+('project.',)) or key=='input_transform':assert torch.equal(value,final[who][key])
    count=0
    for t in times:
        for who in (0,1):
            ref=v13.load(Path(cells['matched']['path'])/f'protocol_{t:04d}_d{who}.npz');new=v13.load(folder/f'protocol_{t:04d}_d{who}.npz')
            assert np.array_equal(new['h'],new['raw_h'])
            assert np.array_equal(new['policy_h'],new['raw_h']*np.float32(alphas[who]))
            assert np.array_equal(new['policy_h'],new['value_h']) and np.array_equal(new['policy_h'],ref['h']);count+=4
            if t==0:
                for key in ('first_logits','second_logits','greedy_message','receiver_logits'):assert np.array_equal(ref[key],new[key]);count+=1
    return dict(seed=seed,partition=p,all_original_tensors_equal=True,only_branch_buffers_added=True,scales_fixed=True,latent_and_initial_policy_equalities=count)


def phases(update):return ('all','entropy_on' if update<=2100 else 'entropy_off')

def diagnostics(runs,times):
    role_rows=[];manip_rows=[];trace_checks=[];hashes={}
    for run in runs:
        folder=Path(run['path']);cfg=read(folder/'config.json');seed,p,arm=run['seed'],run['partition'],run['arm'];n=cfg['batch']//2
        logs=[json.loads(line) for line in (folder/'training.jsonl').read_text().splitlines()];hashes[str(folder/'training.jsonl')]=sha(folder/'training.jsonl')
        assert len(logs)==times[-1]
        for who in (0,1):
            for phase in PHASES:
                selected=[row for row in logs if phase in phases(row['update'])]
                if not selected:continue
                for role in ('sender','receiver'):
                    norms=np.asarray([row['agents'][who]['gradient_norm_by_role'][role] for row in selected],np.float32)
                    coefficients=np.minimum(np.float32(1),np.float32(2)/(norms+np.float32(1e-6)))
                    role_rows.append(dict(seed=seed,partition=p,arm=arm,person=who,role=role,phase=phase,person_updates=len(selected),active_count=int((coefficients<1).sum()),rate=float((coefficients<1).mean()),max_norm=float(norms.max()),mean_norm=float(norms.astype(np.float64).mean()),min_coefficient=float(coefficients.min())))
                if arm=='matched':
                    manip_rows.append(dict(seed=seed,partition=p,arm=arm,person=who,phase=phase,available=False,reason='Historical matched logs lack baseline vectors/intensity summaries; do not infer their distribution from shuffled runs'))
                    continue
                metadata=[row['agents'][who]['sender_baseline'] for row in selected]
                for row,meta in zip(selected,metadata):
                    entropy=[17017,seed,p,row['update']-1,who];idx=np.random.default_rng(np.random.SeedSequence(entropy)).permutation(n).astype(np.int64,copy=False)
                    assert meta['mode']=='shuffle' and meta['namespace']==17017 and meta['step_zero_based']==row['update']-1 and meta['permutation_seed_entropy']==entropy
                    assert meta['n']==n and meta['permutation_sha256']==array_sha(idx) and meta['fixed_point_count']==int((idx==np.arange(n)).sum())
                    assert meta['original_sorted_sha256']==meta['used_sorted_sha256'] and meta['multiset_preserved']
                    assert 0<=meta['changed_value_count']<=n and 0<=meta['unchanged_value_count']<=n
                    for key in ('original_std','used_minus_original_mean_abs','used_minus_original_max_abs','used_minus_original_rms'):assert np.isfinite(meta[key]) and meta[key]>=0
                examples=n*len(selected)
                manip_rows.append(dict(seed=seed,partition=p,arm=arm,person=who,phase=phase,available=True,person_updates=len(selected),examples=examples,
                    fixed_point_count=sum(m['fixed_point_count'] for m in metadata),fixed_point_rate=sum(m['fixed_point_count'] for m in metadata)/examples,
                    changed_value_count=sum(m['changed_value_count'] for m in metadata),changed_value_rate=sum(m['changed_value_count'] for m in metadata)/examples,
                    mean_original_std=float(np.mean([m['original_std'] for m in metadata])),
                    mean_abs_change=float(np.mean([m['used_minus_original_mean_abs'] for m in metadata])),
                    max_abs_change=max(m['used_minus_original_max_abs'] for m in metadata),
                    rms_change=float(np.sqrt(np.mean([m['used_minus_original_rms']**2 for m in metadata]))),
                    all_multiset_hashes_equal=True,all_permutations_independently_reproduced=True,
                    intensity_scope='All-step intensity summaries read from logs; raw arrays independently recomputed only at steps1/2101'))
        if arm=='shuffled':
            for update in (1,2101):
                if update>times[-1]:continue
                path=folder/f'train_{update:04d}.npz';z=v13.load(path);hashes[str(path)]=sha(path)
                for key in ('baseline_permutation','baseline_original','baseline_used','baseline_target','baseline_logp','baseline_entropy'):assert z[key].shape==(2,n)
                for who in (0,1):
                    meta=logs[update-1]['agents'][who]['sender_baseline'];original=z['baseline_original'][who];used=z['baseline_used'][who];idx=z['baseline_permutation'][who]
                    assert idx.dtype==np.int64 and original.dtype==used.dtype==np.float32
                    assert np.array_equal(used,original[idx]) and array_sha(idx)==meta['permutation_sha256']
                    assert array_sha(original)==meta['original_sha256'] and array_sha(used)==meta['used_sha256']
                    assert array_sha(np.sort(original.view(np.uint32)))==meta['original_sorted_sha256']
                    assert array_sha(np.sort(used.view(np.uint32)))==meta['used_sorted_sha256']
                    diff=used.astype(np.float64)-original.astype(np.float64)
                    vals=dict(original_std=float(np.std(original.astype(np.float64))),used_minus_original_mean_abs=float(np.mean(np.abs(diff))),used_minus_original_max_abs=float(np.max(np.abs(diff))),used_minus_original_rms=float(np.sqrt(np.mean(diff*diff))))
                    assert all(abs(meta[k]-v)<1e-12 for k,v in vals.items())
                    assert meta['changed_value_count']==int(np.count_nonzero(diff))
                    mask=z['scout']==who;assert np.array_equal(z['baseline_target'][who],z['reward'][mask]-np.float32(1))
                trace_checks.append(dict(seed=seed,partition=p,arm=arm,update=update,persons=2,baseline_indices_values_targets_hashes_intensities_verified=True))
    summary={}
    for arm in ARMS:
        summary[arm]={}
        for phase in PHASES:
            selected=[r for r in role_rows if r['arm']==arm and r['phase']==phase]
            if not selected:continue
            row={}
            for role in ('sender','receiver'):
                rr=[r for r in selected if r['role']==role];n=sum(r['person_updates'] for r in rr)
                row[role]=dict(person_updates=n,active_count=sum(r['active_count'] for r in rr),rate=sum(r['active_count'] for r in rr)/n,max_norm=max(r['max_norm'] for r in rr),mean_norm=sum(r['mean_norm']*r['person_updates'] for r in rr)/n,min_coefficient=min(r['min_coefficient'] for r in rr))
            mm=[r for r in manip_rows if r['arm']==arm and r['phase']==phase and r['available']]
            if mm:
                examples=sum(r['examples'] for r in mm);count=sum(r['person_updates'] for r in mm)
                row['manipulation']=dict(available=True,examples=examples,fixed_point_rate=sum(r['fixed_point_count'] for r in mm)/examples,changed_value_rate=sum(r['changed_value_count'] for r in mm)/examples,
                    mean_original_std=sum(r['mean_original_std']*r['person_updates'] for r in mm)/count,
                    mean_abs_change=sum(r['mean_abs_change']*r['person_updates'] for r in mm)/count,
                    max_abs_change=max(r['max_abs_change'] for r in mm),rms_change=float(np.sqrt(sum(r['rms_change']**2*r['examples'] for r in mm)/examples)))
            else:row['manipulation']=dict(available=False)
            summary[arm][phase]=row
    return dict(role_rows=role_rows,manipulation_rows=manip_rows,trace_checks=trace_checks,summary=summary,input_hashes=hashes,all_step_permutations_reproduced=True,raw_value_recomputation_scope='steps1 and2101 only; other intensity values are logged diagnostics',clip_formula='float32 min(1,2/(norm+1e-6))')


def report(r):
    s=r['summary'];c=s['contrasts']['state_correspondence'];lines=['# 基线与场景对应：分析草稿','',
        '全部12新增shuffle完成后统一分析，复用v0.15 matched12；四旧来源各自先平均三分区、两方向，n=4。24运行不是24新训练或24独立来源；单支路结果仅属于上一轮背景。','',
        '唯一主比较为2400步sealed自然N的matched−shuffled。八点全程归一化AUC及old/added、通信干预和操纵/裁剪均为辅助。','',
        '| 条件 | old N | added N | sealed N | sealed AUC |','| --- | ---: | ---: | ---: | ---: |']
    for arm in ARMS:
        m=s['aggregate'][arm]['metrics'];lines.append(f'| {LABELS[arm]} | '+' | '.join(f'{100*m[k]:.2f}%' for k in ('normal.old.N','normal.added.N','normal.sealed.N','auc.sealed.N'))+' |')
    lines+=['','| 指标 | 四来源matched−shuffled（百分点） | 平均差 |','| --- | --- | ---: |']
    for key in ('normal.sealed.N','auc.sealed.N','normal.old.N','auc.old.N','normal.added.N','auc.added.N'):
        lines.append(f'| {key} | '+'、'.join(f'{100*v:+.4f}' for v in c['seed_differences'][key])+f' | {100*c["mean_differences"][key]:+.4f} |')
    lines+=['','| 条件 | common30原生联合MAP | 原生Q | 原生G | 贪心N | 原生联合条件熵 | 贪心完整码数 |','| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for arm in ARMS:
        m=s['aggregate'][arm]['metrics'];pre='protocol.common30.'
        lines.append(f'| {LABELS[arm]} | '+' | '.join(f'{100*m[pre+k]:.2f}%' for k in ('native_joint_oracle','Q','G','N'))+f' | {m[pre+"joint_entropy"]:.6f} | {m[pre+"greedy_used_codes"]:.2f} |')
    lines+=['','协议使用固定16照片对，主自然评价使用9600世界；两者不直接相减。Q为原生双方解析随机期望，G为原生发送与贪心接收，N另使用逐符号贪心发送。原生MAP只约束自身发送政策和共同支持，不与另一政策N相减，也不拼接子组oracle。','',
        '| 条件 | 模式 | 全30图J | sealed J |','| --- | --- | ---: | ---: |']
    for arm in ARMS:
        m=s['aggregate'][arm]['metrics']
        for mode in MODES:lines.append(f'| {LABELS[arm]} | {mode} | {100*m[f"intervention.{mode}.all.joint"]:.2f}% | {100*m[f"intervention.{mode}.sealed.joint"]:.2f}% |')
    lines+=['','## 操纵强度与裁剪','',
        '每步独立重建置换并核对SHA/固定点/多重集记录；实际baseline数组只在1/2101步保存并重算，其余幅度读自日志，不称完整轨迹value前向重放。matched旧日志缺少原value向量及幅度摘要，因此不填造匹配臂的基线方差。','',
        '| 条件 | 阶段 | sender裁剪次数/更新 | receiver裁剪次数/更新 | sender最大范数 |','| --- | --- | ---: | ---: | ---: |']
    for arm,phases0 in r['diagnostics']['summary'].items():
        for phase,v in phases0.items():lines.append(f'| {LABELS[arm]} | {phase} | {v["sender"]["active_count"]}/{v["sender"]["person_updates"]} | {v["receiver"]["active_count"]}/{v["receiver"]["person_updates"]} | {v["sender"]["max_norm"]:.6f} |')
    lines+=['','| shuffle阶段 | 索引固定率 | 数值改变率 | 原基线std均值 | 改变量mean abs | 改变量合并RMS | 最大绝对改变量 |','| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for phase,v in r['diagnostics']['summary']['shuffled'].items():
        m=v['manipulation'];lines.append(f'| {phase} | {100*m["fixed_point_rate"]:.4f}% | {100*m["changed_value_rate"]:.4f}% | {m["mean_original_std"]:.6f} | {m["mean_abs_change"]:.6f} | {m["rms_change"]:.6f} | {m["max_abs_change"]:.6f} |')
    lines+=['','置换只保持该臂当前batch的baseline多重集，包含固定点和重复值；不声称完全消除状态信息，或跨臂完整基线时程、拟合速度相同。训练只替换sender policy优势的baseline，价值MSE仍按原世界回归，receiver原式保持。','',
        '若新轨迹激活裁剪，不能以旧四臂零激活排除新条件中的裁剪；若未激活，入口可收窄至baseline优势数值及后续反馈。结果下降也不单独证明梯度方差机制；没有执行固定政策直接梯度测量、片段拼接或新增语言语法检验。','',
        '完整四来源差保留，不按结果筛选。old任务变化与封存结果并列，避免把一般学习损伤说成专门的组合能力；n4和已观察来源只构成开发证据。']
    for figure in r['figures']:lines+=['',f'![基线对应关系]({figure["png"]})']
    return '\n'.join(lines)+'\n'


def self_test():
    with tempfile.TemporaryDirectory() as folder:
        f=Path(folder);groups,missing=inventory(f/'new',f/'source',f/'reset',f/'scaled',f/'cal',SEEDS,PARTITIONS,TIMES)
        assert len(groups)==12 and len(missing)==103
    assert CONTRASTS=={'state_correspondence':{'matched':1,'shuffled':-1}}
    assert v13.auc([0,1,4],[0,1,0])==.5
    norm=np.array([0.,1.,3.],np.float32);coeff=np.minimum(np.float32(1),np.float32(2)/(norm+np.float32(1e-6)));assert list(coeff<1)==[False,False,True]
    assert phases(2100)==('all','entropy_on') and phases(2101)==('all','entropy_off')
    return dict(passed=True,checks=['All12 new plus12 matched and source/calibration gate before effects','Matched minus shuffled fixed primary','Irregular-time normalized AUC','Float32 clip and entropy phase boundaries'])


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('batch',nargs='?',type=Path,default=ROOT/'results/baseline_001');ap.add_argument('--smoke',action='store_true');ap.add_argument('--self-test',action='store_true');args=ap.parse_args();tests=self_test()
    if args.self_test:print(json.dumps(tests));return
    batch=args.batch.resolve();formal=not args.smoke
    if formal:source,reset_source,scaled_source,calibration,seeds,parts,times=SOURCE,RESET_SOURCE,SCALED_SOURCE,CALIBRATION,SEEDS,PARTITIONS,TIMES
    else:
        assert batch.name.startswith('smoke_');inv=read(batch/'invocation.json');assert not inv['formal'];cfg=inv['args']
        source=Path(cfg['source_batch']);reset_source=Path(cfg['reset_batch']);scaled_source=Path(cfg['scaled_batch']);calibration=Path(cfg['calibration']);seeds=tuple(cfg['seeds']);parts=tuple(cfg['partitions'])
        times=tuple(sorted({0,cfg['social_updates'],*(t for t in TIMES if t<cfg['social_updates'])}))
    groups,missing=inventory(batch,source,reset_source,scaled_source,calibration,seeds,parts,times)
    if missing:print(json.dumps(dict(status='pending',missing=missing,result_content_read=False)));return
    done=read(batch/'training_complete.json');assert done['status']=='complete' and done['new_private_runs']==0
    assert done['new_social_runs']==len(groups) if formal else done['new_social_runs']>=len(groups)
    sr=read(batch/'source_receipt.json')
    for receipt in (sr['retained'],sr['reset'],sr['scaled']):
        for path,digest in receipt['files'].items():assert sha(path)==digest,path
    cr=read(batch/'calibration_receipt.json');assert sha(calibration/'calibration_complete.json')==cr['sha256']
    for path,digest in cr['files'].items():assert sha(calibration/path)==digest,path
    runs=[];calibrations=[];states=[]
    for seed,p,folders in groups:
        cal=previous.verify_calibration(calibration/f's{seed}_p{p}',seed,p,formal);calibrations.append(cal)
        cells={arm:load_social(path,source,seed,p,arm,times,formal) for arm,path in folders.items()}
        assert len({r['world_sequence_sha256'] for r in cells.values()})==1
        assert len({r['curve'][0]['scores']['normal']['world_sha256'] for r in cells.values()})==1
        states.append(verify_pair(seed,p,cells,cal,reset_source,times));runs.extend(cells.values())
    result=dict(status='complete' if formal else 'development_smoke',seeds=list(seeds),partitions=list(parts),times=list(times),
        new_social_runs=len(groups),reused_social_runs=len(groups),total_social_runs=2*len(groups),new_private_runs=0,
        runs=runs,summary=summarize(runs,seeds,parts,times),diagnostics=diagnostics(runs,times),calibration=calibrations,state_checks=states,
        source_batch=str(source),reset_source_batch=str(reset_source),scaled_source_batch=str(scaled_source),calibration_batch=str(calibration),
        analysis_sha256=sha(__file__),helper_sha256=sha(Path(v13.__file__)),previous_analysis_helper_sha256=sha(Path(previous.__file__)),
        source_receipt_sha256=sha(batch/'source_receipt.json'),calibration_receipt_sha256=sha(batch/'calibration_receipt.json'),self_tests=tests,created_utc=datetime.now(timezone.utc).isoformat(),
        primary='matched minus shuffled sealed N at2400',audit=dict(complete_before_effect_analysis=True,all_raw_endpoints_and_protocols_recomputed=True,raw_calibration_recomputed=True,
        base_tensors_equal_matched=True,branch_scales_and_interface_frozen=True,raw_effective_h_semantics_verified=True,external_worlds_paired=True,no_model_forward=True),
        reused_references_are_previously_seen_development_evidence=True)
    if formal:
        output=batch/'figures';output.mkdir(exist_ok=True);result['figures']=figures(result,output);(batch/'基线与场景对应_分析草稿.md').write_text(report(result))
    name='baseline_analysis.json' if formal else 'analysis_smoke.json';(batch/name).write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'],new_social_runs=len(groups),output=str(batch/name),audit=result['audit']),ensure_ascii=False))

if __name__=='__main__':main()

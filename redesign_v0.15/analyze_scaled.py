"""Complete-batch three-interface analysis from saved artifacts; no new inference."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib,itertools,json
from pathlib import Path
import sys,tempfile
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'redesign_v0.14'))
import analyze_reset as previous
v13=previous.v13
import numpy as np
import torch
SEEDS=previous.SEEDS;PARTITIONS=previous.PARTITIONS;TIMES=previous.TIMES;MODES=previous.MODES;GROUPS=previous.GROUPS
ARMS=('retained','reset','reset_scaled')
CONTRASTS={'remaining':('retained','reset_scaled'),'repair':('reset_scaled','reset')}
LABELS={'retained':'Retained interface','reset':'Reset interface','reset_scaled':'Reset + fixed scale'}
COLORS={'retained':'#a47746','reset':'#566fab','reset_scaled':'#3f9274'}
PREFIXES=previous.PREFIXES
SOURCE=ROOT.parent/'redesign_v0.13/results/spatial_001'
RESET_SOURCE=ROOT.parent/'redesign_v0.14/results/reset_001'
CALIBRATION=ROOT/'calibration_001'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())


def inventory(batch,source,reset_source,calibration,seeds,partitions,times):
    groups=[];missing=[]
    names=['config.json','result.json','curve.json','training.jsonl','initial.pt','final.pt',
        *(f'final_{m}.npz' for m in MODES),*(f'protocol_{t:04d}.json' for t in times),
        *(f'protocol_{t:04d}_d{d}.npz' for t in times for d in (0,1))]
    for seed,p in itertools.product(seeds,partitions):
        folders=dict(retained=source/f'social_s{seed}_p{p}_control',reset=reset_source/f'social_s{seed}_p{p}_reset',reset_scaled=batch/f'social_s{seed}_p{p}_reset_scaled')
        groups.append((seed,p,folders))
        for folder in folders.values():
            absent=[n for n in names if not (folder/n).is_file()]
            if absent:missing.append(dict(path=str(folder),files=absent))
        folder=calibration/f's{seed}_p{p}';absent=[n for n in ('calibration.json','person0.npz','person1.npz') if not (folder/n).is_file()]
        if absent:missing.append(dict(path=str(folder),files=absent))
    for folder,name in ((batch,'training_complete.json'),(source,'training_complete.json'),(reset_source,'training_complete.json'),
                        (batch,'source_receipt.json'),(batch,'calibration_receipt.json'),(calibration,'calibration_complete.json')):
        if not (folder/name).is_file():missing.append(dict(path=str(folder),files=[name]))
    return groups,missing


def load_social(path,source,seed,p,label,times,formal):
    cfg,curve,result=[read(path/name) for name in ('config.json','curve.json','result.json')]
    actual_arm={'retained':'control','reset':'reset','reset_scaled':'reset_scaled'}[label]
    for obj in (cfg,result):assert (obj['seed'],obj['partition'],obj['arm'],obj['updates'])==(seed,p,actual_arm,times[-1])
    assert cfg['checkpoints']==list(times)==[r['update'] for r in curve]
    assert cfg['training_pool']==v13.private.partition_maps(p)['old'].tolist()
    assert cfg['sealed_never_trained'] and result['sealed_never_trained'] and result['frozen_modules_verified']
    assert (cfg['rng_world_namespace'],cfg['rng_policy_namespace'],cfg['training_policy_stream'],cfg['evaluation_policy_stream'])==(13014,11011,11,91)
    assert (cfg['learning_rate'],cfg['entropy_coefficient'],cfg['entropy_off_after'],cfg['role_loss_weights'])==(.0007,.02,2100,{'sender':.5,'receiver':.5})
    if formal:assert cfg['batch']==512 and cfg['eval_n']==9600
    for name,digest in cfg['source_hashes'].items():assert sha(name)==digest,name
    assert cfg['source_checkpoint']['sha256']==sha(cfg['source_checkpoint']['path'])
    if label=='reset':
        for key in ('private_initial','retained_social_initial'):
            ref=cfg['source_inputs'][key];assert sha(ref['path'])==ref['sha256']
        assert cfg['source_inputs']['reset_prefixes']==list(PREFIXES)
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


def verify_calibration(folder,seed,p,formal):
    cfg=read(folder/'calibration.json');assert cfg['status']=='complete' and (cfg['seed'],cfg['partition'])==(seed,p)
    assert cfg['training_updates']==0 and cfg['no_test_or_held_maps'] and cfg['chunk_size']==512
    assert cfg['device']=='cpu' and cfg['torch_threads']==1 and not cfg['grad_enabled'] and cfg['model_training']
    for path,digest in cfg['source_hashes'].items():assert sha(path)==digest,path
    old=v13.private.partition_maps(p)['old'];pools=cfg['train_photo_ids']
    assert cfg['old_map_ids']==old.tolist()
    if formal:assert [len(x) for x in pools]==[22,22]
    tuples=np.asarray(list(itertools.product(old,pools[0],pools[1])),dtype=np.int64)
    values=[]
    for who,row in enumerate(cfg['persons']):
        assert who==row['person'];path=folder/f'person{who}.npz';assert sha(path)==row['raw_sha256']
        z=v13.load(path)
        assert np.array_equal(z['world_id'],np.arange(len(tuples)))
        assert np.array_equal(z['map_ids'],tuples[:,0]) and np.array_equal(z['photo_ids'],tuples[:,1:])
        assert np.array_equal(z['positions'],v13.private.MAPS[tuples[:,0]])
        assert np.array_equal(z['photo_pair_ids'],np.tile(np.arange(len(pools[0])*len(pools[1])),18))
        means={a:float(np.linalg.norm(z[a+'_h'].astype(np.float64),axis=1).mean()) for a in ('retained','reset')}
        assert all(np.isfinite(z[a+'_h']).all() and z[a+'_h'].shape==(len(tuples),96) for a in ('retained','reset'))
        ratio=means['retained']/means['reset'];effective=float(np.float32(ratio))
        for k,x in {'retained_mean_l2':means['retained'],'reset_mean_l2':means['reset'],'alpha_float64':ratio}.items():assert abs(x-row[k])<=1e-12,k
        assert effective==row['alpha_float32']
        measured=float(np.linalg.norm((z['reset_h']*np.float32(effective)).astype(np.float64),axis=1).mean())
        assert abs(measured-row['scaled_mean_l2'])<=1e-12
        values.append(dict(person=who,worlds=len(tuples),**{k:row[k] for k in ('retained_mean_l2','reset_mean_l2','alpha_float64','alpha_float32','scaled_mean_l2')}))
    return dict(seed=seed,partition=p,path=str(folder),persons=values,source_sha256=sha(folder/'calibration.json'),raw_means_ratio_and_train_only_support_recomputed=True)


def verify_triplet(source,seed,p,runs,calibration):
    first=previous.verify_states(source,seed,p,runs['retained'],runs['reset'])
    reset=torch.load(Path(runs['reset']['path'])/'initial.pt',weights_only=True)
    folder=Path(runs['reset_scaled']['path']);initial=torch.load(folder/'initial.pt',weights_only=True);final=torch.load(folder/'final.pt',weights_only=True)
    cfg=read(folder/'config.json');assert cfg['scales_float32']==[x['alpha_float32'] for x in calibration['persons']]
    assert runs['reset']['parameter_partition']==runs['reset_scaled']['parameter_partition']
    for source_key in ('reset_initial','retained_social_initial','calibration'):
        ref=cfg['source_inputs'][source_key];assert sha(ref['path'])==ref['sha256']
    checks=[]
    for who in (0,1):
        assert set(initial[who])==set(reset[who])|{'visual_scale'}==set(final[who])
        for k,v in reset[who].items():assert torch.equal(v,initial[who][k]),k
        alpha=initial[who]['visual_scale'];assert alpha.ndim==0 and alpha.dtype==torch.float32 and float(alpha)==cfg['scales_float32'][who]
        assert torch.equal(alpha,final[who]['visual_scale'])
        for k,v in initial[who].items():
            if k.startswith(PREFIXES+('project.',)) or k in ('input_transform','visual_scale'):assert torch.equal(v,final[who][k]),k
        checks.append(dict(person=who,alpha=float(alpha),base_tensors_equal_reset=True,scale_and_interface_frozen=True))
    return dict(seed=seed,partition=p,retained_reset=first,scaled=checks)


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
    for name,(left,right) in CONTRASTS.items():
        differences={k:[lookup[s,left]['metrics'][k]-lookup[s,right]['metrics'][k] for s in seeds] for k in cells[0]['metrics']}
        contrasts[name]=dict(left=left,right=right,seed_differences=differences,
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
    fig.suptitle('Four inherited source seeds; only scaled runs are new; thin lines average partitions and directions within seed',fontsize=9)
    save(fig,'01_three_interface_learning')
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


def report(r):
    s=r['summary'];lines=['# 固定视觉幅度校正：分析草稿','',
        '新增12个reset_scaled社会运行完成后统一分析；retained复用v0.13全部12个control，reset复用v0.14全部12个reset。每个既有来源种子内平均3分区×2方向，独立单位仍为4个来源；来源和图片已被观察，不是独立确认。','',
        '唯一主要终点是2400步封存6图自然N的remaining差：retained−reset_scaled。repair差reset_scaled−reset为辅助，两个方向不因结果改变。AUC为预定8个检查点梯形面积除以2400。','',
        '| 条件 | old N | added N | sealed N（主） | sealed AUC |','| --- | ---: | ---: | ---: | ---: |']
    for arm,a in s['aggregate'].items():lines.append(f'| {LABELS[arm]} | '+' | '.join(f'{100*a["metrics"][k]:.2f}%' for k in ('normal.old.N','normal.added.N','normal.sealed.N','auc.sealed.N'))+' |')
    lines+=['','| 比较 | 指标 | 四来源差（百分点） | 平均差 |','| --- | --- | --- | ---: |']
    for name,c in s['contrasts'].items():
        for k in ('normal.sealed.N','auc.sealed.N','normal.old.N','auc.old.N','normal.added.N','auc.added.N'):
            lines.append(f'| {name}: {c["left"]}−{c["right"]} | {k} | '+ '、'.join(f'{100*v:+.4f}' for v in c['seed_differences'][k])+f' | {100*c["mean_differences"][k]:+.4f} |')
    alpha=[x['alpha_float32'] for c in r['calibration'] for x in c['persons']]
    lines+=['',f'校准24个人倍率范围{min(alpha):.6f}–{max(alpha):.6f}；每人只使用old18×22×22个训练照片世界。倍率为固定meanL2比量化到float32，未扫描、未训练、未使用test或未训练地图；44张train特征才进入校准投影。原始h/世界及倍率已重算。','',
        '| 条件 | common30原生联合MAP | 原生Q | 原生G | 贪心N | 原生联合条件熵 | 贪心完整码数 |','| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for arm,a in s['aggregate'].items():
        m=a['metrics'];pre='protocol.common30.'
        lines.append(f'| {LABELS[arm]} | '+' | '.join(f'{100*m[pre+k]:.2f}%' for k in ('native_joint_oracle','Q','G','N'))+f' | {m[pre+"joint_entropy"]:.6f} | {m[pre+"greedy_used_codes"]:.2f} |')
    lines+=['','社会N使用同一9600世界；协议用固定16照片对，二者不直接相减。Q是双方原生随机策略的解析期望，G为原生发送加贪心接收，N改为逐符号贪心发送。每臂common30原生oracle只对自身发送政策有效，不能套到另一臂或贪心政策，也不能由子组oracle拼接。','',
        '| 条件 | 终点模式 | 全30图J | sealed J | 全30图收益 |','| --- | --- | ---: | ---: | ---: |']
    for arm,a in s['aggregate'].items():
        m=a['metrics']
        for mode in MODES:lines.append(f'| {LABELS[arm]} | {mode} | {100*m[f"intervention.{mode}.all.joint"]:.2f}% | {100*m[f"intervention.{mode}.sealed.joint"]:.2f}% | {m[f"intervention.{mode}.all.reward"]:.6f} |')
    lines+=['','stochastic是固定流抽样轨迹，不能与协议解析Q混称；通道干预是功能依赖诊断，非语法证据。','',
        'reset_scaled只新增persistent正标量buffer，所有基态张量与reset逐位一致，通信初值/世界/预算相同。倍率在observe输出处作用于发送策略和sender价值，协议也经过同一路径；接收者不直接看到h。社会阶段视觉接口及倍率冻结。','',
        '校正匹配一个训练分布上的平均L2幅度，没有匹配逐世界范数、坐标方向、协方差或非线性饱和。正标量在实数表示下不增加每场景信息，但改变发送策略/critic的输入幅度、熵、梯度和有限预算学习。若修复出现，说明本项固定幅度校正可以恢复部分表现；若余差保留，说明此单一校正不足，均不能唯一归因于空间绑定或语言能力。','',
        '旧图学习和未训练表达须一起看；不将一般学习修复称为独有的组合性改变。没有等效界或独立确认，均值接近不能宣称等价。本轮未新增片段拼接、替代倍率或参数扫描。']
    for f in r['figures']:lines+=['',f'![固定幅度与共同通信]({f["png"]})']
    return '\n'.join(lines)+'\n'


def self_test():
    with tempfile.TemporaryDirectory() as folder:
        f=Path(folder);items,missing=inventory(f/'new',f/'retained',f/'reset',f/'cal',SEEDS,PARTITIONS,TIMES)
        assert len(items)==12 and len(missing)==54
    assert v13.auc([0,1,4],[0,1,0])==.5
    assert CONTRASTS=={'remaining':('retained','reset_scaled'),'repair':('reset_scaled','reset')}
    return dict(passed=True,checks=['All36 source/new runs and calibration before effect analysis','Fixed remaining and repair direction','Normalized irregular-time AUC'])


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('batch',nargs='?',type=Path,default=ROOT/'results/scaled_001');ap.add_argument('--smoke',action='store_true');ap.add_argument('--self-test',action='store_true');args=ap.parse_args()
    tests=self_test()
    if args.self_test:print(json.dumps(tests));return
    batch=args.batch.resolve();formal=not args.smoke
    if formal:source,reset_source,calibration,seeds,parts,times=SOURCE,RESET_SOURCE,CALIBRATION,SEEDS,PARTITIONS,TIMES
    else:
        assert batch.name.startswith('smoke_');inv=read(batch/'invocation.json');assert not inv['formal']
        cfg=inv['args'];source=Path(cfg['source_batch']);reset_source=Path(cfg['reset_batch']);calibration=Path(cfg['calibration']);seeds=tuple(cfg['seeds']);parts=tuple(cfg['partitions'])
        times=tuple(sorted({0,cfg['social_updates'],*(t for t in TIMES if t<cfg['social_updates'])}))
    groups,missing=inventory(batch,source,reset_source,calibration,seeds,parts,times)
    if missing:print(json.dumps(dict(status='pending',missing=missing,result_content_read=False)));return
    done=read(batch/'training_complete.json');assert done['status']=='complete' and done['new_private_runs']==0
    assert done['new_social_runs']==len(groups) if formal else done['new_social_runs']>=len(groups)
    sr=read(batch/'source_receipt.json')
    for receipt in (sr['retained'],sr['reset']):
        for path,digest in receipt['files'].items():assert sha(path)==digest,path
    cr=read(batch/'calibration_receipt.json');assert sha(calibration/'calibration_complete.json')==cr['sha256']
    for path,digest in cr['files'].items():assert sha(calibration/path)==digest,path
    runs=[];calibrations=[];states=[]
    for seed,p,folders in groups:
        c=verify_calibration(calibration/f's{seed}_p{p}',seed,p,formal);calibrations.append(c)
        triplet={arm:load_social(path,source,seed,p,arm,times,formal) for arm,path in folders.items()}
        assert len({r['world_sequence_sha256'] for r in triplet.values()})==1
        assert len({r['curve'][0]['scores']['normal']['world_sha256'] for r in triplet.values()})==1
        states.append(verify_triplet(source,seed,p,triplet,c));runs.extend(triplet.values())
    result=dict(status='complete' if formal else 'development_smoke',seeds=list(seeds),partitions=list(parts),times=list(times),
        new_social_runs=len(groups),retained_reference_runs=len(groups),reset_reference_runs=len(groups),new_private_runs=0,
        runs=runs,summary=summarize(runs,seeds,parts,times),calibration=calibrations,state_checks=states,source_batch=str(source),reset_source_batch=str(reset_source),calibration_batch=str(calibration),
        analysis_sha256=sha(__file__),helper_sha256=sha(Path(v13.__file__)),previous_analysis_helper_sha256=sha(Path(previous.__file__)),
        source_receipt_sha256=sha(batch/'source_receipt.json'),calibration_receipt_sha256=sha(batch/'calibration_receipt.json'),self_tests=tests,
        created_utc=datetime.now(timezone.utc).isoformat(),primary='retained minus reset_scaled sealed N at2400',secondary_repair='reset_scaled minus reset',
        audit=dict(complete_before_effect_analysis=True,all_raw_endpoints_and_protocols_recomputed=True,raw_calibration_recomputed=True,
            base_tensors_equal_reset=True,scale_and_interface_frozen=True,external_worlds_paired=True,no_model_forward=True),
        reused_references_are_previously_seen_development_evidence=True)
    if formal:
        out=batch/'figures';out.mkdir(exist_ok=True);result['figures']=figures(result,out);(batch/'固定幅度校正_分析草稿.md').write_text(report(result))
    name='scaled_analysis.json' if formal else 'analysis_smoke.json';(batch/name).write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'],new_social_runs=len(groups),output=str(batch/name),audit=result['audit']),ensure_ascii=False))


if __name__=='__main__':main()

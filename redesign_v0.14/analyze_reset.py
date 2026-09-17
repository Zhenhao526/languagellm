"""Complete-batch retained-versus-reset analysis, using only saved artifacts."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import sys
import tempfile

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'redesign_v0.9/.analysis_deps'))
sys.path.insert(0,str(ROOT.parent/'redesign_v0.13'))
import numpy as np
import torch
import analyze_spatial as v13

SEEDS=(31101,31102,31103,31104)
PARTITIONS=(1,2,3)
ARMS=('retained','reset')
TIMES=(0,100,300,600,1200,1800,2100,2400)
GROUPS=('old','added','sealed')
MODES=v13.MODES
PREFIXES=('memory.','slot_phi.')
SOURCE=ROOT.parent/'redesign_v0.13/results/spatial_001'
COLORS={'retained':'#a47746','reset':'#566fab'}
LABELS={'retained':'Retained interface','reset':'Reset interface'}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())


def inventory(batch,source,seeds,partitions,times):
    pairs=[];missing=[]
    names=['config.json','result.json','curve.json','training.jsonl','initial.pt','final.pt',
        *(f'final_{m}.npz' for m in MODES),*(f'protocol_{t:04d}.json' for t in times),
        *(f'protocol_{t:04d}_d{d}.npz' for t in times for d in (0,1))]
    for seed,p in itertools.product(seeds,partitions):
        retained=source/f'social_s{seed}_p{p}_control';reset=batch/f'social_s{seed}_p{p}_reset'
        pairs.append((seed,p,retained,reset))
        for folder in (retained,reset):
            absent=[n for n in names if not (folder/n).is_file()]
            if absent:missing.append(dict(path=str(folder),files=absent))
    for folder,name in ((batch,'training_complete.json'),(batch,'source_receipt.json'),(source,'training_complete.json')):
        if not (folder/name).is_file():missing.append(dict(path=str(folder),files=[name]))
    return pairs,missing


def load_social(path,source,seed,p,label,times,formal):
    cfg,curve,result=[read(path/name) for name in ('config.json','curve.json','result.json')]
    actual_arm='control' if label=='retained' else 'reset'
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


def verify_states(source,seed,p,retained,reset):
    r=torch.load(Path(retained['path'])/'initial.pt',weights_only=True)
    z=torch.load(Path(reset['path'])/'initial.pt',weights_only=True)
    original=torch.load(source/f'private_s{seed}_p{p}_control/initial.pt',weights_only=True)['agents']
    final=torch.load(Path(reset['path'])/'final.pt',weights_only=True)
    assert retained['parameter_partition']==reset['parameter_partition']
    persons=[]
    for who in (0,1):
        assert set(r[who])==set(z[who])==set(original[who])==set(final[who])
        changed=[];reset_keys=[]
        for key in r[who]:
            assert torch.equal(z[who][key],original[who][key]),key
            if key.startswith(PREFIXES):
                reset_keys.append(key)
                if not torch.equal(r[who][key],z[who][key]):changed.append(key)
                assert torch.equal(z[who][key],final[who][key]),key
            else:assert torch.equal(r[who][key],z[who][key]),key
        for key in ('input_transform','project.0.weight','project.0.bias'):assert torch.equal(z[who][key],final[who][key])
        persons.append(dict(person=who,reset_tensor_count=len(reset_keys),changed_tensor_count=len(changed),
            reset_parameters=sum(z[who][k].numel() for k in reset_keys),changed_parameters=sum(z[who][k].numel() for k in changed),
            changed_keys=changed))
    return dict(seed=seed,partition=p,reset_matches_private_initial=True,other_initial_parameters_identical=True,
        reset_interface_frozen_to_final=True,persons=persons)


def summarize(runs,seeds,partitions,times):
    cells=[]
    for seed,arm in itertools.product(seeds,ARMS):
        chosen=[r for r in runs if r['seed']==seed and r['arm']==arm]
        assert {r['partition'] for r in chosen}==set(partitions)
        curve=[]
        for t in times:
            points=[v13.point_metrics(next(x for x in r['curve'] if x['update']==t)) for r in chosen]
            curve.append(dict(update=t,metrics={key:float(np.mean([x[key] for x in points])) for key in points[0]}))
        metrics=dict(curve[-1]['metrics'])
        for group in GROUPS:metrics[f'auc.{group}.N']=v13.auc(times,[r['metrics'][f'normal.{group}.N'] for r in curve])
        for mode in MODES:
            for group in (*GROUPS,'all'):
                values=[r['final'][mode] if group=='all' else r['final'][mode]['map_groups'][group] for r in chosen]
                metrics[f'intervention.{mode}.{group}.joint']=float(np.mean([v['both_accuracy'] for v in values]))
                metrics[f'intervention.{mode}.{group}.reward']=float(np.mean([v['mean_reward'] for v in values]))
        cells.append(dict(seed=seed,arm=arm,partitions_averaged=len(partitions),directions_per_partition=2,curve=curve,metrics=metrics))
    aggregate={}
    for arm in ARMS:
        selected=[c for c in cells if c['arm']==arm]
        aggregate[arm]=dict(metrics={k:float(np.mean([c['metrics'][k] for c in selected])) for k in cells[0]['metrics']},
            curve=[dict(update=t,metrics={k:float(np.mean([next(x for x in c['curve'] if x['update']==t)['metrics'][k] for c in selected])) for k in cells[0]['curve'][0]['metrics']}) for t in times])
    lookup={(r['seed'],r['arm']):r for r in cells}
    differences={k:[lookup[s,'retained']['metrics'][k]-lookup[s,'reset']['metrics'][k] for s in seeds] for k in cells[0]['metrics']}
    return dict(seed_cells=cells,aggregate=aggregate,contrast='retained_minus_reset',seed_differences=differences,
        mean_differences={k:float(np.mean(v)) for k,v in differences.items()},difference_ranges={k:[min(v),max(v)] for k,v in differences.items()})


def figures(result,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
    summary=result['summary'];seeds=result['seeds'];times=result['times'];paths=[]
    def save(fig,name):
        fig.savefig(out/f'{name}.png');fig.savefig(out/f'{name}.pdf');plt.close(fig)
        paths.append(dict(png=str(out/f'{name}.png'),pdf=str(out/f'{name}.pdf')))
    fig,axes=plt.subplots(1,3,figsize=(12,4.25),sharey=True,constrained_layout=True)
    for ax,group,title in zip(axes,GROUPS,('Old 18: trained','Added 6: never trained','Sealed 6: primary never-trained set')):
        for arm in ARMS:
            for seed in seeds:
                cell=next(c for c in summary['seed_cells'] if c['seed']==seed and c['arm']==arm)
                ax.plot(times,[100*p['metrics'][f'normal.{group}.N'] for p in cell['curve']],color=COLORS[arm],alpha=.25,lw=.8)
            ax.plot(times,[100*p['metrics'][f'normal.{group}.N'] for p in summary['aggregate'][arm]['curve']],color=COLORS[arm],lw=2.1,label=LABELS[arm])
        ax.set(title=title,xlabel='Social learning updates',ylim=(-2,102));ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Greedy sender + greedy receiver: N (%)');axes[1].legend(frameon=False,fontsize=8)
    fig.suptitle('Four inherited source seeds; retained control is reused; thin lines are within-seed means over partitions/directions',fontsize=9)
    save(fig,'01_retained_reset_learning')
    fig,axes=plt.subplots(1,3,figsize=(12,4.25),constrained_layout=True)
    panels=[('native_joint_oracle','G','Native sender: oracle and G','Joint success on common 30 maps (%)',100),
        ('N',None,'Greedy sender + greedy receiver','Natural N on common 30 maps (%)',100),
        ('joint_entropy',None,'Native sender ambiguity','H(food, water | message), nats',1)]
    for ax,(key,other,title,ylabel,scale) in zip(axes,panels):
        for arm in ARMS:
            for k in (key,other) if other else (key,):
                style='--' if k=='native_joint_oracle' else '-'
                for seed in seeds:
                    cell=next(c for c in summary['seed_cells'] if c['seed']==seed and c['arm']==arm)
                    ax.plot(times,[scale*p['metrics'][f'protocol.common30.{k}'] for p in cell['curve']],color=COLORS[arm],ls=style,alpha=.16,lw=.7)
                label=LABELS[arm]+(': oracle' if k=='native_joint_oracle' else ': G' if k=='G' else '')
                ax.plot(times,[scale*p['metrics'][f'protocol.common30.{k}'] for p in summary['aggregate'][arm]['curve']],color=COLORS[arm],ls=style,lw=2,label=label)
        ax.set(title=title,ylabel=ylabel,xlabel='Social learning updates');ax.grid(axis='y',alpha=.2)
        ax.tick_params(axis='y',labelleft=True)
        if scale==100:
            ax.set_ylim(-2,102)
            ax.set_yticks(np.arange(0,101,20))
    axes[0].legend(frameon=False,fontsize=7);axes[1].legend(frameon=False,fontsize=8)
    fig.suptitle('Fixed 16 photo pairs and exact 49-code distributions; each native oracle belongs to its own sender policy',fontsize=9)
    save(fig,'02_common_support_and_natural_use')
    return paths


def report(r):
    s=r['summary'];lines=['# 私人空间接口保留与重置：分析草稿','',
        '本轮复用v0.13全部12个control社会运行，新增同源12个reset运行，未重跑私人准备或正式control。4个既有来源种子各自平均3分区、2方向；不是新的独立确认。主要终点固定2400步封存6图自然N；差值统一为retained−reset，正值表示保留接口占优。','',
        'reset只将memory和slot_phi恢复同源私人初值，并在社会阶段固定。DINO、200×64基础资源后果练习的投影、输入变换、通信参数初值及社会训练预算/外生世界均匹配。memory在这里是从零状态单步调用的视觉编码器，不能称为删除时间记忆。','',
        '| 条件 | old N | added N | sealed N（主） | sealed归一化AUC |','| --- | ---: | ---: | ---: | ---: |']
    for arm,a in s['aggregate'].items():
        m=a['metrics'];lines.append(f'| {LABELS[arm]} | '+' | '.join(f'{100*m[k]:.2f}%' for k in ('normal.old.N','normal.added.N','normal.sealed.N','auc.sealed.N'))+' |')
    lines+=['','| retained−reset | 四个来源种子差，百分点 | 均值 |','| --- | --- | ---: |']
    for k in ('normal.sealed.N','auc.sealed.N','normal.old.N','auc.old.N','normal.added.N','auc.added.N'):
        lines.append(f'| {k} | '+'、'.join(f'{100*v:+.4f}' for v in s['seed_differences'][k])+f" | {100*s['mean_differences'][k]:+.4f} |")
    lines+=['','社会指标使用与v0.13相同的9600世界和已使用过的test照片；normal是逐符号贪心发送、贪心接收。同源世界、需求、照片、菜单和随机流配对，实际消息/行动可改变。AUC按预定八检查点的梯形面积除以2400，不新增观测点或挑选最佳阶段。','',
        '| 条件 | common30原生联合MAP | 原生Q | 原生G | 贪心N | 原生联合条件熵 | 贪心使用码数 |','| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for arm,a in s['aggregate'].items():
        m=a['metrics'];pre='protocol.common30.'
        lines.append(f'| {LABELS[arm]} | '+' | '.join(f'{100*m[pre+k]:.2f}%' for k in ('native_joint_oracle','Q','G','N'))+f" | {m[pre+'joint_entropy']:.6f} | {m[pre+'greedy_used_codes']:.2f} |")
    lines+=['','协议使用固定16照片对×30图；Q为双方原生概率期望，G为原生发送加贪心接收，N改变成逐符号贪心发送。common30参照按每个条件自身消息分布求一张共同表，不可跨臂借用、不可拼接子组oracle。N不能直接与原生oracle求差，9600世界指标与固定16照片指标也不直接相减。','',
        '| 条件 | 终点模式 | 全30图联合成功 | 封存图联合成功 | 全30图收益 |','| --- | --- | ---: | ---: | ---: |']
    for arm,a in s['aggregate'].items():
        m=a['metrics']
        for mode in MODES:lines.append(f"| {LABELS[arm]} | {mode} | {100*m[f'intervention.{mode}.all.joint']:.2f}% | {100*m[f'intervention.{mode}.sealed.joint']:.2f}% | {m[f'intervention.{mode}.all.reward']:.6f} |")
    lines+=['','stochastic为固定流下的实际抽样轨迹，另与解析期望Q区分。shuffle/blank/erase_memory检查当前策略对通道或场景的依赖，不提供自然语法证据。','',
        '已从全部原始终点动作、菜单、正确位置和反馈重算成绩；所有检查点协议由保存logits重算。同源初值核验确认仅覆盖两个允许前缀，重置后的全状态等于私人初态，其余通信参数与retained逐位一致；社会结束时接口保持冻结。分析未增加模型推理或训练。','',
        '同一私人终点头接重置接口的无训练重放属于固定读出兼容性诊断，不代表reset重新训练后的能力。私人头不进入社会通信。参数或h尺度差不能当作关系能力、信息丢失或中介机制的证明。','',
        '本比较测量准备过的冻结接口的综合迁移收益，可能同时涉及表示内容、尺度、激活饱和和新通信头的可读性。若旧图学习也下降，首先是一般任务学习差异，不能仅用封存差命名为组合机制。正结果不证明某项非语言能力普遍必要或充分；接近的结果也不是等价性检验。来源和照片已被观察，控制参考复用，不夸大为独立确认。']
    for f in r['figures']:lines+=['',f'![保留与重置接口]({f["png"]})']
    return '\n'.join(lines)+'\n'


def self_test():
    assert v13.auc([0,1,4],[0,1,0])==.5
    with tempfile.TemporaryDirectory() as folder:
        p=Path(folder);pairs,missing=inventory(p/'new',p/'old',SEEDS,PARTITIONS,TIMES)
        assert len(pairs)==12 and len(missing)==27
    return dict(passed=True,checks=['Normalized irregular-time AUC','Twelve new and twelve retained results plus completion/source receipts required','Primary contrast explicitly retained minus reset'])


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('batch',nargs='?',type=Path,default=ROOT/'results/reset_001')
    parser.add_argument('--self-test',action='store_true');parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    tests=self_test()
    if args.self_test:print(json.dumps(tests));return
    batch=args.batch.resolve();formal=not args.smoke
    if formal:source,seeds,parts,times=SOURCE,SEEDS,PARTITIONS,TIMES
    else:
        assert batch.name.startswith('smoke_');inv=read(batch/'invocation.json');assert not inv['formal']
        cfg=inv['args'];source=Path(cfg['source_batch']);seeds=tuple(cfg['seeds']);parts=tuple(cfg['partitions'])
        times=tuple(sorted({0,cfg['social_updates'],*(t for t in TIMES if t<cfg['social_updates'])}))
    pairs,missing=inventory(batch,source,seeds,parts,times)
    if missing:print(json.dumps(dict(status='pending',missing=missing,result_content_read=False)));return
    done=read(batch/'training_complete.json');assert done['status']=='complete' and done['new_private_runs']==0
    assert done['new_social_runs']==len(pairs) if formal else done['new_social_runs']>=len(pairs)
    receipt=read(batch/'source_receipt.json')
    for path,digest in receipt['files'].items():assert sha(path)==digest,path
    runs=[];state_checks=[]
    for seed,p,r,z in pairs:
        retained=load_social(r,source,seed,p,'retained',times,formal);reset=load_social(z,source,seed,p,'reset',times,formal)
        assert retained['world_sequence_sha256']==reset['world_sequence_sha256']
        assert retained['curve'][0]['scores']['normal']['world_sha256']==reset['curve'][0]['scores']['normal']['world_sha256']
        state_checks.append(verify_states(source,seed,p,retained,reset));runs.extend((retained,reset))
    result=dict(status='complete' if formal else 'development_smoke',seeds=list(seeds),partitions=list(parts),times=list(times),
        new_social_runs=len(pairs),retained_reference_runs=len(pairs),new_private_runs=0,runs=runs,summary=summarize(runs,seeds,parts,times),
        state_checks=state_checks,source_batch=str(source),source_receipt_sha256=sha(batch/'source_receipt.json'),analysis_sha256=sha(__file__),
        helper_sha256=sha(Path(v13.__file__)),self_tests=tests,created_utc=datetime.now(timezone.utc).isoformat(),
        audit=dict(all_new_runs_complete_before_analysis=True,all_raw_endpoints_recomputed=True,all_saved_protocols_recomputed=True,
            same_external_worlds=True,reset_only_allowed_interface=True,interface_frozen=True,no_model_forward=True),
        primary='retained-minus-reset sealed N at2400; four inherited source seeds; partitions and directions nested',
        reused_control_is_previously_seen_development_evidence=True)
    if formal:
        out=batch/'figures';out.mkdir(exist_ok=True);result['figures']=figures(result,out)
        (batch/'保留与重置接口_分析草稿.md').write_text(report(result))
    name='reset_analysis.json' if formal else 'analysis_smoke.json'
    (batch/name).write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'],new_social_runs=len(pairs),output=str(batch/name),audit=result['audit']),ensure_ascii=False))


if __name__=='__main__':main()

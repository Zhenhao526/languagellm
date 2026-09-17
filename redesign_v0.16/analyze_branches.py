"""Complete-batch three-interface analysis from saved artifacts; no new inference."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib,itertools,json
from pathlib import Path
import sys,tempfile
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'redesign_v0.15'))
import analyze_scaled as previous
v13=previous.v13
import numpy as np
import torch
SEEDS=previous.SEEDS;PARTITIONS=previous.PARTITIONS;TIMES=previous.TIMES;MODES=previous.MODES;GROUPS=previous.GROUPS
ARMS=('neither','policy_only_scale','value_only_scale','both')
CONTRASTS={
 'route_difference':{'policy_only_scale':1,'value_only_scale':-1},
 'policy_given_value0':{'policy_only_scale':1,'neither':-1},
 'policy_given_value1':{'both':1,'value_only_scale':-1},
 'value_given_policy0':{'value_only_scale':1,'neither':-1},
 'value_given_policy1':{'both':1,'policy_only_scale':-1},
 'interaction':{'both':1,'policy_only_scale':-1,'value_only_scale':-1,'neither':1}}
LABELS={'neither':'Neither (00)','policy_only_scale':'Policy only (10)','value_only_scale':'Value only (01)','both':'Both (11)'}
COLORS={'neither':'#566fab','policy_only_scale':'#bd7444','value_only_scale':'#9b67a1','both':'#3f9274'}
PREFIXES=previous.PREFIXES
SOURCE=ROOT.parent/'redesign_v0.13/results/spatial_001'
RESET_SOURCE=ROOT.parent/'redesign_v0.14/results/reset_001'
SCALED_SOURCE=ROOT.parent/'redesign_v0.15/results/scaled_001'
CALIBRATION=ROOT.parent/'redesign_v0.15/calibration_001'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())


def load_social(path,source,seed,p,label,times,formal):
    cfg,curve,result=[read(path/name) for name in ('config.json','curve.json','result.json')]
    actual_arm={'neither':'reset','policy_only_scale':'policy_only_scale','value_only_scale':'value_only_scale','both':'reset_scaled'}[label]
    for obj in (cfg,result):assert (obj['seed'],obj['partition'],obj['arm'],obj['updates'])==(seed,p,actual_arm,times[-1])
    assert cfg['checkpoints']==list(times)==[r['update'] for r in curve]
    assert cfg['training_pool']==v13.private.partition_maps(p)['old'].tolist()
    assert cfg['sealed_never_trained'] and result['sealed_never_trained'] and result['frozen_modules_verified']
    assert (cfg['rng_world_namespace'],cfg['rng_policy_namespace'],cfg['training_policy_stream'],cfg['evaluation_policy_stream'])==(13014,11011,11,91)
    assert (cfg['learning_rate'],cfg['entropy_coefficient'],cfg['entropy_off_after'],cfg['role_loss_weights'])==(.0007,.02,2100,{'sender':.5,'receiver':.5})
    if formal:assert cfg['batch']==512 and cfg['eval_n']==9600
    for name,digest in cfg['source_hashes'].items():assert sha(name)==digest,name
    assert cfg['source_checkpoint']['sha256']==sha(cfg['source_checkpoint']['path'])
    if label=='neither':
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
    fig.suptitle('Four inherited source seeds; only single-branch runs are new; thin lines average partitions and directions within seed',fontsize=9)
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


def inventory(batch,source,reset_source,scaled_source,calibration,seeds,partitions,times):
    items=[];missing=[]
    names=['config.json','result.json','curve.json','training.jsonl','initial.pt','final.pt',
        *(f'final_{m}.npz' for m in MODES),*(f'protocol_{t:04d}.json' for t in times),*(f'protocol_{t:04d}_d{d}.npz' for t in times for d in (0,1))]
    for seed,p in itertools.product(seeds,partitions):
        folders=dict(neither=reset_source/f'social_s{seed}_p{p}_reset',both=scaled_source/f'social_s{seed}_p{p}_reset_scaled',
            **{a:batch/f'social_s{seed}_p{p}_{a}' for a in ('policy_only_scale','value_only_scale')})
        items.append((seed,p,folders))
        for folder in folders.values():
            absent=[n for n in names if not (folder/n).is_file()]
            if absent:missing.append(dict(path=str(folder),files=absent))
        folder=calibration/f's{seed}_p{p}';absent=[n for n in ('calibration.json','person0.npz','person1.npz') if not (folder/n).is_file()]
        if absent:missing.append(dict(path=str(folder),files=absent))
    for folder,name in ((batch,'training_complete.json'),(source,'training_complete.json'),(reset_source,'training_complete.json'),(scaled_source,'training_complete.json'),
        (batch,'source_receipt.json'),(batch,'calibration_receipt.json'),(calibration,'calibration_complete.json')):
        if not (folder/name).is_file():missing.append(dict(path=str(folder),files=[name]))
    return items,missing


def verify_cells(seed,p,runs,calibration,times):
    base=torch.load(Path(runs['neither']['path'])/'initial.pt',weights_only=True)
    whole=torch.load(Path(runs['both']['path'])/'initial.pt',weights_only=True)
    alphas=[x['alpha_float32'] for x in calibration['persons']]
    for who in (0,1):
        assert set(whole[who])==set(base[who])|{'visual_scale'}
        assert float(whole[who]['visual_scale'])==alphas[who]
        assert all(torch.equal(v,whole[who][k]) for k,v in base[who].items())
    records=[]
    for arm,factors in (('policy_only_scale',(1,0)),('value_only_scale',(0,1))):
        run=runs[arm];folder=Path(run['path']);cfg=read(folder/'config.json')
        assert cfg['factors']==list(factors) and cfg['calibration_scales_float32']==alphas
        assert cfg['parameter_partition']==runs['neither']['parameter_partition']
        for key in ('reset_initial','retained_social_initial','calibration','scaled_initial'):
            ref=cfg['source_inputs'][key];assert sha(ref['path'])==ref['sha256']
        states=[torch.load(folder/name,weights_only=True) for name in ('initial.pt','final.pt')]
        for who in (0,1):
            initial,final=states[0][who],states[1][who]
            assert set(initial)==set(base[who])|{'send_context.h_scale','send_value.h_scale'}==set(final)
            assert all(torch.equal(v,initial[k]) for k,v in base[who].items())
            expected={key:alphas[who] if factor else 1. for key,factor in zip(('policy','value'),factors)}
            assert cfg['branch_scales_float32'][who]==expected
            for key,branch in (('send_context.h_scale','policy'),('send_value.h_scale','value')):
                assert initial[key].ndim==0 and initial[key].dtype==torch.float32 and float(initial[key])==expected[branch]
                assert torch.equal(initial[key],final[key])
            for k,v in initial.items():
                if k.startswith(PREFIXES+('project.',)) or k=='input_transform':assert torch.equal(v,final[k])
        records.append(dict(arm=arm,factors=list(factors),old_tensors_equal_reset=True,only_two_new_fixed_buffers=True))
    latent_checks=0
    for t in times:
        for who in (0,1):
            old=v13.load(Path(runs['neither']['path'])/f'protocol_{t:04d}_d{who}.npz')
            reference=v13.load(Path(runs['both']['path'])/f'protocol_{t:04d}_d{who}.npz')
            assert np.array_equal(reference['h'],old['h']*np.float32(alphas[who]));latent_checks+=1
            for arm,factors in (('policy_only_scale',(1,0)),('value_only_scale',(0,1))):
                new=v13.load(Path(runs[arm]['path'])/f'protocol_{t:04d}_d{who}.npz')
                assert np.array_equal(new['h'],old['h']) and np.array_equal(new['raw_h'],old['h']);latent_checks+=2
                for name,factor in zip(('policy_h','value_h'),factors):
                    assert np.array_equal(new[name],old['h']*np.float32(alphas[who] if factor else 1.));latent_checks+=1
    return dict(seed=seed,partition=p,cells=records,latent_array_equalities=latent_checks,
        raw_policy_value_semantics_verified=True,source_effective_h_matches_raw_times_scale=True)


def report(r):
    s=r['summary'];lines=['# 策略与价值输入尺度：分析草稿','',
        '全部24个新增单支路运行完成后统一分析，00复用v0.14 reset的12运行，11复用v0.15 reset_scaled的12运行。48个总运行中只有24个是新训练；4个既有来源种子各自先平均3分区×2方向。照片、来源和参考已被观察，不是独立确认。','',
        '唯一主终点：2400步封存6图自然双目标N的policy_only_scale−value_only_scale（route_difference）。四个简单效应和2×2交互为辅助，全部方向固定；AUC由8个预定检查点梯形面积除以2400，不挑选最佳阶段。','',
        '| 单元 | old N | added N | sealed N | sealed AUC |','| --- | ---: | ---: | ---: | ---: |']
    for arm in ARMS:
        m=s['aggregate'][arm]['metrics'];lines.append(f'| {LABELS[arm]} | '+' | '.join(f'{100*m[k]:.2f}%' for k in ('normal.old.N','normal.added.N','normal.sealed.N','auc.sealed.N'))+' |')
    lines+=['','| 对比 | 指标 | 四来源差（百分点） | 平均差 |','| --- | --- | --- | ---: |']
    for name,c in s['contrasts'].items():
        for k in ('normal.sealed.N','auc.sealed.N','normal.old.N','auc.old.N','normal.added.N','auc.added.N'):
            lines.append(f'| {name} | {k} | '+'、'.join(f'{100*v:+.4f}' for v in c['seed_differences'][k])+f' | {100*c["mean_differences"][k]:+.4f} |')
    lines+=['','简单效应：policy_given_value0=10−00；policy_given_value1=11−01；value_given_policy0=01−00；value_given_policy1=11−10。interaction=11−10−01+00。交互描述当前成功率尺度上的非加性，不是一般能力的必要性；未设等效界，均值接近不表示等价。','',
        '| 单元 | common30原生联合MAP | 原生Q | 原生G | 贪心N | 原生联合条件熵 | 贪心完整码数 |','| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for arm in ARMS:
        m=s['aggregate'][arm]['metrics'];pre='protocol.common30.'
        lines.append(f'| {LABELS[arm]} | '+' | '.join(f'{100*m[pre+k]:.2f}%' for k in ('native_joint_oracle','Q','G','N'))+f' | {m[pre+"joint_entropy"]:.6f} | {m[pre+"greedy_used_codes"]:.2f} |')
    lines+=['','自然N主成绩使用原9600世界；协议用固定16照片对，两种支持不直接相减。Q为双方原生策略的解析期望，G为原生发送加贪心接收，N改变为逐符号贪心发送。common30联合MAP表只约束自身原生发送政策，不跨单元借用，也不拼接子组oracle。','',
        '| 单元 | 终点模式 | 全30图J | sealed J | 全30图收益 |','| --- | --- | ---: | ---: | ---: |']
    for arm in ARMS:
        m=s['aggregate'][arm]['metrics']
        for mode in MODES:lines.append(f'| {LABELS[arm]} | {mode} | {100*m[f"intervention.{mode}.all.joint"]:.2f}% | {100*m[f"intervention.{mode}.sealed.joint"]:.2f}% | {m[f"intervention.{mode}.all.reward"]:.6f} |')
    lines+=['','stochastic为固定流抽样轨迹，与协议解析Q分开；shuffle/blank/erase_memory检查功能依赖，不是语法证据。','',
        '倍率完整复用v0.15 old18训练照片校准，未重估或扫描。新接口observe保持raw；两个Sequential只放大前96维h，后4维不变。协议raw_h、policy_h、value_h已逐数组与旧reset raw及历史11有效h对齐；send仍只接raw，没有乘后相除或二次缩放。全部既有tensor与reset初值相同，只有两个persistent buffer新增且冻结。','',
        '价值支路有两条算法路径：value数值进入.detach优势权重，改变有限批次策略梯度；value MSE梯度又与其余send_共同norm2裁剪。理想未裁剪期望中的基线不偏性不意味着有限batch/clip/Adam轨迹相同。本2×2不区分这两条价值路径，也不识别正式中介比例。','',
        '结果应解释为同一冻结表示和预算下输入路径的总影响。消息策略输入放大可改变激活/熵/优化；价值支路改善不表示新增世界信息。旧图与封存结果并列，避免将一般学习差异命名为独有组合能力。来源复用、n4与旧图片的限制仍在；本轮没有新增片段拼接实验、倍率或训练矩阵。']
    for f in r['figures']:lines+=['',f'![策略和价值输入尺度]({f["png"]})']
    return '\n'.join(lines)+'\n'


def self_test():
    with tempfile.TemporaryDirectory() as folder:
        f=Path(folder);items,missing=inventory(f/'new',f/'source',f/'reset',f/'scaled',f/'cal',SEEDS,PARTITIONS,TIMES)
        assert len(items)==12 and len(missing)==67
    assert v13.auc([0,1,4],[0,1,0])==.5
    assert CONTRASTS['route_difference']=={'policy_only_scale':1,'value_only_scale':-1}
    v={'neither':.1,'policy_only_scale':.3,'value_only_scale':.2,'both':.6}
    assert abs(sum(w*v[k] for k,w in CONTRASTS['interaction'].items())-.2)<1e-12
    return dict(passed=True,checks=['All48 total cells plus calibration/completion before effect analysis','Only24 cells counted as new','Fixed primary route contrast and secondary interaction','Normalized irregular-time AUC'])


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('batch',nargs='?',type=Path,default=ROOT/'results/branches_001');ap.add_argument('--smoke',action='store_true');ap.add_argument('--self-test',action='store_true');args=ap.parse_args()
    tests=self_test()
    if args.self_test:print(json.dumps(tests));return
    batch=args.batch.resolve();formal=not args.smoke
    if formal:source,reset_source,scaled_source,calibration,seeds,parts,times=SOURCE,RESET_SOURCE,SCALED_SOURCE,CALIBRATION,SEEDS,PARTITIONS,TIMES
    else:
        assert batch.name.startswith('smoke_');inv=read(batch/'invocation.json');assert not inv['formal']
        cfg=inv['args'];source=Path(cfg['source_batch']);reset_source=Path(cfg['reset_batch']);scaled_source=Path(cfg['scaled_batch']);calibration=Path(cfg['calibration']);seeds=tuple(cfg['seeds']);parts=tuple(cfg['partitions'])
        times=tuple(sorted({0,cfg['social_updates'],*(t for t in TIMES if t<cfg['social_updates'])}))
    groups,missing=inventory(batch,source,reset_source,scaled_source,calibration,seeds,parts,times)
    if missing:print(json.dumps(dict(status='pending',missing=missing,result_content_read=False)));return
    done=read(batch/'training_complete.json');assert done['status']=='complete' and done['new_private_runs']==0
    assert done['new_social_runs']==2*len(groups) if formal else done['new_social_runs']>=2*len(groups)
    sr=read(batch/'source_receipt.json')
    for receipt in (sr['retained'],sr['reset'],sr['scaled']):
        for path,digest in receipt['files'].items():assert sha(path)==digest,path
    cr=read(batch/'calibration_receipt.json');assert sha(calibration/'calibration_complete.json')==cr['sha256']
    for path,digest in cr['files'].items():assert sha(calibration/path)==digest,path
    runs=[];calibrations=[];states=[]
    for seed,p,folders in groups:
        c=previous.verify_calibration(calibration/f's{seed}_p{p}',seed,p,formal);calibrations.append(c)
        cells={arm:load_social(path,source,seed,p,arm,times,formal) for arm,path in folders.items()}
        assert len({r['world_sequence_sha256'] for r in cells.values()})==1
        assert len({r['curve'][0]['scores']['normal']['world_sha256'] for r in cells.values()})==1
        states.append(verify_cells(seed,p,cells,c,times));runs.extend(cells.values())
    result=dict(status='complete' if formal else 'development_smoke',seeds=list(seeds),partitions=list(parts),times=list(times),
        new_social_runs=2*len(groups),reused_social_runs=2*len(groups),total_social_runs=4*len(groups),new_private_runs=0,
        runs=runs,summary=summarize(runs,seeds,parts,times),calibration=calibrations,state_checks=states,source_batch=str(source),reset_source_batch=str(reset_source),scaled_source_batch=str(scaled_source),calibration_batch=str(calibration),
        analysis_sha256=sha(__file__),helper_sha256=sha(Path(v13.__file__)),previous_analysis_helper_sha256=sha(Path(previous.__file__)),
        source_receipt_sha256=sha(batch/'source_receipt.json'),calibration_receipt_sha256=sha(batch/'calibration_receipt.json'),self_tests=tests,
        created_utc=datetime.now(timezone.utc).isoformat(),primary='policy_only_scale minus value_only_scale sealed N at2400',
        audit=dict(complete_before_effect_analysis=True,all_raw_endpoints_and_protocols_recomputed=True,raw_calibration_recomputed=True,
            base_tensors_equal_reset=True,branch_scales_and_interface_frozen=True,raw_effective_h_semantics_verified=True,external_worlds_paired=True,no_model_forward=True),
        reused_references_are_previously_seen_development_evidence=True)
    if formal:
        out=batch/'figures';out.mkdir(exist_ok=True);result['figures']=figures(result,out);(batch/'策略与价值尺度_分析草稿.md').write_text(report(result))
    name='branches_analysis.json' if formal else 'analysis_smoke.json';(batch/name).write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'],new_social_runs=2*len(groups),output=str(batch/name),audit=result['audit']),ensure_ascii=False))


if __name__=='__main__':main()

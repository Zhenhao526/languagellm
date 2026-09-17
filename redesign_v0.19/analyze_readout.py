"""Fresh private readout analysis from complete raw evaluations; no training."""
from __future__ import annotations
import argparse,hashlib,itertools,json,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
SEEDS=(31101,31102,31103,31104)
TIMES=(0,100,300,600,1200,1800,2100,2400)
INTERFACES=('retained','reset_scaled');SIGNALS=('reward','ce')
GROUPS=('old','added','sealed','common30')
METRICS=('J','single','Q','native_single','entropy','nll','exact_max_tie_rate')
MAPS=np.asarray(list(itertools.permutations(range(6),2)),np.int64)
MATCHINGS=(((0,1),(2,3),(4,5)),((0,2),(1,4),(3,5)),((0,3),(1,5),(2,4)))
COLORS={'retained':'#24648f','reset_scaled':'#bb703c'}
LABELS={'retained':'Retained interface','reset_scaled':'Reset + fixed scale'}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def pools(partition):
    assert partition in (1,2,3)
    def group(index):
        pairs={pair for a,b in MATCHINGS[index] for pair in ((a,b),(b,a))}
        return np.asarray([i for i,p in enumerate(MAPS) if tuple(p) in pairs])
    added=group(partition-1);sealed=group(partition%3)
    old=np.setdiff1d(np.arange(30),np.r_[added,sealed])
    return dict(old=old,added=added,sealed=sealed,all=np.arange(30),common30=np.arange(30))

def metrics(raw,partition):
    logits=np.asarray(raw['logits'],np.float64);position=raw['positions'];ids=raw['map_ids']
    assert logits.shape==(len(ids),2,6) and np.isfinite(logits).all()
    assert np.array_equal(position,MAPS[ids])
    centered=logits-logits.max(-1,keepdims=True)
    logp=centered-np.log(np.exp(centered).sum(-1,keepdims=True));prob=np.exp(logp)
    success=logits.argmax(-1)==position
    target_logp=np.take_along_axis(logp,position[...,None],axis=-1)[...,0]
    target_prob=np.exp(target_logp)
    entropy=-(prob*logp).sum(-1);ties=(logits==logits.max(-1,keepdims=True)).sum(-1)>1
    answer={}
    for group,maps in pools(partition).items():
        mask=np.isin(ids,maps);correct=success[mask]
        answer[group]=dict(n=int(mask.sum()),J=float(correct.all(-1).mean()),single=float(correct.mean()),
            Q=float(target_prob[mask].prod(-1).mean()),native_single=float(target_prob[mask].mean()),
            entropy=float(entropy[mask].mean()),nll=float(-target_logp[mask].mean()),
            correct_joint=int(correct.all(-1).sum()),correct_goals=int(correct.sum()),
            exact_max_tie_goal_count=int(ties[mask].sum()),exact_max_tie_rate=float(ties[mask].mean()))
    return answer

def check_nested(a,b,path=''):
    if isinstance(b,dict):
        for k,v in b.items():check_nested(a[k],v,path+'.'+k)
    elif isinstance(b,list):
        assert len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)):check_nested(x,y,path+f'[{i}]')
    elif isinstance(b,(int,float)) and not isinstance(b,bool):
        assert np.isclose(a,b,atol=2e-12,rtol=2e-12),(path,a,b)
    else:assert a==b,(path,a,b)

def auc(times,values):return float(np.trapz(values,times)/(times[-1]-times[0]))

def summarize(runs,seeds,parts,times):
    expected=set(itertools.product(seeds,parts,(0,1),INTERFACES,SIGNALS))
    assert {(r['seed'],r['partition'],r['direction'],r['interface'],r['signal']) for r in runs}==expected and len(runs)==len(expected)
    seed_rows=[];aggregate=[]
    for seed,interface,signal in itertools.product(seeds,INTERFACES,SIGNALS):
        children=[r for r in runs if (r['seed'],r['interface'],r['signal'])==(seed,interface,signal)]
        curve=[]
        for j,t in enumerate(times):
            assert all(r['curve'][j]['update']==t for r in children)
            curve.append(dict(update=t,scores={g:{k:float(np.mean([r['curve'][j]['scores'][g][k] for r in children])) for k in METRICS} for g in GROUPS}))
        areas={g:{k:float(np.mean([r['auc'][g][k] for r in children])) for k in METRICS} for g in GROUPS}
        for g,k in itertools.product(GROUPS,METRICS):check_nested(areas[g][k],auc(times,[p['scores'][g][k] for p in curve]))
        seed_rows.append(dict(seed=seed,interface=interface,signal=signal,curve=curve,auc=areas))
    for interface,signal in itertools.product(INTERFACES,SIGNALS):
        children=[r for r in seed_rows if (r['interface'],r['signal'])==(interface,signal)]
        curve=[dict(update=t,scores={g:{k:float(np.mean([r['curve'][j]['scores'][g][k] for r in children])) for k in METRICS} for g in GROUPS}) for j,t in enumerate(times)]
        aggregate.append(dict(interface=interface,signal=signal,curve=curve,auc={g:{k:float(np.mean([r['auc'][g][k] for r in children])) for k in METRICS} for g in GROUPS}))
    contrasts=[]
    for seed in seeds:
        lookup={(r['interface'],r['signal']):r for r in seed_rows if r['seed']==seed}
        values={f'{i}_{s}':lookup[i,s]['curve'][-1]['scores']['sealed']['J'] for i,s in itertools.product(INTERFACES,SIGNALS)}
        contrasts.append(dict(seed=seed,values=values,
            primary_reward_retained_minus_reset_scaled=values['retained_reward']-values['reset_scaled_reward'],
            secondary_ce_retained_minus_reset_scaled=values['retained_ce']-values['reset_scaled_ce'],
            secondary_reset_scaled_ce_minus_reward=values['reset_scaled_ce']-values['reset_scaled_reward'],
            secondary_retained_ce_minus_reward=values['retained_ce']-values['retained_reward']))
    paired_curves=[]
    for signal in SIGNALS:
        for seed in seeds:
            lookup={r['interface']:r for r in seed_rows if r['seed']==seed and r['signal']==signal}
            curve=[dict(update=t,scores={g:{k:lookup['retained']['curve'][j]['scores'][g][k]-lookup['reset_scaled']['curve'][j]['scores'][g][k] for k in METRICS} for g in GROUPS}) for j,t in enumerate(times)]
            paired_curves.append(dict(seed=seed,signal=signal,contrast='retained_minus_reset_scaled',curve=curve,
                auc={g:{k:lookup['retained']['auc'][g][k]-lookup['reset_scaled']['auc'][g][k] for k in METRICS} for g in GROUPS}))
    return dict(seed_rows=seed_rows,aggregate=aggregate,contrasts=contrasts,paired_curves=paired_curves,
        primary_mean=float(np.mean([r['primary_reward_retained_minus_reset_scaled'] for r in contrasts])))

def analyze(out,dev=False):
    done=read(out/'training_complete.json');inv=read(out/'invocation.json')
    assert done['status']=='complete' and done['formal']==inv['formal']==(not dev)
    seeds=(99513,) if dev else SEEDS;parts=(1,) if dev else (1,2,3)
    assert inv['seeds']==list(seeds) and inv['partitions']==list(parts) and inv['interfaces']==list(INTERFACES) and inv['signals']==list(SIGNALS)
    updates=inv['updates'];assert dev or updates==2400
    times=sorted({0,updates,*[t for t in TIMES if t<updates]})
    expected_names={f's{s}_p{p}_d{d}_{i}_{t}' for s,p,d,i,t in itertools.product(seeds,parts,(0,1),INTERFACES,SIGNALS)}
    assert set(done['runs'])==expected_names and len(done['runs'])==done['head_fits']==len(expected_names)
    assert done['head_updates']==updates*len(expected_names) and done['simulated_selected_goal_actions']==updates*1024*len(expected_names)
    for path,digest in {**inv['source_hashes'],**inv['input_hashes']}.items():assert sha(path)==digest,path
    def verify(path):
        assert sha(path)==done['files'][str(path.relative_to(out))],str(path)
    verify(out/'invocation.json')
    runs=[];initials={};cache_receipts={};eval_world_sha={};head_seed_values={}
    for name in sorted(expected_names):
        folder=out/name
        for filename in ('config.json','result.json','curve.json'):verify(folder/filename)
        cfg=read(folder/'config.json');result=read(folder/'result.json');curve=read(folder/'curve.json')
        assert result['status']=='complete' and result['only_head_trained'] and result['no_communication']
        assert cfg['updates']==result['updates']==updates and cfg['checkpoints']==times==[p['update'] for p in curve]
        assert cfg['batch_pairs']==512 and cfg['worlds_per_update']==1024 and cfg['learning_rate']==.0007 and cfg['gradient_clip']==2.
        assert cfg['entropy_coefficient']==.02 and cfg['entropy_off_after']==2100 and cfg['only_head_trainable'] and not cfg['new_communication_training']
        assert cfg['training_pool']==pools(cfg['partition'])['old'].tolist() and cfg['test_world_count']==1920
        assert cfg['correct_labels_available']==(cfg['signal']=='ce')
        assert all(cfg[k]==result[k] for k in ('seed','partition','direction','interface','signal'))
        assert cfg['initial_sha256']==result['initial_sha256']
        identity=(cfg['seed'],cfg['partition'],cfg['direction'])
        initials.setdefault(identity,set()).add(cfg['initial_sha256']);head_seed_values.setdefault(identity,set()).add(cfg['head_initialization_seed'])
        cp=Path(cfg['cache_receipt']);assert sha(cp)==cfg['cache_receipt_sha256']==result['cache_receipt_sha256']
        cache_receipts[str(cp)]=cfg['cache_receipt_sha256']
        points=[]
        for point in curve:
            file=folder/f"evaluation_{point['update']:04d}.npz";verify(file);assert sha(file)==point['raw_sha256']
            with np.load(file,allow_pickle=False) as z:
                raw={k:z[k] for k in z.files}
            assert len(raw['map_ids'])==1920 and np.array_equal(np.bincount(raw['map_ids'],minlength=30),np.full(30,64))
            photo_pairs=np.unique(raw['photo_ids'],axis=0);assert photo_pairs.shape==(64,2)
            assert len(np.unique(photo_pairs[:,0]))==len(np.unique(photo_pairs[:,1]))==8
            assert len(set((int(m),int(f),int(w)) for m,(f,w) in zip(raw['map_ids'],raw['photo_ids'])))==1920
            worldhash=hashlib.sha256(b''.join(np.ascontiguousarray(raw[k]).tobytes() for k in ('map_ids','positions','photo_ids'))).hexdigest()
            eval_world_sha.setdefault(identity,set()).add(worldhash)
            score=metrics(raw,cfg['partition']);check_nested(score,point['scores'])
            points.append(dict(update=point['update'],scores=score))
        check_nested(points[-1]['scores'],result['scores'])
        row={k:cfg[k] for k in ('seed','partition','direction','interface','signal')}
        row.update(curve=points,auc={g:{k:auc(times,[p['scores'][g][k] for p in points]) for k in METRICS} for g in GROUPS},
            initial_sha256=cfg['initial_sha256'],head_initialization_seed=cfg['head_initialization_seed'],updates=updates,
            source_folder=name,cache_receipt_sha256=cfg['cache_receipt_sha256'])
        runs.append(row)
    assert all(len(v)==1 for v in initials.values()) and all(len(v)==1 for v in head_seed_values.values())
    assert len({next(iter(v)) for v in head_seed_values.values()})==len(head_seed_values)
    assert all(len(v)==1 for v in eval_world_sha.values())
    summary=summarize(runs,seeds,parts,times)
    return dict(status='complete',formal=not dev,created_utc=datetime.now(timezone.utc).isoformat(),analysis_source_sha256=sha(__file__),
        training_receipt_sha256=sha(out/'training_complete.json'),independent_recount_used_as_input=False,
        seeds=list(seeds),partitions=list(parts),times=times,updates=updates,head_fits=len(runs),runs=runs,**summary,
        initial_head_pairing_verified=True,unique_head_initializations=len(head_seed_values),cache_receipts=cache_receipts,
        evaluation_worlds_per_head=1920,photos_per_category=8,
        primary='Reward retained minus reset_scaled, sealed greedy joint J at2400; all directions/partitions within four reused source seeds',
        labels=dict(J='Greedy success for both private goals',Q='Independent stochastic action success for both private goals',
            nll='Mean selected-goal negative log probability, nats per query',entropy='Mean policy entropy across the two goal branches, nats'),
        boundaries=['Frozen interfaces; newly initialized heads only','CE has correct selected-goal location labels and does not use own simulated reward in its loss',
            'No head or label supervision transfers to communication','Finite head family and budget; low performance is not absent information',
            '64 old test-photo pairs per map; cannot subtract prior16-pair scores','Four reused development sources; no significance/equivalence claim'])

def compare(out):
    data=read(out/'analysis.json');ref=read(out/'independent_recount.json');assert ref['passed']
    count=0;maxerror=0.
    def check(a,b,path=''):
        nonlocal count,maxerror
        if isinstance(b,dict):
            for k,v in b.items():check(a[k],v,path+'.'+k)
        elif isinstance(b,list):
            assert len(a)==len(b),path
            for i,(x,y) in enumerate(zip(a,b)):check(x,y,path+f'[{i}]')
        elif isinstance(b,(float,int)) and not isinstance(b,bool):
            check_nested(a,b,path);count+=1;maxerror=max(maxerror,abs(a-b))
        else:assert a==b,(path,a,b)
    for name,ids in [('runs',('seed','partition','direction','interface','signal')),('seed_rows',('seed','interface','signal')),('aggregate',('interface','signal')),('contrasts',('seed',))]:
        actual={tuple(r[k] for k in ids):r for r in data[name]};expected={tuple(r[k] for k in ids):r for r in ref[name]}
        assert actual.keys()==expected.keys()
        for key,row in expected.items():check(actual[key],row,name+str(key))
    check(data['primary_mean'],ref['primary_mean'],'primary_mean')
    result=dict(passed=True,comparisons=count,max_absolute_error=maxerror,analysis_sha256=sha(out/'analysis.json'),analysis_source_sha256=sha(__file__),
        independent_recount_sha256=sha(out/'independent_recount.json'),scope='Every raw-evaluation metric/count, curve and normalized AUC; nested source/condition means; all four predeclared endpoint contrasts')
    write(out/'analysis_comparison.json',result);return result

def figures(data,out):
    sys.path.insert(0,str(ROOT.parent/'redesign_v0.9/.analysis_deps'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
    dest=out/'figures';dest.mkdir(exist_ok=True);files=[]
    def save(fig,stem):
        fig.canvas.draw()
        for ax in fig.axes:ax.get_xticklabels();ax.get_yticklabels()
        fig.canvas.draw()
        for ext in ('png','pdf'):
            path=dest/f'{stem}.{ext}';fig.savefig(path,dpi=180);files.append(path)
        plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13.8,4.8));fig.subplots_adjust(left=.055,right=.985,top=.79,bottom=.24,wspace=.27)
    for ax,g in zip(axes,('old','added','sealed')):
        for interface,signal in itertools.product(INTERFACES,SIGNALS):
            subset=[r for r in data['seed_rows'] if r['interface']==interface and r['signal']==signal]
            for row in subset:ax.plot(data['times'],[100*p['scores'][g]['J'] for p in row['curve']],color=COLORS[interface],ls='-' if signal=='reward' else '--',alpha=.16,lw=.8)
            row=next(r for r in data['aggregate'] if r['interface']==interface and r['signal']==signal)
            ax.plot(data['times'],[100*p['scores'][g]['J'] for p in row['curve']],color=COLORS[interface],ls='-' if signal=='reward' else '--',lw=1.8,label=LABELS[interface]+(' | reward' if signal=='reward' else ' | CE reference'))
        ax.set_title({'old':'Old: trained maps','added':'Added: untrained maps','sealed':'Sealed: untrained maps'}[g]);ax.set_ylim(0,102)
        ax.set_xlabel('New-head updates');ax.set_xticks([0,data['updates']//2,data['updates']]);ax.set_ylabel('Greedy two-goal success J (%)')
        ax.tick_params(labelleft=True);ax.grid(axis='y',alpha=.16)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.52,.925),ncol=2,frameon=False,fontsize=9)
    fig.suptitle('Fresh private action learning from frozen visual interfaces',y=.99)
    fig.text(.055,.065,f"{'DEVELOPMENT ONLY. ' if not data['formal'] else ''}Thin lines: {len(data['seeds'])} source seeds, each averaging {len(data['partitions'])} partitions × 2 persons. Thick lines: source means.\nCE uses correct selected-goal location labels; reward uses own binary action feedback. No interface or communication training.",fontsize=9)
    save(fig,'01_private_learning')
    fig,axes=plt.subplots(1,2,figsize=(10.6,4.9));fig.subplots_adjust(left=.08,right=.97,bottom=.25,top=.82,wspace=.34)
    for ax,signal in zip(axes,SIGNALS):
        for index,seed in enumerate(data['seeds']):
            values=[]
            for interface in INTERFACES:
                row=next(r for r in data['seed_rows'] if (r['seed'],r['interface'],r['signal'])==(seed,interface,signal))
                values.append(100*row['curve'][-1]['scores']['sealed']['J'])
            ax.plot([0,1],values,'o-',color=plt.get_cmap('tab10')(index),ms=4,lw=1,label=str(seed))
        avg=[100*next(r for r in data['aggregate'] if (r['interface'],r['signal'])==(i,signal))['curve'][-1]['scores']['sealed']['J'] for i in INTERFACES]
        ax.scatter([0,1],avg,marker='_',s=350,c='black',lw=2.5,zorder=4,label='Source mean')
        ax.set_xticks([0,1],['Retained','Reset + fixed scale']);ax.set_xlim(-.2,1.2);ax.set_ylim(0,102)
        ax.set_ylabel('Sealed greedy two-goal success J (%)');ax.tick_params(labelleft=True);ax.grid(axis='y',alpha=.16)
        ax.set_title('Primary: reward-trained head' if signal=='reward' else 'Auxiliary: supervised CE readout')
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.52,.94),ncol=5,frameon=False,fontsize=9)
    fig.suptitle(f"Paired frozen-interface readouts after {data['updates']} new-head updates",y=.995)
    fig.text(.08,.065,f"{'DEVELOPMENT ONLY. ' if not data['formal'] else ''}n = {len(data['seeds'])} reused source seeds; new head initialized identically within each four-condition source.\nEvaluation: 64 old test-photo pairs per map. CE labels and newly trained heads are never transferred to communication.",fontsize=9)
    save(fig,'02_paired_sealed_readout')
    write(out/'figure_manifest.json',dict(analysis_sha256=sha(out/'analysis.json'),analysis_source_sha256=sha(__file__),files={str(p.relative_to(out)):sha(p) for p in files}))

def draft(data,out):
    lines=['# 公平新私人读出：分析草稿','',('正式96头完整分析' if data['formal'] else '开发流程核查，不纳入正式结论')+'。所有视觉接口冻结，只训练相同新初值的小型私人行动头。','',
        '**唯一主比较：reward条件下，retained−reset_scaled的2400步sealed贪心双目标成功率。** CE属于有正确地点标签的监督可读性参照，不改变主比较。','','| 接口 | 训练信号 | old J | added J | sealed J | sealed随机Q | sealed J AUC |','|---|---|---:|---:|---:|---:|---:|']
    for r in data['aggregate']:
        score=r['curve'][-1]['scores'];lines.append(f"| {r['interface']} | {r['signal']} | {100*score['old']['J']:.3f}% | {100*score['added']['J']:.3f}% | {100*score['sealed']['J']:.3f}% | {100*score['sealed']['Q']:.3f}% | {100*r['auc']['sealed']['J']:.3f}% |")
    lines+=['','| 来源 | reward retained | reward reset_scaled | 主差，百分点 | CE接口差，百分点 |','|---|---:|---:|---:|---:|']
    for r in data['contrasts']:
        lines.append(f"| {r['seed']} | {100*r['values']['retained_reward']:.3f}% | {100*r['values']['reset_scaled_reward']:.3f}% | {100*r['primary_reward_retained_minus_reset_scaled']:+.3f} | {100*r['secondary_ce_retained_minus_reset_scaled']:+.3f} |")
    delta=[100*r['primary_reward_retained_minus_reset_scaled'] for r in data['contrasts']]
    lines+=['',f"预定主差的四来源均值为 {100*data['primary_mean']:+.3f} 个百分点，来源差范围 [{min(delta):+.3f}, {max(delta):+.3f}]。先平均每来源的分区和两人，再平均来源；不把96头或1920世界当独立来源。",'',
        'J为两次私人需求都按贪心行动正确，Q为各需求按其原生随机政策独立行动时的双成功期望；single/native_single相应为单目标平均。熵和NLL均按两需求平均，每个需求查询为单位。原始计数、并列率、全部8检查点和归一化AUC均保留在JSON中。',
        '', '这项实验去除了原私人头的兼容优势，但仍只测特定新头架构、学习信号和预算。低读出成功不是信息不存在；两接口接近也不构成等效证明。retained与reset_scaled沿用既有尺度，未重调均值/坐标统计。CE获得正确地点标签，不能将其分数当成无监督能力；所有新头与标签均不回灌社会主体。',
        '', '评价为每图64个旧测试照片对，不能直接减去v0.13或v0.14中每图16照片对的旧头分数。四个来源沿用开发主体，不是独立新材料确认；本轮没有训练或评价新的共同通信协议，不能单独证明语言能力或非语言准备的唯一机制。','',
        '[结构化结果](analysis.json) · [独立比较](analysis_comparison.json) · [学习曲线](figures/01_private_learning.png) · [配对终点](figures/02_paired_sealed_readout.png)','']
    (out/'公平私人读出_分析草稿.md').write_text('\n'.join(lines))

def self_test():
    logits=np.zeros((30,2,6),np.float32);raw=dict(logits=logits,positions=MAPS.copy(),map_ids=np.arange(30))
    result=metrics(raw,1)
    assert result['common30']['J']==0 and result['common30']['single']==1/6 and result['common30']['exact_max_tie_rate']==1
    check_nested(result['common30']['Q'],1/36);check_nested(result['common30']['nll'],np.log(6))
    perfect=np.full((30,2,6),-30.,np.float32);perfect[np.arange(30)[:,None],np.arange(2)[None,:],MAPS]=30.
    raw['logits']=perfect;assert metrics(raw,1)['common30']['J']==1
    assert auc([0,100,300],[0.,1.,0.])==.5
    runs=[];times=[0,2]
    for seed,p,d,i,s in itertools.product(SEEDS,(1,2,3),(0,1),INTERFACES,SIGNALS):
        curve=[dict(update=t,scores={g:{k:(seed-31100)*.01+p*.001+d*.002+(i=='retained')*.03+(s=='ce')*.04+t*.001 for k in METRICS} for g in GROUPS}) for t in times]
        runs.append(dict(seed=seed,partition=p,direction=d,interface=i,signal=s,curve=curve,auc={g:{k:auc(times,[x['scores'][g][k] for x in curve]) for k in METRICS} for g in GROUPS}))
    summ=summarize(runs,SEEDS,(1,2,3),times);check_nested(summ['primary_mean'],.03)
    assert len(summ['seed_rows'])==16 and len(summ['contrasts'])==4
    return dict(passed=True,source_sha256=sha(__file__),checks=['Uniform/perfect policies and argmax ties','Greedy J vs stochastic Q and per-query NLL','Irregular checkpoint normalized trapezoidal AUC','Four-source nested paired aggregation'])

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path);p.add_argument('--dev',action='store_true');p.add_argument('--self-test',action='store_true');p.add_argument('--compare-only',action='store_true');p.add_argument('--figures-only',action='store_true');args=p.parse_args()
    if args.self_test:print(json.dumps(self_test()));return
    assert args.out is not None;out=args.out.resolve()
    if args.compare_only:print(json.dumps(compare(out)));return
    if args.figures_only:figures(read(out/'analysis.json'),out);print('FIGURES COMPLETE');return
    assert not (out/'analysis.json').exists(),'refuse silent overwrite'
    data=analyze(out,args.dev);write(out/'analysis.json',data);draft(data,out);figures(data,out)
    if (out/'independent_recount.json').exists():compare(out)
    print(json.dumps(dict(status='complete',formal=data['formal'],head_fits=data['head_fits'],primary_mean=data['primary_mean']),ensure_ascii=False))
if __name__=='__main__':main()

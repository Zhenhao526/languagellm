"""Independent complete two-frame private-capability analysis; no training."""
from __future__ import annotations
import argparse,hashlib,itertools,json,sys
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
SEEDS=(32101,32102,32103,32104);ARMS=('full','detach');MODES=('immediate','delayed')
TIMES=(0,100,300,600,1200,1800,2100,2400);GROUPS=('old','added','sealed','common30')
METRICS=('J','single','moved','stationary','Q','entropy','nll','exact_max_tie_rate','history_pair_J')
MAPS=np.asarray(list(itertools.permutations(range(6),2)),np.int64)
MATCHINGS=(((0,1),(2,3),(4,5)),((0,2),(1,4),(3,5)),((0,3),(1,5),(2,4)))
COLORS={'full':'#24648f','detach':'#bb703c'}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def pools(p):
    def group(k):
        pairs={pair for a,b in MATCHINGS[k] for pair in ((a,b),(b,a))}
        return np.asarray([i for i,pair in enumerate(MAPS) if tuple(pair) in pairs])
    added=group(p-1);sealed=group(p%3)
    return dict(old=np.setdiff1d(np.arange(30),np.r_[added,sealed]),added=added,sealed=sealed,common30=np.arange(30))

def events(p):
    rows=[];index={tuple(row):i for i,row in enumerate(MAPS)}
    for src in pools(p)['old']:
        for mover in (0,1):
            for dest in range(6):
                if dest in MAPS[src]:continue
                target=MAPS[src].copy();target[mover]=dest
                rows.append((int(src),index[tuple(target)],mover))
    assert len(rows)==144
    return rows

def validate_world(raw,p):
    n=len(raw['target_map']);assert n==5760 and raw['positions'].shape==(n,2)
    assert np.array_equal(raw['positions'],MAPS[raw['target_map']])
    assert np.array_equal(np.bincount(raw['target_map'],minlength=30),np.full(30,192))
    old=set(pools(p)['old']);photos=np.unique(raw['photo_ids'],axis=0)
    assert photos.shape==(16,2) and len(np.unique(photos[:,0]))==len(np.unique(photos[:,1]))==4
    expected={(src,tgt,m,int(f),int(w)):(3 if tgt in old else 2) for src,tgt,m in events(p) for f,w in photos}
    got=Counter((int(s),int(t),int(m),int(f),int(w)) for s,t,m,(f,w) in zip(raw['source_map'],raw['target_map'],raw['mover'],raw['photo_ids']))
    assert got==expected,'evaluation must have exact old-prior/terminal-balanced event multiplicities'
    return hashlib.sha256(b''.join(np.ascontiguousarray(raw[k]).tobytes() for k in ('source_map','target_map','mover','photo_ids','positions'))).hexdigest()

def paired_history(actions,raw,mask):
    context={}
    for r in np.flatnonzero(mask):
        mover=int(raw['mover'][r]);target=raw['positions'][r];station=int(target[1-mover])
        key=(mover,int(target[mover]),*map(int,raw['photo_ids'][r]))
        counts=context.setdefault(key,{})
        n,c=counts.get(station,(0,0));counts[station]=(n+1,c+int(actions[r,1-mover]==station))
    total=correct=0
    for classes in context.values():
        n=np.asarray([x[0] for x in classes.values()],np.int64);c=np.asarray([x[1] for x in classes.values()],np.int64)
        total+=int((n.sum()**2-n@n)//2);correct+=int((c.sum()**2-c@c)//2)
    return dict(history_pair_J=correct/total if total else None,history_pairs=total,history_pair_correct=correct)

def metrics(raw,p):
    logits=np.asarray(raw['logits'],np.float64);n=len(raw['target_map'])
    assert logits.shape==(n,2,6) and np.isfinite(logits).all()
    centered=logits-logits.max(-1,keepdims=True);lp=centered-np.log(np.exp(centered).sum(-1,keepdims=True));prob=np.exp(lp)
    positions=raw['positions'];actions=logits.argmax(-1);correct=actions==positions
    rows=np.arange(n);mover=raw['mover'];true_lp=np.take_along_axis(lp,positions[...,None],-1)[...,0]
    entropy=-(prob*lp).sum(-1);tie=(logits==logits.max(-1,keepdims=True)).sum(-1)>1
    scores={}
    for group,maps in pools(p).items():
        mask=np.isin(raw['target_map'],maps);good=correct[mask]
        pair=paired_history(actions,raw,mask)
        assert pair['history_pairs']=={'old':20736,'common30':69120,'added':0,'sealed':0}[group]
        scores[group]=dict(n=int(mask.sum()),J=float(good.all(-1).mean()),single=float(good.mean()),
            moved=float(correct[rows,mover][mask].mean()),stationary=float(correct[rows,1-mover][mask].mean()),
            Q=float(np.exp(true_lp[mask]).prod(-1).mean()),entropy=float(entropy[mask].mean()),nll=float(-true_lp[mask].mean()),
            exact_max_tie_rate=float(tie[mask].mean()),correct_joint=int(good.all(-1).sum()),correct_goals=int(good.sum()),**pair)
    return scores

def close(a,b,path=''):
    if isinstance(b,dict):
        for k,v in b.items():close(a[k],v,path+'.'+k)
    elif isinstance(b,list):
        assert len(a)==len(b),path
        for j,(x,y) in enumerate(zip(a,b)):close(x,y,path+f'[{j}]')
    elif b is None:assert a is None,path
    elif isinstance(b,(int,float)) and not isinstance(b,bool):assert np.isclose(a,b,atol=2e-12,rtol=2e-12),(path,a,b)
    else:assert a==b,(path,a,b)

def mean(values):
    if any(v is None for v in values):assert all(v is None for v in values);return None
    return float(np.mean(values))
def auc(times,values):
    if any(v is None for v in values):assert all(v is None for v in values);return None
    return float(np.trapz(values,times)/(times[-1]-times[0]))
def subtract(a,b):
    if a is None or b is None:assert a is b;return None
    return a-b

def summarize(runs,seeds,parts,times):
    expected=set(itertools.product(seeds,parts,(0,1),ARMS))
    assert len(runs)==len(expected) and {(r['seed'],r['partition'],r['direction'],r['arm']) for r in runs}==expected
    def averaged(children,base):
        curve=[dict(update=t,scores={mode:{g:{k:mean([r['curve'][j]['scores'][mode][g][k] for r in children]) for k in METRICS} for g in GROUPS} for mode in MODES}) for j,t in enumerate(times)]
        areas={mode:{g:{k:mean([r['auc'][mode][g][k] for r in children]) for k in METRICS} for g in GROUPS} for mode in MODES}
        for mode,g,k in itertools.product(MODES,GROUPS,METRICS):close(areas[mode][g][k],auc(times,[r['scores'][mode][g][k] for r in curve]))
        erasure={g:{k:mean([r['erase_scores'][g][k] for r in children]) for k in METRICS} for g in GROUPS}
        return dict(**base,curve=curve,auc=areas,erase_scores=erasure)
    seed_rows=[averaged([r for r in runs if r['seed']==seed and r['arm']==arm],dict(seed=seed,arm=arm)) for seed,arm in itertools.product(seeds,ARMS)]
    aggregate=[averaged([r for r in seed_rows if r['arm']==arm],dict(arm=arm)) for arm in ARMS]
    contrasts=[];paired_curves=[]
    for seed in seeds:
        lookup={r['arm']:r for r in seed_rows if r['seed']==seed}
        full,detach=[lookup[a] for a in ARMS];F,D=full['curve'][-1]['scores'],detach['curve'][-1]['scores']
        contrasts.append(dict(seed=seed,primary_common30_delayed_J=F['delayed']['common30']['J']-D['delayed']['common30']['J'],
            old_delayed_J=F['delayed']['old']['J']-D['delayed']['old']['J'],old_immediate_J=F['immediate']['old']['J']-D['immediate']['old']['J'],
            common30_delayed_J_AUC=full['auc']['delayed']['common30']['J']-detach['auc']['delayed']['common30']['J']))
        paired_curves.append(dict(seed=seed,curve=[dict(update=t,scores={mode:{g:{k:subtract(full['curve'][j]['scores'][mode][g][k],detach['curve'][j]['scores'][mode][g][k]) for k in METRICS} for g in GROUPS} for mode in MODES}) for j,t in enumerate(times)],
            auc={mode:{g:{k:subtract(full['auc'][mode][g][k],detach['auc'][mode][g][k]) for k in METRICS} for g in GROUPS} for mode in MODES}))
    lookup={r['arm']:r['curve'][-1]['scores'] for r in aggregate}
    gate_values=dict(full_old_immediate_J=lookup['full']['immediate']['old']['J'],detach_old_immediate_J=lookup['detach']['immediate']['old']['J'],
        old_immediate_absolute_difference=abs(lookup['full']['immediate']['old']['J']-lookup['detach']['immediate']['old']['J']),
        full_old_delayed_J=lookup['full']['delayed']['old']['J'],old_delayed_difference=lookup['full']['delayed']['old']['J']-lookup['detach']['delayed']['old']['J'],
        old_delayed_source_differences=[r['old_delayed_J'] for r in contrasts])
    criteria=dict(both_immediate_at_least_090=gate_values['full_old_immediate_J']>=.9 and gate_values['detach_old_immediate_J']>=.9,
        immediate_difference_at_most_005=gate_values['old_immediate_absolute_difference']<=.05,
        full_delayed_at_least_080=gate_values['full_old_delayed_J']>=.8,
        delayed_difference_at_least_010=gate_values['old_delayed_difference']>=.1,
        all_source_old_delayed_differences_positive=all(x>0 for x in gate_values['old_delayed_source_differences']))
    eligible=list(seeds)==list(SEEDS) and set(parts)=={1,2,3}
    return dict(seed_rows=seed_rows,aggregate=aggregate,contrasts=contrasts,paired_curves=paired_curves,
        primary_mean=mean([r['primary_common30_delayed_J'] for r in contrasts]),
        capability_gate=dict(passed=eligible and all(criteria.values()),eligible_for_formal_decision=eligible,values=gate_values,criteria=criteria,scope='Old-test only descriptive whole-batch decision; requires all four formal sources, no individual/source filtering or statistical equivalence claim'))

def analyze(out,dev=False):
    done=read(out/'training_complete.json');inv=read(out/'invocation.json')
    assert done['status']=='complete' and done['formal']==inv['formal']==(not dev)
    seeds=(99520,) if dev else SEEDS;parts=(1,) if dev else (1,2,3)
    assert inv['seeds']==list(seeds) and inv['partitions']==list(parts)
    updates=inv['updates'];assert dev or updates==2400
    times=sorted({0,updates,*[t for t in TIMES if t<updates]})
    expected={f's{s}_p{p}_d{d}_{a}' for s,p,d,a in itertools.product(seeds,parts,(0,1),ARMS)}
    assert set(done['runs'])==expected and len(done['runs'])==len(expected)
    assert done['head_fits']==len(expected) and done['private_updates']==updates*len(expected)
    assert done['selected_goal_actions']==updates*len(expected)*512 and done['new_social_training']==0
    for name,digest in {**inv['source_hashes'],**inv['input_hashes']}.items():assert sha(name)==digest,name
    def verify(path):assert sha(path)==done['files'][str(path.relative_to(out))],str(path)
    runs=[];worlds={};initials={}
    for name in sorted(expected):
        folder=out/name
        for filename in ('config.json','result.json','curve.json'):verify(folder/filename)
        cfg,result,curve=[read(folder/f) for f in ('config.json','result.json','curve.json')]
        assert result['status']=='complete' and cfg['updates']==result['updates']==updates
        assert cfg['checkpoints']==times==[r['update'] for r in curve]
        ident=(cfg['seed'],cfg['partition'],cfg['direction']);initials.setdefault(ident,set()).add((cfg['initial_agent_sha256'],cfg['initial_head_sha256']))
        assert cfg['initial_agent_sha256']==curve[0]['agent_sha256'] and cfg['initial_head_sha256']==curve[0]['head_sha256']
        assert cfg['batch_events']==256 and cfg['action_rows']==512 and cfg['trainable_parameters']==155598
        assert cfg['learning_rate']==.0007 and cfg['gradient_clip']==2. and cfg['entropy_on_updates']==2100 and cfg['entropy_weight']==.02
        assert cfg['only_private_preparation'] and cfg['no_communication'] and result['no_communication'] and result['frozen_verified']
        assert cfg['group_maps']=={k:v.tolist() for k,v in pools(cfg['partition']).items()}
        assert all(cfg[k]==result[k] for k in ('seed','partition','direction','arm'))
        points=[]
        for point in curve:
            score={}
            for mode in MODES:
                file=folder/f"evaluation_{point['update']:04d}_{mode}.npz";verify(file)
                with np.load(file,allow_pickle=False) as z:raw={k:z[k] for k in z.files}
                worlds.setdefault(ident,set()).add(validate_world(raw,cfg['partition']))
                score[mode]=metrics(raw,cfg['partition'])
            close(score,point['scores']);points.append(dict(update=point['update'],scores=score))
        close(points[-1]['scores'],result['scores'])
        file=folder/f'evaluation_{updates:04d}_delayed_erase.npz';verify(file)
        with np.load(file,allow_pickle=False) as z:raw={k:z[k] for k in z.files}
        worlds.setdefault(ident,set()).add(validate_world(raw,cfg['partition']))
        erased=metrics(raw,cfg['partition']);close(erased,result['erase_scores'])
        assert erased['common30']['history_pair_correct']==0,'identical last-frame deterministic policy cannot solve distinct stationary pairs'
        assert erased['common30']['J']<=.2+2e-12 and erased['old']['J']<=1/3+2e-12
        row={k:cfg[k] for k in ('seed','partition','direction','arm')}
        row.update(curve=points,erase_scores=erased,auc={mode:{g:{k:auc(times,[r['scores'][mode][g][k] for r in points]) for k in METRICS} for g in GROUPS} for mode in MODES},
            initial_agent_sha256=cfg['initial_agent_sha256'],initial_head_sha256=cfg['initial_head_sha256'],updates=updates,source_folder=name,config=cfg)
        runs.append(row)
    assert all(len(v)==1 for v in initials.values()) and all(len(v)==1 for v in worlds.values())
    return dict(status='complete',formal=not dev,created_utc=datetime.now(timezone.utc).isoformat(),analysis_source_sha256=sha(__file__),
        training_receipt_sha256=sha(out/'training_complete.json'),independent_recount_used_as_input=False,
        seeds=list(seeds),partitions=list(parts),times=times,updates=updates,person_fits=len(runs),runs=runs,**summarize(runs,seeds,parts,times),
        observations='Same two legal frames; full versus boundary gradient detachment is not memory removal',
        scope='5760 weighted worlds per mode; 16 test photo pairs; repeated events and history pairs are not independent sources',
        primary='Full minus detach, common30 delayed greedy joint J at2400',
        history_pairs='Unordered distinct-stationary-target rows with same mover,destination,foodphoto,waterphoto; old20736/common3069120 pairs per model/mode',
        boundaries=['Whole-batch ability gate uses old only, not sealed or selected persons','Full/detach share recurrent parameters across steps',
            'Erasure is information removal and may change hidden-state distribution','Matching-only subgroup can be solved from last frame; its own oracle is not the common30 oracle',
            'Private learning only; no social learning performed in this matrix'])

def compare(out):
    a=read(out/'analysis.json');b=read(out/'independent_recount.json');assert b['passed']
    n=0;maximum=0.
    def check(x,y,path=''):
        nonlocal n,maximum
        if isinstance(y,dict):
            for k,v in y.items():check(x[k],v,path+'.'+k)
        elif isinstance(y,list):
            assert len(x)==len(y),path
            for j,(v,w) in enumerate(zip(x,y)):check(v,w,path+str(j))
        elif isinstance(y,(int,float)) and not isinstance(y,bool):close(x,y,path);n+=1;maximum=max(maximum,abs(x-y))
        else:assert x==y,(path,x,y)
    for name,keys in [('runs',('seed','partition','direction','arm')),('seed_rows',('seed','arm')),('aggregate',('arm',)),('contrasts',('seed',))]:
        actual={tuple(r[k] for k in keys):r for r in a[name]};expected={tuple(r[k] for k in keys):r for r in b[name]}
        assert actual.keys()==expected.keys()
        for ident,row in expected.items():check(actual[ident],row,name+str(ident))
    check(a['primary_mean'],b['primary_mean'],'primary_mean')
    if 'capability_gate' in b:check(a['capability_gate'],b['capability_gate'],'capability_gate')
    receipt=dict(passed=True,comparisons=n,max_absolute_error=maximum,analysis_sha256=sha(out/'analysis.json'),analysis_source_sha256=sha(__file__),independent_recount_sha256=sha(out/'independent_recount.json'))
    write(out/'analysis_comparison.json',receipt);return receipt

def figures(data,out):
    sys.path.insert(0,str(ROOT.parent/'redesign_v0.9/.analysis_deps'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
    dest=out/'figures';dest.mkdir(exist_ok=True);files=[]
    def save(fig,name):
        fig.canvas.draw()
        for ax in fig.axes:ax.get_xticklabels();ax.get_yticklabels()
        fig.canvas.draw()
        for ext in ('png','pdf'):
            f=dest/f'{name}.{ext}';fig.savefig(f,dpi=180);files.append(f)
        plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13.7,4.8));fig.subplots_adjust(left=.06,right=.985,bottom=.26,top=.82,wspace=.3)
    specs=[('immediate','common30','J','Immediate: both goals'),('delayed','common30','J','Delayed: both goals'),('delayed','common30','stationary','Delayed: stationary resource')]
    for ax,(mode,group,metric,title) in zip(axes,specs):
        for arm in ARMS:
            for row in [r for r in data['seed_rows'] if r['arm']==arm]:ax.plot(data['times'],[100*p['scores'][mode][group][metric] for p in row['curve']],color=COLORS[arm],alpha=.2,lw=.7)
            row=next(r for r in data['aggregate'] if r['arm']==arm)
            ax.plot(data['times'],[100*p['scores'][mode][group][metric] for p in row['curve']],color=COLORS[arm],lw=1.8,label='Full BPTT' if arm=='full' else 'Boundary detach')
        if mode=='delayed':ax.axhline(20,color='#777777',ls=':',lw=1,label='Last-frame-only bound')
        ax.set_title(title);ax.set_ylim(0,102);ax.set_xlabel('Private training updates');ax.set_ylabel('Common30 success (%)');ax.tick_params(labelleft=True);ax.grid(axis='y',alpha=.15)
    handles,labels=axes[1].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.52,.94),ncol=3,frameon=False)
    fig.suptitle('Private temporal updating on the shared 30-map support',y=.995)
    fig.text(.06,.08,f"{'DEVELOPMENT ONLY. ' if not data['formal'] else ''}Thin lines: {len(data['seeds'])} source seeds, each averaging {len(data['partitions'])} partitions × 2 persons. Thick lines: source means.\nEqual forward inputs and initial states; detach truncates learning across the state boundary, not the stored information.",fontsize=9)
    save(fig,'01_temporal_learning')
    fig,axes=plt.subplots(1,2,figsize=(11.4,4.9));fig.subplots_adjust(left=.075,right=.97,bottom=.26,top=.8,wspace=.34)
    for j,(metric,title) in enumerate([('J','Primary: delayed two-goal J'),('history_pair_J','Same last frame, distinct histories')]):
        ax=axes[j]
        for index,seed in enumerate(data['seeds']):
            vals=[100*next(r for r in data['seed_rows'] if r['seed']==seed and r['arm']==a)['curve'][-1]['scores']['delayed']['common30'][metric] for a in ARMS]
            ax.plot([0,1],vals,'o-',color=plt.get_cmap('tab10')(index),lw=1,ms=4,label=str(seed))
        means=[100*next(r for r in data['aggregate'] if r['arm']==a)['curve'][-1]['scores']['delayed']['common30'][metric] for a in ARMS]
        erased=[100*next(r for r in data['aggregate'] if r['arm']==a)['erase_scores']['common30'][metric] for a in ARMS]
        ax.scatter([0,1],means,marker='_',s=300,color='black',lw=2,label='Source mean')
        ax.scatter([0,1],erased,marker='x',s=40,color='#888888',label='Erased-history mean')
        ax.set_xticks([0,1],['Full BPTT','Boundary detach']);ax.set_xlim(-.2,1.2);ax.set_ylim(-2,102);ax.set_title(title)
        ax.set_ylabel('Common30 success (%)');ax.tick_params(labelleft=True);ax.grid(axis='y',alpha=.15)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.52,.95),ncol=3,frameon=False,fontsize=9)
    fig.suptitle(f"Paired temporal capability after {data['updates']} private updates",y=.995)
    fig.text(.075,.07,f"{'DEVELOPMENT ONLY. ' if not data['formal'] else ''}n = {len(data['seeds'])} source seeds. History pairs require both stationary predictions to be correct.\nWeighted repeated events/pairs are measurement support, not independent samples. No social training in this matrix.",fontsize=9)
    save(fig,'02_paired_history_capability')
    write(out/'figure_manifest.json',dict(analysis_sha256=sha(out/'analysis.json'),source_sha256=sha(__file__),files={str(p.relative_to(out)):sha(p) for p in files}))

def draft(data,out):
    lines=['# 两帧私人更新能力：分析草稿','',('正式整批分析' if data['formal'] else '开发核查，不作为正式效应')+'。主比较为共同30图delayed贪心双目标J的full−detach。','',
        '| 条件 | immediate共同J | delayed共同J | delayed静止资源 | 同末帧历史配对 | erase共同J |','|---|---:|---:|---:|---:|---:|']
    for r in data['aggregate']:
        end=r['curve'][-1]['scores'];lines.append(f"| {r['arm']} | {100*end['immediate']['common30']['J']:.3f}% | {100*end['delayed']['common30']['J']:.3f}% | {100*end['delayed']['common30']['stationary']:.3f}% | {100*end['delayed']['common30']['history_pair_J']:.3f}% | {100*r['erase_scores']['common30']['J']:.3f}% |")
    lines+=['','| 来源 | 主差，百分点 | old delayed差 | old immediate差 | 共同delayed J AUC差 |','|---|---:|---:|---:|---:|']
    for r in data['contrasts']:lines.append(f"| {r['seed']} | {100*r['primary_common30_delayed_J']:+.3f} | {100*r['old_delayed_J']:+.3f} | {100*r['old_immediate_J']:+.3f} | {100*r['common30_delayed_J_AUC']:+.3f} |")
    lines+=['',f"主差均值 {100*data['primary_mean']:+.3f} 个百分点。只按old测试的整批能力验收结果：{'通过' if data['capability_gate']['passed'] else '未通过'}。各预定标准和原始值完整保存在analysis.json；该工程验收不选择个人、不宣称统计显著或等效。",'',
        'full与detach保留相同两帧及前向状态；差别是跨h0边界的学习梯度，detach仍通过第二步更新共享参数。历史擦除是真信息消融，零隐藏状态也可能造成分布偏移。共同支持仅末帧贪心J上界为20%，同末帧不同静止目标的双历史配对上界为0；matching的added/sealed单组末帧上界可为100%，因此不能单独用其高分验收记忆。',
        '', '评价包含360个加权事件×16照片对。重复事件和平行历史对只是测量权重，不是独立来源；两级汇总先平均分区/两人，再平均四个新来源。私人能力矩阵完成不等于共同符号形成：本轮未训练社会协议，不据私人阳性直接宣称语言能力。','',
        '[结构化结果](analysis.json) · [独立比较](analysis_comparison.json) · [学习曲线](figures/01_temporal_learning.png) · [历史配对](figures/02_paired_history_capability.png)','']
    (out/'两帧私人能力_分析草稿.md').write_text('\n'.join(lines))

def synthetic_world(p):
    photo=np.asarray(list(itertools.product(range(4),range(4,8))),np.int64);old=set(pools(p)['old'])
    rows=[(src,tgt,m,f,w) for src,tgt,m in events(p) for repeat in range(3 if tgt in old else 2) for f,w in photo]
    arr=np.asarray(rows,np.int64)
    return dict(source_map=arr[:,0],target_map=arr[:,1],mover=arr[:,2],photo_ids=arr[:,3:],positions=MAPS[arr[:,1]])

def self_test():
    for p in (1,2,3):
        raw=synthetic_world(p);validate_world(raw,p);n=len(raw['target_map'])
        logits=np.full((n,2,6),-30.);logits[np.arange(n)[:,None],np.arange(2)[None,:],raw['positions']]=30.;raw['logits']=logits
        s=metrics(raw,p);assert s['common30']['J']==s['common30']['history_pair_J']==s['old']['history_pair_J']==1.
        assert s['added']['history_pair_J'] is None and s['sealed']['history_pairs']==0
        logits=np.full_like(logits,-30.)
        for r in range(n):
            m=int(raw['mover'][r]);dest=int(raw['positions'][r,m]);choices=[int(MAPS[t,1-m]) for t in pools(p)['old'] if MAPS[t,m]==dest]
            logits[r,m,dest]=30;logits[r,1-m,min(choices)]=30
        raw['logits']=logits;s=metrics(raw,p)
        close(s['common30']['J'],.2);close(s['old']['J'],1/3)
        assert s['common30']['moved']==1 and s['common30']['history_pair_J']==s['old']['history_pair_J']==0
        # The closed-form pair count equals explicit unordered pairs for one key.
        m,d,f,w=0,0,0,4;idx=np.flatnonzero((raw['mover']==m)&(raw['positions'][:,m]==d)&(raw['photo_ids'][:,0]==f)&(raw['photo_ids'][:,1]==w))
        assert len(idx)==30 and sum(raw['positions'][a,1-m]!=raw['positions'][b,1-m] for a,b in itertools.combinations(idx,2))==360
    # Synthetic summaries check that subgroup scores cannot change the old-only
    # decision, while one negative formal source can reject a positive mean.
    runs=[];times=[0,2400]
    for seed,p,d,arm in itertools.product(SEEDS,(1,2,3),(0,1),ARMS):
        curve=[]
        for t in times:
            scores={}
            for mode in MODES:
                scores[mode]={}
                for group in GROUPS:
                    v=.1 if t==0 else (.95 if mode=='immediate' else (.9 if arm=='full' else .7))
                    scores[mode][group]={k:(None if k=='history_pair_J' and group in ('added','sealed') else v) for k in METRICS}
            curve.append(dict(update=t,scores=scores))
        runs.append(dict(seed=seed,partition=p,direction=d,arm=arm,curve=curve,
            auc={mode:{g:{k:auc(times,[r['scores'][mode][g][k] for r in curve]) for k in METRICS} for g in GROUPS} for mode in MODES},
            erase_scores={g:{k:(None if k=='history_pair_J' and g in ('added','sealed') else 0.) for k in METRICS} for g in GROUPS}))
    result=summarize(runs,SEEDS,(1,2,3),times)
    assert result['capability_gate']['passed'];close(result['primary_mean'],.2)
    for r in runs:
        r['curve'][-1]['scores']['delayed']['sealed']['J']=.01 if r['arm']=='full' else .99
        r['auc']['delayed']['sealed']['J']=auc(times,[x['scores']['delayed']['sealed']['J'] for x in r['curve']])
    assert summarize(runs,SEEDS,(1,2,3),times)['capability_gate']['passed']
    for r in runs:
        if r['seed']==32104 and r['arm']=='full':
            r['curve'][-1]['scores']['delayed']['old']['J']=.6
            r['auc']['delayed']['old']['J']=auc(times,[x['scores']['delayed']['old']['J'] for x in r['curve']])
    result=summarize(runs,SEEDS,(1,2,3),times)
    assert not result['capability_gate']['passed'] and result['capability_gate']['criteria']['delayed_difference_at_least_010']
    return dict(passed=True,source_sha256=sha(__file__),checks=['Independent144-event enumeration and360weighted support','All16photo pairs, equal terminal/mover marginals','Perfect predictions and null matching-only history pairs','Last-frame policy reaches1/5 common and1/3 old with0history pairs','Closed-form unordered pair count independently enumerated'])

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path);p.add_argument('--dev',action='store_true');p.add_argument('--self-test',action='store_true');p.add_argument('--compare-only',action='store_true');p.add_argument('--figures-only',action='store_true');args=p.parse_args()
    if args.self_test:print(json.dumps(self_test()));return
    assert args.out is not None;out=args.out.resolve()
    if args.compare_only:print(json.dumps(compare(out)));return
    if args.figures_only:figures(read(out/'analysis.json'),out);print('FIGURES COMPLETE');return
    assert not (out/'analysis.json').exists(),'refuse silent overwrite'
    data=analyze(out,args.dev);write(out/'analysis.json',data);draft(data,out);figures(data,out)
    if (out/'independent_recount.json').exists():compare(out)
    print(json.dumps(dict(status='complete',formal=data['formal'],person_fits=data['person_fits'],primary_mean=data['primary_mean'],capability_gate=data['capability_gate']),ensure_ascii=False))
if __name__=='__main__':main()

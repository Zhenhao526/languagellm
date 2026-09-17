"""Read-only complete-batch analysis of spatial preparation and social formation."""
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
sys.path.insert(0,str(ROOT.parent/'redesign_v0.12'))
import numpy as np
from receiver_metrics import evaluate_groups
import private_preparation as private

SEEDS=(31101,31102,31103,31104)
ARMS=('control','equivariant')
TIMES=(0,100,300,600,1200,1800,2100,2400)
MODES=('normal','shuffle','blank','stochastic','erase_memory')
GROUPS=('old','added','sealed')
ORACLES=('oracle_branch_ce','oracle_joint_map','oracle_mixed_reward')
COLORS={'control':'#a47746','equivariant':'#277eab'}
LABELS={'control':'Matched control','equivariant':'Spatial consistency'}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def load(path):
    with np.load(path) as z:return {k:z[k] for k in z.files}


def compact(raw):
    out={k:v for k,v in raw.items() if np.isscalar(v)}
    for key in ('stochastic','greedy',*ORACLES):
        out[key]={k:v for k,v in raw[key].items() if np.isscalar(v)}
    return out


def nested_equal(x,y,where='root'):
    if isinstance(x,dict):
        assert set(x)==set(y),(where,set(x),set(y))
        for k in x:nested_equal(x[k],y[k],where+'.'+k)
    elif isinstance(x,list):
        assert len(x)==len(y),where
        for i,(a,b) in enumerate(zip(x,y)):nested_equal(a,b,where+f'[{i}]')
    elif isinstance(x,(int,float)) and not isinstance(x,bool):
        assert np.isclose(x,y,atol=1e-9,rtol=1e-10),(where,x,y)
    else:assert x==y,(where,x,y)


def inventory(batch,seeds,partitions,times):
    paths=[];missing=[]
    for seed,p,arm in itertools.product(seeds,partitions,ARMS):
        social=batch/f'social_s{seed}_p{p}_{arm}';personal=batch/f'private_s{seed}_p{p}_{arm}'
        paths.append((social,personal,seed,p,arm))
        for path,names in ((social,['config.json','curve.json','result.json','initial.pt','final.pt','training.jsonl',
            *(f'final_{mode}.npz' for mode in MODES),*(f'protocol_{t:04d}.json' for t in times),
            *(f'protocol_{t:04d}_d{d}.npz' for t in times for d in (0,1))]),
            (personal,['config.json','curve.json','result.json','transferred.pt','initial.pt','final.pt',f'evaluation_{times[-1]:04d}.npz'])):
            absent=[n for n in names if not (path/n).is_file()]
            if absent:missing.append(dict(path=str(path),files=absent))
    if not (batch/'training_complete.json').is_file():missing.append(dict(path=str(batch),files=['training_complete.json']))
    return paths,missing


def pair_stats(reward,success,goals):
    n=len(reward);assert n>0
    natural=np.take_along_axis(success,np.argsort(goals,axis=1),axis=1).astype(np.int64)
    code=natural@np.array([2,1]);values=reward.astype(np.float64)
    return dict(n=n,decisions=2*n,reward_sum=float(values.sum()),mean_reward=float(values.mean()),
        reward_variance=float(values.var()),positive_rewards=int((values>0).sum()),single_correct=int(success.sum()),
        single_accuracy=float(success.astype(np.float64).mean()),both_correct=int(success.prod(1).sum()),
        both_accuracy=float(success.prod(1).astype(np.float64).mean()),food_correct=int(natural[:,0].sum()),
        water_correct=int(natural[:,1].sum()),outcome_counts={key:int((code==i).sum()) for i,key in enumerate(('00','01','10','11'))})


def recompute_social(raw,p):
    n=len(raw['positions']);assert n%120==0
    assert np.array_equal(np.sort(raw['goals'],axis=1),np.tile([0,1],(n,1)))
    assert np.array_equal(np.sort(raw['menu'],axis=2),np.broadcast_to(np.arange(6),(n,2,6)))
    assert not raw['inventory'].any() and not raw['history'].any()
    assert np.all((raw['sent']>=0)&(raw['sent']<7)) and np.all((raw['delivered']>=0)&(raw['delivered']<7))
    place=np.take_along_axis(raw['menu'],raw['action'][:,:,None],axis=2).squeeze(-1)
    assert np.array_equal(place,raw['place'])
    correct=(place==np.take_along_axis(raw['positions'],raw['goals'],axis=1)).astype(np.float32)
    assert np.array_equal(correct,raw['successes'])
    reward=.25*correct.sum(1)+.5*correct.prod(1)
    assert np.array_equal(reward,raw['reward'])
    mids=private.map_index(raw['positions']);assert np.array_equal(private.MAPS[mids],raw['positions'])
    for d in (0,1):assert np.array_equal(np.bincount(mids[raw['scout']==d],minlength=30),np.full(30,n//60))
    def score(mask):return pair_stats(reward[mask],correct[mask],raw['goals'][mask])
    stats=score(np.ones(n,dtype=bool));groups=private.partition_maps(p)
    stats['map_groups']={g:score(np.isin(mids,groups[g])) for g in GROUPS}
    stats['direction_groups']=[{g:score(np.isin(mids,groups[g])&(raw['scout']==d)) for g in GROUPS} for d in (0,1)]
    wh=hashlib.sha256()
    for d in (0,1):
        ix=raw['scout']==d
        for key in ('positions','photo_ids','goals','menu'):wh.update(np.ascontiguousarray(raw[key][ix]).tobytes())
    stats['world_sha256']=wh.hexdigest()
    stats['trace_sha256']=hashlib.sha256(b''.join(np.ascontiguousarray(raw[k]).tobytes() for k in ('sent','delivered','action','place','successes','reward'))).hexdigest()
    stats['sealed_maps_never_in_this_social_training']=True
    return stats


def probability(raw):
    def softmax(x):
        x=np.asarray(x,np.float64);p=np.exp(x-x.max(-1,keepdims=True));return p/p.sum(-1,keepdims=True)
    n=len(raw['positions']);assert raw['first_logits'].shape==(n,7) and raw['second_logits'].shape==(n,7,7)
    first=raw['first_logits'].argmax(-1)
    second=raw['second_logits'][np.arange(n),first].argmax(-1)
    assert np.array_equal(raw['greedy_message'],np.column_stack((first,second)))
    native=(softmax(raw['first_logits'])[:,:,None]*softmax(raw['second_logits'])).reshape(n,49)
    assert np.allclose(native.sum(1),1,atol=1e-12,rtol=0)
    return dict(native=native,greedy=np.eye(49)[first*7+second])


def load_protocol(folder,t,p,photo_ids):
    saved=read(folder/f'protocol_{t:04d}.json');assert [r['direction'] for r in saved]==[0,1]
    result=[]
    for d in (0,1):
        path=folder/f'protocol_{t:04d}_d{d}.npz';raw=load(path);n=len(raw['positions']);assert n==480
        assert np.array_equal(raw['map_ids'],np.repeat(np.arange(30),16))
        assert np.array_equal(raw['positions'],private.MAPS[raw['map_ids']])
        expected=np.tile(np.asarray(list(itertools.product(*photo_ids))),(30,1))
        assert np.array_equal(raw['photo_ids'],expected)
        masks={g:np.isin(raw['map_ids'],private.partition_maps(p)[g]) for g in GROUPS}
        masks['common30']=np.ones(n,dtype=bool)
        probs=probability(raw)
        values={mode:evaluate_groups(probs[mode],raw['positions'],raw['receiver_logits'],masks) for mode in ('native','greedy')}
        nested_equal(values,saved[d]['metrics'],f'{folder.name}.protocol{t}.d{d}')
        result.append(dict(direction=d,metrics={mode:{g:compact(r) for g,r in values[mode].items()} for mode in values},
            raw_sha256=sha(path),greedy_complete_message_count=int(np.unique(raw['greedy_message']@np.array([7,1])).size),
            native_positive_message_count=values['native']['common30']['positive_message_count']))
    return result


def load_run(path,personal,seed,p,arm,times,formal):
    cfg,curve,result=[read(path/n) for n in ('config.json','curve.json','result.json')]
    pcfg,presult=[read(personal/n) for n in ('config.json','result.json')]
    for obj in (cfg,result,pcfg,presult):assert (obj['seed'],obj['partition'],obj['arm'],obj['updates'])==(seed,p,arm,times[-1])
    assert cfg['checkpoints']==list(times) and [r['update'] for r in curve]==list(times)
    assert cfg['training_pool']==private.partition_maps(p)['old'].tolist()
    assert result['frozen_modules_verified'] and result['sealed_never_trained'] and presult['frozen_verified']
    assert cfg['private_outcomes_never_filter_runs'] and presult['all_persons_proceed_to_social_training']
    assert cfg['source_checkpoint']['sha256']==sha(personal/'transferred.pt')==presult['transferred_sha256']
    assert cfg['initial_sha256']==presult['transfer_agent_sha256']
    assert cfg['batch']==512 if formal else True
    assert cfg['eval_n']==9600 if formal else True
    hashes=cfg['source_hashes'];assert hashes==pcfg['source_hashes']
    for name,digest in hashes.items():assert sha(name)==digest
    reconstructed={}
    raw_normal=None
    for mode in MODES:
        raw=load(path/f'final_{mode}.npz');reconstructed[mode]=recompute_social(raw,p)
        nested_equal(reconstructed[mode],result['scores'][mode],path.name+'.'+mode)
        if mode=='normal':
            raw_normal=raw;assert np.array_equal(raw['sent'],raw['delivered'])
        elif mode=='blank':assert not raw['delivered'].any()
        if raw_normal is not None:
            for key in ('positions','photo_ids','goals','menu','inventory','history'):
                assert np.array_equal(raw[key],raw_normal[key])
    nested_equal(curve[-1]['scores']['normal'],reconstructed['normal'])
    worlds={r['scores']['normal']['world_sha256'] for r in curve};assert len(worlds)==1
    protocol=[]
    for row in curve:
        row['protocol']=load_protocol(path,row['update'],p,pcfg['test_photo_ids'])
        protocol.append(dict(update=row['update'],directions=row['protocol']))
        for d in (0,1):
            for g in GROUPS:
                assert row['scores']['normal']['direction_groups'][d][g]['n']==cfg['eval_n']*len(private.partition_maps(p)[g])//60
    # The entire private ability result is recomputed from saved terminal logits.
    praw=load(personal/f'evaluation_{times[-1]:04d}.npz');assert praw['logits'].shape==(2,480,2,6)
    for who in (0,1):
        person=presult['scores']['persons'][who]
        nested_equal(private.capability_statistics(praw['logits'][who],praw,p),person['groups'])
        nested_equal(private.equivariance_statistics(praw['logits'][who],p),person['equivariance'])
    sequence=hashlib.sha256()
    with (path/'training.jsonl').open() as stream:
        for update,line in enumerate(stream,1):
            line=json.loads(line);assert line['update']==update
            assert line['stats']['n']==cfg['batch'] and line['stats']['map_groups']['added']['n']==line['stats']['map_groups']['sealed']['n']==0
            sequence.update((line['stats']['world_sha256']+'\n').encode())
    assert update==times[-1]
    return dict(seed=seed,partition=p,arm=arm,curve=curve,final=reconstructed,protocol=protocol,
        initial_sha256=cfg['initial_sha256'],world_sequence_sha256=sequence.hexdigest(),source_hashes=hashes,
        private_common_initial_sha256=pcfg['common_initial_sha256'],private_initial_head_sha256=pcfg['initial_head_sha256'],
        path=str(path),file_hashes={n:sha(path/n) for n in ('config.json','curve.json','result.json','initial.pt','final.pt','training.jsonl')},
        private_result_sha256=sha(personal/'result.json'))


def point_metrics(row):
    stats=row['scores']['normal'];out={f'normal.{g}.N':float(np.mean([d[g]['both_accuracy'] for d in stats['direction_groups']])) for g in GROUPS}
    out.update({f'normal.{g}.single':float(np.mean([d[g]['single_accuracy'] for d in stats['direction_groups']])) for g in GROUPS})
    out['normal.all.N']=stats['both_accuracy'];out['normal.all.reward']=stats['mean_reward']
    keys={'native_joint_oracle':('native','oracle_joint_map','J'),'Q':('native','stochastic','J'),
        'G':('native','greedy','J'),'N':('greedy','greedy','J'),'native_reward_oracle':('native','oracle_mixed_reward','mixed_reward'),
        'native_reward':('native','stochastic','mixed_reward')}
    for g in (*GROUPS,'common30'):
        for label,(mode,part,key) in keys.items():
            out[f'protocol.{g}.{label}']=float(np.mean([d['metrics'][mode][g][part][key] for d in row['protocol']]))
        out[f'protocol.{g}.joint_entropy']=float(np.mean([d['metrics']['native'][g]['joint_conditional_entropy'] for d in row['protocol']]))
    out['protocol.common30.native_positive_codes']=float(np.mean([d['native_positive_message_count'] for d in row['protocol']]))
    out['protocol.common30.greedy_used_codes']=float(np.mean([d['greedy_complete_message_count'] for d in row['protocol']]))
    return out


def auc(times,values):
    times=np.asarray(times,float);values=np.asarray(values,float)
    assert len(times)==len(values) and len(times)>=2 and np.all(np.diff(times)>0)
    return float(np.sum(np.diff(times)*(values[:-1]+values[1:])*.5)/(times[-1]-times[0]))


def summarize(runs,seeds,partitions,times):
    cells=[]
    for seed,arm in itertools.product(seeds,ARMS):
        chosen=[r for r in runs if r['seed']==seed and r['arm']==arm]
        assert {r['partition'] for r in chosen}==set(partitions)
        curve=[]
        for t in times:
            points=[point_metrics(next(x for x in r['curve'] if x['update']==t)) for r in chosen]
            curve.append(dict(update=t,metrics={k:float(np.mean([x[k] for x in points])) for k in points[0]}))
        endpoint=dict(curve[-1]['metrics'])
        endpoint.update({f'auc.{g}.N':auc(times,[c['metrics'][f'normal.{g}.N'] for c in curve]) for g in GROUPS})
        for mode in MODES:
            for group in (*GROUPS,'all'):
                stats=[r['final'][mode] if group=='all' else r['final'][mode]['map_groups'][group] for r in chosen]
                endpoint[f'intervention.{mode}.{group}.joint']=float(np.mean([s['both_accuracy'] for s in stats]))
                endpoint[f'intervention.{mode}.{group}.reward']=float(np.mean([s['mean_reward'] for s in stats]))
        cells.append(dict(seed=seed,arm=arm,partitions_averaged=len(partitions),directions_per_partition=2,curve=curve,metrics=endpoint))
    aggregate={arm:dict(curve=[dict(update=t,metrics={k:float(np.mean([next(p for p in c['curve'] if p['update']==t)['metrics'][k] for c in cells if c['arm']==arm])) for k in cells[0]['curve'][0]['metrics']}) for t in times],
        metrics={k:float(np.mean([c['metrics'][k] for c in cells if c['arm']==arm])) for k in cells[0]['metrics']}) for arm in ARMS}
    lookup={(r['seed'],r['arm']):r for r in cells}
    differences={k:[lookup[s,'equivariant']['metrics'][k]-lookup[s,'control']['metrics'][k] for s in seeds] for k in cells[0]['metrics']}
    return dict(seed_cells=cells,aggregate=aggregate,seed_differences=differences,
        mean_differences={k:float(np.mean(v)) for k,v in differences.items()},difference_ranges={k:[min(v),max(v)] for k,v in differences.items()})


def plots(result,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
    paths=[];seeds=result['seeds'];personal=result['private'];summary=result['summary']
    def save(fig,name):
        fig.savefig(out/f'{name}.png');fig.savefig(out/f'{name}.pdf');plt.close(fig)
        paths.append(dict(png=str(out/f'{name}.png'),pdf=str(out/f'{name}.pdf')))
    def dots(ax,values,index):
        offsets=np.linspace(-.07,.07,len(values));ax.scatter(index+offsets,values,color='#222222',s=20,zorder=3)
    fig,axes=plt.subplots(1,3,figsize=(12,4.3),constrained_layout=True)
    groups=('old','added','sealed')
    for a,arm in enumerate(ARMS):
        points=[c for c in personal['seed_cells'] if c['arm']==arm]
        for g,group in enumerate(groups):
            x=g+(-.18 if a==0 else .18);key=f'{group}.J'
            axes[0].bar(x,personal['aggregate'][arm][key]*100,width=.32,color=COLORS[arm],label=LABELS[arm] if g==0 else None)
            dots(axes[0],[c['metrics'][key]*100 for c in points],x)
        axes[1].bar(a,personal['aggregate'][arm]['equivariance.all.jsd'],color=COLORS[arm],width=.6)
        dots(axes[1],[c['metrics']['equivariance.all.jsd'] for c in points],a)
        axes[2].bar(a,personal['aggregate'][arm]['all.policy_entropy'],color=COLORS[arm],width=.6)
        dots(axes[2],[c['metrics']['all.policy_entropy'] for c in points],a)
    axes[0].set(xticks=range(3),xticklabels=['Old 18','Added 6','Sealed 6'],ylim=(0,102),ylabel='Greedy joint action success (%)',title='Private action capability')
    axes[0].legend(frameon=False,fontsize=8)
    for ax,title,ylabel in [(axes[1],'Policy consistency over all 720 permutations','Mean JSD (nats)'),(axes[2],'Action uncertainty','Mean branch policy entropy (nats)')]:
        ax.set(xticks=[0,1],xticklabels=['Control','Consistency'],title=title,ylabel=ylabel)
    fig.suptitle('Private heads are discarded before communication; dots are four source-seed means',fontsize=10)
    save(fig,'01_private_capability')
    fig,axes=plt.subplots(1,3,figsize=(12,4.2),sharey=True,constrained_layout=True)
    for ax,g,title in zip(axes,GROUPS,('Old 18: socially trained','Added 6: never trained','Sealed 6: primary never-trained set')):
        for arm in ARMS:
            for seed in seeds:
                cell=next(c for c in summary['seed_cells'] if c['seed']==seed and c['arm']==arm)
                ax.plot(result['times'],[100*c['metrics'][f'normal.{g}.N'] for c in cell['curve']],color=COLORS[arm],alpha=.25,lw=.8)
            ax.plot(result['times'],[100*c['metrics'][f'normal.{g}.N'] for c in summary['aggregate'][arm]['curve']],color=COLORS[arm],lw=2,label=LABELS[arm])
        ax.set(xlabel='Social learning updates',title=title,ylim=(-2,102));ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Greedy sender + greedy receiver: N (%)');axes[1].legend(frameon=False,fontsize=8)
    fig.suptitle('Four source seeds; mean over three partitions and two directions within each seed',fontsize=10)
    save(fig,'02_social_learning')
    fig,axes=plt.subplots(1,3,figsize=(12,4.3),constrained_layout=True)
    for arm in ARMS:
        curves=summary['aggregate'][arm]['curve']
        for key,style,label in [('native_joint_oracle','--','native sender: common-table oracle'),('G','-','native sender + greedy receiver')]:
            axes[0].plot(result['times'],[100*c['metrics'][f'protocol.common30.{key}'] for c in curves],color=COLORS[arm],ls=style,label=LABELS[arm]+': '+label)
        for seed in seeds:
            cell=next(c for c in summary['seed_cells'] if c['seed']==seed and c['arm']==arm)
            axes[1].plot(result['times'],[100*c['metrics']['protocol.common30.N'] for c in cell['curve']],color=COLORS[arm],alpha=.22,lw=.8)
        axes[1].plot(result['times'],[100*c['metrics']['protocol.common30.N'] for c in curves],color=COLORS[arm],lw=2,label=LABELS[arm])
        for seed in seeds:
            cell=next(c for c in summary['seed_cells'] if c['seed']==seed and c['arm']==arm)
            axes[2].plot(result['times'],[c['metrics']['protocol.common30.joint_entropy'] for c in cell['curve']],color=COLORS[arm],alpha=.22,lw=.8)
        axes[2].plot(result['times'],[c['metrics']['protocol.common30.joint_entropy'] for c in curves],color=COLORS[arm],lw=2)
    axes[0].set(title='Native sender: oracle and G',ylabel='Common 30-map joint success (%)',ylim=(-2,102));axes[0].legend(frameon=False,fontsize=6)
    axes[1].set(title='Greedy sender + greedy receiver',ylabel='Common 30-map natural N (%)',ylim=(-2,102));axes[1].legend(frameon=False,fontsize=8)
    axes[2].set(title='Native sender ambiguity',ylabel='H(food, water | message), nats')
    for ax in axes:ax.set_xlabel('Social learning updates');ax.grid(axis='y',alpha=.2)
    fig.suptitle('Exact 49-code analysis on fixed 16 photo pairs; N changes sender policy and is not subtracted from the native oracle',fontsize=9)
    save(fig,'03_common_support_and_natural_use')
    return paths


def report(r):
    s=r['summary'];p=r['private'];end=r['times'][-1]
    lines=['# 空间行动准备与通信形成：分析草稿','',
        f'预定{r["run_count"]}个私人主体对准备和{r["run_count"]}个社会形成全部完成后才进行效应分析。四个新来源种子各自先平均三分区、两方向；不把24个社会运行、48个人、照片或地图数当作独立来源数。主要终点固定{end}步封存6图的自然N；全程梯形AUC除以总时长，作为辅助过程量。','',
        '两臂私人准备均只训练旧18图，看到相同成对世界与动作随机数；等变臂额外使用环境提供的地点对应g，以0.1权重约束两需求行动分布。私人头被丢弃，准备后的memory与slot接口及原DINO/project在社会阶段固定。发信、接收模块从原共同初值学习。该对照检验所测准备的总作用，不单独识别DINO必要性或内部表征的等变性。','',
        '| 条件 | 旧图私人单目标 | 新增图私人J | 封存图私人J | 全30图私人JSD | 全30图私人策略熵 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for arm,m in p['aggregate'].items():lines.append(f"| {LABELS[arm]} | {100*m['old.single']:.2f}% | {100*m['added.J']:.2f}% | {100*m['sealed.J']:.2f}% | {m['equivariance.all.jsd']:.6f} | {m['all.policy_entropy']:.6f} |")
    gates={arm:sum(c['old_gate_count'] for c in p['seed_cells'] if c['arm']==arm) for arm in ARMS}
    lines+=['',f'旧图0.90单目标门槛：control {gates["control"]}/24人，equivariant {gates["equivariant"]}/24人；所有来源均继续社会训练。低JSD也可由无用的均匀策略得到，须与准确率、策略熵、精确并列记录一起看。私人能力评价的是准备接口与私人头整体，不能把它直接叫作已进入消息的能力。','',
        '| 条件 | 社会旧图N | 新增图N | 封存图N（主） | 封存全程AUC |', '| --- | ---: | ---: | ---: | ---: |']
    for arm,a in s['aggregate'].items():
        m=a['metrics'];lines.append(f"| {LABELS[arm]} | {100*m['normal.old.N']:.2f}% | {100*m['normal.added.N']:.2f}% | {100*m['normal.sealed.N']:.2f}% | {100*m['auc.sealed.N']:.2f}% |")
    lines+=['','| equivariant−control | 四个种子差，百分点 | 平均差 |','| --- | --- | ---: |']
    for key in ('normal.sealed.N','normal.added.N','normal.old.N','auc.sealed.N'):
        lines.append(f"| {key} | "+'、'.join(f'{100*v:+.4f}' for v in s['seed_differences'][key])+f" | {100*s['mean_differences'][key]:+.4f} |")
    lines+=['','本批未观察到预定封存N的稳定提高；四个来源种子的差有正有负，也不能称稳定下降。匹配控制的私人未训练地图行动能力本来已经较强，额外对应约束仅造成很小的平均能力差异，不能将社会结果解释成一次强能力操纵对空间能力的普遍否定。','',
        '社会normal是逐符号贪心发信与贪心接收N。固定9600世界的照片从原8张test池抽取，和精确协议评价的固定4×4照片口径不同，不直接相减或把两次测量当独立复制。新增6和封存6均未进入两阶段任务反馈，主指标固定封存组，不选择两组中表现较好者。','',
        '| 条件 | 共同30图原生消息联合MAP | 原生Q | 原生G | 贪心N | 联合条件熵 | 贪心实际使用码数 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for arm,a in s['aggregate'].items():
        m=a['metrics'];pref='protocol.common30.'
        lines.append(f"| {LABELS[arm]} | "+' | '.join(f'{100*m[pref+k]:.2f}%' for k in ('native_joint_oracle','Q','G','N'))+f" | {m[pref+'joint_entropy']:.6f} | {m[pref+'greedy_used_codes']:.2f} |")
    lines+=['','Q对原生49码概率及两次独立行动采样概率求期望；G使用同一原生发信概率和贪心接收；N改为实际逐符号贪心发送。联合MAP参照只适合与同一原生发信口径的Q/G比较。原生非零消息数可能仍是全部49码，不等于模型稳定使用49个区分；实际贪心使用码数另列。common30是一张适用于全部地图的共同表，old/added/sealed单组最优表不能拼接。','',
        '| 条件 | final模式 | 全30图联合成功 | 原封存图联合成功 | 全30图收益 |', '| --- | --- | ---: | ---: | ---: |']
    for arm,a in s['aggregate'].items():
        m=a['metrics']
        for mode in MODES:lines.append(f"| {LABELS[arm]} | {mode} | {100*m[f'intervention.{mode}.all.joint']:.2f}% | {100*m[f'intervention.{mode}.sealed.joint']:.2f}% | {m[f'intervention.{mode}.all.reward']:.6f} |")
    lines+=['','stochastic行是固定随机种子下的实际采样轨迹，不是协议解析中的精确Q；shuffle在同方向和相同首需求组内打乱整条消息，blank固定全零消息，erase_memory移除场景记忆。它们各自衡量当前约定对输入的依赖，不证明消息有自然语言语法。','',
        '终点社会统计已从原始菜单、动作、位置与反馈重算；全部检查点协议从保存首符号/条件次符号及49码接收logits重新计算；私人终点策略与完整720置换测量也从保存logits核验。不新增模型推理，不以能力、封存结果或协议片段选择预算。旧照片属于开发证据。','',
        '能力改善但封存N不改善，可限制该准备在当前接口中的充分性；能力与共同码表信息及N同步变化也不直接识别中介因果。若能力没有按预期改变，社会零效应不证明空间能力不重要。结构片段、拼接或新语言语法不属于本轮新增测量，不能据本报告推断。']
    for fig in r['figures']:lines+=['',f'![空间准备与通信形成]({fig["png"]})']
    return '\n'.join(lines)+'\n'


def self_test():
    assert auc([0,1,4],[0,1,0])==.5
    assert auc([0,2,7],[.25,.25,.25])==.25
    success=np.array([[1,0],[0,1],[1,1],[0,0]],np.float32);goals=np.array([[0,1],[1,0],[1,0],[0,1]])
    stats=pair_stats(.25*success.sum(1)+.5*success.prod(1),success,goals)
    assert stats['both_correct']==1 and stats['single_accuracy']==.5 and stats['food_correct']==3 and stats['water_correct']==1
    with tempfile.TemporaryDirectory() as folder:
        paths,missing=inventory(Path(folder),SEEDS,(1,2,3),TIMES)
        assert len(paths)==24 and len(missing)==49
    return dict(passed=True,checks=['Normalized nonuniform-time trapezoid','Resource-order-aware direct reward recount','Complete24private plus24social andsentinel inventory gate'])


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('batch',type=Path,nargs='?',default=ROOT/'results/spatial_001')
    parser.add_argument('--smoke',action='store_true');parser.add_argument('--self-test',action='store_true');args=parser.parse_args()
    tests=self_test()
    if args.self_test:print(json.dumps(tests));return
    batch=args.batch.resolve();formal=not args.smoke
    if formal:seeds,partitions,times=SEEDS,(1,2,3),TIMES
    else:
        assert batch.name.startswith('smoke_');inv=read(batch/'invocation.json');assert not inv['formal']
        options=inv['args'];seeds=tuple(options['seeds']);partitions=tuple(options['partitions'])
        assert options['social_updates']==options['private_updates']
        times=tuple(sorted({0,options['social_updates'],*(t for t in TIMES if t<options['social_updates'])}))
    entries,missing=inventory(batch,seeds,partitions,times)
    if missing:print(json.dumps(dict(status='pending',missing=missing,result_content_read=False)));return
    complete=read(batch/'training_complete.json');assert complete['status']=='complete' and complete['private_runs']==complete['social_runs']==len(entries)
    runs=[load_run(*entry,times,formal) for entry in entries]
    hashes=runs[0]['source_hashes'];assert all(r['source_hashes']==hashes for r in runs)
    for seed,p in itertools.product(seeds,partitions):
        pair=[r for r in runs if r['seed']==seed and r['partition']==p];assert len(pair)==2
        for key in ('world_sequence_sha256','private_common_initial_sha256','private_initial_head_sha256'):assert pair[0][key]==pair[1][key]
    personal=private.analyze_private(batch,seeds,partitions);assert personal['status']=='complete'
    summary=summarize(runs,seeds,partitions,times)
    result=dict(status='complete' if formal else 'development_smoke',seeds=list(seeds),partitions=list(partitions),times=list(times),run_count=len(runs),
        runs=runs,private=personal,summary=summary,source_hashes=hashes,analysis_sha256=sha(__file__),
        created_utc=datetime.now(timezone.utc).isoformat(),self_tests=tests,
        audit=dict(full_private_and_social_gate_before_scores=True,all_raw_social_endpoints_recomputed=True,
            all_checkpoint_protocols_recomputed=True,private_terminal_policy_and_full_permutations_recomputed=True,
            paired_world_streams_verified=True,no_model_forward=True,no_source_filtering=True),
        primary='2400 sealed normal both_accuracy N; four new seeds, within-seed three partitions/two directions',
        scopes={'social':'9600 balanced worlds with sampled old test photos','protocol':'fixed16 test-photo pairs x30maps; exact49code distributions',
            'private':'fixed16 test-photo pairs x30maps; all720permutations; head discarded'})
    if formal:
        figout=batch/'figures';figout.mkdir(exist_ok=True);result['figures']=plots(result,figout)
        (batch/'空间行动准备与通信形成_分析草稿.md').write_text(report(result))
    name='spatial_analysis.json' if formal else 'analysis_smoke.json'
    (batch/name).write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'],run_count=len(runs),output=str(batch/name),audit=result['audit']),ensure_ascii=False))


if __name__=='__main__':main()

"""Read-only evaluation of the complete fixed-sender receiver diagnostic batch.

Uses saved sender/receiver logits only. No models, training or new forward passes.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
DEPS = ROOT.parent/'redesign_v0.9/.analysis_deps'
if DEPS.exists(): sys.path.insert(0, str(DEPS))
import numpy as np
from receiver_metrics import evaluate

SEEDS = (29101,29102,29103,29104)
PARTITIONS = (1,2,3)
DIRECTIONS = (0,1)
ARMS = ('warm_rl24','warm_ce24','fresh_ce24','fresh_ce30')
TIMES = (0,100,300,600,1200,1800,2400)
GROUPS = ('old','added','sealed','all','train24')
MODES = ('native','greedy')
ORACLES = ('oracle_branch_ce','oracle_joint_map','oracle_mixed_reward')
INFORMATION = ('branch_nll','branch_conditional_entropy','joint_conditional_entropy',
               'conditional_mutual_information','kl_sum')
LABELS = dict(warm_rl24='Warm RL24',warm_ce24='Warm CE24',fresh_ce24='Fresh CE24',fresh_ce30='Fresh CE30*')
COLORS = dict(warm_rl24='#8b6b47',warm_ce24='#367bb0',fresh_ce24='#25927a',fresh_ce30='#9b6fad')
SOURCE = ROOT.parent/'redesign_v0.10/results/generalization_001'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def near(x,y):assert np.isclose(x,y,atol=1e-9,rtol=1e-10),(x,y)
def arrsha(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def softmax(x):
    x=np.asarray(x,dtype=np.float64);z=np.exp(x-x.max(-1,keepdims=True));return z/z.sum(-1,keepdims=True)


def pools(partition):
    pairs=list(itertools.permutations(range(6),2))
    matching=[[(0,1),(2,3),(4,5)],[(0,2),(1,4),(3,5)],[(0,3),(1,5),(2,4)]]
    def ids(k):
        chosen={x for pair in matching[k] for x in (pair,pair[::-1])}
        return [i for i,pair in enumerate(pairs) if pair in chosen]
    added=ids(partition-1);sealed=ids(partition%3);old=sorted(set(range(30))-set(added)-set(sealed))
    return dict(old=old,added=added,sealed=sealed,all=list(range(30)),train24=sorted(old+added))


def expected_files(times,formal):
    names=['config.json','curve.json','result.json','training.jsonl','initial.pt','final.pt',
           'initial_optimizer.pt','final_optimizer.pt']
    names += [f'{kind}_{t:04d}.{ext}' for t in times for kind,ext in
              [('receiver','npz'),('checkpoint','pt'),('optimizer','pt')]]
    if formal:names += ['forensic_2100.pt','forensic_optimizer_2100.pt','train_0001.npz','train_2101.npz']
    return names


def inventory(batch,seeds,partitions,directions,times,formal):
    runs=[];sources=[];missing=[]
    for seed,p,d in itertools.product(seeds,partitions,directions):
        source=batch/f'source_s{seed}_p{p}_d{d}';sources.append((source,seed,p,d))
        names=['config.json','source.pt',*(f'context_{s}.npz' for s in ('fit','dev','evaluation'))]
        absent=[n for n in names if not (source/n).is_file()]
        if absent:missing.append(dict(folder=source.name,files=absent))
        for arm in ARMS:
            path=batch/f's{seed}_p{p}_d{d}_{arm}';runs.append((path,seed,p,d,arm))
            absent=[n for n in expected_files(times,formal) if not (path/n).is_file()]
            if absent:missing.append(dict(folder=path.name,files=absent))
    if not (batch/'training_complete.json').is_file():missing.append(dict(folder='batch',files=['training_complete.json']))
    return runs,sources,missing


def compact(r):
    out={k:v for k,v in r.items() if np.isscalar(v)}
    for key in ('stochastic','greedy',*ORACLES):out[key]={k:v for k,v in r[key].items() if np.isscalar(v)}
    return out


def flatten(r,prefix=''):
    out={}
    for k,v in r.items():
        name=f'{prefix}.{k}' if prefix else k
        if isinstance(v,dict):out.update(flatten(v,name))
        elif isinstance(v,(int,float)) and not isinstance(v,bool):out[name]=float(v)
    return out


def check_nested(actual,expected):
    assert set(actual)==set(expected),(set(actual),set(expected))
    for k,v in actual.items():
        if isinstance(v,dict):check_nested(v,expected[k])
        else:near(v,expected[k])


def distribution(context,probabilities,keep):
    ix=np.isin(context['map_ids'],keep);n=int(ix.sum());assert n>0
    by_map=np.zeros((36,49));pos=context['positions'][ix]
    np.add.at(by_map,pos[:,0]*6+pos[:,1],probabilities[ix]/n)
    return by_map.T.reshape(49,6,6)


def fixed_table_scores(c,actions):
    a=np.asarray(actions,dtype=int);assert a.shape==(49,2)
    m=np.arange(49);f=float(c.sum(2)[m,a[:,0]].sum());w=float(c.sum(1)[m,a[:,1]].sum())
    j=float(c[m,a[:,0],a[:,1]].sum())
    return dict(J=j,single_accuracy=(f+w)/2,food_accuracy=f,water_accuracy=w,mixed_reward=.25*(f+w)+.5*j)


def load_source(path,seed,p,d,reference_out):
    cfg=read(path/'config.json');assert (cfg['seed'],cfg['partition'],cfg['direction'])==(seed,p,d)
    assert cfg['source_state_file_sha256']==sha(path/'source.pt')
    for file,digest in cfg['source_hashes'].items():assert sha(file)==digest
    contexts={};probabilities={};distributions={};reference={};arrays={}
    map_array=np.array(list(itertools.permutations(range(6),2)))
    photo_sets={}
    for split,size in [('fit',16),('dev',6),('evaluation',8)]:
        file=path/f'context_{split}.npz';assert sha(file)==cfg['context_hashes'][split]
        with np.load(file) as z:ctx={k:z[k] for k in z.files}
        n=30*size*size
        assert ctx['map_ids'].shape==(n,) and ctx['positions'].shape==ctx['photo_ids'].shape==(n,2)
        assert np.array_equal(ctx['positions'],map_array[ctx['map_ids']])
        assert np.array_equal(np.bincount(ctx['map_ids'],minlength=30),np.full(30,size*size))
        rows=[[item['feature_row'] for item in group] for group in cfg['photo_split'][split]]
        assert all(len(group)==size for group in rows)
        expected_photos=np.tile(np.array(list(itertools.product(*rows))),(30,1))
        assert np.array_equal(ctx['photo_ids'],expected_photos)
        photo_sets[split]=[set(group) for group in rows]
        assert ctx['first_logits'].shape==(n,7) and ctx['second_logits'].shape==(n,7,7)
        assert np.isfinite(ctx['first_logits']).all() and np.isfinite(ctx['second_logits']).all()
        first=ctx['first_logits'].argmax(-1);second=ctx['second_logits'][np.arange(n),first].argmax(-1)
        assert np.array_equal(ctx['greedy_message'],np.column_stack((first,second)))
        native=(softmax(ctx['first_logits'])[:,:,None]*softmax(ctx['second_logits'])).reshape(n,49)
        greedy=np.eye(49)[first*7+second]
        assert np.allclose(native.sum(1),1,atol=1e-12,rtol=0)
        contexts[split]=ctx;probabilities[split]={'native':native,'greedy':greedy}
        distributions[split]={};reference[split]={}
        support_names=GROUPS if split=='evaluation' else ('train24','all')
        for mode in MODES:
            distributions[split][mode]={};reference[split][mode]={}
            for group in support_names:
                c=distribution(ctx,probabilities[split][mode],pools(p)[group])
                distributions[split][mode][group]=c;arrays[f'{split}_{mode}_{group}']=c
                raw=evaluate(c,np.zeros((49,2,6)))
                reference[split][mode][group]={k:raw[k] for k in
                    ('joint_conditional_entropy','branch_conditional_entropy','food_conditional_entropy',
                     'water_conditional_entropy','conditional_mutual_information','message_probability',*ORACLES)}
    for k in (0,1):
        assert not photo_sets['fit'][k]&photo_sets['dev'][k]
        assert not photo_sets['fit'][k]&photo_sets['evaluation'][k]
        assert not photo_sets['dev'][k]&photo_sets['evaluation'][k]
    fit_transfer={}
    for support in ('train24','all'):
        fit_transfer[support]={}
        for mode in MODES:
            fit_transfer[support][mode]={}
            for oracle in ORACLES:
                actions=reference['fit'][mode][support][oracle]['actions']
                fit_transfer[support][mode][oracle]=dict(actions=actions,fit_scores=fixed_table_scores(distributions['fit'][mode][support],actions),
                    evaluation={g:fixed_table_scores(distributions['evaluation'][mode][g],actions) for g in GROUPS})
    reference_file=reference_out/f'{path.name}_distributions.npz';np.savez_compressed(reference_file,**arrays)
    serial=dict(seed=seed,partition=p,direction=d,key=path.name,path=str(path),config_sha256=sha(path/'config.json'),
        context_hashes=cfg['context_hashes'],source_state_file_sha256=cfg['source_state_file_sha256'],
        source_receiver_sha256=cfg['source_receiver_sha256'],frozen_sender_sha256=cfg['frozen_sender_sha256'],
        references=reference,fit_oracle_transfer=fit_transfer,distributions_file=str(reference_file),distributions_sha256=sha(reference_file))
    return serial,distributions


def load_run(path,seed,p,d,arm,source,distributions,times,formal):
    cfg,curve,result=[read(path/n) for n in ('config.json','curve.json','result.json')]
    for obj in (cfg,result):assert (obj['seed'],obj['partition'],obj['direction'],obj['arm'],obj['updates'])==(seed,p,d,arm,times[-1])
    assert result['complete'] and result['frozen_verified'] and result['frozen_sender_verified']
    assert cfg['batch']==(512 if formal else cfg['batch'])
    assert Path(cfg['source_folder']).resolve()==Path(source['path']).resolve()
    assert cfg['source_config_sha256']==source['config_sha256'] and cfg['frozen_sender_sha256']==source['frozen_sender_sha256']
    assert tuple(cfg['checkpoints'])==tuple(times)==tuple(x['update'] for x in curve)
    assert cfg['extra_supervision']==(arm!='warm_rl24') and cfg['sealed_supervised_exposure']==(arm=='fresh_ce30')
    group='all' if arm=='fresh_ce30' else 'train24'
    assert cfg['training_pool']==pools(p)[group]
    assert cfg['learning_rate']==.0007 and cfg['entropy_off_after']==2100
    assert cfg['map_groups']=={k:pools(p)[k] for k in ('old','added','sealed')}
    assert result['final_support_scores']==curve[-1]['scores']
    if arm.startswith('warm'):assert cfg['initial_sha256']==source['source_receiver_sha256']
    expected_selected=min(curve,key=lambda x:(x['scores']['dev']['branch_nll'],x['update']))['update'] if arm!='warm_rl24' else None
    assert result['selected_dev_update']==expected_selected
    evaluated=[]
    for point in curve:
        t=point['update']
        with np.load(path/f'receiver_{t:04d}.npz') as z:logits=z['logits']
        assert logits.shape==(49,2,6) and np.isfinite(logits).all()
        for split in ('fit','dev'):
            check_nested(compact(evaluate(distributions[split]['native'][group],logits)),point['scores'][split])
        evaluation={mode:{g:compact(evaluate(distributions['evaluation'][mode][g],logits)) for g in GROUPS} for mode in MODES}
        scalars=flatten(evaluation)
        scalars.update({f'{split}.{key}':point['scores'][split][key] for split in ('fit','dev') for key in INFORMATION})
        evaluated.append(dict(update=t,fit=point['scores']['fit'],dev=point['scores']['dev'],evaluation=evaluation,
            metrics=scalars,fit_close_to_branch_entropy=point['scores']['fit']['kl_sum']<=1e-3,
            logits_sha256=sha(path/f'receiver_{t:04d}.npz')))
    training_worlds=hashlib.sha256();messages=hashlib.sha256();exposure=dict(old=0,added=0,sealed=0)
    with (path/'training.jsonl').open() as stream:
        for step,line in enumerate(stream,1):
            row=json.loads(line);assert row['update']==step and sum(row['exposure'].values())==cfg['batch']
            if arm!='fresh_ce30':assert row['exposure']['sealed']==0
            for g in exposure:exposure[g]+=row['exposure'][g]
            training_worlds.update((row['world_sha256']+'\n').encode());messages.update((row['message_sha256']+'\n').encode())
    assert step==times[-1]
    return dict(seed=seed,partition=p,direction=d,arm=arm,path=str(path),source_key=source['key'],
        initial_sha256=cfg['initial_sha256'],frozen_sender_sha256=cfg['frozen_sender_sha256'],
        source_hashes=cfg['source_hashes'],fit_support=group,sealed_supervised_exposure=cfg['sealed_supervised_exposure'],
        selected_dev_update=expected_selected,curve=evaluated,
        selected_dev_point=next((x for x in evaluated if x['update']==expected_selected),None),
        training_exposure=exposure,training_world_sequence_sha256=training_worlds.hexdigest(),
        training_message_sequence_sha256=messages.hexdigest(),
        file_hashes={n:sha(path/n) for n in ('config.json','curve.json','result.json','initial.pt','final.pt','training.jsonl')})


def summarize_points(runs,seeds,partitions,directions,times):
    cells=[];expected=len(partitions)*len(directions)
    for seed,arm,t in itertools.product(seeds,ARMS,times):
        selected=[r for r in runs if r['seed']==seed and r['arm']==arm]
        assert {(r['partition'],r['direction']) for r in selected}==set(itertools.product(partitions,directions))
        points=[next(p for p in r['curve'] if p['update']==t) for r in selected]
        metrics={k:float(np.mean([p['metrics'][k] for p in points])) for k in points[0]['metrics']}
        cells.append(dict(seed=seed,arm=arm,update=t,directions_averaged=expected,metrics=metrics,
            fit_close_count=sum(p['fit_close_to_branch_entropy'] for p in points)))
    aggregate={}
    for arm,t in itertools.product(ARMS,times):
        cs=[c for c in cells if c['arm']==arm and c['update']==t];assert [c['seed'] for c in cs]==list(seeds)
        aggregate.setdefault(arm,{})[str(t)]=dict(seed_count=len(seeds),
            metrics={k:float(np.mean([c['metrics'][k] for c in cs])) for k in cs[0]['metrics']},
            ranges={k:[min(c['metrics'][k] for c in cs),max(c['metrics'][k] for c in cs)] for k in cs[0]['metrics']},
            fit_close_count=sum(c['fit_close_count'] for c in cs),fit_direction_count=expected*len(seeds))
    comparisons={}
    lookup={(c['seed'],c['arm'],c['update']):c for c in cells}
    for left,right in [('warm_ce24','warm_rl24'),('fresh_ce24','warm_ce24')]:
        name=f'{left}_minus_{right}';comparisons[name]={}
        for t in times:
            diff={k:[lookup[s,left,t]['metrics'][k]-lookup[s,right,t]['metrics'][k] for s in seeds] for k in cells[0]['metrics']}
            comparisons[name][str(t)]=dict(seeds=list(seeds),seed_differences=diff,
                mean_differences={k:float(np.mean(x)) for k,x in diff.items()},difference_ranges={k:[min(x),max(x)] for k,x in diff.items()})
    return cells,aggregate,comparisons


def selected_summary(runs,seeds):
    cells=[]
    for seed,arm in itertools.product(seeds,ARMS[1:]):
        selected=[r for r in runs if r['seed']==seed and r['arm']==arm]
        points=[r['selected_dev_point'] for r in selected]
        cells.append(dict(seed=seed,arm=arm,selected_updates=[r['selected_dev_update'] for r in selected],
            metrics={k:float(np.mean([p['metrics'][k] for p in points])) for k in points[0]['metrics']}))
    return dict(seed_cells=cells,aggregate={arm:{k:float(np.mean([c['metrics'][k] for c in cells if c['arm']==arm]))
        for k in cells[0]['metrics']} for arm in ARMS[1:]},selection='Native dev NLL on each arm fit support; earliest checkpoint tie; never evaluation scores')


def summarize_references(sources,references,seeds,partitions,directions):
    """Aggregate scalar reference scores, never combine the per-group action tables."""
    cells=[]
    for seed in seeds:
        selected=[s for s in sources if s['seed']==seed]
        assert {(s['partition'],s['direction']) for s in selected}==set(itertools.product(partitions,directions))
        values=[flatten(dict(baseline=s['baseline'],**references[s['key']])) for s in selected]
        cells.append(dict(seed=seed,directions_averaged=len(selected),
            metrics={k:float(np.mean([v[k] for v in values])) for k in values[0]}))
    return dict(seed_cells=cells,aggregate={k:float(np.mean([c['metrics'][k] for c in cells])) for k in cells[0]['metrics']},
        ranges={k:[min(c['metrics'][k] for c in cells),max(c['metrics'][k] for c in cells)] for k in cells[0]['metrics']},
        note='Scores are averaged within source seed; actions stay separate per source and objective. No group-oracle tables are combined.')


def figures(r,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
    saved=[];times=r['times'];final=str(times[-1]);lookup={(c['seed'],c['arm'],c['update']):c for c in r['seed_cells']}
    def save(fig,name):
        fig.savefig(out/f'{name}.png');fig.savefig(out/f'{name}.pdf');plt.close(fig);saved.append(dict(png=str(out/f'{name}.png'),pdf=str(out/f'{name}.pdf')))
    fig,axes=plt.subplots(1,2,figsize=(11,4.4),constrained_layout=True)
    for ax,key,title,ylabel,mult in [(axes[0],'fit.kl_sum','Fit-support branch NLL gap','Nats/world (symmetric-log scale)',1),
        (axes[1],'native.train24.stochastic.J','Common 24 maps: native sender + random receiver','Expected joint success Q (%)',100)]:
        for arm in ARMS:
            for seed in r['inherited_training_seeds']:
                ax.plot(times,[lookup[seed,arm,t]['metrics'][key]*mult for t in times],color=COLORS[arm],alpha=.22,linewidth=.8)
            ax.plot(times,[r['aggregate'][arm][str(t)]['metrics'][key]*mult for t in times],color=COLORS[arm],linewidth=2.1,label=LABELS[arm])
        ax.set(title=title,xlabel='Receiver fitting updates',ylabel=ylabel);ax.grid(axis='y',alpha=.2)
    axes[0].set_yscale('symlog',linthresh=1e-3);axes[0].axhline(1e-3,color='#777777',ls='--',lw=1)
    axes[1].set_ylim(-2,102);axes[1].legend(frameon=False,fontsize=9)
    fig.suptitle('Four inherited source seeds; CE30* is supervised on all 30 maps, including sealed maps',fontsize=10)
    save(fig,'01_receiver_learning')
    fig,axes=plt.subplots(1,3,figsize=(12,4.2),constrained_layout=True)
    panels=[('native.train24.branch_nll','Branch NLL','Nats/world',1,'native.train24.branch_conditional_entropy'),
            ('native.train24.stochastic.J','Random-receiver joint success Q','Joint success (%)',100,'native.train24.oracle_joint_map.J'),
            ('native.train24.stochastic.mixed_reward','Random-receiver mixed reward','Expected reward',1,'native.train24.oracle_mixed_reward.mixed_reward')]
    for ax,(key,title,ylabel,mult,refkey) in zip(axes,panels):
        ax.bar(range(4),[r['aggregate'][a][final]['metrics'][key]*mult for a in ARMS],color=[COLORS[a] for a in ARMS],alpha=.85,width=.65)
        for i,arm in enumerate(ARMS):
            ax.scatter(i+np.array([-.12,-.04,.04,.12]),[lookup[s,arm,times[-1]]['metrics'][key]*mult for s in r['inherited_training_seeds']],s=18,color='#222222',zorder=3)
        ax.axhline(r['aggregate']['warm_rl24'][final]['metrics'][refkey]*mult,color='#222222',ls='--',lw=1,label='Mean target-specific group oracle')
        ax.set(xticks=range(4),xticklabels=['Warm\nRL24','Warm\nCE24','Fresh\nCE24','Fresh\nCE30*'],title=title,ylabel=ylabel);ax.grid(axis='y',alpha=.2)
    axes[1].set_ylim(0,102);axes[2].set_ylim(0,1.02);axes[0].legend(frameon=False,fontsize=8)
    fig.suptitle('Fixed 2400-update endpoint on common 24-map evaluation support; different objectives have different bounds',fontsize=10)
    save(fig,'02_target_specific_endpoints')
    fig,axes=plt.subplots(1,3,figsize=(11,4.1),sharey=True,constrained_layout=True)
    for ax,group,title in zip(axes,('old','added','sealed'),('Old 18 maps','Added 6 maps','Sealed 6 maps: CE30* was taught these')):
        key=f'native.{group}.stochastic.J'
        ax.bar(range(4),[100*r['aggregate'][a][final]['metrics'][key] for a in ARMS],color=[COLORS[a] for a in ARMS],alpha=.85,width=.65)
        for i,arm in enumerate(ARMS):
            ax.scatter(i+np.array([-.12,-.04,.04,.12]),[100*lookup[s,arm,times[-1]]['metrics'][key] for s in r['inherited_training_seeds']],color='#222222',s=18,zorder=3)
        ax.set(xticks=range(4),xticklabels=['Warm\nRL24','Warm\nCE24','Fresh\nCE24','Fresh\nCE30*'],title=title,ylim=(0,102));ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Native sender + random receiver: Q (%)')
    fig.suptitle('Labels describe source maps; only CE30 exposes the original sealed targets during fitting',fontsize=10)
    save(fig,'03_support_specific_outcomes')
    return saved


def report(r):
    def pct(x):return f'{100*x:.2f}%'
    end=str(r['times'][-1]);lines=['# v0.12 固定发送者接收诊断：分析草稿','',
        '全部96次单方向接收拟合完成，固定发送者为v0.10的24个来源方向。四个继承主体对种子是外层单位，三个分区和两个方向在种子内平均。没有根据评价照片或原封存图选择检查点；主要终点固定为2400步，600步单独列为短预算，监督条件最低开发NLL选点仅作补充。','',
        '三种组合分别记为Q=原生随机发送π＋随机接收、G=π＋贪心接收、N=逐符号贪心发送＋贪心接收。π由首符号和给定首符号的第二符号概率相乘并枚举49码；G仍对π求和。N不使用完整消息联合概率的argmax。全部结果来自已保存logits，不增加模型推理。','',
        'warm_rl24与warm_ce24同接收起点、同24图、同拟合世界和实际消息；CE另获得两个正确地点的监督，并改变损失。fresh_ce24与fresh_ce30同新初始化，后者直接监督原封存6图。fresh_ce30是全支持特权诊断，其原sealed成绩不是未训练组合泛化。原测试8×8照片对已在旧研究使用，仍属开发数据。','',
        '每世界两分支NLL之和的下界是H(F|M)+H(W|M)，比联合条件熵多I(F;W|M)。KL残差还可能包含函数类、有限预算或评价分布迁移，不能全部叫优化失败。分支CE最优边缘的贪心动作、联合MAP和混合收益最优动作分别报告；各分组单独oracle不可拼成一张共同接收表。','',
        '| 条件 | 终点fit KL残差 | 达到1e−3的方向数 | 共同24图评价NLL | Q | G | N | π随机混合收益 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for arm in ARMS:
        s=r['aggregate'][arm][end];m=s['metrics'];lines.append(f"| {LABELS[arm]} | {m['fit.kl_sum']:.6f} | {s['fit_close_count']}/{s['fit_direction_count']} | {m['native.train24.branch_nll']:.4f} | {pct(m['native.train24.stochastic.J'])} | {pct(m['native.train24.greedy.J'])} | {pct(m['greedy.train24.greedy.J'])} | {m['native.train24.stochastic.mixed_reward']:.4f} |")
    lines+=['','1e−3 nats/世界是事前诊断精度标记；只有接近对应fit支持分支下界才称接近该参照。训练平台或较高J本身不证明达到信息极限。RL与CE目标不同，RL未达到CE下界不自动构成其任务优化失败。','',
        '| 条件 | 预算 | old Q | added Q | 原sealed Q | all Q |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for t in (600,2400):
        for arm in ARMS:
            m=r['aggregate'][arm][str(t)]['metrics'];lines.append(f"| {LABELS[arm]} | {t} | "+' | '.join(pct(m[f'native.{g}.stochastic.J']) for g in ('old','added','sealed','all'))+' |')
    lines+=['','以下差按29101、29102、29103、29104顺序列出。主比较warm_ce24−warm_rl24；次比较fresh_ce24−warm_ce24。NLL差的负值表示降低，J或收益差的正值表示提高，不将不同指标差合并成一个优化结论。','',
        '| 比较 | 共同24图指标（2400步） | 四种子差 | 均值 |',
        '| --- | --- | --- | ---: |']
    for name,points in r['comparisons'].items():
        c=points[end]
        for key,label,mult in [('native.train24.branch_nll','分支NLL，nats',1),('native.train24.stochastic.J','Q，百分点',100),('native.train24.stochastic.mixed_reward','混合收益',1),('greedy.train24.greedy.J','N，百分点',100)]:
            vals='、'.join(f'{mult*x:+.4f}' for x in c['seed_differences'][key]);lines.append(f"| {name} | {label} | {vals} | {mult*c['mean_differences'][key]:+.4f} |")
    lines+=['','| 原生发送、共同24图参照 | 数值 |', '| --- | ---: |']
    m=r['aggregate']['warm_rl24'][end]['metrics']
    for key,label in [('branch_conditional_entropy','分支熵下界'),('joint_conditional_entropy','联合条件熵'),('conditional_mutual_information','条件互信息'),('oracle_branch_ce.J','CE最优边缘的贪心J'),('oracle_joint_map.J','联合MAP最优J'),('oracle_mixed_reward.mixed_reward','混合收益最优值')]:
        lines.append(f"| {label} | {m['native.train24.'+key]:.6f} |")
    lines+=['','完整引用表在JSON的references.by_split中：同时保存evaluation各组独立oracle，fit24／fit30支持上的共同动作表，以及保持该表不变在evaluation各组的J和收益。后一项仍是用拟合标签构造的分析者oracle，不是神经接收器；它避免把评价组标签重新拟合的oracle伪装成共同可用策略。原R₀的所有组合指标在sources.baseline中，所有照片划分和联合分布文件均可追溯。reference_summary按来源种子汇总这些分数，动作表仍逐来源分开。','',
        '| 在fit支持求出的共同表，保持不变用于evaluation | old J | added J | 原sealed J | train24 J | all J |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    ref=r['reference_summary']['aggregate']
    for support in ('train24','all'):
        for oracle in ORACLES:
            lines.append(f'| {support} / native / {oracle} | '+' | '.join(pct(ref[f'fit_oracle_transfer.{support}.native.{oracle}.evaluation.{g}.J']) for g in ('old','added','sealed','train24','all'))+' |')
    lines+=['','三种最优表并不回答同一个目标。一个纯解析校准例是：常量消息配均匀10张(a,b)／(b,a)地图，其中b≠a；两边缘均最偏a。CE边缘贪心与混合收益最优表均可选(a,a)，此时J=0、混合收益=0.25；联合MAP选(a,b)可得J=0.1、混合收益=0.20。它只说明任务收益与联合成功可能不同步，并非本轮数据发现。','',
        '| CE条件 | 开发NLL选点后的共同24图评价NLL | Q | 原sealed Q |', '| --- | ---: | ---: | ---: |']
    for arm,m in r['selected_dev_summary']['aggregate'].items():
        lines.append(f"| {LABELS[arm]} | {m['native.train24.branch_nll']:.4f} | {pct(m['native.train24.stochastic.J'])} | {pct(m['native.sealed.stochastic.J'])} |")
    lines+=['','开发选点只使用对应拟合地图支持的6×6开发照片；平局取最早，允许选择0步，所有运行仍完成2400步。它不是主要终点的替代，也不构成独立的新种子。所有逐方向的选择时间和全部7点评价都保存在JSON。','',
        '该诊断借鉴固定发送者后拟合接收者的既有研究；监督标签、任务目标、两分支架构与有限照片口径均与原始工作存在差别。更高监督成绩只表明额外教学可以利用部分固定消息信息，不能称主体自行创造了更好的语言。若探针未达适当参照，仍需分清函数类、训练过程和照片分布，不能只看一个上界差就指定原因。', '',
        '三个24图条件共享实际训练世界和消息；CE30支持不同，只匹配初始化和预算。旧／新增／sealed单独最优的表可能冲突，须同时检查共同24图和全30图参照。所有实际J、单目标和收益均在同一输入c下计量；不同发信政策或照片分布不直接相减。图中粗线／柱为四来源种子均值，细线／散点是各来源种子；目标参照的虚线也是四来源均值，不是每个散点各自的界限。']
    for fig in r['figures']:lines+=['',f'![固定发送者接收诊断]({fig["png"]})']
    return '\n'.join(lines)+'\n'


def self_test():
    pairs=np.array([[0,1],[2,3],[0,1]]);ctx={'positions':pairs,'map_ids':np.array([0,12,0])}
    prob=np.zeros((3,49));prob[:,0]=1;c=distribution(ctx,prob,[0,12]);near(c[0,0,1],2/3);near(c[0,2,3],1/3)
    actions=np.tile([0,1],(49,1));near(fixed_table_scores(c,actions)['J'],2/3)
    # A fit-supported fixed table can fail a group whose own oracle is perfect.
    unseen=np.zeros((49,6,6));unseen[0,2,3]=1;near(fixed_table_scores(unseen,actions)['J'],0)
    near(evaluate(unseen,np.zeros((49,2,6)))['oracle_joint_map']['J'],1)
    star=np.zeros((49,6,6))
    for b in range(1,6):star[0,0,b]=star[0,b,0]=.1
    example=evaluate(star,np.zeros((49,2,6)))
    assert example['oracle_mixed_reward']['actions'][0]==[0,0]
    near(example['oracle_mixed_reward']['J'],0);near(example['oracle_mixed_reward']['mixed_reward'],.25)
    near(example['oracle_joint_map']['J'],.1);near(example['oracle_joint_map']['mixed_reward'],.20)
    first=np.array([[2.,1.,0.,0.,0.,0.,0.]]);second=np.zeros((1,7,7));second[0,1,6]=30
    p=(softmax(first)[:,:,None]*softmax(second)).reshape(1,49)
    assert p.argmax()//7==1 and first.argmax()==0
    fake=[];test_seeds=(1,2);test_times=(0,4)
    for seed,p,d,arm in itertools.product(test_seeds,(1,),(0,1),ARMS):
        curve=[dict(update=t,metrics={'x':float(seed+d+t)},fit_close_to_branch_entropy=False) for t in test_times]
        fake.append(dict(seed=seed,partition=p,direction=d,arm=arm,curve=curve))
    cells,agg,comparisons=summarize_points(fake,test_seeds,(1,),(0,1),test_times)
    near(agg['warm_rl24']['4']['metrics']['x'],6.);assert all(x==0 for c in comparisons.values() for pp in c.values() for x in pp['seed_differences']['x'])
    try:summarize_points(fake[:-1],test_seeds,(1,),(0,1),test_times);raise RuntimeError('Partial cell accepted')
    except AssertionError:pass
    with tempfile.TemporaryDirectory() as tmp:
        _,_,missing=inventory(Path(tmp),SEEDS,PARTITIONS,DIRECTIONS,TIMES,True)
        assert len(missing)==121
    return dict(passed=True,checks=['Uniform world joint distribution','Fit-fixed table differs from group oracle',
        'Sequential-greedy is not global-message argmax','Seed-first aggregation and paired differences',
        'Incomplete cell rejected','96 fits plus24 sources pluscompletion filename gate',
        'Reward-optimal same-site action may have lower joint success than joint MAP'])


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('batch',nargs='?',type=Path,default=ROOT/'results/receiver_001')
    parser.add_argument('--self-test',action='store_true');parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    tests=self_test()
    if args.self_test:print(json.dumps(tests));return
    batch=args.batch.resolve();formal=not args.smoke
    if formal:seeds,partitions,directions,times=SEEDS,PARTITIONS,DIRECTIONS,TIMES
    else:
        assert batch.name.startswith('smoke_'),'--smoke may not open a formal batch'
        invocation=read(batch/'invocation.json');assert not invocation['formal']
        seeds,partitions,directions=tuple(invocation['seeds']),tuple(invocation['partitions']),tuple(invocation['directions'])
        times=tuple(sorted(set([0,invocation['updates']]+[t for t in TIMES if t<invocation['updates']])))
    entries,source_entries,missing=inventory(batch,seeds,partitions,directions,times,formal)
    if missing:
        print(json.dumps(dict(status='pending',expected_runs=len(entries),missing=missing,evaluation_content_read=False,analysis_written=False)));return
    complete=read(batch/'training_complete.json');assert complete['complete'] and complete['runs']==len(entries)
    refout=batch/'analysis_references';refout.mkdir(exist_ok=True)
    sources=[];runs=[];references={}
    for source_path,seed,p,d in source_entries:
        source,distributions=load_source(source_path,seed,p,d,refout)
        for arm in ARMS:
            path=batch/f's{seed}_p{p}_d{d}_{arm}';runs.append(load_run(path,seed,p,d,arm,source,distributions,times,formal))
        selected={r['arm']:r for r in runs if (r['seed'],r['partition'],r['direction'])==(seed,p,d)}
        for arm in ARMS[:3]:
            assert selected[arm]['training_world_sequence_sha256']==selected['warm_rl24']['training_world_sequence_sha256']
            assert selected[arm]['training_message_sequence_sha256']==selected['warm_rl24']['training_message_sequence_sha256']
        assert selected['warm_rl24']['curve'][0]['evaluation']==selected['warm_ce24']['curve'][0]['evaluation']
        assert selected['fresh_ce24']['initial_sha256']==selected['fresh_ce30']['initial_sha256']
        assert selected['fresh_ce24']['curve'][0]['evaluation']==selected['fresh_ce30']['curve'][0]['evaluation']
        source['baseline']=selected['warm_rl24']['curve'][0]['evaluation']
        references[source['key']]=dict(by_split=source.pop('references'),fit_oracle_transfer=source.pop('fit_oracle_transfer'))
        sources.append(source)
    frozen=runs[0]['source_hashes'];assert all(r['source_hashes']==frozen for r in runs)
    assert all(sha(path)==digest for path,digest in frozen.items())
    cells,aggregate,comparisons=summarize_points(runs,seeds,partitions,directions,times)
    result=dict(status='complete' if formal else 'development_smoke',inherited_training_seeds=list(seeds),
        times=list(times),run_count=len(runs),source_direction_count=len(sources),sources=sources,references=references,
        runs=runs,seed_cells=cells,aggregate=aggregate,comparisons=comparisons,selected_dev_summary=selected_summary(runs,seeds),
        reference_summary=summarize_references(sources,references,seeds,partitions,directions),
        self_tests=tests,analysis_sha256=sha(__file__),receiver_metrics_sha256=sha(ROOT/'receiver_metrics.py'),
        source_hashes=frozen,primary_endpoint=2400 if formal else times[-1],short_budget=600 if formal else None,
        combinations={'Q':'native sender + stochastic receiver','G':'native sender + greedy receiver','N':'sequential greedy sender + greedy receiver'},
        inferential_unit='four inherited source pairs; three partitions and two directions averaged within seed' if formal else 'development API check only',
        audit=dict(full_file_inventory_before_evaluation=True,all_fit_dev_curves_recomputed=True,
            matched_three24_worlds_messages=True,paired_fresh_initialization=True,all_source_hashes_verified=True,
            no_model_forward=True,dev_selection_recomputed_without_evaluation=True))
    if formal:
        out=batch/'figures';out.mkdir(exist_ok=True);result['figures']=figures(result,out)
        name='receiver_analysis.json';(batch/'接收诊断分析草稿.md').write_text(report(result))
    else:name='analysis_smoke.json'
    (batch/name).write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'],runs=len(runs),sources=len(sources),output=str(batch/name),audit=result['audit']),ensure_ascii=False))


if __name__=='__main__':main()

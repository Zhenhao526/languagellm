"""Complete finite-support fixed-policy analysis; no model inference or training."""
from __future__ import annotations
import argparse, hashlib, itertools, json, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
TIMES=(0,100,600)
SEEDS=(31101,31102,31103,31104)
METRICS=('variance_matched','variance_permuted','variance_optimal','permutation_minus_matched',
    'role_variance_matched','role_variance_permuted','role_variance_optimal','role_permutation_minus_matched',
    'value_mse_against_mean_target','value_mse_against_optimal','weighted_optimal_distance')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def close(a,b):
    assert np.allclose(a,b,atol=2e-11,rtol=2e-10),(a,b)

def policy_metrics(result):
    worlds=result['world_summaries'];assert len(worlds)==18
    A,B,C,mu2,b,target,ez2,dot=[np.asarray([w[k] for w in worlds],np.float64) for k in
        ('A','B','C','mu_norm_sq','baseline','expected_target','mean_score_norm_sq','mu_dot_mean_score')]
    assert all(np.isfinite(x).all() for x in (A,B,C,mu2,b,target,ez2,dot)) and (A>=0).all()
    assert [w['world_index'] for w in worlds]==list(range(18))
    assert [w['map_id'] for w in worlds]==sorted({w['map_id'] for w in worlds})
    n=len(worlds);opt=np.divide(B,A,out=np.zeros_like(B),where=A>0)
    assert ((A!=0)|(B==0)).all() and ((A!=0)|(C==0)).all() and ((A!=0)|(mu2==0)).all()
    vm=float(np.sum(C-2*b*B+b*b*A-mu2)/n**2)
    vp=float(np.sum(C-2*b.mean()*B+np.mean(b*b)*A-mu2)/n**2)
    vo=float(np.sum(C-np.divide(B*B,A,out=np.zeros_like(B),where=A>0)-mu2)/n**2)
    distance=float(np.sum(A*(b-opt)**2)/n**2)
    row={k:result[k] for k in ('seed','partition','direction','checkpoint')}
    row.update(variance_matched=vm,variance_permuted=vp,variance_optimal=vo,permutation_minus_matched=vp-vm,
        role_variance_matched=vm/4,role_variance_permuted=vp/4,role_variance_optimal=vo/4,
        role_permutation_minus_matched=(vp-vm)/4,weighted_optimal_distance=distance,
        relative_reduction=(vp-vm)/vp if vp>0 else None,
        value_mse_against_mean_target=float(np.mean((b-target)**2)),value_mse_against_optimal=float(np.mean((b-opt)**2)),
        finite_precision_match_variance_correction=float(np.sum(2*b*dot-b*b*ez2)/n**2),
        finite_precision_perm_variance_absolute_correction_bound=float(np.sum(2*abs(b.mean())*abs(dot)+np.mean(b*b)*ez2)/n**2+np.var(b)*ez2.sum()/(n*(n-1))),
        max_expected_score_l2=max(w['expected_score_l2'] for w in worlds),
        zero_A_worlds=int(np.sum(A==0)),near_zero_A_worlds=int(np.sum(A<=1e-12)))
    close(vm-vo,distance)
    for k,v in row.items():
        if k in result['aggregate']:close(v,result['aggregate'][k])
    return row

def summarize(policies,seeds,partitions):
    assert len(policies)==len(seeds)*len(partitions)*2*len(TIMES)
    assert {(r['seed'],r['partition'],r['direction'],r['checkpoint']) for r in policies}==set(itertools.product(seeds,partitions,(0,1),TIMES))
    seed_rows=[];aggregate=[]
    for seed,t in itertools.product(seeds,TIMES):
        rows=[r for r in policies if (r['seed'],r['checkpoint'])==(seed,t)]
        out={k:float(np.mean([r[k] for r in rows])) for k in METRICS}
        out.update(seed=seed,checkpoint=t,policies=len(rows))
        out['relative_reduction']=out['permutation_minus_matched']/out['variance_permuted'] if out['variance_permuted']>0 else None
        seed_rows.append(out)
    for t in TIMES:
        rows=[r for r in seed_rows if r['checkpoint']==t]
        out={k:float(np.mean([r[k] for r in rows])) for k in METRICS}
        out.update(checkpoint=t,sources=len(rows),positive_sources=sum(r['permutation_minus_matched']>0 for r in rows),negative_sources=sum(r['permutation_minus_matched']<0 for r in rows))
        out['relative_reduction']=out['permutation_minus_matched']/out['variance_permuted'] if out['variance_permuted']>0 else None
        aggregate.append(out)
    return seed_rows,aggregate

def analyze(out,smoke=False):
    done=read(out/'measurement_complete.json');assert done['status']=='complete' and done['formal']==(not smoke)
    seeds=(99513,) if smoke else SEEDS;partitions=(1,) if smoke else (1,2,3)
    expected=len(seeds)*len(partitions)*2*len(TIMES)
    assert (done['policies'],done['worlds'],done['message_scores'],done['training_updates'])==(expected,expected*18,expected*18*49,0)
    for path,digest in done['files'].items():assert sha(out/path)==digest,path
    inv=read(out/'invocation.json');assert inv['seeds']==list(seeds) and inv['partitions']==list(partitions) and inv['times']==list(TIMES)
    for path,digest in {**done['source_hashes'],**done['input_hashes']}.items():assert sha(path)==digest,path
    policies=[];sources=[]
    for entry in read(out/'policy_results.json'):
        path=out/entry['file'];assert sha(path)==entry['sha256'];result=read(path)
        assert all(entry[k]==result[k] for k in ('seed','partition','direction','checkpoint'))
        row=policy_metrics(result)
        for w in result['world_summaries']:
            assert sha(out/w['file'])==w['sha256']
            assert w['policy_parameter_count']==43319 and w['policy_tensor_count']==9 and w['message_count']==49
            assert w['expected_score_l2']<=w['expected_score_gate_limit']
            assert w['mu_dot_mean_score']==w['mu_dot_expected_score'] and w['mean_score_norm_sq']==w['expected_score_norm_sq']
        policies.append(row);sources.append(dict(file=entry['file'],sha256=entry['sha256']))
    seed_rows,aggregate=summarize(policies,seeds,partitions)
    return dict(status='complete',formal=not smoke,analysis_sha256=sha(__file__),created_utc=datetime.now(timezone.utc).isoformat(),
        measurement_receipt_sha256=sha(out/'measurement_complete.json'),policy_results_sha256=sha(out/'policy_results.json'),
        independent_recount_used_as_input=False,seeds=list(seeds),partitions=list(partitions),times=list(TIMES),
        policies=policies,seed_rows=seed_rows,aggregate=aggregate,sources=sources,
        max_abs_finite_precision_match_correction=max(abs(r['finite_precision_match_variance_correction']) for r in policies),
        max_finite_precision_perm_correction_bound=max(r['finite_precision_perm_variance_absolute_correction_bound'] for r in policies),
        max_expected_score_l2=max(r['max_expected_score_l2'] for r in policies),zero_A_worlds=sum(r['zero_A_worlds'] for r in policies),
        estimand='Conditional variance trace of the reward-score batch mean on 18 fixed old worlds; sender role weight 1/2, variance multiplier 1/4',
        scope='Same frozen v15 policy: matched baseline vs analytic expectation of uniform within-support permutations; not actual 256-row training batches',
        aggregation='Mean three partitions and two directions within source/time, then mean four sources; relative reduction is ratio of mean variances',
        exclusions=['world/photo sampling','entropy terms and covariance','critic gradients','clipping','Adam','longitudinal learning'],
        uncertainty='Four reused development sources; all timepoints and paired source differences retained; no significance test')

def compare(out):
    analysis=read(out/'analysis.json');reference=read(out/'independent_recount.json');assert reference['passed']
    checks=0;maximum=0.
    def check(a,b):
        nonlocal checks,maximum
        if a is None or b is None:assert a is b
        else:
            close(a,b);maximum=max(maximum,float(np.max(np.abs(np.asarray(a)-np.asarray(b)),initial=0)))
        checks+=1
    for name,keys in [('policies',('seed','partition','direction','checkpoint')),('seed_rows',('seed','checkpoint')),('aggregate',('checkpoint',))]:
        actual={tuple(r[k] for k in keys):r for r in analysis[name]};expected={tuple(r[k] for k in keys):r for r in reference[name]}
        assert actual.keys()==expected.keys()
        for ident,row in expected.items():
            for k,v in row.items():check(actual[ident][k],v)
    for k in ('max_abs_finite_precision_match_correction','max_finite_precision_perm_correction_bound'):check(analysis[k],reference[k])
    receipt=dict(passed=True,comparisons=checks,max_absolute_error=maximum,analysis_sha256=sha(out/'analysis.json'),
        analysis_source_sha256=sha(__file__),independent_recount_sha256=sha(out/'independent_recount.json'),
        scope='All common per-policy, seed/time, aggregate metrics, relative reductions and numerical-correction summaries')
    write(out/'analysis_comparison.json',receipt);return receipt

def figures(data,out):
    sys.path.insert(0,str(ROOT.parent/'redesign_v0.9/.analysis_deps'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(12.5,4.8));fig.subplots_adjust(left=.09,right=.98,bottom=.25,top=.81,wspace=.36)
    styles=[('matched','Matched','#266b99',-.22),('permuted','Permuted expectation','#bc7435',0),('optimal','Optimal baseline','#3f9274',.22)]
    for kind,label,color,offset in styles:
        xs=np.arange(3)+offset
        for j,t in enumerate(TIMES):
            vals=[r[f'role_variance_{kind}'] for r in data['seed_rows'] if r['checkpoint']==t]
            jitter=np.linspace(-.045,.045,len(vals))
            axes[0].scatter(j+offset+jitter,vals,s=25,color=color,alpha=.6,zorder=3)
        means=[r[f'role_variance_{kind}'] for r in data['aggregate']]
        axes[0].plot(xs,means,marker='_',markersize=14,lw=1.3,color=color,label=label)
    axes[0].set_xticks(range(3),[str(t) for t in TIMES]);axes[0].set_xlabel('Saved policy checkpoint')
    axes[0].set_ylabel('Conditional reward-score variance trace\n(sender role weighted by 1/2)')
    axes[0].set_ylim(bottom=0);axes[0].ticklabel_format(axis='y',style='sci',scilimits=(-2,2),useMathText=True)
    axes[0].legend(loc='upper left',bbox_to_anchor=(0,1.18),fontsize=8.6,ncol=2,frameon=False)
    colors=plt.get_cmap('tab10')
    for i,seed in enumerate(data['seeds']):
        rows=[next(r for r in data['seed_rows'] if r['seed']==seed and r['checkpoint']==t) for t in TIMES]
        ys=[100*r['relative_reduction'] if r['relative_reduction'] is not None else np.nan for r in rows]
        axes[1].plot(range(3),ys,'o-',color=colors(i),ms=4,lw=1,label=str(seed))
    axes[1].plot(range(3),[100*r['relative_reduction'] if r['relative_reduction'] is not None else np.nan for r in data['aggregate']],marker='D',color='black',ls='--',ms=4,lw=1.3,label='Ratio of source means')
    axes[1].axhline(0,color='#888888',lw=.8);axes[1].set_xticks(range(3),[str(t) for t in TIMES])
    axes[1].set_xlabel('Saved policy checkpoint');axes[1].set_ylabel('Matched-baseline variance reduction (%)\n(V permuted − V matched) / V permuted')
    axes[1].legend(loc='upper left',bbox_to_anchor=(0,1.18),fontsize=8.3,ncol=3,frameon=False)
    for ax in axes:
        ax.tick_params(axis='y',labelleft=True);ax.grid(axis='y',alpha=.16)
    fig.suptitle('Fixed-policy baseline correspondence on a finite 18-world support',y=.995,fontsize=12)
    note=f"{'DEVELOPMENT CHECK ONLY; ' if not data['formal'] else ''}n = {len(data['seeds'])} source seeds; each source averages {len(data['partitions'])} partitions × 2 directions. Dots: source means.\nOne fixed training-photo pair; no training, entropy term, optimizer, or world-sampling variance. Not the original 256-world training batch."
    fig.text(.09,.055,note,fontsize=8.7,va='bottom')
    dest=out/'figures';dest.mkdir(exist_ok=True)
    fig.canvas.draw()
    for ax in axes:ax.get_xticklabels();ax.get_yticklabels()
    fig.canvas.draw()
    files=[]
    for suffix in ('png','pdf'):
        path=dest/f'01_fixed_policy_variance.{suffix}';fig.savefig(path,dpi=180);files.append(path)
    plt.close(fig)
    write(out/'figure_manifest.json',dict(analysis_sha256=sha(out/'analysis.json'),analysis_source_sha256=sha(__file__),files={str(p.relative_to(out)):sha(p) for p in files}))

def draft(data,out):
    lines=['# 固定策略条件方差：分析草稿','',('正式测量' if data['formal'] else '开发核查，不纳入正式结论')+'；未进行训练。仅估计固定 18 张旧地图及一对训练照片条件下的奖励策略项梯度方差。以下方差已乘发送角色权重平方 1/4。','',
        '| 检查点 | 匹配基线方差 | 排列期望方差 | 最优基线方差 | 匹配相对降低 | 四来源方向 |','|---|---:|---:|---:|---:|---|']
    for r in data['aggregate']:
        ratio='未定义' if r['relative_reduction'] is None else f"{100*r['relative_reduction']:.3f}%"
        lines.append(f"| {r['checkpoint']} | {r['role_variance_matched']:.8g} | {r['role_variance_permuted']:.8g} | {r['role_variance_optimal']:.8g} | {ratio} | {r['positive_sources']} 正 / {r['negative_sources']} 负 |")
    lines+=['','相对降低按方差先在分区/方向内汇总成来源均值，再对四个来源均值取比值；不平均各策略比例。以下保留全部来源。','','| 来源 | 检查点 | 排列减匹配的半角色方差 | 相对降低 |','|---|---:|---:|---:|']
    for r in data['seed_rows']:
        ratio='未定义' if r['relative_reduction'] is None else f"{100*r['relative_reduction']:.3f}%"
        lines.append(f"| {r['seed']} | {r['checkpoint']} | {r['role_permutation_minus_matched']:+.8g} | {ratio} |")
    lines+=['',f"最大 score 均值残差 L2 为 {data['max_expected_score_l2']:.4g}；匹配方差中心修正最大绝对值 {data['max_abs_finite_precision_match_correction']:.4g}，排列方差数值修正绝对值上界最大为 {data['max_finite_precision_perm_correction_bound']:.4g}（这两个修正按未加角色权重方差口径）。零 A 世界数为 {data['zero_A_worlds']}。",'',
        '这些是同一个冻结策略上的基线替换比较，不能直接解释 v0.17 两条独立学习轨迹的最终差异。固定 18 世界支持不同于实际 256 行训练批次；它排除了世界/照片抽样、熵项及协方差、价值回归、裁剪、Adam 与随后伙伴反馈。最优基线是当前奖励 score 方差目标的解析参照，不是已训练或回灌的模型。',
        '', '四个来源继续沿用开发种子；三个时点全部保留，不以时点或正负方向选择结论。经典最优基线公式不构成本项目的新理论，也不能由该方差测量直接推出语言结构或未训练组合能力。','',
        '[结构化分析](analysis.json) · [科学图](figures/01_fixed_policy_variance.png) · [独立比较](analysis_comparison.json)','']
    (out/'固定策略条件方差_分析草稿.md').write_text('\n'.join(lines))

def self_test():
    rows=[]
    for seed,p,d,t in itertools.product(SEEDS,(1,2,3),(0,1),TIMES):
        v=(seed-31100)*(p+d+1)*(t+1);r=dict(seed=seed,partition=p,direction=d,checkpoint=t)
        for k in METRICS:r[k]=float(v)
        r['variance_permuted']=2*v;r['permutation_minus_matched']=v/3
        rows.append(r)
    sr,agg=summarize(rows,SEEDS,(1,2,3));assert len(sr)==12 and len(agg)==3
    assert all(abs(r['relative_reduction']-1/6)<1e-15 for r in sr+agg)
    # Enforce ratio-of-means rather than mean-of-ratios under heterogeneity.
    rows[0]['variance_permuted']*=100
    sr,agg=summarize(rows,SEEDS,(1,2,3));subset=[r for r in rows if r['seed']==31101 and r['checkpoint']==0]
    expected=sum(r['permutation_minus_matched'] for r in subset)/sum(r['variance_permuted'] for r in subset)
    close(sr[0]['relative_reduction'],expected)
    return dict(passed=True,scope='Synthetic inventory and nested source means; ratio of mean variances',analysis_source_sha256=sha(__file__))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path);p.add_argument('--smoke',action='store_true');p.add_argument('--self-test',action='store_true');p.add_argument('--compare-only',action='store_true');p.add_argument('--figures-only',action='store_true');args=p.parse_args()
    if args.self_test:print(json.dumps(self_test()));return
    assert args.out is not None;out=args.out.resolve()
    if args.compare_only:print(json.dumps(compare(out)));return
    if args.figures_only:figures(read(out/'analysis.json'),out);print('FIGURES COMPLETE');return
    assert not (out/'analysis.json').exists(),'refuse silent analysis overwrite'
    data=analyze(out,args.smoke);write(out/'analysis.json',data);draft(data,out);figures(data,out)
    if (out/'independent_recount.json').exists():compare(out)
    print(json.dumps(dict(status='complete',formal=data['formal'],aggregate=data['aggregate']),ensure_ascii=False))
if __name__=='__main__':main()

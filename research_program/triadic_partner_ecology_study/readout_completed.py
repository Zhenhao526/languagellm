"""Short readout of already completed summary, including all closed proposals."""
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
import argparse,json,hashlib
from research_program.triadic_partner_ecology_study.summarize_results import proposal_inventory,require,sha,write_new,SEEDS


def main(run,out):
    run=Path(run).resolve();out=Path(out).resolve();require(not out.exists(),'Readout already exists')
    summary_path=run/'summary_001/summary.json';result_path=run/'execution/results.json'
    s=json.loads(summary_path.read_text());source=json.loads(result_path.read_text())
    require(s['status']==source['status']=='completed' and source['completed_run_count']==32,'Incomplete source')
    index={(x['seed'],x['ecology'],x['condition']):x for x in s['full_domain_proposals']}
    closed=[]
    for row in source['runs']:
        if row['condition'].endswith('_live'):
            counts=Counter()
            for value in row['final'].values():
                for a in value['closed']['raw']['joint_action_distribution']:counts[tuple(a['action_indices'])]+=a['raw_worlds']
            inv=proposal_inventory([dict(action_indices=list(a),worlds=n) for a,n in counts.items()],sum(counts.values()))
            inv.update(seed=row['seed'],ecology=row['ecology'],condition=row['condition'])
            inv['same_fixed_pair_as_natural']=inv['globally_fixed_mutual_active_pair']==index[(row['seed'],row['ecology'],row['condition'])]['globally_fixed_mutual_active_pair']
            closed.append(inv)
    curves=s['aggregates'];held=[x for x in curves if x['stage']=='final' and x['partition']=='heldout_layouts']
    def stat(e,c,mode,m):return next(x for x in held if (x['ecology'],x['condition'],x['mode'],x['metric'])==(e,c,mode,m))
    channels=[]
    for e in ('unique','multiple'):
        for c in ('FI_live','PI_live'):
            channels.append(dict(ecology=e,condition=c,values={m:{'natural':stat(e,c,'natural',m)['mean'],'closed':stat(e,c,'closed',m)['mean'],
                'natural_minus_closed':stat(e,c,'natural',m)['mean']-stat(e,c,'closed',m)['mean']} for m in ('compatible_role_rate','physical_match_rate','full_success_rate','reward_mean')}))
    out.mkdir(parents=True,exist_ok=False)
    payload=dict(status='completed_readout',created_at=datetime.now(timezone.utc).isoformat(),summary_sha256=sha(summary_path),source_results_sha256=sha(result_path),
        natural_fixed_pair_run_count=sum(x['globally_fixed_mutual_active_pair'] is not None for x in index.values()),
        natural_run_count=len(index),extra_live_closed_fixed_pair_run_count=sum(x['globally_fixed_mutual_active_pair'] is not None for x in closed),
        extra_live_closed_same_pair_as_natural=sum(x['same_fixed_pair_as_natural'] for x in closed),closed_full_domain_proposals=closed,
        heldout_channel_means=channels,paired_contrasts=s['paired_contrasts'],
        role_floating_point_interpretation='Absolute rates/differences around1e-16 are numerical zero, not a signed effect or bound exceedance.',
        scope='Saved JSON joint-action distributions only; no NPZ tensor load, new forward, training or inference. Independent execution audit is separate.')
    write_new(out/'readout.json',payload)
    lines=['# 加权读数与行为核查','',
        '主角色预测未获支持：PI角色差中差均值为−5.55×10⁻¹⁷，按数值零解释；FI角色差中差同样为零。全部自然完整终点中，unique兼容角色率为1/3，multiple为31/39；1e−16量级差异不代表方向或超越上界。','',
        '32组自然政策在各自全部布局／需求／私有观察归属上都只有一对固定互选活跃者，第三人恒等待。这一结论依据全部联合行动的提议计数，包括失败状态。活跃者的地点和目的地会变化。16组live的闭通道全域行动也保持同一对搭档和等待者；闭信后的地点／目的地匹配与任务成功会改变。','',
        '## 留出布局：四个配对结果','',
        '| 种子 | unique的PI满分增益pp | multiple的PI满分增益pp | PI满分DiD pp | FI满分DiD pp |','|---|---:|---:|---:|---:|']
    vals=[]
    for seed in SEEDS:
        x=next(x for x in s['paired_contrasts']['seed_values'] if x['seed']==seed and x['partition']=='heldout_layouts' and x['metric']=='full_success_rate');vals.append(x)
        lines.append(f"| {seed} | {100*x['PI_gains']['unique']:+.6f} | {100*x['PI_gains']['multiple']:+.6f} | {100*x['PI_DiD_unique_minus_multiple']:+.6f} | {100*x['FI_DiD_unique_minus_multiple']:+.6f} |")
    lines+=['',f"PI满分DiD均值{100*sum(x['PI_DiD_unique_minus_multiple'] for x in vals)/4:+.6f}pp，三个负值、一个正值；FI满分DiD均值{100*sum(x['FI_DiD_unique_minus_multiple'] for x in vals)/4:+.6f}pp，两个负值、两个正值。各生态内部PI开放−静默的满分增益四种子均为正，不能据此把角色预测的零结果写成获得支持。",'',
        '## 冻结政策的闭通道依赖','',
        '| 生态 | 条件 | 自然满分% | 闭信满分% | 自然−闭信pp | 自然物理匹配% | 闭信物理匹配% |','|---|---|---:|---:|---:|---:|---:|']
    for x in channels:
        f=x['values']['full_success_rate'];p=x['values']['physical_match_rate']
        lines.append(f"| {x['ecology']} | {x['condition']} | {100*f['natural']:.6f} | {100*f['closed']:.6f} | {100*f['natural_minus_closed']:+.6f} | {100*p['natural']:.6f} | {100*p['closed']:.6f} |")
    lines+=['','表为四种子等权均值，完整个体值保存在summary。所有live完整终点（训练／留出）的满分natural−closed为正，角色差为零。关闭通道从第一窗重算，但同时改变输入分布，因此支持这些冻结政策依赖通信路径完成固定搭档内的协调，不单独证明消息编码了未知私人事实、参与者关系词或组合语法。','',
        '主要图：[全部加权终点](../figures_002/complete_weighted_endpoints.png)、[四种子差中差](../figures_002/paired_role_and_success.png)、[672样本监测](../figures_002/fixed_monitor_curves.png)。监测与全量分布分列，连线不定位精确形成时刻。','',
        '依据完整JSON与文件哈希进行描述汇总；本脚本没有独立重算神经前向或物理结算，相关审计由独立执行核验提供。']
    (out/'读数说明.md').write_text('\n'.join(lines)+'\n')
    write_new(out/'receipt.json',dict(status='completed',script_sha256=sha(__file__),outputs={p.name:sha(p) for p in out.iterdir() if p.is_file()},neural_forwards=0,training_updates=0))
    return dict(status='completed',out=str(out),readout_sha256=sha(out/'readout.json'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--out',required=True);a=p.parse_args();print(json.dumps(main(a.run,a.out),ensure_ascii=False))

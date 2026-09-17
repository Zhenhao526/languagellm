"""One pure JSON comparison against the independent completed endpoint audit."""
from pathlib import Path
from collections import Counter
from hashlib import sha256
from itertools import product
from datetime import datetime,timezone
import json,math,statistics

HERE=Path(__file__).resolve().parent
RUN=HERE.parent
TOL=2e-12
counts=Counter();max_error=0.;max_content_error=0.
def read(p):return json.loads(p.read_text())
def sha(p):return sha256(p.read_bytes()).hexdigest()
def eq(x,y,kind):
    assert x==y,(kind,x,y)
    counts[kind]+=1
def close(x,y,kind):
    global max_error,max_content_error
    assert math.isfinite(x) and math.isfinite(y)
    error=abs(x-y);assert error<=TOL,(kind,x,y,error)
    max_error=max(max_error,error)
    if kind.startswith('content_'):max_content_error=max(max_content_error,error)
    counts[kind]+=1

def main():
    output=HERE/'summary_comparison.json';assert not output.exists()
    files=[HERE/'verification.json',RUN/'summary_001/summary.json',RUN/'summary_001/enriched_final_records.json',RUN/'execution/results.json']
    hashes={str(p):sha(p) for p in files}
    assert hashes[str(files[0])]=='7c0eab86edff602ec6def4e54e92cacf3403cb495e39b195594b0185caeb250a'
    assert hashes[str(files[1])]=='95b61d165b43966ac997b77895fea9b081530873874b225336fc5ff911bb6fa8'
    independent,summary,enriched,execution=map(read,files)
    eq(independent['status'],'passed','status');eq(summary['status'],'completed','status')
    eq(sha(files[2]),summary['enriched_final_records_sha256'],'source_sha')
    eq(sha(files[3]),independent['artifacts_sha256'][str(files[3])],'source_sha')
    seeds=(51101,51102,51103,51104);conds=('FI_silent','FI_live','PL_silent','PL_live','LL_silent','LL_live')
    parts=('train','new_needs','new_layouts','new_needs_and_layouts');modes=('natural','closed')
    eq([(r['seed'],r['condition']) for r in enriched],list(product(seeds,conds)),'complete_grid')
    native={(r['seed'],r['condition']):r for r in execution['runs']}
    finals=independent['final_metrics'];cells={}
    for row in enriched:
        seed,cond=row['seed'],row['condition'];eq(set(row['final']),set(parts),'partition_grid')
        for part in parts:
            eq(set(row['final'][part]),set(modes),'mode_grid')
            for mode in modes:
                key=f'{seed}/{cond}/{part}/{mode}';ref=finals[key];actual=row['final'][part][mode]
                eq(actual['source_record'],native[seed,cond]['final'][part][mode],'audited_original_reference')
                w=actual['world_metrics'];content=actual['semantic_metrics']['content']
                eq(w['worlds'],ref['world']['worlds'],'world_denominator')
                eq(w['joint_action_distribution'],ref['world']['raw_joint_action_counts'],'joint_action_counts')
                eq(actual['semantic_metrics']['coverage'],'full_partition','full_endpoint_scope')
                eq(len(actual['semantic_metrics']['saved_background_indices']),ref['content']['backgrounds_per_case'],'background_denominator')
                close(content['macro']['both_endpoints_apt'],ref['content']['both_endpoints_apt'],'content_macro')
                eq(content['case_count'],sum(a['cases'] for a in ref['content']['axes']),'content_case_count')
                for axis in ref['content']['axes']:
                    name=('kind','length','destination')[axis['axis']];x=content['by_axis'][name]
                    close(x['metrics']['both_endpoints_apt'],axis['both_endpoints_apt'],'content_axis')
                    eq(sum(s['case_count'] for s in x['strata'].values()),axis['cases'],'content_axis_count')
                    for stratum in axis['strata']:
                        tag='ABC'[stratum['sender']]+'>'+'ABC'[stratum['listener']];s=x['strata'][tag]
                        close(s['metrics']['both_endpoints_apt'],stratum['both_endpoints_apt'],'content_sender_listener')
                        eq(s['case_count'],stratum['cases'],'content_sender_listener_count')
                for dst,src in [('reward_mean','reward_mean'),('full_success_rate','full_success_rate'),('role_success_rate','role_success_rate'),
                    ('physical_execution_rate','physical_execution_rate'),('expected_reward_given_saved_messages','expected_reward_given_greedy_messages'),('full_probability_given_saved_messages','full_probability_given_greedy_messages')]:
                    close(w[dst],ref['world'][src],'world_common_scalar')
                cells[seed,cond,part,mode]=dict(content_both_endpoints_apt=ref['content']['both_endpoints_apt'],**{k:ref['world'][k] for k in ('reward_mean','full_success_rate','role_success_rate','physical_execution_rate')})
    contrasts={
        'PL_live_minus_LL_live':{'PL_live':1,'LL_live':-1},
        'PL_communication_effect':{'PL_live':1,'PL_silent':-1},
        'LL_communication_effect':{'LL_live':1,'LL_silent':-1},
        'FI_communication_effect':{'FI_live':1,'FI_silent':-1},
        'communication_difference_in_differences':{'PL_live':1,'PL_silent':-1,'LL_live':-1,'LL_silent':1}}
    def paired(saved,values):
        eq([r['seed'] for r in saved['seed_values']],list(seeds),'paired_seed_identity')
        for r,v in zip(saved['seed_values'],values):close(r['difference'],v,'paired_seed_scalar')
        close(saved['equal_seed_mean'],statistics.fmean(values),'paired_mean')
        close(saved['minimum'],min(values),'paired_minimum');close(saved['maximum'],max(values),'paired_maximum')
    for part in parts:
        p=summary['primary_comparison']['partitions'][part]
        eq([r['seed'] for r in p['seed_rows']],list(seeds),'cell_seed_identity')
        for r in p['seed_rows']:
            for cond,mode in product(conds,modes):
                for metric,value in cells[r['seed'],cond,part,mode].items():close(r['cells'][cond][mode][metric],value,'paired_input_cell')
        for metric in next(iter(cells.values())):
            for cond,mode in product(conds,modes):
                values=[cells[s,cond,part,mode][metric] for s in seeds]
                close(p['equal_seed_cell_means'][cond][mode][metric],statistics.fmean(values),'equal_seed_cell_mean')
            for name,coefs in contrasts.items():
                values=[sum(k*cells[s,c,part,'natural'][metric] for c,k in coefs.items()) for s in seeds]
                paired(p['paired_contrasts'][name][metric],values)
            for info in ('FI','PL','LL'):
                values=[cells[s,info+'_live',part,'natural'][metric]-cells[s,info+'_live',part,'closed'][metric] for s in seeds]
                paired(p['natural_minus_closed'][info][metric],values)
    primary=summary['primary_comparison']['primary'];secondary=summary['primary_comparison']['secondary_task_difference_in_differences']
    paired(primary,[r['primary_PL_live_minus_LL_live'] for r in independent['paired_primary']])
    paired(secondary,[r['secondary_full_success_DiD'] for r in independent['paired_primary']])
    eq(counts['content_macro'],192,'scope');eq(counts['content_axis'],576,'scope');eq(counts['content_sender_listener'],3456,'scope')
    for path,digest in hashes.items():eq(sha(Path(path)),digest,'source_unchanged')
    result=dict(status='passed',completed_at=datetime.now(timezone.utc).isoformat(),comparison_source_sha256=sha(Path(__file__)),input_sha256=hashes,
        counts=dict(counts),max_absolute_error=max_error,max_content_absolute_error=max_content_error,tolerance=TOL,
        primary=primary,secondary=secondary,logical_full_endpoint_cells=192,distinct_full_endpoint_npz=144,
        independent_reference='Existing audit uses independent new predicate/feature/MLP/native arithmetic and separately enumerated content cases. This comparison imports no main metrics or summarizer.',
        additional_neural_forwards=0,additional_npz_loads=0,additional_training=0,
        exclusions=['No new independent recalculation of role-case semantic metrics or endpoint-probability content summaries.',
            'Monitor outcomes/hashes were audited; monitor semantic aggregation is outside this scalar comparison.',
            'Site/destination component statistics absent from independent final_metrics are not claimed as independently scalar-matched.'])
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    (HERE/'汇总交叉核对.md').write_text(f'''纯JSON交叉核对通过。以完整独立执行审计的计算为参照，没有导入主metrics／summarizer、再读NPZ、前向或训练。

核对192个逻辑完整末点单元（24组×4分区×自然／关闭通信，其中48单元为静默引用，实际144个NPZ）：两端适切宏值192项、轴值576项、六发送者—听者分层3456项，共4224个内容适切率，以及相应案例／背景分母。另核共同6项世界标量共1152项、全部联合行动计数及原始结果引用。

四分区内，独立参考具备的5项指标全部核对6条件×2模式×4种子的单元、四种子均值、5类配对对比及自然－关闭通信差；主、次预设对比另逐种子复核。最大绝对误差为{max_error:.3g}；4224个内容适切值误差为{max_content_error:.3g}。来源SHA在比较前后不变。

范围：这次标量交叉核对没有独立重算role案例、内容概率质量或监测语义聚合；未覆盖的指标不冒称已逐项独立复核。末点原生结算、全部保存概率、监测记录与SHA已在先前完整执行审计中核过。
''')
    print(json.dumps({k:result[k] for k in ('status','counts','max_absolute_error','max_content_absolute_error')},ensure_ascii=False))

if __name__=='__main__':main()

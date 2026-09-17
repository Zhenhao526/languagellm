"""Read-review receipt and independently bound report/table checks; no model calls.

The semantic findings below record a complete human-readable agent review of
sections 1–7, not a claim that keyword presence proves scientific correctness.
Numeric checks use the previous independent raw recount, not build_report.py.
"""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import hashlib,json,re
from urllib.parse import unquote
import numpy as np
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent;OUT=ROOT/'results/gradient_001'
REPORT=OUT/'固定策略下的通信学习梯度方差研究报告.md'
EXPECTED='6a02d7a768620848652b9b53f76336714ff5896baa148de4aa9e10584f2758d2'

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    text=REPORT.read_text();failures=[];checks=0;numeric=[];links=[]
    def check(ok,name,detail=None):
        nonlocal checks
        checks+=1
        if not bool(ok):failures.append(dict(check=name,detail=detail))
    def shown(token,value,decimals,name,percent=False,signed=False):
        expected=(f'{value*100:+.{decimals}f}%' if signed else f'{value*100:.{decimals}f}%') if percent else f'{value:.{decimals}f}'
        check(token==expected,name,dict(actual=token,expected=expected))
        numeric.append(dict(claim=name,display=token,unrounded=float(value)))
    def near(a,b,name):check(np.allclose(a,b,atol=2e-12,rtol=2e-12),name)
    check(sha(REPORT)==EXPECTED,'review_binds_read_final_report')
    recount=read(OUT/'independent_recount.json');analysis=read(OUT/'analysis.json');numbers=read(OUT/'report_numbers.json')
    execution=read(OUT/'audit_execution.json');gate=read(ROOT/'preflight_qa.json');comparison=read(OUT/'analysis_comparison.json')
    done=read(OUT/'measurement_complete.json');inv=read(OUT/'invocation.json')
    check(all(x['passed'] for x in (recount,execution,gate,comparison)),'successful_execution_math_and_statistical_checks')
    check(numbers['report_sha256']==EXPECTED and numbers['source_recount_sha256']==sha(OUT/'independent_recount.json'),'report_number_receipt_current_bound')
    check(numbers['aggregate']==recount['aggregate'] and numbers['seed_rows']==recount['seed_rows'],'generator_numbers_exactly_same_as_independent_recount')
    check(comparison['analysis_sha256']==sha(OUT/'analysis.json') and comparison['analysis_source_sha256']==sha(ROOT/'analyze_gradient.py') and comparison['independent_recount_sha256']==sha(OUT/'independent_recount.json'),'analysis_comparison_binds_final_sources')
    check(comparison['comparisons']==1514,'analysis_comparison_count')
    table_rows=[[c.strip() for c in line.strip('|').split('|')] for line in text.splitlines() if re.match(r'^\|\s*\d+\s*\|',line)]
    check(len(table_rows)==7,'complete_two_tables_three_times_four_sources')
    agg={r['checkpoint']:r for r in recount['aggregate']}
    seedrows={(r['seed'],r['checkpoint']):r for r in recount['seed_rows']}
    for cells,t in zip(table_rows[:3],(0,100,600)):
        check(int(cells[0])==t,'table1_time_order',t);r=agg[t]
        for col,field in zip((1,2,3),('role_variance_matched','role_variance_permuted','role_variance_optimal')):
            shown(cells[col],r[field],6,f'table1_{t}_{field}')
        shown(cells[4],r['relative_reduction'],2,f'table1_{t}_relative_percent',percent=True)
        check(cells[5]==f"{r['positive_sources']}/4",'table1_source_signs',t)
        near(r['relative_reduction'],(r['variance_permuted']-r['variance_matched'])/r['variance_permuted'],'relative_ratio_of_aggregate_variances')
        for suffix in ('matched','permuted','optimal'):near(r['role_variance_'+suffix],r['variance_'+suffix]/4,'role_loss_half_squared_variance_quarter')
    for cells,seed in zip(table_rows[3:],(31101,31102,31103,31104)):
        check(int(cells[0])==seed,'table2_source_order',seed)
        for i,t in enumerate((0,100,600)):shown(cells[i+1],seedrows[seed,t]['relative_reduction'],2,f'table2_{seed}_{t}',percent=True,signed=True)
    # Check headings' leading numbers against the same independent values.
    lead=text.split('## 1.')[0]
    for t in (600,0,100):check(f"**{agg[t]['relative_reduction']*100:.2f}%**" in lead,'leading_relative_percent',t)
    for key in ('value_mse_against_mean_target','value_mse_against_optimal','weighted_optimal_distance'):
        check(f"{agg[600][key]:.6f}" in text,'body_mse_or_weighted_gap',key)
    near(agg[600]['weighted_optimal_distance'],agg[600]['variance_matched']-agg[600]['variance_optimal'],'unweighted_role_gap_is_weighted_baseline_distance')
    # Recompute the finite-precision sensitivity bounds from all stored moments;
    # this does not rederive all underlying mu/Ez vectors (sample audit does that).
    corrections=[];bounds=[];amin=[];ezmax=[]
    for row in read(OUT/'policy_results.json'):
        r=read(OUT/row['file']);worlds=r['world_summaries'];n=len(worlds)
        b=np.asarray([w['baseline'] for w in worlds]);dot=np.asarray([w['mu_dot_expected_score'] for w in worlds]);e2=np.asarray([w['expected_score_norm_sq'] for w in worlds])
        corrections.append(np.sum(2*b*dot-b*b*e2)/(n*n))
        diagonal_bound=(2*abs(b.mean())*np.sum(np.abs(dot))+np.mean(b*b)*e2.sum())/(n*n)
        permutation_mean_bound=np.var(b)*e2.sum()/(n*(n-1))
        bounds.append(diagonal_bound+permutation_mean_bound)
        amin.extend(w['A'] for w in worlds);ezmax.extend(w['expected_score_l2'] for w in worlds)
    near(max(abs(x) for x in corrections),recount['max_abs_finite_precision_match_correction'],'all_policy_max_center_correction')
    near(max(bounds),recount['max_finite_precision_perm_correction_bound'],'all_policy_max_full_permutation_bound')
    check(f"{max(abs(x) for x in corrections):.3g}" in text and f"{max(bounds):.3g}" in text,'reported_numeric_bounds_exact_display')
    check(min(amin)>1e-12 and analysis['zero_A_worlds']==0,'all_A_above_recorded_degeneracy_threshold')
    check((done['policies'],done['worlds'],done['message_scores'])==(72,1296,63504),'actual_measurement_totals')
    check(inv['seeds']==[31101,31102,31103,31104] and inv['partitions']==[1,2,3] and inv['times']==[0,100,600],'four_inherited_sources_and_all_times')
    check(inv['training_updates']==inv['optimizer_steps']==inv['new_dino_inferences']==done['training_updates']==0,'zero_new_training_optimizer_DINO')
    check([p['feature_row'] for p in inv['photos']]==[0,31] and all(p['split']=='train' for p in inv['photos']),'fixed_train_photo_rows_0_31')
    check(inv['threads']==1 and inv['device']=='cpu','CPU_single_thread_measurement')
    check(f"{done['seconds']:.2f}秒" in text and f"{done['max_rss_platform_units']/1024**2:.1f} MiB" in text,'measurement_only_timing_and_macOS_memory_units')
    check(f"{execution['checks']:,}项" in text and f"{recount['checks']:,}次" in text and f"{recount['max_absolute_comparison_error']:.3g}" in text,'reported_audit_counts_and_error')
    check(execution['counts']['sample_worlds']==4 and gate['counts']['sample_worlds']==18,'declared_full_score_sample_coverage')
    check(execution['script_sha256']==gate['script_sha256']==sha(ROOT/'audit_gradient.py'),'same_preflight_and_formal_auditor_source')
    old=PROJECT/'redesign_v0.17/results/baseline_001';oldnum=read(old/'report_numbers.json')
    oldreport=old/'发送基线与场景对应关系的通信形成研究报告.md'
    check(sha(oldreport)==numbers['old_v17_report_sha256']=='6807c27159c5b110bfeee384ceb011eb8eff12a9d360a5d0ea23c1ce5df77136','v17_frozen_report_not_rewritten')
    for arm in ('matched','shuffled'):check(f"{oldnum['arms'][arm]['normal']['sealed']*100:.2f}%" in text,'historical_v17_natural_success',arm)
    encoder=read(PROJECT/'redesign_v0.4/data/encoder_report.json')
    check(encoder['model']=='official DINOv2 ViT-L/14, backbone only' and encoder['trainable_parameters']==0,'actual_frozen_visual_backbone')
    # Metadata classification is recounted from annotations, not treated as pixels.
    meta=PROJECT/'paper_program/visual_confirmation_v2_water_review';annotation=read(meta/'root_independent_review.json');mqa=read(meta/'final_review_qa.json');packet=read(meta/'packet_qa.json')
    counts=dict(Counter(row['water_text_evidence'] for row in annotation['rows']))
    check(len(annotation['rows'])==79 and counts==annotation['counts']=={'E':28,'U':43,'X':8},'metadata_complete79_descriptive_counts')
    check(mqa['status']=='passed' and mqa['root_annotation_sha256']==sha(meta/'root_independent_review.json'),'metadata_final_clarified_annotation_bound')
    check(all(annotation[k]==0 for k in ('new_usable_images','pixel_requests','new_network_requests','model_calls')) and mqa['source_files_unchanged']==77,'metadata_no_pixels_network_model_or_new_accepted_images')
    check(packet['checks']==1517 and packet['source_file_count']==77,'threshold_metadata_technical_scope')
    for value in (28,43,8):check(f'{value}项' in text,'reported_metadata_label_count',value)
    # Link validation is local; cited primary literature was previously full-text
    # checked and archived, no new web/legal/identity verification is asserted.
    threshold=PROJECT/'paper_program/证据与投稿门槛.md'
    for document in (REPORT,threshold):
        for raw in re.findall(r'!?\[[^\]]*\]\(([^\n]*?)\)',document.read_text()):
            target=raw.strip().strip('<>')
            if target.startswith(('http://','https://','#','mailto:')):continue
            resolved=(document.parent/unquote(target.split('#',1)[0])).resolve() if not Path(target).is_absolute() else Path(unquote(target.split('#',1)[0]))
            check(resolved.exists(),'local_report_or_threshold_link',dict(document=str(document),target=target));links.append(str(resolved))
    history=ROOT/'index_history/paper_program';update=read(history/'门槛更新凭证.json')
    check(update['updated_sha256']==sha(threshold) and update['original_sha256']==sha(history/'证据与投稿门槛.md'),'threshold_update_snapshot_and_current_hash')
    check(update['original_sha256']=='a14d603d832677409f2a5adb152b7c92d9ba9a0f80afd0530c4e73d474d0632a','threshold_original_v17_bytes_preserved')
    figure=read(OUT/'figure_qa.json');rootfig=read(OUT/'root_figure_qa.json')
    check(figure['passed'] and rootfig['passed'] and figure['analysis_sha256']==sha(OUT/'analysis.json'),'final_figure_review_bound_to_analysis')
    for path,digest in figure['files'].items():check(sha(OUT/path)==digest,'reviewed_PNG_PDF_exact_hash',path)
    # Explicit full-text review judgments, tied to final hash, with supporting
    # interpretation and the exact scope inspected. They complement numeric QA.
    reviews=[
      dict(sections='opening, 1, 3, 7',finding='pass',assessment='62.12% is the ratio-of-mean variance reduction at t600, not percentage points. All earlier checkpoints and mixed source signs are retained. Four inherited sources are the replication level; policies/worlds/messages do not increase n.'),
      dict(sections='1–2',finding='pass',assessment='No new learning, optimizer step or DINO forward occurred. Source v15 observer and contemporaneous partner are fixed. Each map uses train feature rows 0/31; independent message and receiver sampling is conditional on fixed worlds. The revised text no longer implies world sampling or saving full mu vectors.'),
      dict(sections='2',finding='pass',assessment='ET2 includes two independent receiver actions sharing the message. Nine policy tensors include both-token shared-coordinate cross terms. The value network is action-independent and excluded from score coordinates. Float32 forward / double logsoftmax / float32 derivative / double moments is not asserted identical to training PRNG bins.'),
      dict(sections='2–4',finding='pass',assessment='The n² and role-weight-square 1/4 factors match the estimand. Optimal b=B/A is an analyst reference and is neither fitted nor transferred. Ordinary unweighted prediction MSE is distinguished from the score-norm-weighted variance gap; absolute changes across policies are not attributed solely to baseline learning.'),
      dict(sections='5',finding='pass',assessment='Matched finite-Ez center correction has sign +2b mu·Ez−b²||Ez||². The permutation absolute bound includes variance of the permutation-dependent conditional mean; Var(b)[n sum||Ez_i||²−||sum Ez_i||²]/[n²(n−1)] is bounded by Var(b)sum||Ez_i||²/[n(n−1)]. This is a sensitivity bound, not PRNG-exact recovery.'),
      dict(sections='5',finding='pass',assessment='All saved moment arrays/source forwards are checked, but complete score/Jacobian verification is limited to development18 and formal4 worlds. Other mu/Ez vectors are not independently rederived. Runtime and peak memory are attributed only to the measurement process.'),
      dict(sections='6',finding='pass',assessment='The 28/43/8 metadata labels are descriptive review annotations, not image acceptance or exclusion. Seventy-nine original representatives and source files are retained; new usable images and pixel/model actions remain zero. Uploaders/institutions do not establish independent photographer identities.'),
      dict(sections='opening, 4, 7, linked next-step note',finding='pass',assessment='Local variance reduction is not equated with the v17 long-run sealed outcome or a causal mediation fraction. Classical score-weighted baselines are not claimed novel. The optimization branch is bounded; next nonlinguistic-capability and image work remain candidates, not completed experiments or ICLR readiness.'),
      dict(sections='mutable submission-threshold index',finding='pass',assessment='Updated only this index after preserving v17 bytes. v18 is complete with finite-support and inherited-n4 limits; v2 metadata review is separated from pixel confirmation. Near-neighbor novelty, independent image/source confirmation, a falsifiable central claim and a coherent reproduction/manuscript remain unmet.'),
    ]
    receipt=dict(passed=not failures,report_sha256=sha(REPORT),created_utc=datetime.now(timezone.utc).isoformat(),script_sha256=sha(__file__),
      checks=checks,failures=failures,report=str(REPORT),numeric_claims=numeric,full_text_review=reviews,
      source_bindings={str(p.relative_to(PROJECT)):sha(p) for p in (OUT/'analysis.json',OUT/'independent_recount.json',OUT/'audit_execution.json',ROOT/'preflight_qa.json',OUT/'analysis_comparison.json',OUT/'report_numbers.json',meta/'root_independent_review.json',meta/'元数据复核报告.md',threshold)},
      local_links_checked=len(links),source_files_not_modified=True,new_training_or_model_inference=0,
      review_scope='Read final full report and linked interpretation/metadata reports; independently checked every numeric table cell and reported precision, numerical correction bounds, all actual local links, lineage and manual scientific inferences. No repeat of full training or complete gradient audit.',
      unverified_scope='No new primary-literature novelty certification, photographer/rights verification, image acceptance, unseen-photo model confirmation or causal mediation estimate.')
    path=OUT/'audit_report_review.json'
    if path.exists():
        previous=OUT/f'audit_report_review_previous_{sha(path)[:12]}.json';previous.write_bytes(path.read_bytes())
    path.write_text(json.dumps(receipt,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    (OUT/'audit_report_gradient_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(dict(passed=receipt['passed'],checks=checks,failures=failures,report_sha256=receipt['report_sha256'],threshold_sha256=sha(threshold)),ensure_ascii=False))
    if failures:raise SystemExit(1)

if __name__=='__main__':main()

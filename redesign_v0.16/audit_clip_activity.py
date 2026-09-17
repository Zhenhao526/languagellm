"""Independent post-outcome log and installed-PyTorch clipping audit; no model fit."""
from pathlib import Path
from itertools import product
from datetime import datetime,timezone
import hashlib,inspect,json
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent;B=ROOT/'results/branches_001'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
record=read(B/'clip_activity_exploratory.json');inv=read(B/'invocation.json');args=inv['args'];checks=[];hashes={};rows=[];pooled={};root_index={(r['seed'],r['partition'],r['arm'],r['person'],r['role'],r['phase']):r for r in record['rows']}
def check(ok,name,context=None):checks.append(dict(check=name,passed=bool(ok),context=context))
check(read(B/'audit_execution.json')['passed'],'full_execution_audit_previously_passed')
check(sha(ROOT/'clip_activity.py')==record['source_sha256'],'root_exploratory_source_bound')
check(torch.__version__==inv['torch'],'installed_torch_matches_executed_version')
for arm in ('neither','policy_only_scale','value_only_scale','both'):
 parent=Path(args['reset_batch']) if arm=='neither' else Path(args['scaled_batch']) if arm=='both' else B
 name='reset' if arm=='neither' else 'reset_scaled' if arm=='both' else arm
 pooled[arm]={role:[] for role in ('sender','receiver')}
 for seed,p in product(args['seeds'],args['partitions']):
  f=parent/f'social_s{seed}_p{p}_{name}/training.jsonl';hashes[str(f)]=sha(f)
  check(record['source_hashes'].get(str(f))==hashes[str(f)],'raw_log_hash_exact',str(f))
  data=[json.loads(s) for s in f.read_text().splitlines()]
  check(len(data)==2400 and all(q['update']==i+1 for i,q in enumerate(data)),'all_2400_updates_in_order',str(f))
  for who,role in product((0,1),('sender','receiver')):
   norms=np.array([q['agents'][who]['gradient_norm_by_role'][role] for q in data],dtype=np.float32)
   check(np.isfinite(norms).all() and (norms>=0).all(),'finite_preclip_norm_sequence',[seed,p,arm,who,role])
   # Use installed Torch tensor arithmetic independently of the root NumPy code.
   coefficient=(2.0/(torch.from_numpy(norms.copy())+1e-6)).clamp(max=1.0).numpy()
   pooled[arm][role].append((seed,norms,coefficient))
   for phase,lo,hi in [('all',0,2400),('entropy_on',0,2100),('entropy_off',2100,2400)]:
    ns=norms[lo:hi];co=coefficient[lo:hi];out=dict(seed=seed,partition=p,arm=arm,person=who,role=role,phase=phase,count=len(ns),active_count=int((co!=np.float32(1)).sum()),rate=float((co!=np.float32(1)).mean()),min_coefficient=float(co.min()),max_norm=float(ns.max()),mean_norm=float(ns.astype(np.float64).mean()))
    ref=root_index[seed,p,arm,who,role,phase]
    check(out==ref,'all_root_per_person_phase_fields_exact',[seed,p,arm,who,role,phase]);rows.append(out)
summary={}
for arm,roles in pooled.items():
 summary[arm]={}
 for role,parts in roles.items():
  norms=np.concatenate([q[1] for q in parts]);coefficients=np.concatenate([q[2] for q in parts])
  value=dict(person_updates=len(norms),active_count=int((coefficients!=1).sum()),rate=float((coefficients!=1).mean()),max_norm=float(norms.max()),min_coefficient=float(coefficients.min()),seed_rates={str(seed):float(np.mean(np.concatenate([c for s,n,c in parts if s==seed])!=1)) for seed in args['seeds']})
  check(value==record['summary'][arm][role],'root_summary_exact',[arm,role]);check(value['person_updates']==57600 and value['active_count']==0 and value['min_coefficient']==1.,'all_logged_updates_have_unit_clip_coefficient',[arm,role]);summary[arm][role]=value
clip_src=inspect.getsource(torch.nn.utils.clip_grad_norm_);scale_src=inspect.getsource(torch.nn.utils.clip_grads_with_norm_);adam_src=inspect.getsource(torch.optim.Adam._init_group)
check(clip_src.index('total_norm = _get_total_norm')<clip_src.index('_clip_grads_with_norm_(parameters')<clip_src.index('return total_norm'),'installed_clip_returns_preclip_norm')
check('clip_coef = max_norm / (total_norm + 1e-6)' in scale_src and 'torch.clamp(clip_coef, max=1.0)' in scale_src,'installed_coefficient_formula')
check('state = self.state[p]' in adam_src and 'state["exp_avg"]' in adam_src and 'state["exp_avg_sq"]' in adam_src,'Adam_moments_keyed_per_parameter')
p=torch.nn.Parameter(torch.zeros(2));p.grad=torch.tensor([3.,4.]);before=p.grad.clone();reported=torch.nn.utils.clip_grad_norm_([p],2.);after=p.grad.clone()
check(float(reported)==5. and torch.linalg.vector_norm(after)<2. and torch.equal(after,before*torch.clamp(2./(reported+1e-6),max=1.)),'synthetic_preclip_return_and_nonunit_scaling')
max_norm=max(v['sender']['max_norm'] for v in summary.values());q=torch.nn.Parameter(torch.zeros(1));q.grad=torch.tensor([max_norm]);qb=q.grad.clone();qreported=torch.nn.utils.clip_grad_norm_([q],2.)
check(torch.equal(q.grad,qb) and float(qreported)==max_norm,'synthetic_observed_max_is_exact_unit_scaling')
run_source=PROJECT/'redesign_v0.13/run_spatial.py';run_text=run_source.read_text()
check("norms[role]=float(torch.nn.utils.clip_grad_norm_(params,2.))" in run_text,'logged_scalar_is_returned_preclip_norm')
check('policy=-(lp*(target-value).detach()).mean()' in run_text,'value_prediction_enters_detached_advantage_weight')
clip_path=Path(inspect.getfile(torch.nn.utils.clip_grad_norm_));adam_path=Path(inspect.getfile(torch.optim.Adam))
result=dict(passed=all(q['passed'] for q in checks),created_utc=datetime.now(timezone.utc).isoformat(),script_sha256=sha(__file__),exploratory_result_sha256=sha(B/'clip_activity_exploratory.json'),check_count=len(checks),checks=checks,summary=summary,person_role_updates_checked=sum(x['person_updates'] for y in summary.values() for x in y.values()),source_hashes=hashes,implementation_files={str(p):sha(p) for p in (clip_path,adam_path,run_source)},installed_source=dict(clip_grad_norm=clip_src,clip_grads_with_norm=scale_src,Adam_init_group=adam_src),synthetic_clip_check=dict(preclip_return=float(reported),actual_postclip_norm=float(torch.linalg.vector_norm(after))),scope='Exploratory after outcomes. Independent complete logged-norm reconstruction using Torch float32 arithmetic; installed implementation/source and synthetic return-value verification. No new training or model forward, and not a replay of every historical gradient tensor.',scientific_assessment=dict(nonunit_shared_clipping_not_active_on_these_observed_trajectories=True,gradient_clipping_function_still_called_with_unit_multiplier=True,scope_all_four_cells_sender_and_receiver=True,existing_frozen_visual_and_disjoint_policy_value_parameters_from_execution_audit=True,Adam_has_per_parameter_moments=True,value_to_policy_detached_advantage_and_subsequent_feedback_paths_remain=True,does_not_establish_advantage_path_as_causal_mediator=True,does_not_claim_clipping_inactive_under_other_data_seeds_budgets_or_counterfactuals=True))
out=B/'clip_activity_independent_qa.json'
if out.exists():out.rename(out.with_name(out.stem+'_'+sha(out)[:12]+'.json'))
out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(passed=result['passed'],checks=len(checks),person_role_updates=result['person_role_updates_checked'],summary=summary,failures=[q for q in checks if not q['passed']]),ensure_ascii=False,indent=2));assert result['passed']

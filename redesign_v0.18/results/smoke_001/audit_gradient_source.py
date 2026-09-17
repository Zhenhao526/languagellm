"""Independent v18 finite-support score/moment audit. No production moment import.

Old checkpoint construction is reused. The float32 score replay is assembled here;
its sensitivity reference is a separately written double Linear/GRU network with
Jacobian and central finite differences. All-array moment recount and selected
full-score checks have deliberately different coverage, recorded in the receipt.
"""
from pathlib import Path
from itertools import product, permutations
from datetime import datetime, timezone
import argparse, hashlib, json, sys, traceback
import numpy as np
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parent; PROJECT=ROOT.parent
sys.path.insert(0,str(PROJECT/'redesign_v0.15'))
import run_scaled as old

NAMES=('send_context.0.weight','send_context.0.bias','send_embedding.weight',
       'send_recur.weight_ih','send_recur.weight_hh','send_recur.bias_ih',
       'send_recur.bias_hh','send_out.weight','send_out.bias')
SHAPES=((96,100),(96,),(7,16),(288,16),(288,96),(288,),(288,),(7,96),(7,))
MAPS=np.asarray(list(permutations(range(6),2)),np.int64)
MATCHINGS=(((0,1),(2,3),(4,5)),((0,2),(1,4),(3,5)),((0,3),(1,5),(2,4)))
MESSAGES=np.asarray(list(product(range(7),repeat=2)),np.int64)
FORMAL_SAMPLES={(31101,1,0,0,0),(31102,2,1,100,8),(31103,3,0,600,17),(31104,1,1,600,8)}
DOUBLE_ATOL=2e-6; DOUBLE_RTOL=2e-4; FD_EPS=1e-5; FD_ATOL=1e-7

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def arrsha(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def state_sha(a):
    d=hashlib.sha256()
    for k,v in a.state_dict().items():d.update(k.encode());d.update(v.detach().cpu().contiguous().numpy().tobytes())
    return d.hexdigest()

def old_maps(p):
    excluded=set()
    for which in ((p-1)%3,p%3):
        for a,b in MATCHINGS[which]:excluded.update(((a,b),(b,a)))
    return np.asarray([i for i,x in enumerate(MAPS) if tuple(x) not in excluded],np.int64)

class Audit:
    def __init__(self):self.checks=0;self.failed=[];self.max_errors={};self.counts={};self.samples=[]
    def check(self,condition,tag,context=None):
        self.checks+=1
        if not bool(condition):self.failed.append(dict(check=tag,context=context))
    def close(self,a,b,tag,context=None,atol=2e-12,rtol=2e-12):
        x=np.asarray(a,dtype=np.float64);y=np.asarray(b,dtype=np.float64)
        error=float(np.max(np.abs(x-y),initial=0)) if x.shape==y.shape else 1e300
        self.max_errors[tag]=max(error,self.max_errors.get(tag,0.))
        self.check(x.shape==y.shape and np.isfinite(x).all() and np.isfinite(y).all() and np.allclose(x,y,atol=atol,rtol=rtol),tag,context)
    def count(self,key,n=1):self.counts[key]=self.counts.get(key,0)+n

def softmax(x):
    x=np.asarray(x,np.float64);e=np.exp(x-x.max(-1,keepdims=True));return e/e.sum(-1,keepdims=True)

def enumerate_targets(logits,pos):
    q=softmax(logits);et=np.zeros(49);et2=np.zeros(49)
    for f,w in product(range(6),repeat=2):
        correct_f=float(f==pos[0]);correct_w=float(w==pos[1])
        t=.25*(correct_f+correct_w)+.5*correct_f*correct_w-1.
        mass=q[:,0,f]*q[:,1,w];et+=mass*t;et2+=mass*t*t
    return q,et,et2

@torch.no_grad()
def replay_receiver(a):
    msg=torch.tensor(MESSAGES)
    embedded=F.embedding(msg,a.receive_embedding.weight).flatten(1)
    inputs=torch.cat((embedded,embedded.new_zeros(49,20)),1)
    hidden=torch.tanh(F.linear(inputs,a.actor[0].weight,a.actor[0].bias))
    return F.linear(hidden,a.actor[2].weight,a.actor[2].bias).reshape(49,2,6).numpy()

@torch.no_grad()
def replay_h(a,projected,photo_ids,pos):
    slots=projected.new_zeros(1,6,65)
    for k in range(2):slots[0,int(pos[k]),:64]=projected[int(photo_ids[k])];slots[0,int(pos[k]),64]=1.
    mixed=torch.einsum('ij,bjd->bid',a.input_transform,slots)
    encoded=torch.tanh(F.linear(mixed,a.slot_phi[0].weight,a.slot_phi[0].bias)).flatten(1)
    seen=torch.cat((encoded,encoded.new_ones(1,1)),1)
    raw=a.memory(seen,seen.new_zeros(1,96))
    return raw*a.visual_scale

def f32_graph(a,h):
    local=torch.cat((h,h.new_zeros(1,4)),1)
    hidden=a.send_context(local)
    first=a.send_out(hidden).reshape(7)
    second=[]
    for token in range(7):
        embedded=a.send_embedding(torch.tensor([token],dtype=torch.int64))
        nxt=a.send_recur(embedded,hidden);second.append(a.send_out(nxt).reshape(7))
    second=torch.stack(second)
    lp1=torch.log_softmax(first.to(torch.float64),-1)
    lp2=torch.log_softmax(second.to(torch.float64),-1)
    return (lp1[:,None]+lp2).flatten(),first,second,lp1,lp2

def manual_double_graph(params,h):
    """GRU equations written independently; no sender modules or moment helpers."""
    w,b,emb,wih,whh,bih,bhh,wo,bo=params
    local=torch.cat((h.to(torch.float64),torch.zeros(1,4,dtype=torch.float64)),1)
    state=torch.tanh(local@w.T+b)
    first=(state@wo.T+bo).reshape(7)
    rows=[]
    for token in range(7):
        gi=emb[token:token+1]@wih.T+bih;gh=state@whh.T+bhh
        ir,iz,inn=gi.chunk(3,1);hr,hz,hn=gh.chunk(3,1)
        reset=torch.sigmoid(ir+hr);update=torch.sigmoid(iz+hz)
        new=torch.tanh(inn+reset*hn)
        next_state=new+update*(state-new)
        rows.append((next_state@wo.T+bo).reshape(7))
    second=torch.stack(rows)
    lp1=first-torch.logsumexp(first,0)
    lp2=second-torch.logsumexp(second,1,keepdim=True)
    return (lp1[:,None]+lp2).flatten(),lp1,lp2,first,second

def sample_scores(a,h,raw,summary,au,identity):
    named=dict(a.named_parameters());params=[named[n] for n in NAMES]
    before=state_sha(a);rng=torch.random.get_rng_state().clone()
    logjoint,first,second,lp1,lp2=f32_graph(a,h)
    actual=[]
    for i in range(49):
        grad=torch.autograd.grad(logjoint[i],params,retain_graph=True)
        actual.append(torch.cat([g.reshape(-1) for g in grad]).detach().numpy().astype(np.float64))
    actual=np.stack(actual)
    au.check(arrsha(actual)==summary['score_matrix_sha256'],'sample_full_score_sha_exact',identity)
    au.close(np.einsum('ij,ij->i',actual,actual),raw['score_norm_sq'],'sample_score_norm',identity,atol=1e-10,rtol=1e-12)
    pi=raw['probability'];et=raw['expected_target'];ez=pi@actual;mu=(pi*et)@actual
    au.close(mu@mu,summary['mu_norm_sq'],'sample_mu_sq',identity)
    au.close(ez@ez,summary['expected_score_norm_sq'],'sample_Ez_sq',identity)
    au.close(np.linalg.norm(ez),summary['expected_score_l2'],'sample_Ez_l2',identity)
    au.close(np.max(np.abs(ez)),summary['expected_score_max_abs'],'sample_Ez_max',identity)
    au.close(mu@ez,summary['mu_dot_expected_score'],'sample_mu_dot_Ez',identity)
    baseline=summary['baseline']
    au.close((mu-baseline*ez)@(mu-baseline*ez),summary['matched_mean_norm_sq_finite_precision'],'sample_finite_precision_mean',identity)
    doubles=tuple(p.detach().double().clone().requires_grad_() for p in params)
    # Functional Jacobian of independently handwritten equations, all nine tensors.
    pieces=torch.autograd.functional.jacobian(lambda *p:manual_double_graph(p,h)[0],doubles,vectorize=False)
    reference=np.concatenate([piece.reshape(49,-1).detach().numpy() for piece in pieces],axis=1)
    per_tensor=[];offset=0
    for name,p in zip(NAMES,params):
        sl=slice(offset,offset+p.numel());offset+=p.numel()
        err=float(np.max(np.abs(reference[:,sl]-actual[:,sl])))
        au.close(reference[:,sl],actual[:,sl],'double_jacobian_vs_float32_score',dict(sample=identity,tensor=name),atol=DOUBLE_ATOL,rtol=DOUBLE_RTOL)
        per_tensor.append(dict(name=name,max_abs=err))
    dbl_joint,dbl_lp1,dbl_lp2,dbl_first,dbl_second=manual_double_graph(doubles,h)
    au.close(np.exp(dbl_joint.detach().numpy()),pi,'double_pi_sensitivity',identity,atol=2e-6,rtol=2e-5)
    # Both factors differentiated separately in shared coordinates. Message (3,3)
    # is fixed before any results, including the later embedding token selection.
    message=24
    grad1=torch.autograd.grad(dbl_lp1[3],doubles,retain_graph=True,allow_unused=True)
    grad2=torch.autograd.grad(dbl_lp2[3,3],doubles,retain_graph=True)
    z1=torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1) for g,p in zip(grad1,doubles)]).detach().numpy()
    z2=torch.cat([g.reshape(-1) for g in grad2]).detach().numpy()
    au.close(z1+z2,reference[message],'token_score_sum',identity)
    cross=float(2*z1@z2)
    au.close(reference[message]@reference[message],z1@z1+z2@z2+cross,'shared_parameter_cross_term',identity)
    # Pre-fixed coordinates cover all nine tensors; embedding uses the prefix row.
    indices=[0,17,3*16+5,21,117,31,63,48,3]
    fd_rows=[];offset=0
    for j,(p,index) in enumerate(zip(doubles,indices)):
        index=min(index,p.numel()-1);original=float(p.reshape(-1)[index].detach())
        with torch.no_grad():
            p.reshape(-1)[index]=original+FD_EPS;plus=float(manual_double_graph(doubles,h)[0][message])
            p.reshape(-1)[index]=original-FD_EPS;minus=float(manual_double_graph(doubles,h)[0][message])
            p.reshape(-1)[index]=original
        fd=(plus-minus)/(2*FD_EPS);expected=float(reference[message,offset+index]);offset+=p.numel()
        au.close(fd,expected,'manual_double_finite_difference',dict(sample=identity,tensor=NAMES[j],index=index),atol=FD_ATOL,rtol=1e-5)
        fd_rows.append(dict(name=NAMES[j],index=index,derivative=expected,finite_difference=fd,absolute_error=abs(fd-expected)))
    au.check(state_sha(a)==before and torch.equal(rng,torch.random.get_rng_state()) and all(p.grad is None for p in a.parameters()),'sample_no_weight_grad_or_rng_mutation',identity)
    au.samples.append(dict(identity=identity,full_score_sha256=arrsha(actual),parameters=actual.shape[1],messages=49,double_tensor_errors=per_tensor,finite_differences=fd_rows,token_cross_term_message_24=cross))
    au.count('sample_worlds');au.count('independently_replayed_full_message_scores',49)

def independent_aggregate(worlds):
    n=len(worlds);A,B,C,mu2,b=[np.asarray([w[k] for w in worlds],np.float64) for k in ('A','B','C','mu_norm_sq','baseline')]
    mean=float(b.mean());second=float(np.mean(b*b));positive=A>0;opt=np.zeros(n);opt[positive]=B[positive]/A[positive]
    vm=float(np.sum(C-2*b*B+b*b*A-mu2)/(n*n))
    vp=float(np.sum(C-2*mean*B+second*A-mu2)/(n*n))
    vo=float(np.sum(C-2*opt*B+opt*opt*A-mu2)/(n*n))
    distance=float(np.sum(A*(b-opt)**2)/(n*n))
    perm_dist=float(np.sum(A*((mean-opt)**2+float(np.var(b))))/(n*n))
    scale=max(1.,float(np.sum(np.abs(C)+2*np.abs(b*B)+b*b*A+mu2)/(n*n)),abs(vp));tol=1e-9*scale
    return dict(world_count=n,baseline_mean=mean,baseline_second_moment=second,baseline_std=float(np.std(b)),
      variance_matched=vm,variance_permuted=vp,variance_optimal=vo,permutation_minus_matched=vp-vm,
      matched_excess_over_optimal=vm-vo,permutation_excess_over_optimal=vp-vo,
      weighted_optimal_distance=distance,permutation_weighted_optimal_distance=perm_dist,
      optimal_identity_residual=vm-vo-distance,permutation_optimal_identity_residual=vp-vo-perm_dist,
      role_weight=.5,role_variance_multiplier=.25,role_variance_matched=vm/4,
      role_variance_permuted=vp/4,role_variance_optimal=vo/4,role_permutation_minus_matched=(vp-vm)/4,
      zero_A_count=int((A==0).sum()),near_zero_A_count=int((A<=1e-12).sum()),
      optimal_baselines=[float(v) if pos else None for v,pos in zip(opt,positive)],
      max_expected_score_l2=max(w['expected_score_l2'] for w in worlds),
      max_expected_score_relative=max(w['expected_score_relative'] for w in worlds),numeric_tolerance=tol)

def synthetic_test(au):
    # Actual normalized categorical score vectors, nonuniform messages, stochastic
    # receiver actions and nonconstant baselines. Enumerate the entire joint draw,
    # not merely reuse the analytic per-world variance formula.
    worlds=[];outcomes=[];baselines=[-.1,-.9,.25]
    for i,p in enumerate((.2,.45,.7)):
        pi=np.asarray([p,1-p]);z=np.asarray([[1-p,-(1-p)],[-p,p]])
        items=[]
        for m,f,w in product(range(2),repeat=3):
            pf=.15+.2*i+.07*m;pw=.8-.15*i-.11*m
            probability=pi[m]*(pf if f else 1-pf)*(pw if w else 1-pw)
            t=.25*(f+w)+.5*f*w-1.;items.append((probability,t,z[m]))
        outcomes.append(items)
        mu=sum(prob*t*score for prob,t,score in items)
        worlds.append(dict(A=sum(prob*(score@score) for prob,t,score in items),
          B=sum(prob*t*(score@score) for prob,t,score in items),C=sum(prob*t*t*(score@score) for prob,t,score in items),
          mu_norm_sq=float(mu@mu),baseline=baselines[i],expected_score_l2=0.,expected_score_relative=0.))
    agg=independent_aggregate(worlds);variances=[];means=[]
    for order in permutations(range(3)):
        mean=np.zeros(2);second=0.
        for draws in product(*outcomes):
            prob=np.prod([d[0] for d in draws]);vector=sum((draws[i][1]-baselines[order[i]])*draws[i][2] for i in range(3))/3
            mean+=prob*vector;second+=prob*float(vector@vector)
        means.append(mean);variances.append(second-float(mean@mean))
    au.close(variances[0],agg['variance_matched'],'synthetic_full_enumeration_matched')
    au.close(np.mean(variances),agg['variance_permuted'],'synthetic_all_permutations_and_actions')
    au.close(np.stack(means),np.repeat(means[0][None],6,0),'synthetic_baseline_independent_mean')
    au.close(np.mean(variances)/4,agg['role_variance_permuted'],'synthetic_role_half_variance_quarter')
    au.close(agg['matched_excess_over_optimal'],agg['weighted_optimal_distance'],'synthetic_optimal_distance')
    au.check(all(-1<=x<=0 for x in agg['optimal_baselines']),'synthetic_variance_optimal_baseline_target_range')
    zero=dict(A=0.,B=0.,C=0.,mu_norm_sq=0.,baseline=9.,expected_score_l2=0.,expected_score_relative=0.)
    za=independent_aggregate([zero]);au.check(za['zero_A_count']==1 and za['variance_matched']==0 and za['optimal_baselines']==[None],'synthetic_zero_score_degenerate')
    au.count('synthetic_complete_joint_draws',6*8**3)


def audit_batch(out,au):
    inv=read(out/'invocation.json');done=read(out/'measurement_complete.json');formal=bool(inv['formal'])
    source=Path(inv['source']);expected_source=PROJECT/('redesign_v0.15/results/scaled_001' if formal else 'redesign_v0.15/results/smoke_002')
    au.check(source==expected_source and done['status']=='complete' and done['formal']==formal,'complete_and_preselected_source')
    seeds=[31101,31102,31103,31104] if formal else [99513];parts=[1,2,3] if formal else [1]
    au.check(inv['seeds']==seeds and inv['partitions']==parts and inv['times']==[0,100,600] and inv['directions']==[0,1],'fixed_factorial_checkpoint_list')
    inherited=read(source/'invocation.json')['source_hashes'];own={str(ROOT/n):sha(ROOT/n) for n in ('run_gradient.py','gradient_moments.py','固定执行方案.md','前置审查.md')};expected_hashes={**inherited,**own}
    au.check(inv['source_hashes']==expected_hashes==done['source_hashes'],'exact_frozen_source_hashes')
    for path,digest in expected_hashes.items():
        au.check(sha(path)==digest and sha(out/'frozen_sources'/Path(path).relative_to(PROJECT))==digest,'frozen_source_current_and_archive',path)
    au.check(inv['inputs']==done['input_hashes'],'input_receipt_consistent')
    for path,digest in inv['inputs'].items():au.check(sha(path)==digest,'consumed_input_unchanged',path)
    if formal:
        manifest=read(source/'completion_manifest.json');au.check(manifest['status']=='complete','source_v15_manifest_complete')
        for path,digest in inv['inputs'].items():
            p=Path(path)
            if p.is_relative_to(source) and p.name!='completion_manifest.json':au.check(manifest['artifacts'].get(str(p.relative_to(PROJECT)))==digest,'input_bound_to_v15_original_manifest',path)
        gate=inv['preflight'];au.check(sha(gate['path'])==gate['sha256'] and read(gate['path'])['passed'] and read(gate['path'])['source_hashes']==expected_hashes,'successful_unchanged_preflight_gate')
    for name,digest in done['files'].items():au.check(sha(out/name)==digest,'measurement_completion_file_hash',name)
    au.check(all(inv[k]==0 for k in ('training_updates','optimizer_steps','new_dino_inferences')) and done['training_updates']==0,'no_training_optimizer_or_DINO_inference')
    entries=read(PROJECT/'redesign_v0.4/data/manifest.json')['images']
    photos=[min(i for i,x in enumerate(entries) if x['split']=='train' and x['category']==category) for category in ('food','water')]
    au.check([x['feature_row'] for x in inv['photos']]==photos and all(x['split']=='train' for x in inv['photos']),'preselected_minimum_original_train_photo_rows')
    bank=old.ImageBank();all_rows=read(out/'policy_results.json');au.check(len(all_rows)==len(seeds)*len(parts)*6,'full_policy_result_count')
    rows_by_key={(r['seed'],r['partition'],r['direction'],r['checkpoint']):r for r in all_rows}
    au.check(len(rows_by_key)==len(all_rows),'unique_policy_records')
    layout=[dict(name=k,shape=list(shape),numel=int(np.prod(shape))) for k,shape in zip(NAMES,SHAPES)]
    for seed,p,t in product(seeds,parts,(0,100,600)):
        checkpoint=source/f'social_s{seed}_p{p}_reset_scaled/checkpoint_{t:04d}.pt'
        states=torch.load(checkpoint,weights_only=True);prepared=torch.load(source/f'prepared_{seed}.pt',weights_only=True)
        agents=old.scaled.remake_scaled_agents(seed,prepared,states)
        for agent in agents:
            agent.requires_grad_(False)
            for name,param in agent.named_parameters():param.requires_grad_(name in NAMES)
            agent.train()
        snapshots=[state_sha(a) for a in agents]
        with torch.no_grad():projected=[a.project(bank.features).detach() for a in agents]
        maps=old_maps(p);au.check(len(maps)==18 and set(maps)==set(old.v13.private.partition_maps(p)['old']),'independent_old18_partition',(seed,p))
        for who in (0,1):
            key=(seed,p,who,t);top=rows_by_key[key];folder=out/f's{seed}_p{p}_d{who}_t{t:04d}'
            result=read(folder/'result.json');au.check(top['file']==str((folder/'result.json').relative_to(out)) and sha(folder/'result.json')==top['sha256'],'policy_result_hash',key)
            au.check(result['source_checkpoint']==str(checkpoint) and result['source_checkpoint_sha256']==sha(checkpoint) and result['sender_state_sha256']==snapshots[who] and result['partner_state_sha256']==snapshots[1-who],'loaded_exact_original_pair_checkpoint',key)
            agent=agents[who];receiver=replay_receiver(agents[1-who]);summaries=result['world_summaries']
            named=[(n,pv) for n,pv in agent.named_parameters() if pv.requires_grad]
            au.check([n for n,pv in named]==list(NAMES) and [tuple(pv.shape) for n,pv in named]==list(SHAPES) and sum(pv.numel() for n,pv in named)==43319,'exact_actual_policy_coordinates',key)
            au.check(len(summaries)==18,'all18_world_summaries',key)
            for wi,(mid,summary) in enumerate(zip(maps,summaries)):
                identity=(*key,wi);path=folder/f'world_{wi:02d}.npz'
                au.check(summary['world_index']==wi and summary['map_id']==int(mid) and summary['file']==str(path.relative_to(out)) and summary['sha256']==sha(path),'world_identity_file_hash',identity)
                with np.load(path) as z:raw={k:z[k] for k in z.files}
                au.check(np.array_equal(raw['message_ids'],MESSAGES) and np.array_equal(raw['positions'],MAPS[mid]) and int(raw['map_id'])==mid and int(raw['world_index'])==wi and np.array_equal(raw['photo_ids'],photos),'world_legal_fixed_support_and_49_message_order',identity)
                au.check(not any(k in raw for k in ('score_matrix','scores','mu','mean_score')),'no_formal_large_score_vector_output',identity)
                au.close(raw['receiver_logits'],receiver,'all_worlds_receiver_logit_source_replay',identity,atol=0,rtol=0)
                h=replay_h(agent,projected[who],photos,MAPS[mid])
                au.close(raw['h'],h.numpy(),'all_worlds_private_observer_replay',identity,atol=0,rtol=0)
                with torch.no_grad():joint,first,second,_,_=f32_graph(agent,h);baseline=float(agent.send_value(torch.cat((h,h.new_zeros(1,4)),1)).squeeze())
                au.close(raw['first_logits'],first.numpy(),'all_worlds_first_token_logits',identity,atol=0,rtol=0)
                au.close(raw['second_logits'],second.numpy(),'all_worlds_all_seven_prefix_logits',identity,atol=0,rtol=0)
                pi=(softmax(raw['first_logits'])[:,None]*softmax(raw['second_logits'])).flatten()
                q,et,et2=enumerate_targets(raw['receiver_logits'],MAPS[mid]);norm=raw['score_norm_sq']
                au.close(raw['probability'],pi,'all_worlds_native_joint_probability',identity)
                au.close(raw['receiver_probabilities'],q,'all_worlds_receiver_categorical_probability',identity)
                au.close(raw['expected_target'],et,'all_worlds_36_action_ET',identity)
                au.close(raw['expected_target_sq'],et2,'all_worlds_36_action_ET2',identity)
                au.close([raw['baseline'],summary['baseline']],[baseline,baseline],'all_worlds_original_action_independent_value',identity,atol=0,rtol=0)
                A=float(pi@norm);B=float((pi*et)@norm);C=float((pi*et2)@norm)
                au.close([summary['A'],summary['B'],summary['C']],[A,B,C],'all_worlds_moments_from_saved_score_norms',identity)
                au.close([summary['expected_target'],summary['expected_target_sq'],summary['probability_mass']],[pi@et,pi@et2,pi.sum()],'all_worlds_unconditional_targets_and_mass',identity)
                au.check(np.isfinite(norm).all() and (norm>=0).all() and np.all(et2>=et*et-2e-12) and abs(pi.sum()-1)<=5e-13,'valid_probability_score_norm_and_conditional_variance',identity)
                au.check(summary['policy_parameter_layout']==layout and summary['policy_parameter_count']==43319 and summary['policy_tensor_count']==9 and summary['prefix_batch_size']==1 and summary['message_count']==49,'saved_complete_parameter_layout_and_batch_definition',identity)
                limit=2e-6*max(1.,np.sqrt(A));ez=summary['expected_score_l2'];mu2=summary['mu_norm_sq']
                au.check(np.isfinite(mu2) and mu2>=0 and np.isfinite(ez) and 0<=ez<=limit and summary['zero_A']==(A==0) and summary['near_zero_A']==(A<=1e-12),'recorded_mean_score_residual_and_degenerate_flag',identity)
                au.close(summary['expected_score_gate_limit'],limit,'score_zero_gate_limit',identity)
                au.close(summary['matched_second_moment'],C-2*baseline*B+baseline*baseline*A,'matched_raw_second_moment',identity)
                au.close(summary['matched_mean_shift_l2'],abs(baseline)*ez,'finite_precision_baseline_mean_shift_bound',identity)
                if A>0:au.close(summary['optimal_baseline'],B/A,'world_optimal_baseline',identity)
                sample=(not formal and wi in (0,8,17)) or (formal and identity in FORMAL_SAMPLES)
                if sample:sample_scores(agent,h,raw,summary,au,identity)
                au.count('worlds');au.count('receiver_action_pairs_enumerated',49*36)
            aggregate=independent_aggregate(summaries)
            for field,expected in aggregate.items():
                if field=='optimal_baselines':au.check(result['aggregate'][field]==expected,'aggregate_optimal_baselines',key)
                else:au.close(result['aggregate'][field],expected,'all_policy_aggregate_'+field,key,atol=1e-11,rtol=2e-12)
                if field=='optimal_baselines':au.check(top[field]==expected,'root_table_optimal_baselines',key)
                else:au.close(top[field],expected,'root_table_'+field,key,atol=1e-11,rtol=2e-12)
            au.check(result['aggregate']['variance_clamped'] is False and min(aggregate[k] for k in ('variance_matched','variance_permuted','variance_optimal'))>=-aggregate['numeric_tolerance'],'no_hidden_negative_variance_clipping',key)
            au.count('policies');print(json.dumps(dict(audited_policy=folder.name,sample_worlds=au.counts.get('sample_worlds',0),failures=len(au.failed))),flush=True)
        au.check([state_sha(a) for a in agents]==snapshots and all(pv.grad is None for a in agents for pv in a.parameters()),'no_model_state_or_parameter_grad_mutation',(seed,p,t))
    au.check(done['policies']==au.counts['policies'] and done['worlds']==au.counts['worlds'] and done['message_scores']==au.counts['worlds']*49,'complete_measurement_counts')
    au.check(au.counts['sample_worlds']==(4 if formal else 18),'precommitted_independent_score_sample_count')
    return expected_hashes,formal


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--preflight',action='store_true');args=ap.parse_args()
    torch.set_num_threads(1);out=args.out.resolve();au=Audit();hashes={};formal=None;fatal=None
    try:
        synthetic_test(au);hashes,formal=audit_batch(out,au)
        if args.preflight:
            au.check(not formal,'preflight_uses_development_only')
            formal_inherited=read(PROJECT/'redesign_v0.15/results/scaled_001/invocation.json')['source_hashes']
            formal_hashes={**formal_inherited,**{str(ROOT/n):sha(ROOT/n) for n in ('run_gradient.py','gradient_moments.py','固定执行方案.md','前置审查.md')}}
            au.check(hashes==formal_hashes,'preflight_development_and_formal_production_sources_identical')
    except Exception:
        fatal=traceback.format_exc();au.check(False,'audit_fatal_exception',fatal)
    archive=out/'audit_gradient_source.py'
    if archive.exists() and sha(archive)!=sha(__file__):archive.rename(out/('audit_gradient_source_'+sha(archive)[:12]+'.py'))
    archive.write_bytes(Path(__file__).read_bytes())
    receipt=dict(passed=not au.failed,created_utc=datetime.now(timezone.utc).isoformat(),script_sha256=sha(__file__),
      output=str(out),formal=formal,source_hashes=hashes,checks=au.checks,failures=au.failed,
      counts=au.counts,max_errors=au.max_errors,selected_score_samples=au.samples,
      coverage=dict(all_raw_moments='π/36-action ET/ET2/A/B/C recomputed from recorded per-message score norms; all h/receiver/sender logits replayed from original source tensors',
       mu_and_score_vectors='only the preselected sample worlds have all 49 full parameter scores, μ and Ez independently replayed; other μ/Ez are recorded diagnostic inputs, not claimed independently rederived',
       double_reference='independent handwritten double Linear/GRU functional Jacobian, with nine preselected coordinate finite differences per sampled world',
       source_files='all consumed inputs and frozen source archives rehashed; formal originals bound to v15 completion manifest',
       statistical_unit='four inherited source seeds; partitions/persons/checkpoints/worlds are nested, no new independent training'),
      tolerances=dict(double_score_atol=DOUBLE_ATOL,double_score_rtol=DOUBLE_RTOL,finite_difference_step=FD_EPS,finite_difference_atol=FD_ATOL),
      formal_fixed_samples=[list(x) for x in sorted(FORMAL_SAMPLES)],production_moment_helpers_called=False,training_updates=0,optimizer_steps=0)
    dest=ROOT/'preflight_qa.json' if args.preflight else out/'audit_execution.json'
    if dest.exists():
        prior=dest.with_name(dest.stem+'_previous_'+sha(dest)[:12]+'.json');prior.write_bytes(dest.read_bytes())
    write(dest,receipt)
    if args.preflight:write(out/'audit_preflight.json',receipt)
    print(json.dumps(dict(passed=receipt['passed'],checks=au.checks,failures=len(au.failed),counts=au.counts,receipt=str(dest)),ensure_ascii=False),flush=True)
    if fatal:print(fatal,flush=True)
    if au.failed:raise SystemExit(1)

if __name__=='__main__':main()

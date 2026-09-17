"""Independent numeric/compatibility audit of frozen saved protocol probes.

The optimization axis is reversed: maximize old correctness at each achievable
new count. This does not call the probe's DP, summary function, or model forward.
"""
from __future__ import annotations
import argparse
import ast
from datetime import datetime,timezone
import hashlib
import importlib.util
from itertools import product
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import audit_anchoring as saved
MAPS=saved.MAPS;SEEDS=saved.SEEDS;ARMS=saved.ARMS
read,sha=saved.read,saved.sha


class QA:
    def __init__(self):self.counts={};self.failures=[]
    def check(self,ok,name,context=None):
        self.counts[name]=self.counts.get(name,0)+1
        if not bool(ok):self.failures.append(dict(check=name,context=context))


def load_arrays(path):
    with np.load(path) as z:return {k:z[k] for k in z.files}


def reverse_dp(oc,nc):
    old=oc.max(0);new=nc.max(0)
    best=np.full(int(new.sum())+1,-1,dtype=int);best[0]=0
    for code in range(oc.shape[1]):
        after=np.full_like(best,-1)
        for n in np.flatnonzero(best>=0):
            after[n]=max(after[n],best[n]+int(old[code]))
            at=n+int(new[code]);after[at]=max(after[at],best[n])
        best=after
    return best


def reverse_optimum(dp,threshold):
    feasible=np.flatnonzero(dp>=threshold)
    assert len(feasible)
    n=int(feasible[-1]);return int(dp[n]),n


def pools(p):return {k:np.array(v,dtype=int) for k,v in saved.v10.partitions(p).items()}


def legacy_truth(base,current,p):
    group=pools(p);old,new=group['old'],group['added']
    oi=(old[:,None]*16+np.arange(16)).reshape(-1);ni=(new[:,None]*16+np.arange(16)).reshape(-1)
    bid=7*base['greedy_message'][:,0]+base['greedy_message'][:,1]
    cid=7*current['greedy_message'][:,0]+current['greedy_message'][:,1]
    bd=base['receiver_logits'].argmax(-1);cd=current['receiver_logits'].argmax(-1)
    oc=np.array([np.bincount(bid[m*16:(m+1)*16],minlength=49) for m in old])
    nc=np.array([np.bincount(cid[m*16:(m+1)*16],minlength=49) for m in new])
    old_positions=np.repeat(MAPS[old],16,axis=0);new_positions=np.repeat(MAPS[new],16,axis=0)
    def score(decoder):return (int((decoder[bid[oi]]==old_positions).all(1).sum()),
                              int((decoder[cid[ni]]==new_positions).all(1).sum()))
    t0=score(bd)[0];theta,natural=score(cd)
    dp=reverse_dp(oc,nc);used=oc.sum(0)>0
    independent=int(nc.max(0).sum());old_at_cs=int(oc.max(0)[nc.max(0)==0].sum())
    strict_new=sum(int(nc[k,m]) for k,z in enumerate(new) for m in np.flatnonzero(used) if np.array_equal(bd[m],MAPS[z]))
    strict_new+=int(nc.max(0)[~used].sum())
    return dict(old=old,new=new,oi=oi,ni=ni,bid=bid,cid=cid,bd=bd,cd=cd,oc=oc,nc=nc,used=used,score=score,
        dp=dp,t0=t0,theta=theta,natural=natural,independent=independent,old_at_cs=old_at_cs,strict_new=strict_new,
        current_sender_old_receiver_old_J=float((bd[cid[oi]]==old_positions).all(1).mean()),
        current_old_J=float((cd[cid[oi]]==old_positions).all(1).mean()),
        old_message_change=float((bid[oi]!=cid[oi]).mean()),new_using_old_used_codes=float(used[cid[ni]].mean()))


def scalar_truth(t):
    return dict(new_J=t['natural']/96,legacy_old_J=t['theta']/288,
        current_sender_old_receiver_old_J=t['current_sender_old_receiver_old_J'],current_old_J=t['current_old_J'],
        actual_retention_bound=reverse_optimum(t['dp'],t['theta'])[1]/96,
        fixed_retention_bound=reverse_optimum(t['dp'],t['t0'])[1]/96,
        strict_legacy_actions_bound=t['strict_new']/96,independent_new_CS=t['independent']/96,
        min_old_loss_for_new_CS_pp=100*max(0,t['t0']-t['old_at_cs'])/288,
        old_message_change=t['old_message_change'],new_using_old_used_codes=t['new_using_old_used_codes'],
        fixed_threshold_met_fraction=float(t['theta']>=t['t0']))


def audit_legacy(qa,actual,base,current,p,label):
    t=legacy_truth(base,current,p)
    expected=dict(old_n=288,new_n=96,initial_old_correct=t['t0'],actual_legacy_old_correct=t['theta'],
        actual_new_correct=t['natural'],new_J=t['natural']/96,legacy_old_J=t['theta']/288,
        current_sender_old_receiver_old_J=t['current_sender_old_receiver_old_J'],current_old_J=t['current_old_J'],
        meets_fixed_old_threshold=t['theta']>=t['t0'],independent_new_CS=t['independent']/96,
        min_old_loss_for_new_CS_pp=100*max(0,t['t0']-t['old_at_cs'])/288,
        old_message_change=t['old_message_change'],new_using_old_used_codes=t['new_using_old_used_codes'])
    for key,value in expected.items():qa.check(actual[key]==value,'legacy_metrics_independent_integer_counts',[label,key])
    qa.check(np.array_equal(actual['old_counts'],t['oc']) and np.array_equal(actual['new_counts'],t['nc']),
             'old_S0_and_current_added_frequency_tables',label)
    for name,threshold in [('actual_retention_bound',t['theta']),('fixed_retention_bound',t['t0']),('new_CS_witness',t['old_at_cs'])]:
        row=actual[name];o,n=reverse_optimum(t['dp'],threshold);decoder=np.array(row['decoder'])
        qa.check(row['threshold']==threshold and row['old_correct']==o and row['new_correct']==n and row['new_J']==n/96,
                 'reversed_axis_DP_optimum',[label,name])
        qa.check(decoder.shape==(49,2) and np.issubdtype(decoder.dtype,np.integer) and ((decoder>=0)&(decoder<6)).all(),
                 'valid_49_code_decoder_witness',[label,name])
        qa.check(t['score'](decoder)==(o,n),'oracle_witness_replay_raw_photo_rows',[label,name])
    strict=actual['strict_legacy_actions_bound'];decoder=np.array(strict['decoder'])
    qa.check(t['score'](decoder)==(t['t0'],t['strict_new']) and strict['old_correct']==t['t0'] and
             strict['new_correct']==t['strict_new'] and strict['new_J']==t['strict_new']/96,'strict_all_old_actions_optimum',label)
    qa.check(np.array_equal(decoder[t['bid'][t['oi']]],t['bd'][t['bid'][t['oi']]]),
             'strict_keeps_every_old_S0_action_including_errors',label)
    qa.check(t['natural']<=actual['actual_retention_bound']['new_correct'],'actual_new_bounded_by_actual_legacy_retention',label)
    if t['theta']>=t['t0']:
        qa.check(t['natural']<=actual['fixed_retention_bound']['new_correct'],'fixed_threshold_only_enforced_if_met',label)
    qa.check(t['strict_new']<=actual['fixed_retention_bound']['new_correct']<=t['independent'],
             'nested_strict_fixed_independent_bounds',label)
    return scalar_truth(t)


def probability_groups(z,p):
    # logaddexp-based normalization is independent of the probe's exp/sum helper.
    def probs(x):
        x=x.astype(np.float64);return np.exp(x-np.logaddexp.reduce(x,axis=-1,keepdims=True))
    f=probs(z['sender_first_logits']);s=probs(z['sender_second_logits']);r=probs(z['receiver_logits'])
    q=(f[:,:,None]*s).reshape(-1,49);decoder=z['receiver_logits'].argmax(-1);pos=z['positions']
    codes=7*z['greedy_message'][:,0]+z['greedy_message'][:,1];natural=(decoder[codes]==pos).all(1)
    reachable=(decoder[None,:,:]==pos[:,None,:]).all(2).any(1)
    pf=r[np.arange(49)[None,:],0,pos[:,0,None]];pw=r[np.arange(49)[None,:],1,pos[:,1,None]]
    both=pf*pw;Q=(q*both).sum(1);B=both.max(1);reward=(q*(.25*(pf+pw)+.5*both)).sum(1)
    ids=pos[:,0]*5+pos[:,1]-(pos[:,1]>pos[:,0]);counts=np.zeros((30,49),int);np.add.at(counts,(ids,codes),1)
    result={}
    for name,pool in {**pools(p),'all':np.arange(30)}.items():
        ix=np.isin(ids,pool);n=int(ix.sum())
        result[name]=dict(n=n,natural_correct=int(natural[ix].sum()),N=float(natural[ix].mean()),U=float(reachable[ix].mean()),
            Q=float(Q[ix].mean()),B=float(B[ix].mean()),expected_mixed_reward=float(reward[ix].mean()),C_S=float(counts[pool].max(0).sum()/n))
    return result


def self_test(qa):
    rng=np.random.default_rng(11011621)
    for trial in range(48):
        no,nn,ncode=(int(x) for x in rng.integers(1,[4,4,6]));oc=rng.integers(0,4,(no,ncode));nc=rng.integers(0,4,(nn,ncode))
        attainable=[]
        for assignments in product(range(no+nn+1),repeat=ncode):
            o=n=0
            for code,a in enumerate(assignments):
                if a<no:o+=int(oc[a,code])
                elif a<no+nn:n+=int(nc[a-no,code])
            attainable.append((o,n))
        dp=reverse_dp(oc,nc)
        for threshold in range(int(oc.max(0).sum())+1):
            wanted=max((n,o) for o,n in attainable if o>=threshold);o,n=reverse_optimum(dp,threshold)
            qa.check((n,o)==wanted,'reverse_DP_exhaustive_full_assignments',[trial,threshold])
    # Execute only the frozen pure metric function, without importing its runner,
    # model, extract function, or any inference. Compare to the independent DP.
    path=ROOT/'probe_anchoring.py';module=ast.parse(path.read_text())
    node=next(x for x in module.body if isinstance(x,ast.FunctionDef) and x.name=='legacy_bound')
    compat_path=ROOT.parent/'redesign_v0.10/analyze_compatibility.py'
    spec=importlib.util.spec_from_file_location('old_compat_for_frozen_metric_test',compat_path)
    compat=importlib.util.module_from_spec(spec);spec.loader.exec_module(compat)
    ns=dict(np=np,MAPS=MAPS,compat=compat,run=SimpleNamespace(v10=SimpleNamespace(partition=pools)))
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),ns)
    source=ROOT.parent/'redesign_v0.10/results/generalization_001/protocol'
    base=load_arrays(source/'s29101_p1_base_d0.npz');current=load_arrays(source/'s29101_p1_expand_receiver_d0.npz')
    for name,z in [('base_equals_current',base),('prior_receiver_only',current)]:
        actual=ns['legacy_bound'](base,z,1);audit_legacy(qa,actual,base,z,1,name)
        if name=='base_equals_current':
            qa.check(actual['actual_retention_bound']==actual['fixed_retention_bound'] and actual['old_message_change']==0,
                     'base_current_base_limit',name)
    # Deliberate counterexample: all new codes reuse six perfect old codes;
    # rewriting them attains new=96 but old=192, violating original T0=288.
    group=pools(1);messages=np.zeros((480,2),int);decoder=np.zeros((49,2),int)
    for j,m in enumerate(group['old']):messages[m*16:(m+1)*16]=[j//7,j%7];decoder[j]=MAPS[m]
    for j,m in enumerate(group['added']):messages[m*16:(m+1)*16]=[j//7,j%7]
    def logits(table):
        x=np.full((49,2,6),-1.,dtype=np.float32)
        for m,g in product(range(49),range(2)):x[m,g,table[m,g]]=1.
        return x
    b=dict(positions=np.repeat(MAPS,16,axis=0),greedy_message=messages,receiver_logits=logits(decoder))
    changed=decoder.copy()
    for j,m in enumerate(group['added']):changed[j]=MAPS[m]
    c=dict(b,receiver_logits=logits(changed));actual=ns['legacy_bound'](b,c,1);audit_legacy(qa,actual,b,c,1,'threshold_violation_constructed')
    qa.check(actual['initial_old_correct']==288 and actual['actual_legacy_old_correct']==192 and actual['actual_new_correct']==96 and
             actual['fixed_retention_bound']['new_correct']==0 and actual['actual_retention_bound']['new_correct']==96 and
             not actual['meets_fixed_old_threshold'],'do_not_apply_T0_bound_when_actual_old_violates_T0')
    return dict(random_exhaustive_trials=48,rng_seed=11011621,real_cases=['v10 s29101 p1 d0 base=current','v10 s29101 p1 d0 receiver-only'],
                constructed_case='old=288 to 192, new=96: Ttheta bound96, T0 bound0, T0 not met',
                current_v11_training_data_read=False,model_inference=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--self-test',action='store_true');p.add_argument('--root',type=Path,default=ROOT/'results/anchoring_001')
    p.add_argument('--out',type=Path);args=p.parse_args();batch=args.root.resolve()
    out=args.out or (ROOT/'audit_protocol_preflight.json' if args.self_test else batch/'audit_protocol_independent.json')
    if out.exists():raise FileExistsError(out)
    qa=QA();small=self_test(qa);records=[];direction_rows=[]
    manifest=read(ROOT/'protocol_fixed_manifest.json')
    for path,digest in manifest['hashes'].items():qa.check(sha(path)==digest,'frozen_protocol_source_hash',path)
    if not args.self_test:
        archived=ROOT/'protocol_sources'
        qa.check(read(archived/'protocol_fixed_manifest.json')==manifest,'archived_protocol_manifest_exact')
        for name in ('probe_anchoring.py','协议测量方案.md'):
            qa.check(sha(archived/name)==manifest['hashes'][str(ROOT/name)],'archived_protocol_source_exact',name)
        qa.check(datetime.fromisoformat(manifest['frozen_utc'])<datetime.fromisoformat(read(batch/'invocation.json')['started_utc']),
                 'protocol_manifest_fixed_before_training_started')
        for seed,pnum,arm in product(SEEDS,(1,2,3),ARMS):
            qa.check((batch/f's{seed}_p{pnum}_{arm}/result.json').is_file(),'all_formal_training_finished',[seed,pnum,arm])
        source=read(batch/'protocol_analysis.json');records=source['records'];expected={(s,p,a,u) for s,p in product(SEEDS,(1,2,3)) for a,u in [('base',0)]+list(product(ARMS,(300,600)))}
        qa.check(source['complete'] and {(r['seed'],r['partition'],r['arm'],r['update']) for r in records}==expected and len(records)==84,
                 'exact_84_model_pairs')
        qa.check(source['source_hashes']==manifest['hashes'],'analysis_matches_frozen_protocol_sources')
        arrays={};lookup={}
        for record in records:
            seed,pnum,arm,step=(record[k] for k in ('seed','partition','arm','update'));key=(seed,pnum,arm,step)
            checkpoint=(ROOT.parent/f'redesign_v0.10/results/generalization_001/s{seed}_p{pnum}_base/final.pt') if arm=='base' else batch/f's{seed}_p{pnum}_{arm}/checkpoint_{step:04d}.pt'
            qa.check(Path(record['source_checkpoint']).resolve()==checkpoint.resolve() and record['source_sha256']==sha(checkpoint),
                     'probe_checkpoint_source_hash',list(key))
            qa.check([d['direction'] for d in record['directions']]==[0,1],'exact_two_directions',list(key))
            for d in record['directions']:
                direction=d['direction'];label=(*key,direction);path=batch/'protocol'/d['arrays'];z=load_arrays(path)
                arrays[label]=z;lookup[label]=d
                qa.check(sha(path)==d['sha256'],'saved_probe_npz_hash',list(label))
                qa.check(np.array_equal(z['positions'],np.repeat(MAPS,16,axis=0)) and
                         np.array_equal(z['photo_ids'],np.tile(np.asarray(source['photos']),(30,1))),
                         'same_30_maps_16_photo_pairs',list(label))
                first=z['sender_first_logits'].argmax(-1);second=z['sender_second_logits'][np.arange(480),first].argmax(-1)
                qa.check(np.array_equal(z['greedy_message'],np.column_stack((first,second))),
                         'autoregressive_greedy_not_global_code_argmax',list(label))
                qa.check(z['receiver_logits'].shape==(49,2,6) and z['sender_first_logits'].shape==(480,7) and
                         z['sender_second_logits'].shape==(480,7,7) and all(z[k].dtype==np.float32 and np.isfinite(z[k]).all() for k in
                         ('receiver_logits','sender_first_logits','sender_second_logits')),'finite_logit_shapes',list(label))
                decoder=z['receiver_logits'].argmax(-1);ordered=np.sort(z['receiver_logits'],axis=-1)
                qa.check((ordered[:,:,-1]>ordered[:,:,-2]).all(),'receiver_unique_maxima',list(label))
                menu=d['menu_audit'];physical=np.repeat(decoder[:,:,None],720,axis=2)
                qa.check(menu['all_menu_permutations_physically_equivalent'] and menu['enumerated_messages']==49 and menu['goals']==2 and
                         menu['menus']==720 and menu['cases']==70560 and menu['physical_action_sha256']==hashlib.sha256(physical.tobytes()).hexdigest(),
                         'menu_audit_matches_constant_physical_table',list(label))
                expected_groups=probability_groups(z,pnum)
                for group,values in expected_groups.items():
                    for name,value in values.items():qa.check(np.isclose(d['groups'][group][name],value,rtol=0,atol=2e-12),
                                                             'independent_group_counts_and_probabilities',[*label,group,name])
        for key,z in arrays.items():
            seed,pnum,arm,step,direction=key;base=arrays[seed,pnum,'base',0,direction]
            scalar=audit_legacy(qa,lookup[key]['legacy'],base,z,pnum,list(key))
            qa.check(np.isclose(scalar['new_J'],lookup[key]['groups']['added']['N'],atol=0,rtol=0),'legacy_new_J_matches_natural_added_N',list(key))
            direction_rows.append(dict(seed=seed,partition=pnum,arm=arm,update=step,direction=direction,metrics=scalar))
            if arm=='release' and step==300:
                paired=arrays[seed,pnum,'anchor',300,direction]
                qa.check(all(np.array_equal(z[k],paired[k]) for k in z),'anchor_release_midpoint_all_logits_messages_identical',list(key))
            if arm=='base':qa.check(lookup[key]['legacy']['actual_retention_bound']==lookup[key]['legacy']['fixed_retention_bound'] and
                                    scalar['old_message_change']==0,'all_base_limit_equalities',list(key))
        for arm,step in [('base',0)]+list(product(ARMS,(300,600))):
            seed_means=[]
            for seed in SEEDS:
                selected=[x['metrics'] for x in direction_rows if (x['seed'],x['arm'],x['update'])==(seed,arm,step)]
                qa.check(len(selected)==6,'six_repeated_directions_per_seed',[seed,arm,step])
                mean={k:float(np.mean([x[k] for x in selected])) for k in selected[0]};seed_means.append(mean)
                actual=next(x['metrics'] for x in source['seed_cells'] if (x['seed'],x['arm'],x['update'])==(seed,arm,step))
                qa.check(all(np.isclose(actual[k],v,rtol=0,atol=2e-12) for k,v in mean.items()),'within_seed_aggregation', [seed,arm,step])
            mean={k:float(np.mean([x[k] for x in seed_means])) for k in seed_means[0]}
            qa.check(all(np.isclose(source['aggregate'][f'{arm}_{step}'][k],v,rtol=0,atol=2e-12) for k,v in mean.items()),
                     'four_seed_equal_weight_aggregate',[arm,step])
    report=dict(passed=not qa.failures,self_test_only=args.self_test,checks=qa.counts,total_checks=sum(qa.counts.values()),failures=qa.failures,
        model_pairs_checked=len(records),directions_checked=len(direction_rows),self_tests=small,source_hashes=manifest['hashes'],
        audit_script_sha256=sha(__file__),completed_utc=datetime.now(timezone.utc).isoformat(),
        scope='Independent saved-logit probability calculations and reverse-axis integer frontier, raw-photo witness replay, source hashes and four-seed aggregation. No model forward or optimizer execution; no claim to independently regenerate saved logits from weights.')
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(passed=report['passed'],checks=report['total_checks'],failures=len(qa.failures),directions=len(direction_rows),out=str(out)),ensure_ascii=False))
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':main()

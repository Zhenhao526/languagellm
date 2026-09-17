"""Independent zero-forward audit of frozen local-need response diagnostics.

No production case/environment/kernel/runner function is imported. Pinned earlier
independent physical settlement and native acceptance are reused. Cases, weighting,
exact permutation expectation and primary are implemented independently here.
"""
import argparse, hashlib, json, re, time, traceback
from collections import Counter
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
import numpy as np
from research_program.triadic_reciprocal_execution_study import audit_execution as prior

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]
require,read,sha,compare,close,json_bytes=prior.require,prior.read,prior.sha,prior.compare,prior.close,prior.json_bytes
env=prior.env
PARTS=env.PARTS; SEEDS=(57101,57102,57103,57104); RULES=('strict','reciprocal')
AXES=('kind','length','destination'); PAIR_NAMES=('AB','AC','BC')
COUNTS=('both_correct','only_first_correct','only_second_correct','one_correct','neither_correct',
        'partner_change','both_executed_pair_change','execution_status_change')
PRIOR_SHA='ee995a50d32ee44018a1f1fb63bfab743f84a81fb78045cd4d40d62dd389faa1'
ORIGINAL_SHA='e555037fa8a9d72a2ef150a0d99d46e5a893139b2efa02314dffa3e2a001a299'


def references():
    refs=prior.references();p=Path(prior.__file__).resolve()
    require(sha(p)==PRIOR_SHA,'Changed independent settlement helper');refs[str(p)]=PRIOR_SHA
    return refs


def build_cases(spec):
    """Flip resource acceptance sets by material XOR, not resource-id shortcuts."""
    needs=[tuple(map(int,row)) for row in spec['needs']];lookup={row:i for i,row in enumerate(needs)}
    require(len(needs)==len(lookup)>1,'Unique needs required')
    bg=list(product(range(len(spec['layouts'])),range(len(spec['private_sites']))));B=len(bg)
    states=np.asarray([list(row)+[0,1,2,3,1,2,3] for row in needs],np.int16)
    native=env.rewards(states);require(np.all((native==1).sum(1)==1),'Unique correct physical plan')
    targets=(native.argmax(1)//8).astype(int).tolist()
    masks=(3,12,5,10,1,2,4,8);edges=[]
    for i,row in enumerate(needs):
        for who in range(3):
            resource,dest=divmod(row[who],3)
            for axis in range(3):
                if axis<2:
                    xor=2 if axis==0 else 1
                    flipped=sum(1<<(m^xor) for m in range(4) if masks[resource]&(1<<m))
                    changed=masks.index(flipped)*3+dest
                else:changed=resource*3+(1-dest if dest<2 else dest)
                other=list(row);other[who]=changed;other=tuple(other)
                if other<=row or other not in lookup:continue
                j=lookup[other]
                if targets[i]!=targets[j]:edges.append((i,j,who,axis,targets[i],targets[j]))
    edges.sort(key=lambda e:(e[2],e[3],needs[e[0]],needs[e[1]]));strata=[]
    for who,axis in product(range(3),repeat=2):
        ids=[k for k,e in enumerate(edges) if e[2]==who and e[3]==axis]
        counts=Counter((edges[k][4],edges[k][5]) for k in ids)
        strata.append(dict(changed_person=who,axis=AXES[axis],axis_index=axis,edge_indices=ids,
            need_edges=len(ids),state_edges=len(ids)*B,empty=not ids,
            target_transition_counts={f'{a}_{b}':counts[a,b] for a,b in product(range(3),repeat=2) if a!=b}))
    require(spec['world_count']==len(needs)*B,'Cartesian cardinality')
    return dict(schema='single_need_executed_pair_response_v1',partition=spec['partition'],
        spec_sha256=hashlib.sha256(json.dumps(spec,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest(),
        state_order='need-major, then layout, then owner',needs=[list(n) for n in needs],layouts=spec['layouts'],private_sites=spec['private_sites'],
        need_worlds=len(needs),n_backgrounds=B,world_count=len(needs)*B,
        background_layout_owner_indices=[list(b) for b in bg],need_target_pairs=targets,
        edge_need_indices=[[e[0],e[1]] for e in edges],changed_person=[e[2] for e in edges],axis_index=[e[3] for e in edges],
        target_pairs=[[e[4],e[5]] for e in edges],need_edges=len(edges),state_edges=len(edges)*B,strata=strata,
        empty_strata=[dict(changed_person=s['changed_person'],axis=s['axis']) for s in strata if s['empty']],
        actor_inputs=False,policy_dependent_filter=False)


def validate_partition(cases,states,indices):
    ids=np.arange(cases['world_count'],dtype=np.int64)
    require(np.asarray(indices).dtype.kind in 'iu' and np.array_equal(indices,ids),'Complete ordered state indices')
    wanted=env.packed(cases,ids)
    require(np.asarray(states).dtype.kind in 'iu' and np.array_equal(states,wanted),'Complete ordered Cartesian states')


def metrics(cases,actual):
    actual=np.asarray(actual);N=cases['need_worlds'];B=cases['n_backgrounds']
    require(actual.shape==(N*B,) and actual.dtype.kind in 'iu' and np.isin(actual,(-1,0,1,2)).all(),'Actual pair vector')
    output=actual.reshape(N,B);edges=np.asarray(cases['edge_need_indices'],np.int64).reshape(-1,2)
    truth=np.asarray(cases['target_pairs'],np.int64).reshape(-1,2)
    require(np.all(truth[:,0]!=truth[:,1]),'Shuffle formula requires distinct endpoint truths')
    marginals=np.stack([(output==p).sum(0) for p in range(3)])
    backgrounds=[]
    for b,(li,oi) in enumerate(cases['background_layout_owner_indices']):
        backgrounds.append(dict(background_index=b,layout_index=li,owner_index=oi,layout=cases['layouts'][li],
            private_sites=cases['private_sites'][oi],worlds=N,actual_pair_counts={name:int((output[:,b]==p).sum()) for p,name in enumerate(('none',)+PAIR_NAMES,start=-1)},
            shuffle_ordered_world_pairs=N*(N-1)))
    strata=[]
    scalar=('Q','Q_shuffle','Q_excess')+tuple(k+'_rate' for k in COUNTS)
    for structure in cases['strata']:
        chosen=np.asarray(structure['edge_indices'],np.int64);E=len(chosen);records=[]
        if E:
            observed=output[edges[chosen]];wanted=truth[chosen,:,None]
            left,right=observed[:,0],observed[:,1];c0=left==wanted[:,0];c1=right==wanted[:,1]
            flags=(c0&c1,c0&~c1,~c0&c1,c0^c1,~c0&~c1,left!=right,
                (left>=0)&(right>=0)&(left!=right),(left>=0)^(right>=0))
            totals=np.stack([f.sum(0) for f in flags])
            numerators=(marginals[truth[chosen,0]]*marginals[truth[chosen,1]]).sum(0)
        else:totals=np.zeros((len(COUNTS),B),np.int64);numerators=np.zeros(B,np.int64)
        for b in range(B):
            raw={k:int(totals[j,b]) for j,k in enumerate(COUNTS)};den=N*(N-1)*E
            r=dict(background_index=b,edges=E,raw_counts=raw,shuffle_numerator_sum=int(numerators[b]),shuffle_denominator=den)
            r.update({k+'_rate':raw[k]/E if E else None for k in COUNTS})
            r.update(Q=raw['both_correct']/E if E else None,Q_shuffle=int(numerators[b])/den if den else None)
            r['Q_excess']=r['Q']-r['Q_shuffle'] if E else None;records.append(r)
        row={k:structure[k] for k in ('changed_person','axis','axis_index','need_edges','state_edges','empty','target_transition_counts')}
        row.update(raw_counts={k:sum(r['raw_counts'][k] for r in records) for k in COUNTS},backgrounds=records)
        row.update({k:float(np.mean([r[k] for r in records])) if E else None for k in scalar});strata.append(row)
    all_nonempty=not cases['empty_strata'];raw={k:sum(s['raw_counts'][k] for s in strata) for k in COUNTS}
    out=dict(schema='single_need_response_metrics_v1',partition=cases['partition'],worlds=N*B,
        need_worlds_per_background=N,background_count=B,need_edges=len(edges),state_edges=len(edges)*B,
        complete_nine_strata=all_nonempty,empty_strata=cases['empty_strata'],strata=strata,backgrounds=backgrounds,raw_counts=raw,
        raw_pooled_Q=raw['both_correct']/(len(edges)*B) if len(edges) else None)
    out.update({k:float(np.mean([s[k] for s in strata])) if all_nonempty else None for k in scalar})
    return out


def primary(rows):
    indexed={(r['seed'],r['training_rule'],r['part']):r for r in rows}
    require(set(indexed)==set(product(SEEDS,RULES,PARTS)) and len(rows)==32,'Exactly32 old endpoints')
    values=[]
    for seed in SEEDS:
        q=indexed[seed,'reciprocal','new_needs_and_layouts']['settlements']['reciprocal']
        require(all(isinstance(q[k],float) and np.isfinite(q[k]) for k in ('Q','Q_shuffle','Q_excess')),'Defined primary')
        values.append(dict(seed=seed,**{k:q[k] for k in ('Q','Q_shuffle','Q_excess')}))
    return dict(name='reciprocal_Q_excess_complete_double_holdout',paired_units=4,by_seed=values,
        **{'mean_'+k:float(np.mean([v[k] for v in values])) for k in ('Q','Q_shuffle','Q_excess')})


def audit(run,plan_sha):
    run=Path(run).resolve();refs=references();artifacts=dict(refs)
    require(re.fullmatch('[0-9a-f]{64}',plan_sha) is not None,'Explicit frozen plan hash')
    def bind(p,digest=None):
        p=Path(p).resolve();got=sha(p)
        if digest is not None:require(got==digest,'Artifact checksum '+str(p))
        artifacts[str(p)]=got
    for p in (run/'plan.json',run/'freeze.json',run/'status.json',run/'result.json'):bind(p)
    plan=read(run/'plan.json');result=read(run/'result.json');status=read(run/'status.json');prior.finite_tree(result)
    require(status['status']==result['status']=='completed','Only complete diagnostic')
    require(sha(run/'result.json')==status['result_sha256'],'Result completion hash')
    require(sha(run/'plan.json')==plan_sha==read(run/'freeze.json')['plan_sha256']==result['plan_sha256'],'Plan freeze chain')
    require(plan['schema']=='triadic_need_response_v1' and plan['status']=='prepared_without_endpoint_reads','Diagnostic schema')
    expected_budget=dict(endpoint_files=32,settlement_evaluations=64,new_network_forwards=0,new_training_updates=0,new_model_calls=0)
    require(plan['budget']==result['budget']==expected_budget,'Zero-forward fixed budget')
    for p,d in plan['source_sha256'].items():bind(p,d);bind(run/'source_snapshot'/Path(p).relative_to(ROOT),d)
    for p,d in plan['inputs_sha256'].items():bind(p,d)
    frozen=read(HERE/'audit_freeze_001.json');bind(HERE/'audit_freeze_001.json')
    require(frozen['status']=='frozen_before_formal_diagnostic_audit','Independent audit freeze')
    for p,d in frozen['source_sha256'].items():bind(p,d)
    original=ROOT/'research_program/triadic_action_dependency_study/results/context_001/prepared.json';bind(original,ORIGINAL_SHA)
    specs=read(original)['partitions'];independent,_,_=env.independent_specs()
    for p in PARTS:compare(specs[p],independent[p],'Independent support/')
    cc={p:build_cases(specs[p]) for p in PARTS}
    source=ROOT/'research_program/triadic_reciprocal_execution_study/results/rules_001/execution/results.json'
    bind(source,'28dc1ee3ba9a6a460ad783b42f3d7faf6caef52d55be9753693122d894984a5e')
    wanted=[]
    for row in read(source)['runs']:
        for part in PARTS:
            item=row['final'][part];wanted.append(dict(seed=row['seed'],training_rule=row['rule'],part=part,path=item['path'],sha256=item['data_sha256']))
    require(plan['endpoint_records']==wanted,'Original complete endpoint manifest')
    require(len(result['records'])==32,'Exactly32 records');computed=[];counts=Counter()
    for meta,record in zip(wanted,result['records']):
        compare(record,meta,'Endpoint identity/');bind(meta['path'],meta['sha256'])
        with np.load(meta['path'],allow_pickle=False) as f:states=f['states'];ids=f['state_indices'];actions=f['action_indices']
        c=cc[meta['part']];validate_partition(c,states,ids)
        stats={}
        require(set(record['settlements'])==set(RULES),'Both settlements required')
        for rule in RULES:
            actual=prior.settle(states,actions,rule)['actual_pair_index'];stats[rule]=metrics(c,actual)
            compare(record['settlements'][rule],stats[rule],'Independent local response/')
            counts['settlements']+=1;counts['edge_background_evaluations']+=c['state_edges']
        computed.append(dict(meta,settlements=stats));counts['endpoint_files']+=1;counts['worlds']+=len(states)
        print(json.dumps(dict(stage='endpoint_response_audited',seed=meta['seed'],rule=meta['training_rule'],part=meta['part'])),flush=True)
    calculated=primary(computed);compare(result['primary'],calculated,'Independent diagnostic primary/')
    require(dict(counts)==dict(endpoint_files=32,settlements=64,worlds=6193152,edge_background_evaluations=4036608),'Complete audit scope')
    for p,d in artifacts.items():require(sha(p)==d,'Artifact changed during audit '+p)
    return dict(status='passed',audit_source_sha256=sha(__file__),plan_sha256=plan_sha,scope=dict(counts),
        new_network_forwards=0,new_training_updates=0,primary=calculated,artifacts_sha256=artifacts,
        limits=['This is an additional descriptive analysis of previously inspected policies, not independent confirmation.',
            'No neural/optimizer/gradient replay. Physical settlements and local response cases are independently recomputed.',
            'Exact shuffle is an analytic marginal reference, not a p-value or optimal silent bound.',
            'Overlapping edges/backgrounds are repeated measurements; four societies are independent units.'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--out',required=True);p.add_argument('--plan-sha',required=True);a=p.parse_args()
    out=Path(a.out).resolve();require(not out.exists(),'Never overwrite audit');out.mkdir(parents=True);start=time.perf_counter()
    (out/'started.json').write_bytes(json_bytes(dict(at=datetime.now(timezone.utc).isoformat(),run=a.run,plan_sha256=a.plan_sha,audit_source_sha256=sha(__file__))))
    try:
        v=audit(a.run,a.plan_sha);v.update(elapsed_seconds=time.perf_counter()-start,completed_at=datetime.now(timezone.utc).isoformat())
        (out/'verification.json').write_bytes(json_bytes(v));print(json.dumps({k:v[k] for k in ('status','scope','elapsed_seconds')}))
    except BaseException as e:
        (out/'failure.json').write_bytes(json_bytes(dict(status='failed',error=repr(e),traceback=traceback.format_exc(),automatic_retry=False,elapsed_seconds=time.perf_counter()-start)));raise

if __name__=='__main__':main()

"""Independent finite certificate for reciprocal-pair execution, no models.

Only the old SHA-bound static partition JSON is read. No candidate numerical
kernel, old environment implementation, NumPy, policy or checkpoint is imported.
"""
from collections import Counter,defaultdict
from datetime import datetime,timezone
from fractions import Fraction
from hashlib import sha256
from itertools import combinations,permutations,product
import json
from pathlib import Path
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
PREPARED=ROOT/'research_program/triadic_action_dependency_study/results/context_001/prepared.json'
PREPARED_SHA='e555037fa8a9d72a2ef150a0d99d46e5a893139b2efa02314dffa3e2a001a299'
RESOURCE=({0,1},{2,3},{0,2},{1,3},{0},{1},{2},{3})
DESTINATION=({0},{1},{0,1})
PAIRS=tuple(combinations(range(3),2))
LAYOUTS=tuple(permutations(range(4)))
OWNERS=tuple(permutations((1,2,3)))


def digest(path):return sha256(Path(path).read_bytes()).hexdigest()
def accepts(need,material,destination):return material in RESOURCE[need//3] and destination in DESTINATION[need%3]


def full_plans(needs):
    return [(i,j,m,d) for i,j in PAIRS for m,d in product(range(4),range(2))
            if accepts(needs[i],m,d) and accepts(needs[j],m,d)]


def encode(actor,partner,site,dest):return 1+4*site+2*dest+[x for x in range(3) if x!=actor].index(partner)


def decode(actor,action):
    if action==0:return None
    value=action-1
    return ([x for x in range(3) if x!=actor][value%2],value//4,(value//2)%2)


def settlement(needs,layout,actions):
    decoded=[decode(i,a) for i,a in enumerate(actions)];executed=[]
    for i,j in PAIRS:
        a,b=decoded[i],decoded[j]
        if a is not None and b is not None and a[0]==j and b[0]==i and a[1:]==b[1:]:
            executed.append((i,j,a[1],a[2]))
    assert len(executed)<=1
    if not executed:return None,0
    i,j,site,dest=executed[0]
    return (i,j),int(accepts(needs[i],layout[site],dest))+int(accepts(needs[j],layout[site],dest))


def local_counts(needs,truth,layout):
    active_full=[defaultdict(Counter) for _ in range(3)]
    active_partner=[defaultdict(Counter) for _ in range(3)]
    proposal_role=[defaultdict(Counter) for _ in range(3)]
    for n in needs:
        i,j,material,dest=truth[n];site=layout.index(material)
        for actor in range(3):
            if actor in (i,j):
                peer=j if actor==i else i
                active_full[actor][n[actor]][encode(actor,peer,site,dest)]+=1
                active_partner[actor][n[actor]][peer]+=1
                proposal_role[actor][n[actor]][peer]+=1
            else:proposal_role[actor][n[actor]]['wait']+=1
    maximum=lambda tables:[sum(max(c.values(),default=0) for c in actor.values()) for actor in tables]
    return dict(active_full_numerators=maximum(active_full),active_partner_numerators=maximum(active_partner),
        proposal_role_numerators=maximum(proposal_role)),active_full,active_partner


def bound_record(needs,truth):
    canonical,full,partner=local_counts(needs,truth,LAYOUTS[0]);n=len(needs)
    all_layouts=[]
    for layout in LAYOUTS:
        count,_,_=local_counts(needs,truth,layout)
        assert count==canonical
        all_layouts.append(dict(layout=list(layout),**count))
    details=[]
    for actor in range(3):
        details.append(dict(actor=actor,by_own_need=[dict(own_need=need,
            full_action_counts={str(k):v for k,v in sorted(full[actor][need].items())},
            active_partner_counts={str(k):v for k,v in sorted(partner[actor][need].items())},
            max_full_count=max(full[actor][need].values(),default=0),
            max_partner_count=max(partner[actor][need].values(),default=0)) for need in range(24)]))
    return dict(need_count=n,truth_pair_counts={''.join('ABC'[a] for a in pair):sum(truth[x][:2]==pair for x in needs) for pair in PAIRS},
        **canonical,full_success_upper=str(Fraction(sum(canonical['active_full_numerators']),2*n)),
        executed_pair_correct_upper=str(Fraction(sum(canonical['active_partner_numerators']),2*n)),
        old_all_proposal_roles_correct_upper=str(Fraction(min(canonical['proposal_role_numerators']),n)),
        layout_checks=all_layouts,local_count_tables=details)


def main():
    start=time.perf_counter();assert digest(PREPARED)==PREPARED_SHA
    prepared=json.loads(PREPARED.read_text());parts=prepared['partitions']
    alltruth={}
    for needs in product(range(24),repeat=3):
        plans=full_plans(needs)
        if len(plans)==1:alltruth[needs]=plans[0]
    assert len(alltruth)==5376
    train=[tuple(n) for n in parts['train']['needs']]
    held=[tuple(n) for n in parts['new_needs']['needs']]
    assert len(train)==len(set(train))==3888 and len(held)==len(set(held))==1488
    assert not set(train)&set(held) and set(train)|set(held)==set(alltruth)
    cartesian=[]
    for part,spec in parts.items():
        ns=held if part in ('new_needs','new_needs_and_layouts') else train
        assert spec['needs']==[list(n) for n in ns]
        assert len(spec['layouts'])==len(set(map(tuple,spec['layouts'])))
        assert set(map(tuple,spec['layouts']))<=set(LAYOUTS)
        assert spec['private_sites']==[list(p) for p in OWNERS]
        assert spec['world_count']==len(ns)*len(spec['layouts'])*len(OWNERS)
        cartesian.append(dict(partition=part,needs=len(ns),layouts=len(spec['layouts']),owners=len(OWNERS),worlds=spec['world_count'],
            independent_public_backgrounds_per_need=len(spec['layouts'])*len(OWNERS)))
    bounds={name:bound_record(needs,alltruth) for name,needs in [('train',train),('heldout',held),('full',sorted(alltruth))]}
    assert bounds['heldout']['active_full_numerators']==[304]*3
    assert bounds['heldout']['active_partner_numerators']==[496]*3
    assert bounds['heldout']['full_success_upper']=='19/62'
    assert bounds['heldout']['executed_pair_correct_upper']=='1/2'
    assert bounds['heldout']['old_all_proposal_roles_correct_upper']=='23/62'
    checks=0;executable=0;fixture_records=[]
    fixtures=((12,12,15),(12,15,12),(15,12,12))
    for needs in fixtures:
        pair=alltruth[needs][:2];i,j,m,d=alltruth[needs];target={i:encode(i,j,m,d),j:encode(j,i,m,d)}
        full_count=pair_count=proposal_count=0
        for actions in product(range(17),repeat=3):
            actual,units=settlement(needs,LAYOUTS[0],actions)
            full=units==2;correct=actual==pair
            active_correct=sum(actions[actor]==action for actor,action in target.items())
            partner_correct=sum(actions[actor]!=0 and decode(actor,actions[actor])[0]==(j if actor==i else i) for actor in pair)
            proposal=all((None if actions[actor]==0 else decode(actor,actions[actor])[0])==((j if actor==i else i) if actor in pair else None) for actor in range(3))
            assert 2*int(full)<=active_correct and 2*int(correct)<=partner_correct
            full_count+=full;pair_count+=correct;proposal_count+=proposal;checks+=1
            if needs==fixtures[0]:executable+=actual is not None
        assert full_count==17 and pair_count==136 and proposal_count==64
        fixture_records.append(dict(needs=list(needs),all_joint_actions_checked=17**3,full_success_joint_actions=full_count,
            correct_executed_pair_joint_actions=pair_count,all_proposal_roles_correct_joint_actions=proposal_count))
    assert executable==408
    # One executing pair may succeed even though the third proposer chose a peer.
    demonstration=[]
    for actions in ((1,1,1),(5,5,1),(1,2,1)):
        actual,units=settlement(fixtures[0],LAYOUTS[0],actions)
        demonstration.append(dict(needs=list(fixtures[0]),layout=list(LAYOUTS[0]),actions=list(actions),
            executed_pair=None if actual is None else list(actual),full_success=units==2,
            correct_executed_pair=actual==(0,1),old_all_proposal_roles_correct=False))
    result=dict(status='passed',at=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-start,
        source_sha256={str(Path(__file__).resolve()):digest(__file__),str(PREPARED):PREPARED_SHA},
        enumeration=dict(candidate_need_triples=24**3,unique_full_plan_need_triples=len(alltruth),
            need_partition_layout_combinations_checked=sum(len(g)*24 for g in (train,held,list(alltruth))),
            joint_action_fixtures=3,joint_actions_checked=checks,executable_joint_actions_per_world=executable),
        model_parameter_loads=0,neural_forward_calls=0,training_updates=0,
        independence='Only standard library; candidate kernel and all environment/learner numerical implementations are not imported.',
        applicability='Uniform Cartesian needs x layout x ownership; each silent actor observes only own need plus independent public background and actor identity; randomness independent of current private needs.',
        partition_cartesian_checks=cartesian,bounds=bounds,settlement_fixtures=fixture_records,metric_counterexamples=demonstration,
        exact_optimum_claimed=False)
    path=HERE/'independent_bounds_review_001.json';assert not path.exists()
    path.write_text(json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],elapsed_seconds=result['elapsed_seconds'],bounds={k:{key:v[key] for key in ('need_count','active_full_numerators','active_partner_numerators','proposal_role_numerators','full_success_upper','executed_pair_correct_upper','old_all_proposal_roles_correct_upper')} for k,v in bounds.items()},receipt=str(path)),ensure_ascii=False))


if __name__=='__main__':main()

"""Single-person need flips and necessary listener-action changes; no models."""
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
from hashlib import sha256
from itertools import combinations, permutations, product
from pathlib import Path
import json
import sys

HERE=Path(__file__).resolve().parent
STUDY=HERE.parent
ROOT=STUDY.parent.parent
sys.path.insert(0,str(ROOT))
from research_program.triadic_task import environment as env

AGENTS=('A','B','C')
AXES=('kind_wood_fiber','length_short_long','destination_L_R')
LAYOUTS=list(permutations(range(4)))
OWNERS=list(permutations((1,2,3)))


def sha(p):return sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
    with Path(p).open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')


def flips(need):
    r,d=divmod(need,3)
    yield ('kind_wood_fiber' if r<2 else 'length_short_long'),3*(r^1)+d
    if d in (0,1):yield 'destination_L_R',3*r+(1-d)


def role(action_set):
    if action_set=={0}:return 'always_wait'
    if 0 not in action_set:return 'always_participate'
    return 'wait_or_participate'


def semantic_action(agent,index,layout=(0,1,2,3)):
    action=dict(env.all_actions(agent)[index])
    if index:
        action['material']=list(env.MATERIALS[layout[env.SITES.index(action['site'])]])
    return action


def counts(rows):
    out={}
    for ecology,axis in product(('unique','multiple'),AXES):
        selected=[r for r in rows if r['ecology']==ecology and r['axis']==axis]
        listeners=[x for r in selected for x in r['listeners']]
        necessary=[x for x in listeners if x['full_success_action_sets_disjoint']]
        relation=[x for x in listeners if x['unique_participation_wait_switch']]
        out[ecology+':'+axis]={
            'legal_unordered_demand_pairs':len(selected),'legal_pair_listener_cases':len(listeners),
            'demand_pairs_with_at_least_one_disjoint_listener':sum(any(x['full_success_action_sets_disjoint'] for x in r['listeners']) for r in selected),
            'disjoint_listener_cases':len(necessary),
            'unique_edge_changed_demand_pairs':sum(r['unique_compatible_pair_changed'] for r in selected),
            'unique_participation_wait_switch_listener_cases':len(relation),
            'disjoint_without_participation_wait_switch':sum(not x['unique_participation_wait_switch'] for x in necessary),
            'disjoint_by_listener':{a:sum(x['listener']==a for x in necessary) for a in AGENTS},
            'disjoint_by_changed_person':{a:sum(sum(x['full_success_action_sets_disjoint'] for x in r['listeners']) for r in selected if r['changed_person']==a) for a in AGENTS},
        }
    return out


def main():
    assert not (HERE/'results.json').exists() and not (HERE/'demand_pairs.json').exists()
    prepared_path=STUDY/'design_audit_002/prepared_inspected.json'
    audit_path=STUDY/'design_audit_002/verification.json'
    support_path=STUDY/'design_static_001/weighted_semantic_support.json'
    source_paths=(Path(__file__),Path(env.__file__),prepared_path,audit_path,support_path,STUDY/'design.py')
    sources={str(p):sha(p) for p in source_paths}
    p=read(prepared_path);assert read(audit_path)['design_sha256']==sha(STUDY/'design.py')
    domains={e:{tuple(n) for layer in p['common_destination_layers'] for n in layer['supports'][e]} for e in ('unique','multiple')}
    common_D={tuple(layer['destinations']) for layer in p['common_destination_layers']}
    weights={e:{tuple(row['needs']):Fraction(row['semantic_world_weight']['fraction']) for row in data} for e,data in read(support_path).items()}
    assert all(set(weights[e])==domains[e] and sum(weights[e].values())==1 for e in domains)
    actions={a:env.all_actions(a) for a in AGENTS}
    plans=[]
    for i,j in combinations(range(3),2):
        for site,dest in product(env.SITES,env.DESTINATIONS):
            joint={a:{'kind':'wait'} for a in AGENTS}
            for active,partner in ((i,j),(j,i)):
                joint[AGENTS[active]]={'kind':'transport','site':site,'destination':dest,'partner':AGENTS[partner]}
            plans.append({'pair':AGENTS[i]+AGENTS[j],'actions':joint,'indices':[actions[a].index(joint[a]) for a in AGENTS]})
    assert len(plans)==len({tuple(x['indices']) for x in plans})==24
    all_needs=sorted(set().union(*domains.values()))
    full={};projections={};edges={};settlements=0
    for needs in all_needs:
        state=env.State(needs,(0,1,2,3),(1,2,3))
        winning=[]
        for k,plan in enumerate(plans):
            outcome=env.settle(state,plan['actions'],require_match=True);settlements+=1
            assert sum(x['executed'] for x in outcome['individual_feedback'].values())==2
            if outcome['reward']==1:winning.append(k)
        assert winning
        full[needs]=winning
        projections[needs]=[{plans[k]['indices'][a] for k in winning} for a in range(3)]
        edges[needs]=sorted({plans[k]['pair'] for k in winning})
        assert edges[needs]==[''.join(AGENTS[a] for a in pair) for pair in env.compatible_pairs(needs)]
    transforms={}
    for layout in LAYOUTS:
        transforms[layout]={}
        for agent in AGENTS:
            mapped=[0]
            for action in actions[agent][1:]:
                target=dict(action);material=env.SITES.index(action['site'])
                target['site']=env.SITES[layout.index(material)]
                mapped.append(actions[agent].index(target))
            assert sorted(mapped)==list(range(17))
            transforms[layout][agent]=mapped
    rows=[];excluded=[];not_applicable=Counter();seen=set();observation_checks=layout_set_checks=0
    for ecology in ('unique','multiple'):
        for source in sorted(domains[ecology]):
            for who in range(3):
                if source[who]%3==2:not_applicable[ecology]+=1
                for axis,newneed in flips(source[who]):
                    target=list(source);target[who]=newneed;target=tuple(target)
                    if target not in domains[ecology]:
                        excluded.append({'ecology':ecology,'axis':axis,'changed_person':AGENTS[who],
                            'source_needs':list(source),'target_needs':list(target),
                            'target_compatible_edges':len(env.compatible_pairs(target)),
                            'reason':'target_D_not_in_common_21' if tuple(n%3 for n in target) not in common_D else 'target_not_in_same_ecology'})
                        continue
                    left,right=sorted((source,target));key=(ecology,axis,who,left,right)
                    if key in seen:continue
                    seen.add(key)
                    pair_changed=ecology=='unique' and edges[left]!=edges[right]
                    row={'ecology':ecology,'axis':axis,'changed_person':AGENTS[who],
                        'needs_before':list(left),'needs_after':list(right),
                        'D_before':[n%3 for n in left],'D_after':[n%3 for n in right],
                        'endpoint_target_need_weights':[str(weights[ecology][n]) for n in (left,right)],
                        'compatible_pairs_before':edges[left],'compatible_pairs_after':edges[right],
                        'full_plan_ids_before':full[left],'full_plan_ids_after':full[right],
                        'unique_compatible_pair_changed':pair_changed,'listeners':[]}
                    assert sum(a!=b for a,b in zip(left,right))==1
                    for listener in range(3):
                        if listener==who:continue
                        before,after=projections[left][listener],projections[right][listener]
                        disjoint=not(before & after)
                        switches=pair_changed and {role(before),role(after)}=={'always_wait','always_participate'}
                        assert not pair_changed or switches
                        a,b=(env.State(n,(0,1,2,3),(1,2,3)) for n in (left,right))
                        assert env.observe(a,AGENTS[listener],shared_needs=False)==env.observe(b,AGENTS[listener],shared_needs=False)
                        observation_checks+=1
                        for layout,by_agent in transforms.items():
                            permutation=by_agent[AGENTS[listener]]
                            assert (not({permutation[x] for x in before}&{permutation[x] for x in after}))==disjoint
                            layout_set_checks+=1
                        row['listeners'].append({'listener':AGENTS[listener],
                            'successful_actions_before':sorted(before),'successful_actions_after':sorted(after),
                            'intersection':sorted(before & after),'full_success_action_sets_disjoint':disjoint,
                            'role_before':role(before),'role_after':role(after),
                            'unique_participation_wait_switch':switches,'PI_observation_identical':True})
                    if ecology=='multiple':assert not any(x['full_success_action_sets_disjoint'] for x in row['listeners'])
                    rows.append(row)
    rows.sort(key=lambda x:(x['ecology'],x['axis'],x['needs_before'],x['needs_after'],x['changed_person']))
    for i,row in enumerate(rows):row['pair_id']=f'need_pair_{i:05d}'
    summary=counts(rows)
    physical_rows=[]
    train_layouts={tuple(x) for x in p['partitions']['unique']['train']['layouts']}
    for layout,owner in product(LAYOUTS,OWNERS):
        physical_rows.append({'layout':list(layout),'private_sites':list(owner),
            'partition':'train' if layout in train_layouts else 'heldout_layouts',
            'counts_by_ecology_axis':summary})
    examples={}
    for ecology,axis in product(('unique','multiple'),AXES):
        selected=[r for r in rows if r['ecology']==ecology and r['axis']==axis]
        forced=[(r,l) for r in selected for l in r['listeners'] if l['full_success_action_sets_disjoint']]
        fixed=[(r,l) for r,l in forced if not l['unique_participation_wait_switch']]
        relation=[(r,l) for r,l in forced if l['unique_participation_wait_switch']]
        def example(items):
            if not items:return None
            r,l=items[0]
            return {'pair_id':r['pair_id'],'changed_person':r['changed_person'],'listener':l['listener'],
                'needs_before':r['needs_before'],'needs_after':r['needs_after'],
                'layout':[0,1,2,3],'private_sites':[1,2,3],
                'successful_actions_before':[semantic_action(l['listener'],a) for a in l['successful_actions_before']],
                'successful_actions_after':[semantic_action(l['listener'],a) for a in l['successful_actions_after']],
                'unique_participation_wait_switch':l['unique_participation_wait_switch']}
        examples[ecology+':'+axis]={'first_legal_pair_id':selected[0]['pair_id'] if selected else None,
            'first_disjoint_listener_example':example(forced),
            'first_disjoint_without_role_switch':example(fixed),'first_role_switch':example(relation)}
    def expand_counts(value):
        return {k:expand_counts(v) for k,v in value.items()} if isinstance(value,dict) else value*144
    full_counts=expand_counts(summary)
    # Denominators below are unordered endpoint pairs; each future directed
    # intervention would have two directions. No intervention is executed here.
    outputs={
        'demand_pairs.json':rows,'excluded_directed_candidates.json':excluded,
        'layout_owner_counts.json':physical_rows,'lexicographic_examples.json':examples,
        'structural_plans_and_action_index.json':{'canonical_layout':[0,1,2,3],
            'canonical_private_sites':[1,2,3],'all_24_plans':plans,'agent_action_menus':actions}}
    for name,value in outputs.items():write(HERE/name,value)
    assert all(sha(path)==digest for path,digest in sources.items())
    result={'status':'completed_static_only','created_at_utc':datetime.now(timezone.utc).isoformat(),
        'sources_sha256':sources,'support_need_counts':{e:len(v) for e,v in domains.items()},
        'legal_unordered_demand_pairs':len(rows),'candidate_listener_cases':2*len(rows),
        'excluded_directed_candidates':len(excluded),
        'excluded_by_ecology_axis_reason':{str(k):v for k,v in sorted(Counter((r['ecology'],r['axis'],r['reason']) for r in excluded).items())},
        'destination_L_or_R_not_applicable_sender_observations':dict(not_applicable),
        'canonical_counts':summary,'world_expansion_multiplier':144,
        'world_pair_counts':full_counts,
        'partition_multipliers':{'train':108,'heldout_layouts':36},
        'checks':{'native_canonical_settlements':settlements,'all_24_physical_plans_per_need':True,
            'canonical_listener_observation_equal_checks':observation_checks,'layout_action_bijection_checks':layout_set_checks,
            'owner_count':6,'owner_invariance':'Physical success ignores owner; listener own need and full layout/owner remain fixed.',
            'multiple_disjoint_listener_count_is_zero':True},
        'pair_weighting':'No intervention weighting chosen; both endpoints original target weights retained. Uniform pair counts are not claimed to follow the original state distribution.',
        'unit':'Unordered legal demand pair; listener cases separate; all24 layouts and6 owners expansion. Reverse-directed probes are not executed.',
        'model_forward_calls':0,'training_updates':0,'solver_calls':0,'trained_actions_read':False,
        'outputs_sha256':{name:sha(HERE/name) for name in outputs}}
    write(HERE/'results.json',result)
    print(json.dumps({'status':'completed_static_only','demand_pairs':len(rows),'canonical_counts':summary},ensure_ascii=False))


if __name__=='__main__':main()

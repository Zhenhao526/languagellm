"""Prespecified nonmatching-support design: finite arithmetic, no learning."""
import collections
import hashlib
import itertools
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
MAPS=list(itertools.permutations(range(6),2));UNIVERSE=set(MAPS)
H={(i,(i+s)%6) for i in range(6) for s in (-1,1)}
Q_CONNECTED=(2,3,4,5,0,1)
Q_BLOCKED=(2,5,4,1,0,3)
RELABELINGS=((0,1,2,3,4,5),(0,2,1,4,3,5),(0,3,1,5,2,4))


def cycles(q):
    left=set(range(6));out=[]
    while left:
        seq=[];x=min(left)
        while x not in seq:seq.append(x);x=q[x]
        assert x==seq[0]
        left-=set(seq);out.append(seq)
    return out


def conjugate(q,r):
    result=[None]*6
    for i in range(6):result[r[i]]=r[q[i]]
    return tuple(result)


def graph(t,h):
    a=[[int((i,j) in t) for j in range(6)] for i in range(6)]
    adjacency={i:set() for i in range(12)}
    for i,j in t:adjacency[i].add(6+j);adjacency[6+j].add(i)
    unseen=set(range(12));components=[]
    while unseen:
        todo=[min(unseen)];component=set()
        for x in todo:
            if x in component:continue
            component.add(x);todo.extend(adjacency[x]-component)
        unseen-=component;components.append(sorted(component))
    target=[]
    for i,j in sorted(h):
        distances={i:0};todo=[i]
        for x in todo:
            for y in adjacency[x]:
                if y not in distances:distances[y]=distances[x]+1;todo.append(y)
        three=sum(a[i][b]*a[c][b]*a[c][j] for b in range(6) for c in range(6))
        target.append(dict(pair=[i,j],map_id=MAPS.index((i,j)),shortest_path=distances.get(6+j),three_step_paths=three))
    overlap=[sum(a[i][k]*a[j][k] for k in range(6)) for i in range(6) for j in range(i)]
    return dict(adjacency=a,training_pairs=sorted(t),training_map_ids=[m for m,pair in enumerate(MAPS) if pair in t],
        row_degrees=[sum(row) for row in a],column_degrees=[sum(a[i][j] for i in range(6)) for j in range(6)],
        components=components,component_sizes=sorted(map(len,components)),
        reciprocal_directed_edges=sum((j,i) in t for i,j in t),
        directed_triangles=sum((i,j) in t and (j,k) in t and (k,i) in t for i,j,k in itertools.permutations(range(6),3))//3,
        four_cycles=sum(n*(n-1)//2 for n in overlap),
        target_paths=target,
        target_shortest_path_counts=dict(collections.Counter('unreachable' if r['shortest_path'] is None else str(r['shortest_path']) for r in target)),
        target_three_step_count_histogram=dict(collections.Counter(str(r['three_step_paths']) for r in target)))


def bounds(h):
    # Exact finite receiver enumeration for each observation, not simulation.
    choices=list(itertools.product(range(6),repeat=2));targets=sorted(h)
    def score(rows,action):
        cf=sum(f==action[0] for f,w in rows);cw=sum(w==action[1] for f,w in rows)
        joint=sum((f,w)==action for f,w in rows)
        return joint,cf+cw,(cf+cw)/4+joint/2
    out={}
    for name,fn in [('no_information',lambda pair:0),('fixed_food_location',lambda pair:pair[0]),('fixed_water_location',lambda pair:pair[1])]:
        subsets=collections.defaultdict(list)
        for pair in targets:subsets[fn(pair)].append(pair)
        sums=[0.,0.,0.]
        for rows in subsets.values():
            values=[score(rows,action) for action in choices]
            for k in range(3):sums[k]+=max(v[k] for v in values)
        out[name]=dict(joint_success=sums[0]/len(targets),mean_single_success=sums[1]/(2*len(targets)),mixed_reward=sums[2]/len(targets))
    assert out['no_information']==dict(joint_success=1/12,mean_single_success=1/6,mixed_reward=1/8)
    for name in ('fixed_food_location','fixed_water_location'):
        assert out[name]==dict(joint_success=.5,mean_single_success=.75,mixed_reward=.625)
    # A sender that sees the WHOLE map can choose which role/location to report.
    # Find a one-to-one edge->incident endpoint assignment (12 endpoints total).
    assignment=[]
    def search(k,used):
        if k==len(targets):return True
        f,w=targets[k]
        for endpoint in ((0,f),(1,w)):
            if endpoint not in used:
                assignment.append(endpoint)
                if search(k+1,used|{endpoint}):return True
                assignment.pop()
        return False
    assert search(0,set()) and len(set(assignment))==12
    out['adaptive_role_choice_counterexample']=dict(joint_success=1.,
        assumption='Sender uses both target positions to choose which resource endpoint to report; receiver stores a12-entry dictionary.',
        witness=[dict(target=list(pair),chosen_role='food' if endpoint[0]==0 else 'water',chosen_position=endpoint[1]) for pair,endpoint in zip(targets,assignment)])
    return out


def main():
    allowed=UNIVERSE-H;enumeration=[]
    for q in itertools.permutations(range(6)):
        removed={(i,q[i]) for i in range(6)}
        if not removed<=allowed:continue
        t=allowed-removed;g=graph(t,H)
        enumeration.append(dict(q=q,cycle_lengths=sorted(map(len,cycles(q))),component_sizes=g['component_sizes'],reciprocal_directed_edges=g['reciprocal_directed_edges']))
    assert len(enumeration)==20
    panels=[]
    for number,r in enumerate(RELABELINGS,1):
        h={(r[i],r[j]) for i,j in H};arms={}
        for arm,q in [('connected_cycle12',Q_CONNECTED),('three_products',Q_BLOCKED)]:
            cq=conjugate(q,r);remaining={(i,cq[i]) for i in range(6)};t=UNIVERSE-h-remaining
            g=graph(t,h)
            assert len(t)==len(h)==12 and len(remaining)==6
            assert not (t&h or t&remaining or h&remaining)
            assert g['row_degrees']==g['column_degrees']==[2]*6
            assert g['reciprocal_directed_edges']==6 and g['directed_triangles']==2
            assert sorted(map(len,cycles(cq)))==[3,3]
            arms[arm]=dict(q=cq,remaining_untrained_pairs=sorted(remaining),remaining_untrained_map_ids=[MAPS.index(pair) for pair in sorted(remaining)],**g)
        c,d=arms.values();ct=set(map(tuple,c['training_pairs']));dt=set(map(tuple,d['training_pairs']))
        assert c['component_sizes']==[12] and d['component_sizes']==[4,4,4]
        assert c['target_shortest_path_counts']=={'5':6,'3':6}
        assert d['target_shortest_path_counts']=={'unreachable':12}
        common=ct&dt;unseen=UNIVERSE-ct-dt
        assert len(common)==9 and len(unseen)==15
        panels.append(dict(panel=number,coordinate_permutation=r,common_test_pairs=sorted(h),common_test_map_ids=[MAPS.index(pair) for pair in sorted(h)],
            conditions=arms,shared_training_map_ids=[MAPS.index(pair) for pair in sorted(common)],
            unique_training_map_ids={name:sorted(set(g['training_map_ids'])-{MAPS.index(pair) for pair in common}) for name,g in arms.items()},
            all_common_untrained_map_ids=[MAPS.index(pair) for pair in sorted(unseen)],
            extra_common_untrained_map_ids=[MAPS.index(pair) for pair in sorted(unseen-h)]))
    categories=collections.Counter((tuple(r['component_sizes']),r['reciprocal_directed_edges']) for r in enumeration)
    project=ROOT.parents[1]
    sources=[project/'redesign_v0.28/world.py',project/'redesign_v0.21/social_model.py',project/'redesign_v0.29/support.py']
    result=dict(status='finite_design_not_training',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        universe=MAPS,common_test_definition='H={(i,(i-1)mod6),(i,(i+1)mod6):i=0..5}',
        train_definition='T=(legal30 minus H) minus perfect matching Q',
        enumeration_count=20,enumeration_classes=[dict(component_sizes=k[0],reciprocal_directed_edges=k[1],count=v) for k,v in sorted(categories.items())],
        enumeration=enumeration,panels=panels,information_bounds=bounds(H),
        invariant_information_nats=dict(train_H_F='ln6',train_H_W='ln6',train_H_FW='ln12',train_H_W_given_F='ln2',train_MI='ln3'),
        fixed_components=dict(conditions=2,panels=3,formal_sources_if_reused=4,new_communication_runs_if_executed=24,new_private_training=0),
        existing_interface_sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        interpretation_limits=['Connectivity, cycle decomposition, local product completeness and target reachability covary.',
            'All H targets cross components in the three-products training support.',
            'J>.5 rules out only a fixed single-resource signal with map-independent nuisance; adaptive reporting can encode both positions.',
            '49 complete messages remain enough to memorize all30 worlds; task success is not token factorization.',
            'Private all has seen all30 layouts; novelty is limited to communication experience.',
            'Three panels are coordinate repetitions, and inherited development sources are not independent confirmation.'])
    target=ROOT/'nonmatching_candidates.json';target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(path=str(target),legal_q=20,selected_conditions=2,panels=3,shared_train=9,unique_each=3,shared_untrained=15,common_primary_test=12)))

if __name__=='__main__':main()

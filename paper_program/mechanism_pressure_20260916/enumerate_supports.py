"""Exact finite support enumeration only: no images, models or training."""
import collections
import hashlib
import itertools
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
MAPS=list(itertools.permutations(range(6),2))
P=(1,0,3,2,5,4)
Q3=(2,3,4,5,0,1)
Q2=(2,4,5,1,3,0)
COORDINATES=((0,1,2,3,4,5),(0,2,1,4,3,5),(0,3,1,5,2,4))


def matrix(p,q):
    return [[int(i!=j and j not in (p[i],q[i])) for j in range(6)] for i in range(6)]


def cycle_type(p):
    unseen=set(range(6));lengths=[]
    while unseen:
        x=min(unseen);component=[]
        while x not in component:
            component.append(x);x=p[x]
        assert x==component[0]
        unseen-=set(component);lengths.append(len(component))
    return sorted(lengths)


def properties(a,p,q):
    unseen=set(range(12));components=[]
    while unseen:
        stack=[min(unseen)];component=set()
        while stack:
            x=stack.pop()
            if x in component:continue
            component.add(x)
            stack.extend([6+j for j in range(6) if a[x][j]] if x<6 else [i for i in range(6) if a[i][x-6]])
        unseen-=component;components.append(sorted(component))
    overlaps=[sum(a[i][k]*a[j][k] for k in range(6)) for i in range(6) for j in range(i)]
    paths=[[sum(a[i][b]*a[c][b]*a[c][j] for b in range(6) for c in range(6)) for j in range(6)] for i in range(6)]
    row_degrees=[sum(row) for row in a];column_degrees=[sum(a[i][j] for i in range(6)) for j in range(6)]
    edges=[(i,j) for i,j in MAPS if a[i][j]]
    assert len(edges)==18 and row_degrees==column_degrees==[3]*6
    return dict(adjacency=a,training_pairs=edges,training_map_ids=[MAPS.index(e) for e in edges],
        row_degree=row_degrees,column_degree=column_degrees,components=components,
        reciprocal_directed_edges=sum(a[i][j]*a[j][i] for i in range(6) for j in range(6)),
        four_cycles=sum(s*(s-1)//2 for s in overlaps),
        food_context_overlap_counts=dict(sorted(collections.Counter(overlaps).items())),
        water_context_overlap_counts=dict(sorted(collections.Counter(sum(a[k][i]*a[k][j] for k in range(6)) for i in range(6) for j in range(i)).items())),
        directed_triangles=sum(a[i][j]*a[j][k]*a[k][i] for i,j,k in itertools.permutations(range(6),3))//3,
        target_three_paths=[paths[i][p[i]] for i in range(6)],other_held_three_paths=[paths[i][q[i]] for i in range(6)],
        target_pairs=[(i,p[i]) for i in range(6)],target_map_ids=[MAPS.index((i,p[i])) for i in range(6)],
        other_held_pairs=[(i,q[i]) for i in range(6)],other_held_map_ids=[MAPS.index((i,q[i])) for i in range(6)],
        removed_permutation_cycle_type=cycle_type(q))


def conjugate(p,r):
    result=[None]*6
    for i in range(6):result[r[i]]=r[p[i]]
    return tuple(result)


def main():
    candidates=[]
    for q in itertools.permutations(range(6)):
        if any(q[i] in (i,P[i]) for i in range(6)):continue
        a=matrix(P,q);prop=properties(a,P,q)
        candidates.append(dict(q=q,**prop))
    counts=collections.Counter((c['four_cycles'],c['reciprocal_directed_edges'],tuple(sorted(c['target_three_paths']))) for c in candidates)
    assert len(candidates)==80
    panels=[]
    for number,r in enumerate(COORDINATES,1):
        p=conjugate(P,r);conditions={}
        for name,q in [('target_paths3',Q3),('target_paths2',Q2)]:
            cq=conjugate(q,r);conditions[name]=dict(q=cq,**properties(matrix(p,cq),p,cq))
        left=conditions['target_paths3'];right=conditions['target_paths2']
        shared_train=sorted(set(left['training_map_ids'])&set(right['training_map_ids']))
        common_unseen=sorted(set(range(30))-set(left['training_map_ids'])-set(right['training_map_ids']))
        assert len(shared_train)==13 and len(common_unseen)==7
        assert left['target_three_paths']==[3]*6 and right['target_three_paths']==[2]*6
        assert left['four_cycles']==3 and right['four_cycles']==6
        assert left['reciprocal_directed_edges']==right['reciprocal_directed_edges']==12
        assert len(left['components'])==len(right['components'])==1
        assert left['directed_triangles']==right['directed_triangles']==8
        assert left['removed_permutation_cycle_type']==right['removed_permutation_cycle_type']==[3,3]
        assert set(left['target_map_ids'])<=set(common_unseen)
        panels.append(dict(panel=number,coordinate_permutation=r,p=p,conditions=conditions,
            shared_training_map_ids=shared_train,all_common_unseen_map_ids=common_unseen,
            extra_common_unseen_map_ids=sorted(set(common_unseen)-set(left['target_map_ids']))))
    a,b=matrix(P,Q3),matrix(P,Q2)
    equivalences=[r for r in itertools.permutations(range(6)) if all(r[P[i]]==P[r[i]] for i in range(6))
                  and all(a[i][j]==b[r[i]][r[j]] for i in range(6) for j in range(6))]
    assert not equivalences
    old_matchings=(((0,1),(2,3),(4,5)),((0,2),(1,4),(3,5)),((0,3),(1,5),(2,4)))
    historical=[]
    for ix in range(3):
        removals={edge for matching in (old_matchings[ix],old_matchings[(ix+1)%3]) for f,w in matching for edge in ((f,w),(w,f))}
        aa=[[int(i!=j and (i,j) not in removals) for j in range(6)] for i in range(6)]
        triangles=sum(aa[i][j]*aa[j][k]*aa[k][i] for i,j,k in itertools.combinations(range(6),3))
        historical.append(dict(partition=ix+1,adjacency=aa,undirected_triangles=triangles,
            map_ids=[index for index,(i,j) in enumerate(MAPS) if aa[i][j]],degree=[sum(row) for row in aa]))
    historical_equivalent=[any(all(historical[0]['adjacency'][i][j]==item['adjacency'][r[i]][r[j]] for i in range(6) for j in range(6)) for r in itertools.permutations(range(6))) for item in historical]
    assert historical_equivalent==[True]*3
    result=dict(status='exact_enumeration_no_training',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        universe=MAPS,definition='A[i,j]=1 iff i!=j and j!=P[i] and j!=Q[i]',
        number_valid_q=80,enumeration=[dict(four_cycles=k[0],reciprocal_directed_edges=k[1],sorted_target_three_paths=k[2],count=n) for k,n in sorted(counts.items())],
        candidates=candidates,panels=panels,preserving_P_coordinate_equivalences=equivalences,
        historical_v23_v28=historical,historical_common_site_relabel_equivalence=historical_equivalent,
        exact_common_metrics=dict(layout_count=18,food_and_water_degree=3,train_joint_no_message_best='1/18',train_single_no_message_best='1/6',
            train_mixed_reward_no_message_best='1/9',target_six_joint_no_message_best='1/6',
            training_information_nats=dict(H_F='ln6',H_W='ln6',H_FW='ln18',H_W_given_F='ln3',I_FW='ln2')),
        limitations=['Fixed common six targets form a matching, so either resource location identifies the other within that evaluation subset.',
            'Changing all twelve common unseen pairs while retaining exactly eighteen complementary training pairs is impossible.',
            'Cycle count, overlap coverage and target three-path counts change together; no isolated cycle mechanism is identified.',
            'Three panels are coordinate repetitions, not independent training seeds.'])
    target=ROOT/'support_enumeration.json';target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(path=str(target),valid_q=80,panels=3,shared_train=13,shared_unseen=7,primary_unseen=6,coordinate_equivalences=len(equivalences))))

if __name__=='__main__':main()

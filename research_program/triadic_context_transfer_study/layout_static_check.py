"""Finite layout/visibility proof check. No policies or trained results read."""
from pathlib import Path
from itertools import permutations,combinations
from collections import Counter
from hashlib import sha256
import json

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'triadic_action_dependency_study/results/context_001/prepared.json'

def check():
    prepared=json.loads(SOURCE.read_text());result=[]
    for part in ('train','new_layouts'):
        layouts=prepared['partitions'][part]['layouts'];n=len(layouts)
        edges=[[j for j in range(n) if all(x!=y for x,y in zip(layouts[i],layouts[j]))] for i in range(n)]
        match={}
        def augment(i,seen):
            for j in edges[i]:
                if j in seen:continue
                seen.add(j)
                if j not in match or augment(match[j],seen):match[j]=i;return True
            return False
        for i in range(n):augment(i,set())
        row=dict(partition=part,layouts=layouts,all_four_sites_changed_edges=edges,
            maximum_matching_size=len(match),matching_rows_to_donors={str(i):j for j,i in sorted(match.items())})
        if n==6:
            perfect=[list(p) for p in permutations(range(n)) if all(p[i] in edges[i] for i in range(n))]
            row.update(bijections_enumerated=720,perfect_bijections=perfect,hall_witnesses=[])
            for size in range(1,n+1):
                for selected in combinations(range(n),size):
                    neighbors=sorted(set(j for i in selected for j in edges[i]))
                    if len(neighbors)<len(selected):row['hall_witnesses'].append(dict(rows=list(selected),neighbors=neighbors,deficit=len(selected)-len(neighbors)))
            assert not perfect and len(match)==4 and edges[0]==[]
        else:assert len(match)==18
        result.append(row)
    # Exhaust all nonidentical layouts and six public owner assignments.
    # Each case has an ordered sender/listener plus the other non-sender.
    layouts=list(permutations(range(4)));stats=Counter();by_distance={d:Counter() for d in (2,3,4)}
    for first,second in permutations(layouts,2):
        distance=sum(x!=y for x,y in zip(first,second))
        for owner in permutations((1,2,3)):
            same=[all(first[k]==second[k] for k in (0,owner[a])) for a in range(3)]
            for sender in range(3):
                listeners=[a for a in range(3) if a!=sender]
                assert not all(same[a] for a in listeners)
                stats['sender_layout_owner_cases']+=1
                stats['both_non_sender_visible_observations_unchanged']+=int(all(same[a] for a in listeners))
                for listener in listeners:
                    stats['ordered_sender_listener_layout_owner_cases']+=1
                    stats['listener_visible_observation_unchanged']+=int(same[listener])
                    by_distance[distance]['ordered_sender_listener_cases']+=1
                    by_distance[distance]['listener_observation_unchanged']+=int(same[listener])
                    if same[listener]:assert distance==2
    return dict(status='passed_static_only',source_sha256={str(SOURCE):sha256(SOURCE.read_bytes()).hexdigest(),
        str(Path(__file__).resolve()):sha256(Path(__file__).read_bytes()).hexdigest()},partitions=result,
        general_nonself_visibility=dict(stats),by_hamming_distance={str(k):dict(v) for k,v in by_distance.items()},
        inference_calls=0,parameter_loads=0,training_calls=0,trained_outputs_read=False,
        no_donor_mapping_selected=True,interpretation='A designated LL listener can retain the same view only under a two-site change. The two non-senders cannot both retain the same view under any non-self permutation. This does not remove the separate endpoint-swap identity for both-window interventions.')

if __name__=='__main__':
    output=HERE/'layout_static_check.json';assert not output.exists()
    data=check();output.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:data[k] for k in ('status','general_nonself_visibility','by_hamming_distance')},ensure_ascii=False))

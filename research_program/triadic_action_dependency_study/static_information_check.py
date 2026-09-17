"""Independent finite-set derivations only; no model, training or solver."""
from itertools import product, combinations
from collections import Counter, defaultdict
from pathlib import Path
from hashlib import sha256
from fractions import Fraction
from datetime import datetime, timezone
import json

ROOT=Path(__file__).resolve().parents[2]
PAIRS=((0,1),(0,2),(1,2))
PARTIAL=(frozenset((0,1)),frozenset((2,3)),frozenset((0,2)),frozenset((1,3)))
EXACT=tuple(frozenset((m,)) for m in range(4))
DEST=((0,),(1,),(0,1))


def action_indices(plan):
    edge, local=divmod(plan,8); site,destination=divmod(local,2); i,j=PAIRS[edge]
    actions=[0,0,0]
    for a,b in ((i,j),(j,i)):
        peer=[x for x in range(3) if x!=a].index(b)
        actions[a]=1+4*site+2*destination+peer
    return actions


def flips(resource, style):
    if style=='partial' or style=='mixed' and resource<4:
        yield ('kind' if resource<2 else 'length'),resource^1
    else:
        offset=4 if style=='mixed' else 0
        material=resource-offset
        yield 'kind',offset+(material^2)
        yield 'length',offset+(material^1)


def analyze(resources,style):
    size=len(resources)*3
    accept=[sum(1<<(2*m+d) for m in r for d in DEST[k]) for r in resources for k in range(3)]
    plans={}; edges={}; hist=Counter(); single={}; unique={}; totals=Counter()
    for need in product(range(size),repeat=3):
        bitsets=[accept[need[i]]&accept[need[j]] for i,j in PAIRS]
        p=sum(mask<<(8*k) for k,mask in enumerate(bitsets))
        e=sum(bool(x) for x in bitsets); hist[e]+=1; plans[need]=p; edges[need]=e
        if e==1: unique[need]=p
        if p.bit_count()==1: single[need]=p
    support_stats={}
    for support_name,support in (('all_solvable',{n:p for n,p in plans.items() if p}),('single_full_plan',single)):
        counts=Counter(); cardinality=defaultdict(Counter); examples={}; ordered=Counter(); pairs_seen=set()
        for before,p0 in support.items():
            for sender in range(3):
                r,d=divmod(before[sender],3)
                changes=list((axis,3*rr+d) for axis,rr in flips(r,style))
                if d<2: changes.append(('destination',3*r+1-d))
                for axis,replacement in changes:
                    after=list(before);after[sender]=replacement;after=tuple(after)
                    if after not in support or not before<after:continue
                    pairs_seen.add((axis,sender,before,after))
                    p1=support[after]
                    projections=[]
                    for p in (p0,p1):
                        inds=[i for i in range(24) if p>>i&1]
                        projections.append([{action_indices(i)[a] for i in inds} for a in range(3)])
                    for listener in range(3):
                        if listener==sender:continue
                        a,b=projections[0][listener],projections[1][listener]
                        disjoint=not (a&b)
                        content=disjoint and 0 not in a and 0 not in b
                        role=disjoint and ((a=={0} and 0 not in b) or (b=={0} and 0 not in a))
                        group='content' if content else 'role' if role else 'other'
                        counts[axis,group]+=1
                        if content:
                            cardinality[axis][(len(a),len(b))]+=1; ordered[axis,sender,listener]+=1
                            examples.setdefault(axis,dict(needs_before=before,needs_after=after,sender=sender,listener=listener,
                                successful_actions=[sorted(a),sorted(b)],full_plan_count=[p0.bit_count(),p1.bit_count()]))
        support_stats[support_name]=dict(need_tables=len(support),unordered_demand_pairs=len(pairs_seen),
            listener_counts={a:{g:counts[a,g] for g in ('content','role','other')} for a in ('kind','length','destination')},
            content_cardinalities={a:{str(k):v for k,v in c.items()} for a,c in cardinality.items()},
            ordered_sender_listener_content={a:{f'{s}>{l}':ordered[a,s,l] for s in range(3) for l in range(3) if s!=l}
                for a in ('kind','length','destination')},first_content_examples=examples)
    # Necessary individual-role predictor yields a rigorous JOINT-role upper bound.
    role_counts=[defaultdict(Counter) for _ in range(3)]; pair_counts=Counter()
    for needs,mask in single.items():
        plan=mask.bit_length()-1; pair=PAIRS[plan//8];pair_counts[str(pair)]+=1
        for a in range(3):
            correct=0 if a not in pair else 1+[x for x in range(3) if x!=a].index(next(x for x in pair if x!=a))
            role_counts[a][needs[a]][correct]+=1
    upper=[]
    for a in range(3):
        numerator=sum(max(c.values()) for c in role_counts[a].values())
        upper.append(dict(agent=a,numerator=numerator,denominator=len(single),fraction=str(Fraction(numerator,len(single))),
            counts_by_own_need={str(k):dict(v) for k,v in role_counts[a].items()}))
    return dict(resource_sets=[sorted(r) for r in resources],demand_types=size,
        total_need_tables=size**3,compatible_edge_histogram=dict(hist),unique_pair_tables=len(unique),
        single_full_plan_tables=len(single),support_statistics=support_stats,
        uniform_single_plan_fixed_pair_counts=dict(pair_counts),
        uniform_single_plan_individual_role_Bayes_upper_bounds=upper)


def main():
    out=Path(__file__).with_suffix('.json')
    assert not out.exists(),'Do not overwrite static receipt'
    old=ROOT/'research_program/triadic_semantic_probe_study/dataset_001/cases.json'
    cases=json.loads(old.read_text())
    old_counts={a:dict(Counter(str(tuple(map(len,c['canonical_success_actions']))) for c in cases
        if c['classification']=='content' and c['axis']==a)) for a in
        ('kind_wood_fiber','length_short_long','destination_L_R')}
    report=dict(status='completed_static_enumeration',created_at=datetime.now(timezone.utc).isoformat(),
        source_sha256={str(p):sha256(p.read_bytes()).hexdigest() for p in (Path(__file__).resolve(),old,ROOT/'research_program/triadic_task/environment.py')},
        frozen_content_cardinality=old_counts,
        partial_only=analyze(PARTIAL,'partial'),exact_only=analyze(EXACT,'exact'),
        partial_plus_exact=analyze(PARTIAL+EXACT,'mixed'),
        canonical_layout=[0,1,2,3],layout_owner_multiplier=144,
        interpretation='Uniform all-demand-table counts only; not a frozen new training distribution. No actions are removed.',
        model_forwards=0,parameter_loads=0,training_updates=0,solver_calls=0)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    for name in ('partial_only','exact_only','partial_plus_exact'):
        r=report[name];s=r['support_statistics']['single_full_plan']
        print(name,json.dumps({k:r[k] for k in ('compatible_edge_histogram','single_full_plan_tables')}),
              'single_plan_content_role',s['listener_counts'],
              'role_upper',r['uniform_single_plan_individual_role_Bayes_upper_bounds'][0]['fraction'])


if __name__=='__main__':main()

"""Zero-policy structural and identification certificate for directed W1 study.

Uses only pinned prior independent native acceptance/case reconstruction.
No checkpoint, current policy output, model forward, or production new case code.
"""
from pathlib import Path
from datetime import datetime,timezone
from itertools import product
import numpy as np
from research_program.triadic_need_response_study import audit_response as a

HERE=Path(__file__).resolve().parent

def run():
    refs=a.references();source=a.ROOT/'research_program/triadic_action_dependency_study/results/context_001/prepared.json'
    a.require(a.sha(source)==a.ORIGINAL_SHA,'Frozen native support')
    rows={}
    for part,spec in a.read(source)['partitions'].items():
        c=a.build_cases(spec);needs=[tuple(n) for n in spec['needs']];lookup={n:i for i,n in enumerate(needs)}
        edges=np.asarray(c['edge_need_indices']);truth=np.asarray(c['target_pairs']);actors=c['changed_person'];axes=c['axis_index']
        index={(int(u),int(v),who,axis):k for k,((u,v),who,axis) in enumerate(zip(edges,actors,axes))}
        base=np.asarray([list(n)+spec['layouts'][0]+spec['private_sites'][0] for n in needs],np.int16)
        native=a.env.rewards(base);features=a.env.features(base,'PL');seen=set();groups=[]
        for k,((u,v),who,axis) in enumerate(zip(edges,actors,axes)):
            other=[j for j in range(3) if j!=who];third=a.prior.PAIRS.index(tuple(other))
            a.require(truth[k,0]!=truth[k,1] and third not in truth[k],'Both changed truth pairs contain changed actor')
            a.require(np.all(native[[u,v],third*8:third*8+8]!=1),'Other pair cannot attain full success at either end')
            mirrored=[]
            for endpoint in (u,v):
                n=list(needs[endpoint]);n[other[0]],n[other[1]]=n[other[1]],n[other[0]];mirrored.append(lookup[tuple(n)])
            mate=index[mirrored[0],mirrored[1],who,axis]
            a.require(mate!=k,'No fixed mirror edge');a.require(len({int(u),int(v),*mirrored})==4,'Four distinct need worlds')
            a.require(np.array_equal(truth[k],truth[mate][::-1]),'Demand-to-partner mapping reverses')
            a.require(np.array_equal(features[[u,v],who],features[mirrored,who]),'Sender PL view context invariant')
            for recipient in other:a.require(np.array_equal(features[u,recipient],features[v,recipient]),'Recipient observation invariant across sender own need')
            if k not in seen:
                a.require(mate not in seen,'Mirror disjointness');seen.update((k,mate));groups.append((k,mate,who,axis))
        a.require(len(seen)==len(edges) and 2*len(groups)==len(edges),'Complete unordered group support')
        L=len(spec['layouts']);O=len(spec['private_sites']);mapping=[(li+L//2)%L for li in range(L)]
        a.require(L%2==0 and all(mapping[li]!=li and mapping[mapping[li]]==li for li in range(L)),'Prespecified cross-layout no-fixed-point involution')
        counts=[sum(g[2:]==(who,axis) for g in groups) for who,axis in product(range(3),repeat=2)]
        expected=([20,20,16] if part in ('new_needs','new_needs_and_layouts') else [76,76,84])*3
        a.require(counts==expected,'All nine independently derived group strata')
        rows[part]=dict(need_edges=len(edges),mirror_groups=len(groups),group_strata=counts,backgrounds=L*O,
            group_backgrounds=len(groups)*L*O,main_cells_per_policy_time=len(groups)*L*O*8,
            native_sham_cells_per_policy_time=len(groups)*L*O*4,donor_layout_map=mapping,
            unchanged_other_pair_full_feasible_worlds=0,actor_feature_checks=len(edges)*2,recipient_feature_checks=len(edges)*2)
    # Concrete generic suppression counterexample: packet1 decreases both needs'
    # willingness to propose the sender, but does so more for need0-compatible x.
    # ctx0 recipients=(x,y), ctx1=(y,x); same packet pair reused unchanged.
    probabilities=np.array([[[.8,.2],[.4,.3]],[[.4,.3],[.8,.2]]]) # [context,recipient,packet]
    compatible=np.array([[0,1],[1,0]])
    delta=np.array([[probabilities[c,j,compatible[c,j]]-probabilities[c,j,1-compatible[c,j]] for j in range(2)] for c in range(2)])
    a.close(delta,np.array([[.6,-.1],[-.1,.6]]),'Counterexample signed effects')
    a.close(delta.mean(),.25,'D can be positive under monotone suppression');a.close(delta.min(),-.1,'L exposes missing bidirectional selectivity')
    # Any context-independent per-recipient packet effect cancels in D, with
    # opposite signs in each recipient's two contexts; min is never positive.
    fixed_examples=[]
    for effect0,effect1 in product((-.8,-.1,0.,.2,.7),repeat=2):
        effects=np.array([[effect0,-effect1],[-effect0,effect1]])
        a.close(effects.mean(),0.,'Context-independent null D');a.require(effects.min()<=0,'Context-independent null L')
        fixed_examples.append(dict(raw_effects=[effect0,effect1],D=float(effects.mean()),L=float(effects.min())))
    main=sum(r['main_cells_per_policy_time'] for r in rows.values());sham=sum(r['native_sham_cells_per_policy_time'] for r in rows.values())
    return dict(status='passed',at=datetime.now(timezone.utc).isoformat(),source_sha256=a.sha(__file__),
        original_prepared_sha256=a.sha(source),independent_helpers_sha256=refs,policy_output_reads=0,checkpoint_loads=0,network_forwards=0,
        partitions=rows,main_cells_per_policy_time=main,native_sham_cells_per_policy_time=sham,
        four_policies_two_times=dict(main_cells=main*8,native_sham_cells=sham*8,total_cells=(main+sham)*8,
            full_nine_head_upper_bound_module_samples=(main+sham)*8*9),
        monotone_suppression_counterexample=dict(probabilities=probabilities.tolist(),compatible_packet=compatible.tolist(),delta=delta.tolist(),D=float(delta.mean()),L=float(delta.min())),
        context_independent_identity_null_examples=fixed_examples,
        mathematical_claims=['If unique full pair changes after changing only A need, the other pair BC has no full plan at either end; both correct pairs contain A.',
            'Swapping the other two needs is an involution on every native partition and reverses the need-value-to-partner mapping while leaving sender PL observation unchanged.',
            'A context-independent effect of each packet on each fixed recipient gives zero D and nonpositive L.',
            'If one packet never raises any recipient proposal probability relative to the other, at least one compatibility-signed slope is nonpositive; L cannot be positive.',
            'Positive L is stronger than ordinary protocol dependence, but does not identify linguistic semantics or exclude all context-dependent perturbation mechanisms.'])

if __name__=='__main__':
    out=HERE/'static_identification_proof_001.json';a.require(not out.exists(),'Never overwrite certificate')
    result=run();out.write_bytes(a.json_bytes(result));print({k:result[k] for k in ('status','main_cells_per_policy_time','native_sham_cells_per_policy_time','four_policies_two_times')})

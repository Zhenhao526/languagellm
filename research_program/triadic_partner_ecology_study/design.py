"""Paired destination-stratified demand ecologies; no model initialization.

Every common destination triple has equal mass; resource triples are uniform
within that stratum and ecology. Raw world counts are not target weights.
"""
from fractions import Fraction
from itertools import product, permutations
from pathlib import Path
import hashlib
import json
import numpy as np
from research_program.triadic_message_study import runner as r

SEEDS=(49101,49102,49103,49104)
ECOLOGIES=('unique','multiple')
CONDITIONS=r.CONDITIONS
PARTITIONS=('train','heldout_layouts')
MONITOR_PER_STRATUM=32
MONITOR_SEED=61917001


def demand_layers():
    layers=[];excluded=[]
    for destinations in product(range(3),repeat=3):
        supports={e:[] for e in ECOLOGIES}
        for resources in product(range(4),repeat=3):
            needs=tuple(3*resources[a]+destinations[a] for a in range(3))
            k=len(r.base.env.compatible_pairs(needs))
            if k==1:supports['unique'].append(needs)
            elif k>=2:supports['multiple'].append(needs)
        row={'destinations':list(destinations),'supports':{e:[list(n) for n in v] for e,v in supports.items()}}
        (layers if all(supports.values()) else excluded).append(row)
    r.require(len(layers)==21 and len(excluded)==6,'Unexpected common destination domain')
    r.require([sum(len(x['supports'][e]) for x in layers) for e in ECOLOGIES]==[324,996],'Unexpected need supports')
    return layers,excluded


def exact_marginals(layers,ecology):
    margins=[[Fraction(0) for _ in range(12)] for _ in range(3)]
    for layer in layers:
        needs=layer['supports'][ecology];weight=Fraction(1,len(layers)*len(needs))
        for n in needs:
            for a in range(3):margins[a][n[a]]+=weight
    return [[str(x) for x in row] for row in margins]


def make_prepared():
    layers,excluded=demand_layers()
    r.require(exact_marginals(layers,'unique')==exact_marginals(layers,'multiple'),'Unbalanced personal needs')
    # Reused development layout split; all demands train, no unseen-demand arm.
    previous=r.base.make_prepared()['partitions']
    layouts={'train':previous['train']['layouts'],'heldout_layouts':previous['new_layouts']['layouts']}
    owners=[list(p) for p in permutations((1,2,3))]
    partitions={};monitor_draws={}
    for pi,part in enumerate(PARTITIONS):
        physical_count=len(layouts[part])*6
        for di in range(len(layers)):
            rng=np.random.default_rng(np.random.SeedSequence([MONITOR_SEED,pi,di]))
            # Unique physical arrangements guarantee unique state indices, even
            # if demand draws repeat. These draws are shared across ecologies.
            monitor_draws[(part,di)]=(rng.choice(physical_count,MONITOR_PER_STRATUM,replace=False),rng.random(MONITOR_PER_STRATUM))
    for ecology in ECOLOGIES:
        needs=[];strata=[]
        for layer in layers:
            start=len(needs);needs.extend(layer['supports'][ecology])
            strata.append({'destinations':layer['destinations'],'need_indices':list(range(start,len(needs)))})
        r.require(len(needs)==len({tuple(n) for n in needs}),'Duplicate demand tuple')
        partitions[ecology]={}
        for part in PARTITIONS:
            nphysical=len(layouts[part])*6;monitor=[]
            for di,layer in enumerate(strata):
                physical,u=monitor_draws[(part,di)]
                selected=np.array(layer['need_indices'])[np.floor(u*len(layer['need_indices'])).astype(int)]
                monitor.extend((selected*nphysical+physical).tolist())
            monitor=sorted(map(int,monitor))
            r.require(len(monitor)==len(set(monitor))==21*32,'Invalid monitor indices')
            partitions[ecology][part]=dict(ecology=ecology,partition=part,needs=needs,layouts=layouts[part],
                private_sites=owners,demand_strata=strata,world_count=len(needs)*nphysical,monitor_indices=monitor,
                population_weighting='equal destination strata, uniform conditional resources, layouts and owners',
                monitor_weighting='equal strata; each32 frozen sampled worlds equal weight')
    return dict(schema='destination_stratified_partner_ecologies_v1',seeds=list(SEEDS),ecologies=list(ECOLOGIES),
        conditions=list(CONDITIONS),partitions=partitions,common_destination_layers=layers,excluded_destination_layers=excluded,
        exact_personal_need_marginals={e:exact_marginals(layers,e) for e in ECOLOGIES},
        monitor_seed=MONITOR_SEED,monitor_per_stratum=MONITOR_PER_STRATUM,
        layout_split_scope='same18/6 development layout split as earlier; all demands train; not independent confirmation',
        runs=[dict(seed=seed,ecology=e,condition=c,directory=f'seed_{seed}_{e}_{c}') for seed in SEEDS for e in ECOLOGIES for c in CONDITIONS],
        updates_per_run=6000,batch_size=256,training_state_samples_total=32*6000*256,
        sampled_complete_message_trajectories_total=32*6000*256*2,
        full_natural_worlds_total=4*4*(324+996)*24*6,
        full_closed_worlds_total=4*2*(324+996)*24*6,
        checkpoints=list(r.CHECKPOINTS),automatic_followon_experiment=False)


def sample_indices(spec,uniforms):
    u=np.asarray(uniforms,dtype=float)
    r.require(u.ndim==2 and u.shape[1]==4 and np.isfinite(u).all() and ((u>=0)&(u<1)).all(),'Invalid world uniforms')
    strata=spec['demand_strata'];d=np.floor(u[:,0]*len(strata)).astype(np.int64)
    needs=np.empty(len(u),dtype=np.int64)
    for di,layer in enumerate(strata):
        rows=np.flatnonzero(d==di);indices=np.array(layer['need_indices'],dtype=np.int64)
        needs[rows]=indices[np.floor(u[rows,1]*len(indices)).astype(np.int64)]
    layout=np.floor(u[:,2]*len(spec['layouts'])).astype(np.int64)
    owner=np.floor(u[:,3]*len(spec['private_sites'])).astype(np.int64)
    return (needs*len(spec['layouts'])+layout)*len(spec['private_sites'])+owner


def evaluation_weights(spec,indices=None):
    if indices is not None:
        ids=np.asarray(indices,dtype=np.int64)
        r.require(np.array_equal(ids,np.array(spec['monitor_indices'])),'Only the frozen stratified monitor may be reweighted as monitor')
        return np.full(len(ids),1/len(ids),dtype=np.float64)
    nphysical=len(spec['layouts'])*len(spec['private_sites'])
    weights=np.empty(spec['world_count'],dtype=np.float64)
    for layer in spec['demand_strata']:
        mass=1/(len(spec['demand_strata'])*len(layer['need_indices'])*nphysical)
        for ni in layer['need_indices']:weights[ni*nphysical:(ni+1)*nphysical]=mass
    r.require(abs(weights.sum()-1)<1e-12,'Population weights do not sum to1')
    return weights


def pairing_fields(spec,indices):
    """Researcher-only fields; never append to a policy observation."""
    ids=np.asarray(indices);owners=len(spec['private_sites']);nl=len(spec['layouts'])
    ni=ids//(owners*nl);needs=np.array(spec['needs'],dtype=np.int16)[ni]
    return dict(destinations=needs%3,layout_indices=(ids//owners)%nl,owner_indices=ids%owners)

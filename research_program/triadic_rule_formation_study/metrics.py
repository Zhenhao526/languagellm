"""Fixed society-level trajectory estimand and descriptive message panels.

Pure NumPy arithmetic, no model, result-file loader, SciPy or learned decoder.
All message positions, actors, windows, backgrounds and adjacent times remain.
"""
from itertools import product
import numpy as np

SEEDS=tuple(range(60101,60117))
RULES=('strict','reciprocal')
UPDATES=(0,100,500,1500,3000,6000)
HORIZON=6000
STATE_ORDER='need-major, then layout, then owner'
TARGET='new_needs_and_layouts'
PRIMARY='mean_baseline_centered_native_Q_communication_DiD_time_AUC'
# NIST official df15/.975 table gives2.131. The additional digits were fixed
# before results by independent standard-library t15 CDF integration/bisection.
T15_975=2.1314495455597715
MEASURES={'native_Q':('native','Q'),'native_Q_excess':('native','Q_excess'),
          'common_reciprocal_Q':('common_reciprocal','Q'),
          'common_reciprocal_Q_excess':('common_reciprocal','Q_excess')}
SCALARS=('raw_AUC','centered_AUC','baseline_DiD','endpoint_DiD','endpoint_centered_DiD')


def require(ok,message):
    if not ok:raise ValueError(message)


def society_statistics(values):
    """Sixteen initialization blocks, not worlds or policies, define uncertainty."""
    values=np.asarray(values)
    require(values.shape==(16,) and values.dtype.kind in 'fiu' and np.isfinite(values).all(),
            'Exactly sixteen finite society values required')
    mean=float(np.mean(values));sd=float(np.std(values,ddof=1));se=sd/4;half=T15_975*se
    return dict(n=16,mean=mean,sample_sd=sd,standard_error=se,df=15,t_critical=T15_975,
        interval_level=.95,ci95_lower=mean-half,ci95_upper=mean+half,
        interval_method='Approximate two-sided Student-t interval across16 paired initializations;not a world-level interval or equivalence proof.')


def trajectory_contrasts(cells):
    """Four six-point Q or Q-excess curves in a single initialization block."""
    expected={(rule,live) for rule in RULES for live in (False,True)}
    require(set(cells)==expected,'Four rule/communication cells required')
    curves={key:np.asarray(value,dtype=np.float64) for key,value in cells.items()}
    require(all(a.shape==(6,) and np.isfinite(a).all() for a in curves.values()),'Six finite trajectory points required')
    strict=curves['strict',True]-curves['strict',False]
    reciprocal=curves['reciprocal',True]-curves['reciprocal',False]
    did=reciprocal-strict;centered=did-did[0];widths=np.diff(np.asarray(UPDATES,dtype=np.float64))
    def integral(curve):return float(np.sum(widths*(curve[:-1]+curve[1:])*.5)/HORIZON)
    return dict(communication_strict=strict.tolist(),communication_reciprocal=reciprocal.tolist(),
        DiD=did.tolist(),centered_DiD=centered.tolist(),baseline_DiD=float(did[0]),
        raw_AUC=integral(did),centered_AUC=integral(centered),
        endpoint_DiD=float(did[-1]),endpoint_centered_DiD=float(centered[-1]))


def primary(runs):
    """Unique native-Q primary, with all predefined paired auxiliaries retained."""
    require(isinstance(runs,list) and len(runs)==64,'Exactly64 complete training runs required')
    by={}
    for run in runs:
        seed=run['seed'];rule=run['rule'];live=run['live']
        require(type(seed) is int and seed in SEEDS and rule in RULES and type(live) is bool,'Unexpected society or condition')
        key=(seed,rule,live);require(key not in by,'Duplicate society condition')
        require([row['update'] for row in run['trajectory']]==list(UPDATES),'Six frozen checkpoints in order required')
        by[key]=run
    require(set(by)=={(seed,rule,live) for seed in SEEDS for rule in RULES for live in (False,True)},'Incomplete64-cell grid')
    rows=[]
    for seed in SEEDS:
        cells={};measures={}
        for rule,live in product(RULES,(False,True)):
            run=by[seed,rule,live];curves={}
            for name,(settlement,field) in MEASURES.items():
                values=[r['evaluation']['need_response'][settlement][field] for r in run['trajectory']]
                require(all(type(v) in (int,float) and np.isfinite(v) and (-1 if field=='Q_excess' else 0)<=v<=1 for v in values),
                        'Invalid stored Q/Q-excess trajectory')
                curves[name]=list(values)
            cells[f'{rule}_PL_{"live" if live else "silent"}']=dict(rule=rule,live=live,**curves)
        for name in MEASURES:
            values={(v['rule'],v['live']):v[name] for v in cells.values()}
            measures[name]=trajectory_contrasts(values)
        rows.append(dict(seed=seed,cells=cells,measures=measures))
    statistics=society_statistics([r['measures']['native_Q']['centered_AUC'] for r in rows])
    summaries={name:{field:society_statistics([r['measures'][name][field] for r in rows]) for field in SCALARS}
               for name in MEASURES}
    return dict(name=PRIMARY,partition=TARGET,independent_societies=16,training_runs=64,checkpoints=list(UPDATES),horizon_updates=HORIZON,
        primary_measure='native_Q',primary_statistic='centered_AUC',statistics=statistics,mean_centered_AUC=statistics['mean'],
        by_seed=rows,auxiliary_statistics=summaries,
        definition='Per seed:[Q_Rlive−Q_Rsilent]−[Q_Slive−Q_Ssilent],subtract its own step0 DiD,integrate by the trapezoid rule over update count and divide by6000;then mean16seeds.',
        unit='Probability difference averaged over training time;multiply by100 for percentage points.',
        scope='Native-rule Q primary only. Common-reciprocal scoring,Q-excess,uncentered AUC and endpoints are auxiliary. Q is a behavior response,not a language-formation score. No world-level p values,seed exclusion or automatic extension.')


def _messages(messages,spec):
    raw=np.asarray(messages);n=spec['world_count'];b=len(spec['layouts'])*len(spec['private_sites']);needs=len(spec['needs'])
    require(spec.get('state_order')==STATE_ORDER and type(n) is int and b>0 and needs>0 and n==b*needs,
            'Original complete need-major Cartesian specification required')
    require(raw.shape==(n,2,3,4) and raw.dtype.kind in 'iu' and np.all((raw>=0)&(raw<8)),
            'Full natural integer messages[N,2,3,4] with symbols0..7 required')
    codes=np.sum(raw.astype(np.int64,copy=False)*np.asarray([512,64,8,1]),axis=-1)
    return raw,codes,needs,b


def adjusted_rand_index(left,right):
    """Sparse pair-count ARI; no dense vocabulary-square contingency matrix."""
    left=np.asarray(left);right=np.asarray(right)
    require(left.ndim==1 and left.shape==right.shape and len(left)>0 and left.dtype.kind in 'iu' and right.dtype.kind in 'iu',
            'Equal nonempty integer cluster label vectors required')
    _,a,ca=np.unique(left,return_inverse=True,return_counts=True)
    _,b,cb=np.unique(right,return_inverse=True,return_counts=True)
    _,joint=np.unique(a.astype(np.int64)*len(cb)+b,return_counts=True)
    def pairs(counts):return int(np.sum(counts*(counts-1)//2,dtype=np.int64))
    total=len(left)*(len(left)-1)//2;aa=pairs(ca);bb=pairs(cb);ab=pairs(joint)
    denominator=total*(aa+bb)-2*aa*bb
    if denominator==0:return 1.0
    return float(2*(total*ab-aa*bb)/denominator)


def _entropy(codes):
    _,counts=np.unique(codes,return_counts=True);p=counts.astype(float)/len(codes)
    entropy=float(-np.sum(p*np.log2(p)))
    if entropy==0:entropy=0.0
    return dict(entropy_bits=entropy,effective_packet_count=float(2**entropy),observed_packet_count=len(counts),
                constant_code=len(counts)==1)


def message_snapshot(messages,spec):
    """Six separate actor/window panels, conditional on every public background."""
    _,codes,needs,b=_messages(messages,spec);panels=[];owners=len(spec['private_sites'])
    for actor,window in product(range(3),range(2)):
        values=codes[:,window,actor].reshape(needs,b);backgrounds=[]
        for background in range(b):
            backgrounds.append(dict(background_index=background,layout_index=background//owners,owner_index=background%owners,
                need_worlds=needs,**_entropy(values[:,background])))
        conditional=float(np.mean([bg['entropy_bits'] for bg in backgrounds]))
        panels.append(dict(actor=actor,window=window,conditional_entropy_bits=conditional,
            conditional_effective_packet_count=float(2**conditional),backgrounds=backgrounds,
            global_observed_packet_count=int(len(np.unique(values)))))
    return dict(schema='natural_message_snapshot_v1',worlds=spec['world_count'],need_worlds=needs,background_count=b,
        panel_order='actor,window',panels=panels,
        definition='Four-position base8 packet;H(M|public background)=equal-background mean Shannon entropy in bits;effective packets=2^H,not raw distinct count.',
        scope='All six actor/window panels reported separately. Full natural transcript in frozen state order is caller responsibility. Conditional entropy/effective code count do not identify meaning,compositionality or communication use.')


def message_transition(before,after,spec):
    """Only adjacent saved checkpoints; retain raw changes and both ARI scopes."""
    first,codes_a,needs,b=_messages(before,spec);last,codes_b,_,_=_messages(after,spec)
    panels=[];owners=len(spec['private_sites'])
    for actor,window in product(range(3),range(2)):
        a=codes_a[:,window,actor];z=codes_b[:,window,actor];abg=a.reshape(needs,b);zbg=z.reshape(needs,b)
        backgrounds=[]
        for background in range(b):
            left=abg[:,background];right=zbg[:,background]
            left_entropy=_entropy(left);right_entropy=_entropy(right)
            left_count=left_entropy['observed_packet_count'];right_count=right_entropy['observed_packet_count']
            backgrounds.append(dict(background_index=background,layout_index=background//owners,owner_index=background%owners,
                ARI=adjusted_rand_index(left,right),observed_packets_before=left_count,observed_packets_after=right_count,
                constant_before=left_count==1,constant_after=right_count==1,
                entropy_bits_before=left_entropy['entropy_bits'],entropy_bits_after=right_entropy['entropy_bits']))
        first_count=int(len(np.unique(a)));last_count=int(len(np.unique(z)))
        panels.append(dict(actor=actor,window=window,packet_equal_rate=float(np.mean(a==z)),
            token_hamming_fraction=float(np.mean(first[:,window,actor]!=last[:,window,actor])),
            global_ARI=adjusted_rand_index(a,z),mean_background_ARI=float(np.mean([bg['ARI'] for bg in backgrounds])),
            global_observed_packets_before=first_count,global_observed_packets_after=last_count,
            global_constant_before=first_count==1,global_constant_after=last_count==1,
            conditional_entropy_bits_before=float(np.mean([bg['entropy_bits_before'] for bg in backgrounds])),
            conditional_entropy_bits_after=float(np.mean([bg['entropy_bits_after'] for bg in backgrounds])),backgrounds=backgrounds))
    return dict(schema='adjacent_natural_message_transition_v1',worlds=spec['world_count'],need_worlds=needs,background_count=b,
        panel_order='actor,window',panels=panels,
        definition='Same-world original packets and raw token Hamming;ARI compares world partitions induced by packet equality,globally and within every public background.',
        boundary='Two constant partitions,even with different symbols,haveARI1;two singleton partitions also haveARI1. One constant and one nonconstant partition haveARI0. Display with snapshot entropy and packet counts.',
        scope='Only five adjacent checkpoint transitions;no initial-to-all matrix,no relabeling before raw Hamming,no actor/window composite score. Global ARI can reflect public-background grouping;neither ARI scope is semantic agreement.')

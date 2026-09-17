"""One fixed per-person scale at the visual interface; no model fitting.

``attach_scale`` preserves the existing parameters and their identities. Checkpoint
readers must use ``load_scaled_state`` before strict loading a scaled state dict.
``calibrate_pair`` reads only old-map/train-photo scenes and returns JSON metadata.
"""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'redesign_v0.13'))
import private_preparation as private
from camp import CampAgent, MAPS, remake_agents, scene_visual

CALIBRATION_CHUNK=512
RESET_PREFIXES=('memory.','slot_phi.')
ACTIVE_PREFIXES=('send_','receive_embedding.','actor.','receive_value.')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def _effective_scale(alpha,reference):
    value=float(alpha)
    if not np.isfinite(value) or value<=0:raise ValueError('visual scale must be finite and positive')
    result=reference.new_tensor(value)
    if not bool(torch.isfinite(result)) or not bool(result>0):raise ValueError('visual scale is not finite positive in model dtype')
    return result


class ScaledCampAgent(CampAgent):
    """Scale the completed observation once, before every legal consumer of h."""
    def observe(self,visual,delay=0,memory_mode='retain'):
        return super().observe(visual,delay,memory_mode)*self.visual_scale


def attach_scale(agent,alpha):
    """Promote this instance only; return it without allocating/reinitializing parameters.

    A persistent scalar buffer is the only new state entry. Existing parameters,
    storage, training flags, and external Torch/NumPy RNG states are unchanged.
    An already attached scale cannot be changed to another value.
    """
    if type(agent) not in (CampAgent,ScaledCampAgent):raise TypeError('expected an unwrapped CampAgent or ScaledCampAgent')
    value=_effective_scale(alpha,agent.memory.weight_ih)
    if isinstance(agent,ScaledCampAgent):
        if not torch.equal(agent.visual_scale,value):raise ValueError('a fixed visual scale cannot be changed')
        return agent
    if 'visual_scale' in agent.state_dict() or hasattr(agent,'visual_scale'):raise ValueError('visual_scale already exists')
    agent.__class__=ScaledCampAgent
    agent.register_buffer('visual_scale',value,persistent=True)
    return agent


def load_scaled_state(agent,state):
    """Register the stored fixed scale, then load the complete checkpoint strictly."""
    if 'visual_scale' not in state:raise KeyError('scaled checkpoint must include visual_scale')
    if state['visual_scale'].ndim!=0:raise ValueError('visual_scale must be scalar')
    attach_scale(agent,state['visual_scale'].detach().cpu().item())
    agent.load_state_dict(state,strict=True)
    return agent


def remake_scaled_agents(seed,prepared,states):
    """The standard fresh pair factory followed by scaled checkpoint loading."""
    agents=remake_agents(seed,prepared,7,2,'identity')
    if len(states)!=2:raise ValueError('expected two personal states')
    for agent,state in zip(agents,states):load_scaled_state(agent,state)
    return agents


def calibration_worlds(bank,partition):
    """Map-major order, then increasing food row, then increasing water row."""
    old=private.partition_maps(partition)['old']
    pools=[np.sort(np.asarray(bank.pools['train',kind],dtype=np.int64)) for kind in (0,1)]
    for kind,pool in enumerate(pools):
        assert len(pool)>0 and len(np.unique(pool))==len(pool)
        assert all(bank.entries[int(row)]['split']=='train' for row in pool)
        assert all(bank.entries[int(row)]['category']==('food','water')[kind] for row in pool)
    pairs=np.asarray(list(itertools.product(*pools)),dtype=np.int64)
    rows=np.asarray(list(itertools.product(old,range(len(pairs)))),dtype=np.int64)
    world=dict(world_id=np.arange(len(rows),dtype=np.int64),map_ids=rows[:,0],
        photo_pair_ids=rows[:,1],positions=MAPS[rows[:,0]].copy(),photo_ids=pairs[rows[:,1]].copy())
    assert len(rows)==18*len(pools[0])*len(pools[1])
    assert not np.isin(world['map_ids'],np.concatenate([private.partition_maps(partition)[g] for g in ('added','sealed')])).any()
    return world,pools


def _social_flags(agent):
    for key,param in agent.named_parameters():param.requires_grad_(key.startswith(ACTIVE_PREFIXES))


def _train_projected(agent,bank,pools):
    ids=np.unique(np.concatenate(pools));assert all(bank.entries[int(i)]['split']=='train' for i in ids)
    features=bank.features[torch.from_numpy(ids)]
    projected=agent.project(features)
    table=projected.new_zeros((len(bank.features),projected.shape[1]))
    table[torch.from_numpy(ids)]=projected
    return table


@torch.no_grad()
def calibrate_pair(seed,partition,prepared,bank,retained_states,reset_states,out,*,source_hashes=None):
    """Save fixed old/train support and raw h, compute the only permitted multiplier.

    CPU, float32 model arithmetic, 1 intra-op thread, no_grad, default train mode,
    explicit social-stage requires_grad flags, 512 scene chunks. The ratio is
    computed in NumPy float64 after concatenating all raw h in fixed world order.
    Calibration uses no reward, private head, message or test/sealed scene.
    """
    if torch.get_num_threads()!=1:raise ValueError('calibration requires torch.set_num_threads(1)')
    if bank.features.device.type!='cpu' or bank.features.dtype!=torch.float32:raise ValueError('CPU float32 features required')
    if len(retained_states)!=2 or len(reset_states)!=2:raise ValueError('expected two personal state dictionaries per interface')
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
    source_hashes={} if source_hashes is None else dict(source_hashes)
    for path,digest in source_hashes.items():assert sha(path)==digest,path
    world,pools=calibration_worlds(bank,partition)
    rows=[]
    # Model construction must not consume any subsequent training RNG stream.
    with torch.random.fork_rng(devices=[]):
        retained=remake_agents(seed,prepared,7,2,'identity')
        reset=remake_agents(seed,prepared,7,2,'identity')
        for who,(r,z,rs,zs) in enumerate(zip(retained,reset,retained_states,reset_states)):
            assert set(rs)==set(zs) and 'visual_scale' not in rs
            for key in rs:
                if not key.startswith(RESET_PREFIXES):assert torch.equal(rs[key],zs[key]),key
            r.load_state_dict(rs,strict=True);z.load_state_dict(zs,strict=True)
            for agent in (r,z):
                _social_flags(agent);agent.train()
                assert all(x.device.type=='cpu' and x.dtype==torch.float32 for x in agent.parameters())
            projected=_train_projected(r,bank,pools)
            for key,value in r.project.state_dict().items():assert torch.equal(value,z.project.state_dict()[key])
            all_h={}
            for name,agent in [('retained',r),('reset',z)]:
                pieces=[]
                for lo in range(0,len(world['world_id']),CALIBRATION_CHUNK):
                    sl=slice(lo,lo+CALIBRATION_CHUNK)
                    h=agent.observe(scene_visual(world['positions'][sl],world['photo_ids'][sl],projected))
                    pieces.append(h.numpy())
                all_h[name]=np.concatenate(pieces)
                assert all_h[name].shape==(len(world['world_id']),96) and np.isfinite(all_h[name]).all()
            norms={name:float(np.linalg.norm(h.astype(np.float64),axis=1).mean()) for name,h in all_h.items()}
            if norms['reset']<=0:raise ValueError('zero reset mean norm; multiplier undefined')
            ratio=norms['retained']/norms['reset'];effective=float(_effective_scale(ratio,z.memory.weight_ih).item())
            raw_path=out/f'person{who}.npz';np.savez_compressed(raw_path,**world,retained_h=all_h['retained'],reset_h=all_h['reset'])
            scaled=all_h['reset']*np.float32(effective)
            rows.append(dict(person=who,world_count=len(world['world_id']),retained_mean_l2=norms['retained'],reset_mean_l2=norms['reset'],
                alpha_float64=ratio,alpha_float32=effective,scaled_mean_l2=float(np.linalg.norm(scaled.astype(np.float64),axis=1).mean()),
                raw_path=str(raw_path),raw_sha256=sha(raw_path),retained_state_sha256=private.module_sha(r),reset_state_sha256=private.module_sha(z),
                requires_grad={key:param.requires_grad for key,param in z.named_parameters()}))
    result=dict(status='complete',seed=seed,partition=partition,persons=rows,training_updates=0,
        old_map_ids=private.partition_maps(partition)['old'].tolist(),train_photo_ids=[p.tolist() for p in pools],
        photo_manifest=[[dict(feature_row=int(i),image_id=bank.entries[int(i)]['id'],sha256=bank.entries[int(i)]['sha256']) for i in pool] for pool in pools],
        world_order='increasing old map ID; increasing food feature row; increasing water feature row',chunk_size=CALIBRATION_CHUNK,
        calibration_dtype='float32 forward; float64 concatenated-world L2 and mean ratio; float32 persistent scale buffer',
        device='cpu',torch_threads=1,torch_interop_threads=torch.get_num_interop_threads(),torch_version=str(torch.__version__),
        mkldnn_enabled=torch.backends.mkldnn.enabled,grad_enabled=False,model_training=True,
        no_test_or_held_maps=True,source_hashes=source_hashes,calibration_source_sha256=sha(__file__),created_utc=datetime.now(timezone.utc).isoformat(),
        estimand='One old/train-distribution mean-L2 match; not per-world, coordinatewise, covariance, direction or information matching.')
    private.write_json(out/'calibration.json',result)
    return result


def self_test():
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(99515)
        a=CampAgent(nn.Sequential(nn.Linear(8,64)),7,2);_social_flags(a)
        before=copy.deepcopy(a.state_dict());b=copy.deepcopy(a);pointers=[p.data_ptr() for p in b.parameters()]
        rng=torch.random.get_rng_state().clone();attach_scale(b,1.)
        assert torch.equal(rng,torch.random.get_rng_state()) and pointers==[p.data_ptr() for p in b.parameters()]
        assert set(b.state_dict())==set(before)|{'visual_scale'} and 'visual_scale' not in dict(b.named_parameters())
        assert all(torch.equal(x,b.state_dict()[k]) for k,x in before.items())
        visual=torch.randn(17,390)
        with torch.no_grad():
            for mode in ('retain','reset','replay'):
                for delay in (0,2):assert torch.equal(a.observe(visual,delay,mode),b.observe(visual,delay,mode))
            h=a.observe(visual);lhs=a.send(h,torch.zeros(17,2),torch.zeros(17,2),np.random.default_rng(15),False)
            rhs=b.send(b.observe(visual),torch.zeros(17,2),torch.zeros(17,2),np.random.default_rng(15),False)
            assert all(torch.equal(x,y) for x,y in zip(lhs,rhs))
        c=copy.deepcopy(a);attach_scale(c,3.)
        with torch.no_grad():
            assert torch.equal(c.observe(visual),a.observe(visual)*3.)
            assert not c.observe(visual,memory_mode='reset').any()
        reload=copy.deepcopy(a);load_scaled_state(reload,c.state_dict())
        assert all(torch.equal(value,reload.state_dict()[key]) for key,value in c.state_dict().items())
        assert torch.equal(reload.observe(visual),c.observe(visual))
        for bad in (0.,-1.,float('inf'),float('nan'),1e-100,1e100):
            try:attach_scale(copy.deepcopy(a),bad)
            except ValueError:pass
            else:raise AssertionError('invalid multiplier accepted')
    return dict(passed=True,checks=['No parameter or RNG change on attach','Only persistent scalar buffer added','Scale one exactly preserves observation and sampled send including value',
        'All memory modes/delays scaled once; erase stays zero','Full checkpoint reload preserves scalar and outputs','Invalid/underflow/overflow scales rejected'])


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--self-test',action='store_true',required=True);ap.parse_args()
    print(json.dumps(self_test()))

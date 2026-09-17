"""Read-only fixed private-head compatibility and latent-scale check; no fitting."""
from pathlib import Path
import argparse,json,sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'redesign_v0.13'))
import run_spatial as v13
pvt=v13.private

@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,default=ROOT.parent/'redesign_v0.13/results/spatial_001')
    ap.add_argument('--out',type=Path,default=ROOT/'manipulation_001');args=ap.parse_args()
    torch.set_num_threads(1);args.source=args.source.resolve();args.out=args.out.resolve();args.out.mkdir(exist_ok=False)
    inv=json.loads((args.source/'invocation.json').read_text());bank=v13.ImageBank();world=pvt.evaluation_worlds(bank)
    rows=[];hashes={str(Path(__file__).resolve()):pvt.sha(__file__)}
    for seed in inv['args']['seeds']:
        prep_path=args.source/f'prepared_{seed}.pt';prepared=torch.load(prep_path,weights_only=True);hashes[str(prep_path)]=pvt.sha(prep_path)
        for part in inv['args']['partitions']:
            folder=args.source/f'private_s{seed}_p{part}_control';cfg=json.loads((folder/'config.json').read_text());step=cfg['updates']
            for name in ('initial.pt','final.pt',f'evaluation_{step:04d}.npz'):
                hashes[str(folder/name)]=pvt.sha(folder/name)
            initial=torch.load(folder/'initial.pt',weights_only=True);final=torch.load(folder/'final.pt',weights_only=True)
            heads=pvt.make_heads(seed);agents=v13.remake_agents(seed,prepared,7,2,'identity');banks=v13.projected_banks(agents,bank)
            saved=np.load(folder/f'evaluation_{step:04d}.npz')
            for key,value in world.items():assert np.array_equal(value,saved[key]),key
            for who,(a,head,b) in enumerate(zip(agents,heads,banks)):
                head.load_state_dict(final['heads'][who]);# Keep original parameter flags to reproduce original inference arithmetic.
                values={};out=dict(world);out={k:v for k,v in world.items()}
                for arm,state in [('retained',final['agents'][who]),('reset',initial['agents'][who])]:
                    a.load_state_dict(state)
                    h=a.observe(pvt.scene_visual(world['positions'],world['photo_ids'],b));logits=head(h).reshape(480,2,6).numpy()
                    if arm=='retained':assert np.array_equal(logits,saved['logits'][who]),(seed,part,who)
                    out[arm+'_h']=h.numpy();out[arm+'_logits']=logits
                    values[arm]=pvt.capability_statistics(logits,world,part)
                support=dict(pvt.partition_maps(part),common30=np.arange(30));geometry={}
                for group,pool in support.items():
                    mask=np.isin(world['map_ids'],pool);r=out['retained_h'][mask].astype(float);z=out['reset_h'][mask].astype(float);delta=r-z
                    geometry[group]=dict(retained_mean_l2=float(np.linalg.norm(r,axis=1).mean()),reset_mean_l2=float(np.linalg.norm(z,axis=1).mean()),paired_mean_l2=float(np.linalg.norm(delta,axis=1).mean()),paired_rms=float(np.sqrt(np.mean(delta**2))))
                np.savez_compressed(args.out/f's{seed}_p{part}_person{who}.npz',**out)
                rows.append(dict(seed=seed,partition=part,person=who,private_head_readout=values,geometry=geometry))
    pvt.write_json(args.out/'result.json',dict(status='complete',persons=len(rows),source_hashes=hashes,rows=rows,retained_logits_exact_replay=True,training_updates=0,
        interpretation='Fixed trained private head compatibility only; reset failure is not evidence of absent information or its separately relearned capacity. Latent norm/RMS are scale checks, not information or mediation.'))
    print(json.dumps(dict(status='complete',persons=len(rows),retained_logits_exact_replay=True)))

if __name__=='__main__':main()

"""Frozen private-agent visual material probe; no training or network calls."""
import argparse, hashlib, itertools, json, platform, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent; PROJECT=ROOT.parent
sys.path.insert(0,str(PROJECT/'redesign_v0.23'))
import run_experience as old
from camp import ImageBank,remake_agents
import temporal_model as temporal
import world,metrics
sys.path.insert(0,str(PROJECT/'redesign_v0.4'))
import encode_images
BASE=PROJECT/'paper_program/visual_confirmation_v1'
PRIOR=PROJECT/'redesign_v0.23/results/experience_001'
SEEDS=[33101,33102,33103,33104]
CELLS=('old_old','new_old','old_new','new_new')

def read(p):return json.loads(Path(p).read_text())
def sha(p):return encode_images.sha(p)
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def now():return datetime.now(timezone.utc).isoformat()

def selected():
    status=read(BASE/'pixel_workflow_003/curation_status_20260916.json')
    ids=status['cumulative']['usable_ids']; assert len(ids)==len(set(ids))==11
    meta={r['id']:r for r in read(BASE/'manifest.json')['images']}
    downloads={r['blind_id']:r for b in (1,2,3) for r in read(BASE/f'pixel_workflow_{b:03d}/download_manifest.json')['records']}
    rows=[]
    for ident in sorted(ids):
        r=downloads[ident];m=meta[r['candidate_id']]
        assert r['status']=='downloaded' and sha(r['processed_path'])==r['processed_sha256']
        rows.append(dict(r,stratum=m['stratum'],cluster_id=m['cluster_id']))
    assert {k:sum(r['stratum']==k for r in rows) for k in ('apple','banana','orange','water')}==dict(apple=3,banana=3,orange=3,water=2)
    return rows

def freeze(out):
    assert not out.exists(), 'Use a new output directory; never overwrite a completed or failed probe'
    rows=selected(); paths=[Path(__file__), ROOT/'固定执行方案.md',BASE/'manifest.json',BASE/'candidate_review_order.json', BASE/'pixel_workflow_003/curation_status_20260916.json']
    paths += [BASE/f'pixel_workflow_{b:03d}'/n for b in (1,2,3) for n in ('download_manifest.json','curation_status_20260916.json')]
    paths += [Path(r['processed_path']) for r in rows]
    paths += [PROJECT/'redesign_v0.4/data'/n for n in ('manifest.json','features.npz','encoder_report.json')]
    paths += [PROJECT/n for n in ('redesign_v0.4/encode_images.py','redesign_v0.4/agents.py','redesign_v0.4/run_pilot.py','redesign_v0.4/resource_env.py','redesign_v0.8/camp.py','redesign_v0.20/temporal_model.py','redesign_v0.20/temporal_world.py','redesign_v0.21/social_model.py','redesign_v0.23/world.py','redesign_v0.23/metrics.py','redesign_v0.23/run_experience.py')]
    legacy_entries=read(PROJECT/'redesign_v0.4/data/manifest.json')['images'];assert len(legacy_entries)==60
    for entry in legacy_entries[:8]:
        path=PROJECT/'redesign_v0.4'/entry['path'];assert sha(path)==entry['sha256'];paths.append(path)
    paths += [PRIOR/f'prepared_{s}.pt' for s in SEEDS]
    paths += [PRIOR/'private'/f's{s}_p{p}_d{d}_{scope}'/n for s,p,d,scope in itertools.product(SEEDS,(1,2,3),(0,1),('old','all')) for n in ('final.pt','evaluation_2400.npz')]
    weight=PROJECT/'redesign_v0.4/weights/dinov2_vitl14_pretrain.pth';paths.append(weight)
    assert sha(weight)==read(PROJECT/'redesign_v0.4/data/encoder_report.json')['weight_sha256']
    paths+=list((PROJECT/'redesign_v0.4/vendor/dinov2/dinov2').rglob('*.py'))
    hashes={str(p):sha(p) for p in sorted(set(paths))}
    out.mkdir(parents=True); (out/'tables').mkdir()
    write(out/'selection.json',dict(rows=rows,scope='11 previously accepted limited workflow images; all included'))
    write(out/'freeze.json',dict(created_utc=now(),before_new_model_inference=True,input_hashes=hashes,selection_sha256=sha(out/'selection.json'),training_updates=0,confirmation_images=0))
    return rows,hashes

def build_table(food,water):
    pairs=np.asarray(list(itertools.product(food,water)),np.int64)
    mids=np.repeat(np.arange(30),len(pairs));ids=np.tile(pairs,(30,1));n=len(mids)
    return dict(map_id=np.tile(mids,2),photo_ids=np.tile(ids,(2,1)),positions=np.tile(world.MAPS[mids],(2,1)),shown=np.repeat(np.arange(2),n))

def dino(rows,out,bank):
    sys.path.insert(0,str(PROJECT/'redesign_v0.4/vendor/dinov2'))
    from dinov2.hub.backbones import dinov2_vitl14
    assert torch.backends.mps.is_available(), 'Fixed MPS encoder device unavailable'
    model=dinov2_vitl14(pretrained=False)
    model.load_state_dict(torch.load(PROJECT/'redesign_v0.4/weights/dinov2_vitl14_pretrain.pth',map_location='cpu',weights_only=True),strict=True)
    model.requires_grad_(False).eval().to('mps')
    t=time.monotonic()
    with torch.inference_mode():
        anchor=model(torch.stack([encode_images.preprocess(PROJECT/'redesign_v0.4'/e['path']) for e in bank.entries[:8]]).to('mps')).float().cpu().numpy()
        original=np.load(PROJECT/'redesign_v0.4/data/features.npz')['features'][:8]
        passed=bool(np.allclose(anchor,original,atol=1e-4,rtol=1e-4))
        np.savez_compressed(out/'encoder_anchor.npz',reencoded=anchor,cached=original)
        write(out/'encoder_anchor_qa.json',dict(passed=passed,n=8,atol=1e-4,rtol=1e-4,max_abs_error=float(np.max(np.abs(anchor-original)))))
        assert passed, 'Encoder anchor failed; preserve result and stop without changing tolerance'
        outputs=[]
        write(out/'model_exposure_started.json',dict(utc=now(),blind_ids=[r['blind_id'] for r in rows],note='All11 workflow images enter inference; not eligible for future unexposed confirmation'))
        for lo in range(0,len(rows),8):
            x=torch.stack([encode_images.preprocess(r['processed_path']) for r in rows[lo:lo+8]])
            outputs.append(model(x.to('mps')).float().cpu().numpy())
    raw=np.concatenate(outputs);assert raw.shape==(11,1024) and np.isfinite(raw).all()
    np.savez_compressed(out/'new_features.npz',features=raw)
    np.savez_compressed(out/'normalization.npz',center=bank.center,scale=np.asarray(bank.scale))
    write(out/'encoder_receipt.json',dict(created_utc=now(),new_images=11,anchor_images=8,forward_batches=3,training_updates=0,device='mps',torch=torch.__version__,python=platform.python_version(),parameters=sum(p.numel() for p in model.parameters()),seconds=time.monotonic()-t,features_sha256=sha(out/'new_features.npz'),input='pixels only',normalization='old training-only center and scale; unchanged'))
    del model;torch.mps.empty_cache()
    return torch.cat((bank.features,torch.from_numpy((raw-bank.center)/max(bank.scale,1e-6))))

def summarize(raw,p):
    result=metrics.private(raw,p)
    correct=raw['logits'].argmax(-1)==raw['positions'];ix=np.arange(len(correct));shown=raw['shown']
    for group,selected_mask in metrics.masks(raw,p).items():
        for label,mask in [('pooled',selected_mask),('food_only',selected_mask&(shown==0)),('water_only',selected_mask&(shown==1))]:
            result[group][label].update(visible=float(correct[ix,shown][mask].mean()),historical=float(correct[ix,1-shown][mask].mean()))
    return result

def aggregate(records):
    sources=[]
    for s,scope,cell in itertools.product(SEEDS,('old','all'),CELLS):
        rr=[r for r in records if (r['seed'],r['scope'],r['cell'])==(s,scope,cell)]; assert len(rr)==6
        scores={g:{m:{key:float(np.mean([r['scores'][g][m][key] for r in rr])) for key in ('J','Q','food','water','visible','historical')} for m in ('pooled','food_only','water_only')} for g in ('old','new12','common30')}
        sources.append(dict(seed=s,scope=scope,cell=cell,scores=scores))
    overall=[]
    for scope,cell in itertools.product(('old','all'),CELLS):
        rr=[r for r in sources if (r['scope'],r['cell'])==(scope,cell)]
        scores={g:{m:{key:float(np.mean([r['scores'][g][m][key] for r in rr])) for key in ('J','Q','food','water','visible','historical')} for m in ('pooled','food_only','water_only')} for g in ('old','new12','common30')}
        overall.append(dict(scope=scope,cell=cell,scores=scores))
    gate=[dict(seed=r['seed'],mask=m,J=r['scores']['new12'][m]['J'],passed=r['scores']['new12'][m]['J']>=.8) for r in sources if r['scope']=='all' and r['cell']=='new_new' for m in ('food_only','water_only')]
    return dict(sources=sources,overall=overall,primary='private all/new_new/new12 J; source mean',gate=gate,gate_all_passed=all(r['passed'] for r in gate))

@torch.no_grad()
def run(out):
    torch.set_num_threads(1);rows,hashes=freeze(out);started=time.monotonic();bank=ImageBank()
    features=dino(rows,out,bank)
    pools={'old':[np.sort(bank.pools['test',k])[:4] for k in (0,1)],'new':[np.asarray([60+i for i,r in enumerate(rows) if (r['stratum']=='water')==bool(k)]) for k in (0,1)]}
    tables={cell:build_table(pools[cell.split('_')[0]][0],pools[cell.split('_')[1]][1]) for cell in CELLS}
    assert {k:len(v['map_id']) for k,v in tables.items()}==dict(old_old=960,new_old=2160,old_new=480,new_new=1080)
    np.savez_compressed(out/'combined_normalized_features.npz',features=features.numpy())
    write(out/'world_manifest.json',dict(pools={key:[x.tolist() for x in value] for key,value in pools.items()},counts={k:len(v['map_id']) for k,v in tables.items()},old_food=[bank.entries[i] for i in pools['old'][0]],model_inputs='world.frames places image vectors; metadata IDs/labels do not reach temporal model or head'))
    records=[];comparisons=[]
    for s,p,d,scope in itertools.product(SEEDS,(1,2,3),(0,1),('old','all')):
        prepared=torch.load(PRIOR/f'prepared_{s}.pt',weights_only=True)
        a=remake_agents(s,prepared,7,2,'identity')[d];head=temporal.make_head(world.seed_value(s,p,d,0))
        folder=PRIOR/'private'/f's{s}_p{p}_d{d}_{scope}';blob=torch.load(folder/'final.pt',weights_only=True)
        a.load_state_dict(blob['agent'],strict=True);head.load_state_dict(blob['head'],strict=True)
        for name,value in a.named_parameters():value.requires_grad_(name.startswith(('memory.','slot_phi.')))
        a.eval();head.eval();before=old.state_sha(a.state_dict());head_before=old.state_sha(head.state_dict())
        projected=a.project(features)
        for cell,w in tables.items():
            h=old.encode(a,projected,w)
            logits=np.concatenate([head(torch.from_numpy(h[lo:lo+256])).reshape(-1,2,6).numpy() for lo in range(0,len(h),256)])
            raw=dict(w,h=h,logits=logits);file=out/'tables'/f's{s}_p{p}_d{d}_{scope}_{cell}.npz';np.savez_compressed(file,**raw)
            if cell=='old_old':
                ref=np.load(folder/'evaluation_2400.npz')
                assert all(np.array_equal(w[k],ref[k]) for k in w)
                ok=bool(np.allclose(logits,ref['logits'],atol=1e-6,rtol=1e-5));greedy=bool(np.array_equal(logits.argmax(-1),ref['logits'].argmax(-1)))
                comparisons.append(dict(seed=s,partition=p,direction=d,scope=scope,passed=ok and greedy,max_abs_logits_error=float(np.max(np.abs(logits-ref['logits']))),same_actions=greedy))
                assert ok and greedy, 'Legacy positive control failed'
            records.append(dict(seed=s,partition=p,direction=d,scope=scope,cell=cell,file=str(file),sha256=sha(file),scores=summarize(raw,p)))
        assert before==old.state_sha(a.state_dict()) and head_before==old.state_sha(head.state_dict())
        assert all(v.grad is None for v in list(a.parameters())+list(head.parameters()))
        print(json.dumps(dict(completed_private_endpoints=len(comparisons),total=48)),flush=True)
    assert len(records)==192
    write(out/'raw_summaries.json',records);write(out/'summary.json',aggregate(records));write(out/'legacy_control_qa.json',dict(passed=all(r['passed'] for r in comparisons),rows=comparisons))
    assert all(sha(p)==h for p,h in hashes.items())
    write(out/'completion.json',dict(status='complete',created_utc=now(),seconds=time.monotonic()-started,private_endpoints=48,tables=192,private_world_forwards=224640,new_images=11,training_updates=0,social_runs=0,confirmation_images=0,network_requests=0,frozen_inputs_unchanged=True))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True,type=Path);args=parser.parse_args()
    run(args.out.resolve())

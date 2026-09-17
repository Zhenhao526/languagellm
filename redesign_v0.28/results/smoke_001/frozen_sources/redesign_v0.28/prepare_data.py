"""Freeze all twelve accepted images and encode once using official DINOv2-L."""
import hashlib,json,platform,sys,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent;OUT=ROOT/'data'
sys.path.insert(0,str(PROJECT/'redesign_v0.4'))
from encode_images import preprocess,sha

def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def now():return datetime.now(timezone.utc).isoformat()

def main():
    assert not OUT.exists(),'Do not overwrite frozen data or restart a completed encoder'
    fruit_path=PROJECT/'redesign_v0.26/results/material_001/selection.json'
    water_path=PROJECT/'paper_program/water_source_followup_001/curation_result.json'
    fruit=read(fruit_path)['rows'];water=read(water_path)['rows'];rows=[]
    for r in fruit:
        if r['stratum']=='water':continue
        rows.append(dict(id=r['blind_id'],stratum=r['stratum'],category='food',source_cluster=r['cluster_id'],
            processed_path=r['processed_path'],sha256=r['processed_sha256'],candidate_id=r['candidate_id'],
            prior_model_exposure='v26/v27',source_page=r['source_page'],license=r['license'],license_url=r['license_url']))
    for r in water:
        if not r['limited_workflow_usable']:continue
        f=r['files'];at=r['attribution']
        rows.append(dict(id=r['id'],stratum='water',category='water',source_cluster=r['source_cluster'],
            processed_path=f['processed_path'],sha256=f['processed_sha256'],candidate_id=r['candidate_id'],
            prior_model_exposure='none_before_this_batch',source_page=at['source_page'],license=at['license_name'],license_url=at['license_url']))
    assert len(rows)==len({r['id'] for r in rows})==len({r['source_cluster'] for r in rows})==12
    for kind in ('apple','banana','orange','water'):
        block=[r for r in rows if r['stratum']==kind];assert len(block)==3
        for r in block:r['split_rank_hash']=hashlib.sha256(('v28|20260916|split-v1|'+r['source_cluster']+'|'+r['id']).encode()).hexdigest()
        for i,r in enumerate(sorted(block,key=lambda x:x['split_rank_hash'])):r['split']='train' if i<2 else 'test'
    rows.sort(key=lambda r:r['id'])
    assert not ({r['sha256'] for r in rows if r['split']=='train'}&{r['sha256'] for r in rows if r['split']=='test'})
    for r in rows:assert sha(r['processed_path'])==r['sha256']
    weight=PROJECT/'redesign_v0.4/weights/dinov2_vitl14_pretrain.pth'
    assert sha(weight)==read(PROJECT/'redesign_v0.4/data/encoder_report.json')['weight_sha256']
    legacy=read(PROJECT/'redesign_v0.4/data/manifest.json')['images'][:8]
    paths=[Path(__file__),ROOT/'固定执行方案.md',fruit_path,water_path,
        PROJECT/'paper_program/water_source_followup_001/artifact_manifest.json',
        PROJECT/'redesign_v0.4/encode_images.py',weight]
    paths += [PROJECT/'redesign_v0.4/data'/n for n in ('manifest.json','features.npz','encoder_report.json')]
    paths += [Path(r['processed_path']) for r in rows]
    paths += [PROJECT/'redesign_v0.4'/r['path'] for r in legacy]
    paths += list((PROJECT/'redesign_v0.4/vendor/dinov2/dinov2').rglob('*.py'))
    hashes={str(p):sha(p) for p in sorted(set(paths))}
    OUT.mkdir()
    write(OUT/'selection.json',dict(images=rows,training_images=8,test_images=4,scope='development_formation_replication',salt='v28|20260916|split-v1|'))
    write(OUT/'freeze.json',dict(created_utc=now(),before_inference=True,input_hashes=hashes,selection_sha256=sha(OUT/'selection.json')))
    torch.set_num_threads(1);assert torch.backends.mps.is_available()
    sys.path.insert(0,str(PROJECT/'redesign_v0.4/vendor/dinov2'))
    from dinov2.hub.backbones import dinov2_vitl14
    model=dinov2_vitl14(pretrained=False)
    model.load_state_dict(torch.load(weight,map_location='cpu',weights_only=True),strict=True)
    model.requires_grad_(False).eval().to('mps');start=time.monotonic()
    with torch.inference_mode():
        anchor=model(torch.stack([preprocess(PROJECT/'redesign_v0.4'/r['path']) for r in legacy]).to('mps')).float().cpu().numpy()
        cached=np.load(PROJECT/'redesign_v0.4/data/features.npz')['features'][:8]
        passed=bool(np.allclose(anchor,cached,atol=1e-4,rtol=1e-4))
        np.savez_compressed(OUT/'anchor.npz',reencoded=anchor,cached=cached)
        write(OUT/'anchor_qa.json',dict(passed=passed,max_abs_error=float(np.abs(anchor-cached).max()),atol=1e-4,rtol=1e-4))
        assert passed,'Keep failed anchor; do not relax tolerance'
        write(OUT/'model_exposure_started.json',dict(utc=now(),image_ids=[r['id'] for r in rows],new_to_model=[r['id'] for r in rows if r['category']=='water']))
        outputs=[]
        for lo in range(0,len(rows),8):
            outputs.append(model(torch.stack([preprocess(r['processed_path']) for r in rows[lo:lo+8]]).to('mps')).float().cpu())
            print(json.dumps(dict(encoded=min(lo+8,len(rows)),total=len(rows))),flush=True)
    z=torch.cat(outputs);assert z.shape==(12,1024) and torch.isfinite(z).all()
    torch.save(dict(features=z,image_ids=[r['id'] for r in rows]),OUT/'feature_cache.pt')
    assert all(sha(p)==h for p,h in hashes.items())
    write(OUT/'encoder_receipt.json',dict(status='complete',ended_utc=now(),images=12,first_model_exposure_images=3,anchor_images=8,
        parameters=sum(p.numel() for p in model.parameters()),model='official DINOv2 ViT-L/14',device='mps',training_updates=0,
        network_requests=0,confirmation_images=0,raw_unnormalized=True,feature_sha256=sha(OUT/'feature_cache.pt'),selection_sha256=sha(OUT/'selection.json'),
        python=platform.python_version(),torch=str(torch.__version__),numpy=np.__version__,seconds=time.monotonic()-start))
    del model;torch.mps.empty_cache()
    print(json.dumps(dict(status='complete',images=12,first_exposures=3)),flush=True)

if __name__=='__main__':main()

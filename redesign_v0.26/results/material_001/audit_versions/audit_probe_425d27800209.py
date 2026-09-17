"""Independent finite-material QA. No DINO inference, network, or optimization.

Recomputes all raw metrics and tables; replays only the two prespecified private
endpoints. Uses the frozen Camp/temporal modules, but no v26 runner, v23 world,
or production metric/aggregation function.
"""
import argparse, hashlib, io, itertools, json, sys, time, traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch

ROOT=Path(__file__).resolve().parent; PROJECT=ROOT.parent
sys.path.insert(0,str(PROJECT/'redesign_v0.8'))
from camp import remake_agents
sys.path.insert(0,str(PROJECT/'redesign_v0.20'))
import temporal_model as temporal
sys.path.insert(0,str(PROJECT/'redesign_v0.4'))
import encode_images

BASE=PROJECT/'paper_program/visual_confirmation_v1'
PRIOR=PROJECT/'redesign_v0.23/results/experience_001'
SEEDS=(33101,33102,33103,33104)
CELLS=('old_old','new_old','old_new','new_new')
MAPS=np.asarray(list(itertools.permutations(range(6),2)),np.int64)
MATCHINGS=(((0,1),(2,3),(4,5)),((0,2),(1,4),(3,5)),((0,3),(1,5),(2,4)))
EXPECTED_COUNTS=dict(old_old=960,new_old=2160,old_new=480,new_new=1080)
WORLD_KEYS=('map_id','positions','photo_ids','shown')
METRICS=('J','Q','food','water','visible','historical')

def read(path):return json.loads(Path(path).read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def load(path):
    with np.load(path,allow_pickle=False) as data:return {k:data[k] for k in data.files}
def now():return datetime.now(timezone.utc).isoformat()

class QA:
    def __init__(self):self.checks=0;self.comparisons=0;self.max_abs_error=0.;self.failures=[]
    def check(self,value,label):
        self.checks+=1
        if not bool(value):self.failures.append(label);raise AssertionError(label)
    def close(self,a,b,label,atol=1e-12,rtol=0):
        a=np.asarray(a);b=np.asarray(b);self.check(a.shape==b.shape,label+' shape')
        err=float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64)))) if a.size else 0.
        self.comparisons+=1;self.max_abs_error=max(self.max_abs_error,err)
        self.check(np.allclose(a,b,atol=atol,rtol=rtol),label+f' max_error={err}')
    def tree(self,a,b,label):
        if isinstance(a,dict):
            self.check(set(a)==set(b),label+' keys')
            for k in a:self.tree(a[k],b[k],label+'/'+k)
        elif isinstance(a,list):
            self.check(len(a)==len(b),label+' length')
            for i,(x,y) in enumerate(zip(a,b)):self.tree(x,y,label+f'/{i}')
        elif isinstance(a,(float,int)) and not isinstance(a,bool):self.close(a,b,label)
        else:self.check(a==b,label)

def groups(p):
    def ids(matching):
        pairs={pair for a,b in matching for pair in ((a,b),(b,a))}
        return np.asarray([i for i,x in enumerate(MAPS) if tuple(x) in pairs],np.int64)
    added=ids(MATCHINGS[p-1]);sealed=ids(MATCHINGS[p%3]);new=np.r_[added,sealed]
    return dict(old=np.setdiff1d(np.arange(30),new),added=added,sealed=sealed,common30=np.arange(30),new12=new)

def table(food,water):
    # Explicit enumeration works for rectangular pools and preserves fixed order.
    rows=[(m,f,w,k) for k in (0,1) for m in range(30) for f in food for w in water]
    z=np.asarray(rows,np.int64)
    return dict(map_id=z[:,0],positions=MAPS[z[:,0]],photo_ids=z[:,1:3],shown=z[:,3])

def scores(raw,p):
    z=np.asarray(raw['logits'],np.float64);probs=np.exp(z-z.max(-1,keepdims=True));probs/=probs.sum(-1,keepdims=True)
    correct=raw['logits'].argmax(-1)==raw['positions']
    target=np.take_along_axis(probs,raw['positions'][...,None],axis=-1)[...,0]
    q=target[:,0]*target[:,1];shown=raw['shown'];r=np.arange(len(shown));out={}
    for g,ids in groups(p).items():
        selected=np.isin(raw['map_id'],ids);out[g]={}
        for label,mask in [('pooled',selected),('food_only',selected&(shown==0)),('water_only',selected&(shown==1))]:
            c=correct[mask]
            value=dict(n=int(mask.sum()),J=float(np.logical_and(c[:,0],c[:,1]).mean()),food=float(c[:,0].mean()),water=float(c[:,1].mean()),Q=float(q[mask].mean()))
            value.update(visible=float(correct[r,shown][mask].mean()),historical=float(correct[r,1-shown][mask].mean()))
            out[g][label]=value
    return out

def aggregate(rows):
    def mean_scores(rr):
        return {g:{m:{k:float(np.mean([r['scores'][g][m][k] for r in rr])) for k in METRICS} for m in ('pooled','food_only','water_only')} for g in ('old','new12','common30')}
    sources=[];overall=[]
    for s,scope,cell in itertools.product(SEEDS,('old','all'),CELLS):
        rr=[r for r in rows if (r['seed'],r['scope'],r['cell'])==(s,scope,cell)];assert len(rr)==6
        sources.append(dict(seed=s,scope=scope,cell=cell,scores=mean_scores(rr)))
    for scope,cell in itertools.product(('old','all'),CELLS):
        rr=[r for r in sources if (r['scope'],r['cell'])==(scope,cell)];assert len(rr)==4
        overall.append(dict(scope=scope,cell=cell,scores=mean_scores(rr)))
    gate=[dict(seed=r['seed'],mask=m,J=r['scores']['new12'][m]['J'],passed=r['scores']['new12'][m]['J']>=.8) for r in sources if r['scope']=='all' and r['cell']=='new_new' for m in ('food_only','water_only')]
    return dict(sources=sources,overall=overall,primary='private all/new_new/new12 J; source mean',gate=gate,gate_all_passed=all(r['passed'] for r in gate))

def image_crop(path):
    im=ImageOps.exif_transpose(Image.open(path)).convert('RGB');w,h=im.size
    size=(round(256*w/min(w,h)),round(256*h/min(w,h)))
    im=im.resize(size,Image.Resampling.BICUBIC);x=(size[0]-224)//2;y=(size[1]-224)//2
    return im.crop((x,y,x+224,y+224))

def pixels(rows,qa):
    for row in rows:
        ident=row['blind_id']
        for name in ('original','processed','crop','full'):
            qa.check(sha(row[name+'_path'])==row[name+'_sha256'],ident+'/'+name+' SHA')
        im=ImageOps.exif_transpose(Image.open(row['original_path'])).convert('RGB')
        im.thumbnail((640,640),Image.Resampling.LANCZOS);buffer=io.BytesIO();im.save(buffer,format='JPEG',quality=95)
        qa.check(hashlib.sha256(buffer.getvalue()).hexdigest()==row['processed_sha256'],ident+' recreated640')
        crop=np.asarray(image_crop(row['processed_path']))
        qa.check(np.array_equal(crop,np.asarray(Image.open(row['crop_path']).convert('RGB'))),ident+' exact224')
        x=crop.astype(np.float32)/255
        x=(x-np.asarray([.485,.456,.406],np.float32))/np.asarray([.229,.224,.225],np.float32)
        qa.close(x.transpose(2,0,1),encode_images.preprocess(row['processed_path']).numpy(),ident+' normalized224',atol=0)

def render(projected,w):
    n=len(w['map_id']);frame=[];r=torch.arange(n)
    for side in (0,1):
        visual=torch.zeros(n,6,64,dtype=torch.float32);exists=torch.zeros(n,6,dtype=torch.float32)
        for k in (0,1):
            chosen=np.arange(n) if side==0 else np.flatnonzero(w['shown']==k)
            rr=r[chosen];places=torch.from_numpy(w['positions'][chosen,k]);ids=torch.from_numpy(w['photo_ids'][chosen,k])
            visual[rr,places]=projected[ids];exists[rr,places]=1.
        frame.append(torch.cat((visual.flatten(1),exists),dim=1))
    bits=torch.ones(n,2,dtype=torch.float32);bits[:,1]=0
    return torch.stack(frame,dim=1),bits

@torch.no_grad()
def replay(features,tables,out,qa):
    seed,p,d=33101,1,0;prepared=torch.load(PRIOR/f'prepared_{seed}.pt',weights_only=True)
    records=[]
    for scope in ('old','all'):
        a=remake_agents(seed,prepared,7,2,'identity')[d]
        # Head initialization is irrelevant after strict full-state load.
        head=temporal.make_head(0);blob=torch.load(PRIOR/'private'/f's{seed}_p{p}_d{d}_{scope}'/'final.pt',weights_only=True)
        a.load_state_dict(blob['agent'],strict=True);head.load_state_dict(blob['head'],strict=True)
        for name,value in a.named_parameters():value.requires_grad_(name.startswith(('memory.','slot_phi.')))
        a.eval();head.eval();before=[v.clone() for v in list(a.state_dict().values())+list(head.state_dict().values())]
        projected=a.project(torch.from_numpy(features))
        for cell,w in tables.items():
            hs=[];ls=[]
            for lo in range(0,len(w['map_id']),256):
                chunk={k:v[lo:lo+256] for k,v in w.items()};f,b=render(projected,chunk)
                hs.append(temporal.observe_sequence(a,f,b,'full').numpy())
            h=np.concatenate(hs)
            for lo in range(0,len(h),256):ls.append(head(torch.from_numpy(h[lo:lo+256])).reshape(-1,2,6).numpy())
            logits=np.concatenate(ls);raw=load(out/'tables'/f's{seed}_p{p}_d{d}_{scope}_{cell}.npz')
            qa.close(h,raw['h'],scope+'/'+cell+' independent h',atol=0)
            qa.close(logits,raw['logits'],scope+'/'+cell+' independent logits',atol=0)
            records.append(dict(seed=seed,partition=p,direction=d,scope=scope,cell=cell,worlds=len(h),h_exact=True,logits_exact=True))
        after=list(a.state_dict().values())+list(head.state_dict().values())
        qa.check(all(torch.equal(x,y) for x,y in zip(before,after)),scope+' immutable state')
        qa.check(all(x.grad is None for x in list(a.parameters())+list(head.parameters())),scope+' no gradients')
    return records

def selftest():
    qa=QA()
    for p in (1,2,3):qa.check({g:len(v) for g,v in groups(p).items()}==dict(old=18,added=6,sealed=6,common30=30,new12=12),'partition')
    w=table([1,2,3],[9,10]);n=len(w['map_id']);qa.check(n==360,'rectangle')
    logits=np.zeros((n,2,6),np.float32);row=np.arange(n)
    for k in (0,1):logits[row,k,w['positions'][:,k]]=100
    s=scores(dict(w,logits=logits),1)
    qa.check(all(s[g][m]['J']==1 and s[g][m]['Q']==1 for g in s for m in s[g]),'perfect')
    s=scores(dict(w,logits=np.zeros_like(logits)),1)
    qa.close(s['common30']['pooled']['Q'],1/36,'uniformQ');qa.check(s['common30']['pooled']['J']==0,'firstargmax sameplace')
    qa.close(s['common30']['pooled']['food'],1/6,'uniform greedy food')
    return dict(passed=True,checks=qa.checks)

def audit(out):
    started=time.monotonic();qa=QA();torch.set_num_threads(1)
    complete=read(out/'completion.json');qa.check(complete['status']=='complete','complete gate')
    for key,value in dict(private_endpoints=48,tables=192,private_world_forwards=224640,new_images=11,training_updates=0,social_runs=0,confirmation_images=0,network_requests=0,frozen_inputs_unchanged=True).items():qa.check(complete[key]==value,'completion/'+key)
    freeze=read(out/'freeze.json');inputs=freeze['input_hashes']
    for path,digest in inputs.items():qa.check(sha(path)==digest,'frozen/'+path)
    qa.check(sha(out/'selection.json')==freeze['selection_sha256'],'selection SHA')
    selected=read(out/'selection.json')['rows'];ids=[r['blind_id'] for r in selected]
    status=read(BASE/'pixel_workflow_003/curation_status_20260916.json')
    qa.check(ids==sorted(status['cumulative']['usable_ids']) and len(ids)==11,'all11 fixed sorted selection')
    qa.check({k:sum(r['stratum']==k for r in selected) for k in ('apple','banana','orange','water')}==dict(apple=3,banana=3,orange=3,water=2),'strata counts')
    reserved=read(BASE/'provisional_allocations.json')['records'];confirm={r['pageid'] for r in reserved if r['provisional_assignment']=='confirmation'}
    qa.check(len(confirm)==48 and not ({int(r['candidate_id'].split(':')[1]) for r in selected}&confirm),'no confirmation identity')
    exposure=read(out/'model_exposure_started.json');qa.check(exposure['blind_ids']==ids,'all11 exposure record')
    pixels(selected,qa)
    manifest=read(PROJECT/'redesign_v0.4/data/manifest.json')['images'];z=load(PROJECT/'redesign_v0.4/data/features.npz')['features'].astype(np.float32)
    qa.check(z.shape==(60,1024),'old60 cache');train=np.asarray([e['split']=='train' for e in manifest])
    qa.check(int(train.sum())==44,'old44 normalization support');center=z[train].mean(0);scale=float(np.sqrt(((z[train]-center)**2).mean()))
    norms=load(out/'normalization.npz');qa.close(norms['center'],center,'center old train only',atol=0);qa.close(norms['scale'],scale,'scale old train only',atol=0)
    new=load(out/'new_features.npz')['features'];qa.check(new.shape==(11,1024) and new.dtype==np.float32 and np.isfinite(new).all(),'newfeatures shape')
    features=load(out/'combined_normalized_features.npz')['features'];expected=np.concatenate(((z-center)/max(scale,1e-6),(new-center)/max(scale,1e-6)))
    qa.close(features,expected,'combined old normalization',atol=0)
    anchor=load(out/'encoder_anchor.npz');anchorqa=read(out/'encoder_anchor_qa.json')
    qa.close(anchor['cached'],z[:8],'anchor old cache',atol=0);qa.close(anchor['reencoded'],z[:8],'saved DINO anchor',atol=1e-4,rtol=1e-4)
    qa.check(anchorqa['passed'] and anchorqa['n']==8 and anchorqa['atol']==anchorqa['rtol']==1e-4,'fixed anchor tolerance')
    qa.close(anchorqa['max_abs_error'],np.max(np.abs(anchor['reencoded']-z[:8])),'anchor maxerror')
    receipt=read(out/'encoder_receipt.json');qa.check(receipt['features_sha256']==sha(out/'new_features.npz'),'newfeatures receipt')
    for k,v in dict(new_images=11,anchor_images=8,forward_batches=3,training_updates=0,device='mps',input='pixels only').items():qa.check(receipt[k]==v,'encoder/'+k)
    oldpools=[np.asarray([i for i,e in enumerate(manifest) if e['split']=='test' and e['category']==name],np.int64)[:4] for name in ('food','water')]
    newpools=[np.asarray([60+i for i,r in enumerate(selected) if (r['stratum']=='water')==bool(k)],np.int64) for k in (0,1)]
    pools={'old':oldpools,'new':newpools};tables={cell:table(pools[cell.split('_')[0]][0],pools[cell.split('_')[1]][1]) for cell in CELLS}
    wm=read(out/'world_manifest.json');qa.tree(wm['pools'],{k:[x.tolist() for x in v] for k,v in pools.items()},'world pools');qa.tree(wm['counts'],EXPECTED_COUNTS,'world counts')
    source_rows=read(out/'raw_summaries.json');lookup={(r['seed'],r['partition'],r['direction'],r['scope'],r['cell']):r for r in source_rows}
    keys=list(itertools.product(SEEDS,(1,2,3),(0,1),('old','all'),CELLS));qa.check(len(lookup)==len(source_rows)==192 and set(lookup)==set(keys),'all192 table inventory')
    outputs={};rows=[];legacy=[];total=0
    for s,p,d,scope,cell in keys:
        record=lookup[s,p,d,scope,cell];path=out/'tables'/f's{s}_p{p}_d{d}_{scope}_{cell}.npz'
        qa.check(Path(record['file']).resolve()==path.resolve(),'raw path');digest=sha(path);qa.check(digest==record['sha256'],'raw SHA');outputs[str(path)]=digest
        raw=load(path);w=tables[cell];n=EXPECTED_COUNTS[cell];total+=n
        qa.check(set(raw)==set(WORLD_KEYS)|{'h','logits'},'raw keys')
        for k in WORLD_KEYS:qa.check(raw[k].dtype==np.int64 and np.array_equal(raw[k],w[k]),path.name+'/'+k)
        qa.check(raw['h'].shape==(n,96) and raw['logits'].shape==(n,2,6),'raw tensor shapes')
        qa.check(all(raw[k].dtype==np.float32 and np.isfinite(raw[k]).all() for k in ('h','logits')),'finite float32')
        calculated=scores(raw,p);qa.tree(calculated,record['scores'],path.name+'/scores')
        rows.append(dict(seed=s,partition=p,direction=d,scope=scope,cell=cell,file=str(path),sha256=digest,scores=calculated))
        if cell=='old_old':
            ref=load(PRIOR/'private'/f's{s}_p{p}_d{d}_{scope}'/'evaluation_2400.npz')
            for k in WORLD_KEYS:qa.check(np.array_equal(raw[k],ref[k]),'legacy world/'+k)
            qa.close(raw['logits'],ref['logits'],'legacy logits',atol=1e-6,rtol=1e-5)
            same=np.array_equal(raw['logits'].argmax(-1),ref['logits'].argmax(-1));qa.check(same,'legacy greedy')
            legacy.append(dict(seed=s,partition=p,direction=d,scope=scope,passed=True,max_abs_logits_error=float(np.max(np.abs(raw['logits']-ref['logits']))),same_actions=bool(same)))
    qa.check(total==224640,'world total');qa.tree(dict(passed=True,rows=legacy),read(out/'legacy_control_qa.json'),'all48 legacy control')
    summary=aggregate(rows);qa.tree(summary,read(out/'summary.json'),'independent source aggregate')
    replays=replay(features,tables,out,qa)
    for path,digest in inputs.items():qa.check(sha(path)==digest,'post-audit frozen/'+path)
    for name in ('completion.json','freeze.json','selection.json','normalization.npz','combined_normalized_features.npz','new_features.npz','encoder_anchor.npz','encoder_anchor_qa.json','encoder_receipt.json','model_exposure_started.json','world_manifest.json','raw_summaries.json','summary.json','legacy_control_qa.json'):outputs[str(out/name)]=sha(out/name)
    analysis=dict(status='complete',kind='independent finite-material metrics',source_sha256=sha(__file__),rows=rows,**summary)
    write(out/'analysis.json',analysis)
    comparison=dict(passed=True,scalar_and_array_comparisons=qa.comparisons,max_abs_error=qa.max_abs_error,analysis_sha256=sha(out/'analysis.json'),production_summary_sha256=sha(out/'summary.json'),note='Maximum includes prespecified numerical DINO-anchor and legacy tolerances; independent metrics use atol1e-12.')
    write(out/'comparison.json',comparison)
    result=dict(status='passed',passed=True,created_utc=now(),seconds=time.monotonic()-started,checks=qa.checks,failures=[],comparisons=qa.comparisons,max_abs_error=qa.max_abs_error,source_sha256=sha(__file__),input_hashes=inputs,output_hashes=outputs,analysis_sha256=sha(out/'analysis.json'),comparison_sha256=sha(out/'comparison.json'),selftest=selftest(),replays=replays,
        coverage=dict(raw_tables=192,private_endpoints=48,raw_worlds=total,legacy_controls=48,replayed_endpoints=2,replayed_tables=8,replayed_worlds=sum(r['worlds'] for r in replays),new_images_pixel_preprocessing=11,DINO_forward_replayed=0,training_updates=0,confirmation_pixels_read=0),
        limits=['Full metrics/world/hash checks for all48 private endpoints; model forward replay restricted before results to33101/p1/d0 old and all.', 'DINO anchor/raw features checked from saved arrays and hashed source only; no independent DINO inference.', 'Frozen Camp and temporal atomic modules reused; world construction, preprocessing, scalar metrics and aggregation independently implemented.', 'Readonly exposure boundary checks use the11 selected identities against confirmation48 metadata; this is not an OS-wide pixel-access audit.', 'The11 selected workflow images are already exposed materials, not an independent held-out confirmation sample.'])
    write(out/'audit_execution.json',result)
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path);parser.add_argument('--selftest',action='store_true');args=parser.parse_args()
    if args.selftest:print(json.dumps(selftest()));sys.exit(0)
    if args.out is None:parser.error('--out or --selftest is required')
    out=args.out.resolve()
    try:
        result=audit(out);print(json.dumps({k:result[k] for k in ('status','checks','comparisons','max_abs_error','coverage','source_sha256')},ensure_ascii=False))
    except Exception as exc:
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
        receipt=dict(status='failed',created_utc=now(),source_sha256=sha(__file__),error=repr(exc),traceback=traceback.format_exc())
        if out.exists():write(out/f'audit_failure_{stamp}.json',receipt)
        raise

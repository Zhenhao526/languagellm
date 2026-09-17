"""Fixed source-supported prefix acquisition using prior verified primitives."""
import importlib.util,itertools,json,hashlib,sys,shutil,time
from pathlib import Path
from datetime import datetime,timezone,timedelta
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parents[1]
BASE=PROJECT/'paper_program/visual_confirmation_v1';OUT=ROOT/'pixel_001'
REVIEW=PROJECT/'paper_program/image_review_batches/batch_004'
PRIMITIVE=BASE/'pixel_workflow_003/collect_pixels.py'
spec=importlib.util.spec_from_file_location('prior_pixel_primitives',PRIMITIVE);helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
helper.OUT=OUT;helper.ROOT=PROJECT
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def now():return datetime.now(timezone.utc)

def prepare():
    assert not OUT.exists() and not REVIEW.exists(),'No overwrite/restart'
    source=read(ROOT/'source_decisions.json');assert source['status']=='complete_finite_source_review'
    for path,expected in source['evidence_hashes'].items():
        assert sha(path)==expected,'Source evidence changed: '+path
    selected=read(ROOT/'source_001/selection.json')['rows'];meta={r['id']:r for r in read(PROJECT/'paper_program/visual_confirmation_v2_water/manifest.json')['images']}
    chosen=[]
    for row in selected:
        if row['ordinal'] in (8,19,22,42) and row['ordinal'] in source['supported_ordinals']:
            m=meta[row['id']];ident=hashlib.sha256(('water_source_followup_001|pixel-blind|'+row['id']).encode()).hexdigest()[:12]
            chosen.append(dict(blind_id=ident,ordinal=row['ordinal'],metadata=m))
    assert len(chosen)<=4 and [r['ordinal'] for r in chosen]==[v for v in (8,19,22,42) if v in source['supported_ordinals']]
    assert len({r['metadata']['cluster_id'] for r in chosen})==len(chosen)
    assert all(r['metadata']['original_url'].startswith('https://upload.wikimedia.org/') for r in chosen)
    old=read(PROJECT/'redesign_v0.4/data/candidates_downloaded.json')['images'];refs=[];missing=[]
    for r in old:
        p=PROJECT/'redesign_v0.4'/r['path']
        if r.get('download_status')=='ok' and p.is_file():
            assert sha(p)==r['sha256'];refs.append(dict(id=r['id'],group='old90',path=str(p),sha256=r['sha256'],original_sha1=r.get('original_sha1')))
        else:missing.append(r['id'])
    for b in (1,2,3):
        for r in read(BASE/f'pixel_workflow_{b:03d}/download_manifest.json')['records']:
            p=Path(r['processed_path']);assert sha(p)==r['processed_sha256']
            refs.append(dict(id=r['blind_id'],group=f'v1_batch{b}',path=str(p),sha256=r['processed_sha256'],original_sha1=r['original_sha1']))
    assert len(refs)==150 and len(missing)==11
    paths=[Path(__file__),PRIMITIVE,ROOT/'像素执行方案.md',ROOT/'来源核查固定方案.md',ROOT/'source_decisions.json',ROOT/'source_001/freeze.json',ROOT/'source_001/selection.json',ROOT/'source_001/completion.json',PROJECT/'redesign_v0.4/encode_images.py',PROJECT/'redesign_v0.4/data/candidates_downloaded.json']
    paths += [Path(p) for p in source['evidence_hashes']]
    paths += [PROJECT/'paper_program/visual_confirmation_v2_water/manifest.json']
    paths += [Path(r['path']) for r in refs]+[BASE/f'pixel_workflow_{b:03d}/download_manifest.json' for b in (1,2,3)]
    hashes={str(p):sha(p) for p in sorted(set(paths))}
    OUT.mkdir();[ (OUT/n).mkdir() for n in ('responses','originals','processed','blind') ]
    write(OUT/'selection_manifest.json',dict(rows=chosen,original_workflow_ordinals=[8,19,22,42],reserved_ordinals=[46,51,62,70],no_replacement=True))
    write(OUT/'comparison_inputs.json',dict(reference=refs,old_unavailable=missing,confirmation_pixels=False))
    write(OUT/'freeze.json',dict(created_utc=now().isoformat(),before_pixel_requests=True,input_hashes=hashes,selection_sha256=sha(OUT/'selection_manifest.json'),comparison_sha256=sha(OUT/'comparison_inputs.json'),planned=len(chosen),old_reference_count=150))
    return chosen,refs,hashes

def finish(records,refs,hashes,planned,stopped):
    prior=[dict(r,**helper.hash_image(Image.open(r['path']))) for r in refs];flags=[];count=0
    for r in records:
        if 'processed_path' not in r:continue
        for other in prior:
            count+=1;pd=(int(r['hashes']['phash'],16)^int(other['phash'],16)).bit_count();dd=(int(r['hashes']['dhash'],16)^int(other['dhash'],16)).bit_count()
            exact1=r['original_sha1']==other.get('original_sha1');exact2=r['processed_sha256']==other['sha256']
            if pd<=8 or dd<=6 or exact1 or exact2:flags.append(dict(a=r['blind_id'],b=other['id'],b_group=other['group'],phash_distance=pd,dhash_distance=dd,exact_original=exact1,exact_processed=exact2))
        prior.append(dict(id=r['blind_id'],group='current',sha256=r['processed_sha256'],original_sha1=r['original_sha1'],**r['hashes']))
    n=sum('processed_path' in r for r in records);assert count==n*150+n*(n-1)//2
    write(OUT/'near_duplicate_flags.json',dict(comparisons=count,reference_count=150,flags=flags,rows=prior,unavailable_old=11))
    REVIEW.mkdir(parents=True);ordinary=sorted([r for r in records if r.get('ordinary_single_image')],key=lambda x:x['blind_id']);images=[]
    for r in ordinary:
        for kind in ('full','crop'):shutil.copy2(r[kind+'_path'],REVIEW/f'{r["blind_id"]}_{kind}.png')
        images.append(dict(id=r['blind_id'],full=str(REVIEW/f'{r["blind_id"]}_full.png'),crop=str(REVIEW/f'{r["blind_id"]}_crop.png')))
    sheets=[]
    for lo in range(0,len(images),4):
        canvas=Image.new('RGB',(1200,650),'white');draw=ImageDraw.Draw(canvas);font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',18)
        for j,r in enumerate(images[lo:lo+4]):
            x,y=(j%2)*600,(j//2)*325;draw.text((x+10,y+8),r['id'],fill='black',font=font)
            full=Image.open(r['full']);full.thumbnail((350,270));canvas.paste(full,(x+8+(350-full.width)//2,y+38+(270-full.height)//2));canvas.paste(Image.open(r['crop']),(x+370,y+55))
        p=REVIEW/f'sheet_{lo//4+1:02d}.png';canvas.save(p);sheets.append(str(p))
    packet=dict(ids=[r['id'] for r in images],sheets=sheets,images=images,criteria=['real photograph','food or visible water in transparent container identifiable','crop retains resource','no prominent text or watermark','no mixed foods or ambiguous liquid'],notes='Full image left; actual224 crop right. Ice and liquid water are the same resource; visible liquid and transparent vessel still required. Judge pixels only, not filenames or presumed labels.',pixel_hashes={p:sha(p) for r in images for p in (r['full'],r['crop'])},sheet_hashes={p:sha(p) for p in sheets})
    write(REVIEW/'packet.json',packet)
    assert all(sha(p)==h for p,h in hashes.items())
    status='acquisition_stopped_visual_review_pending' if stopped else 'acquisition_finished_visual_review_pending'
    write(OUT/'completion.json',dict(status=status,ended_utc=now().isoformat(),planned=planned,processed=len(records),unattempted=planned-len(records),stopped=stopped,downloaded=n,ordinary=len(ordinary),errors=[r for r in records if r.get('status')=='error'],requests=sum(1 for _ in (OUT/'requests.jsonl').open()) if (OUT/'requests.jsonl').exists() else 0,near_duplicate_comparisons=count,near_duplicate_flags=len(flags),review_packet=str(REVIEW/'packet.json'),review_packet_sha256=sha(REVIEW/'packet.json'),source_inputs_unchanged=True,model_calls=0,confirmation_pixels=0,reserved_pixel_reads=0))
    print(json.dumps(dict(status=status,downloaded=n,ordinary=len(ordinary),near_duplicate_flags=len(flags))),flush=True)

def main():
    chosen,refs,hashes=prepare();records=[];prev=None;stop=False
    write(OUT/'process_handle.json',dict(pid=helper.os.getpid(),started_utc=now().isoformat(),source_sha256=sha(__file__)))
    for index,row in enumerate(chosen,1):
        record=dict(candidate_id=row['metadata']['id'],blind_id=row['blind_id'],status='pending')
        for attempt in (1,2):
            ctx=dict(planned=len(chosen),processed=len(records),model_calls=0,confirmation_requests=0)
            if prev:helper.wait_until('serial_interval',prev['finished']['monotonic']+10,datetime.fromisoformat(prev['finished']['utc'])+timedelta(seconds=10),ctx)
            req=helper.request_once(row,attempt,index);prev=req;code=req['http_status']
            if req['explicit_refusal'] or (code is not None and 300<=code<400):record.update(status='error',error='access_refusal_or_redirect');stop=True;break
            if code==200 and req['error'] is None and req['body_complete']:
                try:record=helper.process_image(row,req)
                except Exception as e:record.update(status='error',error=str(e),raw_path=req['raw_path'])
                break
            record.update(status='error',error=req['error'] or f'HTTP {code}',raw_path=req['raw_path'])
            if code in (429,503):
                if attempt==2:stop=True;break
                try:delay,_=helper.retry_delay(req['headers'],datetime.fromisoformat(req['finished']['utc']))
                except Exception:stop=True;break
                helper.wait_until('retry_after',req['finished']['monotonic']+delay+1,datetime.fromisoformat(req['finished']['utc'])+timedelta(seconds=delay+1),ctx);continue
            if attempt==1 and (req['transient_network_error'] or (code is not None and code>=500)):
                helper.wait_until('transient',req['finished']['monotonic']+10,datetime.fromisoformat(req['finished']['utc'])+timedelta(seconds=10),ctx);continue
            break
        records.append(record);write(OUT/'download_manifest.json',dict(records=records,planned=len(chosen),source_sha256=sha(__file__)))
        print(json.dumps(dict(processed=len(records),status=record['status'],planned=len(chosen))),flush=True)
        if stop:break
    finish(records,refs,hashes,len(chosen),stop)
if __name__=='__main__':main()

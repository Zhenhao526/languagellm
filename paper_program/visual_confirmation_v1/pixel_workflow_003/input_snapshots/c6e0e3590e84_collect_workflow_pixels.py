"""Collect only frozen workflow reservations, with no model calls or new sampling."""
from pathlib import Path
from datetime import datetime,timezone
from email.utils import parsedate_to_datetime
import hashlib,io,json,platform,sys,time,urllib.request,urllib.error
import numpy as np
import PIL
from PIL import Image,ImageOps,ImageDraw,ImageFont

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=HERE/'pixel_workflow_001'
SALT='visual_confirmation_v1|2026-09-15|prospective|no_model_scores'
LIMIT=50*1024*1024

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def now():return datetime.now(timezone.utc).isoformat()
def hash_image(im):
    gray=ImageOps.exif_transpose(im).convert('RGB').convert('L')
    a=np.asarray(gray.resize((32,32),Image.Resampling.LANCZOS),dtype=np.float64)
    x=np.arange(32);k=np.arange(32)
    c=np.cos(np.pi*(2*x[None,:]+1)*k[:,None]/64)*np.sqrt(2/32);c[0]/=np.sqrt(2)
    low=(c@a@c.T)[:8,:8];pb=(low>np.median(low)).reshape(-1)
    small=np.asarray(gray.resize((9,8),Image.Resampling.LANCZOS));db=(small[:,1:]>small[:,:-1]).reshape(-1)
    return dict(phash=np.packbits(pb).tobytes().hex(),dhash=np.packbits(db).tobytes().hex())

def crop_image(im):
    im=ImageOps.exif_transpose(im).convert('RGB');w,h=im.size
    size=(round(w*256/min(w,h)),round(h*256/min(w,h)))
    im=im.resize(size,Image.Resampling.BICUBIC);x,y=(size[0]-224)//2,(size[1]-224)//2
    return im.crop((x,y,x+224,y+224))

def main():
    assert (platform.python_version(),PIL.__version__,np.__version__)==('3.12.14','12.3.0','1.26.4')
    assert hash_image(Image.new('RGB',(40,40),'gray'))==hash_image(Image.new('RGB',(40,40),'gray'))
    files=[Path(__file__),HERE/'pixel_workflow_plan.md',HERE/'prospective_sampling_plan.md',HERE/'manifest.json',
        HERE/'candidate_review_order.json',ROOT/'redesign_v0.4/encode_images.py',ROOT/'redesign_v0.4/data/candidates_downloaded.json']
    hashes={str(p):sha(p) for p in files}
    OUT.mkdir(exist_ok=False);(OUT/'originals').mkdir();(OUT/'processed').mkdir();(OUT/'blind').mkdir()
    write(OUT/'freeze.json',dict(started_utc=now(),source_hashes=hashes,python=platform.python_version(),
        pillow=PIL.__version__,numpy=np.__version__,model_calls=0,confirmation_requests=0))
    rows=json.loads((HERE/'manifest.json').read_text())['images']
    selected=[r for r in rows if r['provisional_assignment']=='workflow']
    assert len(selected)==24 and all(not r['exclusion_reasons'] for r in selected)
    selected.sort(key=lambda r:r['cluster_rank'])
    records=[];last=0.;stopped=False
    for row in selected:
        ident=hashlib.sha256((SALT+'|pixel-blind|'+row['id']).encode()).hexdigest()[:12]
        record=dict(candidate_id=row['id'],blind_id=ident,status='pending')
        for attempt in range(2):
            time.sleep(max(0,10-(time.monotonic()-last)))
            started=now();last=time.monotonic();headers={};code=None
            request=urllib.request.Request(row['original_url'],headers={'User-Agent':'LanguageEmergenceResearch/0.13 (local research prototype; serialized requests)'})
            try:
                with urllib.request.urlopen(request,timeout=50) as response:
                    code=response.status;headers={k.lower():v for k,v in response.headers.items()}
                    raw=response.read(LIMIT+1)
                if len(raw)>LIMIT:raise ValueError('Original exceeds frozen50MiB budget')
                assert hashlib.sha1(raw).hexdigest()==row['original_sha1'],'Original version SHA1 changed'
                im=ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert('RGB')
                assert sorted(im.size)==sorted((row['width'],row['height'])),'Original dimensions changed'
                suffix='.png' if row['mime']=='image/png' else '.jpg'
                original=OUT/'originals'/f'{ident}{suffix}';original.write_bytes(raw)
                im.thumbnail((640,640),Image.Resampling.LANCZOS);processed=OUT/'processed'/f'{ident}.jpg'
                im.save(processed,quality=95,format='JPEG');image=Image.open(processed)
                crop=crop_image(image);crop.save(OUT/'blind'/f'{ident}_crop.png')
                Image.open(original).convert('RGB').save(OUT/'blind'/f'{ident}_full.png')
                # Use existing preprocessing solely as a tensor equality check; no backbone load.
                sys.path.insert(0,str(ROOT/'redesign_v0.4'))
                from encode_images import preprocess
                arr=np.asarray(crop).astype(np.float32)/255
                arr=(arr-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
                assert np.array_equal(arr.transpose(2,0,1),preprocess(processed).numpy())
                record.update(status='downloaded',original_path=str(original),original_sha256=sha(original),
                    original_sha1=hashlib.sha1(raw).hexdigest(),processed_path=str(processed),processed_sha256=sha(processed),
                    crop_sha256=sha(OUT/'blind'/f'{ident}_crop.png'),hashes=hash_image(image),
                    preprocess_matches_legacy=True,bytes=len(raw),source_page=row['source_page'],
                    license=row['license'],license_url=row['license_url'],artist_html=row['artist_html'],
                    credit_html=row['credit_html'],modification='EXIF orientation/RGB; max640 LANCZOS/JPEG95; fixed224crop')
                error=None
            except Exception as e:
                error=f'{type(e).__name__}: {e}';record.update(status='error',error=error)
                if isinstance(e,urllib.error.HTTPError):code=e.code;headers={k.lower():v for k,v in e.headers.items()}
            with (OUT/'requests.jsonl').open('a') as f:f.write(json.dumps(dict(started_utc=started,finished_utc=now(),
                candidate_id=row['id'],url=row['original_url'],attempt=attempt+1,http_status=code,headers=headers,error=error))+'\n')
            if record['status']=='downloaded':break
            if code in (403,429,503):
                if code==403 or attempt==1:stopped=True;break
                retry=headers.get('retry-after','30')
                try:delay=float(retry)
                except ValueError:delay=max(0,(parsedate_to_datetime(retry)-datetime.now(timezone.utc)).total_seconds())
                time.sleep(max(0,delay));continue
            if attempt==0 and (code is None or code>=500):time.sleep(5);continue
            break
        records.append(record)
        write(OUT/'download_manifest.json',dict(status='running',records=records,planned=24,model_calls=0,confirmation_requests=0))
        print(json.dumps(dict(processed=len(records),planned=24,status=record['status'],blind_id=ident)),flush=True)
        if stopped:break
    old=json.loads((ROOT/'redesign_v0.4/data/candidates_downloaded.json').read_text())['images'];oldhash=[]
    assert len(old)==101
    for row in old:
        path=ROOT/'redesign_v0.4'/row['path'];assert sha(path)==row['sha256']
        oldhash.append(dict(id=row['id'],path=str(path),sha256=sha(path),**hash_image(Image.open(path))))
    flags=[];new=[r for r in records if r['status']=='downloaded']
    for i,a in enumerate(new):
        for b,group in [(r,'old101') for r in oldhash]+[(dict(id=r['blind_id'],**r['hashes']),'new_workflow') for r in new[:i]]:
            pd=(int(a['hashes']['phash'],16)^int(b['phash'],16)).bit_count()
            dd=(int(a['hashes']['dhash'],16)^int(b['dhash'],16)).bit_count()
            if pd<=8 or dd<=6:flags.append(dict(a=a['blind_id'],b=b['id'],b_group=group,phash_distance=pd,dhash_distance=dd))
    write(OUT/'near_duplicate_flags.json',dict(old_images=101,new_images=len(new),flags=flags,old_pixel_hashes=oldhash,
        review_pending=True,confirmation_pixels_compared=False))
    font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',18)
    ordered=sorted(new,key=lambda r:r['blind_id']);sheets=[]
    for start in range(0,len(ordered),4):
        canvas=Image.new('RGB',(1200,650),'white');draw=ImageDraw.Draw(canvas)
        for j,row in enumerate(ordered[start:start+4]):
            ident=row['blind_id'];x=(j%2)*600;y=(j//2)*325
            draw.text((x+10,y+8),ident,fill='black',font=font)
            full=ImageOps.exif_transpose(Image.open(row['original_path'])).convert('RGB');full.thumbnail((350,270))
            canvas.paste(full,(x+8+(350-full.width)//2,y+38+(270-full.height)//2))
            canvas.paste(Image.open(OUT/'blind'/f'{ident}_crop.png'),(x+370,y+55))
        path=OUT/'blind'/f'sheet_{start//4+1:02d}.png';canvas.save(path);sheets.append(str(path))
    write(OUT/'blind_review_packet.json',dict(ids=[r['blind_id'] for r in ordered],sheets=sheets,
        criteria=['real photograph','food or visible water in transparent container identifiable','crop retains resource',
            'no prominent text or watermark','no mixed foods or ambiguous liquid'],model_scores_available=False))
    assert all(sha(p)==v for p,v in hashes.items())
    write(OUT/'download_manifest.json',dict(status='stopped_access_limit' if stopped else 'download_phase_finished',records=records,
        planned=24,downloaded=len(new),model_calls=0,confirmation_requests=0,source_hashes=hashes,finished_utc=now()))

if __name__=='__main__':main()

"""Finite pixel technical QA; PIL/NumPy only, no viewing, network or model."""
import argparse,hashlib,io,json,sys,traceback,platform
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import PIL
from PIL import Image,ImageOps

ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parents[1]
OLD=PROJECT/'paper_program/visual_confirmation_v1'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def dt(x):return datetime.fromisoformat(x)
def now():return datetime.now(timezone.utc).isoformat()

class Checks:
    def __init__(self):self.rows=[]
    def check(self,passed,label):
        self.rows.append(dict(check=label,passed=bool(passed)))
        if not passed:raise AssertionError(label)

def hashes(path):
    # Independent implementation of the frozen pHash64/dHash64 definitions.
    gray=ImageOps.exif_transpose(Image.open(path)).convert('RGB').convert('L')
    pixels=np.asarray(gray.resize((32,32),Image.Resampling.LANCZOS),np.float64)
    cosine=np.cos(np.pi*(2*np.arange(32)[None,:]+1)*np.arange(32)[:,None]/64)*np.sqrt(2/32)
    cosine[0]/=np.sqrt(2);freq=(cosine@pixels@cosine.T)[:8,:8]
    phash=np.packbits((freq>np.median(freq)).ravel()).tobytes().hex()
    small=np.asarray(gray.resize((9,8),Image.Resampling.LANCZOS))
    dhash=np.packbits((small[:,1:]>small[:,:-1]).ravel()).tobytes().hex()
    return dict(phash=phash,dhash=dhash)

def crop(path):
    im=ImageOps.exif_transpose(Image.open(path)).convert('RGB');w,h=im.size
    size=(round(w*256/min(w,h)),round(h*256/min(w,h)))
    im=im.resize(size,Image.Resampling.BICUBIC);left=(size[0]-224)//2;top=(size[1]-224)//2
    return im.crop((left,top,left+224,top+224))

def main(out):
    q=Checks();complete=read(out/'completion.json');freeze=read(out/'freeze.json');selected=read(out/'selection_manifest.json');comparison=read(out/'comparison_inputs.json');source=read(ROOT/'source_decisions.json')
    requests=[json.loads(x) for x in (out/'requests.jsonl').read_text().splitlines()];waits=[json.loads(x) for x in (out/'waits.jsonl').read_text().splitlines()]
    download=read(out/'download_manifest.json');records=download['records'];chosen=selected['rows'];byid={r['blind_id']:r for r in chosen}
    q.check(complete['status']=='acquisition_finished_visual_review_pending' and not complete['stopped'] and complete['unattempted']==0,'complete fixed acquisition')
    q.check([r['ordinal'] for r in chosen]==[8,19,22,42] and selected['reserved_ordinals']==[46,51,62,70] and selected['no_replacement'],'fixed front4 no backfill')
    q.check(all(r['ordinal'] in source['supported_ordinals'] for r in chosen),'all selected source-supported')
    for p,h in source['evidence_hashes'].items():q.check(sha(p)==h,'source evidence hash '+p)
    for p,h in freeze['input_hashes'].items():q.check(sha(p)==h,'frozen input '+p)
    q.check(sha(out/'selection_manifest.json')==freeze['selection_sha256'] and sha(out/'comparison_inputs.json')==freeze['comparison_sha256'],'frozen selection/comparison')
    q.check(dt(freeze['created_utc'])<dt(requests[0]['started']['utc']) and freeze['before_pixel_requests'],'freeze before first request')
    q.check(len(requests)==complete['requests']==4 and all(r['attempt']==1 for r in requests),'actual4 HTTP no retries')
    q.check(len(records)==len(chosen)==complete['downloaded']==complete['ordinary']==complete['processed']==4,'four downloaded ordinary records')
    q.check(len({r['metadata']['id'] for r in chosen})==4 and len(byid)==4,'unique identities and blind IDs')
    reserved={r['pageid'] for r in read(OLD/'provisional_allocations.json')['records'] if r['provisional_assignment']=='confirmation'}
    source_rows=read(ROOT/'source_001/selection.json')['rows'];later={r['id'] for r in source_rows if r['ordinal'] in (46,51,62,70)}
    q.check(len(reserved)==48 and not {r['metadata']['pageid'] for r in chosen}&reserved,'no confirmation48 ID')
    q.check(not {r['metadata']['id'] for r in chosen}&later,'no retained last4 ID')
    gaps=[]
    for i,r in enumerate(requests):
        meta=byid[r['blind_id']]['metadata'];path=Path(r['raw_path'])
        q.check(r['candidate_id']==meta['id'] and r['url']==meta['original_url'] and r['url'].startswith('https://upload.wikimedia.org/'),'fixed original URL '+r['blind_id'])
        q.check(r['http_status']==200 and r['body_complete'] and not r['body_capped'] and r['error'] is None and not r['explicit_refusal'],'successful complete response '+r['blind_id'])
        q.check(path.parent==out/'responses' and sha(path)==r['raw_sha256'] and path.stat().st_size==r['raw_bytes']<=50*1024*1024,'raw bytes/SHA256 '+r['blind_id'])
        q.check(hashlib.sha1(path.read_bytes()).hexdigest()==meta['original_sha1'],'fixed original SHA1 '+r['blind_id'])
        q.check(r['timeout_seconds']==60 and r['read_limit_bytes']==50*1024*1024,'fixed timeout/limit')
        q.check(dt(r['finished']['utc'])>=dt(r['started']['utc']) and r['finished']['monotonic']>=r['started']['monotonic'],'nonnegative request interval')
        if i:
            prev=requests[i-1];utc=(dt(r['started']['utc'])-dt(prev['finished']['utc'])).total_seconds();mono=r['started']['monotonic']-prev['finished']['monotonic']
            gaps.append(dict(previous=prev['blind_id'],next=r['blind_id'],utc_seconds=utc,monotonic_seconds=mono));q.check(min(utc,mono)>=10,'both-clock serial interval')
    q.check(len(waits)==3,'three serial waits')
    for r in waits:q.check(r['reason']=='serial_interval' and dt(r['finished']['utc'])>=dt(r['deadline_utc']) and r['finished']['monotonic']>=r['deadline_monotonic'] and r['utc_deadline_met'] and r['monotonic_deadline_met'],'wait deadlines')
    expectedrefs=[];missing=[]
    for r in read(PROJECT/'redesign_v0.4/data/candidates_downloaded.json')['images']:
        p=PROJECT/'redesign_v0.4'/r['path']
        if r.get('download_status')=='ok' and p.is_file():expectedrefs.append(dict(id=r['id'],group='old90',path=str(p),sha256=r['sha256'],original_sha1=r.get('original_sha1')))
        else:missing.append(r['id'])
    for b in (1,2,3):
        for r in read(OLD/f'pixel_workflow_{b:03d}/download_manifest.json')['records']:
            expectedrefs.append(dict(id=r['blind_id'],group=f'v1_batch{b}',path=r['processed_path'],sha256=r['processed_sha256'],original_sha1=r['original_sha1']))
    q.check(comparison['reference']==expectedrefs and len(expectedrefs)==150,'exact150 old reference inventory')
    q.check(comparison['old_unavailable']==missing and len(missing)==11 and not comparison['confirmation_pixels'],'11 unavailable originals retained')
    reference=[]
    for r in expectedrefs:
        q.check(sha(r['path'])==r['sha256'],'reference bytes '+r['id']);reference.append(dict(r,**hashes(r['path'])))
    reconstructed=[];pixelrecords=[]
    for r in records:
        ident=r['blind_id'];meta=byid[ident]['metadata'];req=next(v for v in requests if v['blind_id']==ident)
        q.check(r['status']=='downloaded' and r['ordinary_single_image'] and r['candidate_id']==meta['id'],'ordinary image record '+ident)
        for kind in ('original','processed','crop','full'):
            p=Path(r[kind+'_path']);q.check(p.is_relative_to(out) and sha(p)==r[kind+'_sha256'],'new output/hash '+kind+'/'+ident)
        q.check(Path(req['raw_path']).read_bytes()==Path(r['original_path']).read_bytes(),'original matches response '+ident)
        decoded=Image.open(r['original_path']);oriented=ImageOps.exif_transpose(decoded).convert('RGB')
        q.check(decoded.format in ('JPEG','PNG') and getattr(decoded,'n_frames',1)==1,'ordinary decoded format '+ident)
        q.check(tuple((meta['width'],meta['height'])) in (decoded.size,oriented.size),'metadata size '+ident)
        q.check(np.array_equal(np.asarray(oriented),np.asarray(Image.open(r['full_path']).convert('RGB'))),'full EXIF/RGB pixels '+ident)
        reduced=oriented.copy();reduced.thumbnail((640,640),Image.Resampling.LANCZOS);buf=io.BytesIO();reduced.save(buf,format='JPEG',quality=95)
        q.check(hashlib.sha256(buf.getvalue()).hexdigest()==r['processed_sha256'],'exact640 JPEG95 reconstruction '+ident)
        cropped=np.asarray(crop(r['processed_path']));saved=np.asarray(Image.open(r['crop_path']).convert('RGB'))
        q.check(cropped.shape==(224,224,3) and np.array_equal(cropped,saved),'exact256/center224 pixels '+ident)
        # Verify finite standard tensor arithmetic without importing Torch/model code.
        normalized=(cropped.astype(np.float32)/255-np.asarray([.485,.456,.406],np.float32))/np.asarray([.229,.224,.225],np.float32)
        q.check(normalized.shape==(224,224,3) and np.isfinite(normalized).all() and r['preprocess_matches_legacy'],'finite fixed normalization '+ident)
        h=hashes(r['processed_path']);q.check(h==r['hashes'],'independent pHash/dHash '+ident)
        reconstructed.append(dict(id=ident,group='current',sha256=r['processed_sha256'],original_sha1=r['original_sha1'],**h))
        pixelrecords.append(dict(id=ident,ordinary=True,original_sha1=r['original_sha1'],reconstructed640_exact=True,crop224_exact=True,full_oriented_exact=True,hashes=h))
    flags=[];prior=list(reference);count=0
    for r in reconstructed:
        for other in prior:
            count+=1;pd=(int(r['phash'],16)^int(other['phash'],16)).bit_count();dd=(int(r['dhash'],16)^int(other['dhash'],16)).bit_count();sameoriginal=r['original_sha1']==other.get('original_sha1');sameprocessed=r['sha256']==other['sha256']
            if pd<=8 or dd<=6 or sameoriginal or sameprocessed:flags.append(dict(a=r['id'],b=other['id'],b_group=other['group'],phash_distance=pd,dhash_distance=dd,exact_original=sameoriginal,exact_processed=sameprocessed))
        prior.append(r)
    dup=read(out/'near_duplicate_flags.json');q.check(count==606==dup['comparisons'] and dup['reference_count']==150 and dup['unavailable_old']==11,'all606 comparisons')
    q.check(dup['flags']==flags and dup['rows']==prior,'all duplicate flags and154 hash rows')
    packetpath=Path(complete['review_packet']);packet=read(packetpath)
    q.check(packetpath==PROJECT/'paper_program/image_review_batches/batch_004/packet.json' and sha(packetpath)==complete['review_packet_sha256'],'isolated review packet')
    q.check(packet['ids']==sorted(r['blind_id'] for r in records) and len(packet['images'])==4 and len(packet['sheets'])==1,'fixed blind packet inventory')
    q.check(set(packet)=={'ids','sheets','images','criteria','notes','pixel_hashes','sheet_hashes'},'anonymous packet fields')
    for r in packet['images']:
        source=next(x for x in records if x['blind_id']==r['id']);q.check(set(r)=={'id','full','crop'},'no source mapping in review image entry')
        for kind in ('full','crop'):q.check(Path(r[kind]).parent==packetpath.parent and sha(r[kind])==source[kind+'_sha256']==packet['pixel_hashes'][r[kind]],'anonymous exact copy '+kind+'/'+r['id'])
    for p in packet['sheets']:q.check(Path(p).parent==packetpath.parent and sha(p)==packet['sheet_hashes'][p] and Image.open(p).size==(1200,650),'sheet bytes and dimensions')
    q.check(len(packet['criteria'])==5 and 'actual224' in packet['notes'],'five criteria and actual crop label')
    for p,h in freeze['input_hashes'].items():q.check(sha(p)==h,'post-audit frozen input '+p)
    q.check(complete['near_duplicate_comparisons']==count and complete['near_duplicate_flags']==len(flags) and complete['model_calls']==complete['confirmation_pixels']==complete['reserved_pixel_reads']==0 and complete['source_inputs_unchanged'],'completion counters and scope')
    result=dict(passed=True,status='passed_technical_not_visual_review',created_utc=now(),checks=len(q.rows),failures=[],check_records=q.rows,source_sha256=sha(__file__),runtime=dict(python=platform.python_version(),numpy=np.__version__,pillow=PIL.__version__),coverage=dict(downloads=4,requests=4,original_sha1_checked=4,processed_reconstructions=4,crop_reconstructions=4,prior_hash_rows=150,near_duplicate_comparisons=606,blind_packet_images=4,visual_judgments=0,network_requests_by_auditor=0,model_calls=0,confirmation_pixels=0,reserved_pixel_reads=0),gaps=gaps,pixel_records=pixelrecords,near_duplicate_flags=flags,
        input_hashes=freeze['input_hashes'],output_hashes={str(p):sha(p) for p in [out/'freeze.json',out/'selection_manifest.json',out/'comparison_inputs.json',out/'download_manifest.json',out/'requests.jsonl',out/'waits.jsonl',out/'near_duplicate_flags.json',out/'completion.json',packetpath]},
        limitations=['PIL/NumPy decodes are technical array checks, not visual content judgments; no view_image or learned model used.', 'Four successful fixed-front-prefix acquisitions only; no inference about all ordinary water pictures or verified independent photographers.', 'pHash/dHash recall plus exact file checks do not establish absence of every visual/source duplicate.', 'No alternate network/error branch was exercised in this all200 batch; source was reviewed but retries are not claimed dynamically tested.', 'ImageNet normalization arithmetic is checked numerically and the frozen old preprocessing source is hashed; no model or Torch module is imported by this auditor.', 'Confirmation avoidance is verified against selection/reference inventory and code/log scope, not OS-wide access tracing.'])
    write(out/'independent_technical_qa.json',result);print(json.dumps(dict(passed=True,checks=len(q.rows),comparisons=count,near_duplicate_flags=len(flags),source_sha256=sha(__file__),output_sha256=sha(out/'independent_technical_qa.json'))))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=ROOT/'pixel_001');args=parser.parse_args();out=args.out.resolve()
    try:main(out)
    except Exception as error:
        if out.exists():write(out/('technical_failure_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')+'.json'),dict(passed=False,error=repr(error),source_sha256=sha(__file__),traceback=traceback.format_exc()))
        raise

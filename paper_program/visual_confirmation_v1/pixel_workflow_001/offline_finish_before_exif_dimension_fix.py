"""Offline finish after downloads; explicitly account for unavailable old pixels."""
from pathlib import Path
from datetime import datetime,timezone
from email.utils import parsedate_to_datetime
import hashlib,json,shutil,collections
import sys
import numpy as np
from PIL import Image,ImageOps,ImageDraw,ImageFont
import collect_workflow_pixels as original

HERE=Path(__file__).resolve().parent;OUT=HERE/'pixel_workflow_001';ROOT=HERE.parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')

def main():
    current=json.loads((OUT/'download_manifest.json').read_text());new=[r for r in current['records'] if r['status']=='downloaded']
    assert len(new)==24 and len(current['records'])==24
    frozen=json.loads((OUT/'freeze.json').read_text())
    assert all(sha(p)==v for p,v in frozen['source_hashes'].items())
    selected=[r for r in json.loads((HERE/'manifest.json').read_text())['images'] if r['provisional_assignment']=='workflow']
    assert {r['id'] for r in selected}=={r['candidate_id'] for r in new}
    metadata={r['id']:r for r in selected}
    req=[json.loads(x) for x in (OUT/'requests.jsonl').read_text().splitlines()]
    assert len([r for r in req if r['http_status']==200 and r['error'] is None])==24
    waits=[];normal_intervals=[]
    for i,r in enumerate(req):
        if i:
            gap=(datetime.fromisoformat(r['started_utc'])-datetime.fromisoformat(req[i-1]['started_utc'])).total_seconds()
            assert gap>=10;normal_intervals.append(gap)
        if r['http_status'] in (429,503):
            nxt=req[i+1];assert nxt['candidate_id']==r['candidate_id'] and nxt['attempt']==2
            elapsed=(datetime.fromisoformat(nxt['started_utc'])-datetime.fromisoformat(r['finished_utc'])).total_seconds()
            header=r['headers'].get('retry-after','30')
            try:required=float(header)
            except ValueError:required=max(0,(parsedate_to_datetime(header)-datetime.fromisoformat(r['finished_utc'])).total_seconds())
            waits.append(dict(required_seconds=required,recorded_utc_seconds=elapsed,
                utc_interval_satisfies_header=elapsed>=required,shortfall_seconds=max(0,required-elapsed),
                source_requested_sleep_seconds=required,monotonic_end_not_recorded=True))
    old=json.loads((ROOT/'redesign_v0.4/data/candidates_downloaded.json').read_text())['images'];available=[];unavailable=[]
    for row in old:
        path=ROOT/'redesign_v0.4'/row['path']
        if row.get('download_status')!='ok' or not path.is_file():
            unavailable.append(dict(id=row['id'],download_status=row.get('download_status'),path=str(path)));continue
        assert sha(path)==row['sha256']
        available.append(dict(id=row['id'],path=str(path),sha256=sha(path),**original.hash_image(Image.open(path))))
    assert len(old)==101 and len(available)==90 and len(unavailable)==11
    flags=[];full_revisions=[]
    sys.path.insert(0,str(ROOT/'redesign_v0.4'))
    from encode_images import preprocess
    for i,a in enumerate(new):
        assert sha(a['original_path'])==a['original_sha256'] and sha(a['processed_path'])==a['processed_sha256']
        assert hashlib.sha1(Path(a['original_path']).read_bytes()).hexdigest()==metadata[a['candidate_id']]['original_sha1']==a['original_sha1']
        decoded=Image.open(a['original_path']);meta=metadata[a['candidate_id']]
        assert decoded.format=={'image/jpeg':'JPEG','image/png':'PNG'}[meta['mime']]
        assert decoded.size==(meta['width'],meta['height'])
        cp=OUT/'blind'/f"{a['blind_id']}_crop.png";assert sha(cp)==a['crop_sha256']
        crop=Image.open(cp);assert np.array_equal(np.asarray(crop),np.asarray(original.crop_image(Image.open(a['processed_path']))))
        arr=np.asarray(crop).astype(np.float32)/255
        arr=(arr-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
        assert np.array_equal(arr.transpose(2,0,1),preprocess(a['processed_path']).numpy())
        full=OUT/'blind'/f"{a['blind_id']}_full.png";before=sha(full)
        corrected=ImageOps.exif_transpose(Image.open(a['original_path'])).convert('RGB')
        if not np.array_equal(np.asarray(Image.open(full)),np.asarray(corrected)):
            corrected.save(full);full_revisions.append(dict(blind_id=a['blind_id'],before_sha256=before,after_sha256=sha(full)))
        assert original.hash_image(Image.open(a['processed_path']))==a['hashes']
        for b,group in [(r,'old_available90') for r in available]+[(dict(id=r['blind_id'],**r['hashes']),'new_workflow') for r in new[:i]]:
            pd=(int(a['hashes']['phash'],16)^int(b['phash'],16)).bit_count();dd=(int(a['hashes']['dhash'],16)^int(b['dhash'],16)).bit_count()
            if pd<=8 or dd<=6:flags.append(dict(a=a['blind_id'],b=b['id'],b_group=group,phash_distance=pd,dhash_distance=dd))
    write(OUT/'near_duplicate_flags.json',dict(old_records=101,old_available_pixels=90,old_unavailable=unavailable,new_images=24,
        comparisons=24*90+24*23//2,flags=flags,old_pixel_hashes=available,review_pending=True,confirmation_pixels_compared=False))
    already=set(json.loads((OUT/'review_round1/packet.json').read_text())['ids'])
    rows=sorted([r for r in new if r['blind_id'] not in already],key=lambda r:r['blind_id']);assert len(rows)==14
    dest=OUT/'review_round2';dest.mkdir(exist_ok=False);sheets=[]
    font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',18)
    for start in range(0,len(rows),4):
        canvas=Image.new('RGB',(1200,650),'white');draw=ImageDraw.Draw(canvas)
        for j,row in enumerate(rows[start:start+4]):
            ident=row['blind_id'];x=j%2*600;y=j//2*325;draw.text((x+10,y+8),ident,fill='black',font=font)
            full=ImageOps.exif_transpose(Image.open(row['original_path'])).convert('RGB');full.thumbnail((350,270))
            canvas.paste(full,(x+8+(350-full.width)//2,y+38+(270-full.height)//2));canvas.paste(Image.open(OUT/'blind'/f'{ident}_crop.png'),(x+370,y+55))
        path=dest/f'sheet_{start//4+1:02d}.png';canvas.save(path);sheets.append(str(path))
    write(dest/'packet.json',dict(ids=[r['blind_id'] for r in rows],sheets=sheets,image_sha256={p:sha(p) for p in sheets},scope='Remaining14 workflow photos; same frozen criteria; no source title/metadata/model scores'))
    round1=OUT/'review_round1';packet1=json.loads((round1/'packet.json').read_text())
    write(round1/'review_input_receipt.json',dict(packet_sha256=sha(round1/'packet.json'),
        image_sha256={p:sha(p) for p in packet1['sheets']},note='Computed after both independent reviews; original packet and sheets unchanged.'))
    shutil.copy2(OUT/'download_manifest.json',OUT/'download_manifest_before_offline_finish.json')
    current.update(status='download_complete_offline_review_pending',downloaded=24,model_calls=0,confirmation_requests=0,
        finished_utc=datetime.now(timezone.utc).isoformat(),old_available_pixels=90,old_unavailable_pixels=11,
        source_hashes=frozen['source_hashes'],postprocess_source_sha256=sha(__file__))
    write(OUT/'download_manifest.json',current)
    write(OUT/'offline_finish_qa.json',dict(data_integrity_passed=True,retry_after_utc_qa_passed=all(w['utc_interval_satisfies_header'] for w in waits),downloaded=24,original_sha1_checks=24,legacy_crop_equal_checks=24,
        request_count=len(req),retry_after_waits=waits,old_pixels=90,old_pixel_gaps=11,comparisons=2436,near_duplicate_flags=len(flags),
        minimum_request_start_interval=min(normal_intervals),standalone_full_orientation_corrections=full_revisions,
        actual_format_and_original_dimension_checks=24,request_timeout_seconds=50,plan_timeout_seconds=60,
        unsuccessful_response_body_archived=False,
        timing_deviation='Both source sleep requests were600seconds; second recorded UTC interval599.990350seconds is9.65ms short. Original strict offline check failed and is preserved; no claim that the unrecorded monotonic duration met the limit. No more network requests made to fix this record. Future collector should record monotonic endpoints and use a small positive margin.',
        source_sha256=sha(__file__),collector_source_unchanged=True,
        deviation='The frozen collector assumed all101 historical candidate records had downloaded pixels. In fact90 succeeded,9 failed,2 excluded for license. Its post-download loop failed; this offline-only finish preserves the original code and downloads and records missing pixels instead of falsely claiming101 comparisons. No model scores, new source requests or threshold changes.'))
    print(json.dumps(dict(downloaded=24,old_available_pixels=90,flags=len(flags),next_review_images=14)))

if __name__=='__main__':main()

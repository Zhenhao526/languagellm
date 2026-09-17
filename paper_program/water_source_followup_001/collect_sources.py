"""Three fixed text-only source requests; never fetch page subresources."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
from email.utils import parsedate_to_datetime
from urllib.parse import urlencode
import urllib.request,urllib.error,json,hashlib,socket,time,os,sys,math
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parents[1]
V2=PROJECT/'paper_program/visual_confirmation_v2_water'
OUT=ROOT/'source_001';ORDINALS=(8,19,22,42,46,51,62,70)
LIMIT=8*1024*1024
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,obj):
    p=Path(p);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');t.replace(p)
def now():return datetime.now(timezone.utc)
def stamp():return dict(utc=now().isoformat(),monotonic=time.monotonic())
def append(p,obj):
    with p.open('a') as f:f.write(json.dumps(obj,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
def wait(reason,end,seconds):
    target_m=end['monotonic']+seconds;target_u=datetime.fromisoformat(end['utc'])+timedelta(seconds=seconds)
    record=dict(reason=reason,start=stamp(),deadline_utc=target_u.isoformat(),deadline_monotonic=target_m)
    write(OUT/'live_status.json',dict(status='waiting',reason=reason,deadline_utc=target_u.isoformat()))
    while True:
        remaining=max(target_m-time.monotonic(),(target_u-now()).total_seconds())
        if remaining<=0:break
        time.sleep(min(15,remaining))
    record['end']=stamp();record['utc_pass']=now()>=target_u;record['monotonic_pass']=time.monotonic()>=target_m
    append(OUT/'waits.jsonl',record)
class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,req,fp,code,msg,headers,newurl):return None
def request(row,attempt):
    data=bytearray();response=None
    record=dict(logical_id=row['id'],url=row['url'],attempt=attempt,start=stamp(),http_status=None,headers={},error=None,complete=False,limit=LIMIT)
    try:
        try:response=urllib.request.build_opener(NoRedirect).open(urllib.request.Request(row['url'],headers={'User-Agent':'LanguageEmergenceSourceReview/1.0 (local academic data curation; text-only, serialized requests)'}),timeout=60)
        except urllib.error.HTTPError as e:response=e
        record.update(http_status=response.code,headers=dict(response.headers.items()))
        content_len=response.headers.get('Content-Length');content_len=int(content_len) if content_len and content_len.isdigit() else None
        while len(data)<LIMIT:
            chunk=response.read(min(65536,LIMIT-len(data)))
            if not chunk:record['complete']=True;break
            data.extend(chunk)
        if content_len==len(data):record['complete']=True
    except (urllib.error.URLError,TimeoutError,socket.timeout,ConnectionError,OSError) as e:record['error']=str(e);record['transient']=True
    finally:
        if response is not None:response.close()
    body=bytes(data);p=OUT/'raw'/f'{row["id"]}_a{attempt}.response';p.write_bytes(body)
    record.update(end=stamp(),raw_path=str(p),sha256=sha(p),bytes=len(body))
    h={k.lower():v for k,v in record['headers'].items()}
    denial=any(s in body[:262144].lower() for s in (b'access denied',b'robot verification',b'cf-chl-'))
    record['explicit_refusal']=record['http_status'] in (401,403) or ('text/html' in h.get('content-type','') and denial)
    append(OUT/'requests.jsonl',record);return record
def retry_delay(record):
    h={k.lower():v for k,v in record['headers'].items()};value=h.get('retry-after')
    if value is None:return 601.
    try:
        n=float(value)
        if not math.isfinite(n) or n<0:raise ValueError('invalid')
        return n+1
    except ValueError:
        date=parsedate_to_datetime(value)
        if date.tzinfo is None:date=date.replace(tzinfo=timezone.utc)
        return max(0.,(date-datetime.fromisoformat(record['end']['utc'])).total_seconds())+1
def main():
    assert not OUT.exists(),'No overwrite or automatic restart'
    packet=read(PROJECT/'paper_program/confirmation_readiness_20260916/水来源核查简表.json')
    rows=[r for r in packet['rows'] if r['ordinal'] in ORDINALS];assert [r['ordinal'] for r in rows]==list(ORDINALS)
    users=['User:Benjamín Núñez González','User:Victor Blacus','User:Tamorlan','User:WalkingRadiance','User:MF-Warburg','User:Јана Смилеска','User:Ghostis']
    endpoint='https://commons.wikimedia.org/w/api.php'
    common=dict(action='query',format='json',formatversion='2',maxlag='5')
    queries=[dict(id='commons_files',url=endpoint+'?'+urlencode(dict(common,pageids='|'.join(str(r['pageid']) for r in rows),prop='info|revisions|imageinfo',rvprop='ids|timestamp|content',rvslots='main',iiprop='url|sha1|mime|size|extmetadata|user|timestamp',iiextmetadatalanguage='en',curtimestamp='1'))),
             dict(id='commons_authors',url=endpoint+'?'+urlencode(dict(common,titles='|'.join(users),redirects='1',prop='info',curtimestamp='1'))),
             dict(id='flickr_original',url='https://www.flickr.com/photos/dmatos/1081916080/')]
    inputs=[Path(__file__),ROOT/'来源核查固定方案.md',V2/'manifest.json',V2/'author_groups.json',V2/'exclusion_registry.json',PROJECT/'paper_program/confirmation_readiness_20260916/水来源核查简表.json']
    hashes={str(p):sha(p) for p in inputs}
    OUT.mkdir();(OUT/'raw').mkdir()
    write(OUT/'selection.json',dict(rows=rows,workflow_ordinals=list(ORDINALS[:4]),pixel_reserved_ordinals=list(ORDINALS[4:]),queries=queries))
    write(OUT/'freeze.json',dict(created_utc=now().isoformat(),before_first_request=True,input_hashes=hashes,selection_sha256=sha(OUT/'selection.json'),logical_requests=3,http_cap=6,pixel_requests=0,model_calls=0))
    write(OUT/'process_handle.json',dict(pid=os.getpid(),started_utc=now().isoformat(),command='collect_sources.py',source_sha256=sha(__file__)))
    results=[];prev=None;stop=False
    for query in queries:
        for attempt in (1,2):
            if prev:wait('serial_interval',prev['end'],10)
            write(OUT/'live_status.json',dict(status='requesting',logical_id=query['id'],attempt=attempt))
            rec=request(query,attempt);prev=rec;code=rec['http_status'];results.append(rec)
            print(json.dumps(dict(request=query['id'],attempt=attempt,status=code,bytes=rec['bytes'],complete=rec['complete'])),flush=True)
            if rec['explicit_refusal'] or (code is not None and 300<=code<400):stop=True;break
            if code==200 and rec['complete'] and not rec['error']:break
            if code in (429,503):
                if attempt==2:stop=True;break
                try:seconds=retry_delay(rec)
                except (ValueError,TypeError,OverflowError):stop=True;break
                wait('retry_after',rec['end'],seconds);continue
            if attempt==1 and (rec.get('transient') or (code is not None and code>=500)):
                wait('transient',rec['end'],10);continue
            break
        if stop:break
    assert len(results)<=6 and all(sha(p)==h for p,h in hashes.items())
    terminal=dict(status='stopped_access_or_rate_limit' if stop else 'text_collection_complete',ended_utc=now().isoformat(),requests=len(results),logical_attempted=len({r['logical_id'] for r in results}),response_hashes={r['raw_path']:r['sha256'] for r in results},source_judgment='pending',pixel_requests=0,model_calls=0,inputs_unchanged=True)
    write(OUT/'completion.json',terminal);write(OUT/'live_status.json',terminal);print(json.dumps(terminal),flush=True)
if __name__=='__main__':main()

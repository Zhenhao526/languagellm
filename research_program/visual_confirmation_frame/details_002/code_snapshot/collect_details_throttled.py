"""Explicitly authorized new metadata round; prepare makes no network request."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
from email.utils import parsedate_to_datetime
from urllib.parse import urlencode,urlparse
from urllib.request import Request,urlopen
from urllib.error import HTTPError
import argparse
import hashlib
import importlib.util
import json
import shutil
import time
import traceback

ROOT=Path(__file__).resolve().parent
FIRST=ROOT/'details_001'
ENDPOINT='https://commons.wikimedia.org/w/api.php'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def now():return datetime.now(timezone.utc).isoformat()
def write(path,data):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False)
def require(value,message):
    if not value:raise AssertionError(message)


def prepare(out):
    out=Path(out).resolve();require(not out.exists(),'Never overwrite an existing preparation')
    first=read(FIRST/'result.json');records=read(FIRST/'records.json');old=read(FIRST/'old_index_evidence.json')
    require(first['status']=='partial' and first['resolved']==325 and first['unresolved']==1551,'Fixed source partial result')
    require(sha(FIRST/'records.json')==first['records_sha256'],'First record hash')
    pending=sorted(r['pageid'] for r in records if r['status']!='resolved')
    require(len(pending)==len(set(pending))==1551 and pending==first['unresolved_pageids'],'Fixed unknown list')
    sources={str(p.resolve()):sha(p) for p in (Path(__file__),ROOT/'metadata_detail_followup_plan.md',
      FIRST/'result.json',FIRST/'records.json',FIRST/'old_index_evidence.json',FIRST/'invocation.json',FIRST/'request_frame.json',
      FIRST/'failure.json',FIRST/'rate_limit_pause.json',FIRST/'rate_limit_termination.json',FIRST/'collect_details.py')}
    for path,digest in first['source_files_sha256'].items():require(sha(path)==digest,'First source changed');sources[path]=digest
    for path,digest in first['response_and_receipt_sha256'].items():require(sha(path)==digest,'First response changed');sources[path]=digest
    constraints=[];last=None
    for path in sorted((FIRST/'responses').glob('*_receipt.json')):
        r=read(path)
        if r.get('http_status')!=429:continue
        ended=datetime.fromisoformat(r['completed_utc']);last=max(last,ended) if last else ended
        delay=r.get('response_headers',{}).get('retry-after')
        deadline=ended+timedelta(seconds=int(delay)) if str(delay).isdigit() else parsedate_to_datetime(delay)
        constraints.append(dict(batch=r['batch'],completed_utc=ended.isoformat(),retry_after=delay,deadline_utc=deadline.isoformat()))
    require(last is not None,'Expected recorded rate limit evidence')
    not_before=max([last+timedelta(seconds=60)]+[datetime.fromisoformat(x['deadline_utc']) for x in constraints])
    fields=('Artist','Credit','Source','LicenseShortName','LicenseUrl','UsageTerms','AttributionRequired','Copyrighted',
            'Restrictions','ImageDescription','DateTimeOriginal','ObjectName','Categories')
    batches=[]
    for i in range(0,len(pending),25):
        ids=pending[i:i+25]
        params=dict(action='query',format='json',formatversion=2,prop='imageinfo',pageids='|'.join(map(str,ids)),
          iilimit=1,iiprop='url|size|mime|sha1|extmetadata|user|timestamp|canonicaltitle',iiextmetadatalanguage='en',
          iiextmetadatafilter='|'.join(fields))
        batches.append(dict(batch=i//25,pageids=ids,url=ENDPOINT+'?'+urlencode(params)))
    require(len(batches)==63 and len({b['url'] for b in batches})==63,'Unique bounded batch URLs')
    plan=dict(status='prepared_not_executed',created_utc=now(),source_partial_result=str(FIRST/'result.json'),
      source_files_sha256=sources,unresolved_pageids=pending,batches=batches,max_requests=63,batch_size_max=25,
      request_timeout_seconds=20,minimum_delay_after_request_completion_seconds=5,
      retry_after_constraints=constraints,not_before_utc=not_before.isoformat(),
      stop_entire_round_on_any_request_error=True,automatic_retry=False,resume=False,
      source_resolved_preserved=325,source_full_frame=1876,pixel_downloads=0,model_calls=0,
      scope='New one-attempt completion of prior unknown metadata only; independent output, no third round.')
    out.mkdir(parents=True,exist_ok=False);(out/'code_snapshot').mkdir()
    for path in (Path(__file__),ROOT/'metadata_detail_followup_plan.md',FIRST/'collect_details.py'):
        shutil.copyfile(path,out/'code_snapshot'/path.name)
    write(out/'plan.json',plan);write(out/'freeze.json',dict(plan_sha256=sha(out/'plan.json'),status='prepared_not_executed'))
    return dict(status='prepared_not_executed',output=str(out),unknown_pageids=1551,max_requests=63,
                source_sha256=sha(__file__),plan_sha256=sha(out/'plan.json'),not_before_utc=plan['not_before_utc'])


def verify(out):
    out=Path(out).resolve();plan=read(out/'plan.json')
    require(sha(out/'plan.json')==read(out/'freeze.json')['plan_sha256'],'Preparation changed')
    for path,digest in plan['source_files_sha256'].items():require(sha(path)==digest,'Frozen source changed: '+path)
    for path in (Path(__file__),ROOT/'metadata_detail_followup_plan.md',FIRST/'collect_details.py'):
        require(sha(out/'code_snapshot'/path.name)==sha(path),'Snapshot changed')
    return plan


def merged_record(base,page,batch,parsers,old):
    infos=page.get('imageinfo',[]);info=infos[0] if len(infos)==1 else {};digest=parsers.original_sha(info.get('sha1'))
    if not(info and digest and info.get('url') and info.get('mime')):
        return dict(base,unknown_reason='second_round_returned_missing_imageinfo',metadata_round='details_002',api_batch=batch)
    title=page.get('title',base['title']);authors=parsers.author_keys(info);sources=parsers.source_keys(info)
    return dict(base,status='resolved',unknown_reason=None,metadata_round='details_002',api_batch=batch,current_title=title,
      original_file_sha1=digest,old_title_exact_match=any(t in old['exact_titles'] for t in (base['title'],title)),
      old_title_normalized_match=any(parsers.norm(t) in old['normalized_titles'] for t in (base['title'],title)),
      old_sha1_match=digest in old['original_file_sha1s'],author_keys=authors,source_keys=sources,
      old_author_key_match=bool(set(authors)&set(old['author_keys'])),old_source_key_match=bool(set(sources)&set(old['source_keys'])),
      artist_raw=parsers.extract(info,'Artist'),artist_normalized_text=parsers.norm(parsers.plain(parsers.extract(info,'Artist'))),
      upload_user=info.get('user'),license_short_name=parsers.plain(parsers.extract(info,'LicenseShortName')),
      license_url=parsers.extract(info,'LicenseUrl'),license_pattern_supported=parsers.allowed_license(info),
      mime=info.get('mime'),width=info.get('width'),height=info.get('height'),size=info.get('size'),original_url=info.get('url'),
      description_url=info.get('descriptionurl'),timestamp=info.get('timestamp'),extmetadata=info.get('extmetadata',{}),
      page_flags={k:v for k,v in page.items() if k!='imageinfo'})


def execute(out):
    out=Path(out).resolve();plan=verify(out)
    require(datetime.now(timezone.utc)>=datetime.fromisoformat(plan['not_before_utc']),'Respect all Retry-After and 60-second lower bound')
    directory=out/'execution';directory.mkdir(exist_ok=False);(directory/'responses').mkdir()
    spec=importlib.util.spec_from_file_location('frozen_metadata_rules',out/'code_snapshot/collect_details.py')
    parsers=importlib.util.module_from_spec(spec);spec.loader.exec_module(parsers)
    old=read(FIRST/'old_index_evidence.json');records={r['pageid']:r for r in read(FIRST/'records.json')}
    write(directory/'started.json',dict(started_utc=now(),plan_sha256=sha(out/'plan.json'),new_network_round=True))
    previous_completed=None;receipts=[];stopped=None;context={}
    try:
        for job in plan['batches']:
            if previous_completed is not None:
                wait=5-(time.monotonic()-previous_completed)
                if wait>0:time.sleep(wait)
            context=job;start=time.monotonic()
            receipt=dict(job,requested_utc=now(),gap_after_previous_completion_seconds=None if previous_completed is None else start-previous_completed)
            write(directory/'responses'/f'batch_{job["batch"]:03d}_started.json',receipt)
            raw=directory/'responses'/f'batch_{job["batch"]:03d}.bin'
            try:
                with urlopen(Request(job['url'],headers={'User-Agent':'LanguageFormationResearch/1.0 (metadata-only local research)','Accept':'application/json'}),timeout=20) as response:
                    require(urlparse(response.geturl()).netloc=='commons.wikimedia.org','Unexpected endpoint redirect')
                    receipt.update(http_status=response.status,final_url=response.geturl(),response_headers=dict(response.headers));blob=response.read()
                with raw.open('xb') as f:f.write(blob)
                receipt.update(response_file=str(raw),response_sha256=sha(raw),response_bytes=len(blob))
                data=json.loads(blob);require('error' not in data,'API error: '+json.dumps(data.get('error')))
                pages=data['query']['pages'];ids={p.get('pageid') for p in pages}
                require(len(pages)==len(ids) and ids==set(job['pageids']),'Returned page identities incomplete or duplicated')
                for page in pages:records[page['pageid']]=merged_record(records[page['pageid']],page,job['batch'],parsers,old)
                receipt.update(status='received',returned_pages=len(pages),warnings=data.get('warnings'),
                    continuation_not_followed_latest_revision_only=data.get('continue'))
            except Exception as error:
                receipt.update(status='failed',error=repr(error),traceback=traceback.format_exc());stopped=dict(batch=job['batch'],error=repr(error))
                if isinstance(error,HTTPError):
                    receipt.update(http_status=error.code,response_headers=dict(error.headers));blob=error.read()
                    if not raw.exists():
                        with raw.open('xb') as f:f.write(blob)
                    receipt.update(response_file=str(raw),response_sha256=sha(raw),response_bytes=len(blob))
                for pid in job['pageids']:
                    if records[pid]['status']!='resolved':records[pid]=dict(records[pid],unknown_reason='second_round_request_failed')
            previous_completed=time.monotonic()
            receipt.update(completed_utc=now(),elapsed_seconds=previous_completed-start)
            write(directory/'responses'/f'batch_{job["batch"]:03d}_receipt.json',receipt);receipts.append(receipt)
            print(json.dumps({k:receipt[k] for k in ('batch','status','elapsed_seconds')},ensure_ascii=False),flush=True)
            if stopped:break
        full=[records[k] for k in sorted(records)];summary=parsers.summarize(full,old)
        write(directory/'records.json',full);write(directory/'feasibility.json',summary)
        verify(out)
        unknown=[r['pageid'] for r in full if r['status']!='resolved']
        result=dict(status='complete' if not unknown and not stopped else 'partial',completed_utc=now(),
          plan_sha256=sha(out/'plan.json'),request_attempts=len(receipts),requests_planned=63,
          successful_requests=sum(r['status']=='received' for r in receipts),stopped_on_error=stopped,
          source_resolved_preserved=325,new_resolved=1876-len(unknown)-325,total_resolved=1876-len(unknown),
          unresolved_pageids=unknown,unresolved=len(unknown),full_frame=1876,
          records_sha256=sha(directory/'records.json'),feasibility_sha256=sha(directory/'feasibility.json'),
          minimum_observed_gap_seconds=min((r['gap_after_previous_completion_seconds'] for r in receipts[1:]),default=None),
          pixel_downloads=0,model_calls=0,automatic_retry=False,third_round_started=False)
        write(directory/'result.json',result);return result
    except BaseException as error:
        write(directory/'failure.json',dict(status='failed',context=context,error=repr(error),traceback=traceback.format_exc(),automatic_retry=False))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('prepare','verify','execute'))
    parser.add_argument('--out',type=Path,default=ROOT/'details_002');args=parser.parse_args()
    result=prepare(args.out) if args.command=='prepare' else execute(args.out) if args.command=='execute' else verify(args.out)
    print(json.dumps(result,ensure_ascii=False,indent=2))

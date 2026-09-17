"""Recover partial metadata from saved response bytes only; no requests/retries."""
from pathlib import Path
import importlib.util
import json
import hashlib
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'details_001'
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())
def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)


def main():
    spec=importlib.util.spec_from_file_location('frozen_metadata_parsers',OUT/'collect_details.py')
    parsers=importlib.util.module_from_spec(spec);spec.loader.exec_module(parsers)
    def forbidden(*args,**kwargs): raise AssertionError('Offline recovery cannot request a URL')
    parsers.urlopen=forbidden
    invocation=read(OUT/'invocation.json')
    assert all(sha(p)==h for p,h in invocation['source_files_sha256'].items())
    assert sha(OUT/'request_frame.json')==invocation['frame_sha256']
    assert sha(OUT/'old_index_evidence.json')==invocation['old_index_sha256']
    frame=read(OUT/'request_frame.json');old=read(OUT/'old_index_evidence.json')
    receipts={}
    record_map={}
    files_hashes={}
    for path in sorted((OUT/'responses').glob('*_receipt.json')):
        receipt=read(path);receipts[receipt['batch']]=receipt;files_hashes[str(path)]=sha(path)
        if receipt.get('response_file'):
            raw=Path(receipt['response_file']);assert sha(raw)==receipt['response_sha256'];files_hashes[str(raw)]=sha(raw)
        if receipt['status']!='received':continue
        data=json.loads(Path(receipt['response_file']).read_bytes())
        for page in data['query']['pages']:
            assert page['pageid'] not in record_map
            record_map[page['pageid']]=page
    failure=read(OUT/'failure.json');interrupted_batch=failure['context']['batch']
    output=[]
    for row in frame['files']:
        batch=next(job['batch'] for job in frame['batches'] if row['pageid'] in job['pageids'])
        page=record_map.get(row['pageid'],{});infos=page.get('imageinfo',[]);info=infos[0] if len(infos)==1 else {}
        current_title=page.get('title',row['title']);digest=parsers.original_sha(info.get('sha1'))
        authors=parsers.author_keys(info);sources=parsers.source_keys(info)
        resolved=bool(info and digest and info.get('url') and info.get('mime'))
        reason=None if resolved else ('interrupted_request_no_response' if batch==interrupted_batch else
            'not_requested' if batch not in receipts else 'http_429' if receipts[batch].get('http_status')==429 else
            'other_request_error' if receipts[batch]['status']!='received' else 'returned_missing_imageinfo')
        output.append(dict(row,status='resolved' if resolved else 'unresolved',unknown_reason=reason,current_title=current_title,
            api_batch=batch,original_file_sha1=digest,
            old_title_exact_match=any(t in old['exact_titles'] for t in (row['title'],current_title)),
            old_title_normalized_match=any(parsers.norm(t) in old['normalized_titles'] for t in (row['title'],current_title)),
            old_sha1_match=bool(digest and digest in old['original_file_sha1s']),
            old_author_key_match=bool(set(authors)&set(old['author_keys'])),old_source_key_match=bool(set(sources)&set(old['source_keys'])),
            author_keys=authors,source_keys=sources,artist_raw=parsers.extract(info,'Artist'),artist_normalized_text=parsers.norm(parsers.plain(parsers.extract(info,'Artist'))),
            upload_user=info.get('user'),license_short_name=parsers.plain(parsers.extract(info,'LicenseShortName')),
            license_url=parsers.extract(info,'LicenseUrl'),license_pattern_supported=parsers.allowed_license(info),
            mime=info.get('mime'),width=info.get('width'),height=info.get('height'),size=info.get('size'),
            original_url=info.get('url'),description_url=info.get('descriptionurl'),timestamp=info.get('timestamp'),
            extmetadata=info.get('extmetadata',{}),page_flags={k:v for k,v in page.items() if k!='imageinfo'}))
    summary=parsers.summarize(output,old)
    write(OUT/'records.json',output);write(OUT/'feasibility.json',summary)
    unresolved=[r['pageid'] for r in output if r['status']!='resolved']
    unknown_counts=dict(parsers.Counter(r['unknown_reason'] for r in output if r['status']!='resolved'))
    result=dict(status='partial',offline_recovery=True,recovered_utc=datetime.now(timezone.utc).isoformat(),
        original_collector_status='interrupted_after_rate_limit',original_failure_sha256=sha(OUT/'failure.json'),
        recovery_script_sha256=sha(__file__),request_receipts=len(receipts),successful_requests=sum(r['status']=='received' for r in receipts.values()),
        failed_http429_requests=sum(r.get('http_status')==429 for r in receipts.values()),interrupted_batch=interrupted_batch,
        requests_planned=len(frame['batches']),unique_files=len(output),resolved=len(output)-len(unresolved),unresolved=len(unresolved),
        unknown_reason_counts=unknown_counts,unresolved_pageids=unresolved,first_retry_after_seconds=37,
        response_and_receipt_sha256=files_hashes,source_files_sha256=invocation['source_files_sha256'],
        records_sha256=sha(OUT/'records.json'),feasibility_sha256=sha(OUT/'feasibility.json'),
        pixel_downloads=0,model_calls=0,recovery_network_calls=0,automatic_retry=False,
        scope='Incomplete ascending-pageid metadata frame. Unknown files are not rejected or unavailable; no population sufficiency inference.')
    write(OUT/'result.json',result)
    followup=dict(status='proposed_not_executed',source_partial_result_sha256=sha(OUT/'result.json'),
        pageids=unresolved,unique_pageids=len(unresolved),new_output_required=True,suggested_output='details_002',
        batch_size=25,requests_max=(len(unresolved)+24)//25,timeout_seconds=20,
        minimum_delay_after_each_response_seconds=5,wait_before_start_seconds='At least every still-active Retry-After deadline; also minimum 60 seconds since last request.',
        stop_immediately_on_http429=True,stop_on_other_network_or_api_error=True,
        failed_current_batch_not_retried=True,each_unresolved_pageid_at_most_one_new_attempt=True,
        preserve_prior_successes_without_refetch=True,automatic_retry=False,network_calls_started=0,
        note='A new explicitly authorized completion attempt includes previous HTTP429 and unrequested IDs; not an automatic retry of details_001.')
    write(OUT/'proposed_throttled_completion_plan.json',followup)
    print(json.dumps({k:result[k] for k in ('status','request_receipts','successful_requests','failed_http429_requests','unique_files','resolved','unresolved','unknown_reason_counts')},ensure_ascii=False))
    print(json.dumps({name:{k:v[k] for k in ('listed','resolved','unresolved','fresh_files','metadata_license_pattern_supported','supported_with_author_field')}|
           {'component_count':v['metadata_components']['count'],'new_author_component_count':v['excluding_old_author_or_source_components']['count']}
           for name,v in summary.items()},ensure_ascii=False))


if __name__=='__main__':main()

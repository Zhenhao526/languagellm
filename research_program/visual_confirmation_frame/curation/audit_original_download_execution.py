"""Read-only execution audit: hashes bytes without decoding any image.

No network, image, ML, or acquisition imports; writes only new audit outputs.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = HERE / 'water_download_001'

def read(path): return json.loads(Path(path).read_text())
def hashes(path):
    h1, h256, size = hashlib.sha1(), hashlib.sha256(), 0
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h1.update(block); h256.update(block); size += len(block)
    return {'bytes': size, 'sha1': h1.hexdigest(), 'sha256': h256.hexdigest()}
def sha(path): return hashes(path)['sha256']
def utc(value): return datetime.fromisoformat(value)

def main():
    plan = read(RUN/'plan.json'); result = read(RUN/'execution/result.json')
    freeze = read(RUN/'freeze.json'); started = read(RUN/'execution/started.json')
    assert freeze['plan_sha256'] == started['plan_sha256'] == sha(RUN/'plan.json')
    assert plan['stage'] == 'water' and plan['fixed_download_order'] == ['water']
    assert len(plan['jobs']) == plan['original_get_limit'] == result['planned_requests'] == 106
    assert plan['retries'] == 0 and plan['redirects'] is False and plan['stop_round_on_request_error'] is True
    assert plan['timeout_seconds'] == 20 and plan['delay_after_completion_seconds'] == 5
    assert plan['file_byte_limit'] == 16*1024**2 and plan['full_pool_byte_limit'] == 2*1024**3
    for path, expected in plan['source_sha256'].items(): assert sha(path) == expected
    for snapshot, current in [('download_originals.py','download_originals.py'),
                              ('curation_plan.md','plan.md'),('test_download_originals.py','test_download_originals.py')]:
        assert sha(RUN/snapshot) == sha(HERE/current)
    rdir = RUN/'execution/receipts'
    receipt_files = sorted(rdir.glob('*_receipt.json')); start_files = sorted(rdir.glob('*_started.json'))
    assert [p.name for p in receipt_files] == [f'{i:03d}_receipt.json' for i in range(11)]
    assert [p.name for p in start_files] == [f'{i:03d}_started.json' for i in range(11)]
    receipts = [read(p) for p in receipt_files]; total = 0; output=[]; gaps=[]; mono=[]; original_files=[]
    for i, receipt in enumerate(receipts):
        job=plan['jobs'][i]; beginning=read(start_files[i])
        assert receipt['request_index'] == job['request_index'] == i
        for key in ('pageid','review_id'): assert receipt[key] == job[key]
        assert receipt['url'] == job['original_url'] and receipt['requested_utc'] == beginning['requested_utc']
        assert all(receipt[k] == v for k,v in beginning.items())
        assert utc(receipt['completed_utc']) >= utc(receipt['requested_utc']) >= utc(started['started_utc'])
        elapsed=(utc(receipt['completed_utc'])-utc(receipt['requested_utc'])).total_seconds()
        assert abs(elapsed-receipt['elapsed_seconds']) < 0.001
        if i:
            gap=(utc(receipt['requested_utc'])-utc(receipts[i-1]['completed_utc'])).total_seconds()
            assert gap >= 5 and receipt['gap_after_previous_completion_seconds'] >= 5
            assert abs(gap-receipt['gap_after_previous_completion_seconds']) < 0.001
            gaps.append(gap);mono.append(receipt['gap_after_previous_completion_seconds'])
        else: assert receipt['gap_after_previous_completion_seconds'] is None
        headers={k.lower():v for k,v in receipt['response_headers'].items()}
        row={'request_index':i,'pageid':job['pageid'],'review_id':job['review_id'],
             'url':job['original_url'],'status':receipt['status'],'http_status':receipt['http_status'],
             'started_utc':receipt['requested_utc'],'completed_utc':receipt['completed_utc']}
        if i<10:
            assert receipt['status']=='downloaded' and receipt['http_status']==200
            assert receipt['final_url']==job['original_url']
            raw=Path(receipt['file']); assert raw.resolve()==(RUN/'execution/originals'/(job['review_id']+'.bin')).resolve()
            h=hashes(raw); original_files.append(raw.resolve())
            assert h['sha1']==receipt['original_sha1']==job['original_file_sha1']
            assert h['sha256']==receipt['sha256']
            assert h['bytes']==receipt['response_bytes']==job['size']<=plan['file_byte_limit']
            if 'content-length' in headers: assert h['bytes']==int(headers['content-length'])
            total += h['bytes'];row.update(h)
        else:
            assert receipt['status']=='request_failed' and receipt['http_status']==429
            assert headers['retry-after']=='600' and headers['content-length']=='2256'
            raw=Path(receipt['error_body']); assert raw.resolve()==(rdir/'010_error.bin').resolve()
            h=hashes(raw);assert h['bytes']==receipt['error_body_bytes']==2256
            assert h['sha256']==receipt['error_body_sha256']
            assert 'file' not in receipt and 'partial_file' not in receipt
            total += h['bytes'];row.update(error_body_bytes=h['bytes'],error_body_sha256=h['sha256'],retry_after_seconds=600)
        assert receipt['cumulative_response_bytes']==total<=plan['full_pool_byte_limit']
        output.append(row)
    assert len({r['url'] for r in receipts})==len({r['pageid'] for r in receipts})==11
    assert sorted(original_files)==sorted(p.resolve() for p in (RUN/'execution/originals').iterdir())
    assert {p.name for p in rdir.iterdir()}==({p.name for p in receipt_files+start_files}|{'010_error.bin'})
    assert result['status']=='stopped_on_error' and result['requests']==11 and result['downloaded']==10
    assert result['metadata_mismatch_rejected']==0 and result['stopped']['request_index']==10
    assert result['response_bytes']==total==22402712
    assert result['not_requested_pageids']==[j['pageid'] for j in plan['jobs'][11:]]
    assert len(result['not_requested_pageids'])==95
    assert result['minimum_observed_gap_seconds']==min(mono)
    assert utc(result['completed_utc'])>=utc(receipts[-1]['completed_utc'])
    assert result['visual_acceptance_completed'] is False and result['model_calls']==0
    assert not (RUN/'execution/failure.json').exists()
    record={'status':'passed_read_only_execution_audit','audited_at_utc':datetime.now(timezone.utc).isoformat(),
        'source_sha256':{str(p):sha(p) for p in [Path(__file__),RUN/'plan.json',RUN/'freeze.json',RUN/'execution/result.json',RUN/'download_originals.py']},
        'frozen_source_files_verified':len(plan['source_sha256']),'requests':11,'unique_urls':11,'unique_pageids':11,
        'downloaded_originals':10,'original_sha1s_verified':10,'original_sha256s_verified':10,
        'original_bytes':total-2256,'error_body_bytes':2256,'total_response_bytes':total,
        'final_http_status':429,'retry_after_seconds':600,'not_requested':95,
        'min_gap_utc_seconds':min(gaps),'min_gap_recorded_monotonic_seconds':min(mono),
        'all_ten_request_gaps_at_least_5_seconds':True,'request_after_index10_in_run':False,
        'no_redirect_or_duplicate_request_in_archived_run':True,'receipts':output,
        'audit_network_requests':0,'audit_image_decodes_or_views':0,'audit_model_calls':0,
        'scope':'Local frozen code, plan, receipts and bytes; not an independent network-traffic capture.',
        'interpretation':'10 acquired checksum-matching originals is not visual/source acceptance; interruption is a download-stage failure, not failure of the 106 metadata candidates to meet the 84-material quota.'}
    with (HERE/'原图下载执行核验.json').open('x') as stream:
        json.dump(record,stream,ensure_ascii=False,indent=2);stream.write('\n')
    print(json.dumps({k:v for k,v in record.items() if k not in ['receipts','source_sha256']},ensure_ascii=False,indent=2))

if __name__=='__main__':main()

"""Offline-only thumbnail feasibility and request planning; no image I/O.

No requests, opener, image decoder, torch, or model imports. Does not modify old
files. Output is a review draft, never permission to execute network requests.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
CURATION=HERE.parent
WIDTHS=(960,1280,1920,3840)
LABELS=('apple','banana','orange','water')
TARGETS={'apple':28,'banana':28,'orange':28,'water':84}

def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def choose_width(width,height,steps=WIDTHS,strict=True):
    """Integer lower bound: do not depend on a service's rounding direction."""
    assert type(width) is int and type(height) is int and width>0 and height>0
    for candidate in steps:
        if candidate>width or (strict and candidate==width):continue
        if min(candidate,height*candidate//width)>=512:return candidate
    return None

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=HERE/'offline_001');args=parser.parse_args()
    out=args.out.resolve()
    if out.exists():raise FileExistsError('Do not overwrite an earlier offline draft')
    paths={'records':CURATION.parent/'details_002/execution/records.json',
           'order':CURATION/'ranking_001/candidate_order.json',
           'inventory':CURATION/'ranking_001/candidate_inventory.json',
           'components':CURATION/'ranking_001/global_components.json',
           'ranking_freeze':CURATION/'ranking_001/freeze.json',
           'old_plan':CURATION/'plan.md',
           'acquisition_plan':CURATION/'water_download_001/plan.json',
           'acquisition_result':CURATION/'water_download_001/execution/result.json',
           'acquisition_audit':CURATION/'原图下载执行核验.json',
           'script':Path(__file__).resolve(),'draft':HERE/'plan_draft.md'}
    source_hashes={str(p.resolve()):sha(p) for p in paths.values()}
    for item in read(paths['ranking_freeze'])['source_files']:
        assert sha(item['path'])==item['sha256']
    records=read(paths['records']);byid={r['pageid']:r for r in records};order=read(paths['order'])
    assert len(records)==len(byid)==1876 and set(order)==set(LABELS)
    nominees=[n for label in LABELS for n in order[label]]
    assert len(nominees)==232==len({n['pageid'] for n in nominees})==len({n['component_id'] for n in nominees})
    inventory={r['pageid']:r for r in read(paths['inventory'])}
    legacy=read(paths['acquisition_result']);legacy_audit=read(paths['acquisition_audit'])
    assert legacy['requests']==legacy_audit['requests']==11 and legacy['downloaded']==10
    assert legacy['response_bytes']==legacy_audit['total_response_bytes']==22402712
    assert legacy['status']=='stopped_on_error' and legacy_audit['retry_after_seconds']==600
    cases=[]
    for label in LABELS:
        for n in order[label]:
            r=byid[n['pageid']];assert inventory[n['pageid']]['metadata_eligible'] is True
            for key in ('title','original_file_sha1','width','height','size','mime','author_keys','source_keys'):
                assert r[key]==n[key]
            w,h=r['width'],r['height'];selected=choose_width(w,h)
            variants={name:choose_width(w,h,steps,strict) for name,steps,strict in (
                ('fixed1280_nonupscale',(1280,),False),('fixed1280_strict',(1280,),True),
                ('adaptive_nonupscale',WIDTHS,False),('adaptive_strict',WIDTHS,True))}
            nonupscale=variants['adaptive_nonupscale']
            why=None if selected else ('source_width_below_960' if w<960 else
                'only_equal_width_standard_available' if nonupscale==w else
                'no_standard_width_with_short_side_512_and_strict_downscale')
            cases.append({'legacy_nomination':n,'source_timestamp':r['timestamp'],
                'expected_current_title':r['current_title'],'original_metadata_record_sha256':
                    hashlib.sha256(json.dumps(r,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
                'strategy_comparison':variants,'recommended_width':selected,
                'technical_metadata_eligible':selected is not None,'technical_metadata_exclusion':why,
                'predicted_thumbnail_width':selected,
                'predicted_thumbnail_height_floor':h*selected//w if selected else None,
                'predicted_thumbnail_height_ceil':(h*selected+w-1)//w if selected else None,
                'source_orientation_verified':False,'actual_thumbnail_received':False,
                'visual_acceptance':None,'source_acceptance':None,'material_split':None})
    comparisons={}
    for name in cases[0]['strategy_comparison']:
        comparisons[name]={}
        for label in LABELS:
            selected=[r for r in cases if r['legacy_nomination']['label']==label and r['strategy_comparison'][name] is not None]
            comparisons[name][label]={'count':len(selected),'target':TARGETS[label],
                'original_primary_eligible':sum(r['legacy_nomination']['role']=='primary' for r in selected),
                'original_reserve_eligible':sum(r['legacy_nomination']['role']=='reserve' for r in selected),
                'technical_surplus_over_final_target':len(selected)-TARGETS[label],
                'selected_widths':dict(sorted(Counter(r['strategy_comparison'][name] for r in selected).items()))}
    api_jobs=[]
    for stage in ('water','food'):
        stage_cases=[r for r in cases if r['technical_metadata_eligible'] and
                     (r['legacy_nomination']['label']=='water')==(stage=='water')]
        for width in WIDTHS:
            relevant=[r for r in stage_cases if r['recommended_width']==width]
            for start in range(0,len(relevant),25):
                group=relevant[start:start+25];ids=[r['legacy_nomination']['pageid'] for r in group]
                api_jobs.append({'batch_index':len(api_jobs),'stage':stage,'requested_width':width,'pageids':ids,
                    'endpoint':'https://commons.wikimedia.org/w/api.php',
                    'parameters':{'action':'query','format':'json','formatversion':2,'pageids':'|'.join(map(str,ids)),
                        'prop':'imageinfo','iilimit':1,'iiprop':'canonicaltitle|url|size|sha1|timestamp|mime|thumbmime',
                        'iiurlwidth':width,'maxlag':5},'network_requested':False})
    ids=[pid for j in api_jobs for pid in j['pageids']]
    eligible=[r for r in cases if r['technical_metadata_eligible']]
    assert len(ids)==len(set(ids))==len(eligible)==209
    assert Counter(j['stage'] for j in api_jobs)=={'water':4,'food':6}
    # Pure arithmetic boundary examples; no images involved.
    assert choose_width(960,1280) is None and choose_width(960,1280,strict=False)==960
    assert choose_width(1000,512) is None and choose_width(2000,800)==1280
    assert choose_width(4800,600) is None and choose_width(4000,3000)==960
    counts={'original_candidates':232,'recommended_technical_candidates':len(eligible),
        'recommended_technical_exclusions':232-len(eligible),'comparisons':comparisons,
        'recommended_width_counts':dict(Counter(r['recommended_width'] for r in eligible)),
        'recommended_exclusion_reasons':dict(Counter(r['technical_metadata_exclusion'] for r in cases if not r['technical_metadata_eligible'])),
        'planned_metadata_batches':len(api_jobs),'planned_metadata_batches_by_stage':dict(Counter(j['stage'] for j in api_jobs)),
        'planned_max_new_media_gets':len(eligible),'legacy_media_gets':11,
        'combined_max_media_gets':11+len(eligible),'legacy_response_body_bytes':22402712,
        'media_cumulative_byte_ceiling':2*1024**3,'remaining_media_byte_ceiling':2*1024**3-22402712,
        'note':'Metadata feasibility only; no API verification, pixels, source acceptance or material allocation.'}
    out.mkdir(parents=True)
    write(out/'candidates.json',cases);write(out/'metadata_api_batches.json',api_jobs);write(out/'counts.json',counts)
    manifest={'status':'offline_draft_pending_root_review_not_network_frozen','prepared_utc':datetime.now(timezone.utc).isoformat(),
        'recommended_strategy':'adaptive_strict','source_sha256':source_hashes,
        'output_sha256':{p.name:sha(p) for p in out.iterdir() if p.is_file()},
        'network_coordination_status':'not_cleared_other_user_metadata_task_live_at_assignment',
        'new_network_requests':0,'original_image_bytes_read_by_this_script':0,'pixel_views':0,'model_calls':0,
        'media_urls_resolved':False,'network_executor_implemented':False,'final_material_splits_assigned':False}
    write(out/'manifest.json',manifest)
    assert all(sha(p)==h for p,h in source_hashes.items())
    print(json.dumps(counts,ensure_ascii=False,indent=2))

if __name__=='__main__':main()

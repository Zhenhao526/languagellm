"""Independent offline receipt, identity, exclusion and grouping audit."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
from collections import defaultdict,Counter
from html.parser import HTMLParser
from urllib.parse import urlparse,parse_qs,unquote
import hashlib,html,json,re,unicodedata

OUT=Path(__file__).resolve().parent
ROOT=OUT.parent
FIRST=ROOT/'details_001'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def need(x,message):
    if not x:raise AssertionError(message)
def norm(s):return unicodedata.normalize('NFKC',str(s)).replace('_',' ').strip().casefold()
class HTML(HTMLParser):
    def __init__(self,s):super().__init__(convert_charrefs=True);self.text=[];self.links=[];self.feed(s)
    def handle_data(self,s):self.text.append(s)
    def handle_starttag(self,tag,attrs):
        if tag=='a':self.links.extend(v for k,v in attrs if k=='href' and v)
def text(s):return ' '.join(' '.join(HTML(s).text).split())
def field(info,k):return str(info.get('extmetadata',{}).get(k,{}).get('value',''))
UNKNOWN={'','unknown','unknown author','anonymous','anonymous author','not provided','n/a'}


def metadata_keys(info):
    artist=HTML(field(info,'Artist'));label=norm(text(field(info,'Artist')));author=[];source=[]
    if label not in UNKNOWN:author.append('artist_text:'+label)
    for link in artist.links:
        p=urlparse('https:'+link if link.startswith('//') else link);host=p.netloc.lower();path=unquote(p.path)
        if host in ('commons.wikimedia.org','en.wikipedia.org') and path.lower().startswith(('/wiki/user:','/wiki/creator:')):
            author.append('creator_url:'+host+norm(path))
        words=path.strip('/').split('/')
        if host in ('flickr.com','www.flickr.com') and len(words)>=2 and words[0] in ('photos','people'):
            author.append('creator_flickr:'+norm(words[1]))
    for k in ('Credit','Source'):
        for link in HTML(field(info,k)).links:
            p=urlparse('https:'+link if link.startswith('//') else link);host=p.netloc.lower();path=unquote(p.path);words=path.strip('/').split('/')
            if host in ('flickr.com','www.flickr.com') and len(words)>=3 and words[0]=='photos' and words[2].isdigit():
                source.append('source_flickr_photo:'+words[2])
            if host=='commons.wikimedia.org' and path.startswith('/wiki/File:'):
                source.append('source_commons_file:'+norm(path[6:]))
    label_license=text(field(info,'LicenseShortName')).strip();license_url=html.unescape(field(info,'LicenseUrl')).strip()
    short=bool(re.fullmatch(r'CC BY(?:-SA)? (?:1\.0|2\.0|2\.5|3\.0|4\.0)',label_license,re.I)) or norm(label_license) in ('cc0','cc0 1.0','public domain')
    url_ok=bool(re.match(r'https?://creativecommons\.org/(?:licenses/(?:by|by-sa)/(?:1\.0|2\.0|2\.5|3\.0|4\.0)(?:/|$)|publicdomain/(?:zero|mark)/1\.0(?:/|$))',license_url,re.I))
    return sorted(set(author)),sorted(set(source)),label,short or url_ok


def components(rows):
    """Build key-to-file graph, then traverse connected components (not producer DSU)."""
    keys=defaultdict(list)
    for row in rows:
        for k in row['author_keys']+row['source_keys']+['sha1:'+row['original_file_sha1']]:keys[k].append(row['pageid'])
    neighbors={r['pageid']:set() for r in rows}
    for ids in keys.values():
        for pid in ids:neighbors[pid].update(ids)
    groups=[];seen=set()
    for pid in sorted(neighbors):
        if pid in seen:continue
        stack=[pid];group=set()
        while stack:
            current=stack.pop()
            if current in group:continue
            group.add(current);stack.extend(neighbors[current]-group)
        seen.update(group);groups.append(sorted(group))
    return sorted(groups)


def main():
    hashes={}
    def bind(path,digest=None):
        path=Path(path).resolve();actual=sha(path)
        if digest is not None:need(actual==digest,'SHA mismatch '+str(path))
        hashes[str(path)]=actual
    plan=read(OUT/'plan.json');result=read(OUT/'execution/result.json')
    need(result['status']=='complete' and result['unresolved']==0 and result['unresolved_pageids']==[] and
         result['request_attempts']==result['successful_requests']==63 and result['new_resolved']==1551 and result['total_resolved']==1876,
         'Completed coverage/budget')
    need(result['stopped_on_error'] is None and result['automatic_retry'] is False and result['third_round_started'] is False,
         'No retries or stopped round')
    bind(OUT/'plan.json',result['plan_sha256']);bind(OUT/'freeze.json')
    need(read(OUT/'freeze.json')['plan_sha256']==sha(OUT/'plan.json'),'Freeze consistency')
    for path,digest in plan['source_files_sha256'].items():bind(path,digest)
    for name in ('collect_details_throttled.py','metadata_detail_followup_plan.md','collect_details.py'):
        source=ROOT/name if name!='collect_details.py' else FIRST/name
        bind(OUT/'code_snapshot'/name,sha(source))
    bind(OUT/'execution/result.json');bind(OUT/'execution/records.json',result['records_sha256'])
    bind(OUT/'execution/feasibility.json',result['feasibility_sha256']);bind(OUT/'execution/started.json')
    old_records=read(FIRST/'records.json');merged=read(OUT/'execution/records.json');old=read(FIRST/'old_index_evidence.json')
    byid={r['pageid']:r for r in merged};base={r['pageid']:r for r in old_records}
    need(len(merged)==len(byid)==len(base)==1876 and set(byid)==set(base),'Same complete pageid frame')
    preserved={r['pageid'] for r in old_records if r['status']=='resolved'}
    pending=set(plan['unresolved_pageids'])
    need(len(preserved)==325 and len(pending)==1551 and not preserved&pending and preserved|pending==set(byid),'First/new partition')
    need(all(base[pid]==byid[pid] for pid in preserved),'First 325 records changed')
    need(all(r['status']=='resolved' for r in merged),'All resolved')
    raw_pages={};gaps=[];utc_gaps=[];last_end=None;new_ids=[];requests=[]
    for i,job in enumerate(plan['batches']):
        need(job['batch']==i and len(job['pageids'])<=25,'Batch order/size')
        folder=OUT/'execution/responses';start=read(folder/f'batch_{i:03d}_started.json');r=read(folder/f'batch_{i:03d}_receipt.json')
        for path in (folder/f'batch_{i:03d}_started.json',folder/f'batch_{i:03d}_receipt.json'):bind(path)
        need(all(r[k]==start[k]==job[k] for k in ('batch','pageids','url')) and r['requested_utc']==start['requested_utc'],'Start/end receipt consistency')
        need(r['status']=='received' and r['http_status']==200 and r['final_url']==job['url'],'Successful official response')
        url=urlparse(r['url']);query=parse_qs(url.query)
        need(url.scheme=='https' and url.netloc=='commons.wikimedia.org' and url.path=='/w/api.php' and
             query['prop']==['imageinfo'] and query['iilimit']==['1'] and 'iiurlwidth' not in query,
             'Metadata-only request endpoint')
        need(list(map(int,query['pageids'][0].split('|')))==job['pageids'],'Actual pageid request')
        requested=datetime.fromisoformat(r['requested_utc']);completed=datetime.fromisoformat(r['completed_utc'])
        need(completed>=requested and r['elapsed_seconds']>=0,'Receipt timing')
        if last_end is not None:
            utc_gap=(requested-last_end).total_seconds();utc_gaps.append(utc_gap);gaps.append(r['gap_after_previous_completion_seconds'])
            need(utc_gap>=5 and r['gap_after_previous_completion_seconds']>=5,'Minimum 5-second interval')
        else:need(requested>=datetime.fromisoformat(plan['not_before_utc']),'Start cooldown')
        last_end=completed
        bind(r['response_file'],r['response_sha256']);data=read(r['response_file'])
        need('error' not in data and {p['pageid'] for p in data['query']['pages']}==set(job['pageids']),'Exact raw API page response')
        for page in data['query']['pages']:
            need(page['pageid'] not in raw_pages,'Duplicate new raw page');raw_pages[page['pageid']]=page
        new_ids+=job['pageids'];requests.append(r)
    need(len(requests)==63 and len({r['url'] for r in requests})==63 and len(new_ids)==len(set(new_ids))==1551 and set(new_ids)==pending,
         'No missing/repeated new request')
    need(len(list((OUT/'execution/responses').glob('*.bin')))==len(list((OUT/'execution/responses').glob('*_receipt.json')))==
         len(list((OUT/'execution/responses').glob('*_started.json')))==63,'No extra requests/raw files')
    need(min(gaps)==result['minimum_observed_gap_seconds'],'Reported minimum actual gap')
    old_429=[]
    for path in sorted((FIRST/'responses').glob('*_receipt.json')):
        r=read(path)
        if r.get('http_status')==429:old_429.append(r)
        if r['status']=='received':
            for page in read(r['response_file'])['query']['pages']:
                need(page['pageid'] not in raw_pages,'Old/new page overlap');raw_pages[page['pageid']]=page
    need(set(raw_pages)==set(byid),'All records have one raw successful API source')
    first_new=datetime.fromisoformat(requests[0]['requested_utc'])
    need((first_new-max(datetime.fromisoformat(r['completed_utc']) for r in old_429)).total_seconds()>=60,'60-second cooldown from last429')
    for rule in plan['retry_after_constraints']:need(first_new>=datetime.fromisoformat(rule['deadline_utc']),'Retry-After deadline')
    for pid,row in byid.items():
        page=raw_pages[pid];need(len(page['imageinfo'])==1,'Latest one version');info=page['imageinfo'][0]
        author,source,artist,license_ok=metadata_keys(info)
        need(row['current_title']==page['title'] and row['mime']==info['mime'] and row['original_file_sha1']==info['sha1'] and
             row['width']==info['width'] and row['height']==info['height'] and row['size']==info['size'] and
             row['original_url']==info['url'] and row['extmetadata']==info['extmetadata'],'Raw metadata extraction')
        need(row['author_keys']==author and row['source_keys']==source and row['artist_normalized_text']==artist and
             row['license_pattern_supported'] is license_ok,'Author/source/license parsing')
        need(row['old_title_exact_match']==any(t in old['exact_titles'] for t in (row['title'],page['title'])) and
             row['old_title_normalized_match']==any(norm(t) in old['normalized_titles'] for t in (row['title'],page['title'])) and
             row['old_sha1_match']==(info['sha1'] in old['original_file_sha1s']) and
             row['old_author_key_match']==bool(set(author)&set(old['author_keys'])) and
             row['old_source_key_match']==bool(set(source)&set(old['source_keys'])),'Exclusion flags')
        need(row['title']==base[pid]['title'] and row['categories']==base[pid]['categories'],'Original category identity')
    saved=read(OUT/'execution/feasibility.json');summary={};eligible_by_resource={}
    for category in ('Apples','Bananas','Oranges','food_union','Glasses of water'):
        subset=[r for r in merged if any(c in ('Apples','Bananas','Oranges') for c in r['categories'])] if category=='food_union' else [r for r in merged if category in r['categories']]
        fresh=[r for r in subset if not(r['old_title_normalized_match'] or r['old_sha1_match'])]
        raster=[r for r in fresh if r['mime'] in ('image/jpeg','image/png','image/tiff','image/webp')]
        licensed=[r for r in raster if r['license_pattern_supported']];authored=[r for r in licensed if r['author_keys']]
        new_authors=[r for r in authored if not(r['old_author_key_match'] or r['old_source_key_match'])]
        numbers=dict(listed=len(subset),resolved=len(subset),unresolved=0,
          old_exact_title_matches=sum(r['old_title_exact_match'] for r in subset),old_normalized_title_matches=sum(r['old_title_normalized_match'] for r in subset),
          old_sha1_matches=sum(r['old_sha1_match'] for r in subset),old_sha1_without_old_title=sum(r['old_sha1_match'] and not r['old_title_normalized_match'] for r in subset),
          fresh_files=len(fresh),fresh_distinct_sha1=len({r['original_file_sha1'] for r in fresh}),fresh_static_raster_mime=len(raster),
          metadata_license_pattern_supported=len(licensed),supported_with_author_field=len(authored),supported_without_author_field=len(licensed)-len(authored),
          distinct_artist_text_fields=len({r['artist_normalized_text'] for r in authored if r['artist_normalized_text'] not in UNKNOWN}),
          exclusions_old_author_or_source=len(authored)-len(new_authors))
        need(all(saved[category][k]==v for k,v in numbers.items()),'Independent count mismatch '+category)
        need(saved[category]['license_labels']==dict(Counter(r['license_short_name'] for r in subset)) and
             saved[category]['mime_counts']==dict(Counter(r['mime'] for r in subset)),'Label/MIME histograms')
        for key,rows in (('metadata_components',authored),('excluding_old_author_or_source_components',new_authors)):
            groups=components(rows);need(saved[category][key]['count']==len(groups) and sorted(saved[category][key]['groups'])==groups,'Independent graph components '+category)
            numbers[key+'_count']=len(groups)
        summary[category]=numbers
        if category in ('food_union','Glasses of water'):eligible_by_resource[category]={r['pageid'] for r in new_authors}
    both_ids=set.union(*eligible_by_resource.values());global_groups=components([byid[i] for i in sorted(both_ids)])
    capacities={resource:sum(bool(set(group)&ids) for group in global_groups) for resource,ids in eligible_by_resource.items()}
    # Two-resource cluster allocation is feasible only as metadata arithmetic; no selection is made.
    capacity_possible=all(n>=84 for n in capacities.values()) and len(global_groups)>=168
    shared=[r for r in merged if 'Glasses of water' in r['categories'] and any(c in ('Apples','Bananas','Oranges') for c in r['categories'])]
    need(all(sha(p)==h for p,h in hashes.items()),'Source changed during audit')
    verification=dict(status='passed',checked_utc=datetime.now(timezone.utc).isoformat(),audit_sha256=sha(__file__),
      counts=dict(full_pageids=1876,source_preserved_exact_json=325,new_pageids=1551,new_requests=63,
                  response_metadata_entries=1876,bound_hash_files=len(hashes),old_exact_titles=len(old['exact_titles']),
                  old_normalized_titles=len(old['normalized_titles']),old_original_file_sha1s=len(old['original_file_sha1s'])),
      minimum_monotonic_gap_seconds=min(gaps),minimum_utc_gap_seconds=min(utc_gaps),all_retry_after_deadlines_respected=True,
      raw_response_pageid_coverage_exact=True,extraction_and_exclusion_flags_exact=True,independent_graph_groups_exact=True,
      summary=summary,global_components_after_old_author_exclusion=len(global_groups),global_components_available_by_resource=capacities,
      metadata_only_two_resource_84_each_capacity_possible=capacity_possible,
      cross_resource_category_overlap=[dict(pageid=r['pageid'],title=r['title']) for r in shared],
      source_sha256=hashes,new_network_calls=0,pixel_downloads=0,model_calls=0,
      limitations=['Author text/link keys and source-file SHA1 are metadata evidence, not verified independent photographers/series.',
        'Raster MIME and license-pattern matches are not visual suitability or full rights clearance.',
        'Counts exceeding84 do not constitute completed dataset acceptance; cross-resource cluster allocation is arithmetic only.'])
    with (OUT/'独立核验.json').open('x',encoding='utf-8') as f:json.dump(verification,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({k:verification[k] for k in ('status','counts','minimum_monotonic_gap_seconds','minimum_utc_gap_seconds','global_components_after_old_author_exclusion','global_components_available_by_resource','metadata_only_two_resource_84_each_capacity_possible','cross_resource_category_overlap')},ensure_ascii=False))


if __name__=='__main__':main()

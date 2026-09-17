"""Offline, independent reconstruction of curation; no network or image libraries.

Does not import prepare_candidates. Checks archived metadata, graph, exclusions,
bounded nomination ordering, and the anonymous review interface. Final image
acceptance and train/validation/confirmation partition are not executed.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import unicodedata
from urllib.parse import urlparse, unquote

HERE = Path(__file__).resolve().parent
SALT = 'visual-curation-20260915-v1'
OLD = ('old_title_exact_match', 'old_title_normalized_match', 'old_sha1_match',
       'old_author_key_match', 'old_source_key_match')
LABEL = {'Apples': 'apple', 'Bananas': 'banana', 'Oranges': 'orange', 'Glasses of water': 'water'}
FOOD = ('apple', 'banana', 'orange')
ALL = (*FOOD, 'water')
PATTERNS = (
    ('ai_generated', r'\bAI[\s-]*generated\b'),
    ('artificial_intelligence_generated', r'\bartificial\s+intelligence[\s-]+generated\b'),
    ('midjourney', r'\bMidjourney\b'), ('stable_diffusion', r'\bStable\s+Diffusion\b'),
    ('dall_e', r'\bDALL[\s-]*E\b'), ('computer_generated', r'\bcomputer[\s-]*generated\b'),
    ('3d_render', r'\b3D[\s-]+render(?:ed|ing|s)?\b'),
    ('digital_illustration', r'\bdigital\s+illustrations?\b'),
    ('drawings_of', r'\bdrawings?\s+of\b'), ('paintings_of', r'\bpaintings?\s+of\b'))

def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def norm(value): return unicodedata.normalize('NFKC', str(value)).replace('_', ' ').strip().casefold()
def digest(value): return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
def rank(kind, value): return digest([SALT, kind, value])

class Html(HTMLParser):
    def __init__(self, value):
        super().__init__(convert_charrefs=True); self.text=[]; self.links=[]; self.feed(str(value))
    def handle_data(self, text): self.text.append(text)
    def handle_starttag(self, tag, attrs):
        if tag == 'a': self.links += [v for k, v in attrs if k == 'href' and v]
    def plain(self): return ' '.join(' '.join(self.text).split())

def keys_from_ext(row):
    ext=row['extmetadata']; artist=Html(ext.get('Artist', {}).get('value', ''))
    text=norm(artist.plain()); authors=set(); sources=set(); raw_sources=[]
    if text not in {'', 'unknown', 'unknown author', 'anonymous', 'anonymous author', 'not provided', 'n/a'}:
        authors.add('artist_text:'+text)
    for link in artist.links:
        u=urlparse('https:'+link if link.startswith('//') else link); host=u.netloc.casefold(); path=unquote(u.path)
        if host in ('commons.wikimedia.org', 'en.wikipedia.org') and re.match(r'/wiki/(User|Creator):', path, re.I):
            authors.add('creator_url:'+host+norm(path))
        parts=path.strip('/').split('/')
        if host in ('flickr.com','www.flickr.com') and len(parts)>=2 and parts[0] in ('people','photos'):
            authors.add('creator_flickr:'+norm(parts[1]))
    for field in ('Credit','Source'):
        for link in Html(ext.get(field,{}).get('value','')).links:
            u=urlparse('https:'+link if link.startswith('//') else link); host=u.netloc.casefold(); path=unquote(u.path)
            parts=path.strip('/').split('/')
            if host in ('flickr.com','www.flickr.com') and len(parts)>=3 and parts[0]=='photos' and parts[2].isdigit():
                sources.add('source_flickr_photo:'+parts[2])
            if host=='commons.wikimedia.org' and path.startswith('/wiki/File:'):
                sources.add('source_commons_file:'+norm(path[6:])); raw_sources.append({'field':field,'url':link,'raw_title':path[6:]})
    return sorted(authors), sorted(sources), raw_sources

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--ranking',type=Path,default=HERE/'ranking_001'); args=parser.parse_args()
    folder=args.ranking.resolve(); source=HERE.parent/'details_002/execution/records.json'; oldfile=HERE.parent/'details_001/old_index_evidence.json'
    records=read(source); old=read(oldfile); byid={r['pageid']:r for r in records}
    assert len(records)==len(byid)==1876 and all(r['status']=='resolved' for r in records)
    # Check the already audited raw metadata fields again without importing the producer.
    raw_links={}; title_ids=defaultdict(set); key_ids=defaultdict(list)
    for row in records:
        pid=row['pageid']; authors,sources,links=keys_from_ext(row); raw_links[pid]=links
        assert authors==row['author_keys'] and sources==row['source_keys']
        expected=(any(t in old['exact_titles'] for t in (row['title'],row['current_title'])),
                  any(norm(t) in old['normalized_titles'] for t in (row['title'],row['current_title'])),
                  bool(row['original_file_sha1'] in old['original_file_sha1s']),
                  bool(set(authors)&set(old['author_keys'])),bool(set(sources)&set(old['source_keys'])))
        assert tuple(row[k] for k in OLD)==expected
        for title in (row['title'],row['current_title']): title_ids[norm(title)].add(pid)
        for key in authors+sources+(['sha1:'+row['original_file_sha1']] if row['original_file_sha1'] else []):key_ids[key].append(pid)
    parent={pid:pid for pid in byid}
    def find(pid):
        while parent[pid]!=pid:parent[pid]=parent[parent[pid]];pid=parent[pid]
        return pid
    def join(a,b): parent[find(a)]=find(b)
    for ids in key_ids.values():
        for pid in ids[1:]:join(ids[0],pid)
    before={pid:find(pid) for pid in byid}; before_count=len(set(before.values()))
    ambiguous=set(); old_source=set(); new_links=[]; statuses=Counter()
    for row in records:
        pid=row['pageid']
        for key in row['source_keys']:
            if not key.startswith('source_commons_file:'):continue
            title=key[len('source_commons_file:'):]; targets=sorted(title_ids.get(title,()))
            statuses['unresolved_outside_frame' if not targets else 'unique_target' if len(targets)==1 else 'ambiguous_normalized_title']+=1
            if title in old['normalized_titles']:old_source.add(pid)
            if len(targets)>1:ambiguous.add(pid)
            if len(targets)==1:
                target=targets[0]
                if before[pid]!=before[target]:
                    matches=[x for x in raw_links[pid] if norm(x['raw_title'])==title]
                    new_links.append({'from_pageid':pid,'to_pageid':target,'raw_source_links':matches,
                        'frame_title':byid[target]['title'],
                        'exact_raw_title_match':any(x['raw_title'].replace('_',' ')==byid[target]['title'].replace('_',' ') for x in matches)})
                join(pid,target)
    groups=defaultdict(list)
    for pid in sorted(byid):groups[find(pid)].append(pid)
    components={digest(members):members for members in groups.values()}
    cid={pid:c for c,members in components.items() for pid in members}
    oldgroups={cid[pid] for pid in byid if any(byid[pid][k] for k in OLD) or pid in old_source}
    badgroups={cid[pid] for pid in ambiguous}
    reasons={}; hits={}; label={}
    for row in records:
        pid=row['pageid']; why=[]; labs=sorted({LABEL[x] for x in row['categories']}); label[pid]=labs
        if cid[pid] in oldgroups:why.append('global_component_old_material_or_author_or_source')
        if cid[pid] in badgroups:why.append('global_component_ambiguous_commons_source_target')
        if len(labs)!=1:why.append('category_overlap')
        if row['mime'] not in {'image/jpeg','image/png','image/tiff','image/webp'}:why.append('mime_not_static_raster_candidate')
        if min(row['width'] or 0,row['height'] or 0)<512:why.append('short_side_below_512')
        if type(row['size']) is not int or not 0<row['size']<=16*1024**2:why.append('file_size_missing_or_outside_1_to_16MiB')
        if not row['license_pattern_supported']:why.append('license_pattern_not_supported')
        if not row['author_keys']:why.append('author_keys_missing')
        found=[]
        for field in ('Categories','ObjectName','ImageDescription'):
            text=Html(row['extmetadata'].get(field,{}).get('value','')).plain()
            for marker,pattern in PATTERNS:
                for m in re.finditer(pattern,text,re.I):found.append({'field':field,'marker':marker,'match':m.group(),'context':text[max(0,m.start()-60):m.end()+60]})
        if found:why.append('explicit_generated_or_nonphoto_marker')
        reasons[pid]=why;hits[pid]=found
    options=defaultdict(lambda:defaultdict(list))
    for pid in byid:
        if not reasons[pid]: options[cid[pid]][label[pid][0]].append(pid)
    for opt in options.values():
        for pids in opt.values():pids.sort(key=lambda p:(rank('file',p),p))
    water={c for c,opt in options.items() if opt.get('water')}
    ordered=sorted(components,key=lambda c:(rank('component',c),c)); used=set(); selected={k:[] for k in ALL}
    for c in ordered:
        if c in water and len(selected['water'])<126:selected['water'].append(options[c]['water'][0]);used.add(c)
    for round_id in range(42):
        for name in FOOD:
            choices=[c for c in ordered if c not in water and c not in used and options.get(c,{}).get(name)]
            if choices and len(selected[name])<42:
                c=choices[0];selected[name].append(options[c][name][0]);used.add(c)
    result=read(folder/'result.json'); saved=read(folder/'candidate_order.json'); inv=read(folder/'candidate_inventory.json'); graph=read(folder/'global_components.json')
    assert {g['component_id']:g['pageids'] for g in graph}==components
    for g in graph:
        assert g['old_contaminated']==(g['component_id'] in oldgroups)
        assert g['ambiguous_source_title']==(g['component_id'] in badgroups)
    assert [r['pageid'] for r in inv]==sorted(byid)
    for entry in inv:
        pid=entry['pageid']; assert entry['component_id']==cid[pid] and entry['metadata_exclusions']==reasons[pid]
        assert entry['metadata_eligible']==(not reasons[pid]) and entry['marker_hits']==hits[pid]
        assert entry['labels_from_metadata']==label[pid]
        expected_pool=label[pid][0] if not reasons[pid] and (label[pid]==['water'] or cid[pid] not in water) else None
        assert entry['pool']==expected_pool
    nominated=[]; counts={}
    for name in ALL:
        assert [n['pageid'] for n in saved[name]]==selected[name]
        target=84 if name=='water' else 28
        for i,n in enumerate(saved[name],1):
            pid=n['pageid']; assert n['fixed_order']==i and n['role']==('primary' if i<=target else 'reserve')
            assert n['component_id']==cid[pid] and n['nomination_hash']==rank('component',cid[pid]) and n['file_rank']==rank('file',pid)
            assert n['review_id']=='V_'+rank('review_id',pid)[:20]
            for key in ('title','original_url','description_url','original_file_sha1','mime','width','height','size','author_keys','source_keys'):
                assert n[key]==byid[pid][key]
        eligible=[p for p in byid if not reasons[p] and label[p]==[name]]
        pool=[p for p in eligible if name=='water' or cid[p] not in water]
        counts[name]={'metadata_eligible_files':len(eligible),'metadata_eligible_global_components':len({cid[p] for p in eligible}),
            'pool_files_after_water_priority':len(pool),'pool_global_components_after_water_priority':len({cid[p] for p in pool}),
            'nominated':len(selected[name]),'primary':min(target,len(selected[name])),'reserve':max(0,len(selected[name])-target),
            'minimum_shortfall_before_visual_review':max(0,target-len(selected[name]))}
        for k,v in counts[name].items():assert result['summary']['labels'][name][k]==v
        nominated+=saved[name]
    assert len(nominated)==len({x['pageid'] for x in nominated})==len({x['component_id'] for x in nominated})<=252
    expected_review=[{'review_id':n['review_id'],'preview_file':n['preview_file']} for n in sorted(nominated,key=lambda n:(rank('presentation',n['review_id']),n['review_id']))]
    assert read(folder/'reviewer_manifest.json')==expected_review
    templates=[json.loads(x) for x in (folder/'review_template.jsonl').read_text().splitlines()]
    assert [{k:t[k] for k in ('review_id','preview_file')} for t in templates]==expected_review
    for t in templates:
        assert not set(t)&{'title','label','component_id','author_keys','role','fixed_order','original_url'}
        assert t['visual_class'] is None and t['decision'] is None
    total=sum(n['size'] for n in nominated); summary=result['summary']
    assert summary['declared_original_bytes']==total and summary['global_components_before_filtering']==len(components)
    assert summary['global_components_old_contaminated']==len(oldgroups) and summary['water_reserved_global_components']==len(water)
    assert summary['exclusion_counts_nonexclusive']==dict(Counter(x for rr in reasons.values() for x in rr))
    assert summary['material_sets_assigned'] is False and summary['pixels_viewed']==summary['pixel_requests']==0
    for name,expected in result['output_sha256'].items():assert sha(folder/name)==expected
    freeze=read(folder/'freeze.json')
    for item in freeze['source_files']:
        assert sha(item['path'])==item['sha256'] and sha(folder/item['snapshot'])==item['snapshot_sha256']==item['sha256']
    source_paths=[source,oldfile,HERE/'plan.md',HERE/'prepare_candidates.py',Path(__file__),folder/'result.json']
    report={'status':'passed_metadata_reconstruction_only','completed_at_utc':datetime.now(timezone.utc).isoformat(),
        'source_sha256':{str(p):sha(p) for p in source_paths},'ranking':str(folder),
        'counts':{'metadata_records':len(records),'components_before_source_title_edges':before_count,
            'components_after_source_title_edges':len(components),'old_blocked_components':len(oldgroups),
            'old_blocked_files':sum(len(components[c]) for c in oldgroups),'ambiguous_source_components':len(badgroups),
            'explicit_source_status':dict(statuses),'water_reserved_components':len(water),'labels':counts,
            'nominated_total':len(nominated),'declared_original_bytes':total,'exclusions_nonexclusive':dict(Counter(x for rr in reasons.values() for x in rr))},
        'new_normalized_source_title_relationships':new_links,
        'checks':{'all_raw_artist_source_keys_reextracted':True,'all_old_exclusion_flags_recomputed':True,
            'full_graph_before_exclusions':True,'all_1876_decisions_exact':True,'all_nomination_orders_exact':True,
            'globally_unique_component_per_nomination':True,'anonymous_review_manifest_exact':True,
            'source_and_result_hashes_checked':True,'no_pixels_no_network_no_models':True},
        'not_executed':['pixel review','photographer/series identity verification','license chain verification',
                        'acceptance/reserve replacement','final 48/12/24 material allocation'],
        'limitations':['metadata component is not proven independent photographer',
                       'normalization-only title matches can conservatively merge different originals',
                       'fixed marker list only screens disclosed generation/non-photo metadata']}
    target=HERE/'independent_metadata_review.json'
    with target.open('x') as stream:json.dump(report,stream,ensure_ascii=False,indent=2);stream.write('\n')
    print(json.dumps(report['counts'],ensure_ascii=False,indent=2))

if __name__=='__main__':main()

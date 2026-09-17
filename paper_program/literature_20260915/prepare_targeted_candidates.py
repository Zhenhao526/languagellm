"""Transparent fallback input preparation after radar category API HTTP 429.
Does not modify plugin code; full-text packet/review/finalize remain radar.py.
"""
from pathlib import Path
from html.parser import HTMLParser
import urllib.request, json, sqlite3, re, datetime, hashlib, time
BASE=Path(__file__).resolve().parent
IDS=['2209.15342','2601.10169','2604.03266','2402.16247','2505.12872','2605.11695','2601.22041','2407.17960']
class Meta(HTMLParser):
    def __init__(self): super().__init__(); self.m={}
    def handle_starttag(self,tag,attrs):
        d=dict(attrs)
        if tag=='meta': self.m.setdefault(d.get('name',d.get('property','')),[]).append(d.get('content',''))
def main():
    (BASE/'metadata').mkdir(exist_ok=True)
    con=sqlite3.connect(BASE/'radar_data/watcher.sqlite3'); papers=[]
    for aid in IDS:
        path=BASE/'metadata'/f'{aid}.html'
        if not path.exists():
            req=urllib.request.Request(f'https://arxiv.org/abs/{aid}',headers={'User-Agent':'ResearchLiteratureAudit/1.0'})
            path.write_bytes(urllib.request.urlopen(req,timeout=40).read()); time.sleep(.3)
        raw=path.read_text(); p=Meta();p.feed(raw);m=p.m
        title=m['citation_title'][0]; authors=m['citation_author']; abstract=m.get('citation_abstract',m.get('og:description'))[0]
        pub=m['citation_date'][0].replace('/','-')+'T00:00:00+00:00'
        upd=m.get('citation_online_date',m['citation_date'])[0].replace('/','-')+'T00:00:00+00:00'
        versions=re.findall(r'/abs/'+re.escape(aid)+r'v(\d+)',raw);v=max(map(int,versions),default=1)
        cats=re.findall(r'<span class="primary-subject">[^<]*\(([^)]+)\)</span>',raw) or ['cs.MA']
        obj=dict(arxiv_id=aid,versioned_id=f'{aid}v{v}',title=title,abstract=abstract,authors=authors,categories=cats,primary_category=cats[0],published=pub,updated=upd,abs_url=f'https://arxiv.org/abs/{aid}',pdf_url=f'https://arxiv.org/pdf/{aid}v{v}',metadata_priority_score=0,metadata_priority_score_status='not_ranked; targeted fallback',metadata_html_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),retrieval='official arXiv HTML targeted fallback, not successful radar collector')
        now=datetime.datetime.now(datetime.timezone.utc).isoformat()
        con.execute('INSERT OR REPLACE INTO papers (arxiv_id,versioned_id,title,abstract,authors_json,categories_json,primary_category,published,updated,abs_url,pdf_url,first_seen_at,last_seen_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)',(aid,obj['versioned_id'],title,abstract,json.dumps(authors),json.dumps(cats),cats[0],pub,upd,obj['abs_url'],obj['pdf_url'],now,now))
        papers.append(obj);print(aid,title,flush=True)
    con.commit();con.close()
    out=dict(config='radar.toml',since_hours=24000,candidates_count=len(papers),papers=papers,retrieval_status='targeted_fallback_after_429',not_exhaustive=True,requested_collector_pool=20,actual_targeted_arxiv_pool=8,scope='2024–2026 nearest neighbours plus necessary 2022 predecessor; eight additional local/official comparisons handled separately')
    (BASE/'targeted_candidates.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
    (BASE/'collection_status.json').write_text(json.dumps(dict(status='collector_failed_http429; targeted_fallback_prepared',collector_log='collect.log',initial_packet_log='packet_2209.15342.log',database_preparation_script=__file__,plugin_modified=False,packet_review_finalize='original radar.py',systematic_search=False),ensure_ascii=False,indent=2))
if __name__=='__main__':main()

"""Transparent targeted arXiv subset after the category collector failed HTTP429.

The seven-paper triage includes two papers without a verified arXiv ID.
Six papers receive full-method reviews; Imai remains abstract-only because of
Japanese PDF extraction failure. Publisher-only entries are not given fake IDs.
"""
from pathlib import Path
from html.parser import HTMLParser
import urllib.request,json,sqlite3,re,datetime,hashlib,time
HERE=Path(__file__).resolve().parent
IDS=['2004.03868','2002.01335','2305.10920','2407.17960','2605.27532']

class Meta(HTMLParser):
    def __init__(self):super().__init__();self.m={}
    def handle_starttag(self,tag,attrs):
        d=dict(attrs)
        if tag=='meta':self.m.setdefault(d.get('name',d.get('property','')),[]).append(d.get('content',''))

def main():
    (HERE/'metadata').mkdir(exist_ok=True)
    con=sqlite3.connect(HERE/'data/watcher.sqlite3');papers=[]
    for aid in IDS:
        path=HERE/'metadata'/f'{aid}.html'
        if not path.exists():
            req=urllib.request.Request(f'https://arxiv.org/abs/{aid}',headers={'User-Agent':'ResearchLiteratureAudit/1.0'})
            path.write_bytes(urllib.request.urlopen(req,timeout=40).read());time.sleep(3)
        raw=path.read_text();parser=Meta();parser.feed(raw);m=parser.m
        title=m['citation_title'][0];authors=m['citation_author'];abstract=m.get('citation_abstract',m.get('og:description'))[0]
        pub=m['citation_date'][0].replace('/','-')+'T00:00:00+00:00'
        upd=m.get('citation_online_date',m['citation_date'])[0].replace('/','-')+'T00:00:00+00:00'
        version=max(map(int,re.findall(r'/abs/'+re.escape(aid)+r'v(\d+)',raw)),default=1)
        cats=re.findall(r'<span class="primary-subject">[^<]*\(([^)]+)\)</span>',raw) or ['cs.CL']
        item=dict(arxiv_id=aid,versioned_id=f'{aid}v{version}',title=title,abstract=abstract,authors=authors,
                  categories=cats,primary_category=cats[0],published=pub,updated=upd,abs_url=f'https://arxiv.org/abs/{aid}',
                  pdf_url=f'https://arxiv.org/pdf/{aid}',metadata_priority_score=0,
                  metadata_html_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  retrieval='Official arXiv HTML targeted subset after collector429, not a systematic pool')
        now=datetime.datetime.now(datetime.timezone.utc).isoformat()
        con.execute('INSERT OR REPLACE INTO papers (arxiv_id,versioned_id,title,abstract,authors_json,categories_json,primary_category,published,updated,abs_url,pdf_url,first_seen_at,last_seen_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)',
                    (aid,item['versioned_id'],title,abstract,json.dumps(authors),json.dumps(cats),cats[0],pub,upd,item['abs_url'],item['pdf_url'],now,now))
        papers.append(item);print(aid,title,flush=True)
    con.commit();con.close()
    (HERE/'targeted_arxiv_candidates.json').write_text(json.dumps(dict(config=str(HERE/'config.toml'),since_hours=60000,
        candidates_count=len(papers),papers=papers,requested_collector_pool=20,actual_arxiv_subset=len(papers),
        full_text_primary_target=6,not_exhaustive=True,retrieval_status='targeted_after_category_api429'),ensure_ascii=False,indent=2)+'\n')

if __name__=='__main__':main()

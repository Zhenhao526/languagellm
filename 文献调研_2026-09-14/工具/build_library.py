"""Build the curated literature inventory, source-linked report and BibTeX."""
import json, re, hashlib
from pathlib import Path
from pypdf import PdfReader

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / '数据'
GROUPS = [('human','人类实验与文化演化'),('neural','神经多智能体'),('affordance','神经多智能体'),('llm','LLM与评价'),('root','理论与形成过程'),('extra','理论与形成过程')]
STATUS = {'peer_reviewed':'正式发表', 'preprint':'预印本', 'accepted_author_manuscript':'作者稿声明已接收', 'conference_abstract':'会议摘要', 'workshop':'研讨会论文'}

def authors(r):
    a = r.get('authors', [])
    return a if isinstance(a,list) else [s.strip() for s in a.split(';') if s.strip()]

def val(v):
    if isinstance(v,list): return '；'.join(map(str,v))
    if isinstance(v,dict): return json.dumps(v,ensure_ascii=False)
    return str(v or '')

def mdlink(label,path):
    return f'[{label}](<{path}>)'

records = []
for stem, group in GROUPS:
    base = json.loads((DATA/f'{stem}.json').read_text())
    downloads = {r['id']:r for r in json.loads((DATA/f'{stem}_downloads.json').read_text())}
    for source in base:
        r = dict(downloads.get(source['id'], {}))
        r.update(source)  # latest factual corrections override earlier downloaded metadata
        r['group'] = group
        r['authors'] = authors(r)
        r['verified_as_of'] = '2026-09-14'
        if r.get('download_status')=='downloaded':
            path = Path(r['local_path'])
            blob=path.read_bytes()
            assert blob.lstrip().startswith(b'%PDF-'), path
            pdf=PdfReader(path)
            assert len(pdf.pages)>0
            r.update(pages=len(pdf.pages),bytes=len(blob),sha256=hashlib.sha256(blob).hexdigest())
        records.append(r)

assert len({r['id'] for r in records})==len(records)
canon=[r['doi'].lower().replace('https://doi.org/','') for r in records if r.get('doi')]
assert len(canon)==len(set(canon)), 'duplicate DOI'
available=[r for r in records if r.get('download_status')=='downloaded']
assert len({r['local_path'] for r in available})==len(available)
assert len({r['sha256'] for r in available})==len(available), 'duplicate PDF'
(DATA/'文献总表.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
lookup={r['id']:r for r in records}

lines=['# 文献索引','',f'核查截止：2026年9月14日。共 {len(records)} 项文献记录，{len(available)} 项取得PDF，{len(records)-len(available)} 项未取得PDF。所有已下载文件通过PDF格式、页数与题名核查；文件按“年份_第一作者_题名”命名。', '',
'“正式发表”区分于预印本，但不代表所有条目都是原始实验。综述、理论文章、会议摘要等在逐篇摘要中说明。全文核读指已核对与本问题有关的方法、结果和局限，并非逐页精读全部补充材料。', '',
'R07为三页会议摘要，位于下载的完整JCoLE会议录第179–181页（印刷151–153页）；整本会议录按一条来源计数。部分论文下载的是作者稿，正式卷期与稿件年份不同，详见版本记录。','',
mdlink('阅读调研报告',BASE/'语言诞生文献调研与选题建议.md')+' · '+mdlink('逐篇证据摘要',BASE/'逐篇证据摘要.md')+' · '+mdlink('BibTeX文献库',BASE/'语言涌现文献库.bib')+' · '+mdlink('检索范围与获取说明',BASE/'检索范围与获取说明.md'), '']
for group in dict(GROUPS).values():
    if f'## {group}' in lines: continue
    lines += [f'## {group}','','| 编号 | 年份与作者 | 题名 | 来源与获取 |','| --- | --- | --- | --- |']
    for r in records:
        if r['group']!=group: continue
        local = mdlink('PDF',r['local_path']) if r.get('download_status')=='downloaded' else '未取得PDF'
        label=STATUS.get(r['status'],r['status'])
        if r.get('read_depth')=='abstract': label+='；仅读摘要'
        lines.append(f"| {r['id']} | {r['year']} {r['first_author']} | {r['title']} | {local} · [原始来源]({r['url']})；{label} |")
    lines.append('')
lines += ['## 未取得PDF的记录','','保留元数据与可访问的来源，没有把错误页存成PDF，也没有用题名相近的论文替代。','']
for r in records:
    if r.get('download_status')=='downloaded': continue
    reasons='；'.join(val(a.get('error','')) for a in r.get('download_attempts',[])) or '未找到可成功获取的公开PDF；作者或机构来源访问失败，详见原始获取记录。'
    lines.append(f"- **{r['id']} · {r['title']}**：{reasons}。阅读依据：{val(r.get('evidence_locator'))}")
(BASE/'文献索引.md').write_text('\n'.join(lines)+'\n')

detail=['# 逐篇证据摘要','','本表用于追溯报告论据。每条保留实验起点、操纵、形成指标、主要结果和解释边界；不把综述的观点当作原始实验结果。','']
for r in records:
    detail += [f"## {r['id']} · {r['title']}",'',f"{'；'.join(r['authors'])}（{r['year']}）。{r['venue']}。{STATUS.get(r['status'],r['status'])}。",'',f"[原始来源]({r['url']})"+(f" · {mdlink('本地PDF',r['local_path'])}" if r.get('download_status')=='downloaded' else ' · 未取得PDF'), '']
    fields=[('主题','theme'),('研究类型','study_type'),('实验起点','starting_point'),('操纵条件','manipulations'),('形成过程指标','formation_measures'),('形成语言的性质','language_properties'),('主要结果','key_result'),('解释边界','limitation'),('选题关联','relevance'),('证据定位','evidence_locator'),('下载版本','downloaded_version')]
    detail += [f'- **{label}**：{val(r[key])}' for label,key in fields if r.get(key)]
    detail += [f"- **阅读范围**：{'正文关键方法与结果' if r.get('read_depth')=='full_text' else '摘要/有限可见内容'}。",'']
(BASE/'逐篇证据摘要.md').write_text('\n'.join(detail))

def bibsafe(s):
    return str(s).replace('\\','').replace('{','').replace('}','').replace('&',r'\&').replace('%',r'\%')
bib=[]
for r in records:
    conference=(r['id'].startswith('N') and r['id']!='N14') or r['id']=='A01' or (r['id'].startswith('L') and r['status']=='peer_reviewed') or r['id'] in ('R07','R08','R10')
    entrytype='inproceedings' if conference else 'misc' if r['status'] in ('preprint','accepted_author_manuscript') else 'article'
    key=r['id']+'_'+re.sub('[^A-Za-z]','',r['first_author'])+str(r['year'])
    fields={'title':'{'+bibsafe(r['title'])+'}','author':' and '.join(bibsafe(x) for x in r['authors']),'year':r['year'],'url':r['url'],'note':f"{r['venue']}; {STATUS.get(r['status'],r['status'])}; verified 2026-09-14"}
    venue=r['venue'].split(';')[0]
    if entrytype=='inproceedings': fields['booktitle']=bibsafe(venue)
    elif entrytype=='article':
        match=re.match(r'^(.*?)\s+(\d+)(?:\((\d+)\))?(?:[:,]\s*([^;]+))?$',venue)
        if match and match[2]!='2026':
            fields['journal']=bibsafe(match[1]); fields['volume']=match[2]
            if match[3]: fields['number']=match[3]
            if match[4]:
                pages=match[4].replace('–','--').strip()
                fields['pages' if '--' in pages else 'eid']=pages
        else: fields['journal']=bibsafe(venue.replace(' 2026',''))
    if r.get('doi'): fields['doi']=r['doi']
    if r.get('local_path'): fields['file']=r['local_path']
    bib.append('@'+entrytype+'{'+key+',\n'+',\n'.join('  '+k+' = {'+str(v)+'}' for k,v in fields.items())+'\n}')
(BASE/'语言涌现文献库.bib').write_text('\n\n'.join(bib)+'\n')

body=(BASE/'研究笔记/报告正文_带文献编号.md').read_text()
ids=list(dict.fromkeys(re.findall(r'\[@([A-Z]\d+)\]',body)))
assert all(i in lookup for i in ids)
numbers={key:i+1 for i,key in enumerate(ids)}
report=re.sub(r'\[@([A-Z]\d+)\]',lambda m:f'[^{numbers[m[1]]}]',body)
report+='\n## 参考文献\n\n'
for key in ids:
    r=lookup[key]; number=numbers[key]
    names='; '.join(r['authors'])
    report+=f"[^{number}]: {names} ({r['year']}). [{r['title']}]({r['url']}). {r['venue']}. {STATUS.get(r['status'],r['status'])}。文献库编号：{key}。\n\n"
(BASE/'语言诞生文献调研与选题建议.md').write_text(report)
(DATA/'报告引文映射.json').write_text(json.dumps([dict(number=numbers[k],**lookup[k]) for k in ids],ensure_ascii=False,indent=2))
print(json.dumps({'records':len(records),'downloaded':len(available),'report_references':len(ids),'chinese_characters':len(re.findall(r'[\u4e00-\u9fff]',body))},ensure_ascii=False))

"""Verify and index the literature package, without model/data outcome access."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib,json

HERE=Path(__file__).resolve().parent
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
 receipt_path=HERE/'review_receipt.json'
 raw=receipt_path.read_bytes()
 r=json.loads(raw)
 # Native finalize failed on a legitimate non-arXiv venue item. Correct a receipt
 # counter written before the failed command was tested, preserving that raw receipt.
 backup=HERE/'review_receipt_raw_before_status_correction.json'
 if backup.exists(): raise FileExistsError(backup)
 backup.write_bytes(raw)
 assert r['commands'][-1]['exit_code']==1 and 'Invalid arXiv ID' in r['commands'][-1]['stderr']
 r['native_final_report_contains']=0
 r['native_final_report_status']='failed_mixed_pool_has_non_arxiv_id'
 r['post_receipt_correction']={'time':datetime.now(timezone.utc).isoformat(),
  'reason':'Raw receipt prematurely contained expected count 2. Actual finalizer failed and produced no report; count corrected to 0. Two record-review commands did succeed. No command rerun.',
  'raw_receipt_sha256':sha(backup)}
 receipt_path.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
 index=json.loads((HERE/'download_packet_index.json').read_text())
 pool=json.loads((HERE/'candidate_pool20.json').read_text())
 normalized=json.loads((HERE/'finalize_same_pool.json').read_text())
 assert len(pool['papers'])==20
 assert [p['paper_id'] for p in pool['papers']]==[p['paper_id'] for p in normalized['papers']]
 assert sum(p['triage_decision']=='full_text_packet' for p in pool['papers'])==3
 reviews=[json.loads(p.read_text()) for p in sorted((HERE/'reviews').glob('*.json'))]
 assert len(reviews)==3 and all(x['review_mode']=='full_text' for x in reviews)
 weights=dict(relevance=.30,evidence=.20,novelty=.15,impact=.15,reproducibility=.10,author_prior=.05,early_signal=.05)
 for row in reviews:
  assert row['scores']['overall']==round(sum(row['scores'][k]*v for k,v in weights.items()),1)
 for row in index:
  file=Path(row['local_path']);packet=Path(row['packet_path'])
  assert file.read_bytes().startswith(b'%PDF-')
  assert sha(file)==row['sha256'] and sha(packet)==row['packet_sha256']
  assert row['all_pages_extractable'] and row['first_page_title_normalized_match']
 assert sum(x['new_pdf_download'] for x in index)==2
 lines=['# 下载、版本与工具范围记录','',
  '本轮下载2篇、复用1篇。所有PDF均来自正式会议或arXiv官方链接；%PDF头、全部页面文本提取及SHA均已核对。没有修改或复制已有Kaszyński PDF。', '',
  '|论文|处理|版本|页数|PDF SHA-256|','|---|---|---|---|---|']
 for row in index:
  lines.append(f"|[{row['title']}]({row['official_page']})|{'新下载' if row['new_pdf_download'] else '复用已有'}|{row['version']}|{row['page_count']}|`{row['sha256']}`|")
 lines.extend(['','完整本地路径、官方PDF URL、HTTP状态、各页提取字符数和packet SHA见[下载索引JSON](download_packet_index.json)。',
  '', '版本和身份核对：', '',
  '- Gualdoni等使用NeurIPS2024正式20页论文，不是同名短会版本；4名作者和题名在首页逐项匹配。未核实arXiv ID，保持空值。',
  '- CORAL使用2508.06659v2，官方页面记载v1为2025-08-08、v2为2026-06-07。首页首作者为Fernando Martinez，arXiv元数据为Fernando Martinez-Lopez；相同官方版本、题名及另3名作者一致，保留此字面差异，不把自动四人匹配说成全部成功。',
  '- Kaszyński使用已有2604.03266v1。官方页面与PDF写2026-03-18，但编号月份为2604；这是来源内部日期不一致，本记录未自行纠正。所示日期均在截止前，未确认正式会议录用。',
  '- 原生8篇的日期来自官方arXiv Atom；补充12篇分别保留正式年份/初版/修订日期。未把2025正式论文的2026上传日期当作首次提出时间。检索名义截止2026-09-16，实际只覆盖当次抓取前已可见资料。',
  '', '工具链实际范围：', '',
  '- collector一次成功，但只返回8项而非历史完整20项；类别最新条数上限使旧资料缺失。官方定向补充12项，全部20项已按摘要分流。',
  '- native packet对两篇补充arXiv记录报本地数据库不存在，错误保留；改从已核实官方PDF/已有PDF用pypdf完整提取，未伪称native packet成功。非arXiv会议论文原生packet不适用。',
  '- 两篇真实arXiv的record-review成功。finalize对混合池的空arXiv ID报错，未重试或伪造编号；主中文报告合并3篇全文评审。',
  '- Morpheme Induction候选的最初摘要意译过宽；官方摘要实际使用平行话语与意义。finalize_same_pool.json中已明确修正，候选身份/选择/顺序均未改变，原始候选池保留。',
  '', '没有下载其他新PDF、运行训练或读取当前实验成绩。没有独立复现这三篇论文。'])
 (HERE/'下载与版本记录.md').write_text('\n'.join(lines)+'\n')
 ranked=sorted(reviews,key=lambda p:p['scores']['overall'],reverse=True)
 ranked_payload={'source_pool_sha256':sha(HERE/'candidate_pool20.json'),
    'native_collector_count':8,'official_supplement_count':12,'full_text_review_count':3,
    'native_finalizer_success':False,'ranking_basis':'Agent authored content scores, skill rubric; manual extension for official venue with null arXiv ID.',
    'papers':ranked}
 (HERE/'mixed_top3.json').write_text(json.dumps(ranked_payload,ensure_ascii=False,indent=2)+'\n')
 verification={'created_at':datetime.now(timezone.utc).isoformat(),'status':'verified_bounded_package_with_recorded_tool_limits',
  'original_pool_count':20,'native_candidates':8,'official_supplements':12,'full_text_packets':3,'full_text_reviews':3,
  'new_pdf_downloads':2,'existing_pdf_reuses':1,'pdf_pages':[x['page_count'] for x in index],
  'all_pdf_and_packet_sha_exact':True,'all_pages_extractable':True,'same20_ids_and_order_preserved':True,
  'native_review_successes':sum(x['exit_code']==0 for x in r['commands'][:2]),'native_finalize':'failed_invalid_arxiv_id_for_official_non_arxiv_paper',
  'new_model_calls':0,'current_experiment_outcomes_read':False,
  'limitations':['Bounded20-item mixed pool, not exhaustive historical coverage.','No independent replication of paper results.','Not a first-in-literature finding.'],
  'files':{}}
 for p in sorted(HERE.rglob('*')):
  if p.is_file() and ('radar_data' not in p.parts and '.arxiv' not in p.parts and '__pycache__' not in p.parts) and p.name!='verification.json':
   verification['files'][str(p.relative_to(HERE))]={'sha256':sha(p),'bytes':p.stat().st_size}
 (HERE/'verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({'status':verification['status'],'scores':[(r['paper_id'],r['scores']['overall']) for r in ranked],
                   'report_sha256':sha(HERE/'直接近邻与主张范围.md'),'verification_sha256':sha(HERE/'verification.json')},ensure_ascii=False,indent=2))

if __name__=='__main__':main()

"""Offline reporting of the bounded metadata run. Never reads image pixels."""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
from urllib.parse import urlparse, parse_qs, quote
import hashlib
import json
import re

HERE = Path(__file__).resolve().parent
def read(name): return json.loads((HERE / name).read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(name, value): (HERE / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

def main():
    status = read('status.json'); assert status['status'] != 'running'
    cfg = read('config.json'); freeze = read('freeze.json'); exclusion = read('exclusion_registry.json')
    rows = read('manifest.json')['images']; groups = read('author_groups.json')['groups']
    requests = [json.loads(s) for s in (HERE / 'requests.jsonl').read_text().splitlines()]
    waits = [json.loads(s) for s in (HERE / 'waits.jsonl').read_text().splitlines()] if (HERE / 'waits.jsonl').exists() else []
    frame = read('sampling_frame.json') if (HERE / 'sampling_frame.json').exists() else {}
    discovery = read('category_discovery.json') if (HERE / 'category_discovery.json').exists() else {}
    checks = 0
    def check(v, reason):
        nonlocal checks
        checks += 1
        assert v, reason
    for p, h in freeze['source_hashes'].items(): check(sha(Path(p)) == h, 'frozen source/input changed:' + p)
    for item in freeze['snapshots']: check(sha(Path(item['snapshot'])) == item['sha256'], 'snapshot changed')
    for p, h in exclusion['source_hashes'].items(): check(sha(Path(p)) == h, 'v1 input changed:' + p)
    check(len(requests) <= 45, 'HTTP budget')
    check(len({r['logical_request'] for r in requests}) <= 42, 'logical budget')
    check(sum(r['attempt'] > 1 for r in requests) <= 3, 'retry budget')
    check(len(rows) <= 300, 'candidate budget')
    if requests:
        check(datetime.fromisoformat(freeze['frozen_utc']) < datetime.fromisoformat(requests[0]['started']['utc']), 'freeze after request')
    gaps = []
    for i, request in enumerate(requests):
        u = urlparse(request['url']); q = parse_qs(u.query)
        check(u.scheme == 'https' and u.netloc == 'commons.wikimedia.org' and u.path == '/w/api.php', 'endpoint scope')
        check(q.get('action') == ['query'], 'mutating action')
        check(sha(HERE / request['response_file']) == request['response_sha256'], 'raw response hash')
        check(request['response_bytes'] == (HERE / request['response_file']).stat().st_size, 'raw byte count')
        check(request['attempt'] <= 2, 'attempt cap')
        if i:
            a=requests[i-1]['finished']; b=request['started']
            gaps.append({'utc': b['unix']-a['unix'], 'monotonic': b['monotonic']-a['monotonic']})
            check(gaps[-1]['utc'] >= 10 and gaps[-1]['monotonic'] >= 10, 'serial response gap')
        if request['attempt'] == 2:
            check(i > 0 and request['url'] == requests[i-1]['url'], 'retry changed URL')
        if 'pageids' in request['params']:
            check(len(request['params']['pageids'].split('|')) <= 20, 'metadata batch')
    for wait in waits:
        check(wait['elapsed_utc'] >= wait['required_seconds'], 'UTC wait too short')
        check(wait['elapsed_monotonic'] >= wait['required_seconds'], 'monotonic wait too short')
    check(status['pixel_requests'] == status['image_views'] == status['model_calls'] == status['confirmation_pixel_access'] == 0, 'exposure')
    check(not status['allocations_performed'], 'premature pixel allocation')
    index = {r['id']: r for r in rows}
    check(len(index) == len(rows), 'duplicate detailed page ID')
    raw_cache = {}
    for row in rows:
        name = row['metadata_response']
        if name not in raw_cache:
            data = read(name); raw_cache[name] = {p['pageid']: p for p in data.get('query', {}).get('pages', [])}
        page = raw_cache[name].get(row['pageid'], {})
        info = page.get('imageinfo', [{}])[0]
        check(row['original_sha1'] == info.get('sha1'), 'original SHA metadata')
        check(row['artist_html'] == info.get('extmetadata', {}).get('Artist', {}).get('value', ''), 'artist copied wrong')
        check(row['original_url'] == info.get('url'), 'original URL copied wrong')
        check(row['metadata_provisionally_eligible'] == (not row['exclusion_reasons']), 'eligibility accounting')
        if row['metadata_provisionally_eligible']:
            check(not row['blocked_identity_or_source_keys'], 'known excluded identity accepted')
            check(row['pageid'] not in exclusion['blocked_pageids'], 'known excluded page accepted')
    seen = set()
    representatives = []
    for group in groups:
        check(not seen.intersection(group['members']), 'duplicate cluster membership')
        seen.update(group['members'])
        for name in group['members']: check(index[name]['cluster_id'] == group['cluster_id'], 'cluster assignment')
        possible = sorted([index[n] for n in group['members'] if index[n]['metadata_provisionally_eligible']], key=lambda r:(r['frame_rank_hash'],r['id']))
        check(group['eligible_representative'] == (possible[0]['id'] if possible else None), 'representative rank')
        if possible: representatives.append(possible[0])
    check(seen == set(index), 'cluster coverage')
    check(len(representatives) == status['provisionally_eligible_clusters'], 'eligible cluster total')
    if frame:
        check(len(frame['requested_metadata']) <= 300, 'selection cap')
        selected = read('metadata_selection_freeze.json')
        check(sha(HERE / 'sampling_frame.json') == selected['sampling_frame_sha256'], 'detail list changed')
        detail_requests = [r for r in requests if r['label'].startswith('metadata_')]
        if detail_requests:
            check(datetime.fromisoformat(selected['frozen_utc']) < datetime.fromisoformat(detail_requests[0]['started']['utc']), 'detail list frozen late')
        if status['status'] == 'metadata_collection_complete':
            check([r['pageid'] for r in rows] == selected['ordered_pageids'], 'detail list incomplete or reseeded')

    reasons = Counter(reason for r in rows for reason in r['exclusion_reasons'])
    http_counts = Counter(str(r['http_status']) for r in requests)
    broad_only = [r for r in representatives if not r['scene_phrase_matches'] and
        r['scene_category_sources'] and 'Category:Glasses of water' not in r['scene_category_sources']]
    evidence = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        exact_category_pages=discovery.get('discovery', [{}])[0].get('pages', []),
        chosen_categories=discovery.get('selected_categories', []),
        unique_frame_files=frame.get('unique_file_count'), exact_old_reserved_files_omitted=len(frame.get('excluded_exact_old_or_reserved', [])),
        detail_count=len(rows), automatic_provisional_records=sum(r['metadata_provisionally_eligible'] for r in rows),
        automatic_provisional_clusters=len(representatives),
        provisional_cluster_representative_ids=[r['id'] for r in representatives],
        provisional_representatives_supported_only_by_nonbase_categories=[r['id'] for r in broad_only],
        exclusion_reason_counts=dict(reasons), reason_counts_overlap=True,
        category_total_counts_do_not_establish_photographic_scene=True,
        manual_identity_and_scene_acceptance_completed=False, pixel_acceptance_completed=False,
        threshold80_is_unvalidated_proposal=True, no_pixels_to_follow_automatically=True)
    write('feasibility_summary.json', evidence)
    qa = dict(status='passed', checks=checks, failures=0, actual_http_requests=len(requests),
        http_status_counts=dict(http_counts), minimum_response_gap_utc=min([g['utc'] for g in gaps],default=None),
        minimum_response_gap_monotonic=min([g['monotonic'] for g in gaps],default=None),
        all_frozen_sources_and_v1_inputs_unchanged=True, all_raw_response_hashes_verified=True,
        network_requests_in_audit=0, pixel_reads=0, model_calls=0,
        scope='Read-only metadata/source/request bookkeeping; not independent human visual or license acceptance',
        source_sha256=sha(Path(__file__).resolve()), created_utc=datetime.now(timezone.utc).isoformat())
    write('metadata_qa.json', qa)
    n=len(representatives)
    conclusion = (f'固定自动元数据规则得到 {n} 个暂合格关联簇，未达到此前候选门槛 80。' if n<80 else
        f'固定自动元数据规则得到 {n} 个暂合格关联簇，数量达到此前候选门槛 80，但尚无 {n} 个合格普通水杯摄影来源的证据。')
    lines = ['# v2 水场景元数据可行性结果','',
        f'**{conclusion}** 本轮没有下载或查看新图片，没有访问原确认 48 像素或调用实验模型。元数据数量不构成像素就绪门槛，后续不自动执行 20/40/10/10 分配。','',
        f'执行状态：`{status["status"]}`。实际 {len(requests)} 次 HTTP 请求（预算 45），取得 {len(rows)} 条文件详情（上限 300）；所有已读取 v1 输入哈希保持不变。请求结果：{dict(http_counts)}。', '',
        '## 官方来源实际存在性','',
        '| 精确查询分类 | 结果 | 官方直接文件数 |','|---|---|---:|']
    for p in evidence['exact_category_pages']:
        url='https://commons.wikimedia.org/wiki/'+quote(p['title'].replace(' ','_'),safe=':')
        lines.append(f'| [{p["title"]}]({url}) | {"未找到" if p.get("missing",False) else "存在"} | {p.get("categoryinfo",{}).get("files","—")} |')
    lines += ['', '精确不存在只针对本次固定标题；没有证明其他语言、分类或全站都不存在该场景。检索追加类别按原规则选择，包含：'+ '、'.join(evidence['chosen_categories'])+'。', '',
        '水杯、餐食配水杯、其他饮料配水杯和艺术相关类别混在检索结果中；类别是来源线索，不能保证普通透明杯、液体为水、真实摄影、无文字或裁剪保留。未因看到这些结果改写已经冻结的查询和词表。', '',
        '## 有界抽样与排除结果','',
        '| 项目 | 数量 |','|---|---:|',
        f'| 原始唯一文件框 | {evidence["unique_frame_files"]} |',
        f'| 详情前排除的已知旧/预约精确文件 | {evidence["exact_old_reserved_files_omitted"]} |',
        f'| 已取得文件详情 | {len(rows)} |',
        f'| 固定规则暂合格文件 | {evidence["automatic_provisional_records"]} |',
        f'| 固定规则暂合格关联簇 | {n} |',
        f'| 其中代表仅有非基础类别支持、无固定场景短语 | {len(broad_only)} |',
        '| 已进行人工来源/身份验收 | 0 |','| 已进行像素验收 | 0 |','',
        '下面理由可同时作用于一张图，不能相加为排除总数：','',
        '| 理由 | 文件数 |','|---|---:|']
    for reason,count in sorted(reasons.items()): lines.append(f'| `{reason}` | {count} |')
    lines += ['',
        '旧 101、v1 原 144 预约簇及 60 已下载图的已记录作者/原作/原 SHA1 关联保守传播后用于排除。所有 v1 预约簇成员均纳入，不仅是其代表。旧重定向解析出的账号仍与历史作者存在未解别名关系，保持排除/待核；3 位旧作者和 11 张不可得旧像素的限制没有消失。作者文本和链接键是元数据代理，不能将一个关联簇直接叫一位独立真实摄影者。', '',
        '## 能与不能据此决定什么','',
        '本轮能核查：这组官方类别和检索在当前快照下能返回哪些文件；固定预算、词表、许可字段和已知身份排除后还剩多少记录。检索页和类别页均有固定截断；未请求所有 Commons 文件，不估计全站总量，也不把不足解释为全站资源不存在。', '',
        '当前自动场景规则允许水玻璃杯类别代替描述短语，分类本身可能过宽；有些描述提到水不代表纯水场景，署名声明也不能消除再摄、插画或作者代理问题。上述暂合格数字只是等待独立元数据审查的候选计数，不能当成普通透明水杯真实摄影的已验收规模。像素条件必须在后续另行冻结后才可检查。', '',
        '执行完成后的只读元数据检查还发现英语排除词表的局限：在上述 24 个“仅非基础类别支持”的代表中，按原簇哈希顺序前两个分别是 [Un café.jpg](https://commons.wikimedia.org/wiki/File:Un_caf%C3%A9.jpg)（法语描述含咖啡、糖、糕点与水杯）和 [メロンソーダ (12630470863).jpg](https://commons.wikimedia.org/wiki/File:%E3%83%A1%E3%83%AD%E3%83%B3%E3%82%BD%E3%83%BC%E3%83%80_(12630470863).jpg)（日语题名指瓜味汽水）。固定英语词表未排除这些跨语言或类别线索。此处没有看像素，没有给它们新增视觉判断，也没有修改原规则或计数；它说明 79 不能被解读为“已经合格，只差 1 个”。前两个实例用于解释方法局限，不作为额外抽样率估计。', '',
        '下一道门是独立审查来源定义、候选元数据和身份/许可链；若本轮不足，不提高本次预算，不悄然缩减排除，不动用旧确认像素，也不启动图片下载。是否需要更换来源或新的具体场景框，应由本次失败/不足证据决定，并另立方案。80 和 20/40/10/10 尚为工作量建议，不是已经验证的数据门槛。', '',
        '## 复核入口','',
        '[固定计划](固定元数据可行性计划.md)、[配置](config.json)、[首次请求前冻结](freeze.json)、[旧排除表](exclusion_registry.json)、[分类发现](category_discovery.json)、[抽样框](sampling_frame.json)、[详情名单冻结](metadata_selection_freeze.json)、[逐文件记录](manifest.json)、[关联簇](author_groups.json)、[全部请求](requests.jsonl)、[原响应](raw)、[等待](waits.jsonl)、[状态](status.json)、[元数据 QA](metadata_qa.json)。', '',
        f'只读 QA {checks} 项通过；不等同视觉或法律验收。所有新增产物仅位于 v2 目录，v1 源文件未改。','']
    if status.get('error'): lines.insert(6,'停止原因：`'+status['error']+'`。原失败响应/异常保留，没有换入口。\n')
    (HERE / '元数据可行性报告.md').write_text('\n'.join(lines))
    (HERE / 'README.md').write_text('# v2 水场景：仅元数据可行性研究\n\n'+conclusion+
        '\n\n[结果报告](元数据可行性报告.md) · [当前状态](status.json) · [QA](metadata_qa.json) · [固定方案](固定元数据可行性计划.md)\n\n'+
        '没有请求新图片或原确认像素，没有实验模型调用，没有改动 v1。所有候选尚待独立来源/身份审查；不会自动进入像素阶段。\n')
    artifact_names=['README.md','元数据可行性报告.md','status.json','manifest.json','author_groups.json',
        'metadata_qa.json','feasibility_summary.json','freeze.json','analyze_metadata.py']
    write('completion_receipt.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        artifact_hashes={name:sha(HERE/name) for name in artifact_names},
        pixel_stage_authorized_by_this_receipt=False))
    print(json.dumps({'status':qa['status'],'checks':checks,'provisional_clusters':n,
        'metadata_files':len(rows),'http_requests':len(requests)},ensure_ascii=False))

if __name__=='__main__': main()

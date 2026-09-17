"""Review three already-collected official text responses; never fetch subresources."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import re
import importlib.util

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('initial_source_helpers', HERE / 'prepare_initial_source_review.py')
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
read, sha, keys, closure = helpers.read, helpers.sha, helpers.keys, helpers.closure


def main():
    initial_paths = [HERE / '八项来源初审.json', HERE / '八项来源初审.md']
    inputs = initial_paths + [HERE / 'prepare_initial_source_review.py', HERE / '来源核查固定方案.md',
        HERE / 'source_001/selection.json', HERE / 'source_001/freeze.json', HERE / 'source_001/completion.json',
        HERE / 'source_001/raw/commons_files_a1.response', HERE / 'source_001/raw/commons_authors_a1.response',
        HERE / 'source_001/raw/flickr_original_a1.response', helpers.BASE / 'exclusion_registry.json',
        helpers.BASE / 'manifest.json', helpers.REVIEW / 'review_input.json']
    hashes = {str(p): sha(p) for p in inputs}
    completion = read(HERE / 'source_001/completion.json')
    assert completion['status'] == 'text_collection_complete' and completion['requests'] == 3
    assert all(hashes[p] == value for p, value in completion['response_hashes'].items())
    selection = read(HERE / 'source_001/selection.json')
    assert selection['workflow_ordinals'] == [8, 19, 22, 42]
    assert selection['pixel_reserved_ordinals'] == [46, 51, 62, 70]
    initial = read(initial_paths[0])
    frozen = read(helpers.BASE / 'freeze.json')
    def snapshot(suffix):
        entry = next(r for r in frozen['snapshots'] if r['original'].endswith(suffix))
        p = Path(entry['snapshot'])
        assert sha(p) == entry['sha256']
        hashes[str(p)] = entry['sha256']
        return p
    records = [('old101', r) for r in read(snapshot('visual_confirmation_v1/old_exclusion_registry.json'))['images']]
    records += [('v1_metadata800', r) for r in read(snapshot('visual_confirmation_v1/manifest.json'))['images']]
    records += [('v2_metadata300', r) for r in read(helpers.BASE / 'manifest.json')['images']]
    keysets = [keys(r) for _, r in records]
    files_path = HERE / 'source_001/raw/commons_files_a1.response'
    authors_path = HERE / 'source_001/raw/commons_authors_a1.response'
    flickr_path = HERE / 'source_001/raw/flickr_original_a1.response'
    pages = {p['pageid']: p for p in read(files_path)['query']['pages']}
    author_response = read(authors_path)['query']
    assert {'from': 'User:WalkingRadiance', 'to': 'User:Programmatically'} in author_response['redirects']
    flickr_html = flickr_path.read_text()
    structured = []
    for block in re.findall(r'<script[^>]*type=[\"\']application/ld\+json[\"\'][^>]*>(.*?)</script>', flickr_html, re.S):
        structured.extend(json.loads(block).get('@graph', []))
    image_metadata = next(r for r in structured if r.get('@type') == 'ImageObject')
    assert image_metadata['author']['name'] == image_metadata['creator']['name'] == 'Dave Matos'
    assert image_metadata['author']['url'].rstrip('/') == 'https://www.flickr.com/photos/dmatos'
    assert image_metadata['acquireLicensePage'].rstrip('/') == 'https://www.flickr.com/photos/dmatos/1081916080'
    assert image_metadata['license'].rstrip('/') == 'https://creativecommons.org/licenses/by-sa/2.0'
    assert 'photostreamId_8068581%40N02' in flickr_html
    assert re.search(r'class="attribution-username"\s+title="ThrasherDave">ThrasherDave</p>', flickr_html)
    flickr_wiki = pages[162912038]['revisions'][0]['slots']['main']['content']
    review_template = re.search(r'\{\{FlickreviewR\|[^}]+\}\}', flickr_wiki).group()
    for exact in ['status=passed', 'author=ThrasherDave', '8068581@N02/1081916080',
                  'reviewdate=2025-03-28 16:36:44', 'reviewlicense=cc-by-sa-2.0', 'reviewer=FlickreviewR 2']:
        assert exact in review_template
    aliases = [dict(
        kind='official_work_and_account_attribution', target_ordinal=22,
        keys=['text:dave matos', 'text:thrasherdave', 'flickr-author:dmatos', 'flickr-author:8068581@n02', 'flickr-photo:1081916080'],
        evidence=[dict(path=str(files_path), sha256=sha(files_path), excerpt=review_template),
                  dict(path=str(flickr_path), sha256=sha(flickr_path),
                       excerpt='JSON-LD: author.name=creator.name=Dave Matos; author.url=/photos/dmatos; acquireLicensePage=/photos/dmatos/1081916080; license=CC BY-SA 2.0. HTML attribution username=ThrasherDave; photostreamId_8068581%40N02.')],
        limit='Stable account/work linkage supported, not absolute civil identity or unseen-repost independence.'),
        dict(kind='official_commons_user_redirect', target_ordinal=46,
             keys=['commons-user:walkingradiance', 'text:walkingradiance', 'commons-user:programmatically', 'text:programmatically'],
             evidence=[dict(path=str(authors_path), sha256=sha(authors_path),
                            excerpt={'from': 'User:WalkingRadiance', 'to': 'User:Programmatically'})],
             limit='Public account-page redirect explains recorded names; not a real-person identity audit.')]
    all_sets = keysets + [set(x['keys']) for x in aliases]
    registry = read(helpers.BASE / 'exclusion_registry.json')
    excluded, passes = closure(registry['blocked_keys'], all_sets)
    all79 = read(helpers.REVIEW / 'review_input.json')['rows']
    affected79 = []
    for r in all79:
        connected, _ = closure(keys(r['original_manifest_record']), all_sets)
        if connected & excluded:
            affected79.append(r['id'])
    assert affected79 == []
    rows = []
    for prior in initial['rows']:
        p = pages[int(prior['id'].split(':')[1])]
        info, rev = p['imageinfo'][0], p['revisions'][0]
        meta = info['extmetadata']
        wikitext = rev['slots']['main']['content']
        val = lambda k: meta.get(k, {}).get('value', '')
        assert info['sha1'] == prior['provenance']['original_sha1']
        assert rev['revid'] == prior['provenance']['page_revision'][0]['revid']
        assert info['timestamp'] == prior['provenance']['original_version_timestamp']
        assert val('LicenseShortName') == prior['license']['name']
        assert val('LicenseUrl').rstrip('/') == prior['license']['url'].rstrip('/')
        evidence = [line.strip() for line in wikitext.splitlines() if
                    re.match(r'\|(?:source|author|permission)\s*=', line, re.I) or
                    re.search(r'\{\{(?:self|cc-by|FlickreviewR)', line, re.I)]
        assert evidence
        if prior['ordinal'] != 22:
            assert re.search(r'\|source\s*=\s*\{\{own\}\}', wikitext, re.I)
            assert '{{self|' in wikitext
        m = next(r['original_manifest_record'] for r in all79 if r['id'] == prior['id'])
        connected, _ = closure(keys(m), all_sets)
        overlap = sorted(connected & excluded)
        assert not overlap
        notes = prior['specific_followup']
        if prior['ordinal'] == 22:
            notes = 'Flickr 原作 HTML 与 Commons 审核模板一致支持 Dave Matos／dmatos／ThrasherDave／8068581@N02；旧排除传播无交集，具体 alias 待核已解除。'
        if prior['ordinal'] == 46:
            notes = '官方 API 明确 User:WalkingRadiance → User:Programmatically；来源署名与上传者的账户差异有文本解释，旧排除传播无交集，具体待核已解除。'
        if prior['ordinal'] == 62:
            notes = '官方 API 确认用户介绍页 missing；文件原作者、自作和许可模板仍明确，上传署名一致。未查询 civil 身份，不因介绍页缺失否定有限来源支持。'
        rows.append(dict(
            ordinal=prior['ordinal'], id=prior['id'], title=prior['title'],
            source_status='source-supported', source_gate_eligible=True,
            creator_as_credited=prior['creator_as_credited'],
            source_basis='Official file text, recorded attribution, license declaration, exact original file identity, and known-cluster exclusion are consistent.',
            source_notes=notes, scope='limited_documented_source_support_not_absolute_ownership_or_real_person_proof',
            license=dict(name=val('LicenseShortName'), url=val('LicenseUrl'), attribution_required=val('AttributionRequired'),
                         attribution_as_recorded=val('Attribution'), selected_license_only=True),
            official_file_evidence=dict(path=str(files_path), sha256=sha(files_path), pageid=p['pageid'],
                                        revision_id=rev['revid'], revision_timestamp=rev['timestamp'],
                                        original_sha1=info['sha1'], original_version_timestamp=info['timestamp'],
                                        wikitext_relevant_lines=evidence,
                                        old_revision_and_original_file_identity_match=True),
            resolved_aliases=[a for a in aliases if a['target_ordinal']==prior['ordinal']],
            original_source_page=prior['provenance']['source_page'],
            attribution_draft=prior['attribution_draft'],
            exclusion=dict(direct_old1180_overlap=sorted(connected & set(registry['blocked_keys'])),
                           propagated_overlap=overlap, connected_keys=sorted(connected),
                           connected_records=[dict(collection=s,id=r.get('id'),title=r['title']) for (s,r),k in zip(records,keysets) if k&connected]),
            previously_fixed_pixel_role='workflow_candidate' if prior['ordinal'] in selection['workflow_ordinals'] else 'reserved_no_pixel',
            pixel_status='not_reviewed', model_use_approved=False, new_usable_sample=False,
        ))
    assert len(rows) == 8
    assert all(sha(Path(p)) == h for p, h in hashes.items())
    out = dict(schema='water_eight_official_text_source_review_v1', created_utc=datetime.now(timezone.utc).isoformat(),
        reviewer='literature_update; current official requests collected by root; historical v2 metadata collector role disclosed',
        scope='Independent reading of saved official text; no new network/pixels/models; limited data-source support, not legal audit.',
        input_sha256=hashes, initial_review_preserved=True,
        initial_wording_correction=dict(location='八项来源初审.md / 旧排除与关联传播',
          original='第8项还有同作者的另一幅 v2 作品', corrected='第8项的同作者另一幅106394147来自 v1 metadata800；未据此替换代表或访问像素。',
          reason='Collection label prose error; initial JSON already records v1_metadata800 correctly. Initial files retained unchanged.'),
        rows=rows, aliases=aliases,
        summary=dict(source_supported=8,pending=0,contradicted=0,workflow_candidates_source_eligible=[8,19,22,42],
                     reserved_pixel_ordinals=[46,51,62,70],new_usable_images=0),
        exclusion_propagation=dict(old1180_count=len(registry['blocked_keys']), closed_count=len(excluded),
                                   passes=passes, added_exclusion_keys=sorted(excluded-set(registry['blocked_keys'])),
                                   all79_checked=79, affected_representatives=affected79, original_registry_modified=False,
                                   method='Initial read-only link-key extraction over old101+v1metadata800+v2metadata300; add two official account/work co-association sets, close on metadata components. Username ThrasherDave is a text key, not asserted to be a Flickr path alias.'),
        all8_original_sha1_timestamp_and_revision_match=True, input_hashes_unchanged=True,
        new_network_requests=0,pixel_reads=0,model_calls=0)
    dest=HERE/'八项来源复核.json'
    dest.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    lines=['# 八项来源复核：已有官方文本支持', '',
      '2026-09-16。对根代理固定取得的三份官方文本响应作独立阅读后，**八项均可判为有限 source-supported，pending 0、contradicted 0**。这是资料链支持；真实照片、透明容器、液态水可辨及实际224裁剪尚未评审，新增可用图仍为0。原定前四项（8/19/22/42）通过来源门，后四项（46/51/62/70）仍保留像素，不因来源通过补位或提前下载。', '',
      '本复核没有发起请求、加载HTML子资源、读图或运行模型。当前官方请求由根代理收集；复核者曾参与早期元数据采集，不能称为全程独立的实名／权属认证。初审及原v1/v2记录均保留。', '',
      '| 序号 | 署名／当前来源支持 | 采用的许可 | 来源判定 |',
      '|---|---|---|---|']
    for r in rows:
        short={8:'自作；作者和记录上传名一致',19:'自作；另有 Victor Blacus 许可声明',22:'Flickr 原作、署名、稳定账号和历史审核互相对应',42:'自作；Tamorlan 署名一致',46:'自作；WalkingRadiance 重定向到 Programmatically',51:'自作；Attribution 明确 MF-Warburg',62:'自作；作者和上传名一致，介绍页缺失不否定该声明',70:'自作；Adam S. Keck 署名明确链接 Ghostis'}[r['ordinal']]
        lines.append(f"| {r['ordinal']} | [{r['creator_as_credited']}]({r['original_source_page']})：{short} | {r['license']['name']} | source-supported |")
    lines+=['', '两个原待核项有了具体支持：', '',
      '- **Dave Matos（22）：** Flickr 原作 `1081916080` 的 JSON-LD 指定作者与创作者 Dave Matos、路径 `dmatos` 和 CC BY-SA 2.0；可读署名另列 ThrasherDave，页面元数据对应账号 `8068581@N02`。Commons 的 FlickreviewR 模板也指向该稳定账号下的同一作品，记录审核通过、日期 `2025-03-28 16:36:44`、审核许可 `cc-by-sa-2.0`。姓名、路径别名、用户名、稳定账号和作品 ID 现在可一起参与排除，不需再无界查证。',
      '- **WalkingRadiance（46）：** Commons 作者 API 明确返回 `User:WalkingRadiance → User:Programmatically`，解释已有 Artist 与记录上传者的名称差异。该关系只支持公开账户关联，不升级为绝对自然人身份结论。', '',
      '八项新响应的原图 SHA1、原版本时间戳、页面 revision ID 均与原冻结候选一致，没有静默替换图像版本。七项原文明确自作与 self 许可模板；第19项的多许可模板也由 extmetadata 明确给出本次采用的 CC BY-SA 4.0。第62项缺用户介绍页没有推翻相容的文件署名和自作许可声明。水与冰按同一物料处理，不计为添加食物；容器和液态部分仍由像素评审判断。', '',
      f"旧1,180键经已有链接及新账户关系传播后，闭包为{len(excluded)}键。全部79原代表均检查，无新增旧／预约来源交集；8项之间也未据新关系新增合并。旧注册表没有改动，未知别名仍不能绝对排除。部分条目在v1元数据800中出现，不等于曾被下载或预约；已下载60、预约144簇及其成员的排除继续保留。", '',
      '**初审文字更正（不覆盖原文）：** 初审MD称第8项有“另一幅v2作品”，应为“同作者另一幅106394147来自v1元数据800”。初审JSON的 collection 字段已正确；本次明确标注此归属笔误，不改原初审或候选代表。', '',
      f"[机器记录]({dest})包含逐项 wikitext 证据行、原始响应 SHA256、Flickr 必要字段、两组新增账号关系、原图一致性及排除关联；未整段复制HTML。", '',
      '三份已保存响应：', '']
    for p in [files_path,authors_path,flickr_path]:
        lines.append(f'- `{p.name}`：`{sha(p)}`。')
    lines+=['', '来源支持只解除本阶段的具体来源疑点。原定前四项仍须另行固定实际像素名单并通过技术、匿名双视觉评审；不能据来源判定自动计为独立模型确认样本。']
    (HERE/'八项来源复核.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'status':'complete','source_supported':8,'pending':0,'contradicted':0,
                      'original_identity_match':True,'affected_old_exclusion':0,'new_usable_images':0},ensure_ascii=False))


if __name__=='__main__':
    main()

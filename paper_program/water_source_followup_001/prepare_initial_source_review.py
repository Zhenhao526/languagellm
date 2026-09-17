"""Bounded eight-item review of existing metadata. No network, pixels or models."""
from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import urlparse, urljoin, unquote, parse_qs, urlencode
import hashlib
import html
import json
import re
import unicodedata

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / 'visual_confirmation_v2_water'
REVIEW = HERE.parent / 'visual_confirmation_v2_water_review'
ORDINALS = [8, 19, 22, 42, 46, 51, 62, 70]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    return json.loads(p.read_text())


def norm(x):
    x = html.unescape(re.sub('<[^>]*>', '', str(x)))
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFC', x).replace('_', ' ')).strip().casefold()


def keys(r):
    # Existing keys are preserved. New link-derived aliases only strengthen exclusions.
    k = set(r.get('author_keys', [])) | set(r.get('original_source_keys', []))
    if r.get('title'):
        k.add('commons-title:' + norm(re.sub(r'^File:', '', r['title'], flags=re.I)))
    if r.get('original_sha1'):
        k.add('sha1:' + r['original_sha1'])
    for val in [r.get('artist_html', ''), r.get('credit_html', '')]:
        for link in re.findall(r'href=[\"\']([^\"\']+)', val):
            u = urlparse(urljoin('https://commons.wikimedia.org', html.unescape(link)))
            path = unquote(u.path)
            if u.hostname in ('flickr.com', 'www.flickr.com'):
                m = re.match(r'/(?:people|photos)/([^/]+)', path)
                if m:
                    k.add('flickr-author:' + norm(m[1]))
                m = re.match(r'/photos/[^/]+/(\d+)', path)
                if m:
                    k.add('flickr-photo:' + m[1])
            if u.hostname == 'commons.wikimedia.org':
                title = path.split('/wiki/', 1)[-1] if '/wiki/' in path else parse_qs(u.query).get('title', [''])[0]
                m = re.fullmatch(r'(User|Creator):([^/#?]+)', title, re.I)
                if m:
                    k.update(['commons-' + m[1].lower() + ':' + norm(m[2]), 'text:' + norm(m[2])])
    return k


def closure(seed, sets):
    result = set(seed)
    passes = 0
    while True:
        previous = len(result)
        for ks in sets:
            if result & ks:
                result.update(ks)
        passes += 1
        if len(result) == previous:
            return result, passes


NOTES = {
    8: ('source_supported', 'Artist 明确署名 Benjamín Núñez González 并链接同名 Commons 用户；Credit 为 Own work，记录上传者同名，CC BY-SA 4.0 名称/URL一致。可作为有限来源支持，无须仅因没有外部实名证明而无限待核。',
        '未发现具体署名冲突；若后续刷新官方文本，保留作者页/文件修订及许可模板即可。'),
    19: ('source_supported', 'Artist、上传者均为 Victor Blacus，Credit 为 Own work；Permission 另明确由 Victor Blacus 给予许可。选用元数据列明的 CC BY-SA 4.0 作为拟用许可，不把分类中的其他许可名称合并为额外要求。',
         '水与冰是同一化学物料的不同相态，不按添加食物排除。透明容器、液态部分及裁剪后辨识度仍须像素评审。'),
    22: ('pending_specific_alias_check', 'Commons 署名为 Dave Matos，Credit 指向 Flickr 原作 1081916080（路径作者 dmatos），许可为 CC BY-SA 2.0；分类含 FlickreviewR 2 审核标记。这足以支持存在具体原作及许可声明，但现有元数据没有稳定 Flickr 作者 NSID 和实际审核日期。',
         '优先核对既有 Flickr 原作文本中作者/许可及 dmatos 账号的稳定 ID，再将 Dave Matos、dmatos 与 NSID 联到排除图；不把上传者 Infrogmation 当摄影者。'),
    42: ('source_supported', 'Artist 链接 Tamorlan，上传者同名，Credit 为 Own work，CC BY 3.0 名称/URL一致；分类另有 Files by User:Tamorlan 和相机型号，但后者不单独证明摄影真实性。',
         '无具体来源冲突；元数据支持进入另行冻结的像素阶段，尚未通过视觉。'),
    46: ('pending_specific_alias_check', 'Artist 为 WalkingRadiance，链接被原 HTML 标成 mw-redirect；Credit 为 Own work、许可 CC BY-SA 4.0，而记录上传者是 Programmatically。自作/许可声明有来源，两个账号是否为重命名关系尚未在本地文本中解明。',
         '只需有界查询 User:WalkingRadiance 的官方重定向及返回身份，再与 Programmatically/旧键核对。当前把两名作为保守排除关联，不断言已经证明同一人，也不因差异直接否定许可。'),
    51: ('source_supported', 'Artist 为 User:MF-Warburg、上传者为 MF-Warburg、Credit 为 Own work；独立 Attribution 字段还指定 MF-Warburg，CC BY-SA 2.5 名称/URL一致。',
         '署名草案采用 Attribution 指定的 MF-Warburg，不照抄 User: 前缀为姓名；标题不直观但描述明确一杯水，是否可辨另看像素。'),
    62: ('source_supported', 'Artist 与上传者同为 Јана Смилеска，Credit 为 Own work，CC BY-SA 4.0 名称/URL一致。Artist 链接是用户页红链；用户介绍页不存在不等于上传账号不存在或自作声明无效。',
         '有限来源支持可成立；如补官方文本，应使用只读用户查询，不能打开 action=edit 链接。竞赛分类不是独立摄影身份证明。'),
    70: ('source_supported', 'Artist 文字为 Adam S. Keck，明确链接 User:Ghostis；记录上传者 Ghostis，Credit 为 Own work，CC BY-SA 4.0 名称/URL一致。署名文字与账户映射在已有字段中直接给出。',
         '保留 Adam S. Keck ↔ Ghostis 两层身份键；不因署名与账号不同自动待核，仍不声称排尽所有未知别名。'),
}


def main():
    paths = [REVIEW / 'review_input.json', REVIEW / 'root_independent_review.json',
             HERE.parent / 'confirmation_readiness_20260916/水来源核查简表.json',
             BASE / 'manifest.json', BASE / 'author_groups.json', BASE / 'freeze.json', BASE / 'exclusion_registry.json']
    fingerprints = {str(p): sha(p) for p in paths}
    snapshot_entries = read(BASE / 'freeze.json')['snapshots']
    def snapshot(suffix):
        e = next(e for e in snapshot_entries if e['original'].endswith(suffix))
        p = Path(e['snapshot'])
        assert sha(p) == e['sha256']
        fingerprints[str(p)] = e['sha256']
        return p
    old = read(snapshot('visual_confirmation_v1/old_exclusion_registry.json'))['images']
    v1 = read(snapshot('visual_confirmation_v1/manifest.json'))['images']
    v2 = read(BASE / 'manifest.json')['images']
    registry = read(BASE / 'exclusion_registry.json')
    original_blocked = set(registry['blocked_keys'])
    assert (len(old), len(v1), len(v2), len(original_blocked)) == (101, 800, 300, 1180)
    records = [('old101', r) for r in old] + [('v1_metadata800', r) for r in v1] + [('v2_metadata300', r) for r in v2]
    keysets = [keys(r) for _, r in records]
    for i, (_, r) in enumerate(records):
        if r.get('id') == 'commons:119699630':
            keysets[i].update(['commons-user:programmatically', 'text:programmatically'])
    propagated_blocked, passes = closure(original_blocked, keysets)
    packet = read(REVIEW / 'review_input.json')['rows']
    inventory = {r['id']: r for r in read(paths[2])['rows']}
    annotations = {r['id']: r for r in read(REVIEW / 'root_independent_review.json')['rows']}
    targets = [r for r in packet if r['ordinal'] in ORDINALS]
    assert [r['ordinal'] for r in targets] == ORDINALS
    rows = []
    for r in targets:
        a = annotations[r['id']]
        assert a['water_text_evidence'] == 'E' and a['scene_context'] == 'D'
        assert inventory[r['id']]['creator']['raw_html'] == r['artist']['html']
        raw_path = BASE / r['provenance']['metadata_response']
        fingerprints[str(raw_path)] = sha(raw_path)
        assert fingerprints[str(raw_path)] == r['provenance']['metadata_response_sha256']
        raw_page = next(x for x in read(raw_path)['query']['pages'] if x['pageid'] == r['pageid'])
        assert raw_page['imageinfo'][0]['extmetadata'] == r['all_extmetadata']
        ks = keys(r['original_manifest_record'])
        extra_aliases = []
        if r['ordinal'] == 22:
            extra_aliases = ['flickr-author:dmatos']
        if r['ordinal'] == 46:
            extra_aliases = ['commons-user:programmatically', 'text:programmatically']
            ks.update(extra_aliases)
        connected, _ = closure(ks, keysets)
        overlap = sorted(connected & propagated_blocked)
        assert not overlap
        assert r['pageid'] not in registry['blocked_pageids']
        linked = [{'collection': src, 'id': x.get('id'), 'title': x['title']}
                  for (src, x), k in zip(records, keysets) if k & connected]
        status, assessment, followup = NOTES[r['ordinal']]
        creator = 'MF-Warburg' if r['ordinal'] == 51 else r['artist']['text']
        endpoints = [dict(kind='recorded_commons_file', url=r['provenance']['source_page'], visited=False)]
        rev = r['provenance']['page_revision'][0]['revid']
        endpoints.append(dict(kind='derived_frozen_revision_text_api', url='https://commons.wikimedia.org/w/api.php?' + urlencode({
            'action': 'query', 'format': 'json', 'formatversion': 2, 'prop': 'revisions',
            'revids': rev, 'rvprop': 'ids|timestamp|content', 'rvslots': 'main'}), visited=False))
        if r['ordinal'] == 22:
            endpoints += [dict(kind='recorded_flickr_work', url=r['credit']['links'][0], visited=False),
                          dict(kind='derived_flickr_creator_slug_not_verified', url='https://www.flickr.com/people/dmatos/', visited=False)]
        elif r['ordinal'] == 46:
            endpoints.append(dict(kind='derived_official_redirect_query', url='https://commons.wikimedia.org/w/api.php?' + urlencode({
                'action': 'query', 'format': 'json', 'formatversion': 2, 'titles': 'User:WalkingRadiance',
                'redirects': 1, 'prop': 'info'}), visited=False))
        elif r['ordinal'] == 62:
            endpoints.append(dict(kind='derived_official_read_only_user_query', url='https://commons.wikimedia.org/w/api.php?' + urlencode({
                'action': 'query', 'format': 'json', 'formatversion': 2, 'list': 'users', 'ususers': 'Јана Смилеска'}), visited=False))
        else:
            endpoints += [dict(kind='recorded_creator_page', url=u, visited=False) for u in r['artist']['links']]
        evidence = []
        for name in ['Artist', 'Credit', 'ImageDescription', 'Permission', 'Attribution', 'LicenseShortName', 'LicenseUrl', 'AttributionRequired']:
            if name in r['all_extmetadata']:
                evidence.append(dict(field=name, **r['all_extmetadata'][name]))
        rows.append(dict(
            ordinal=r['ordinal'], id=r['id'], title=r['title'],
            source_evidence_status='source_supported_metadata_limited',
            proposed_source_gate_status=status, source_assessment=assessment,
            specific_followup=followup, creator_as_credited=creator,
            artist=r['artist'], credit=r['credit'], license=r['license'],
            source_evidence=evidence, categories=r['categories'],
            provenance={k:v for k,v in r['provenance'].items() if k not in ['original_url','metadata_request_url','frame_sources']},
            raw_response_path=str(raw_path), raw_response_sha256=sha(raw_path),
            evidence_json_pointer=f"/query/pages/[pageid={r['pageid']}]/imageinfo/0/extmetadata",
            additional_exclusion_only_aliases=extra_aliases,
            exclusion_check=dict(direct_old1180_overlap=sorted(ks & original_blocked),
                                 propagated_overlap=overlap, connected_keys=sorted(connected),
                                 linked_metadata_records=linked,
                                 repeated_metadata_id_is_not_new_pixel_exposure=True),
            future_text_entrypoints=endpoints,
            attribution_draft=dict(title=r['title'][5:], creator=creator,
                                   source_page=r['provenance']['source_page'],
                                   license_name=r['license']['name'], license_url=r['license']['url'],
                                   modifications='尚无图像处理；后续若实际裁剪/缩放须按真实处理另记'),
            pixel_review='not_performed', independent_person_identity_proven=False,
            approved_for_model_use=False, new_usable_sample=False,
        ))
    assert all(sha(Path(p)) == h for p, h in fingerprints.items())
    data = dict(
        schema='water_eight_source_initial_review_v1', created_utc=datetime.now(timezone.utc).isoformat(),
        scope='existing_metadata_only_not_legal_audit_or_pixel_acceptance', ordinals=ORDINALS,
        reviewer='literature_update; original metadata collector; separate from future official-source verification',
        source_hashes=fingerprints, rows=rows,
        summary=dict(count=8, metadata_source_claim_supported=8, proposed_source_gate_supported=6,
                     pending_specific_alias_check=2, contradicted=0, new_usable_samples=0),
        exclusion_propagation=dict(original_key_count=1180, conservative_propagated_key_count=len(propagated_blocked),
                                   added_exclusion_keys=sorted(propagated_blocked-original_blocked), closure_passes=passes,
                                   input_metadata_records=dict(old101=101, v1=800, v2=300),
                                   rules='Preserve stored keys; derive Commons User/Creator and Flickr account/work keys from recorded Artist/Credit links. Treat WalkingRadiance/Programmatically as provisional exclusion-only co-association. Propagate on all recorded components. Never infer that every uploader is an author.',
                                   affected_selected_targets=0, original_registry_modified=False,
                                   limitation='Unknown aliases and missing authors remain unresolved; no absolute independence claim.'),
        raw_extmetadata_exactly_matched=True, inputs_unchanged=True,
        new_network_requests=0, pixel_reads=0, model_calls=0,
    )
    out = HERE / '八项来源初审.json'
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    text = [
        '# 八项水图来源初审', '',
        '2026-09-16。本次只核对既有元数据中的 E∩D 八项，不联网、不读图、不改原文件。**六项已有证据足以给出有限的来源支持；另两项有明确作品/许可声明，但各留一个具体作者关联待核。** 这不是法律审计，也不是像素通过：八项新增可用样本仍为 0。整理者曾参与原采集，不能称为独立摄影身份认证；根代理另作官方文本核查。', '',
        '“有限来源支持”指署名、来源、文件标识和许可声明在现有记录中有可追溯且相容的链条，无具体反证；不要求证明绝对真实姓名。Own work 与有链接的署名、相符上传记录和许可一起构成证据，不能只凭两个字证明摄影真实性或作者独立。所有八项都保留独立视觉门槛。', '',
        '| 原序号 | 原署名 | 现有许可 | 建议来源门状态 | 关键证据／后续 |',
        '|---|---|---|---|---|',
    ]
    for r in rows:
        label = '有限 source-supported' if r['proposed_source_gate_status']=='source_supported' else '具体作者关联待核'
        text.append(f"| {r['ordinal']} | [{r['creator_as_credited']}]({r['provenance']['source_page']}) | {r['license']['name']} | {label} | {r['source_assessment']} |")
    text += ['', '## 两个有界核查，以及不必无限待核的情形', '',
        '- **22／Dave Matos：** 原作入口是 [Flickr 1081916080](https://www.flickr.com/photos/dmatos/1081916080/)。先读该作品的作者与许可文本，核实路径名 dmatos 是否对应 Dave Matos，再取得可得的稳定 Flickr 作者 ID；随后将新 ID、文本名和路径别名都对旧注册表传播。Commons 分类中的 FlickreviewR 2 是已有审核线索，本地没有审核时间或审核原文，不能编成已核完整许可历史。该项不是“完全无来源”。',
        '- **46／WalkingRadiance：** 原 Artist 链接标注重定向，上传者为 Programmatically。机器表给出只读 Commons 重定向 API 参数。若官方文本确认别名关系，则保存响应、补记身份键并重跑排除；若不能解明，保留这一具体 pending，不通过推测认定同一人，也不把上传者自动当摄影者。',
        '- **62／Јана Смилеска：** 作者、上传者和自作声明一致，用户介绍页红链不是账号不存在的证据。可给有限来源支持；若补查，使用机器表中的 `list=users` 只读 API，不访问原 `action=edit` 链接。',
        '- **70／Adam S. Keck：** 原署名文字直接链接 User:Ghostis，上传者 Ghostis，与自作声明一致。这是已有的姓名—账号映射，不因名称不同而无限待核。',
        '- **19／Water and ice：** 原描述明确水的液态与固态，Permission 另署 Victor Blacus。冰不是添加食物，不因相态共存排除；是否有可见液态水、透明饮水容器，仍由未执行的像素评审决定。',
        '- **51／MF-Warburg：** 原 Attribution 字段明确指定 MF-Warburg，署名草案采用该写法。其他七项没有单独 Attribution 字段，不代表无需署名；可使用已有 Artist、题名、来源与许可 URL，未来如裁剪/缩放再记录实际修改。', '',
        '## 旧排除与关联传播', '',
        f"只读 [旧 1,180 键注册表]({BASE / 'exclusion_registry.json'})及其冻结的旧101、v1 800条、v2 300条元数据。除保留原键外，补提取已有 Artist/Credit 链接中的账号/作品键；对第46项把两账号作保守的排除关联，**只用于不漏排，不证明二者同一人**。闭包扩大为 {len(propagated_blocked)} 键，八个目标均无直接或传播后的排除交集。新增键完整保存在新 JSON，旧注册表没有回写。", '',
        '第22、42、46、62项也出现在 v1 的800条元数据内；它们不是因此新增的独立来源，但也不能仅因曾收元数据就推定曾看像素。原排除范围是旧101、已下载60、预约144簇及全部关联成员；这些四项在既有注册表及本次保守传播中未命中。第8项还有同作者的另一幅 v2 作品，仍按一个关联簇处理，本次不替换代表或追加候选。', '',
        '原三位作者未明项和待核别名继续保留。未命中有限已知键不等于证明所有隐藏别名均不存在；若后续官方文本新增关联，必须再次传播后再进入新像素计划，而不是改变旧记录。', '',
        '## 可交付范围', '',
        f"[八项机器记录]({out})保存原文证据、修订ID、原SHA1、API响应SHA256、候选官方文本入口、逐项署名草案与传播见证。8条 extmetadata 均与原 API 字节对应，全部读取输入的哈希未改变。后续网页入口仅列出，未访问；没有下载、模型调用或新像素评审。", '',
        '这份初审允许来源证据已足够的六项继续接受另行冻结的像素评审，不要求为实名不可绝对证明而停滞。另两项的缺口具体、有限，可由官方文本核查澄清。任何来源支持都不能自动变成“独立摄影者已确认”“真实照片已确认”或“可用于模型”。',
    ]
    (HERE / '八项来源初审.md').write_text('\n'.join(text) + '\n')
    print(json.dumps({'status':'complete','source_supported':6,'specific_alias_pending':2,
                      'propagated_exclusion_keys':len(propagated_blocked),'new_usable_samples':0},ensure_ascii=False))


if __name__ == '__main__':
    main()

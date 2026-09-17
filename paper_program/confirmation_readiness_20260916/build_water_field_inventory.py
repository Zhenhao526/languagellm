"""Read frozen metadata only; no requests, image decoding, or model imports."""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
from urllib.parse import urlparse
import hashlib
import json

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / 'visual_confirmation_v2_water'
REVIEW = HERE.parent / 'visual_confirmation_v2_water_review'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def key_mode(row):
    keys = row['artist']['original_author_keys']
    if any(k.startswith('flickr-author:') for k in keys):
        return 'flickr_author_key'
    if any(k.startswith('commons-user:') for k in keys):
        return 'commons_user_key'
    return 'text_or_other_key'


def source_mode(row):
    if row['credit']['text'].strip().lower() == 'own work':
        return 'own_work_claim'
    if any(k.startswith('flickr-photo:') for k in row['credit']['original_source_keys']):
        return 'flickr_work_key'
    return 'other_credit'


def summarize(rows):
    predicates = {
        'artist_text': lambda r: r['creator']['text'],
        'artist_links': lambda r: r['creator']['recorded_links'],
        'credit_text': lambda r: r['source']['credit_text'],
        'credit_links': lambda r: r['source']['recorded_credit_links'],
        'original_source_keys': lambda r: r['source']['original_source_keys'],
        'commons_page': lambda r: r['source']['commons_page'],
        'page_revision': lambda r: r['source']['page_revision'],
        'original_sha1': lambda r: r['file_identity']['original_sha1'],
        'license_name': lambda r: r['license']['name'],
        'license_url': lambda r: r['license']['url'],
        'attribution_value': lambda r: r['license']['attribution'],
        'permission_field': lambda r: r['permission_field_present'],
        'nonempty_restrictions': lambda r: r['license']['restrictions'],
    }
    return dict(
        count=len(rows),
        water_labels=dict(Counter(r['root_annotation']['water_text_evidence'] for r in rows)),
        scene_contexts=dict(Counter(r['root_annotation']['scene_context'] for r in rows)),
        author_key_modes=dict(Counter(r['creator']['key_structure'] for r in rows)),
        source_modes=dict(Counter(r['source']['credit_structure'] for r in rows)),
        license_names=dict(Counter(r['license']['name'] for r in rows)),
        nonempty_field_counts={k: sum(bool(f(r)) for r in rows) for k, f in predicates.items()},
    )


def cell(value):
    return str(value).replace('|', '\\|').replace('\n', ' ')


def main():
    inputs = [REVIEW / 'review_input.json', REVIEW / 'root_independent_review.json',
              REVIEW / 'input_freeze.json', REVIEW / 'annotation_clarification.json',
              BASE / 'manifest.json', BASE / 'author_groups.json', BASE / 'freeze.json',
              BASE / 'exclusion_registry.json', BASE / 'feasibility_summary.json',
              BASE / 'metadata_helpers.py', BASE / 'prepare_freeze.py']
    before = {str(p): sha(p) for p in inputs}
    packet = read(inputs[0])
    annotations = read(inputs[1])
    manifest = {r['id']: r for r in read(BASE / 'manifest.json')['images']}
    groups = read(BASE / 'author_groups.json')['groups']
    group_map = {g['eligible_representative']: g for g in groups if g['eligible_representative']}
    registry = read(BASE / 'exclusion_registry.json')
    blocked = set(registry['blocked_keys'])
    annotation_map = {r['id']: r for r in annotations['rows']}
    expected = read(BASE / 'feasibility_summary.json')['provisional_cluster_representative_ids']
    assert [r['id'] for r in packet['rows']] == expected
    assert [r['id'] for r in annotations['rows']] == expected
    assert len(expected) == len(set(expected)) == len(group_map) == 79
    snapshots = read(BASE / 'freeze.json')['snapshots']
    for item in snapshots:
        p = Path(item['snapshot'])
        assert sha(p) == item['sha256'], p
        before[str(p)] = item['sha256']
    rows = []
    raw_cache = {}
    for p in packet['rows']:
        aid = p['id']
        m, a, g = manifest[aid], annotation_map[aid], group_map[aid]
        assert m == p['original_manifest_record']
        assert a['order'] == p['ordinal'] and a['title'] == p['title']
        assert a['cluster_id'] == g['cluster_id'] == p['cluster']['cluster_id']
        assert not g['blocked_by_old_or_reserved']
        assert not (set(g['identity_keys']) & blocked)
        assert p['pageid'] not in registry['blocked_pageids']
        raw_path = BASE / p['provenance']['metadata_response']
        rh = sha(raw_path)
        assert rh == p['provenance']['metadata_response_sha256']
        before[str(raw_path)] = rh
        if str(raw_path) not in raw_cache:
            raw_cache[str(raw_path)] = {x['pageid']: x for x in read(raw_path)['query']['pages']}
        raw_page = raw_cache[str(raw_path)][p['pageid']]
        assert raw_page['title'] == p['title']
        assert raw_page['imageinfo'][0]['extmetadata'] == p['all_extmetadata']
        risk = []
        if key_mode(p) == 'text_or_other_key':
            risk.append('作者键仅文本或未被原解析器覆盖的链接；需独立核实身份/别名。')
        if source_mode(p) == 'own_work_claim':
            risk.append('Own work 是原页面声明；尚未独立确认摄影者和授权链。')
        if p['ordinal'] in (34, 44):
            risk.append(p['artist']['collector_note'])
        if p['ordinal'] == 74:
            risk.append('原 Artist 只说明英文维基上传者；旧 commons-user 键和 parseable=true 不证明摄影作者。')
        if p['ordinal'] == 27:
            risk.append('500px/Archive Team 转存链；不得把归档像素 URL 当作元数据核查入口。')
        risks = risk + [a['rationale']]
        candidate_links = list(dict.fromkeys(p['artist']['links'] + p['credit']['links']))
        no_pixel_links = [u for u in candidate_links if not any(
            s in u.lower() for s in ('upload.wikimedia.org', 'drscdn.', '/web/', '.jpg', '.jpeg', '.png'))]
        r = dict(
            ordinal=p['ordinal'], id=aid, pageid=p['pageid'], title=p['title'],
            description=p['description'], categories=p['categories'],
            cluster=dict(id=g['cluster_id'], original_rank=g['cluster_rank'],
                         original_members=g['members'], original_identity_keys=g['identity_keys'],
                         known_registry_key_overlap=[],
                         interpretation='metadata_connected_component_not_verified_person'),
            root_annotation={k: a[k] for k in ['water_text_evidence', 'scene_context', 'rationale', 'source_flag']},
            creator=dict(text=p['artist']['text'], raw_html=p['artist']['html'],
                         recorded_links=p['artist']['links'], original_keys=p['artist']['original_author_keys'],
                         key_structure=key_mode(p), original_parseable=p['artist']['original_parseable'],
                         uploader_not_assumed_author=p['artist']['original_upload_user_not_author_inference'],
                         independent_identity_status='not_completed'),
            source=dict(commons_page=p['provenance']['source_page'],
                        page_revision=p['provenance']['page_revision'],
                        credit_text=p['credit']['text'], raw_credit_html=p['credit']['html'],
                        recorded_credit_links=p['credit']['links'],
                        original_source_keys=p['credit']['original_source_keys'],
                        credit_structure=source_mode(p),
                        linked_metadata_entrypoints_for_future_review=no_pixel_links,
                        independent_source_status='not_completed'),
            license=p['license'],
            permission_field_present=bool(p['all_extmetadata'].get('Permission', {}).get('value')),
            file_identity={k: p['provenance'][k] for k in ['original_sha1', 'original_version_timestamp', 'mime', 'width', 'height']},
            evidence=dict(raw_response_path=str(raw_path), raw_response_sha256=rh,
                          json_pointer=f"/query/pages/[pageid={p['pageid']}]/imageinfo/0/extmetadata",
                          field_sources={k: p['all_extmetadata'].get(k, {}).get('source')
                                         for k in ['Artist', 'Credit', 'ImageDescription', 'LicenseShortName', 'LicenseUrl', 'Permission', 'Attribution']},
                          packet_path=str(REVIEW / 'review_input.json'), packet_row_ordinal=p['ordinal']),
            original_automatic=p['original_automatic'],
            readiness_notes=risks,
            pixel_status='not_accessed', model_exposure='none', usable_sample_verdict=None,
        )
        rows.append(r)
    E = [r for r in rows if r['root_annotation']['water_text_evidence'] == 'E']
    assert len(E) == 28
    counts, ecounts = summarize(rows), summarize(E)
    registry_summary = {k: registry[k] for k in [
        'old_record_count', 'downloaded_count', 'reserved_cluster_count', 'reserved_member_page_count',
        'unresolved_old_author_count', 'old_unavailable_pixels', 'pending_alias_preserved']}
    registry_summary.update(blocked_key_count=len(blocked), blocked_pageid_count=len(registry['blocked_pageids']),
                            blocked_title_count=len(registry['blocked_titles']))
    result = dict(
        schema='water_source_field_inventory_v1', created_utc=datetime.now(timezone.utc).isoformat(),
        status='complete_read_only_inventory_not_source_or_pixel_acceptance',
        source_hashes=before, rows=rows, all79=counts, explicit_E28=ecounts,
        E28_ordinals=[r['ordinal'] for r in E],
        old_exclusion_registry=dict(path=str(BASE / 'exclusion_registry.json'), sha256=sha(BASE / 'exclusion_registry.json'),
                                    summary=registry_summary, frozen_input_snapshots=snapshots),
        original_automatic_count_unchanged=79,
        collector_is_not_independent_source_reviewer=True,
        new_network_requests=0, pixel_reads=0, image_downloads=0, model_calls=0,
        new_usable_samples_declared=0,
        all_links_are_recorded_candidates_not_live_verified=True,
    )
    out_json = HERE / '水来源核查简表.json'
    write(out_json, result)
    lines = [
        '# 水来源可用字段与核查入口', '',
        '本文件盘点既有 79 个元数据簇代表，并为下一步来源核查提供入口。没有联网、读取像素、调用模型或改动原 v1/v2 数据。原采集者负责本次整理，不能自称完成独立身份或许可评审。新增可用样本仍为 **0**。', '',
        f"79 条完整机器表见[水来源核查简表.json]({out_json})；ID、顺序、簇、逐条全文、原字段、根代理注释、核查链接和原始响应哈希均保留。", '',
        '## 字段是否存在，与事实是否成立分开', '',
        '| 字段或声明 | 全部 79 | E28 | 能说明的范围 |',
        '|---|---:|---:|---|',
        '| Artist 文本、Credit 文本 | 79 | 28 | 页面提供署名/来源文字；不是已核实摄影者 |',
        '| Artist 中有链接 | 73 | 27 | 可继续核对作者页；链接有时只是机构账号、上传者或 Wikidata |',
        '| Commons 来源页、页面版本、原 SHA1、许可 URL | 79 | 28 | 可绑定所读页面/文件版本和许可声明 |',
        '| Credit 中有链接 | 29 | 6 | 可核对原作或转存入口，不能保证链接仍可用 |',
        '| 有原作键（本框均为 Flickr 照片键） | 26 | 5 | 有作品标识；缺键不等于缺来源，键存在也不确认权属 |',
        '| 独立 Attribution 字段非空 | 2 | 1 | 其余条目仍可用 Artist、标题、Credit、许可链接整理署名，不能当作无需署名 |',
        '| Permission 字段非空 | 9 | 3 | 有额外许可/历史审核文字；空值不直接判失败，非空也不代替验证 |',
        '| Restrictions 字段非空 | 0 | 0 | 只能说此字段未填限制，不能推出没有限制 |', '',
        '作者键的互斥结构为 Flickr 作者键 24、Commons 用户键 45、仅文本或其他键 10；E28 对应 4、21、3。来源文字为 Own work 47、Flickr 原作键 26、其他来源 6；E28 为 22、5、1。这些统计按原键进行，保留原解析误差。', '',
        '许可名称按原字段统计：' + '；'.join(f'{k}：{v}' for k, v in counts['license_names'].items()) + '。许可白名单判断只验证名称/URL结构，不证明声明者有授权权利。本材料属于数据来源准备，不作法律结论。', '',
        '## E28 的具体入口', '',
        'E 是根代理事后给出的“标题/描述/标签明确提到水”标签，不能倒用为原自动资格或视觉通过。E28 中 D=普通水描述 8、C=餐饮共现 8、P=实验或布景 11、?=不明 1。类别 D 也没有证明透明杯、真实摄影、无文字和实际 224 裁剪合格。', '',
        '| 原序号 / 来源页 | 场景 | 原 Artist；来源结构 | 原许可 | 水证据与待核问题 |',
        '|---|---|---|---|---|',
    ]
    source_labels = {'own_work_claim': 'Own work 声明', 'flickr_work_key': 'Flickr 原作', 'other_credit': '其他转存/来源'}
    for r in E:
        lines.append('| ' + ' | '.join([
            f"{r['ordinal']} [{cell(r['title'])}]({r['source']['commons_page']})",
            r['root_annotation']['scene_context'],
            cell(r['creator']['text']) + '；' + source_labels[r['source']['credit_structure']],
            f"[{r['license']['name']}]({r['license']['url']})",
            cell(r['root_annotation']['rationale']),
        ]) + ' |')
    lines += [
        '',
        '每行引用的原文和页面入口来自冻结元数据；本次没有确认当前网页。机器表的 `evidence.raw_response_path`、SHA256、pageid 和字段路径可定位原 API 字节。`recorded_credit_links` 保留所有原链接，`linked_metadata_entrypoints_for_future_review` 另排除明显像素/归档像素 URL，防止来源核查误触图像。', '',
        '需重点分辨：第 22 项 Dave Matos 只有作者文本，但已有 Flickr 作品 URL，核对作品署名后再解析账号别名；第 24 项作者链指向 Wikidata，仍需将自作声明与摄影者对应；第 27 项为 500px/Archive Team 转存，原图归档链接不是元数据入口。全 79 中第 34 项为机构账号 SuperJet International，第 44 项机构 FinnishGovernment 与摄影者 Lauri Heikkinen 分属两个层级；第 74 项 Artist 仅称英文维基上传者 Pixel23，原 `commons-user`/`parseable=true` 不足以证明摄影作者。不得据这些键直接宣布独立作者数。', '',
        '## 如何继续读取旧排除，不触碰旧像素', '',
        f"优先读取冻结的 [v2 exclusion_registry.json]({BASE / 'exclusion_registry.json'})，而不是只检查 79 行当前 `blocked=false`。注册表记有旧候选 101、已下载 60、原预约 144 簇及其中 190 个成员页；最终排除键 1,180、页面 ID 209、标题 309。各项会重叠，不能相加为独立人数或图片数。", '',
        f"[freeze.json]({BASE / 'freeze.json'}) 的 `snapshots` 给出以下原输入的不可变副本及 SHA256；机器表亦完整抄录路径。无需打开这些清单中列出的像素路径：", '',
        '- `old_exclusion_registry.json`：保留旧 101 的全部身份/原作/标题/SHA1，以及作者未明项。旧 90 有像素、11 不可得，不能只排除可解码的 90。',
        '- v1 `manifest.json`、`author_groups.json`、`provisional_allocations.json`、`candidate_review_order.json`：恢复 800 元数据中的连接关系与 144 原预约簇，排除所有预约成员，而非只排除代表。原确认 48 的身份继续保留排除；本次不读其像素。',
        '- 三批 `download_manifest.json`：001 的 24、002 的 19、003 的 17，共 60；成功、格式隔离、视觉失败或身份待核均视为已接触候选，不能因未入可用集重新当独立新来源。',
        '- `author_resolution_001/resolution_supplement.json`：保留待核别名和已观察重定向。原注册表仍有 3 个作者未明项、`pending_alias_preserved=true`，不能因未命中解析键视为已证明独立。', '',
        '本次重新核对 79 行的已知簇键与该注册表均无交集；这是既有键范围内的检查。下一步查明新的个人姓名、账号别名或作品 ID 后，应在新目录维护补充关联表，向 v1 全部相关连通簇传播排除；如任一成员命中旧/预约键，则整新簇暂排除。不得回写旧注册表使历史判断看似从未出错，也不得把机构账号直接折算成一个已核实个人摄影者。', '',
        '## 建议的有界执行顺序（候选，尚未冻结或执行）', '',
        '1. **先完成 E28 的来源审查。** 在独立新计划中固定这 28 个 ID 和原顺序，逐条记录摄影者、账号、作品标识、许可授予链、署名文本及旧簇关联。每条最多 1 个 Commons 元数据/页面入口和 1 个已记录的原作或作者入口；E28 涉及的 7 种许可 URL 共享读取各至多 1 次，合计最多 63 次只读请求，不补搜新候选、不因前几条成功而截断。预算不够澄清的项目记 pending；链接失效或需要额外证据不升级为通过。原页面的 Permission/Flickr 历史审核声明可作为证据的一部分，需保留原日期和它能支持的范围。请求方案另固定串行节流、超时、服务限制与停止规则，禁止自动加载页面图片。',
        '2. **来源判断与场景判断分别记账。** 可用状态建议为 `source-supported / contradicted / pending`，旁列 `creator / work / license / old-cluster-check` 的证据和剩余疑问。Own work 本身是来源声明，不是必须拒绝；但账号关系、摄影署名、许可和旧簇排除仍须一致。许可页介绍 CC 条款不证明该作品确实按此授权。E 中的实验、添加物和混合餐饮条目不会因来源核实而自动满足普通饮水场景。',
        '3. **随后只开一个小像素流程框。** 一个保守可执行候选是：以既有 E∩D 的 8 个代表为上限，先完成上述来源核查和新增关联归并；来源支持且无旧排除交集者按原簇排序，前至多 4 个固定为流程试用，其余保留、暂不取像素。若支持者不足 4，则按实际数量执行，不从 U/C/P 补位，不凑 80。冻结名单、原 SHA1/版本、预处理、格式隔离、近重复、匿名双评与停止条件后，才由另一个任务下载固定候选。水与冰等边界在看像素前明确，并沿用旧真实照片/普通杯中清水/混合资源/文字水印标准，不因本批产出调松。',
        '4. **流程像素不充当确认数据。** 流程样本用于检查资料链与图像处理是否可行，不能同时作为模型选择后的独立确认集；其余保留样本在另一个独立冻结方案之前不看像素或模型成绩。最多几个水来源不支持 40 张确认集或全面跨来源泛化。最终确认的任务、其他资源来源、独立单位和估计精度需要单独固定。若小流程仍不足，报告实际产出与失败原因，另论新来源方案，不能改判既有失败或挪用 v1 确认 48。', '',
        '原 80/20/40/10/10 只是未执行的规模建议。当前阻碍是作者/原作/许可及视觉适用性的证据不足，不是差 1 条元数据。上面的有限核查和流程试用可不依赖凑数推进；它仍不保证得到可用照片或论文级独立确认。',
    ]
    doc = HERE / '水来源可用字段与核查入口.md'
    doc.write_text('\n'.join(lines) + '\n')
    assert all(sha(Path(p)) == h for p, h in before.items())
    receipt = dict(status='passed_metadata_join_and_hash_checks_only', created_utc=datetime.now(timezone.utc).isoformat(),
                   count79=len(rows), countE28=len(E), all_original_ids_order_preserved=True,
                   raw_metadata_response_count=len(raw_cache),
                   raw_extmetadata_join_exact=True, known_exclusion_intersections=0,
                   original_inputs_unchanged=True, source_hashes=before,
                   output_sha256={str(p): sha(p) for p in [out_json, doc, Path(__file__)]},
                   new_network_requests=0, pixel_reads=0, model_calls=0, new_usable_samples_declared=0)
    write(HERE / '水来源字段盘点_QA.json', receipt)
    print(json.dumps({k: receipt[k] for k in ['status','count79','countE28','raw_metadata_response_count','original_inputs_unchanged']}, ensure_ascii=False))


if __name__ == '__main__':
    main()

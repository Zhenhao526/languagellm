"""Offline bookkeeping only. Does not import imaging/model/network libraries."""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json

HERE = Path(__file__).resolve().parent
FRAME = HERE.parent
ROOT = FRAME.parents[1]
SNAP = HERE / 'finalization_snapshots'
SNAP.mkdir(exist_ok=True)

def read(path):
    return json.loads(path.read_text())

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

snapshots = []
for source, name in [
    (HERE / 'README.md', 'README_before_final_curation.md'),
    (HERE / 'collection_status.json', 'collection_status_before_final_curation.json'),
    (HERE / 'collection_delivery_qa.json', 'collection_delivery_qa_before_final_curation.json'),
    (FRAME / 'README.md', 'visual_confirmation_v1_README_before_workflow003.md'),
]:
    dest = SNAP / name
    if not dest.exists():
        dest.write_bytes(source.read_bytes())
    snapshots.append({'original': str(source), 'snapshot': str(dest), 'sha256': sha(dest)})
write(SNAP / 'receipt.json', {'created_utc': datetime.now(timezone.utc).isoformat(),
    'note': 'Original bytes retained before final README updates; collection-stage JSON files remain unchanged.',
    'snapshots': snapshots})

previous = read(FRAME / 'pixel_workflow_002/curation_status_20260916.json')
identity = read(HERE / 'identity_license_review_20260916.json')
visual = read(HERE / 'review_round1/visual_merge.json')
technical = read(HERE / 'independent_collection_qa.json')
selection = read(HERE / 'selection_manifest.json')
queues = read(FRAME / 'candidate_review_order.json')['reserved_queues']
usable = [r for r in identity['records'] if r['provisionally_workflow_usable_with_limits']]
pending = [r for r in identity['records'] if not r['provisionally_workflow_usable_with_limits']]
assert len(usable) == 4 and len(pending) == 1
assert len(identity['records']) == visual['dual_pass'] == 5
assert {r['blind_id'] for r in identity['records']} == {r['id'] for r in visual['records'] if r['dual_visual_pass']}
added = Counter(r['stratum'] for r in usable)
counts = {k: previous['cumulative']['counts'][k] + added[k] for k in ['apple', 'banana', 'orange', 'water']}
target = dict(apple=4, banana=4, orange=4, water=12)
gaps = {k: target[k] - counts[k] for k in target}
remaining = {k: previous['remaining_unconsumed_reserved_counts'][k] - selection['quotas'][k] for k in target}
assert counts == dict(apple=3, banana=3, orange=3, water=2)
assert remaining == dict(apple=6, banana=7, orange=7, water=16)

# Verify exact same-stratum queue continuation without exposing unconsumed pixels.
for k in target:
    old_consumed = len(queues['reserve'][k]) - previous['remaining_unconsumed_reserved_counts'][k]
    expected = [r['id'] for r in queues['reserve'][k][old_consumed:old_consumed + selection['quotas'][k]]]
    actual = [r['metadata']['id'] for r in selection['rows'] if r['metadata']['stratum'] == k]
    assert set(expected) == set(actual)

sources = [
    FRAME / 'pixel_workflow_001/curation_status_20260916.json',
    FRAME / 'pixel_workflow_002/curation_status_20260916.json',
    FRAME / 'candidate_review_order.json', HERE / 'freeze.json',
    HERE / 'selection_manifest.json', HERE / 'download_manifest.json',
    HERE / 'collection_status.json', HERE / 'collection_delivery_qa.json',
    HERE / 'independent_collection_qa.json', HERE / 'identity_license_review_20260916.json',
    HERE / 'review_round1/review_a.json', HERE / 'review_round1/review_b.json',
    HERE / 'review_round1/visual_merge.json', HERE / 'review_round1/packet.json',
    HERE / 'author_resolution_001/resolution_supplement.json',
]
status = {
    'status': 'fixed_third_batch_complete_workflow_incomplete',
    'updated_utc': datetime.now(timezone.utc).isoformat(), 'updated_local_date': '2026-09-16',
    'batch': {'planned_reserve_candidates': 17, 'originals_downloaded': 17,
        'original_http_requests': 17, 'http200': 17, 'http429': 0, 'retries': 0,
        'ordinary_candidates': 17, 'mpo_quarantined': 0, 'visual_reviewed': 17,
        'individual_ai_reviews': 34, 'dual_visual_pass': 5,
        'pass_disagreements_excluded': 0, 'both_not_passed': 12,
        'source_license_metadata_supported': 5, 'added_usable_for_limited_workflow': 4,
        'source_medium_pending': 1, 'added_counts': dict(added),
        'prior_author_resolution_requests': 2, 'prior_author_usable_increment': 0,
        'identity_review_additional_network_requests': 0,
        'model_calls': 0, 'confirmation_pixel_requests': 0, 'new_metadata_candidates': 0},
    'cumulative': {'workflow_originals_downloaded': 60, 'ordinary_format_originals': 57,
        'mpo_quarantined': 3, 'ordinary_not_dual_visual_pass': 43,
        'ordinary_dual_visual_pass': 14,
        'source_license_metadata_supported_after_dual_visual_and_format_checks': 14,
        'usable_for_limited_workflow': sum(counts.values()), 'workflow_target': 24,
        'workflow_gap': sum(gaps.values()), 'counts': counts, 'gaps': gaps,
        'resource_semantics_pending': previous['cumulative']['resource_semantics_pending'],
        'author_identity_pending': previous['cumulative']['author_identity_pending'],
        'source_medium_pending': [r['blind_id'] for r in pending],
        'total_pending': 3,
        'usable_ids': previous['cumulative']['usable_ids'] + [r['blind_id'] for r in usable]},
    'collector_exit_code': 0, 'process_pid': 45508, 'process_session_id': 91720,
    'all_network_collection_finished': True, 'reserve_original_assignments_unchanged': True,
    'reserve_consumed_this_batch': selection['quotas'],
    'remaining_unconsumed_reserved_counts': remaining,
    'remaining_unconsumed_reserved_total': sum(remaining.values()),
    'automatic_next_batch': False, 'confirmation_pixels_accessed': 0,
    'confirmation_candidates': 48, 'confirmation_ready': False, 'model_calls': 0,
    'qa': {'independent_technical_checks': technical['checks'], 'failures': technical['failures'],
        'new_pair_comparisons': technical['comparison_count'], 'cumulative_pair_comparisons': 7170,
        'near_duplicate_flags': 0, 'old_available_pixels': 90, 'old_unavailable_pixels': 11,
        'minimum_request_gap_utc_seconds': technical['minimum_serial_utc_seconds'],
        'minimum_request_gap_monotonic_seconds': technical['minimum_serial_monotonic_seconds'],
        'actual_retry_after_waits_this_batch': [], 'retry_logic_was_selftested': True,
        'current_batch_request_timing_passed': True,
        'historical_metadata_and_batch001_timing_failures_retained': True,
        'sealed_visual_reviews_unchanged': True, 'collector_visual_views': 0},
    'limitations': [
        'Eleven accepted images support limited workflow development, not a complete24-image workflow or48-image confirmation dataset.',
        'Three old authors and11 unavailable old pixels limit source independence checks; known metadata keys and pixel-hash recall are not universal proofs.',
        'The old alias redirect target was resolved, but its relation to a historical creator remains unresolved within the two-request cap.',
        'One new water photograph/artwork source-medium chain remains pending, without changing its sealed visual judgment.',
        'Two AI reviews are not human validation; no absence from DINO pretraining is established.',
        'No confirmation pixels, model outcomes, new source frame or automatic fourth batch were used.',
        'The feasibility memorandum is post-curation and pre-model planning, not a preregistered prediction.'],
    'source_hashes': {str(p): sha(p) for p in sources},
}
write(HERE / 'curation_status_20260916.json', status)

readme = '''# 流程图第三批：完成记录

第三批固定 17 张已全部下载、完成双 AI 盲审及对应来源核查，新增 **4 张可作有限流程开发**，1 张来源媒介未决。第一批 5 张、第二批 2 张、第三批 4 张，累计 **11/24**，仍缺 13 张。水层仍只有 **2/12**；本批没有自动追加候选。

| 资源 | 累计可用 / 流程目标 | 尚缺 | 原预约未消费备用 |
|---|---:|---:|---:|
| 苹果 | 3 / 4 | 1 | 6 |
| 香蕉 | 3 / 4 | 1 | 7 |
| 橙子 | 3 / 4 | 1 | 7 |
| 水 | 2 / 12 | 10 | 16 |
| 合计 | 11 / 24 | 13 | 36 |

[当前整理状态](curation_status_20260916.json)按张保存计数与来源哈希；[来源与许可核查](identity_license_review_20260916.md)区分许可元数据、视觉判断与原作链。艺术水图的描述未明确摄影媒介及摄制归属，现有元数据没有作品/馆藏官方 URL，不增加检索，保留待核。艺术题名本身不证明它不是真实照片，也没有把它重标为视觉失败。旧香蕉资源语义未决与旧作者别名关系未决各 1 张仍未计入。

## 收集、评审与技术核查

原 144 预约中同层备用顺序固定选出苹果 3、香蕉 2、橙子 2、水 10。17 次原图请求全部 HTTP 200，17 张均为普通格式，无 MPO 隔离、无 429 或重试。原进程 session 91720 / PID 45508 已 exit 0。响应后串行间隔最短 UTC 10.004014 秒、monotonic 10.003931 秒；Retry-After 解析有离线自检，但本批没有真实重试事件。历史元数据及第一批等待偏差均保留，本批通过不追溯修复历史。

[737 项独立技术核查](independent_collection_qa.json)全部通过：原 SHA1、冻结输入/快照、640 派生 JPEG 字节重建、EXIF 校正全图、224 像素与旧函数张量、匿名隔离和请求双时钟。17 张与旧 90、第一批 24、第二批 19 张及本批内部共 2,397 对比较，近重复召回 0；三批累计 7,170 对。旧 101 中仍有 11 张像素不可得、3 位旧作者未完全消歧，不能把未召回重复等同完全独立。

[匿名包](review_round1/packet.json)包含 17 张及 5 张 sheet。两位 AI 评审独立封存 [A](review_round1/review_a.json) 和 [B](review_round1/review_b.json)，随后[机械合并](review_round1/visual_merge.json)：5 张双全 true，12 张双方均未通过，0 张通过/不通过分歧。仅这 5 张进入来源许可核查，4 张可用、1 张未决。评审不是人类验收；收集者没有读图或重写任何原判断。

三批累计 60 张原图 = 3 张 MPO 隔离 + 43 张普通图未双视觉通过 + 3 张来源/语义待核 + 11 张有限流程可用。确认 48 张像素仍未访问；没有 DINO 或其他实验模型调用、没有读取新图模型成绩。下载数、作者元数据簇数和流程可用数都不是独立模型确认样本数。

## 冻结与历史保留

[执行方案](冻结执行方案.md)、[冻结记录](freeze.json)与[前置状态](prerequisite_status.json)保存原参考绑定。选中清单 SHA256：`6272d0979d78a5eaf52c2617ce2d938f31f9026a5cd02906361b0577a15a3141`；首次请求前最终 freeze：`beb97fe6f6babb306162ef340224409f491bf2008368e91ccb5afe317b7585b8`。启动前发现 auditor 仍有上批 19 的计数常量，改为本批 17 后重新冻结；原源码及失败版本的冻结记录保留在[启动前修订凭证](prelaunch_correction_001/receipt.json)，没有改变候选、配额或已运行收集器。

旧待核作者的两次官方请求解析了重定向目标，但与旧作者明显名称关联尚未消歧，增量仍为 0，详见[补记](author_resolution_001/重定向解析补记.md)。没有改第二批原状态。第三批 [collection_status.json](collection_status.json) 与 [collection_delivery_qa.json](collection_delivery_qa.json)保留技术阶段“尚待评审”的原状态，不能误读为当前进度；当前以整理状态为准。更新前本批及图库总 README 原字节和技术阶段状态另存于[快照凭证](finalization_snapshots/receipt.json)。

## 后续边界

[后续图库可行性审查](后续图库可行性审查.md)给出剩余数量上界、低产出的经验情景和一个未来新水场景来源框候选。它写于流程观察后、模型评分前，尚未冻结或执行。当前不默认消费剩余 16 个水备用，不挪用确认图、不扩大元数据框，也不为配额放宽真实照片、资源可见、裁剪和署名要求。[最终整理 QA](curation_delivery_qa.json)仅验证当前记账和封存关系。
'''
(HERE / 'README.md').write_text(readme)

old_total = (SNAP / 'visual_confirmation_v1_README_before_workflow003.md').read_text()
historical = old_total[old_total.index('## 以下为元数据阶段历史记录'):]
new_top = '''# 独立视觉确认的元数据框 v1

**最新流程整理（2026-09-16）：第三批按原固定备用顺序下载 17 张，均为普通格式；5 张双 AI 视觉通过，来源核查后新增 4 张可用、1 张媒介/摄制归属待核。第一批 5 张、第二批 2 张、第三批 4 张，累计 11/24 张仅用于有限流程开发，仍缺 13 张（苹果 1、香蕉 1、橙子 1、水 10）。确认 48 图像素未访问，无新图实验模型调用。**

详见[第三批完成记录](pixel_workflow_003/README.md)、[当前整理状态](pixel_workflow_003/curation_status_20260916.json)和[最终整理 QA](pixel_workflow_003/curation_delivery_qa.json)。本批 17 次原图请求全 HTTP 200、737 项独立技术检查通过；三批取得 60 张原图，其中 11 张有限流程可用、3 张待核、3 张 MPO 隔离，其余 43 张普通图未双视觉通过。旧 11 张缺失像素、3 位未明作者、历史等待偏差和 AI 评审局限均保留。

水层三批累计从 32 张候选仅取得 2 张可用，仍缺 10 张。原预约剩余备用 36 张，其中水 16 张；[可行性审查](pixel_workflow_003/后续图库可行性审查.md)说明数量上界与经验局限，提出尚未冻结/执行的新水场景来源框候选。不会自动开第四批、挪用确认像素、放宽标准或按模型分数选图。

[第一批](pixel_workflow_001/README.md)和[第二批](pixel_workflow_002/README.md)保留原阶段状态。更新前本 README 原字节已[归档](pixel_workflow_003/finalization_snapshots/visual_confirmation_v1_README_before_workflow003.md)，旧报告与完成清单不回写。以下历史部分中的“未下载”仅指元数据阶段。

'''
(FRAME / 'README.md').write_text(new_top + historical)
print(json.dumps({'added': len(usable), 'cumulative': sum(counts.values()), 'gap': sum(gaps.values()),
    'remaining_reserve': remaining, 'curation_sha256': sha(HERE / 'curation_status_20260916.json')}, ensure_ascii=False))

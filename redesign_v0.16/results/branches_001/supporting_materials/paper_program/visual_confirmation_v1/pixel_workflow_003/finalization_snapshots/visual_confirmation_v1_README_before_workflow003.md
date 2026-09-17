# 独立视觉确认的元数据框 v1

**最新流程整理（2026-09-16）：第二批按原固定reserve顺序新增下载19张，累计43张；本批17张普通图完成双AI评审，2张MPO隔离，3张双视觉通过，最终新增2张可用、1张作者身份待核。累计仅7/24张可作有限流程开发，仍缺17张（苹果3、香蕉2、橙子2、水10）。第一批另1张资源语义未定继续保留。确认48图未访问、无新图模型调用，不自动扩大候选。**

详见[第二批完成记录](pixel_workflow_002/README.md)、[当前整理状态](pixel_workflow_002/curation_status_20260916.json)和[735项技术审计](pixel_workflow_002/independent_collection_qa.json)。当前19图全部SHA1/派生像素可复核，唯一429重试等待满足600+1秒；历史请求等待偏差没有改写。低产出是实际数据局限，不能把7张流程图称为完成独立视觉确认。

[第一批5张与未决项](pixel_workflow_001/README.md)保留为历史阶段记录；更新前本README原字节已[归档](pixel_workflow_002/finalization_snapshots/visual_confirmation_v1_README_before_workflow002.md)。

## 以下为元数据阶段历史记录

以下“未下载、尚待像素验收”等描述仅指当时元数据阶段范围，当前流程图状态以上节为准。固定800条元数据已收齐，后续图片阶段没有扩张元数据框。

下文保留此前元数据阶段范围：固定800条元数据已收齐，作者分组、候选顺序和完整性核查完成；该阶段没有下载图片，也没有运行模型。这些元数据本身不是合格确认图像。

本目录先固定来源、配额、排除及排序规则，再通过 Commons 官方 API 收集。[预先固定方案](prospective_sampling_plan.md)的 SHA256 为 `cf4de1868c61d2830b07093c806bebdae58f619ec6b812a84c3f5ec3f32866da`，[冻结记录](plan_freeze.json)早于首次候选请求。原候选范围、盐、顺序与配额均未改变，也未与其他视觉抽样框合并。

## 实际结果

| 项目 | 结果 |
|---|---|
| 官方分类快照 | 苹果 462、香蕉 555、橙子 641、水杯 237、水瓶 392 个直接文件成员；7 次分类请求完成，没有触及预定页数截断 |
| 合并抽样框 | 2,261 个规范化文件标题；分类间存在重叠，不能简单相加 |
| 旧候选排除 | 全部 101 条旧候选及 101 个原图 SHA1；标题 casefold 后为 100 个键，两条大小写不同标题保守合并，原记录及作者仍分别保留 |
| 固定元数据请求 | 800/800 条完成：苹果、香蕉、橙子各 160，水 320；799 个不同 Commons page ID，跨资源重复记录未纳入预约 |
| 元数据条件暂合格 | 655 条；仍待作者消歧、逐图许可复核、像素与近重复验收 |
| 排除或未决 | 145 条；理由可重叠：许可不符精确白名单 96、MIME 18、尺寸不足 15、作者未决 8、跨资源分类 12、已知旧作者 18、关联簇含旧作者／旧原作 18 |
| 作者／原作关联图 | 603 个元数据关联簇，其中 506 个含暂合格代表；并非已人工确认的 603 位真实作者 |
| 临时候选预约 | 144 个不同簇：流程验收 24、确认 48、备用 72；各层配额均填满。其余 362 个暂合格代表只列来源顺序，不扩大备用配额 |
| 真实 API 请求 | 历史累计 92 次：87 次成功（7 次分类、80 次文件元数据）、2 次参数错误、3 次历史 HTTP 429；全部原响应保留 |
| 本次续取 | 600 条、60 次请求全 HTTP 200；采用响应后至少 10 秒节流，最短请求起点间隔 11.124 秒，无新增 429 |
| 完整性核查 | [metadata_qa.json](metadata_qa.json)通过；包含原始响应哈希、旧资料哈希、顺序、配额、组间已知身份互斥及历史状态保留核验 |

655 条暂合格记录与旧库没有**已记录身份键或原图 SHA1**重合。作者键只是元数据代理，不证明别名均已消歧，也不排除旧未知作者或跨作者转载漏检。未做像素近重复比较，不能据此宣布视觉来源已经完全独立。

## 固定分组与后续顺序

[author_groups.json](author_groups.json)保存完整 603 簇及所有成员、身份键与排除状态。[candidate_review_order.json](candidate_review_order.json)按原方案列出全部 506 个暂合格簇代表，并提供逐集合、逐资源层的预约顺序。每簇的代表为暂合格成员中 `(frame_rank_hash, id)` 最小者，再按 `cluster_rank` 全局排序；[build_review_order.py](build_review_order.py)复算现有分配一致性，没有重新抽样。

| 元数据预约 | 苹果 | 香蕉 | 橙子 | 水 | 合计 |
|---|---:|---:|---:|---:|---:|
| 流程验收 | 4 | 4 | 4 | 12 | 24 |
| 确认 | 8 | 8 | 8 | 24 | 48 |
| 备用 | 12 | 12 | 12 | 36 | 72 |

后续排除或合并作者／原作簇须逐项留痕；替补只能取同层已固定备用顺序，不能动用其余未分配候选扩充预算。当前 144 个预约均标为像素、近重复和人工署名复核未执行；配额填满只表示元数据预约完成。

## 文件入口

- [sampling_frame.json](sampling_frame.json)：全部分类候选、分页来源及固定 800 条请求顺序；续取前后哈希一致。
- [candidate_status_manifest.json](candidate_status_manifest.json)：800 条逐项取得状态；[manifest.json](manifest.json)：完整文件、作者原始 HTML、许可 URL、原作 SHA1、版本时间、页面 revision 和 API 响应链。
- [license_chain.tsv](license_chain.tsv)：800 条署名／许可／来源索引，不能视作已批准发布素材清单。
- [old_exclusion_registry.json](old_exclusion_registry.json)：全部旧 101 条及旧源文件哈希；[author_groups.json](author_groups.json)：完整元数据分组；[provisional_allocations.json](provisional_allocations.json)：144 条预约记录。
- [candidate_review_order.json](candidate_review_order.json)：固定预约及备用顺序，含未分配代表的来源索引。
- [requests.jsonl](requests.jsonl)、[raw](raw)：所有请求、错误和原始响应；没有访问任何图像或缩略图 URL。
- [status.json](status.json)、[metadata_qa.json](metadata_qa.json)：当前完成状态与核查结果；[resume_001.json](resume_001.json)：发出续取请求前的冷却、节流、代码修订记录。

## 故障、续取与历史状态

首次执行停在 200/800。此前本地断言误以为 101 条旧标题 casefold 后仍有 101 个键，修正后保留全部旧记录，未缩小排除范围；见[开发记录](collection_development_log.json)。文件元数据请求曾将 `rvlimit=1` 与多标题组合，API 返回参数错误；去除此参数后继续，见[参数修订记录](api_parameter_fix.json)。两次错误响应及当时源码保留。

第一次执行共遇 3 次 HTTP 429，最后一次有限重试仍失败后停止。续取前发现历史代码对响应头大小写敏感：一次服务端 `retry-after=34` 被误用为默认 30 秒，下一请求起点仅相隔 30.768991 秒，**未满足该次服务端等待要求**。原失败请求、响应和源码均保留；此偏差已写入续行记录与 QA，不能将历史全部请求称为合规。修订后按不区分大小写读取秒数或 HTTP-date，并通过[离线响应头回归检查](retry_header_qa.json)。

本次在最后历史 429 之后冷却 1,791.098 秒，于 `2026-09-15T14:21:47Z` 恢复同一剩余 600 条，`14:32:54Z` 完成。每批最多 10 条、串行响应后至少等待 10 秒；全部 60 次成功。没有轮换端点、IP、身份或代理，也没有改变候选范围。网络收集已结束，无须再次运行收集器。

续取前的 [README](resume_001/README.md)、[status](resume_001/status.json) 和 [metadata QA](resume_001/metadata_qa.json) 原字节已归档；[historical_status_receipt.json](historical_status_receipt.json)核验三者精确匹配 v0.11 完成清单当时记录的 SHA256。当前状态文件已推进到 800 条，不能再声称它们的动态哈希与旧完成清单相同。没有改写 v0.11 冻结报告或完成清单。[partial_author_groups.json](partial_author_groups.json)等部分产物仅作为 200 条阶段历史保留。

## 本地复核与后续边界

以下命令只读取已保存元数据并重建本目录索引／核查记录，不请求网络、图片或模型：

```sh
.venv/bin/python paper_program/visual_confirmation_v1/refresh_outputs.py
.venv/bin/python paper_program/visual_confirmation_v1/build_review_order.py
.venv/bin/python paper_program/visual_confirmation_v1/audit_metadata.py
```

下一阶段仍需固定像素检查实现、下载后的作者／许可链消歧、覆盖全部旧 101 图的近重复检查、盲化人眼资源及实际 224 裁剪验收，随后锁定最终像素清单和确认规则。72 张正式目标尚未验收；当前任务到元数据阶段结束。

共用工作区没有技术权限密封。确认隔离目前仅指未向模型暴露、未看模型成绩的流程约束；正式确认前仍须明确数据保管、开发访问、检查点、指标和一次揭盲程序。

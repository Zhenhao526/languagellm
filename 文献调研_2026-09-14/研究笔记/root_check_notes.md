# root 文献记录独立核查

核查日期：2026-09-14。范围：`数据/root.json` 中10条记录及已下载的8份PDF。已直接修正 `root.json`；未改其他文献组、主报告或原PDF文件。`root_downloads.json` 仍保留下载时的旧题名与阅读状态；合并总表时应以 **root.json 的当前文献字段为准，以 root_downloads.json 仅补下载字段**。

## 文件身份与阅读依据

| ID | PDF题名/作者核验 | 正文依据与状态 |
|---|---|---|
| R01 | 第1–2页与七位作者匹配；恢复题名中的冒号 | 主审已读分类、测量与研究缺口相关正文，`full_text` |
| R02 | 第1–2页与三位作者匹配；恢复冒号 | 主审已读建模前提和替代路径相关正文，`full_text` |
| R03 | 第1页与四位作者匹配；恢复题名标点 | 本次独立阅读第13–14页检索范围及第47–48页涌现交流框架，`full_text`；不是通读130余页 |
| R04 | 文件第1页为City机构库封面，第2页为论文首页；三位作者匹配 | 主审已读naming game、约定形成及集体偏好相关正文，`full_text` |
| R05 | 未下载；机构库与作者个人论文页题名/作者吻合 | 保持 `abstract`；未将机构库的文件列表当作已读取全文 |
| R06 | 第1–2页题名、六位作者匹配 | 颜色刺激获正文直接确认；主审已读关键结果，`full_text` |
| R07 | 首2页是整本会议录；目标题名和Bart de Boer在PDF第179页 | 已核目标完整三页，PDF第179–181页＝印刷第151–153页；`full_text`、`conference_abstract` |
| R08 | 未下载；官方条目及索引的论文首页核题名和两位作者 | 保持 `abstract`；另查看可检索正文结论片段，不冒充完整阅读 |
| R09 | 第1–2页题名、四位作者匹配 | 主审已读Results和Methods关键段落，`full_text` |
| R10 | 第1页题名、三位作者及ICLR workshop标识匹配 | 本次独立读第4–5页机制解释和结论，`full_text` |

这里的 `full_text` 表示结论取自论文正文，不表示逐页读完整篇。具体范围已写入各条 `read_scope`，避免把题名核对或摘要阅读写成正文审读。

## 会影响研究判断的更正

1. **R06确实研究颜色分类。** 正文第2页明确以Munsell色片取样，并在CIE LAB空间数值编码。没有将其他刺激误认作颜色。结果还区分了变化类型：中心权威对网络拓扑变化适应较好，对环境变化适应较差。不能只写成某种拓扑普遍提高适应力。[PLOS原文](https://doi.org/10.1371/journal.pone.0182490)

2. **R03的发表年不能代替综述截止时间。** [官方arXiv记录](https://arxiv.org/abs/2602.11583)的Comments说明TMLR 2026接收，故保留同行评审状态，并记载核验依据；本轮未另核OpenReview接收页。正文§4.1写覆盖2025年前文献，§4.4进一步写截止2024年8月。它适合提供共同框架，不能用于宣称2025–2026前沿已获系统覆盖。所列DOI为arXiv DOI。

3. **R07是三页会议摘要，下载件是完整会议录。** 正式题名已更正为 *Hysteresis in Language Emergence and Evolution*；删除题名中人为增加的“complete JCoLE proceedings”，将文件性质放进 `download_note`。DOI属于会议录整体。第180页展示双峰最大熵分布及MCMC产生滞后；作者明确说认知/社会机制解释和实证检验仍待完成。它已足以否定“首次提出语言形成具有滞后”这种宽泛新意，但不能视作已验证某一具体生态机制。[作者机构记录](https://researchportal.vub.be/en/publications/hysteresis-in-language-emergence-and-evolution/)

4. **R08不能直接当作“语言产生紧急/安静词类”的证据。** 官方可检索结论写明，这一区分由智能体行动产生，但尚未成为能被语言控制的区分。因此其证据更准确地涉及行动与交流的功能组织。已在限制字段注明。正式题名恢复中间破折号。[官方条目](https://escholarship.org/uc/item/6md4n8cw)

5. **R10提出解释机制，没有在该文中完成新的操纵实验。** 记忆限制和说听角色轮换位于“Potential reasons for the mismatch in results”及结论，属于待检验解释。已改写 `manipulations` 和 `limitation`，避免总表读者误以为这是本文确认的因果机制。[官方arXiv记录](https://arxiv.org/abs/2204.10590)

## 两篇未下载论文的替代源检查

- **R05：** [University of Warsaw机构库](https://repozytorium.uw.edu.pl/entities/publication/f18cb536-8bf3-49d9-878f-0cc783858dc7)公开记录列有 *Interaction history as a source.pdf*，932.55 KB，CC-BY。但访问页面超时，未取得经核对的PDF直链。作者[个人论文页](https://tomekkorbak.com/papers/)只链接出版商，没有新的开放PDF。停止寻找。卷期年2021与作者列表年2022并存；机构库的开放日期字段另写2021-02-28，不能据此推翻原有线上发表日，已保留差异说明。
- **R08：** eScholarship页面直连返回HTTP 403；搜索仅得到相同官方noSplash PDF及带查询参数的版本，没有新的经验证作者/机构直链。未取得本地全文，停止下载探测。

本次没有新增PDF，仍为8份。所有已下载文件均与目标文献匹配；R07需用文件内页码定位，不能把会议录封面认作单篇文章首页。

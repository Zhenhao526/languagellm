# 水图元数据复核材料

原自动 79 个簇代表已全部整理，保持原顺序和计数。整理者是原采集者；根代理另行逐条注释。均仅用既有元数据，没有联网、下载、读像素或模型调用，新增可用图片为 0。

- [逐条 packet](/Users/xia/Documents/ChatGPT/语言/paper_program/visual_confirmation_v2_water_review/packet.md)：79 项，每项含最多 400 字符描述、全部分类、作者/来源、许可、自动标记、初评风险和原响应哈希。
- [完整输入 JSON](/Users/xia/Documents/ChatGPT/语言/paper_program/visual_confirmation_v2_water_review/review_input.json)：所有全文与原始字段，无描述截断。
- [根代理独立元数据注释](/Users/xia/Documents/ChatGPT/语言/paper_program/visual_confirmation_v2_water_review/root_independent_review.json)：E=28、U=43、X=8，仅是事后描述标签，不是像素通过数量。
- [元数据复核报告](/Users/xia/Documents/ChatGPT/语言/paper_program/visual_confirmation_v2_water_review/元数据复核报告.md)：解释英文筛选、多语场景和作者身份的限制，尤其第 44、74、79 项。
- [输入冻结](/Users/xia/Documents/ChatGPT/语言/paper_program/visual_confirmation_v2_water_review/input_freeze.json)、[技术 QA](/Users/xia/Documents/ChatGPT/语言/paper_program/visual_confirmation_v2_water_review/packet_qa.json)、[材料哈希](/Users/xia/Documents/ChatGPT/语言/paper_program/visual_confirmation_v2_water_review/packet_receipt.json)：1,517 项资料一致性检查通过，原 v2 全部 77 文件未改。

复现材料生成：`PYTHONDONTWRITEBYTECODE=1 python3 paper_program/visual_confirmation_v2_water_review/prepare_packet.py`。脚本只写本目录，不读取或修改独立根代理注释；重跑前会验证原元数据清单不变。整理过程第 65、79 项描述措辞曾收紧，原注释字节及[首次修订记录](/Users/xia/Documents/ChatGPT/语言/paper_program/visual_confirmation_v2_water_review/collector_note_revision.json)、[第二次修订记录](/Users/xia/Documents/ChatGPT/语言/paper_program/visual_confirmation_v2_water_review/collector_note_revision_002.json)均保留。根代理对其第 61、79 项另有独立[澄清记录](/Users/xia/Documents/ChatGPT/语言/paper_program/visual_confirmation_v2_water_review/annotation_clarification.json)。

不自动补搜或开启像素阶段。下一阶段须先厘清更窄目标场景和可追溯来源的可行性；原 79 簇不能称 79 个已确认独立作者或合格水图。

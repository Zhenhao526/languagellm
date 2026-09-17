# 水图片可行性实验完成记录

固定8项来源核查、前4项像素验收均已完成。新增3张有限流程可用水图，1张因视觉歧义排除；未运行实验模型。后4项继续保留，未补位。

- [结果报告](/Users/xia/Documents/ChatGPT/语言/paper_program/water_source_followup_001/独立水图片可行性实验报告.md)
- [逐图机器结果与署名许可](/Users/xia/Documents/ChatGPT/语言/paper_program/water_source_followup_001/curation_result.json)
- [下一轮完整形成流程建议](/Users/xia/Documents/ChatGPT/语言/paper_program/water_source_followup_001/后续完整形成可行性.md)
- [像素独立技术复核](/Users/xia/Documents/ChatGPT/语言/paper_program/water_source_followup_001/pixel_001/independent_technical_qa.json)
- [全部本批文件清单](/Users/xia/Documents/ChatGPT/语言/paper_program/water_source_followup_001/artifact_manifest.json)

`collect_sources.py`与`collect_pixels.py`是本批已执行的收集入口，输出目录禁止覆盖。来源会话64878、像素会话97038均已取得exit0终态，不存在待续跑的收集进程。`finalize_batch.py`对封存的双视觉意见和技术、来源结果作机械合并。

无需重新联网或运行模型，可从项目根执行以下命令核对封存文件及其绑定输入：

```sh
redesign_v0.3/deployment/.venv/bin/python paper_program/water_source_followup_001/verify_batch.py
```

本目录的`index_history`保留本轮修改前的三个项目索引。原v1/v2数据和v26/v27实验结果未被本轮重新解释或覆盖。本批不是正式确认集，也没有得到新的语言形成指标。

# v0.22：布局保持与末帧可见对象

已完成全部24个私人行动头、48个既有通信方向的冻结评价，没有新训练。保持资源位置不变，主体先看完整布局，下一帧只再次看见食物或水。共同30布局上，通信中可见资源相对历史资源的准确率优势为32.687个百分点，四来源都发生类别优势交叉。

主要限制是新状态支持：旧18图的社会D仅2.344个百分点、双目标J为96.191%；原added/sealed合并12图的D为78.201个百分点、J为6.152%。私人头在这12图的J也只有45.627%，因此不能将失败归为通信独有，也不能将旧布局验收通过推广到全30。本轮定位既有协议的行为，不构成新符号形成实验。下一候选为私人经验范围与通信经验范围的分离，尚未执行。

- [完整研究报告](results/visibility_001/布局保持与末帧可见对象的固定协议研究报告.md)
- [固定执行方案](固定执行方案.md)、[前置科学审查](前置审查.md)、[近邻边界](近邻边界.md)、[结果解释及下一候选](结果解释.md)
- [完整原始输出及分析](results/visibility_001)、[独立分析](results/visibility_001/analysis.json)、[私人适用性检查](results/visibility_001/private_applicability.json)
- [执行审计](results/visibility_001/audit_execution.json)、[统计比较](results/visibility_001/comparison.json)、[图表核验](results/visibility_001/figure_qa.json)
- [开发检查](preflight_qa.json)、[解析指标自检](metric_selftest.json)、[上一轮封存完整性核查](prior_integrity_qa.json)
- [生产评价源码](run_probe.py)、[独立复核源码](analyze_probe.py)、[报告构建](build_report.py)、[封存程序](package_results.py)

## 实际执行范围

采用v0.20完整反传私人终点与原私人头，以及v0.21两种观察条件下的全部通信终点。四来源32101–32104，三次布局划分、双方向都嵌套于来源内。每条件固定30布局×16测试照片对＝480世界，原批大小256＋224；私人世界前向23,040，社会发送世界前向46,080。DINOv2-L、时间接口、私人头及所有通信参数均冻结，接收者仅见两枚离散token与原需求分支。

完整前向独立重放与10,883项执行核查通过；7,129项统计比较最大绝对误差3.331×10⁻¹⁶。审计复用原冻结时间模块及上一轮独立原子消息计算，并未重新执行v0.20/v0.21训练。全部世界和来源保留，没有按适用性筛选。评价计时约1.387秒，源码核查、独立复算、图表和文档另计。

无移动序列与12种新完整初态均超出原训练支持。可见、近期和重复曝光仍共变；private/social D的差不是通信的因果放大量。图片与任务仍为旧开发资料，49个完整消息可覆盖30布局，成功率或消息变化不能独自证明组合语言。

## 复现

从项目根目录运行，使用现有本地环境，输出必须选择尚不存在的新目录，避免覆盖封存结果：

```bash
redesign_v0.3/deployment/.venv/bin/python redesign_v0.22/run_probe.py --out redesign_v0.22/results/replay_001
redesign_v0.3/deployment/.venv/bin/python redesign_v0.22/analyze_probe.py --out redesign_v0.22/results/replay_001
```

Torch 2.14.0、NumPy 1.26.4、CPU单线程。独立分析在导入环境NumPy后使用v0.9已安装绘图依赖；不要把该目录整体放到前置PYTHONPATH，否则可能切换NumPy版本。需要原v0.4视觉缓存、v0.20/v0.21终点与封存记录，具体SHA见[evaluation_complete.json](results/visibility_001/evaluation_complete.json)。本轮无模型下载、DINO推理、行动抽样或优化器更新。

开发批使用独立来源99520；旧开发分析版本保存在对应analysis_history中。主结果、所有旧版本和失败记录保留，完成本轮不代表整体论文目标已经达到。

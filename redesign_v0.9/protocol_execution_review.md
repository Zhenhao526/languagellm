# v0.9 协议追踪执行审查

审查完成：2026-09-15T12:47:20.547393+00:00。只读审查，未修改训练器、追踪源、冻结方案或任何旧结果；没有新模型推理或参考重评分。

结论：未发现阻断执行或影响当前指标计算的实现错误。12个v0.8 mixed起点与36个适应终点均已完整覆盖，96个方向端点的已保存记录通过独立重算。

## 函数与分母

- 对照v0.8函数签名核对了natural_messages、choose_assignment、fragment_report、heldout_stitch_report、whole_message_recoding_reference的参数顺序。固定I视觉路径、隐藏需求、49码及零库存/历史与适应任务一致。
- 发送数组为30地图×16照片×2兼容源目标×2符号。bounds先验证兼容源目标相同并只取0；old/new/all分母分别384/96/480。按实际消息逐图重建自然J、接收可达C_R、发送最优任意解码C_S，两个照片阶段合计576个支持集记录均一致。
- 严格片段完整30图分母7680，old24配对分母4608，old供体→new6拼接分母3072。结构函数保留了两个兼容源目标的重复索引，比例未改变；不能把这些分母当独立主体重复。
- 每方向先枚举49码×2目标×720菜单（70560项）并验证物理地点等价。自然消息改变比例覆盖all30×16验证照片；解码改变比例覆盖49完整码，任一目标动作改变即计该码改变，两者分母不同。

## 起点、冻结端与配对参考

- 24个baseline方向的两阶段codebook与原v0.8记录完全相同（48份阶段codebook）；接收表一致，且baseline的三支持C_S与独立发送可区分性诊断逐项一致。
- sender_only全部24个接收表与baseline一致；receiver_only全部24个方向的calibration和validation发送消息均一致（48份消息网格）。这是固定探针上的函数核验，完整参数冻结另由独立训练审计验证。
- 96方向端点各100次完整码双射有效，直接片段/拼接观察率与参考函数的observed字段一致。24个source seed×split×direction随机源均不同；同源baseline及三学习臂的100条置换序列完全相同，符合预先规定的配对比较。9600是条件化参考评分数，不是独立学习样本数。
- 正式前固定manifest中的两份追踪源哈希与当前文件及归档一致；追踪结果列出的v0.8函数/模型源哈希也一致。

## 必须保留的解释边界

**固定的是old24供体地图集合，不是适应前的旧符号串。** 每个端点独立生成其当前old24自然消息；sender_only和both更新发送路径后，供体消息内容也会改变。该实现可用于比较“当前协议用旧地图消息成分重组新图”的能力，不能写成冻结旧词素的保留实验。

每端点仅使用old24的calibration照片选择符号位置映射，但new6已参加600步适应训练。终点自然J与old→new拼接均不是零样本泛化。人工供体选择仍使用实验者的地图信息，不能称为主体自主构造。

C_S在各支持集分别挑最优解码，尤其new6单独C_S不保证old24成绩保持。主报告应同时引用[发送可区分性诊断](/Users/xia/Documents/ChatGPT/语言/redesign_v0.9/发送可区分性诊断.md)中的训练总数约束上界；它也是分析者oracle，而不是实际可学习性保证。自然J与主9600世界终点评估使用不同照片口径，应并列解释，不能直接相减。

## 来源

[追踪源](/Users/xia/Documents/ChatGPT/语言/redesign_v0.9/analyze_adaptation_protocol.py)；SHA-256 `d8f92713db82955f26c4b6742b9edc40ee722e3af36ae0ab15f5dc12e1be9b16`。

[协议结果](/Users/xia/Documents/ChatGPT/语言/redesign_v0.9/results/adaptation_001/protocol_analysis.json)；SHA-256 `f69d2fbef43fc16c84e28a73d1d302642eef31a6a30b835f3ad8b06ca67e4b64`。

[冻结manifest](/Users/xia/Documents/ChatGPT/语言/redesign_v0.9/protocol_fixed_manifest.json)；[训练执行审计](/Users/xia/Documents/ChatGPT/语言/redesign_v0.9/results/adaptation_001/audit_execution.json)。

复核计数：`{'menu_tables': 96, 'bounds_supports_recounted': 576, 'reference_scores_verified': 9600, 'v8_baseline_phase_codebooks': 48, 'frozen_receiver_tables': 24, 'frozen_sender_phase_grids': 48}`。

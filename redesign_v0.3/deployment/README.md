# VPT 2x 本地部署记录

2026-09-14，已在这台 Mac 上部署官方 VPT 2x foundation 策略，并完成真实 Minecraft 录像输入的动作推理检查。**尚未接通 Minecraft 游戏环境；没有运行多人生态、社会学习或语言形成实验。**

## 模型与来源

| 项目 | 已核实内容 |
| --- | --- |
| 主体 | OpenAI Video PreTraining，标准 foundation-model-2x |
| 参数量 | 248,495,294，按实际加载的模型统计 |
| 官方源码 | https://github.com/openai/Video-Pre-Training |
| 固定提交 | `095519fbd4ee0e9281d19f19601e45629de9ac3f` |
| 权重文件 | `weights/foundation-model-2x.weights`，994,032,309 字节 |
| 权重 SHA-256 | `5b20bd5af5cf1efd322dad727264bbca8fa77c13f51fa5baf2c4e73b13baee43` |
| 配置 | 官方 `2x.model`；转换副本为 `weights/2x.config.json` |
| 权重加载 | `weights_only=True`；全部参数严格匹配，无缺失权重替换 |

下载来源：[官方权重](https://openaipublic.blob.core.windows.net/minecraft-rl/models/foundation-model-2x.weights)、[官方配置](https://openaipublic.blob.core.windows.net/minecraft-rl/models/2x.model)。

这是预训练的视觉行动模型，输出原有键鼠操作。它没有自然语言输入和语言训练目标，但数据筛选流程涉及 CLIP、关键词等，不能称为全部训练流程完全没有语言参与。其经验限定于 Minecraft，也不能直接等同于一般世界知识。能力依据及限制见[模型候选审查](/Users/xia/Documents/ChatGPT/语言/redesign_v0.3/模型候选审查.md)。

## 本机验证结果

机器为 Apple M5 Max、48 GiB 统一内存、40 核 GPU、macOS 26.6.2。独立环境采用 Python 3.12.14、PyTorch 2.14.0、NumPy 1.26.4。旧实验的虚拟环境未修改。

使用公开承包者录像，从第 100 帧起读取，转换为 RGB 128×128。每个后端先预热 4 帧，再连续测量 256 帧；float32、贪心动作选择、4 个 CPU 线程、不计算梯度。

| 后端 | 模型前向速度 | 单帧中位时间 | 单帧 p95 | 动作分布与对数概率 |
| --- | ---: | ---: | ---: | --- |
| CPU | 44.13 帧/秒 | 22.64 ms | 23.14 ms | 均为有限值 |
| MPS | 98.69 帧/秒 | 10.11 ms | 10.67 ms | 均为有限值 |

这段录像中，两后端 256 帧的贪心动作全部一致。这是一次回放检查，不能推广为所有输入下的数值等价，也不证明动作在闭环中有效。

计时只包含策略前向，不包括视频解码、预处理、游戏渲染、环境推进、通信或训练。以上不是游戏帧率，也不能据此估算四主体强化学习的成本。该进程累计峰值 RSS 约 3.26 GiB；MPS 结束时驱动分配约 1.06 GiB。两者口径不同，不能相加，也不是多人训练峰值。

完整逐帧动作与测量见[推理报告](/Users/xia/Documents/ChatGPT/语言/redesign_v0.3/deployment/inference_report.json)。早先 32 帧检查另存于 `inference_report_32frames.json`，以 256 帧报告为本次依据。

录像来源：[官方公开样本](https://openaipublic.blob.core.windows.net/minecraft-rl/data/10.0/cheeky-cornflower-setter-02e496ce4abb-20220421-092639.mp4)。本地文件为 `weights/official_minecraft_sample.mp4`，172,629,917 字节，SHA-256 为 `c3dfede32353f7c19a297284ba671382bfbe32ac654077f1d37fe5bd26a41cfe`；下载长度与发布方 Content-MD5 均已核对。

## 兼容处理与复现

官方 Git 工作区保持原样。`prepare_runtime.py` 复制 `lib/` 到独立的 `runtime/`，仅做两项兼容处理：

1. 将未使用的物品名称函数中的 MineRL 导入延迟到函数内部，录像推理无需安装整个游戏环境。
2. 将已移除的 `torch.has_cuda` 查询改为 `torch.cuda.is_available()`。

未替换模型结构、动作映射或张量计算。[兼容补丁](/Users/xia/Documents/ChatGPT/语言/redesign_v0.3/deployment/compatibility.patch)与[源码记录](/Users/xia/Documents/ChatGPT/语言/redesign_v0.3/deployment/source_provenance.json)保存了变更及文件哈希。配置读取仅允许 gym3 的三个必要类型；权重通过 PyTorch 的限制加载模式读取。

在当前工作区重新检查，无需重新下载：

```bash
cd /Users/xia/Documents/ChatGPT/语言/redesign_v0.3/deployment
.venv/bin/python prepare_runtime.py
.venv/bin/python inference_probe.py --frames 256 --device both
```

第二条命令会覆盖当前 `inference_report.json`，需要保留历史测量时应先另存。脚本只读取录像并输出动作，不训练模型。

全新部署时，先在对应目录准备官方仓库、权重、配置和录像，再创建独立环境：

```bash
git clone https://github.com/openai/Video-Pre-Training.git Video-Pre-Training
git -C Video-Pre-Training checkout 095519fbd4ee0e9281d19f19601e45629de9ac3f
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.lock.txt
```

四个输入文件名和来源见上文。依赖版本见[锁定列表](/Users/xia/Documents/ChatGPT/语言/redesign_v0.3/deployment/requirements.lock.txt)。单独使用 NumPy 1.26.4 是为了满足 gym3 的版本约束，不需要降级旧实验环境。

## 尚待验证

本机尚未配置 Java 与 Minecraft/MineRL 闭环。当前部署不包含真实游戏采集能力验证、同一共享世界中的多主体同步、信号板操作、资源后果学习或策略适配训练。VPT 仓库的推理通过，不意味着这些环节已经可用。

下一实施关口是单主体真实行动与原有技能保留，随后才是两主体共享世界。任何环境适配失败都需要单独记录，不能解释为语言形成失败。总体实验方案见[研究设计 v0.3](/Users/xia/Documents/ChatGPT/语言/redesign_v0.3/研究设计_v0.3_预训练主体与资源协作.md)。

# 预训练主体的真实环境校准

本阶段先检验官方 VPT 2x 在真实 Minecraft 中的基本行动与资源后果，再判断能否进入社会学习。首次执行日期为 2026-09-14。当前执行结果以本目录最终的实验报告及原始运行记录为准；脚本、静态 XML 检查或录像推理通过均不代表主体能力通过。

## 主体与环境

- 主体：既有官方 VPT 2x foundation 权重，248,495,294 个参数；本轮全部冻结，沿用官方随机动作采样。没有新增神经决策网络、语言规划器或通信模块。
- 策略进程：原来的 `deployment/.venv`，Python 3.12、PyTorch/MPS。
- 环境进程：独立 `.venv-env`，Python 3.11.15、NumPy 1.23.5、Gym 0.23.1；具体依赖见 `environment.requirements.lock.txt`。Python 3.10 下载中断后改用本机已有的 3.11，已实测导入和跨版本图像动作传输。
- 游戏：Minecraft 1.16.5，MineRL dev 提交 `cdeae668c2f334e3c9117adf651b5a94436b45f8`；MCP-Reborn 标签 `1.16.5-20210115`，提交 `1e71be5bd4c49bc4d6ab0ee559c31b298b7697a3`。
- Java：项目目录内的官方 Zulu JDK 8.0.504 x86，通过本机已有的 Rosetta 运行；不改系统默认 Java。下载 SHA-256 与发布方元数据一致。模型进程仍为原生 Apple Silicon/MPS。

## 信息与归因边界

`policy_client.py` 只把第一视角 RGB 画面传给 `policy_worker.py`。世界坐标、库存明细、饥饿、方块事件和初始化回执只用于评分，不传给策略。原模型没有文本任务接口，场景名称不是发给它的指令。

控制器分成三种：`script` 验证接口；`noop` 检查环境自动变化；`vpt` 检验冻结模型自主行为。脚本可以用评分坐标瞄准，但不会用于补充 VPT 动作。VPT 没有预测的 ESC、pickItem、swapHands 保持环境空动作默认值，不解释为模型输出。

原生世界中搭建了安全、有限的人工场地。木材来自原生木块的破坏和拾取；进食场景的面包在初始化时提供。该场地不检验自然觅食，也不等于已经实现资源社会。

## 为什么需要环境补丁

当前 MineRL 1.0 的 Java 后端未消费部分旧版 XML 初始化字段。另一个问题是 Python 原始 DrawingDecorator 会转义嵌套的绘图 XML。因此仅看到配置文件有树、有饥饿数值，不能证明实际环境如此。

`calibration_java_patch.py` 为 `Calib_` 单主体任务添加明确的初始化：服务端设置场地、位置、库存、健康、饥饿、时间和天气；逐点核对目标方块，并返回回执。随后使用原生破坏、移动、拾取和食用机制。运行器同时核对服务端回执与实际 reset 观测；不匹配即停止。此补丁没有实现通用 FlatWorldGenerator，也没有改模型或让环境代替主体行动。

macOS 图形兼容处理、环境初始化处理分开保存于 `macos_compatibility.patch` 与 `logs/native_init_patch/`。完整构建日志位于 `logs/`。游戏源代码与大型运行文件保留在本地 vendor/runtime 目录。

## 运行

从项目根目录执行，先完成真实接口正控，再运行主体。命令会创建带时间戳的新记录目录，不覆盖旧运行：

```bash
bash redesign_v0.3/calibration/with_environment.sh \
  redesign_v0.3/calibration/run_calibration.py \
  --controller script --scenario movement --seed 101 --steps 160
```

其他场景为 `wood`、`food_selected`、`food_select`；控制器可换成 `noop` 或 `vpt`。`--food` 指定原生饥饿值；进食场景不允许使用满饥饿来制造机械性的“不进食”对照。`--dry-run` 只验证 XML 与动作格式，不启动游戏，不能算闭环结果。

每次运行保存场景、实际 XML、软件版本、原始/实际动作、逐步状态、初末图片、视频、摘要和终止原因。失败同样保存。视频按 20 帧/秒组织，不等于实测墙钟速度；日志另存每步时间。开发性测试与保留种子复测的规则见[验收方案](/Users/xia/Documents/ChatGPT/语言/redesign_v0.3/calibration/验收方案.md)。

“能使用食物”只支持基本操作保留。自身需要敏感性还需配对需要状态；新资源规则的学习需要单独的体验、反转、保留及技能复测。上述门槛未通过时，不进入使用相应能力的社会学习任务。

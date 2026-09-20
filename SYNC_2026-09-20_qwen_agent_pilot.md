# Qwen 三上下文预实验同步记录

首轮平衡协议与测试已在提交 `848d72a09bb6a7954b46bd96ecaab873ccf66e00` 中提交。36轮预实验、审计脚本、结果报告、模型校验清单和结果哈希清单已推送至 `https://github.com/Zhenhao526/languagellm`。实验归档提交为 `fe4f9d1bfaeb7bab001686d3232ad7ca8c892cee`；推送后核对了 `origin/main`，远端当时与该提交一致。

归档路径为 `research_program/qwen_agent_pilot/`。结果 JSON 为33,662字节，SHA-256 `3b3415687d269cff439b2334e28f59028e1d10fffb71e670fb0fc334f3e0f3d2`；分析报告为3,424字节，SHA-256 `f56d99aa0ffa8ff791376333229fa1fa9ce0d579a8faf01d6823b62df29a4811`。`results/pilot_20260920_manifest.json` 保存了结果、分析和模型锁文件的哈希。测试、固定种子环境重建及逐轮联合结果复算均通过。

本批次的简要观察是：36/36消息符合符号格式，helper结构输出72/72有效，但联合成功1/36；全批次只有两种字符串，第二个区组没有任何一种含义实现跨发送者同码。单种子结果不支持语言涌现或环境因果结论。

## 本地模型与存储

权重保留在 `/Users/xia/Models/Qwen3.5-9B-8bit`，没有上传 Git。模型锁定 `mlx-community/Qwen3.5-9B-8bit` revision `16daa4818c54ce5f5436f929d52542eb65bbed9d`。下载最终通过 `hf-mirror.com` 的普通 Hub HTTP 后端完成；两个权重分片均按官方 Hub 元数据核对了字节数和 SHA-256，详见 `research_program/qwen_agent_pilot/model_lock.json`。虽使用镜像传输，文件内容已由官方版本哈希验证。

模型目录共42个文件、10,453,448,252字节（约9.7 GiB）；两个权重分片合计10,426,592,423字节。Hugging Face Hub 缓存中未发现第二份权重副本。失败的 Xet 下载留下的两个零字节 `.incomplete` 文件已删除；另外删除了两个预实验 Python `__pycache__` 目录，回收30,787字节。当前 Mac 检查到约1.1 TiB可用空间。模型服务已停止，推理环境和模型文件继续保留供后续实验使用。

详情见 `research_program/qwen_agent_pilot/experiment_log.md` 和 `LOCAL_CLEANUP_RECEIPT_2026-09-20_qwen_agent_pilot.json`。

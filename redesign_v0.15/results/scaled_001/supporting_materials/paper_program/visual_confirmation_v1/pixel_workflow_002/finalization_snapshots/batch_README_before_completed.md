# 流程图片补缺第二批

状态：冻结的 19 张原图正在按既定 reserve 顺序收集。当前进度与重试期限见 [live_status.json](live_status.json)；进程及复现命令见 [process_handle.json](process_handle.json)。收集者不做视觉判断，不运行模型，不访问确认图。此目录不会自动增加第 20 张候选。

本批配额为苹果 3、香蕉 3、橙子 3、水 10。来源和候选映射仅供数据保管与后续许可审计；**两位视觉评审只读取 `review_round1/packet.json` 指定的匿名图，不读取 selection / download_manifest / responses / input_snapshots / near_duplicate_flags。** 未终止时 packet 尚不存在。

下载前已固定 [冻结执行方案.md](冻结执行方案.md)、[freeze.json](freeze.json) 与源码 [collect_pixels.py](collect_pixels.py)。选中清单 SHA256 为 `98f29f3f3e122c4e1ecae5697c6910dc0b91fd767160eab2848b784ca6ad7df0`。HTTP 失败原字节按 50 MiB 上限归档；429/503 的重试遵守服务端期限并加 1 秒，以 UTC / monotonic 双时钟记录。403 或明确拒绝即停止，不变更入口。

离线 [audit_collection.py](audit_collection.py) 将在本批终止后独立重算原字节、640 派生 JPEG、224 裁剪、旧函数张量、匿名包来源隔离及请求时钟。旧像素仅覆盖 90 张（101 条中 11 条不可用），另比较第一批全部 24 张和本批内部。技术 QA 不能代替双视觉、作者许可或近重复关系判断。第一批的 Retry-After 时间记录限制保持原样。

# 第一轮照片刺激集

已固定 60 张公开许可的实物照片：30 张食物（苹果、香蕉、橙子各 10 张），30 张透明容器中的水。每类训练 22 张、测试 8 张。所选副本合计 6,035,904 字节。所有图片都从 Wikimedia Commons 获取；下载用途是本地模型实验。

`manifest.json` 是实验唯一正式图片清单。`images/` 还保留了未入选的候选照片，实验必须按清单读取，不能直接遍历目录。每张图片的本地路径、SHA256、Commons 原图 SHA1、作者、许可、原图 URL 和源页面均在清单中。类别和文件元数据仅供环境结算与离线分析，不得传给主体。模型只接收解码后的 RGB 像素。

`selection.json` 固定训练与测试的原图 ID。60 个原图 SHA1 全部不同，训练与测试没有原图或记录作者重叠，见 `verification.json`。这只说明本实验的分割，不保证视觉模型的预训练集没有出现过这些网络图片。

筛选时查看了候选联系表，排除了明显文字/水印、空杯或内容不明确的容器、游泳场景、背景文字和难以识别的资源。`contact_food_*.jpg`、`contact_water_*.jpg` 是最终入选图片的人工检查用联系表，带 ID/标题，不能作为主体输入。人工检查没有发现明显可读文字，不等同于严格 OCR 排查，也不能证明原图全部像素没有文字。

副本只做 EXIF 方向校正、RGB 转换、等比例缩放（最大边 640 像素）与 JPEG 保存，没有填充、文字抹除或语义编辑。缩略版本的分辨率与压缩率不同；原图来源信息保留以便后续重新处理。`ATTRIBUTION.md` 列出逐图署名与许可；照片保留各自许可，发布这些副本时也需要保留相应署名和许可条件。

素材发现使用了 Commons 的[苹果](https://commons.wikimedia.org/wiki/Category:Apples)、[香蕉](https://commons.wikimedia.org/wiki/Category:Bananas)、[橙子](https://commons.wikimedia.org/wiki/Category:Oranges)和[水杯照片](https://commons.wikimedia.org/wiki/Category:Glasses_of_water)分类。元数据来自 Wikimedia 官方 `imageinfo` API。当前 Commons 域名在本机解析失败，因此通过英文维基的官方 API 读取其共享 Commons 文件信息；`metadata/` 保留完整响应。

首轮共有 101 张候选记录。`candidates_downloaded.json` 包含全部下载/排除状态；少量文件受服务器 429 限速或不在本轮许可接受范围内，没有使用。图片数不是统计样本量，后续实验中独立训练种子才是评估群体学习变异的主要单位。

复用现成数据无需联网，直接读取清单即可。重新获取时在包含 Pillow 的 Python 环境执行 `python prepare_images.py`；人工复核完成后执行 `python prepare_images.py --finalize data/selection.json`。预先固定的清单应另存备份，避免未来 Commons 更新或下载失败导致静默改变刺激集。

这些照片足以做小规模视觉资源协作先导测试。它们仍然是经过筛选的图片类别，食物与水的颜色、容器和构图差异都可能帮助选择；表现好不能单独说明主体理解营养、水或真实世界中的资源后果。

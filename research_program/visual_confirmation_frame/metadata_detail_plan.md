# 全类别文件详情元数据抓取固定方案

2026-09-15，在详情API调用前固定。只延伸metadata_001已经穷尽的四个直接类别清单；不递归类别、不另加查询词、不下载原图／缩略图或使用模型。

四类成员为Apples 462、Bananas 555、Oranges 641、Glasses of water 237；类别成员次数1895，按Commons pageid去重1876个文件，保留全部类别归属。新清单按pageid升序固定，**不按大小写折叠标题合并文件**，也不先按许可、题名或旧素材命中筛掉请求。

旧metadata的92+9条记录有101个exact标题，旧规范索引100个；`File:An Orange.jpg`与`File:An orange.jpg`在casefold后碰撞。保留双方的原记录、SHA1、路径和规范化碰撞，不把100当作100个原作品。旧SHA1索引来自所有旧metadata imageinfo，以及candidates／manifest的original_sha1；不能只用正式60张。旧原始标题、规范标题、SHA1各自排除原因单列，分母不混用。

抓取仅使用[Commons官方imageinfo API](https://www.mediawiki.org/wiki/API:Imageinfo)，endpoint固定`https://commons.wikimedia.org/w/api.php`，按pageid每批25条（不超过50），单请求timeout=20秒，串行请求，每个批次只发一次；失败后记录该批未知，继续尚未请求的批次，无自动重试或覆盖。只请求最新文件版本`iilimit=1`，字段为url／size／mime／sha1／extmetadata／user／timestamp／canonicaltitle；extmetadata仅取Artist、Credit、Source、LicenseShortName、LicenseUrl、UsageTerms、AttributionRequired、Copyrighted、Restrictions、ImageDescription、DateTimeOriginal、ObjectName、Categories。不给iiurlwidth，不访问返回的图片URL。API字段user是上传者，不能替代摄影者。

details_001目录必须不存在。先保存源文件SHA、完整去重请求框、旧索引证据、代码与本方案快照，随后存每批实际URL、UTC开始／结束时间、HTTP状态／头、原始响应字节SHA及解析异常。API错误、网络异常、缺页／缺imageinfo保留；有错误则总状态partial，程序异常则failed，不假装完整。核对输入文件在执行结束时仍相同。

仅作初步可行性盘点，不选定84张：每资源目标84张（food为三个果类并集，water为水杯类）。先列全部文件、旧标题／旧SHA1命中、候选静态栅格MIME（JPEG／PNG／TIFF／WEBP）、许可字段和作者字段是否存在。许可标签初筛仅识别CC BY、CC BY-SA（1.0／2.0／2.5／3.0／4.0）、CC0／Public domain及其Creative Commons链接，其他标签单列；这不是法律或来源完整验收。元数据不能确认图像是照片、清水、单一资源或没有文字。

分组参照在数据返回前固定：①完全相同原文件SHA1；②非空且不在固定unknown／anonymous缺失标记内的Artist纯文本规范字段，及Artist字段中明确Commons User／Creator、Flickr摄影者路径；③Credit／Source中明确Flickr照片URL或Commons File文件链接。相同键用连通分量合并，每元数据分量最多1张。作者字段／链接有歧义，不能把字段分量数称为独立摄影者确认；同作者会保守合并不同系列，跨作者／改名／衍生作品仍可能漏合并。未能识别作者的文件不计入“具有作者字段的候选分量”，但保留其数量。系列还需要后续人工原始来源核验；本轮不由题名猜系列、也不视觉判断。

报告同时给未排旧作者与排除精确旧作者／来源键的两套数量，旧作者相同单独标记，不与旧文件SHA1等同。若water在这些初步约束下不足84，应报告当前框明显不足或余量不足；若数量达到84，也只能说元数据计数尚未排除可行性，不能保证通过许可、系列和像素验收。原文件SHA1表示文件字节身份，不能穷尽裁剪／重编码／同次拍摄衍生品。

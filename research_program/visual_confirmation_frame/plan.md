# 新视觉来源的类别清单盘点

2026-09-15。本次只收集公开文件名称／ID，不下载图片、不运行编码器或模型，不选定正式训练／测试照片。目的是检查独立图库是否有可用的候选来源；它不是已经完成的新视觉确认。

来源固定为Wikimedia Commons顶层类别：[Apples](https://commons.wikimedia.org/wiki/Category:Apples)、[Bananas](https://commons.wikimedia.org/wiki/Category:Bananas)、[Oranges](https://commons.wikimedia.org/wiki/Category:Oranges)、[Glasses of water](https://commons.wikimedia.org/wiki/Category:Glasses_of_water)。类别存在已通过官方网页核对。它们的分类可能包括艺术作品、混合场景和其他不合要求的媒体，不能把类别名当作已经通过视觉验收的标签。

使用官方categorymembers API，仅取直接file成员，不递归。每页最多500条，每类最多3页，保留 continuation；达到上限则标为截断，不宣称完整。每个URL仅请求一次，15秒超时，错误原样留档，其他类别继续。只归档响应和其中的pageid/ns/title/type字段。

将全部旧候选与旧metadata所列文件名作为排除索引，规范化Unicode、空格／下划线和大小写后比较，保存旧来源文件SHA。名称不重合仅表示未命中此名称索引；不是按原作SHA1、作者、系列或像素排重的证明。后续仍须先冻结抽样与分组规则、核来源许可、去重和盲法像素验收，再确定新图库。

输出metadata_001目录只新建一次，不覆盖。此阶段不按模型成绩筛选，不查看候选图像，也不把候选数当作有效独立照片数。记录源码／方案SHA、开始结束时间、实际请求、失败及每类是否穷尽。

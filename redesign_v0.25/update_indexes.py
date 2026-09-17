"""Record the new diagnostic outcome while preserving prior living-index bytes."""
import json,hashlib,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent
files=['README.md','paper_program/README.md','paper_program/证据与投稿门槛.md'];history=ROOT/'index_history';history.mkdir(exist_ok=True);backups=[]
for rel in files:
    src=PROJECT/rel;dst=history/rel;assert not dst.exists();dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    backups.append(dict(source=str(src),backup=str(dst),sha256=hashlib.sha256(dst.read_bytes()).hexdigest()))
def paragraph(prefix):
    return f'最新完成v0.25资源角色分别读入诊断：新增12次joint双主体训练，复用v24 mean/attention作历史参照。joint与mean初始49码概率/贪心完全相同，允许两资源读取权重随后分别学习。主old18 J由76.157%升至83.333%，差+7.176个百分点，四来源均正；old归一AUC差+19.898个百分点也四正。new12仍仅2.083%（mean1.042%，AT4.514%），来源两正一负一零。水／食物方向的人工重组由1.080%升至5.633%，四来源均正；食物／水方向2.045%升至2.932%，两正两负，不能择优称全面结构增强。三臂全程裁剪系数均为1；有效参数与读入优化路径仍有差异。见[研究报告]({prefix}redesign_v0.25/results/joint_001/资源角色分别读入与共同通信学习研究报告.md)与[复现索引]({prefix}redesign_v0.25/README.md)。熟悉协议学习得到改善，表达迁移未解决；该联合读入候选已完成，接下来收束此接口分支、统一已验证基线与信息权限，再为一个可反驳迁移问题准备独立材料确认，不继续扩大同图库超参数网格。'
old='保留两预测但无动态读取的联合输入仍只是待固定候选，独立图片确认及中心新颖性仍未齐备。'
new='当时的联合读入候选随后已在v0.25固定并完成；独立图片确认及中心新颖性仍未齐备。'
p=PROJECT/'README.md';s=p.read_text().replace('最新完成v0.24','此前完成v0.24',1).replace(old,new,1);s=s.replace('# 语言诞生研究\n\n','# 语言诞生研究\n\n'+paragraph('')+'\n\n',1);p.write_text(s)
p=PROJECT/'paper_program/README.md';s=p.read_text().replace('最新完成v0.24','此前完成v0.24',1).replace(old,new,1)
needle='目标进行中；一次实验成功、完整报告或审计通过不等于达到论文门槛，更不等于保证录用。';assert needle in s;s=s.replace(needle,needle+'\n\n'+paragraph('../'),1);p.write_text(s)
p=PROJECT/'paper_program/证据与投稿门槛.md';s=p.read_text().replace('更新至v0.24私人预测上的mean/attention结构基线及有限符号重组评估','更新至v0.25相同初始协议下的资源角色分别读入诊断',1)
s=s.replace('**最新结构基线证据（v0.24）：**','**最新联合读入证据（v0.25）：** '+paragraph('../')+'\n\n**此前结构基线证据（v0.24）：**',1).replace('最新完成v0.24','此前完成v0.24',1).replace(old,new,1)
s=s.replace('下一步先停止此超参数分支，明确熟悉任务的可学习性和必要的一个联合输入控制；独立图片和更多来源确认仍未完成。','联合输入控制现已在v0.25完成：从同一初始函数解除资源权重绑定，熟悉学习改善，new12仍仅2.083%。收束此接口诊断，不再把joint列为待执行；先统一已有基线及其信息权限，再固定独立材料确认，更多来源与论文中心贡献仍未完成。',1)
s=s.replace('v0.24结构基线也未获得稳定可迁移符号结构，不自动延长或搜索注意力训练。','v0.24结构基线未获得稳定可迁移符号结构；v0.25联合读入虽改善熟悉学习，迁移仍弱。两个诊断均已完成，不自动延长或搜索attention/joint训练。',1);p.write_text(s)
(history/'manifest.json').write_text(json.dumps(dict(backups=backups),ensure_ascii=False,indent=2)+'\n');print('3 living indexes updated with saved prior versions')

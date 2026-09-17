"""Add current outcomes to living indexes, saving each pre-edit version."""
import json,hashlib,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent
files=['README.md','paper_program/README.md','paper_program/证据与投稿门槛.md'];history=ROOT/'index_history';history.mkdir(exist_ok=True)
backup=[]
for name in files:
    src=PROJECT/name;dst=history/name;assert not dst.exists();dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    backup.append(dict(source=str(src),backup=str(dst),sha256=hashlib.sha256(dst.read_bytes()).hexdigest()))

def paragraph(prefix):
    return f'最新完成v0.24私人预测上的Ri式mean/attention任务内基线：24次新双主体训练，四继承来源。社会未见12组合J为1.042%/4.514%，attention−mean +3.472个百分点，三正一零；old18为76.157%/62.963%，共同30为46.111%/39.583%，不支持整体通信改善。两种预定符号槽位分配的重组结果方向不同，绝对成功率约1%–5%，不能择优宣称结构增强。私人行动已100%只限既有有限测试；多key保留与动态读取共变，额外角色预测并非纯视觉输入。所有训练和独立复核完成，见[研究报告]({prefix}redesign_v0.24/results/attention_001/私人预测动态读取与共同符号结构研究报告.md)与[复现索引]({prefix}redesign_v0.24/README.md)。本轮未获得稳定可迁移的共同符号结构，停止扩大该注意力超参数；保留两预测但无动态读取的联合输入仍只是待固定候选，独立图片确认及中心新颖性仍未齐备。'
p=PROJECT/'README.md';s=p.read_text().replace('最新完成[v0.23','此前完成[v0.23',1)
s=s.replace('下一候选优先复核直接近邻的结构先验基线，尚未实施；独立图片确认及论文中心贡献仍未齐备。','当时的结构基线候选现已在v0.24执行；独立图片确认及论文中心贡献仍未齐备。',1)
s=s.replace('# 语言诞生研究\n\n','# 语言诞生研究\n\n'+paragraph('')+'\n\n',1);p.write_text(s)
p=PROJECT/'paper_program/README.md';s=p.read_text().replace('本轮v0.23完成','此前v0.23完成',1)
s=s.replace('下一候选先对齐直接近邻的结构先验基线，未实施；更多来源与独立图片仍必需。','当时的结构基线候选现已在v0.24执行；更多来源与独立图片仍必需。',1)
needle='目标进行中；一次实验成功、完整报告或审计通过不等于达到论文门槛，更不等于保证录用。'
s=s.replace(needle,needle+'\n\n'+paragraph('../'),1);p.write_text(s)
p=PROJECT/'paper_program/证据与投稿门槛.md';s=p.read_text().replace('更新至v0.23私人状态经验与新共同符号表达迁移，以及待方法复核的结构先验基线候选','更新至v0.24私人预测上的mean/attention结构基线及有限符号重组评估',1)
s=s.replace('**最新私人经验与新通信证据（v0.23）：**','**最新结构基线证据（v0.24）：** '+paragraph('../')+'\n\n**此前私人经验与新通信证据（v0.23）：**',1)
old='下一候选优先核实Feng等2024及相关结构先验基线的实际输入/训练，在私人全30条件检验仅通信old18后的迁移；若获得额外对象槽或类别绑定，须明确额外支持，不声称同信息纯算法提升。该候选尚未实施，之后仍需独立图片和更多来源确认。'
new='该方法核查与任务内结构基线现已在v0.24完成：Feng2024并非拟议注意力架构，实际按Ri2023发送者改编。两个新臂读取同样的私人行动概率和角色标记，结果没有建立较强共同符号结构；多key保留与动态读取的作用未分开，不能把旧h接口作同信息纯算法对照。下一步先停止此超参数分支，明确熟悉任务的可学习性和必要的一个联合输入控制；独立图片和更多来源确认仍未完成。'
assert old in s;s=s.replace(old,new,1)
s=s.replace('尚未证明某一新增架构或训练方案必要。','v0.24结构基线也未获得稳定可迁移符号结构，不自动延长或搜索注意力训练。尚未证明某一新增架构或训练方案必要。',1);p.write_text(s)
(history/'manifest.json').write_text(json.dumps(dict(backups=backup),ensure_ascii=False,indent=2)+'\n');print('3 living indexes updated; prior bytes preserved')

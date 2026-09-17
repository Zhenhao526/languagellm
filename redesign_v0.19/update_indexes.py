"""Update living indexes, preserving the prior frozen copies."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent
receipt=json.loads((ROOT/'index_history/receipt.json').read_text())
for name,digest in receipt.items():assert hashlib.sha256((PROJECT/name).read_bytes()).hexdigest()==digest,name
report='redesign_v0.19/results/readout_001/冻结视觉接口的新私人行动学习研究报告.md'
root=PROJECT/'README.md';t=root.read_text();t=t.replace('最新完成[v0.18','此前完成[v0.18',1)
paragraph=f'最新完成[v0.19新私人行动头实验]({report})：两种冻结视觉接口 × 奖励/监督两信号，共96头，来自四个既有来源。仅行动成败奖励下，未训练组合贪心双成功为保留97.62%、重置并匹配幅度96.44%；主差+1.18个百分点，四来源均正。保留接口第100步优势17.44个百分点，过程AUC优势2.58个百分点；重置端仍能学会该私人任务。随机行动Q的排序相反，不能混称整体能力提升。本轮限制了将旧头失配或通信低分解释为缺乏基本资源知识的说法，没有训练新通信。见[v0.19复现索引](redesign_v0.19/README.md)和[最新近邻文献](paper_program/literature_capability_20260916/短近邻定位.md)；重训读出本身已有前例，下一候选是可行为验收的事件保持与更新能力。\n\n'
t=t.replace('# 语言诞生研究\n\n','# 语言诞生研究\n\n'+paragraph,1);root.write_text(t)
p=PROJECT/'paper_program/README.md';t=p.read_text();t=t.replace('本轮v0.18完成','此前v0.18完成',1)
paragraph2=f'本轮v0.19完成96个新私人行动头拟合，四来源、同初值/经验/预算。奖励条件sealed J为保留97.624%、重置匹配96.441%，主差+1.183个百分点，四来源均正；100步差17.437、AUC差2.580个百分点。重置接口同样支持较高的私人未见组合学习，旧头失配不能定义基本资源知识缺失。奖励终点Q排序相反（87.903%对90.118%），不按一个指标给完整能力排序。CE仅作有正确地点标签的参照，没有新共同通信或独立图像确认。见[完整报告](../{report})、[复现索引](../redesign_v0.19/README.md)和[解释](../redesign_v0.19/结果解释.md)。979,098项执行核查、47,520项原始复算、51,237项分析比较通过；这些保证执行可复算，不能替代新颖性。同期[五篇近邻方法复核](literature_capability_20260916/短近邻定位.md)指出冻结视觉后新通信、私人读出重训已有直接先例。\n\n'
marker='本轮v0.19'
index=t.index('此前v0.18完成');t=t[:index]+paragraph2+t[index:];p.write_text(t)
p=PROJECT/'paper_program/证据与投稿门槛.md';t=p.read_text();t=t.replace('更新至v0.18固定策略条件梯度方差结果、独立数学/执行核验及v2水场景79项元数据复核','更新至v0.19新私人行动头诊断、独立执行/统计核验与五篇近邻方法复核',1)
insert=f'**最新能力证据（v0.19）：** 相同新行动头只从old18的标量行动后果学习时，retained与reset_scaled的sealed J为97.624%和96.441%，差+1.183个百分点，四个继承来源均正；第100步差17.437、归一化AUC差2.580个百分点。两接口都可支持较高私人泛化，不能再由旧私人头失配或既有通信低分定义“缺少可学资源信息”。终点原生随机Q为87.903%和90.118%，方向相反；CE仍是明确有标签参照。本轮96头不构成96个独立来源，没有新共同通信。详见[主报告](../{report})、[独立审计](../redesign_v0.19/results/readout_001/audit_execution.json)与[结果解释](../redesign_v0.19/结果解释.md)。\n\n项目核心仍是：**哪些非语言能力足以支持共同符号形成，哪些能力改变形成过程和表达结构？** 当前基础同时可见资源定位已强，下一项应建立可验证的能力差，例如局部事件后的地点保持与更新；该环境和社会学习尚未执行。不能把读出重训、几何指标或增加任务难度本身当作中心创新。\n\n'
index=t.index('ICLR 2027 的官方审稿问题');t=t[:index]+insert+t[index:]
row=f'| [v0.19 新私人行动头](../{report}) | 96头仍来自4旧来源；奖励sealed J 97.624%/96.441%，主差+1.183个百分点、四来源同向；AUC差2.580个百分点；Q排序相反 | reset_scaled支持指定新头与奖励预算下的私人未见组合学习；私人准备加快读出学习并有较小终点优势 | 两接口等效、所有能力一致、通信差距完全由旧头失配造成、新语言或世界知识必要性 |\n'
marker='| [v0.18 固定策略条件方差]';index=t.index(marker);t=t[:index]+row+t[index:]
near='本轮[五篇近邻全文复核](literature_capability_20260916/短近邻定位.md)进一步找到直接重叠：Feng等AAAI 2024已比较视觉预训练后冻结并新训通信，还做表征读出及冻结发送者后的接收重训；Kouwenhoven等已有冻结DINOv2和对齐；CORAL涉及记忆世界模型与新控制器；IWCS 2025比较场景复杂度。因此v0.19是解释所必需的诊断，尚不是新方法贡献。后续若研究事件记忆，仍需相同信息和预算的明确干预、独立行为验收及新协议自主未见组合表现，不能只换任务名称。\n\n'
marker='较合适的论文中心问题是：';index=t.index(marker);t=t[:index]+near+t[index:]
t=t.replace('v0.13–v0.18进一步区分私人经验、数值条件、基线对应与局部方差；v0.18实例化经典控制变量关系，仍须建立超出近邻的预测及独立确认','v0.13–v0.19进一步区分私人经验、数值条件、基线对应、局部方差与新私人读出；本轮方法仍有直接近邻，须建立超出近邻的预测及独立确认',1)
t=t.replace('新旧用途形成原因仍需可反驳检验；','v0.19新奖励头显示两接口都能私人泛化，限制了目标信息缺失的解释；新旧用途形成原因仍需可反驳检验；',1)
t=t.replace('下一阶段候选应回到可区分的非语言能力及共同符号形成，须另立有界方案；当前未执行。','随后v0.19新私人读出诊断已完成：重置匹配端也可仅奖励学到较高私人泛化，保留端学得更快且贪心终点略优，随机Q排序相反。此后候选回到可区分的事件保持与更新能力，须另立有界时序环境和行为验收；新序列环境与社会训练当前未执行。',1)
p.write_text(t)
print('updated three living indexes from hashed backups')

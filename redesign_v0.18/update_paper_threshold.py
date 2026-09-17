"""Update only the mutable paper threshold index; preserve its previous bytes."""
from pathlib import Path
import hashlib,json
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent
path=PROJECT/'paper_program/证据与投稿门槛.md'
history=ROOT/'index_history/paper_program';history.mkdir(parents=True,exist_ok=True)
backup=history/path.name
assert not backup.exists(),'refuse to replace original snapshot'
original=path.read_bytes();backup.write_bytes(original)
text=original.decode()
def replace(a,b):
    global text
    assert text.count(a)==1,(a[:90],text.count(a))
    text=text.replace(a,b)
replace('更新至v0.17完整结果、独立执行/统计核验及v2水场景元数据可行性结果。','更新至v0.18固定策略条件梯度方差结果、独立数学/执行核验及v2水场景79项元数据复核。')
replace('但尚未证明基线准确性或方差机制。v0.17','但该轮没有直接测量基线准确性或条件方差。v0.17')
replace('仍未支持逐例对应带来稳定优势。独立图片v1','仍未支持逐例对应带来稳定优势。v0.18现已在同一固定策略内直接测量奖励score条件方差：第600步匹配相对均匀排列降低62.12%，四来源同向；第0/100步仅0.63%/0.22%，各3正1负。局部方差作用得到支持，但不能解释为v0.17封存终点的已识别中介或新语言结构。独立图片v1')
replace('v2新水来源仅完成元数据可行性检索，79个暂合格关联簇不代表79张已验收水图。','v2新水来源的79个自动代表现已完成元数据复核，描述性28/43/8标签分别表示提水文字、证据不足/歧义、主题风险，不是像素接受或剔除。79个关联簇不代表79张已验收水图。')
replace('下一项仅是[固定政策的条件梯度方差测量候选](../redesign_v0.17/前置科学审查.md)，尚未执行，不能将这个范围较窄的量等同完整训练更新方差。','当时的[固定政策条件梯度方差测量候选](../redesign_v0.17/前置科学审查.md)现已在v0.18另立方案并完成，不能将该较窄的量等同完整训练更新方差。')
paragraph='''v0.18没有新增社会训练或优化器步骤。读取v0.15的4个既有来源×3分区×2方向×0/100/600检查点，共72个个人政策切片；每个固定old18地图，配原训练照片最小行食物0、水31。全部1,296世界、63,504条完整消息的score覆盖43,319个发送策略参数，保留两token共享参数交叉项和接收动作回报二阶矩。固定18世界批均值的半角色方差（原loss系数1/2，方差乘1/4）在600步为matched 0.036137、permuted 0.095393，相对降低62.12%，来源降幅53.60%–79.16%；0/100步的平均相对降低0.63%/0.22%，各有一个来源反向。比例是方差均值之比，不是成功率百分点。它只支持所测同一政策和有限训练支持上的局部降方差机制；排除世界/照片抽样、熵项及其协方差、价值回归、裁剪、Adam和后续伙伴适应，不能将它与v0.17不同学习轨迹合成为因果中介比例。经典最优基线关系不是新理论。全部源前向和已存score范数矩独立核验；完整梯度另在开发18世界、正式4世界作独立Jacobian/有限差分，未声称全部梯度由双实现逐元素重算。见[主报告](../redesign_v0.18/results/gradient_001/固定策略下的通信学习梯度方差研究报告.md)、[独立执行/数学审计](../redesign_v0.18/results/gradient_001/audit_execution.json)、[独立原始复算](../redesign_v0.18/results/gradient_001/independent_recount.json)、[分析对照](../redesign_v0.18/results/gradient_001/analysis_comparison.json)和[研究含义与未执行下一步](../redesign_v0.18/研究含义与下一步.md)。

'''
replace('## 1. 已完成什么，能够支持什么',paragraph+'## 1. 已完成什么，能够支持什么')
row='| [v0.18 固定策略条件方差](../redesign_v0.18/results/gradient_001/固定策略下的通信学习梯度方差研究报告.md) | 0新训练；72切片仍来自4来源；600步matched相对排列期望降方差62.12%、4来源同向，0/100步小且不全同向 | 同一固定政策、old18和一对训练照片下，场景对应的基线具有局部奖励score降方差作用 | 完整训练梯度方差、sealed收益的因果中介、新符号结构、独立图片确认或经典公式的新颖性 |\n'
replace('| [v0.17 基线逐例对应]',row+'| [v0.17 基线逐例对应]')
replace('v0.13–v0.17进一步限定私人经验、数值条件和基线逐例对应解释，但仍须建立超出近邻的预测及独立确认','v0.13–v0.18进一步区分私人经验、数值条件、基线对应与局部方差；v0.18实例化经典控制变量关系，仍须建立超出近邻的预测及独立确认')
replace('固定政策的奖励策略项条件方差测量仍未执行，新旧用途形成原因仍需可反驳检验','v0.18固定政策的奖励score条件方差测量已完成，600步出现四来源同向降低，但未识别长期泛化机制；新旧用途形成原因仍需可反驳检验')
replace('106暂合格文件、79关联簇尚无独立身份或像素验收，不与v1可用数相加。','106暂合格文件、79关联簇的代表现已全部元数据复核（28提水/43不足或歧义/8风险），仍无独立身份或像素验收，不与v1可用数相加。')
metadata='''随后完成的[79个自动代表元数据复核](visual_confirmation_v2_water_review/元数据复核报告.md)只读原资料，没有新增网络或像素请求、确认48图访问或模型调用。根代理的描述性标签为28项文字提到水、43项证据不足/歧义、8项物质/媒介/目标场景风险；标签不重写原79计数、分组或验收门槛，新增可用图片为0。明确问题包括酒装水瓶、照片合成、雨窗短语误命中，以及上传者字段被解析为作者线索；机构账户也不等于独立摄影者。咖啡旁可能有水、药片标题不能证明已溶入、作品照片不等于合成图等边界已在复核意见中澄清并保留历史。原v2的77文件字节不变，1,517技术检查只证明资料关联与记账，不是视觉或身份接受。后续仍需另定冻结来源/身份/许可及有限像素方案，不能直接下载79项或补一个条目凑80。

'''
replace('共享工作区另有[较早的来源准备]',metadata+'共享工作区另有[较早的来源准备]')
replace('固定政策的奖励策略项条件梯度方差只是下一候选，排除熵项、价值梯度、世界抽样、裁剪和Adam，须另立方案，不将其结果等同sealed效应的因果中介。','v0.18固定政策奖励score条件梯度方差诊断现已完成：中期存在局部降低、早期效应较小；排除熵项、价值梯度、世界抽样、裁剪和Adam，不能等同sealed效应的因果中介。由此收束这一优化解释分支，不再自动扩展基线、倍率或学习率网格。下一阶段候选应回到可区分的非语言能力及共同符号形成，须另立有界方案；当前未执行。')
replace('v2水场景仅完成元数据可行性阶段，后续身份/像素验收和确认规则尚待完成。','v2水场景已完成元数据可行性与79代表文字复核，后续身份/像素验收和确认规则尚待完成。')
path.write_text(text)
receipt=dict(status='updated',created_utc=datetime.now(timezone.utc).isoformat(),target=str(path),
 original_snapshot=str(backup),original_sha256=hashlib.sha256(original).hexdigest(),
 updated_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),update_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 scope='Only mutable paper_program/证据与投稿门槛.md updated; v17 frozen files and three README files untouched',
 added=['v18 complete fixed-policy variance with 0 new training','finite-support/sample-audit/4-source boundaries','v2 metadata review 28/43/8 without pixel acceptance','classic variance theory not novelty and next-stage candidate not executed'])
(history/'门槛更新凭证.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(receipt,ensure_ascii=False))

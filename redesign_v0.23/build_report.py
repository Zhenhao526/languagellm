"""Build the v23 report from the independently recomputed analysis."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results/experience_001'
REPORT=OUT/'私人状态经验与共同符号表达迁移研究报告.md'
def read(p):return json.loads(p.read_text())
def pct(x):return f'{100*x:.3f}%'
def pp(x):return f'{100*x:+.3f}'
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def main():
    a=read(OUT/'analysis.json');done=read(OUT/'training_complete.json');agg=a['aggregate'];rows=a['seed_rows']
    assert done['status']=='complete' and done['formal'] and read(OUT/'comparison.json')['passed']
    labels={'private_old':'私人18','private_all':'私人30','old_old':'A：私人18→通信18','all_old':'B：私人30→通信18','all_all':'C：私人30→通信30'}
    private=table(['私人经验','old18 J','目标12 J','目标12食物末帧 J','目标12水末帧 J','目标12原生Q'],[[labels[k],*[pct(agg[k]['scores'][g][m][v]) for g,m,v in [('old','pooled','J'),('new12','pooled','J'),('new12','food_only','J'),('new12','water_only','J'),('new12','pooled','Q')]]] for k in ['private_old','private_all']])
    social=table(['社会条件','old18 J','目标12 J','目标12 Q','目标12常码 J','目标12置换 J'],[[labels[k],*[pct(agg[k]['scores'][g]['pooled'][v]) for g,v in [('old','J'),('new12','J'),('new12','Q'),('new12','blank_J'),('new12','shuffle_J')]]] for k in ['old_old','all_old','all_all']])
    resources=table(['社会条件','食物末帧：食物准确','食物末帧：水准确','水末帧：食物准确','水末帧：水准确'],[[labels[k],*[pct(agg[k]['scores']['new12'][mask][resource]) for mask in ['food_only','water_only'] for resource in ['food','water']]] for k in ['old_old','all_old','all_all']])
    seeds=sorted({r['seed'] for r in rows});get=lambda s,k:next(r for r in rows if r['seed']==s and r['condition']==k)
    sr=[]
    for s in seeds:
        scores={k:get(s,k)['scores']['new12']['pooled']['J'] for k in labels}
        sr.append([s,pct(scores['private_old']),pct(scores['private_all']),pct(scores['old_old']),pct(scores['all_old']),pp(scores['all_old']-scores['old_old']),pct(scores['all_all'])])
    sources=table(['来源','私人18 J','私人30 J','A J','B J','主差B−A（百分点）','C J'],sr)
    main_difference=agg['all_old']['scores']['new12']['pooled']['J']-agg['old_old']['scores']['new12']['pooled']['J']
    assert abs(main_difference-a['primary_mean'])<1e-12
    applicability=read(OUT/'private_applicability.json')
    interpretation=(ROOT/'结果解释.md').read_text()
    body=f'''# 私人状态经验与共同符号表达迁移研究报告

2026年9月16日，v0.23。完成四个新初始化来源的48次私人拟合、36次共同通信训练。唯一主量为通信未见12布局上，私人经验全30相对私人经验18的终点自然双目标J差：**{pp(main_difference)} 个百分点**。三组使用固定预算和新通信初值，原始失败与全部来源保留。

## 本轮问题

v0.22的新初态中，私人行动本身也失败，不能将通信低分直接归因为语言能力。本轮将两种经验分开：独自学习时是否接触某种资源布局，以及随后共同建立符号时是否接触它。A仅在18图学习私人行动及通信；B私人见全30、通信仅见18；C私人和通信都见全30，作为任务学习可行性的辅助对照。

B中的目标12图是**私人阶段已接触、通信阶段未接触**；C中也接受通信训练。只有A的目标12图在两种任务训练阶段都未接触。此前added/sealed分区名字只用于溯源，不回改历史档案或混称全阶段零样本。所有条件仍采用旧开发图片，不是独立图片确认。

## 主体、环境与学习权限

使用官方自监督DINOv2 ViT-L/14的既有冻结视觉缓存，没有LLM或新DINO推理。每个新主体先通过200×64次资源选择奖励校准各自project；该共同准备没有六地点地图。之后CampAgent只继承project，新的slot_phi、GRU与私人行动头从配对初值学习。大型模型提供图像表征，本项目的小接口负责时序保持、行动和通信；不宣称视觉模型具备完整社会推理能力。

环境有六个固定地点、食物和水各一个，位置不同。t0完整观察，t1只再次展示一种资源，位置不变。发送者看到两帧但不知道需求；接收者只读两枚0–6整数token，按食物与水需求各选一次地点，两个动作共用一条消息，不读取另一需求的动作或反馈。地图ID、正确地点与资源类别只用于环境构造、反馈和分析，不输入发送/私人行动头。

私人old18/all30各24次拟合，每人2400步×512次单需求动作；full历史反传，155598可训练参数，project冻结。所选需求0/1奖励、基线.5、Adam .0007、梯度clip2，前2100步熵.02、后300步0。每步256世界精确配对两mask，旧/全经验臂的照片、需求和抽样均匀数配对；支持映射后布局不同。

随后冻结视觉与时间接口、丢弃私人头和优化器。B/C复用同一私人终点，三组发送与接收模块用同一对应新初值。每社会运行2400更新，每方向128世界配对两mask＝256消息、512动作；两端只有各自参数梯度。奖励为`.25*(cF+cW)+.5*cF*cW`，基线.5，按人发送/接收损失平均，分角色clip2，学习率和熵日程同私人。A/B实际通信世界逐位相同；C扩大至30图但总预算相同。全30缓存只预编码，训练只读取相应允许索引。

## 评价与结果

评价30布局×16测试照片对×2mask＝960行，原train/test照片池分开。采用同一有限支持核验私人行动和通信；评价不更新参数。私人J为原私人头的两需求贪心动作均正确，私人Q为该头两动作成功概率的乘积。社会J为自然逐token贪心消息支持两需求均正确，社会Q对49完整消息及两随机动作精确积分。两类系统的贪心与原生概率均分列。常码00与全支持贪心整码独立置换为无更新参照，置换保留整码而不是随机拆token。

四新来源33101–33104中，各自先平均三分区、两发送方向和两mask，再对来源等权。原added6和sealed6合并为目标12，按相同照片数等权；完整分组保存在analysis.json。36社会运行不算36独立来源。过程在0/100/600/1200/2100/2400步固定评价，AUC为这些点的归一化梯形面积，仅为辅助。

{private}

{social}

{sources}

{resources}

私人预定验收只针对全30经验：四来源各自的目标12 J在两个mask下均≥80%，每项先平均三分区与两人，共八项。实际整批**{'通过' if applicability['passed'] else '未通过'}**，逐来源结果见[能力检查](private_applicability.json)和独立分析。该门槛不筛人或改变矩阵，不保证每位主体每张图都掌握，也不表示所有非语言能力已具备。

![私人行动与通信结果]({(OUT/'figures/01_private_and_communication.png').resolve()})

![通信学习过程]({(OUT/'figures/02_learning_curves.png').resolve()})

{interpretation}

## 执行与复核

正式私人更新{done['private_updates']:,}、所选需求动作{done['private_actions']:,}；社会配对更新{done['pair_updates']:,}、消息{done['messages']:,}、动作{done['actions']:,}。基础资源准备更新{done['initial_preparation_updates']:,}、动作{done['initial_preparation_actions']:,}另计。CPU单线程，Torch/NumPy版本见invocation.json，训练与评价计时{done['seconds']:.3f}秒，独立复核和文档另计。

正式前完成解析世界检查、独立实现读审和40步开发批；[前置记录](../../preflight_qa.json)如实注明当时独立数值/优化器复放尚待完成，没有将其提前记为通过。开发前的辅助启动曾误用缺NumPy的系统解释器，未创建正式目录或训练；改用原环境后继续，见[启动历史](../../launch_history.json)，模型源码不变。

[执行审计](audit_execution.json)及[独立比较](comparison.json)记录实际覆盖；[独立分析](analysis.json)由原始行为重算，不调用生产指标公式。本轮独立重算全部288私人及432社会评价表；终点完整重放46,080私人世界及69,120社会发送世界。所有训练外部流身份核验，训练h缓存只核首末块17,664行、test缓存46,080行全核。关键更新只选33101/p1：私人d0两臂第1/2101步共4次，社会三臂两时点共6次配对更新，参数与Adam逐位一致；不是整个训练轨迹独立重训。开发审计器曾因反向浮点归约顺序不同失败，修正独立图顺序后逐位通过，没有放宽容差或改训练源码，失败源与说明均保留。审计和数值一致不能代替科学解释或新颖性证据。

[固定方案](../../固定执行方案.md) · [前置科学审查](../../前置科学审查.md) · [近邻与可证伪预测](../../近邻与可证伪预测.md) · [复现索引](../../README.md)
'''
    REPORT.write_text(body);print(str(REPORT))
if __name__=='__main__':main()

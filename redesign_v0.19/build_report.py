"""Build a report from completed, independently compared private readouts."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/readout_001'
def read(name):return json.loads((OUT/name).read_text())
def pct(x):return f'{100*x:.3f}%'
def pp(x):return f'{100*x:+.3f}'


def main():
    d=read('independent_recount.json');a=read('analysis.json');done=read('training_complete.json')
    assert done['formal'] and done['head_fits']==96 and d['passed'] and read('analysis_comparison.json')['passed']
    audit=read('audit_execution.json');assert audit['passed']
    groups={(x['interface'],x['signal']):x for x in d['aggregate']}
    labels={'retained':'保留私人准备接口','reset_scaled':'重置接口＋原固定幅度'}
    signals={'reward':'仅行动成败奖励','ce':'正确地点监督（CE）'}
    rr=groups['retained','reward'];zr=groups['reset_scaled','reward']
    endpoint=lambda x:x['curve'][-1]['scores']
    lines=['# 冻结视觉接口的新私人行动学习研究报告',
        '', '2026年9月16日；v0.19，固定方案下的正式开发批 readout_001。', '',
        f'96个全新私人行动头已完成固定训练。仅奖励条件下，保留接口与重置后幅度匹配接口在未训练 sealed 组合上的贪心双目标成功率分别为 **{pct(endpoint(rr)["sealed"]["J"])}、{pct(endpoint(zr)["sealed"]["J"])}**；唯一主差为保留减重置，**{pp(d["primary_mean"])} 个百分点**。96头来自四个既有来源，不能按96个独立来源解释。本轮没有训练新的共同通信。', '',
        '## 问题与实验条件', '',
        '前轮将已训练的旧私人行动头接到重置接口后，表现下降。这同时改变了旧头与表示的适配关系，尚不能说明重置端缺少可重新学到的资源定位能力。本轮为每个来源建立全新小头，给予相同经验、初值和预算，直接检验这一解释。', '',
        '主体继续使用官方冻结 DINOv2 ViT-L/14 的原60行视觉特征，没有新DINO前向或视觉主干训练，也没有LLM参与实验。资源投影、地点槽位与单步观察接口均已有先验；本轮的memory每次从零开始，并不表示跨时间的事件记忆。可训练部分只有96→96→12的Tanh行动头，共10,476参数。', '',
        '| 因素 | 固定设置 |', '|---|---|',
        '| 来源与配对 | 31101–31104四来源 × 三个分区 × 两人；每个私人来源运行两接口 × 两信号 |',
        '| 接口 | retained来自v0.13 control社会初态；reset_scaled来自v0.15社会初态，只施加原训练支持校准的正倍率一次 |',
        '| 新行动头 | 四条件初值逐位相同，与原私人头初值不同；各自建立全新Adam，只更新头 |',
        '| 奖励条件 | 只按被抽中的一个需求行动，得到成功/失败0/1；REINFORCE、常数基线0.5 |',
        '| CE参照 | 对同一个被抽中需求提供正确地点标签；不监督另一需求，也不使用未训练组合标签 |',
        '| 预算 | 每头2400次更新，每次512对世界＝1024个单目标场景；Adam学习率0.0007，梯度范数上限2 |',
        '| 熵项 | 被抽中需求的政策熵；前2100次更新系数0.02，后300次为0 |',
        '| 训练支持 | old18 × 22食物图 × 22水图；合法置换产生的两侧场景均在old，配对共享照片和需求 |',
        '| 评价支持 | 30地图 × 8食物test图 × 8水test图＝1920世界；每头固定8个检查点 |', '',
        '六个地点中食物、水各占不同地点，共30种组合。每分区old18参与训练，added6与sealed6都不参与本轮训练；added是沿用旧研究的组名，本轮没有新增组合学习阶段。正确地点、地图ID、照片ID、置换和组别不作为行动头输入；它们只用于构造世界、计算回报、CE监督或审计。两需求通过选择同一头的对应输出分支给定。', '',
        '四条件共享世界、照片、需求和外生行动随机数；政策变化后实际动作可以不同。两接口只沿用此前平均L2尺度匹配，方向、协方差及非线性响应没有被强行相等。新私人头及CE标签均未回灌既有社会主体。', '',
        '## 固定终点与形成过程', '',
        'J表示同一个私人头在同一场景下，对食物与水两个需求分别贪心行动，均选对地点的比例；Q表示两需求各按其原生随机政策独立行动时的双成功期望。它们都不是双方通信成功率。每来源先平均三分区与两人，再对四来源等权平均。', '',
        '| 接口 | 信号 | old J | added J | sealed J | sealed Q | sealed J AUC |',
        '|---|---|---:|---:|---:|---:|---:|']
    for i in ('retained','reset_scaled'):
        for s in ('reward','ce'):
            x=groups[i,s];e=endpoint(x)
            lines.append(f'| {labels[i]} | {signals[s]} | {pct(e["old"]["J"])} | {pct(e["added"]["J"])} | {pct(e["sealed"]["J"])} | {pct(e["sealed"]["Q"])} | {pct(x["auc"]["sealed"]["J"])} |')
    lines += ['', '| 来源 | 奖励：保留 J | 奖励：重置匹配 J | 主差（百分点） | CE接口差（百分点） |',
        '|---|---:|---:|---:|---:|']
    for x in d['contrasts']:
        v=x['values'];lines.append(f'| {x["seed"]} | {pct(v["retained_reward"])} | {pct(v["reset_scaled_reward"])} | {pp(x["primary_reward_retained_minus_reset_scaled"])} | {pp(x["secondary_ce_retained_minus_reset_scaled"])} |')
    lines += ['', 'AUC为0–2400步、固定8点的归一化梯形面积，属于辅助指标。初始、100、300、600、1200、1800、2100、2400步全部保留，不按最高成绩选检查点。原始单目标、概率、熵、NLL、并列率、原始计数与全部来源曲线见[独立复算](independent_recount.json)和[结构化分析](analysis.json)。熵和NLL以需求查询为单位，对两需求平均。', '',
        f'![私人行动学习曲线]({OUT}/figures/01_private_learning.png)', '',
        f'![四来源配对终点]({OUT}/figures/02_paired_sealed_readout.png)', '',
        (ROOT/'结果解释.md').read_text().strip(), '',
        '## 执行与复核', '',
        f'正式训练耗时约{done["seconds"]:.2f}秒，CPU单线程；审计、分析与文档耗时另计。共230,400次头更新、235,929,600次模拟单目标行动，其中CE条件使用117,964,800次地点标签。这些是重复抽样暴露，不是相同数量的独立世界。开发8头全预算通过后才执行正式96头，正式全部完成前未按中途效应决策。', '',
        f'[独立执行审计](audit_execution.json)通过{audit["checks"]:,}项：核对全部缓存观察前向、四条件配对数据流、头与Adam状态、首步及第2101步重放，以及768个检查点的全部1,474,560世界评价。重放没有调用生产损失或世界生成函数。它不表示所有230,400次更新都重新做了一遍完整反向传播；完整参数与Adam重放固定在每头两步。', '',
        f'[独立原始复算](independent_recount.json)完成{d["checks"]:,}项，最大差{d["max_absolute_comparison_error"]:.3g}；[分析交叉比较](analysis_comparison.json)通过。正式源码、输入、初值、训练轨迹、全部评价logits与开发历史均保留。方案文字曾在完整开发前改为准确描述保存的梯度范数和抽样重放，未改损失、主终点、预算或正式条件；原短开发和原方案见[方案历史](../../plan_history)。', '',
        '## 适用范围与后续', '',
        '这仍是旧视觉任务、四个继承开发来源的结果。每图64个测试照片对与v0.13/14原私人评价的16对不同，不能直接相减声称重训效应；旧照片多次用于开发，也不属于独立图片确认。没有进行显著性或等效判断，没有证明表示信息完全相同、所有非语言能力相同、一般句法或人类语言起源机制。', '',
        '本轮文献复核新增两篇公开PDF、重点核查五篇方法。Feng等的AAAI 2024论文已有视觉预训练、冻结后新通信与读出重训；Kouwenhoven等已有冻结DINOv2和表示对齐。因此，重训读出和“私人任务好、通信弱”本身不足以宣称创新。方法差异与逐页来源见[短近邻定位](../../../paper_program/literature_capability_20260916/短近邻定位.md)。', '',
        '下一阶段优先候选是事件后的地点保持与更新：先看完整场景，再看一个资源移动的局部事件，分别检查即时可见行动和需要保持历史的延迟行动。须先固定合法事件与组合留出、证明信息没有旁路泄漏，并验证私人能力确有可重复差异，才解释其对全新共同符号的作用。当前该序列环境、能力干预和社会学习尚未执行；不以继续扩大基线或倍率网格替代这一问题。候选及限制见[科学解释界限](../../科学解释界限.md)。', '',
        '完整批次完成不等于论文证据已齐。中心贡献、独立来源/图片确认以及自主未见组合表达仍是主要缺口；较早负结果和失败开发记录保留。', '',
        '[固定方案](../../固定执行方案.md) · [前置审查](../../前置审查.md) · [预检凭证](../../preflight_qa.json) · [运行输入](invocation.json) · [训练完成凭证](training_complete.json) · [复现索引](../../README.md)', '']
    (OUT/'冻结视觉接口的新私人行动学习研究报告.md').write_text('\n'.join(lines))
    print('report built')


if __name__=='__main__':main()

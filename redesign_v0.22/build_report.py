"""Render the completed frozen visibility probe into a readable research note."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results/visibility_001'
def read(p):return json.loads(p.read_text())
def pc(x):return f'{100*x:.3f}%'
def pp(x):return f'{100*x:+.3f}'
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,row))+' |' for row in rows])
def main():
    a=read(OUT/'analysis.json');done=read(OUT/'evaluation_complete.json');app=read(OUT/'private_applicability.json');cmp=read(OUT/'comparison.json')
    assert done['formal'] and cmp['passed'];agg=a['aggregate'];labels={'private':'原私人行动头','social_immediate':'原完整条件协议','social_delayed':'原局部条件协议','social_all':'两类协议等权均值'}
    stats=table(['冻结系统','F可见：食物准确','F可见：水准确','F可见：J','W可见：食物准确','W可见：水准确','W可见：J','D（百分点）'],
        [[labels[k],*[pc(v['common30'][mask][m]) for mask in ('food_only','water_only') for m in ('food','water','J')],pp(v['common30']['D_visible'])] for k,v in agg.items()])
    groups=table(['系统','支持','两mask平均J','D（百分点）','两mask均双正确','消息改变'],[[labels[k],g,
        pc((v[g]['food_only']['J']+v[g]['water_only']['J'])/2),pp(v[g]['D_visible']),pc(v[g]['stable_correct_J']),
        '不适用' if v[g]['message_change_rate'] is None else pc(v[g]['message_change_rate'])] for k,v in agg.items() if k in ('private','social_all') for g in ('old','added','sealed','common30')])
    source={s:{r['system']:r['scores'] for r in a['seed_rows'] if r['seed']==s} for s in sorted({r['seed'] for r in a['seed_rows']})}
    pairs=table(['来源','私人D','完整协议D','局部协议D','主社会均值D','F可见时F−W','W可见时F−W'],[[s,
        *[pp(d[k]['common30']['D_visible']) for k in ('private','social_immediate','social_delayed','social_all')],
        pp(d['social_all']['common30']['category_gap_food_only']),pp(d['social_all']['common30']['category_gap_water_only'])] for s,d in source.items()])
    figures='\n\n'.join(f'![{p.stem}]({p.resolve()})' for p in sorted((OUT/'figures').glob('*.png')) if '_pdf_preview' not in p.stem)
    interpretation=(ROOT/'结果解释.md').read_text()
    body=f'''# 布局保持与末帧可见对象的固定协议研究报告

2026年9月16日，v0.22。完成全部24个私人头与48个既有社会发送方向的离线评价。唯一主要量是两类既有协议等权后的可见资源准确率减历史资源准确率D，共同30布局均值为 **{pp(a['primary_mean'])} 个百分点**。大效应集中在新初态/布局组合：熟悉布局的社会D仅2.344个百分点，新布局为78.201个百分点；新布局中的私人行动也退化。本轮没有新训练或新语言形成过程。

## 这次改变了什么

每个世界的食物、水位置始终不变。主体先看完整布局，下一帧只再次看见食物，或只再次看见水；两条件首帧、终态、照片与参数相同，第一帧都提供了两项答案。末帧可见对象同时更近期、被展示两次，所以本轮没有进一步分离可见性、近期性和曝光次数。

使用全部v0.20 full私人终点及原私人头，和相应v0.21完整/局部条件训练出的全部协议终点。DINOv2缓存、投影、视觉/时间接口、私人头、发送和接收模块全部冻结；参数requires_grad=False，no_grad，CPU float32，两次原GRU与原无仿射LayerNorm，view_bits=[1,0]。私人与社会共享已核对一致的前端，接收者仍只读取两枚离散token与既有需求分支，不知道哪种资源再次可见。

无移动事件在旧训练中未出现；added/sealed此前未作为完整初态，现在也进入t0。因此这既是固定模型的配对观察干预，也是支持外评价。旧“局部图像总指向刚移动资源”的训练关联可能导致错误更新。不能单独将观察差解释成遗忘、注意或语言机制。

## 世界与主量

共同30布局配原test池4×4照片，共480个世界，每布局16行，没有路线重复权重。旧分区old18/added6/sealed6完整分列。每人两mask各评价480行，按256+224原批大小前向。4来源内先平均三分区、两发送方向、两种原协议训练条件，再对四来源等权；大量世界和端点不增加独立来源数。

令A_food(food_only)表示只再次看见食物时的食物行动准确率：

`D = 0.5 × [(A_food(food_only) − A_food(water_only)) + (A_water(water_only) − A_water(food_only))]`。

这个D等于两mask下平均可见资源正确率减需从历史保持的资源正确率。D为正不保证优势真的交叉；还需食物可见时F−W为正、水可见时F−W为负，以下逐来源报告。J为两需求贪心行动均正确；私人Q为两个原生动作概率乘积，社会Q对49完整消息与两独立行动精确积分。社会发送贪心始终逐token解码，不用49码联合argmax。

{stats}

{pairs}

{groups}

两mask都双目标正确（stable_correct_J）衡量同一世界在两种观察下均能完成任务；消息变化可以是同义表达，不表示结构改善或失稳。D=0可能来自全对、全错或正负抵消。仅末帧策略在共同30的最优J为20%、单目标宏均为60%；可见1/隐藏.2对应D=.8只是特定参照，任意策略的D并不被.8限制。

## 私人适用性检查

事前只用全24私人头的old支持两mask J宏均值均≥80%作描述性检查。实际食物可见为{pc(app['old_J']['food_only'])}，水可见为{pc(app['old_J']['water_only'])}，检查结果为**{'通过' if app['passed'] else '未通过'}**。这不是显著性、等效或人类语言能力门槛。无论结果如何，都没有剔除个人、改变主量或新增社会学习；对既有协议的离线评价完整保留。

{figures}

{interpretation}

## 执行与复核范围

实际私人世界前向{done['private_world_forwards']:,}、社会发送世界前向{done['social_sender_world_forwards']:,}；没有新训练、优化器更新、行动抽样或DINO推理。评价程序约{done['seconds']:.3f}秒，独立复核与文档另计。原文件、失败记录和完整输出均保留。

[执行审计](audit_execution.json)核对实际源权重、冻结状态和世界表，并按原256+224批完整重放全部480行私人h/头输出、社会消息概率/逐token码及49码接收输出。这次任务规模小，前向覆盖全表；它仍不是对旧v20/v21训练的重新执行。

[独立分析](analysis.json)不调用生产统计或聚合函数，独立计算资源正确率、概率、两mask配对与四来源主量；[比较记录](comparison.json)绑定原始文件和分析源码。执行和数值核验不能替代新颖性或跨图片确认。

[固定方案](../../固定执行方案.md) · [前置审查](../../前置审查.md) · [近邻与解释边界](../../近邻边界.md) · [私人检查](private_applicability.json) · [复现索引](../../README.md)
'''
    (OUT/'布局保持与末帧可见对象的固定协议研究报告.md').write_text(body)
    print('report built')
if __name__=='__main__':main()

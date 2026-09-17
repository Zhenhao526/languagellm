"""Finalize documentation only after complete independent verification."""
from pathlib import Path
import json,hashlib,math,datetime
HERE=Path(__file__).resolve().parent
RUN=HERE/'results/trajectory_001'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    result=read(RUN/'execution/results.json');proof=read(RUN/'audit_execution_002/verification.json')
    assert result['status']=='completed' and proof['status']=='passed'
    assert proof['plan_sha256']==result['plan_sha256']==sha(RUN/'plan.json')
    assert proof['new_neural_module_samples']==result['totals']['total_new_module_samples']==203853600
    assert math.isclose(result['primary']['mean_difference'],proof['primary']['mean_difference'],abs_tol=2e-12,rel_tol=0)
    assert sha(HERE/'audit_execution_v2.py')==proof['audit_source_sha256']
    n=proof['numerical_comparison'];scope=proof['independent_scope']
    audit_text=f'''正式审计001在1.9524秒时因比较器把13字段的完整元数据与8字段的独立数据定义直接作整字典相等比较而停止，尚未加载权重或进行模型前向。共同的8个定义字段在两分区完全相同；5个额外字段是分区标签、排序、权重说明、监控说明和语义对摘要。原失败、原源码与快照全部保留。

新增独立审计器v2严格核对完整元数据与原已审计来源，再逐项比较独立重建的8个定义字段；没有删除数据检查或改变主结果。3项实际静态元数据回归通过。此前13项合成测试还发现并修复了布尔值比较错误，原版保留。预检收据补记时间晚于001启动／失败，准确时序见[勘误](audit_preflight_001/时序勘误.md)。

审计002随后完整通过，耗时{proof['elapsed_seconds']:.4f}秒，另重放全部203,853,600模块样本与88次参数加载。消息、行动及物理／反事实结算一致，最大概率误差{scope['maximum_probability_error']:g}。{n['scalars']:,}个数值标量独立交叉比较，最大绝对差{n['max_error']:.3g}；嵌套布尔值按类型和值精确比较。全部2848个NPZ、5952个记录JSON与688个摘要JSON覆盖，旧6000锚仅核对来源与数组，不重复前向或重放优化器。独立静态重建、时间权重、选位、路由及3456个端点交换检查均通过。

[完整独立核验](results/trajectory_001/audit_execution_002/verification.json)SHA为`{sha(RUN/'audit_execution_002/verification.json')}`。该复核确认执行和统计一致，不增加独立群体数量。'''
    p=HERE/'结果与下一步.md';text=p.read_text();assert text.count('AUDIT_PENDING')==1
    text=text.replace('主计算完成；独立重算待完成，当前数值为主程序结果。','主计算和完整独立重算均已完成。').replace('AUDIT_PENDING',audit_text);p.write_text(text)
    p=HERE/'README.md';text=p.read_text().replace('主计算完成，独立复核待完成','主计算和独立复核均完成')
    text=text.replace('独立实现位于`audit_execution.py`','正式通过的独立实现位于`audit_execution_v2.py`，初版与失败记录保留在`audit_execution.py`及`audit_execution_001`')
    text=text.replace('- [正式准备与冻结]', '- [独立复核002](results/trajectory_001/audit_execution_002/verification.json)、[001失败与修复记录](results/trajectory_001/audit_source_snapshot_002/manifest.json)。\n- [正式准备与冻结]');p.write_text(text)
    p=HERE.parent/'README.md';text=p.read_text();assert 'triadic_formation_trajectory_study/结果与下一步.md' not in text
    text=text.replace('- **最新进展：**','- **此前单位置作用：**',1)
    paragraph='- **最新进展：**[六检查点约定形成轨迹](triadic_formation_trajectory_study/结果与下一步.md)完成16项既有政策的六时点测量和6×6跨时间整包测试，全部新增前向经独立重放。公开布局PL相对局部观察LL的累计选择性差均+1.0145个百分点，四种子两正两负；第500步PL四种子所选位置D/S均0，而整包T均正。发送响应、整包功能与单位置作用并不同步；跨时间矩阵不能单独解释成词义漂移或代际传承。0新增训练，固定伙伴与独立确认缺口仍在。审计001在前向前因元数据比较实现问题停止，保留失败；v2完整通过。\n\n'
    first=text.index('- **此前单位置作用：**');text=text[:first]+paragraph+text[first:];p.write_text(text)
    p=HERE.parent/'进展账本.md';text=p.read_text();heading='## 2026-09-16：六检查点的约定形成与跨时间协议兼容'
    assert heading not in text
    entry=f'''{heading}

上一回合完成静态数据、通信模块、固定测量与近邻准备，未执行本轮新模型前向，属于取得进展。本回合30项主测试通过后冻结，prepare会话41562成功；主会话37395一次完成208.0950秒，无训练、无新初始化。计划SHA `{result['plan_sha256']}`，主结果SHA `{sha(RUN/'execution/results.json')}`。

16项既有政策、四配对开发种子×PL/LL开放/静默，固定0/100/500/1500/3000/6000步。发现消息覆盖全部419904训练世界，验证72案例×2布局／144行，接收供体自然并集554世界。每时点训练选位；所有8位置和两轮sham、完整6×6跨时间整包矩阵保留。6000全部同时间输出复用上轮。新增203853600模块样本、88参数加载、0训练更新；静默别名与复制不计前向。

主要S归一化面积，PL/LL四配对为4.6817/0.3791、−0.2170/0.0014、3.9106/2.5159、1.2818/2.7025个百分点；差+4.3027/−0.2185/+1.3947/−1.4207，均+1.0145，两正两负。末点三正一负不代表形成过程的稳定优势。第500步PL四种子当时所选位置D/S均0，整包T分别2.7778/3.1250/3.4722/2.4306；初始化已有35.0001%/39.4079%发送响应而自然全队满分0。仅提供整体作用与局部作用分离候选，不构成统一起源阶段。

固定6000接收者，供体0/100/500/1500/3000/6000的T均值PL为1.0417/2.2569/4.8611/9.1146/11.8924/12.5000，LL为0.5208/0.4340/4.0799/7.6389/8.6806/8.6806个百分点。LL末两项均值相等但四种子差0/−1.0417/+0.6944/+0.3472；不是协议逐群体不变。供体W2含原时点互动，接收方能力也变化，不独占解释为词义漂移、共适应机制或世代传承。

13项合成测试及15子例完成，修独立比较器bool减法错误；原稿保留。审计001会话41111在1.9524秒、加载权重前因13字段元数据与8字段独立定义整字典比较而停止。根新增v2：完整元数据严格匹配原SHA来源，独立定义8字段逐项匹配，3项回归通过，主冻结源与结果不改。预检收据落盘晚于001失败的时序描述错误另有勘误，原收据保留。

审计002会话80234完整通过，{proof['elapsed_seconds']:.4f}秒，另重放同样203853600模块及88加载；概率最大误差{scope['maximum_probability_error']:g}，{n['scalars']:,}个数值标量最大差{n['max_error']:.3g}，2848NPZ/5952记录/688摘要全覆盖。核验SHA `{sha(RUN/'audit_execution_002/verification.json')}`。全部旧6000锚核来源不重复前向，未重训。

三图PNG实际查看通过，PDF同源保存；主报告数字与科学边界另有只读复核。两篇直接近邻完成有界核查：1正式PDF复用、1下载重命名，训练动态与跨代稳定已有先行，不以检查点互换宣称首创。见[完整报告](triadic_formation_trajectory_study/结果与下一步.md)。本回合属于取得进展；下一步需新种子、可区分的发送/接收机制及环境信息负担对照。固定伙伴、组合性、独立确认和论文中心贡献仍缺证据，目标保持进行中。

'''
    text=text.replace('## 论文完成所需的证据',entry+'## 论文完成所需的证据');p.write_text(text)
    record=dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),prepare=dict(session_id=41562,exit_code=0),main=dict(session_id=37395,exit_code=0,elapsed_seconds=result['elapsed_seconds']),audit001=dict(session_id=41111,exit_code=1,neural_forward_samples=0,parameter_loads=0),audit002=dict(session_id=80234,exit_code=0,elapsed_seconds=proof['elapsed_seconds']),main_sha256=sha(RUN/'execution/results.json'),verification_sha256=sha(RUN/'audit_execution_002/verification.json'),report_sha256=sha(HERE/'结果与下一步.md'),report_peer_review='Read-only agent checked primary, all four pairs, fixed-time contrasts, terminal receiver row, final score counts, rewards and interpretation; root clarified W2 recomputation afterward.',training_updates=0)
    with (RUN/'execution_receipt_001.json').open('x') as f:json.dump(record,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(record,ensure_ascii=False))
if __name__=='__main__':main()

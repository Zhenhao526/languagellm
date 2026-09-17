"""Seal this finite probe and update living project indexes without old edits."""
import hashlib,json,re,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent;OUT=ROOT/'results/transfer_001'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
assert json.loads((OUT/'audit_qa.json').read_text())['passed']
assert not (OUT/'completion_manifest.json').exists()
report=OUT/'私人行动与共同通信在新图片上的迁移研究报告.md';assert report.is_file()
backup=ROOT/'index_history';backup.mkdir(exist_ok=False)
lit=PROJECT/'paper_program/confirmation_readiness_20260916/视觉材料与通信读出的近邻定位.md'
support=OUT/'supporting_materials';support.mkdir(exist_ok=False);shutil.copy2(lit,support/lit.name)
paragraph=('最新完成v0.27已有A/B/C协议的新图片通信检验：复用v26的11张流程图表征，72方向、288表、336960个发送世界，无新训练或DINO推理。'
    'B（私人30/通信18）的目标12布局自然J由15.896%变为17.149%，主材料差+1.253个百分点，四来源均小正；C（私人30/通信30）为95.833%，其目标布局已通信训练。'
    '新图B−A均差+5.845个百分点，但一来源反向；B目标12消息TV为0.2223，训练过的18布局仅0.0011。'
    '私人能力成功与低自然表达的差距在这批材料上仍存在，未获得新组合语言或独立形成确认。'
    '该11图冻结探针分支现结束，下一工作为独立训练/测试材料准备及完整A/B/C形成过程检验，不能用更多同图测量代替。')
for i,path in enumerate((PROJECT/'README.md',PROJECT/'paper_program/README.md',PROJECT/'paper_program/证据与投稿门槛.md')):
    shutil.copy2(path,backup/f'{i}_{path.name}');text=path.read_text();first,rest=text.split('\n',1)
    rest=rest.replace('最新完成v0.26冻结私人主体的新图片能力探针','此前完成v0.26冻结私人主体的新图片能力探针',1)
    rest=rest.replace('下一步可另立方案，检查v23既有A/B/C通信协议对相同流程图片的稳健性，再推进独立训练/测试材料上的形成过程确认。','该冻结通信检查现已在v27完成；独立训练/测试材料上的形成过程确认仍待执行。')
    if i==2:
        rest=rest.replace('更新至v0.26新图片上的冻结私人能力探针','更新至v0.27新图片上的冻结通信协议检验',1)
        rest=rest.replace('v26已增加11张流程图的冻结私人能力检验；独立通信确认及形成复现仍未完成。','v26/v27已增加11张流程图的冻结私人和通信能力检验；独立材料确认及形成复现仍未完成。')
        rest=rest.replace('v26另用11流程图完成私人能力探针，尚无通信确认；','v26/v27另用11流程图完成私人和通信探针，尚非独立材料形成确认；')
    prefix='' if i==0 else '../'
    addition=paragraph+f' 见[完整报告]({prefix}redesign_v0.27/results/transfer_001/私人行动与共同通信在新图片上的迁移研究报告.md)及[复现索引]({prefix}redesign_v0.27/README.md)。'
    path.write_text(first+'\n\n'+addition+'\n'+rest)
(ROOT/'README.md').write_text('''# v0.27：已有协议在新图片上的通信

已完成并独立核查。B（私人30/通信18）在目标12布局上的自然J：旧图15.896%、新图17.149%，主材料差+1.253个百分点，四来源小幅正。C（私人30/通信30）为95.833%，这些布局参与过通信训练。新图B−A平均+5.845个百分点，但一来源反向。

同一11图已在v26证明私人all行动100%，此处已有B协议仍低表达；这不构成新语言形成复现、独立材料确认或原创性认证。本探针分支结束，后续优先独立材料准备及完整学习过程检验。

- [完整报告](results/transfer_001/私人行动与共同通信在新图片上的迁移研究报告.md)
- [固定方案](固定执行方案.md)、[前置科学审查](前置科学审查.md)、[数据兼容性核查](前置数据核查.json)、[结果解释审查](结果解释审查.md)
- [原始及来源汇总](results/transfer_001/summary.json)、[独立核查](results/transfer_001/audit_qa.json)
- [生产脚本](run_transfer.py)、[独立核查脚本](audit_transfer.py)、[报告脚本](report_transfer.py)
- [输入冻结](results/transfer_001/freeze.json)、[运行记录](results/transfer_001/completion.json)

36个既有配对终点、72方向×4材料格，共288表336960发送世界，无训练、无新DINO或像素。复现须使用新输出目录，命令见报告。前端参数逐位绑定，但复用私人h与旧社会缓存有末位差；所有旧协议自然token相同，对数概率和概率通过预先容差。11张图仅为已暴露流程材料，原确认48图保持未访问。
''')
links=[]
for p in (ROOT/'README.md',report):
    for target in re.findall(r'\]\(([^)]+)\)',p.read_text()):
        if not target.startswith(('https:','http:','#')):
            resolved=(p.parent/target).resolve();assert resolved.is_file(),str(resolved)
            links.append(dict(document=str(p),target=target,exists=True))
frozen=json.loads((OUT/'freeze.json').read_text())
assert all(sha(Path(p))==h for p,h in frozen['source_hashes'].items())
files=[p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
artifacts={str(p.relative_to(ROOT)):sha(p) for p in files}
write(OUT/'completion_manifest.json',dict(status='complete_bounded_frozen_communication_probe',artifacts=artifacts,artifact_count=len(artifacts),links=links,source_freeze_sha256=sha(OUT/'freeze.json'),all_inputs_unchanged=True,goal_complete=False))
print(json.dumps(dict(sealed=True,artifacts=len(artifacts),links=len(links))))

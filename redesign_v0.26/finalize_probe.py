"""Update living indexes once, keep their old versions, and seal this probe."""
import hashlib,json,re,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent;OUT=ROOT/'results/material_001'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')

assert json.loads((OUT/'audit_qa.json').read_text())['passed']
assert (OUT/'冻结私人主体的新图片能力研究报告.md').is_file()
assert not (OUT/'completion_manifest.json').exists()
backup=ROOT/'index_history';backup.mkdir(exist_ok=False)
paragraph=('最新完成v0.26冻结私人主体的新图片能力探针：首次用当前11张有限流程可用图片进行实验模型推理，未新增训练。'
    '私人all在目标12布局上换食物、换水或同时换图后均保持100%双目标成功；8项来源×mask能力门槛全部通过。'
    '私人old在全部换图后为69.879%（旧图67.220%）。四个既有来源、48私人终点，共224,640场景前向；旧图编码与原行为对照通过。'
    '这说明这批小样本材料上的私人能力可以保持，没有复现共同语言形成，确认48图仍未访问。'
    '下一步可另立方案，检查v23既有A/B/C通信协议对相同流程图片的稳健性，再推进独立训练/测试材料上的形成过程确认。')
index_files=[PROJECT/'README.md',PROJECT/'paper_program/README.md',PROJECT/'paper_program/证据与投稿门槛.md']
for i,path in enumerate(index_files):
    shutil.copy2(path,backup/f'{i}_{path.name}')
    text=path.read_text();first,rest=text.split('\n',1)
    prefix='' if i==0 else '../'
    addition=paragraph+f' 见[研究报告]({prefix}redesign_v0.26/results/material_001/冻结私人主体的新图片能力研究报告.md)和[复现索引]({prefix}redesign_v0.26/README.md)。\n'
    text=first+'\n\n'+addition+rest
    if i==0:
        text=text.replace('最新完成v0.25资源角色分别读入诊断','此前完成v0.25资源角色分别读入诊断',1)
        text=text.replace('确认48图未访问，新图尚未进入模型实验。','确认48图未访问；11张流程图现已进入v0.26私人能力探针，不再是未暴露材料。')
    elif i==1:
        text=text.replace('确认48图未访问、无新图模型调用。','确认48图未访问；11张流程图现已进入v0.26私人能力探针。')
    else:
        text=text.replace('更新至v0.25相同初始协议下的资源角色分别读入诊断','更新至v0.26新图片上的冻结私人能力探针',1)
        text=text.replace('仍无独立图片模型确认。当前仍是旧视觉任务上的开发研究。','v26已增加11张流程图的冻结私人能力检验；独立通信确认及形成复现仍未完成。')
        text=text.replace('确认48图未访问、0模型确认；','确认48图未访问；v26另用11流程图完成私人能力探针，尚无通信确认；')
        text=text.replace('确认48图未访问，无新图模型调用。','截至该像素阶段确认48图未访问、没有实验模型调用；后续v26流程探针另记。')
        text=text.replace('确认48图仍未访问、无新图模型调用。','截至第三批像素整理时确认48图未访问、没有实验模型调用；后续v26已另行使用11流程图。')
    path.write_text(text)

readme='''# v0.26：冻结私人主体的新图片能力探针

已完成。11张新流程图片上，私人all的目标12布局双目标行动仍为100%；私人old为69.879%。四格材料替换、48冻结终点、224640个私人世界，全部数据保留。没有训练新协议，也未使用原确认48图。

- [研究报告](results/material_001/冻结私人主体的新图片能力研究报告.md)
- [固定方案](固定执行方案.md)、[生产程序](run_probe.py)、[独立核查程序](audit_probe.py)
- [结果汇总](results/material_001/summary.json)、[独立核查](results/material_001/audit_qa.json)
- [输入冻结](results/material_001/freeze.json)、[完成记录](results/material_001/completion.json)
- [11图署名和来源](results/material_001/selection.json)、[图片模型暴露记录](results/material_001/model_exposure_started.json)

复现命令见报告。必须使用新的输出目录；原v23权重和v4缓存仅作只读输入，新图沿用旧训练特征的标准化，不重估。现有11图仅为有限流程材料，已暴露于模型，不得作为后续未揭盲确认集。下一步是另立冻结协议材料稳健性检查，独立形成复现仍需新材料训练/测试划分。
'''
(ROOT/'README.md').write_text(readme)
files=[p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
artifacts={str(p.relative_to(ROOT)):sha(p) for p in files}
links=[]
for path in [ROOT/'README.md',OUT/'冻结私人主体的新图片能力研究报告.md']:
    for target in re.findall(r'\]\(([^)]+)\)',path.read_text()):
        if not target.startswith(('http:','https:','#')):
            resolved=(path.parent/target).resolve();assert resolved.is_file(),str(resolved)
            links.append(dict(document=str(path),target=target,exists=True))
frozen=json.loads((OUT/'freeze.json').read_text())
assert all(sha(Path(p))==h for p,h in frozen['input_hashes'].items())
write(OUT/'completion_manifest.json',dict(status='complete_bounded_material_probe',artifacts=artifacts,artifact_count=len(artifacts),input_freeze_sha256=sha(OUT/'freeze.json'),links=links,old_inputs_unchanged=True,goal_complete=False))
print(json.dumps(dict(sealed=True,artifacts=len(artifacts),links=len(links))))

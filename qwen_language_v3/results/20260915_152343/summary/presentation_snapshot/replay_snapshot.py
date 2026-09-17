"""Generate a standalone researcher replay snapshot without model inference.

python3 -m qwen_language_v3.replay_snapshot RUN_DIR --out review_outputs/replay.html
Reads steps/episodes/status only. It never reads inference/private-analysis logs.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path


def _read_jsonl(path, warnings):
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("not an object")
                rows.append(row)
            except ValueError as error:
                warnings.append(f"{path.name}第{number}行未纳入：{error}")
    return rows


def _global_view(state):
    if not isinstance(state, dict):
        return None
    items = {}
    for key, item in state.get("items", {}).items():
        items[key] = {name: item[name] for name in
            ("kind", "length", "condition", "processed", "carriers", "delivered", "location") if name in item}
    event = state.get("event", {})
    return {
        "t": state.get("t"), "max_steps": state.get("max_steps"),
        "positions": state.get("positions", {}), "inventory": state.get("inventory", {}),
        "items": items, "goals": state.get("goals", []),
        "closed_road": event.get("edge") if event.get("occurred") else None,
    }


def collect_snapshot(run_dir):
    directory = Path(run_dir).resolve()
    if not directory.is_dir():
        raise NotADirectoryError(directory)
    warnings = []
    status_path = directory / "status.json"
    try:
        status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
    except ValueError as error:
        warnings.append(f"status.json未读取：{error}")
        status = {}
    entries = []
    for index, row in enumerate(_read_jsonl(directory / "steps.jsonl", warnings)):
        # Whitelist these structured environmental records. Inference text and
        # private analysis have no field or source in this view.
        entry = {key: row.get(key) for key in (
            "phase", "condition", "group", "episode", "step", "observations",
            "actions", "selections", "feedback", "score",
        )}
        entry["id"] = index
        entry["messages"] = [{key: msg.get(key) for key in ("sender", "window", "text", "step")}
            for msg in row.get("messages", []) if isinstance(msg, dict) and isinstance(msg.get("text"), str)]
        entry["before"] = _global_view(row.get("state_before"))
        entry["after"] = _global_view(row.get("state_after"))
        entries.append(entry)
    episodes = [{key: row.get(key) for key in ("phase", "condition", "group", "episode", "score", "steps", "success")}
        for row in _read_jsonl(directory / "episodes.jsonl", warnings)]
    return {
        "run_name": directory.name, "source_directory": str(directory),
        "generated_at": datetime.now().astimezone().isoformat(),
        "status": {key: status.get(key) for key in ("status", "condition", "step", "stop_reason", "completed_physical_steps")},
        "entries": entries, "episodes": episodes, "warnings": warnings,
    }


HTML = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>三主体实验 · 研究者回放快照</title>
<style>
:root{color-scheme:light;--ink:#18303a;--muted:#5c7078;--line:#d9e4e8;--blue:#126778;--warm:#fff5e4;--warm-ink:#88500b;--page:#f3f7f8}
*{box-sizing:border-box;min-width:0}body{margin:0;background:var(--page);color:var(--ink);font:15px/1.65 system-ui,-apple-system,"PingFang SC","Microsoft YaHei",sans-serif;overflow-wrap:anywhere}
main{max-width:1400px;margin:auto;padding:24px}h1{font-size:clamp(23px,3vw,34px);line-height:1.3;margin:12px 0}h2{font-size:20px;line-height:1.45;margin:0 0 12px}h3{font-size:16px;margin:0 0 10px}p{margin:8px 0}.muted{color:var(--muted)}.small{font-size:13px}.tag{display:inline-block;border:1px solid var(--line);border-radius:20px;padding:2px 10px;background:white;font-size:12px}.tag.warm{background:var(--warm);color:var(--warm-ink);border-color:#efd7ad}
.panel{background:#fff;border:1px solid var(--line);border-radius:16px;padding:20px;margin-top:18px}.banner{border-left:4px solid #bc791e;background:var(--warm);border-radius:8px;padding:12px 16px;color:#70420a}.controls{display:flex;flex-wrap:wrap;gap:12px;align-items:end}.field{display:flex;flex-direction:column;gap:4px;flex:1 1 140px;max-width:260px}label{font-size:13px;color:var(--muted)}select,button{font:inherit;border:1px solid #abc3cc;border-radius:8px;background:white;padding:9px 12px;color:var(--ink);max-width:100%}button{cursor:pointer}button:hover:not(:disabled){background:#eaf5f8}button:disabled{opacity:.4;cursor:default}.nav{display:flex;gap:8px;flex-wrap:wrap}.selected{background:var(--blue);border-color:var(--blue);color:white}.selected:hover:not(:disabled){background:#155667}
.row{display:flex;align-items:center;justify-content:space-between;gap:16px;flex-wrap:wrap}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}.card{padding:16px;border:1px solid var(--line);border-radius:12px;background:#fcfefe}.agent-title{display:flex;gap:8px;align-items:center}.agent-dot{border-radius:50%;width:30px;height:30px;display:inline-grid;place-items:center;background:#d9eef2;color:#075263;font-weight:700;flex-shrink:0}.label{font-size:12px;color:var(--muted);margin-top:12px}.value{white-space:pre-wrap;overflow-wrap:anywhere}.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}.success{color:#1c6b42}.failure{color:#a3442f}.state-score{font-size:25px;font-weight:700;color:var(--blue)}.state-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin:14px 0}details{margin-top:12px}summary{cursor:pointer;font-size:13px;color:var(--blue)}pre{white-space:pre-wrap;overflow-wrap:anywhere;word-break:break-word;font-size:12px;line-height:1.5;background:#eff4f6;padding:12px;border-radius:8px;max-height:350px;overflow:auto;margin-bottom:0}.window{margin-top:16px;border-top:1px solid var(--line);padding-top:16px}.message{white-space:pre-wrap;overflow-wrap:anywhere;word-break:break-word;min-height:28px}.message-card{background:#f4f9fa}.empty{color:#71858b;font-style:italic}.goal{padding:7px 0;border-bottom:1px dashed var(--line)}.warning{color:#93401e}.toolbar-note{margin-top:12px}
@media(max-width:780px){main{padding:16px}.grid,.state-grid{grid-template-columns:1fr}.panel{padding:16px}.field{max-width:none}.controls{gap:10px}}
@media(max-width:380px){main{padding:10px}.panel{padding:12px;border-radius:12px}.controls .field{flex-basis:120px}select,button{padding:8px}h2{font-size:18px}.nav button{flex:1}}
</style></head><body><main>
<header><span class="tag">静态快照 · 研究者视图</span><h1>三主体实验逐步回放</h1><p id="run-name"></p><p id="snapshot-meta" class="muted small"></p><div class="banner">此页面仅包含生成时已经写入日志的记录，不自动更新。未结束任务段的当前得分不代表最终结果；私有分析不作为交流内容展示。</div></header>
<section class="panel" aria-label="回放控制"><div class="controls">
<div class="field"><label for="condition">条件</label><select id="condition"></select></div>
<div class="field"><label for="group">群体</label><select id="group"></select></div>
<div class="field"><label for="episode">任务段</label><select id="episode"></select></div>
<div class="field"><label for="step">已结算动作步</label><select id="step"></select></div>
<div class="nav"><button id="prev" type="button">← 前一步</button><button id="next" type="button">后一步 →</button></div>
</div><p id="case-summary" class="toolbar-note small muted"></p><p id="read-warnings" class="small warning"></p></section>
<section id="global-panel" class="panel"><div class="row"><h2><span class="tag warm">研究者全局状态</span> 地点、携带与得分</h2><div class="nav"><button id="before" type="button">动作前</button><button id="after" type="button" class="selected">动作后</button></div></div>
<p class="small muted">此区来自state_before / state_after，包含远处信息及研究者物品编号；是否提供给某个主体，以该主体当步的观察原文为准。它不是主体之间发送的消息。</p>
<div class="row"><div><div id="global-time" class="label"></div><div id="global-score" class="state-score"></div></div><p id="road-state" class="small"></p></div>
<div id="goals"></div><div id="global-agents" class="state-grid"></div><details><summary>查看全局物品状态（研究者编号）</summary><pre id="global-items"></pre></details></section>
<section id="messages-panel" class="panel"><h2>四个通信窗口 · 实际广播</h2><div class="banner small"><strong>下面是研究者的完整通信记录。</strong> 并列展示不表示主体提前看到了同窗消息或延迟消息。<div id="timing-note"></div></div><div id="windows"></div></section>
<section id="agents-panel" class="panel"><h2 id="agents-heading">三主体当步观察与实际行动</h2><p id="agent-input-note" class="small muted"></p><div id="agents" class="grid"></div></section>
<footer class="small muted"><p>仅回放原始环境记录与正式广播。消息长短、重复或主体声称的计划，不用于判定语言已经形成。</p><p id="source"></p></footer>
</main><script id="replay-data" type="application/json">__DATA_JSON__</script>
<script>
'use strict';
const data=JSON.parse(document.getElementById('replay-data').textContent);
const $=id=>document.getElementById(id), agents=['A','B','C'];
const conditionNames={natural:'自然语言 · 局部观察',full_information:'完整信息控制',immediate:'即时符号交流',delayed:'延迟符号共享'};
const statusNames={completed:'运行已结束',running:'运行中',stopped_for_rule_clarity:'已中止以补齐规则说明',failed:'运行中断/失败'};
let stateSide='after';
const text=(id,value)=>{$(id).textContent=value??'未记录'};
function node(tag,content='',cls=''){const e=document.createElement(tag);e.textContent=content;if(cls)e.className=cls;return e;}
function values(rows,key){return [...new Set(rows.map(row=>String(row[key]??'未知')))].sort((a,b)=>a.localeCompare(b,undefined,{numeric:true}));}
function options(id,list,format=x=>x){const select=$(id);select.replaceChildren();for(const value of list){const e=node('option',format(value));e.value=String(value);select.append(e);}}
function filtered(includeStep=false){return data.entries.filter(row=>String(row.condition)===$('condition').value&&String(row.group)===$('group').value&&String(row.episode)===$('episode').value&&(!includeStep||String(row.id)===$('step').value)).sort((a,b)=>Number(a.step)-Number(b.step)||a.id-b.id);}
function current(){return filtered(true)[0];}
function fillGroups(){const rows=data.entries.filter(row=>String(row.condition)===$('condition').value);options('group',values(rows,'group'),x=>'群体 '+x);fillEpisodes();}
function fillEpisodes(){const rows=data.entries.filter(row=>String(row.condition)===$('condition').value&&String(row.group)===$('group').value);options('episode',values(rows,'episode'),x=>'任务段 '+x);fillSteps();}
function fillSteps(){const rows=filtered();options('step',rows.map(row=>String(row.id)),id=>'第 '+rows.find(row=>String(row.id)===id).step+' 步');render();}
function itemLabel(item){if(!item)return'未知物品';let s=[item.handle,item.condition,item.length,item.kind].filter(x=>x!==null&&x!==undefined&&x!=='').join(' ');if(item.kind==='木材'&&typeof item.processed==='boolean')s+=' · '+(item.processed?'可搬运':'不可搬运');if(item.carried_by?.length)s+=' · 搬运者 '+item.carried_by.join('、');return s;}
function itemsLabel(items){return Array.isArray(items)?(items.map(itemLabel).join('\n')||'无'):'未记录';}
function field(parent,label,value){parent.append(node('div',label,'label'),node('div',value,'value'));}
function globalState(row){const state=row[stateSide];$('before').classList.toggle('selected',stateSide==='before');$('after').classList.toggle('selected',stateSide==='after');$('before').setAttribute('aria-pressed',String(stateSide==='before'));$('after').setAttribute('aria-pressed',String(stateSide==='after'));
  $('goals').replaceChildren();$('global-agents').replaceChildren();if(!state){text('global-score','该状态未记录');text('global-time','');text('road-state','');text('global-items','');return;}
  const goals=Array.isArray(state.goals)?state.goals:[], total=goals.reduce((s,g)=>s+(Number(g.quantity)||0),0), delivered=goals.reduce((s,g)=>s+(Number(g.delivered)||0),0);
  text('global-time',(stateSide==='before'?'动作前':'动作后')+' · 物理时钟 '+(state.t??'未知'));
  text('global-score',total?`${delivered} / ${total} 单位 · ${(delivered/total*100).toFixed(1)}%`:'未记录目标分母');
  text('road-state',state.closed_road?'已发生道路关闭：'+state.closed_road.join(' ↔ '):'本版道路永久开放，无道路事件');
  for(const goal of goals)$('goals').append(node('div',[goal.condition,goal.length,goal.kind].filter(Boolean).join(' ')+' → '+goal.destination+' · 已交付 '+goal.delivered+'/'+goal.quantity,'goal'));
  for(const agent of agents){const card=node('div','','card');card.append(node('h3',agent+' · '+(state.positions?.[agent]??'位置未记录')));const inv=state.inventory?.[agent];let invText='未记录';if(inv){const pairs=Object.entries(inv).filter(([,id])=>id!==null&&id!==undefined);invText=pairs.map(([slot,id])=>(slot==='tool'?'工具位':'普通携带位')+'：'+itemLabel(state.items?.[id])+' ['+id+']').join('\n')||'空手';}field(card,'研究者看到的实际携带',invText);$('global-agents').append(card);}
  text('global-items',JSON.stringify(state.items,null,2));
}
function renderMessages(row){const delayed=row.condition==='delayed';text('timing-note',delayed?'延迟组：四个窗口全部生成完后，才向主体公开其他人的本步消息；发言时自己的此前输出可见，过去动作步已收到的消息也可见。':'即时组：每个窗口的三条消息全部收齐后才公开；同一窗口内相互不可见。下一窗口才可回应此前已公开消息。');$('windows').replaceChildren();
  for(let window=1;window<=4;window++){const block=node('div','','window');block.append(node('h3','窗口 '+window));const grid=node('div','','grid');for(const agent of agents){const card=node('div','','card message-card');card.append(node('div',agent+' 的正式广播','label'));const records=(row.messages||[]).filter(msg=>msg.window===window&&msg.sender===agent);if(!records.length)card.append(node('div','该窗口无已记录消息','message empty'));else for(const message of records)card.append(node('div',message.text===''?'[空消息]':message.text,'message'+(message.text===''?' empty':'')));grid.append(card);}block.append(grid);$('windows').append(block);}
}
function renderAgents(row){const full=row.condition==='full_information';text('agents-heading',full?'主体当时收到的观察（完整信息控制）与实际动作':'三主体当步局部观察与实际行动');text('agent-input-note',(full?'本条件实际向每个主体提供了完整当前信息，包括各地点状态和含交付进度的任务板；本版道路永久开放，无道路事件。下面的观察原文保留完整信息，不仅是局部观察。':'本条件各主体收到自己的局部观察。')+' 观察取自动作前的observations，不含完整历史记忆，因此不是完整决策上下文。四个窗口结束后，三人独立提交动作，再同时结算；切换上方研究者动作前/后状态不会改写这里的当步观察。');$('agents').replaceChildren();for(const agent of agents){const observation=row.observations?.[agent], card=node('article','','card'), title=node('h3','','agent-title');title.append(node('span',agent,'agent-dot'),node('span',full?'收到的完整信息观察与动作':'局部观察与动作'));card.append(title);
  if(!observation){card.append(node('p','本步观察未记录'));}else{field(card,'动作前实际位置',observation.location??'未记录');field(card,'动作前自己实际携带',itemsLabel(observation.carried_items));field(card,'本地地上物品',itemsLabel(observation.ground_items));field(card,'同地可见伙伴',Array.isArray(observation.nearby_agents)?(observation.nearby_agents.map(p=>p.agent+(p.visible_carried_items?.length?'（可见携带：'+itemsLabel(p.visible_carried_items)+'）':'')).join('\n')||'无'):'未记录');field(card,'任务板是否在本步观察中','task_board'in observation?'有，见观察原文':'本步未提供（历史记忆可能包含旧信息）');const detail=node('details');detail.append(node('summary','查看该主体实际观察原文'),node('pre',JSON.stringify(observation,null,2)));card.append(detail);}
  const selected=row.selections?.[agent], action=row.actions?.[agent], feedback=row.feedback?.[agent];field(card,'四窗结束后实际提交的动作',selected?(selected.id+'：'+selected.description):action?JSON.stringify(action):'未记录');
  if(feedback){const kind=action?.kind;let label=feedback.action_succeeded===true?(kind==='wait'?'等待已执行':'执行反馈：成功'):feedback.action_succeeded===false?'执行反馈：未发生或未被接收':'执行状态未记录';const p=node('p',label,feedback.action_succeeded===true?'success':'failure');card.append(p);field(card,'该主体实际收到的结果',feedback.result??JSON.stringify(feedback));}else field(card,'动作结果','未记录');$('agents').append(card);
}}
function render(){const row=current();const has=Boolean(row);for(const id of ['global-panel','messages-panel','agents-panel'])$(id).hidden=!has;if(!has){text('case-summary','本快照尚无已结算动作记录。');$('prev').disabled=$('next').disabled=true;return;}const rows=filtered(), index=rows.findIndex(r=>r.id===row.id);$('prev').disabled=index<=0;$('next').disabled=index>=rows.length-1;
  const completed=data.episodes.find(e=>String(e.condition)===String(row.condition)&&String(e.group)===String(row.group)&&String(e.episode)===String(row.episode)&&String(e.phase)===String(row.phase));
  const finalScore=completed&&typeof completed.score==='number'&&Number.isFinite(completed.score)?(completed.score*100).toFixed(1)+'%':'未记录';text('case-summary','阶段：'+row.phase+'；本快照已有 '+rows.length+' 个结算步。'+(completed?` 该任务段已有结束记录：最终完成度 ${finalScore}，共 ${completed.steps??'未记录'} 步。`:' 该任务段尚无结束记录，不把当前得分作最终结果。'));
  globalState(row);renderMessages(row);renderAgents(row);
}
text('run-name',data.run_name);text('snapshot-meta','生成于 '+data.generated_at+' · '+(statusNames[data.status.status]||data.status.status||'运行状态未记录')+' · '+data.entries.length+' 条已结算步记录');text('source','数据来源：'+data.source_directory);text('read-warnings',data.warnings.join('\n'));
options('condition',values(data.entries,'condition'),value=>conditionNames[value]||value);
$('condition').addEventListener('change',fillGroups);$('group').addEventListener('change',fillEpisodes);$('episode').addEventListener('change',fillSteps);$('step').addEventListener('change',render);
for(const [id,offset]of [['prev',-1],['next',1]])$(id).addEventListener('click',()=>{const rows=filtered(),index=rows.findIndex(row=>String(row.id)===$('step').value),next=rows[index+offset];if(next){$('step').value=String(next.id);render();}});
for(const side of ['before','after'])$(side).addEventListener('click',()=>{stateSide=side;const row=current();if(row)globalState(row);});
fillGroups();
</script></body></html>'''


def render_snapshot(snapshot):
    # Escaping '<' prevents an observed message from closing the JSON script.
    # Text shown by the page is then assigned with textContent, never parsed HTML.
    payload = json.dumps(snapshot, ensure_ascii=False, separators=(",", ":"))
    payload = payload.replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    return HTML.replace("__DATA_JSON__", payload, 1)


def write_snapshot(run_dir, out):
    snapshot = collect_snapshot(run_dir)
    target = Path(out).resolve()
    if target.is_dir() or not target.suffix:
        target = target / "replay.html"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name("." + target.name + ".tmp")
    temporary.write_text(render_snapshot(snapshot), encoding="utf-8")
    temporary.replace(target)
    return {"output": str(target), "steps_recorded": len(snapshot["entries"]),
        "status_at_snapshot": snapshot["status"].get("status"), "generated_at": snapshot["generated_at"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(write_snapshot(args.run_dir, args.out), ensure_ascii=False))


if __name__ == "__main__":
    main()

"""Static snapshot data safety and DOM-logic tests; no browser/model service."""

from copy import deepcopy
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

from qwen_language_v3.replay_snapshot import collect_snapshot, render_snapshot, write_snapshot


# This tiny DOM test double executes the page's actual JS and change/click
# handlers. It checks data/control logic, not visual layout or browser rendering.
NODE_HARNESS = r'''
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const html=fs.readFileSync(process.argv[1],'utf8');
class Element{
 constructor(tag){this.tagName=tag;this.children=[];this._text='';this._value=null;this.listeners={};this.attrs={};this.hidden=false;this.disabled=false;this.className='';this.classList={toggle:()=>{}};}
 set textContent(value){this._text=String(value??'');this.children=[];}
 get textContent(){return this._text+this.children.map(x=>x.textContent).join('');}
 set value(value){this._value=String(value);}
 get value(){return this._value??(this.tagName==='select'?(this.children[0]?.value??''):'');}
 append(...children){this.children.push(...children);}
 replaceChildren(...children){this.children=children;this._text='';if(this.tagName==='select')this._value=null;}
 addEventListener(event,listener){this.listeners[event]=listener;}
 setAttribute(name,value){this.attrs[name]=value;}
 fire(event){this.listeners[event]?.({target:this});}
}
const elements={};for(const match of html.matchAll(/<([a-z]+)[^>]*\bid="([^"]+)"/g))elements[match[2]]=new Element(match[1]);
const payload=html.match(/<script id="replay-data" type="application\/json">([\s\S]*?)<\/script>/)[1];
elements['replay-data'].textContent=payload;
const code=html.match(/<script>\s*([\s\S]*?)<\/script>/)[1];
const document={getElementById:id=>elements[id],createElement:tag=>new Element(tag)};
const context=vm.createContext({document,console});vm.runInContext(code,context);
const change=(id,value)=>{elements[id].value=value;elements[id].fire('change');};
let rendered=0;
for(const condition of [...elements.condition.children].map(x=>x.value)){
 change('condition',condition);
 if(condition==='delayed')assert(elements['timing-note'].textContent.includes('四个窗口全部生成完后'));
 else assert(elements['timing-note'].textContent.includes('同一窗口内相互不可见'));
 if(condition==='full_information')assert(elements['agents-heading'].textContent.includes('主体当时收到的观察（完整信息控制）'));
 else assert(elements['agents-heading'].textContent.includes('局部观察'));
 for(const group of [...elements.group.children].map(x=>x.value)){
  change('group',group);
  for(const episode of [...elements.episode.children].map(x=>x.value)){
   change('episode',episode);
   const ids=[...elements.step.children].map(x=>x.value);
   if(ids.length>1){change('step',ids[0]);elements.next.fire('click');assert.strictEqual(elements.step.value,ids[1]);elements.prev.fire('click');assert.strictEqual(elements.step.value,ids[0]);}
   for(const id of ids){
    change('step',id);assert.strictEqual(elements.agents.children.length,3);assert.strictEqual(elements.windows.children.length,4);
    const local=elements.agents.textContent;
    elements.before.fire('click');assert(elements['global-time'].textContent.includes('动作前'));assert.strictEqual(elements.agents.textContent,local);
    elements.after.fire('click');assert(elements['global-time'].textContent.includes('动作后'));assert.strictEqual(elements.agents.textContent,local);
    assert.strictEqual(elements['global-agents'].children.length,3);rendered++;
   }
  }
 }
}
assert.strictEqual(rendered,JSON.parse(payload).entries.length);
console.log(JSON.stringify({dom_logic_steps_rendered:rendered,conditions_tested:elements.condition.children.length,visual_browser_test:false}));
'''


def make_fixture(parent):
    path = Path(parent) / "run"
    path.mkdir()
    state = {"t": 0, "max_steps": 12, "positions": {a: "营地" for a in "ABC"},
        "inventory": {a: {"tool": None, "cargo": None} for a in "ABC"},
        "goals": [{"kind": "纤维", "length": "长", "condition": "干", "quantity": 1, "delivered": 0, "destination": "营地"}],
        "items": {}, "witness": "DO_NOT_SHOW_WITNESS", "seed": 999,
        "event": {"occurred": False, "at_step": 7, "edge": ["营地", "林地"]}}
    rows = []
    for index, (condition, group, episode, step) in enumerate([
            ("immediate", 17, 1, 1), ("immediate", 17, 1, 2), ("immediate", 17, 2, 1),
            ("immediate", 29, 1, 1), ("delayed", 17, 1, 1), ("full_information", 17, 1, 1)]):
        after = deepcopy(state)
        after["t"] = step
        rows.append({"phase": "pilot", "condition": condition, "group": group, "episode": episode, "step": step,
            "observations": {a: {"agent": a, "location": "营地", "ground_items": [], "carried_items": [], "nearby_agents": []} for a in "ABC"},
            "messages": [{"sender": a, "window": window, "step": step,
                "text": '</script><script>globalThis.XSS=1</script>' if index == 0 and a == "A" else "@#",
                "private_analysis": "DO_NOT_SHOW_ANALYSIS"} for window in range(1, 5) for a in "ABC"],
            "private_analysis": "DO_NOT_SHOW_ANALYSIS",
            "actions": {a: {"kind": "wait"} for a in "ABC"},
            "feedback": {a: {"action_succeeded": True, "result": "等待。"} for a in "ABC"},
            "state_before": state, "state_after": after, "score": 0})
    (path / "steps.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    (path / "status.json").write_text(json.dumps({"status": "running"}), encoding="utf-8")
    (path / "inference.jsonl").write_text('DO_NOT_READ_PRIVATE_LOG', encoding="utf-8")
    return path


class ReplayTests(unittest.TestCase):
    def test_snapshot_has_real_messages_but_no_private_inference_or_future_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = make_fixture(tmp)
            snapshot = collect_snapshot(run)
            text = json.dumps(snapshot)
            self.assertEqual(len(snapshot["entries"]), 6)
            self.assertIn('globalThis.XSS=1', snapshot["entries"][0]["messages"][0]["text"])
            self.assertNotIn("DO_NOT_SHOW_ANALYSIS", text)
            self.assertNotIn("DO_NOT_READ_PRIVATE_LOG", text)
            self.assertNotIn("DO_NOT_SHOW_WITNESS", text)
            self.assertNotIn('"at_step"', text)
            self.assertEqual(snapshot["episodes"], [])

    def test_html_embeds_untrusted_text_safely_and_writes_outside_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = make_fixture(tmp)
            before = {path.name: path.read_bytes() for path in run.iterdir()}
            result = write_snapshot(run, Path(tmp) / "review_outputs" / "replay.html")
            html = Path(result["output"]).read_text(encoding="utf-8")
            self.assertEqual(html.count("</script>"), 2)
            self.assertNotIn("innerHTML", html)
            payload = re.search(r'<script id="replay-data" type="application/json">(.*?)</script>', html, re.S).group(1)
            self.assertIn("\\u003c/script>", payload)
            self.assertIn("globalThis.XSS=1", json.loads(payload)["entries"][0]["messages"][0]["text"])
            self.assertIn("同一窗口内相互不可见", html)
            self.assertIn("四个窗口全部生成完后", html)
            self.assertEqual(before, {path.name: path.read_bytes() for path in run.iterdir()})

    @unittest.skipUnless(shutil.which("node"), "Node unavailable; Python data checks still apply")
    def test_actual_page_javascript_controls_with_dom_test_double(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = make_fixture(tmp)
            result = write_snapshot(run, Path(tmp) / "review" / "replay.html")
            completed = subprocess.run([shutil.which("node"), "-e", NODE_HARNESS, result["output"]], capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(json.loads(completed.stdout)["dom_logic_steps_rendered"], 6)


if __name__ == "__main__":
    unittest.main()

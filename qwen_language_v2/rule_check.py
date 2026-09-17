"""Independent declarative rule checks; not a replacement for task controls."""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

from .backend import Backend
from .environment import RULES
from .run import DEFAULT_MODEL, write_json

CASES = [
    ('ground_is_not_carried', '你站在河滩，地上有斧具和纤维，你实际携带的物品为空。你现在已经持有斧具了吗？', 0),
    ('all_attributes_match', '任务要1件干长纤维交付营地，尚未交付任何单位。你在营地携带湿短纤维，选择交付这件物品会被接收吗？', 0),
    ('fiber_needs_no_axe', '任务要1件干长纤维交付营地，尚未交付任何单位。你在营地携带干长纤维，没有斧具。你现在选择交付这件物品会被接收吗？', 1),
    ('processing_preserves_attributes', '你持斧加工了一件湿长木材。加工成功之后，它是否仍然是湿长木材，只是加工状态变了？', 1),
    ('joint_capacity', 'A和B同在林地，地上有已加工长木材。A的普通位已装一件纤维，B普通位空闲。A不先放下纤维，两人能直接开始共同搬运这件木材吗？', 0),
    ('cut_does_not_pickup', '你在林地持斧，地上有未加工的短木材；你执行加工且成功。木材是否仍留在地面，尚未自动进入你的普通携带位？', 1),
    ('drop_does_not_deliver', '你把符合需求的物品搬到指定交付地点，然后选择放下，没有选择交付。这个物品是否已经计入交付完成量？', 0),
    ('matching_joint_action', 'A和B同在林地，普通携带位均空闲，地上有一件已加工长木材，两条相关道路均开放。双方分别选这件木材、互为搭档、共同搬到河滩。该共同搬运能成功吗？', 1),
]


def main():
    out = Path(__file__).parent/'results'/('rule_check_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    out.mkdir(parents=True)
    backend = Backend(DEFAULT_MODEL,out/'inference.jsonl')
    results=[]
    for i,(name,question,expected) in enumerate(CASES):
        prompt=[{'role':'system','content':RULES+'\n这是独立的规则理解检查，不参与任何群体互动。只判断题目所述当前状态。编号0表示否，编号1表示是。'},
                {'role':'user','content':question+'\n可选判断：编号0=否；编号1=是。'}]
        answer=int(backend.decide(prompt,mode='action',choices=[0,1],seed=1000+i,label={'phase':'rule_check','case':name},
            final_instruction='请回答上面的真假判断题，不选择物理动作。只输出一个数字：0表示否，1表示是。不要解释。'))
        row={'case':name,'question':question,'expected':expected,'answer':answer,'correct':answer==expected}
        results.append(row)
        print(json.dumps(row,ensure_ascii=False),flush=True)
    write_json(out/'results.json',{'scope':'declarative rule checks only; full local/natural and full-information task controls are still required','correct':sum(r['correct'] for r in results),'total':len(results),'cases':results,'backend':backend.stats()})
    print(json.dumps({'run_dir':str(out.resolve()),'correct':sum(r['correct'] for r in results),'total':len(results)}),flush=True)


if __name__=='__main__':
    main()

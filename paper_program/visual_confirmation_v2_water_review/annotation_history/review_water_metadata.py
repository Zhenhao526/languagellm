"""Root's manual metadata-only annotations; no pixel access or new requests."""
import hashlib
import html
import json
from pathlib import Path
import re
from collections import Counter

PROJECT = Path(__file__).resolve().parent.parent
SOURCE = PROJECT / 'paper_program/visual_confirmation_v2_water'
OUT = PROJECT / 'paper_program/visual_confirmation_v2_water_review'

# All 79 eligible representatives in the immutable author_groups list order.
# E: explicit water/glass evidence; U: insufficient or ambiguous text;
# X: description identifies an incompatible medium, mixture or scene.
# C: water with food/drink; P: physical experiment/staging; D: ordinary water;
# O: other material/scene; ?: unresolved context. These are NOT acceptance labels.
ANNOTATIONS = '''
1|E|C|法语描述明确咖啡旁一杯水；多物体场景，水的可辨性未看。
2|E|?|标题日语“水”，描述仅编号；容器与饮用场景未确认。
3|U|C|标题为蜜瓜汽水；不能据泛类断定另有纯水。
4|U|C|标题和描述仅咖啡店名，水缺少文字支持。
5|E|P|橙片落入装水玻璃容器的飞溅；食物与水混合场景。
6|E|C|西语描述面包与水、烛光；水的可辨性待像素核验。
7|U|P|描述奇异果落入玻璃杯，没有明确液体成分。
8|E|D|西语标题与描述明确一杯水。
9|U|?|微量元素系列，未明确水或杯中成分。
10|U|C|标题明确咖啡，未文字说明旁边是否有水。
11|U|C|标题与描述为餐馆名；水仅有检索类别支持。
12|U|?|WASH活动描述未说明具体水杯；机构作者也需追溯。
13|U|O|俄语描述为吸附剂产品，不能认作饮水照片。
14|U|C|描述咖啡文化，未明确水杯。
15|U|P|描述折射摄影，没有说明介质就是水。
16|E|C|标题和描述明确浓缩咖啡搭配水、糖和饼干。
17|E|P|描述水杯中的杆演示折射；实验道具场景。
18|U|P|描述非接触温度计，水杯及用途没有明确文字说明。
19|E|D|描述液态水和冰两相；容器和实际可辨性待看。
20|E|P|标题和描述明确水中草莓；混合资源摄影。
21|U|?|描述机场休息室，未明确杯中物质。
22|E|D|标题与描述明确一杯冷水。
23|E|C|明确含冰、柠檬和吸管的水；需固定添加物适用边界。
24|E|C|意语明确咖啡、水和红糖；多饮品场景。
25|U|C|标题早餐，描述为文化遗产模板；没有水的具体描述。
26|U|?|供水研讨会并不说明照片中的水杯；机构作者待追溯。
27|E|P|标签说明水、玻璃与滴落实验；未证明普通饮水场景。
28|U|?|标题及描述编号不足以辨别水与场景。
29|X|O|瑞典语描述明确精液在水杯中；不适合本轮饮水资源类别。
30|U|?|“spökvatten”的实际杯中成分未由当前材料可靠确定。
31|U|C|明确玛奇朵咖啡，未文字证实另有水。
32|U|?|描述仅“creativity”，水和处理方式均不明。
33|U|?|活动名称未说明画面中水的状态。
34|U|?|飞机采购发布会背景，未明确水；机构署名非独立摄影者证明。
35|U|C|咖喱饭与餐馆描述，没有明确水杯文字。
36|E|P|明确水杯中玫瑰花，花瓶式用途需与饮水分开。
37|U|C|卡布奇诺描述，没有文字证实旁边水杯。
38|U|?|景点名称，未明确水杯或介质。
39|U|?|“the water man”不能单独确定杯中饮水与摄影处理。
40|U|C|炸猪排咖喱描述，没有文字证实水。
41|U|C|描述自制面包；标题拼写不足以明确水杯。
42|E|D|西语标题与描述明确一杯水。
43|U|C|荞麦面与炸肉饼描述，没有明确水。
44|U|?|发布会题名；描述署名Lauri Heikkinen与Artist机构不同，需建立关联。
45|U|C|黑咖啡名称，不能直接证明画面旁有纯水。
46|E|D|描述明确glass of water。
47|E|P|标题明确水杯中的回形针；实验物体不等于普通饮水。
48|E|C|日语明确绿茶与“お冷”（冷水）；多饮品场景。
49|X|O|描述明确使用Photoshop制作照片合成；不作自然照片候选。
50|U|C|米饭菜名，没有水的直接文字说明。
51|E|D|描述明确一杯水。
52|X|P|描述苯与水的折射对比；不能直接当饮水资源场景。
53|E|P|明确玻璃杯中用搅拌器制造水旋涡；人为实验场景。
54|X|O|德语明确水瓶中装克里特拉基酒；不是水的证据。
55|E|P|明确水杯与勺子的折射演示。
56|U|C|海滩refreshments没有明确饮料成分。
57|U|P|描述所谓水晶水与石头；记录原说法但不采信其能量宣称，液体与用途待核。
58|E|P|明确两个同水量杯子，搅拌旋转实验。
59|E|C|中文描述拉面店冰水；照片中的水杯清晰性仍未确认。
60|U|?|“glass half full”缺少杯中物质与图像处理的明确描述。
61|X|O|标题与描述为泡腾片，不能直接标为纯饮水。
62|E|D|描述明确日落时的一杯水。
63|U|C|咖啡店Melange，未文字说明水杯。
64|U|C|详细菜品描述没有水的直接说明。
65|E|P|明确人在水杯上方加粉；是否已混入及任务适用边界需看。
66|U|C|拉面描述，没有明确水杯。
67|U|C|仅咖啡店描述，水成分未明。
68|X|O|明确两条河的水色样本；不直接作为可饮用资源。
69|X|P|描述激光穿过超声水雾，所指并非普通杯中液态饮水。
70|E|D|明确窗前阳光反射的一杯水。
71|U|P|“火星上有水”文字可能含双关/摆拍，材料不能证明普通饮水图。
72|U|C|巴黎露台咖啡，没有明确水杯。
73|U|C|维也纳咖啡与蛋糕描述，没有明确水杯。
74|U|O|标题为概念艺术装置，Artist仅原上传者措辞；作者与作品范围待核。
75|E|P|德语明确水滴入饮水杯；滴水摄影语境。
76|U|P|标题水与光、描述反射光影，具体介质及杯中物质待核。
77|U|C|菜谱摄影，未明确水杯；署名要求仍须随源保存。
78|E|C|标题及描述明确香蕉与水；两资源共现可能构成任务混淆。
79|X|O|标题像水杯，但描述为透过雨窗看夜景；元数据内部矛盾。
'''


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def clean(s): return re.sub(r'\s+', ' ', html.unescape(re.sub('<[^>]+>', ' ', s))).strip()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = {name: sha(SOURCE / name) for name in ('author_groups.json', 'manifest.json', 'completion_receipt.json')}
    groups = [g for g in json.loads((SOURCE / 'author_groups.json').read_text())['groups'] if g['eligible_representative']]
    rows = {r['id']: r for r in json.loads((SOURCE / 'manifest.json').read_text())['images']}
    decisions = [s.split('|', 3) for s in ANNOTATIONS.strip().splitlines()]
    assert len(groups) == len(decisions) == 79
    output = []
    for index, (g, (n, evidence, context, reason)) in enumerate(zip(groups, decisions), 1):
        assert int(n) == index
        r = rows[g['eligible_representative']]
        output.append(dict(order=index, id=r['id'], cluster_id=g['cluster_id'], title=r['title'],
            description=clean(r.get('description_html', '')), artist=r['artist_text'],
            artist_html=r.get('artist_html'), credit_html=r.get('credit_html'),
            author_keys=r['author_keys'], original_source_keys=r['original_source_keys'],
            license_as_recorded=r['license'], source_page=r['source_page'],
            water_text_evidence=evidence, scene_context=context, rationale=reason,
            pixel_review='not_accessed', model_use='none', author_identity='not_independently_verified',
            source_flag='uploader_only_not_author' if index == 74 else 'organization_or_named_credit_needs_resolution' if index in (12, 26, 33, 34, 44) else 'metadata_attribution_only'))
    result = dict(status='complete_metadata_review_only', reviewer='root, independent of original collector',
        inputs_sha256=inputs, annotation_source_sha256=sha(Path(__file__)), count=79,
        evidence_legend={'E': 'title/description explicitly mentions water; not visual acceptance',
            'U': 'insufficient, ambiguous or category-only textual support',
            'X': 'incompatible substance, edited medium or internally conflicting scene metadata'},
        context_legend={'C': 'food/drink coexistence', 'P': 'experiment or staging', 'D': 'ordinary water description', 'O': 'other scene/material', '?': 'unknown'},
        counts=dict(Counter(r['water_text_evidence'] for r in output)),
        new_usable_images=0, pixel_requests=0, new_network_requests=0, model_calls=0,
        scope='All 79 prior automatic representatives reviewed. Annotation labels are descriptive, post-collection and not a new acceptance gate. Source and license are copied metadata, not legal or independent identity verification.',
        rows=output)
    dest = OUT / 'root_independent_review.json'
    assert not dest.exists()
    dest.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    assert inputs == {n: sha(SOURCE / n) for n in inputs}
    print(json.dumps({k: result[k] for k in ('status','count','counts','new_usable_images')}, ensure_ascii=False))


if __name__ == '__main__': main()

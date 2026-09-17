"""Agent prompts receive only their own observations/history and delivered text."""
from __future__ import annotations

from copy import deepcopy
import json


def current_status(observation):
    """Literal presentation of own observation, without inferred plans or facts."""
    def item_text(item):
        text = item.get('handle','') + '：' + item.get('condition','') + item.get('length','') + item.get('kind','')
        if item.get('kind') == '木材':
            text += '（已加工）' if item.get('processed') else '（未加工）'
        return text
    carried = '；'.join(item_text(x) for x in observation.get('carried_items',[])) or '无；你现在空手'
    ground = '；'.join(item_text(x) for x in observation.get('ground_items',[])) or '无'
    return {'你现在的位置':observation.get('location'), '你现在实际携带':carried,
        '这里地上的物品':ground, '读取说明':'地上的物品还没有拿到手中。拿取和移动是不同动作，广播中的计划不等于动作已经执行。carried_items只列你实际携带的物品。'}


def build_prompt(agent, *, rules, observation, history, transcript, condition, window=None,
                 remaining=None, menu=None, full_information=None):
    channel = ('本能力检查允许广播自然语言，每窗口至多约50个汉字。' if condition in ('natural','full_information') else
        '广播只可使用8个符号@#%&*+=~。符号没有给定含义，没有共享词典或规定格式。每条0至32字符，每人每动作步总计至多64字符，可以沉默。')
    timing = ('每窗口消息收齐后立即公开；下一窗口可以回应已公开消息。' if condition in ('immediate','natural','full_information') else
        '本动作步四个窗口结束后才一次公开所有人的消息。发言时只能看到自己本步先前发送内容及过去动作步的记录；第四窗口后行动前能读到全部消息。')
    system = (f'你是持续合作群体中的{agent}，伙伴是另外两位A/B/C。\n{rules}\n'
        '每个动作步先获得自己的局部观察，接着固定四个同步广播窗口，再同时提交一个物理动作。'
        '不预设谁指挥谁，各人能力相同，身份跨任务段持续。以截止前完成目标单位最多为先，相同完成度下尽快完成。\n'
        +channel+'\n'+timing+'\n'
        '本地物品句柄只对自己有效，伙伴有不同句柄。可选动作编号仅供自己操作，编号在各次观察中可能变化。'
        '场景每段重新布置，过去物品句柄和需求不能代替当前观察。记录里其他人的广播是对方说的话，不是环境保证的事实。'
        '四个广播窗口期间世界冻结；说要拿取、移动或交付不代表已经执行。判断自己的当前真实位置和携带物以本步观察为准。')
    if condition == 'full_information':
        system += '\n本次是独立的完整信息能力控制：你能查看当前各地点状态和需求；物理动作仍受所在地和携带规则限制。'
    data = {'你自己的历史':deepcopy(history), '当前私有观察':deepcopy(observation),
        '当前观察的直读':current_status(observation), '当前已可见广播':deepcopy(transcript)}
    if menu is not None:
        data['本次可选动作'] = [{'编号':a['id'], '动作':a['description']} for a in menu]
        data['当前决策'] = '四个窗口已结束。独立选择一个动作。其他人的未执行动作不可见。'
    else:
        data['当前决策'] = {'广播窗口':window, '本步剩余符号额度':remaining}
    if full_information is not None:
        data['仅此独立完整信息能力控制可见的全体观察'] = full_information
    return [{'role':'system','content':system}, {'role':'user','content':json.dumps(data,ensure_ascii=False,separators=(',',':'))}]


def remember(histories, *, observations, messages, actions, feedback, episode_end=None):
    for agent in ('A','B','C'):
        record = {'观察':deepcopy(observations[agent]), '公开广播':deepcopy(messages),
                  '自己执行的动作':deepcopy(actions[agent]), '自己可见结果':deepcopy(feedback[agent])}
        if episode_end is not None:
            record['任务段结束'] = deepcopy(episode_end)
        histories[agent].append(record)

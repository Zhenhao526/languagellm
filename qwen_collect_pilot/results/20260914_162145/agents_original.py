"""Private observations and episodic memory; no shared semantic dictionary."""
from __future__ import annotations

from copy import deepcopy
import json

from env import World

RULES = '''你参与一个三人采集合作实验，目标是尽量满足营地需求。
A只采纤维，每轮候选恰有F0和F1，随机放在槽位0、1。
B只采燃料，每轮候选恰有G0和G1，随机放在槽位0、1。
C知道本轮需要哪种纤维和哪种燃料，但不知道两处候选的槽位。
需求与位置在每轮独立随机变化；过去需求不能预测本轮需求。
C可广播一条由@#%&组成、至多两个字符的消息，也可以不发送。符号没有指定含义，不存在预先给定的词典，也不规定字符位置的意义。
A和B各选一个本地槽位，双方都选中所需资源才算群体成功。
采集者在结算后知道自己是否匹配以及群体是否成功；C只知道群体是否成功。
你可以利用自己的观察和过去反馈进行选择。你看不到其他人的私有记录。
文字解释和资源名称只用于环境说明；它们不能进入主体间的消息。'''

KNOWN_PROTOCOL = '''本次仅为独立的已知协议能力对照。约定如下：
@@表示需要F0和G0；@#表示需要F0和G1；#@表示需要F1和G0；##表示需要F1和G1。
按此约定选择本地槽位。'''


def validate_observation(role, obs):
    expected = {'role','demand'} if role == 'C' else {'role','resource_kind','slots'}
    if role not in ('A','B','C') or set(obs) != expected or obs['role'] != role:
        raise ValueError('Observation violates role-specific schema')


def build_messages(role, observation, history, received=None, known_protocol=False):
    """This function accepts no World object or other agent's current data."""
    validate_observation(role, observation)
    if role == 'C' and received is not None:
        raise ValueError('Coordinator has no incoming channel in this pilot')
    if role != 'C' and (not isinstance(received,str) or len(received)>2 or any(x not in '@#%&' for x in received)):
        raise ValueError('Invalid channel message')
    role_instruction = ('你是C。只输出本轮广播消息，最多两个特殊字符。不要解释。'
        if role == 'C' else f'你是{role}。只输出你选择的本地槽位编号0或1，不要解释。')
    system = RULES + '\n' + role_instruction
    if known_protocol:
        system += '\n' + KNOWN_PROTOCOL
    payload = {'你的过去互动':history, '本轮私有观察':observation}
    if role != 'C':
        payload['本轮收到的消息'] = received
    return [{'role':'system','content':system},
            {'role':'user','content':json.dumps(payload,ensure_ascii=False,separators=(',',':'))}]


def remember(histories, world: World, message, actions, scores):
    """Feedback is outcome-only, after both actions are committed."""
    for role in ('A','B','C'):
        record = {'观察':world.private_observation(role)}
        if role == 'C':
            record.update({'发送':message,'反馈':{'群体成功':scores['overall']}})
        else:
            record.update({'收到':message,'动作':actions[role],
                '反馈':{'自己匹配':scores[role],'群体成功':scores['overall']}})
        histories[role].append(deepcopy(record))


def empty_histories():
    return {role:[] for role in ('A','B','C')}

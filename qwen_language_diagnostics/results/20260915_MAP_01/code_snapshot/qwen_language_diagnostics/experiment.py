"""Predeclared rule questions and single-agent past-history interventions."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil

from qwen_language_v3.backend import Backend
from qwen_language_v3.environment import RULES, World
from qwen_language_v3.execute_locked_pipeline import verify_frozen
from qwen_language_v3.run import write_json
from .history_cases import load_history_cases

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
SOURCE_RUN = WORK/'qwen_language_v3/results/20260915_152343/capability_full_information'
RULE_SEED = 20260915
RULE_CARD = ('每人只有一个普通物品携带位。空手主体可独自拿取同地的所有纤维，无论长短或干湿；'
    '短木材也可单人拿取。长木材必须两人共同搬运，占双方普通位，开始时双方必须同地且空手。'
    '双方须选择同一物品、互为搭档、同一种共同动作；共同搬运还须选择同一目的地。'
    '成功搬运时两人和物品一起移动；共同负载期间不能独立移动、拿取或交接。')
CAN = '可以由一名空手主体单独拿取'
MUST = '必须由两名主体共同搬运，不能单人拿取'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_rule_cases(rule_source='full'):
    if rule_source not in ('full', 'card'):
        raise ValueError(rule_source)
    cases = []
    for length in ('长', '短'):
        for order in ('can_first', 'must_first'):
            answers = [CAN, MUST] if order == 'can_first' else [MUST, CAN]
            data = {'题目类型':'一次性规则判断，不执行物理动作；没有过去历史或伙伴广播。',
                '当前状态':{'主体':'A','位置':'河滩','普通携带位':'空闲','同地伙伴':[],
                    '地面物品':[{'句柄':f'物{i+1}','种类':'纤维','干湿':c,'长短':l}
                              for i,(c,l) in enumerate((('干','长'),('干','短'),('湿','长'),('湿','短')))]},
                '问题':f'目标是当前地面上的干{length}纤维。一名普通携带位空闲的主体能否独自拿取它？',
                '答案选项':[{'编号':i,'含义':a} for i,a in enumerate(answers)]}
            prompt = [{'role':'system','content':'依据以下规则回答本题。\n'+(RULES if rule_source == 'full' else RULE_CARD)},
                      {'role':'user','content':json.dumps(data,ensure_ascii=False,separators=(',',':'))}]
            cases.append({'case_id':f'R_{length}_{order}_{rule_source}', 'length':length, 'order':order,
                'rule_source':rule_source,'prompt':prompt,'choices':[0,1], 'answers':answers,
                'correct_choice':answers.index(CAN),'correct_meaning':CAN,'seed':RULE_SEED})
    return cases


def evaluate_rule(backend, case):
    first_call = backend.calls + 1
    choice = int(backend.decide(deepcopy(case['prompt']), mode='action',choices=case['choices'],
        seed=case['seed'],temperature=0,label={'phase':'rule_diagnostic','case_id':case['case_id'],
        'rule_source':case['rule_source'],'length':case['length'],'order':case['order']},
        final_instruction='只输出所选答案的一个编号，不要解释。'))
    if choice not in case['choices']:
        raise ValueError('Invalid rule choice')
    return {'kind':'rule','case_id':case['case_id'],'rule_source':case['rule_source'],
        'length':case['length'],'order':case['order'],'choice':choice,'meaning':case['answers'][choice],
        'correct':choice==case['correct_choice'],'correct_choice':case['correct_choice'],
        'call_ids':[first_call,backend.calls]}


def evaluate_history(backend, case, arm):
    if arm not in ('original','empty_history'):
        raise ValueError(arm)
    record, agent = case['original_record'], case['agent']
    prompt = case['prompt'] if arm == 'original' else case['empty_history_prompt']
    menu = record['menus'][agent]
    first_call = backend.calls + 1
    choice = int(backend.decide(deepcopy(prompt),mode='action',choices=[x['id'] for x in menu],
        seed=case['seed'],temperature=0,label={'phase':'history_diagnostic','case_id':case['case_id'],
        'arm':arm,'agent':agent,'episode':record['episode'],'step':record['step']}))
    selection = next(x for x in menu if x['id']==choice)
    # Other actors' submitted actions are only used AFTER this agent decides.
    actions = deepcopy(record['actions'])
    actions[agent] = deepcopy(selection['action'])
    world = World.from_state_dict(record['state_before'])
    score_before = world.score
    feedback = world.step(actions)
    return {'kind':'history','case_id':case['case_id'],'arm':arm,'agent':agent,
        'episode':record['episode'],'step':record['step'],'selection':deepcopy(selection),
        'recorded_selection':deepcopy(record['selections'][agent]),
        'action_matches_recorded':actions[agent]==record['actions'][agent],
        'actions':actions,'feedback':feedback,'score_before':score_before,'score_after_one_step':world.score,
        'state_after':world.state_dict(),'call_ids':[first_call,backend.calls],
        'diagnostic_target_action':deepcopy(case.get('diagnostic_target_action')),
        'matches_diagnostic_target':(actions[agent]==case['diagnostic_target_action']
            if 'diagnostic_target_action' in case else None),
        'scope':'Single actor replacement; other two recorded actions held fixed, not renegotiated.'}


def run_cases(backend, prepared, on_result=lambda row: None):
    result = {'rule_results':[],'history_results':[],'rule_card_triggered':False}
    def rule(case):
        row=evaluate_rule(backend,case); result['rule_results'].append(row); on_result(row)
    for case in prepared['rule_full']:
        rule(case)
    result['rule_card_triggered']=any(not r['correct'] for r in result['rule_results'])
    if result['rule_card_triggered']:
        for case in prepared['rule_card']:
            rule(case)
    for case in prepared['history']:
        for arm in ('original','empty_history'):
            row=evaluate_history(backend,case,arm); result['history_results'].append(row); on_result(row)
    return result


def source_paths():
    names = ('__init__.py','experiment.py','history_cases.py')
    sources={f'qwen_language_diagnostics/{n}':ROOT/n for n in names}
    frozen=json.loads((WORK/'qwen_language_v3/core_source_frozen.json').read_text())
    sources.update({f'qwen_language_v3/{n}':WORK/'qwen_language_v3'/n for n in frozen['source_sha256']})
    return sources


def prepare(out):
    verify_frozen()
    history=load_history_cases(SOURCE_RUN)
    for case in history:
        record,agent=case['original_record'],case['agent']
        if case['case_id']=='S_B_step2':
            target={'kind':'move','destination':'营地'}
        elif case['case_id']=='S_C_step9':
            other=record['actions']['A']
            handles=record['state_before']['handles']
            oid=next(oid for oid,handle in handles['A'].items() if handle==other['item'])
            target={'kind':'carry_together','item':handles[agent][oid],
                    'partner':'A','destination':other['destination']}
        else:
            raise ValueError('Unexpected history diagnostic case')
        if not any(m['action']==target for m in record['menus'][agent]):
            raise ValueError('Preselected diagnostic target not in the legal menu')
        case['diagnostic_target_action']=target
        case['diagnostic_target_scope']='Researcher-only local movement/matching criterion; other legal actions are not universally incorrect.'
    prepared={'rule_full':make_rule_cases('full'),'rule_card':make_rule_cases('card'),'history':history}
    out.mkdir(parents=True,exist_ok=False)
    write_json(out/'prepared_cases.json',prepared)
    manifest={'prepared_at':datetime.now().astimezone().isoformat(),'source_run':str(SOURCE_RUN),
        'model':str(WORK/'qwen_collect_pilot/models/Qwen3.5-9B-8bit'),
        'model_commit':'16daa4818c54ce5f5436f929d52542eb65bbed9d',
        'cases_sha256':digest(out/'prepared_cases.json'),
        'source_sha256':{name:digest(path) for name,path in source_paths().items()},
        'source_records_sha256':{name:digest(SOURCE_RUN/name) for name in ('manifest.json','steps.jsonl','inference.jsonl','episodes.jsonl')},
        'sequence':['full_rules_four_questions','same_four_rule_card_only_if_any_formal_error',
                    'B_step2_original_then_empty_history','C_step9_original_then_empty_history'],
        'model_parameters':{'private_analysis_tokens':256,'private_analysis_temperature':0,
            'formal_temperature':0,'native_thinking':False,'fresh_cache_every_call':True},
        'calls_without_card':16,'calls_with_card':24,'max_backend_calls':24,
        'rule_seed_shared_across_lengths_orders_and_cards':RULE_SEED,
        'card_intervention':'Replace the full rule text with the frozen carry-rule card; do not append.',
        'history_intervention':'Only past private raw history is removed; all current broadcasts are retained.',
        'no_retries':True,'updates_source_histories':False,'unlocked_symbol_experiment':False,
        'interpretation_limits':['All R truth values are yes; passing does not exclude always-yes semantic response.',
            'R is a synthetic rule judgment, not a physical cooperation or language formation task.',
            'The two S states were selected after observing errors, not unbiased evaluation cases.',
            'Short card changes content, length and salience together.',
            'Deleting past history changes content and prompt length together.',
            'Fixed other-agent actions do not establish renewed agreement or group learning.',
            'A one-step score can stay unchanged even when transportation succeeds.']}
    for name,path in source_paths().items():
        target=out/'code_snapshot'/name; target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(path,target)
    write_json(out/'manifest.json',manifest)
    write_json(out/'status.json',{'status':'prepared','model_calls':0})
    (ROOT/'LATEST').write_text(str(out.resolve())+'\n')
    return manifest


def verify_prepared(out):
    verify_frozen()
    m=json.loads((out/'manifest.json').read_text())
    if digest(out/'prepared_cases.json')!=m['cases_sha256']:
        raise RuntimeError('Prepared inputs changed')
    for name,h in m['source_sha256'].items():
        if digest(WORK/name)!=h or digest(out/'code_snapshot'/name)!=h:
            raise RuntimeError(f'Source changed: {name}')
    for name,h in m['source_records_sha256'].items():
        if digest(Path(m['source_run'])/name)!=h:
            raise RuntimeError(f'Original records changed: {name}')
    return m


def annotate_reproduction(result, prepared, calls):
    by_call={r['call']:r for r in calls}
    cases={c['case_id']:c for c in prepared['history']}
    for row in result['history_results']:
        case=cases[row['case_id']]
        actual=[by_call[i] for i in row['call_ids']]
        old=[case['original_analysis'],case['original_formal']]
        row['reproduction']={
            'analysis_input_exact':actual[0]['messages']==old[0]['messages'],
            'analysis_render_sha_exact':actual[0]['prompt_sha256']==old[0]['prompt_sha256'],
            'analysis_text_exact':actual[0]['output']==old[0]['output'],
            'formal_input_exact':actual[1]['messages']==old[1]['messages'],
            'formal_render_sha_exact':actual[1]['prompt_sha256']==old[1]['prompt_sha256'],
            'formal_output_exact':actual[1]['output']==old[1]['output'],
            'both_seeds_match':all(a['seed']==b['seed'] for a,b in zip(actual,old))}
        row['reproduction']['both_stages_exact']=all(row['reproduction'].values())


def execute(out):
    manifest=verify_prepared(out)
    if json.loads((out/'status.json').read_text())['status']!='prepared' or (out/'inference.jsonl').exists():
        raise RuntimeError('Refuse to rerun or overwrite an executed diagnostic')
    prepared=json.loads((out/'prepared_cases.json').read_text())
    write_json(out/'status.json',{'status':'running','started':datetime.now().astimezone().isoformat()})
    try:
        backend=Backend(Path(manifest['model']),out/'inference.jsonl')
        def progress(row):
            with (out/'decisions.jsonl').open('a') as f:
                f.write(json.dumps(row,ensure_ascii=False)+'\n')
            brief={k:v for k,v in row.items() if k in ('kind','case_id','arm','choice','correct','call_ids','action_matches_recorded')}
            write_json(out/'status.json',{'status':'running','model_calls':backend.calls,'last_decision':brief})
            print(json.dumps(brief,ensure_ascii=False),flush=True)
        result=run_cases(backend,prepared,progress)
        calls=[json.loads(x) for x in (out/'inference.jsonl').read_text().splitlines()]
        expected=24 if result['rule_card_triggered'] else 16
        if len(calls)!=expected or backend.calls!=expected:
            raise RuntimeError('Unexpected model call count')
        annotate_reproduction(result,prepared,calls)
        verify_prepared(out)
        write_json(out/'results.json',result)
        write_json(out/'backend_stats.json',backend.stats())
        write_json(out/'status.json',{'status':'completed','completed':datetime.now().astimezone().isoformat(),
            'model_calls':backend.calls,'rule_card_triggered':result['rule_card_triggered'],
            'symbolic_started':False})
        print(json.dumps({'status':'completed','out':str(out),'calls':backend.calls},ensure_ascii=False),flush=True)
    except BaseException as error:
        write_json(out/'status.json',{'status':'failed','error_type':type(error).__name__,'error':str(error)})
        raise


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['prepare','execute'])
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    out=args.out.resolve()
    if args.mode=='prepare':
        manifest=prepare(out); print(json.dumps({'out':str(out),'max_calls':manifest['max_backend_calls']},ensure_ascii=False))
    else:
        execute(out)


if __name__=='__main__':
    main()

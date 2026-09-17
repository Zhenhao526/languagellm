"""Eight frozen judgments of long-wood action consequences; no model training."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil

from qwen_language_v3.backend import Backend
from qwen_language_v3.run import write_json
from .experiment import ROOT, WORK, SOURCE_RUN, digest, source_paths, verify_prepared
from .semantic_cases import build_semantic_cases

PRIOR_RUN=ROOT/'results/20260915_R_S_01'
TOTAL_CALLS=16


def evaluate_case(backend,case):
    first_call=backend.calls+1
    choice=int(backend.decide(deepcopy(case['prompt']),mode='action',choices=case['choices'],
        seed=case['seed'],temperature=0,
        label={'phase':'action_semantics','case_id':case['case_id'],
               'question_id':case['question_id'],'order':case['order'],'state_id':case['state_id']},
        final_instruction='只输出所选答案的一个编号，不要解释。'))
    if choice not in case['choices']:
        raise ValueError('Formal answer is not an available choice')
    return {'case_id':case['case_id'],'question_id':case['question_id'],
        'order':case['order'],'state_id':case['state_id'],'truth':case['truth'],
        'statement':case['statement'],'choice':choice,'answer':case['answers'][choice],
        'correct_choice':case['correct_choice'],'correct':choice==case['correct_choice'],
        'call_ids':[first_call,backend.calls]}


def validate_case_sequence(prepared):
    cases=prepared['cases']
    expected=[(f'M{i}',order) for i in range(1,5) for order in ('affirm_first','deny_first')]
    if [(c['question_id'],c['order']) for c in cases]!=expected:
        raise ValueError('Cases do not match the fixed eight-question sequence')
    if [c['truth'] for c in cases]!=[True,True,False,False,False,False,True,True]:
        raise ValueError('Unexpected environment truth values')
    if any(c['seed']!=20260916 or c['choices']!=[0,1] for c in cases):
        raise ValueError('Seeds or formal choices differ from the design')


def run_cases(backend,prepared,on_result=lambda row:None):
    validate_case_sequence(prepared)
    rows=[]
    for case in prepared['cases']:
        row=evaluate_case(backend,case)
        rows.append(row)
        on_result(row)
    return rows


def semantic_sources():
    paths=source_paths()
    for name in ('action_semantics.py','semantic_cases.py'):
        paths[f'qwen_language_diagnostics/{name}']=ROOT/name
    return paths


def prepare(out):
    # This verifies the old R/S inputs and all original v3 core sources.
    verify_prepared(PRIOR_RUN)
    prepared=build_semantic_cases()
    validate_case_sequence(prepared)
    out.mkdir(parents=True,exist_ok=False)
    write_json(out/'prepared_cases.json',prepared)
    prior_files=[PRIOR_RUN/name for name in ('manifest.json','prepared_cases.json','results.json','inference.jsonl')]
    manifest={
        'prepared_at':datetime.now().astimezone().isoformat(),
        'source_run':str(SOURCE_RUN),'prior_RS_run':str(PRIOR_RUN),
        'model':str(WORK/'qwen_collect_pilot/models/Qwen3.5-9B-8bit'),
        'model_commit':'16daa4818c54ce5f5436f929d52542eb65bbed9d',
        'cases_sha256':digest(out/'prepared_cases.json'),
        'source_sha256':{name:digest(path) for name,path in semantic_sources().items()},
        'source_records_sha256':{name:digest(SOURCE_RUN/name) for name in ('manifest.json','steps.jsonl','inference.jsonl','episodes.jsonl')},
        'prior_result_files_sha256':{str(path):digest(path) for path in prior_files},
        'sequence':[c['case_id'] for c in prepared['cases']],
        'formal_judgments':8,'model_calls':TOTAL_CALLS,'max_backend_calls':TOTAL_CALLS,
        'shared_inference_seed':20260916,
        'model_parameters':{'private_analysis_max_tokens':256,'private_analysis_temperature':0,
            'formal_temperature':0,'native_thinking':False,'fresh_cache_every_call':True},
        'runtime':{'python':platform.python_version(),'platform':platform.platform(),
            'packages':{name:importlib.metadata.version(name) for name in ('mlx','mlx-lm','transformers')}},
        'no_retries_or_adaptive_questions':True,'updates_source_histories':False,'unlocks_symbolic_experiment':False,
        'interpretation_limits':[
            'Questions test stated action consequences, not autonomous coordination or language formation.',
            'S0 was selected from an observed failure; it is not an unbiased independent evaluation sample.',
            'S1 is an environment-generated counterfactual state, not a new successful model episode.',
            'Each truth is tested in two answer orders; these are not independent group replicates.',
            'Current state and stated hypothetical joint actions are legitimate question information; future settlement and answers are not model inputs.',
            'Formal choices determine scores; private analysis is recorded text, not evidence of hidden mechanisms.']}
    for name,path in semantic_sources().items():
        target=out/'code_snapshot'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target)
    write_json(out/'manifest.json',manifest)
    write_json(out/'status.json',{'status':'prepared','model_calls':0})
    return manifest


def verify_semantic_prepared(out):
    manifest=verify_prepared(out)
    for path,h in manifest['prior_result_files_sha256'].items():
        if digest(Path(path))!=h:
            raise RuntimeError(f'Prior R/S data changed: {path}')
    return manifest


def execute(out):
    manifest=verify_semantic_prepared(out)
    if json.loads((out/'status.json').read_text())['status']!='prepared' or (out/'inference.jsonl').exists():
        raise RuntimeError('Refuse to rerun or overwrite an executed M diagnostic')
    prepared=json.loads((out/'prepared_cases.json').read_text())
    validate_case_sequence(prepared)
    write_json(out/'status.json',{'status':'running','started':datetime.now().astimezone().isoformat()})
    try:
        backend=Backend(Path(manifest['model']),out/'inference.jsonl')
        def progress(row):
            with (out/'decisions.jsonl').open('a') as f:
                f.write(json.dumps(row,ensure_ascii=False)+'\n')
            brief={k:row[k] for k in ('case_id','question_id','order','choice','answer','correct','call_ids')}
            write_json(out/'status.json',{'status':'running','model_calls':backend.calls,'last_decision':brief})
            print(json.dumps(brief,ensure_ascii=False),flush=True)
        rows=run_cases(backend,prepared,progress)
        calls=[json.loads(x) for x in (out/'inference.jsonl').read_text().splitlines()]
        if backend.calls!=TOTAL_CALLS or len(calls)!=TOTAL_CALLS:
            raise RuntimeError('Unexpected total model call count')
        verify_semantic_prepared(out)
        results={'cases':rows,'correct':sum(r['correct'] for r in rows),'formal_judgments':len(rows),
            'by_question':[{'question_id':f'M{i}','truth':rows[(i-1)*2]['truth'],
                'both_orders_correct':all(r['correct'] for r in rows[(i-1)*2:i*2]),
                'answer_meaning_consistent':len({r['answer'] for r in rows[(i-1)*2:i*2]})==1}
                for i in range(1,5)]}
        write_json(out/'results.json',results)
        write_json(out/'backend_stats.json',backend.stats())
        write_json(out/'status.json',{'status':'completed','completed':datetime.now().astimezone().isoformat(),
            'model_calls':backend.calls,'formal_correct':results['correct'],'symbolic_started':False})
        print(json.dumps({'status':'completed','out':str(out),'calls':backend.calls,
            'correct':results['correct']},ensure_ascii=False),flush=True)
    except BaseException as error:
        write_json(out/'status.json',{'status':'failed','error_type':type(error).__name__,'error':str(error)})
        raise


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['prepare','execute'])
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args(); out=args.out.resolve()
    if args.mode=='prepare':
        manifest=prepare(out)
        (ROOT/'LATEST_M').write_text(str(out)+'\n')
        print(json.dumps({'out':str(out),'model_calls':manifest['model_calls']},ensure_ascii=False))
    else:
        execute(out)


if __name__=='__main__':main()

"""Fixed original-versus-enumerated-road judgment pairs; no model training."""
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
from .experiment import ROOT, WORK, SOURCE_RUN, digest, verify_prepared
from .action_semantics import semantic_sources, verify_semantic_prepared
from .map_cases import build_map_cases

PRIOR_M=ROOT/'results/20260915_M_01'
TOTAL_CALLS=16
QUESTIONS=('G1','G2','J1','J2')
REPRESENTATIONS=('original','explicit_edges')


def evaluate_case(backend,case):
    first=backend.calls+1
    choice=int(backend.decide(deepcopy(case['prompt']),mode='action',choices=case['choices'],
        seed=case['seed'],temperature=0,label={'phase':'map_presentation','case_id':case['case_id'],
        'question_id':case['question_id'],'representation':case['representation'],'state_id':case['state_id']},
        final_instruction='只输出所选答案的一个编号，不要解释。'))
    if choice not in case['choices']:
        raise ValueError('Invalid formal choice')
    return {'case_id':case['case_id'],'question_id':case['question_id'],
        'representation':case['representation'],'state_id':case['state_id'],
        'truth':case['truth'],'statement':case['statement'],'choice':choice,
        'answer':case['answers'][choice],'correct_choice':case['correct_choice'],
        'correct':choice==case['correct_choice'],'call_ids':[first,backend.calls]}


def validate_case_sequence(prepared):
    cases=prepared['cases']
    expected=[(q,r) for q in QUESTIONS for r in REPRESENTATIONS]
    if [(c['question_id'],c['representation']) for c in cases]!=expected:
        raise ValueError('Cases differ from the fixed eight-decision sequence')
    if [c['truth'] for c in cases]!=[True,True,False,False,True,True,False,False]:
        raise ValueError('Unexpected truth values')
    if any(c['seed']!=20260916 or c['choices']!=[0,1] or c['order']!='affirm_first' for c in cases):
        raise ValueError('Changed seed or answer order')


def run_cases(backend,prepared,on_result=lambda row:None):
    validate_case_sequence(prepared)
    rows=[]
    for case in prepared['cases']:
        row=evaluate_case(backend,case)
        rows.append(row)
        on_result(row)
    return rows


def map_sources():
    paths=semantic_sources()
    for name in ('map_cases.py','map_presentation.py'):
        paths[f'qwen_language_diagnostics/{name}']=ROOT/name
    return paths


def prepare(out):
    verify_semantic_prepared(PRIOR_M)
    prepared=build_map_cases()
    validate_case_sequence(prepared)
    out.mkdir(parents=True,exist_ok=False)
    write_json(out/'prepared_cases.json',prepared)
    old_manifest=json.loads((PRIOR_M/'manifest.json').read_text())
    previous=deepcopy(old_manifest['prior_result_files_sha256'])
    previous.update({str(PRIOR_M/name):digest(PRIOR_M/name)
                     for name in ('manifest.json','prepared_cases.json','results.json','inference.jsonl')})
    manifest={'prepared_at':datetime.now().astimezone().isoformat(),'source_run':str(SOURCE_RUN),
        'prior_M_run':str(PRIOR_M),'model':old_manifest['model'],'model_commit':old_manifest['model_commit'],
        'cases_sha256':digest(out/'prepared_cases.json'),
        'source_sha256':{name:digest(path) for name,path in map_sources().items()},
        'source_records_sha256':deepcopy(old_manifest['source_records_sha256']),
        'prior_result_files_sha256':previous,
        'sequence':[c['case_id'] for c in prepared['cases']],
        'formal_judgments':8,'model_calls':TOTAL_CALLS,'max_backend_calls':TOTAL_CALLS,
        'paired_seed':20260916,'answer_order':'affirm_first_in_both_representations',
        'presentation_change':deepcopy(prepared['presentation_change']),
        'model_parameters':deepcopy(old_manifest['model_parameters']),
        'runtime':{'python':platform.python_version(),'platform':platform.platform(),
            'packages':{name:importlib.metadata.version(name) for name in ('mlx','mlx-lm','transformers')}},
        'no_retries_or_adaptive_questions':True,'updates_source_histories':False,
        'unlocks_symbolic_experiment':False,
        'interpretation_limits':[
            'Each pair only replaces the map sentence in the system message; length and salience also change.',
            'Four statements in one selected source state do not estimate general model performance.',
            'J1 original is an exact input/seed anchor to M1 affirm-first; reproduction must be reported.',
            'Specified hypothetical actions test judgment, not independent selection or renewed coordination.',
            'There is no new symbolic communication, training, memory reset in a group, or automatic capability unlock.',
            'Private analysis is recorded output; it does not prove a hidden mechanism.']}
    for name,path in map_sources().items():
        target=out/'code_snapshot'/name; target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target)
    write_json(out/'manifest.json',manifest)
    write_json(out/'status.json',{'status':'prepared','model_calls':0})
    return manifest


def verify_map_prepared(out):
    manifest=verify_prepared(out)
    for name,h in manifest['prior_result_files_sha256'].items():
        if digest(Path(name))!=h:
            raise RuntimeError(f'Prior diagnostic records changed: {name}')
    return manifest


def anchor_reproduction(rows,prepared,calls):
    anchor=prepared['anchor']
    row=next(r for r in rows if r['case_id']==anchor['case_id'])
    by_call={r['call']:r for r in calls}
    new=[by_call[c] for c in row['call_ids']]
    old=[anchor['original_analysis'],anchor['original_formal']]
    checks={'analysis_input_exact':new[0]['messages']==old[0]['messages'],
        'analysis_render_sha_exact':new[0]['prompt_sha256']==old[0]['prompt_sha256'],
        'analysis_text_exact':new[0]['output']==old[0]['output'],
        'formal_input_exact':new[1]['messages']==old[1]['messages'],
        'formal_render_sha_exact':new[1]['prompt_sha256']==old[1]['prompt_sha256'],
        'formal_output_exact':new[1]['output']==old[1]['output'],
        'both_seeds_match':all(a['seed']==b['seed'] for a,b in zip(new,old))}
    return {'case_id':row['case_id'],'source_case_id':anchor['source_case_id'],
        'source_call_ids':[r['call'] for r in old],'new_call_ids':row['call_ids'],
        'checks':checks,'both_stages_exact':all(checks.values())}


def execute(out):
    manifest=verify_map_prepared(out)
    if json.loads((out/'status.json').read_text())['status']!='prepared' or (out/'inference.jsonl').exists():
        raise RuntimeError('Refuse to rerun or overwrite an executed map diagnostic')
    prepared=json.loads((out/'prepared_cases.json').read_text()); validate_case_sequence(prepared)
    write_json(out/'status.json',{'status':'running','started':datetime.now().astimezone().isoformat()})
    try:
        backend=Backend(Path(manifest['model']),out/'inference.jsonl')
        def progress(row):
            with (out/'decisions.jsonl').open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
            brief={k:row[k] for k in ('case_id','representation','choice','answer','correct','call_ids')}
            write_json(out/'status.json',{'status':'running','model_calls':backend.calls,'last_decision':brief})
            print(json.dumps(brief,ensure_ascii=False),flush=True)
        rows=run_cases(backend,prepared,progress)
        calls=[json.loads(x) for x in (out/'inference.jsonl').read_text().splitlines()]
        if backend.calls!=TOTAL_CALLS or len(calls)!=TOTAL_CALLS:raise RuntimeError('Unexpected call count')
        verify_map_prepared(out)
        results={'cases':rows,'formal_judgments':8,'correct':sum(r['correct'] for r in rows),
            'by_representation':[{'representation':rep,'correct':sum(r['correct'] for r in rows if r['representation']==rep),
                'judgments':4} for rep in REPRESENTATIONS],
            'pairs':[{'question_id':q,'truth':rows[2*i]['truth'],
                'original_correct':rows[2*i]['correct'],'explicit_edges_correct':rows[2*i+1]['correct'],
                'answer_meaning_changed':rows[2*i]['answer']!=rows[2*i+1]['answer']} for i,q in enumerate(QUESTIONS)],
            'anchor_reproduction':anchor_reproduction(rows,prepared,calls)}
        write_json(out/'results.json',results); write_json(out/'backend_stats.json',backend.stats())
        write_json(out/'status.json',{'status':'completed','completed':datetime.now().astimezone().isoformat(),
            'model_calls':backend.calls,'formal_correct':results['correct'],'symbolic_started':False})
        print(json.dumps({'status':'completed','out':str(out),'calls':backend.calls,
            'by_representation':results['by_representation']},ensure_ascii=False),flush=True)
    except BaseException as error:
        write_json(out/'status.json',{'status':'failed','error_type':type(error).__name__,'error':str(error)})
        raise


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('mode',choices=['prepare','execute'])
    parser.add_argument('--out',type=Path,required=True); args=parser.parse_args(); out=args.out.resolve()
    if args.mode=='prepare':
        manifest=prepare(out); (ROOT/'LATEST_MAP').write_text(str(out)+'\n')
        print(json.dumps({'out':str(out),'model_calls':manifest['model_calls']},ensure_ascii=False))
    else:execute(out)


if __name__=='__main__':main()

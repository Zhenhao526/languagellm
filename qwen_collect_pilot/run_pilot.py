"""Run local Qwen collection: isolated contexts, calibration, and interventions."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import subprocess
import time

import mlx.core as mx

from agents import build_messages, empty_histories, remember
from env import World, enumerate_worlds, sample
from model_backend import Backend
from verify_model import verify

BASE = Path(__file__).resolve().parent

def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')

def hardware_summary():
    info = json.loads(subprocess.check_output(
        ['system_profiler','SPHardwareDataType','SPDisplaysDataType','-json'],text=True))
    hardware = info['SPHardwareDataType'][0]
    return {'hardware':{key:hardware.get(key) for key in
        ('machine_model','chip_type','physical_memory','number_processors')},
        'graphics':[{key:gpu.get(key) for key in
        ('sppci_model','sppci_cores','spdisplays_metal')} for gpu in info['SPDisplaysDataType']]}

def outcome_summary(rows):
    return {'n':len(rows), **{key:sum(row['scores'][key] for row in rows)
        for key in ('A','B','overall')},
        'success_rate':sum(row['scores']['overall'] for row in rows)/len(rows)}

class Experiment:
    def __init__(self, backend, out):
        self.backend = backend
        self.out = out
        self.trace = (out/'trials.jsonl').open('w')

    def act(self, role, world, history, message=None, *, known=False, cache=None, label=None):
        messages = build_messages(role, world.private_observation(role), history,
                                  received=message, known_protocol=known)
        # Deterministic memoization only for identical private prompts in a
        # frozen evaluation. No cache survives an evaluation or crosses roles.
        key = json.dumps(messages, ensure_ascii=False, sort_keys=True)
        if cache is not None and key in cache:
            answer, call = cache[key]
            return answer, {'call':call, 'memoized_identical_prompt':True}
        answer = self.backend.decide(messages, mode='message' if role=='C' else 'action',
                                    temperature=0, seed=0, label=label)
        call = self.backend.calls
        if cache is not None:
            cache[key] = (answer, call)
        return answer, {'call':call, 'memoized_identical_prompt':False}

    def record(self, phase, group, index, world, message, actions, calls, **extra):
        row = {'phase':phase, 'group':group, 'index':index,
            'world':asdict(world), 'message':message, 'actions':actions,
            'scores':world.score(actions), 'calls':calls, **extra}
        self.trace.write(json.dumps(row, ensure_ascii=False)+'\n')
        self.trace.flush()
        return row

    def calibrate(self):
        rows, cache = [], {}
        for index, world in enumerate(enumerate_worlds()):
            message = '@#'[world.d_a] + '@#'[world.d_b]
            actions, calls = {}, {}
            for role in ('A','B'):
                answer, calls[role] = self.act(role,world,[],message,known=True,
                    cache=cache,label={'phase':'known_protocol','role':role,'case':index})
                actions[role] = int(answer)
            rows.append(self.record('known_protocol','calibration',index,world,message,actions,calls))
        result = outcome_summary(rows)
        print(json.dumps({'event':'calibration_complete',**result}),flush=True)
        return result

    def frozen_eval(self, histories, group):
        before = deepcopy(histories)
        cache, codes, encoder_calls = {}, {}, {}
        # C does not see positions, so four private observations exhaust its
        # input space. Label/case numbers exist only in experimenter logs.
        for d_a in (0,1):
            for d_b in (0,1):
                need = f'{d_a}{d_b}'
                codes[need], encoder_calls[need] = self.act('C',World(d_a,d_b,0,0),
                    histories['C'],cache=cache,label={'phase':'frozen_encoder','group':group,'need':need})
        results = {}
        for condition in ('original','empty','permuted'):
            rows = []
            for index, world in enumerate(enumerate_worlds()):
                need = f'{world.d_a}{world.d_b}'
                source = f'{1-world.d_a}{1-world.d_b}' if condition=='permuted' else need
                message = '' if condition=='empty' else codes[source]
                actions, calls = {}, {}
                for role in ('A','B'):
                    answer, calls[role] = self.act(role,world,histories[role],message,
                        cache=cache,label={'phase':'frozen_'+condition,'group':group,'role':role,'case':index})
                    actions[role] = int(answer)
                rows.append(self.record('frozen_'+condition,group,index,world,message,
                    actions,calls,encoder_call=None if condition=='empty' else encoder_calls[source],
                    source_need=None if condition=='empty' else source,
                    feedback_returned=False))
            results[condition] = outcome_summary(rows)
            print(json.dumps({'event':'evaluation_complete','group':group,
                              'condition':condition,**results[condition]}),flush=True)
        assert before == histories, 'Frozen evaluation altered an agent history'
        # Exhaustive iid demands yield exactly 4/16 joint success for any
        # deterministic pair acting only on their private positions.
        if results['empty']['overall'] != 4:
            raise RuntimeError('Empty-message baseline violated; inspect information leakage')
        return {'codes_by_demand':codes,'distinct_messages':len(set(codes.values())),
            'permutation_changed_message_fraction':sum(codes[f'{a}{b}']!=codes[f'{1-a}{1-b}']
                for a in (0,1) for b in (0,1))/4, 'conditions':results}

    def train(self, seed, rounds):
        histories, rows = empty_histories(), []
        worlds = random.Random(seed)
        for index in range(rounds):
            world = sample(worlds)
            messages = build_messages('C',world.private_observation('C'),histories['C'])
            message = self.backend.decide(messages,mode='message',temperature=0.7,
                seed=seed*10000+index,label={'phase':'interaction','group':seed,'role':'C','round':index})
            calls = {'C':{'call':self.backend.calls,'memoized_identical_prompt':False}}
            actions = {}
            for role in ('A','B'):
                answer, calls[role] = self.act(role,world,histories[role],message,
                    label={'phase':'interaction','group':seed,'role':role,'round':index})
                actions[role] = int(answer)
            row = self.record('interaction',seed,index,world,message,actions,calls)
            # Feedback enters both private histories only after both choices.
            remember(histories,world,message,actions,row['scores'])
            rows.append(row)
            print(json.dumps({'event':'round','group':seed,'round':index+1,
                'message':message,'success':row['scores']['overall'],
                'cumulative_success':sum(r['scores']['overall'] for r in rows)}),flush=True)
        group_dir = self.out/f'group_{seed}'
        group_dir.mkdir()
        for role, history in histories.items():
            write_json(group_dir/f'history_{role}.json',history)
        return histories, {'all':outcome_summary(rows),
            'first_half':outcome_summary(rows[:rounds//2]),
            'second_half':outcome_summary(rows[rounds//2:]),
            'messages_observed':sorted({r['message'] for r in rows})}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rounds',type=int,default=24)
    parser.add_argument('--seeds',type=int,nargs='+',default=[17,29])
    parser.add_argument('--smoke-only',action='store_true')
    parser.add_argument('--private-reasoning',action='store_true',help='Use up to 256 private analysis tokens before each constrained output')
    args = parser.parse_args()
    if args.rounds < 2 or len(set(args.seeds)) != len(args.seeds):
        parser.error('Need at least two rounds and unique seeds')
    if not mx.metal.is_available():
        raise RuntimeError('This experiment requires the local Apple Metal backend')
    started = time.perf_counter()
    verification = verify()
    out = BASE/'results'/time.strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True,exist_ok=False)
    print(f'RESULT_DIR={out}',flush=True)
    write_json(out/'config.json',{
        'model':verification['repository'],'revision':verification['revision'],
        'model_verification':verification, 'rounds_per_group':args.rounds,'seeds':args.seeds,
        'alphabet':'@#%&','min_message_chars':0,'max_message_chars':2,
        'sender_temperature_during_interaction':0.7,'collector_temperature':0,
        'frozen_eval_temperature':0,'thinking':False,
        'private_reasoning':args.private_reasoning,'private_reasoning_max_tokens':256,
        'private_reasoning_temperature':0,'private_reasoning_retained_in_episodic_memory':False,
        'updates':'private context only; fixed model weights',
        'private_contexts':['A','B','C'],'fresh_kv_cache_per_inference':True,
        'known_protocol_context_reused_in_experiment':False,
        'evaluation':'all 16 worlds; no feedback; exact private-prompt memoization within frozen phase',
        'hardware':hardware_summary(),
        'packages':{p:importlib.metadata.version(p) for p in ('mlx','mlx-lm','transformers','huggingface-hub')},
        'code_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in BASE.glob('*.py')}
    })
    backend = Backend(BASE/'models/Qwen3.5-9B-8bit',out/'inference.jsonl',private_reasoning=args.private_reasoning)
    smoke = backend.infer([{'role':'user','content':'请只回答两个汉字：你好'}],mode='raw',label={'phase':'smoke'})
    write_json(out/'smoke.json',{'prompt':'请只回答两个汉字：你好','response':smoke})
    print(json.dumps({'event':'smoke','response':smoke},ensure_ascii=False),flush=True)
    if args.smoke_only:
        write_json(out/'summary.json',{'smoke':smoke,'backend':backend.stats()})
        return
    experiment = Experiment(backend,out)
    summary = {'smoke':smoke,'calibration':experiment.calibrate()}
    write_json(out/'summary.json',summary)
    if summary['calibration']['overall'] < 15:
        raise RuntimeError('Known-protocol capability below 15/16; diagnose before free-code interaction')
    summary['zero_interaction'] = experiment.frozen_eval(empty_histories(),'zero_interaction')
    write_json(out/'summary.json',summary)
    summary['groups'] = {}
    for seed in args.seeds:
        histories, training = experiment.train(seed,args.rounds)
        evaluation = experiment.frozen_eval(histories,seed)
        summary['groups'][str(seed)] = {'interaction':training,'evaluation':evaluation}
        write_json(out/'summary.json',summary)
    summary['backend'] = backend.stats()
    summary['elapsed_seconds'] = time.perf_counter()-started
    summary['completed'] = True
    write_json(out/'summary.json',summary)
    print(json.dumps({'event':'pilot_complete','result_dir':str(out),
                      'elapsed_seconds':summary['elapsed_seconds'],**backend.stats()}),flush=True)

if __name__ == '__main__':
    main()

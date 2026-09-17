"""Run the fixed first pilot batch without adapting settings to outcomes."""
import argparse, hashlib, json, shutil, time
from datetime import datetime, timezone
from pathlib import Path
import torch
from .train import pretrain,run,write_json,device_info


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=Path('pilot_round1/results/batch_001'))
    args=parser.parse_args()
    out=args.out
    if (out/'batch_config.json').exists():
        raise RuntimeError('Refusing to overwrite an existing batch; use a new directory')
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    config={'batch':'001','created_utc':datetime.now(timezone.utc).isoformat(),
      'seeds':[101,202,303],'steps_per_run':24000,'eval_every':2000,'eval_n':512,
      'final_eval_n':2048,'history_length':4,'rollout':32,'learning_rate':.002,'entropy':.03,
      'visual_updates_per_agent':250,'visual_batch_size':128,'visual_required_accuracy':.9,
      'visual_device':'mps' if torch.backends.mps.is_available() else 'cpu','communication_device':'cpu',
      'cpu_threads':4,'total_wall_budget_seconds':3600,'max_run_wall_seconds':600,
      'conditions':['H0','H1'],'tasks':['emergent','fixed','no_comm'],
      'no_comm_seeds':[101],'performance_confirmation':'>=0.90 at 3 consecutive scheduled frozen evaluations; report third checkpoint',
      'interpretation':'exploratory pilot; independent dyad is replication unit; no significance claims from 3 seeds',
      'backend_choice':'CPU main-loop 1.735s vs MPS 1.894s for 1024 sequential interactions; identical benchmark final accuracy. MPS used for batched visual preparation.',
      'runtime':device_info()}
    source_dir=Path(__file__).resolve().parent
    snapshot=out/'source_snapshot'; snapshot.mkdir()
    hashes={}
    for name in ['__init__.py','env.py','model.py','train.py','run_batch.py','requirements.lock.txt']:
        path=source_dir/name
        shutil.copy2(path,snapshot/name); hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    config['source_sha256']=hashes
    write_json(out/'batch_config.json',config)
    start=time.monotonic(); prepared={}; results=[]
    for seed in config['seeds']:
        checkpoint,info=pretrain(seed,out/'pretrained',config['visual_device'],250,128)
        prepared[seed]=(checkpoint,info)
        if min(a['accuracy'] for a in info['agents'])<.9:
            write_json(out/'batch_status.json',{'status':'calibration_failed','seed':seed,'info':info})
            raise RuntimeError('Calibration failed: no main trials run')
    schedule=[(task,seed,condition) for task in ['emergent','fixed']
              for seed in config['seeds'] for condition in config['conditions']]
    schedule += [('no_comm',101,c) for c in config['conditions']]
    for index,(task,seed,condition) in enumerate(schedule):
        remaining=3600-(time.monotonic()-start)
        if remaining<60:
            write_json(out/'batch_status.json',{'status':'budget_exhausted','completed':results,'elapsed_seconds':time.monotonic()-start})
            return
        print(json.dumps({'stage':'run_start','run_index':index+1,'run_total':len(schedule),
                          'task':task,'seed':seed,'condition':condition}),flush=True)
        checkpoint,info=prepared[seed]
        run_out=out/f'{task}_seed{seed}_{condition}'
        result=run(checkpoint,info,seed,condition,task,'cpu',run_out,24000,
                   eval_every=2000,eval_n=512,wall_limit=min(600,remaining))
        result_short={'task':task,'seed':seed,'condition':condition,'steps':result['steps'],
                      'accuracy':result['final_eval']['accuracy'],'seconds':result['train_seconds'],
                      'stopped_by_time':result['stopped_by_time'],'path':str(run_out)}
        results.append(result_short)
        write_json(out/'batch_status.json',{'status':'running','completed':results,'elapsed_seconds':time.monotonic()-start})
        if result['stopped_by_time']:
            write_json(out/'batch_status.json',{'status':'run_time_limit','completed':results,'elapsed_seconds':time.monotonic()-start})
            return
    write_json(out/'batch_status.json',{'status':'complete','completed':results,'elapsed_seconds':time.monotonic()-start,
                                      'finished_utc':datetime.now(timezone.utc).isoformat()})
    print(json.dumps({'stage':'batch_complete','runs':len(results),'seconds':time.monotonic()-start}),flush=True)


if __name__=='__main__': main()

"""Run a fixed list of independent single-agent calibration episodes locally."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import argparse
import json
import os
import signal
import subprocess
import time

BASE=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest',type=Path)
    parser.add_argument('--workers',type=int,choices=(1,2),default=2)
    args=parser.parse_args()
    manifest=json.loads(args.manifest.read_text())
    name=manifest['name']
    if not name.replace('_','').isalnum():raise ValueError('Invalid batch name')
    cases=manifest['cases']
    records=[]
    result_path=BASE/'logs'/f'{name}_results.json'
    if result_path.exists():raise FileExistsError('Batch already has results; use a new name')

    def run(case):
        case_id=case['id']
        if not case_id.replace('_','').isalnum():raise ValueError('Invalid case id')
        output=BASE/'runs'/f'{name}_{case_id}'
        command=['bash',str(BASE/'with_environment.sh'),str(BASE/'run_calibration.py'),
                 '--controller',case['controller'],'--scenario',case['scenario'],
                 '--seed',str(case['seed']),'--steps',str(case['steps']),
                 '--food',str(case.get('food',8)),'--output-dir',str(output)]
        logfile=BASE/'logs'/f'{name}_{case_id}.log'
        started=time.monotonic()
        with logfile.open('w') as stream:
            process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,
                env={**os.environ,'MINERL_TMP_INSTANCES':'1'},start_new_session=True)
            timed_out=False
            try:code=process.wait(timeout=600)
            except subprocess.TimeoutExpired:
                timed_out=True
                os.killpg(process.pid,signal.SIGTERM)
                try:code=process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL)
                    code=process.wait(timeout=15)
        record={'case':case,'returncode':code,'timed_out':timed_out,
                'elapsed_wall_seconds':time.monotonic()-started,'output':str(output),'log':str(logfile)}
        for filename in ('status','reset_check','summary'):
            p=output/f'{filename}.json'
            if p.exists():record[filename]=json.loads(p.read_text())
        return record

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(run,c) for c in cases]
        for future in as_completed(futures):
            record=future.result()
            records.append(record)
            result_path.write_text(json.dumps({'manifest':manifest,'workers':args.workers,'records':records},indent=2,ensure_ascii=False)+'\n')
            summary=record.get('summary',{})
            print(json.dumps({'case':record['case']['id'],'returncode':record['returncode'],
                'steps':summary.get('completed_steps'),'movement':summary.get('movement_observed'),
                'wood':summary.get('native_wood_chain_observed'),'food':summary.get('native_food_chain_observed'),
                'finished':len(records),'total':len(cases)},ensure_ascii=False),flush=True)


if __name__=='__main__':main()

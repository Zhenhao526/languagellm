"""Read the latest batch status without importing the model or changing data."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent


def last_complete(path):
    if not path.exists():
        return {}
    with path.open('rb') as stream:
        stream.seek(0, 2)
        position, data = stream.tell(), b''
        while position:
            size = min(position, 262144)
            position -= size
            stream.seek(position)
            data = stream.read(size) + data
            for line in reversed(data.splitlines()):
                try:
                    row = json.loads(line)
                    if isinstance(row, dict):
                        return row
                except (ValueError, UnicodeError):
                    pass
    return {}


batch = Path((ROOT/'LATEST_BATCH').read_text().strip())
status = json.loads((batch/'pipeline_status.json').read_text())
directories = [p for p in batch.iterdir() if p.is_dir() and (p/'manifest.json').exists()]
run = max(directories, key=lambda p:(p/'inference.jsonl').stat().st_mtime if (p/'inference.jsonl').exists() else p.stat().st_mtime)
call = last_complete(run/'inference.jsonl')
step = last_complete(run/'steps.jsonl')
episodes = []
for directory in directories:
    if (directory/'episodes.jsonl').exists():
        episodes.extend({key:row.get(key) for key in ('condition','episode','seed','score','steps')}
            for line in (directory/'episodes.jsonl').read_text().splitlines()
            if isinstance((row:=json.loads(line)), dict))
print(json.dumps({'batch':str(batch),'stage':status['stage'],'outcome':status.get('outcome'),
    'run':run.name,'latest_call':call.get('call'),'current_decision':call.get('label'),
    'settled':{key:step.get(key) for key in ('condition','episode','step','score')},
    'actions':{a:value['description'] for a,value in step.get('selections',{}).items()},
    'feedback':step.get('feedback'), 'completed_episodes':episodes},ensure_ascii=False))

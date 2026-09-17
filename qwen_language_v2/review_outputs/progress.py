"""Read current pilot progress; no inference, writes, or experiment changes."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'results/pilot_immediate_delayed_20260914'


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


pipeline = json.loads((ROOT / 'results/pipeline_status_20260914.json').read_text())
step = last_complete(RUN / 'steps.jsonl')
call = last_complete(RUN / 'inference.jsonl')
episodes = []
if (RUN / 'episodes.jsonl').exists():
    for line in (RUN / 'episodes.jsonl').read_text().splitlines():
        try:
            row = json.loads(line)
            episodes.append({key: row[key] for key in ('condition', 'episode', 'score', 'steps')})
        except ValueError:
            pass
result = {'pipeline_stage': pipeline['stage'], 'settled': {key: step.get(key)
    for key in ('condition', 'episode', 'step', 'score')}, 'latest_call': call.get('call'),
    'current_decision': call.get('label'), 'actions': {a: value['description']
    for a, value in step.get('selections', {}).items()},
    'feedback': step.get('feedback', {}), 'completed_episodes': episodes}
if pipeline['stage'] == 'failed':
    result['pipeline_error'] = pipeline
print(json.dumps(result, ensure_ascii=False))

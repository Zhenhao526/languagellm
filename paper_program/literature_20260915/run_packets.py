from pathlib import Path
import subprocess,json
B=Path(__file__).resolve().parent
radar='/Users/xia/.codex/plugins/cache/zhenhao-arxiv-tools/arxiv-watcher/0.2.0/skills/arxiv-paper-radar/scripts/radar.py'
for p in json.loads((B/'targeted_candidates.json').read_text())['papers']:
    aid=p['arxiv_id']
    with (B/f'packet_{aid}.log').open('a') as log:
        r=subprocess.run([str(B.parents[1]/'.venv/bin/python'),radar,'packet',aid,'--config',str(B/'radar.toml'),'--max-chars','300000','--output',str(B/'packets'/f'{aid}.md')],stdout=log,stderr=subprocess.STDOUT)
    print(aid,r.returncode,flush=True)

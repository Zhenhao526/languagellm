"""Inspect action-format failures in isolated diagnostic contexts only."""
from pathlib import Path
import json
from agents import build_messages
from env import World
from model_backend import Backend

BASE = Path(__file__).resolve().parent
OUT = BASE/'diagnostics'
OUT.mkdir(exist_ok=True)
backend = Backend(BASE/'models/Qwen3.5-9B-8bit',OUT/'calibration_inference.jsonl')
cases = [('A',World(0,0,1,0),'@@'),('B',World(0,1,0,1),'@#'),('B',World(1,0,0,0),'#@')]
clarification = '\n输出的是槽位编号，而不是资源名称末尾的数字。先确定自己要采哪一种资源，再在本轮slots列表里查找该资源所在项的slot。只输出该slot的0或1。'
for role, world, message in cases:
    for variant in ('original_raw','clarified_raw','clarified_constrained'):
        prompt = build_messages(role,world.private_observation(role),[],received=message,known_protocol=True)
        if variant.startswith('clarified'):
            prompt[0]['content'] += clarification
        output = backend.infer(prompt,mode='action' if variant.endswith('constrained') else 'raw',
            label={'variant':variant,'role':role,'world':world.__dict__})
        print(json.dumps({'variant':variant,'role':role,'world':world.__dict__,
            'correct_slot':world.targetslots()[role],'output':output},ensure_ascii=False),flush=True)

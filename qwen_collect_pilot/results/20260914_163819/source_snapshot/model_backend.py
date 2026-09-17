"""One fixed MLX model, fresh inference cache for every private-context call."""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
import re
import time

import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

ALPHABET = '@#%&'


class ChannelMask:
    """Restrict each generated token to one visible symbol or an end token.

    Multi-character tokenizer tokens are deliberately excluded: each channel
    symbol is one decoding step. No meaning or role-specific slot is imposed.
    """
    def __init__(self, token_ids, eos_ids, max_chars, allow_empty):
        self.token_ids = list(token_ids)
        self.eos_ids = list(eos_ids)
        self.max_chars = max_chars
        self.allow_empty = allow_empty
        self.calls = 0

    def __call__(self, tokens, logits):
        n = self.calls
        self.calls += 1
        if n >= self.max_chars:
            allowed = self.eos_ids
        else:
            allowed = self.token_ids + (self.eos_ids if n or self.allow_empty else [])
        mask = mx.full(logits.shape, -float('inf'), dtype=logits.dtype)
        mask[..., mx.array(allowed)] = 0
        return logits + mask


class Backend:
    def __init__(self, model_dir: Path, log_path: Path, private_reasoning=False):
        self.log = log_path.open('w')
        self.calls = 0
        self.elapsed = []
        self.prompt_tokens = []
        self.private_reasoning = private_reasoning
        config = json.loads((model_dir/'config.json').read_text())
        if config.get('model_file'):
            raise ValueError('Repository Python model code is not enabled')
        if config.get('quantization', {}).get('bits') != 8:
            raise ValueError('Expected the requested 8-bit checkpoint')
        started = time.perf_counter()
        self.model, self.tokenizer = load(str(model_dir),
            tokenizer_config={'trust_remote_code': False})
        self.load_seconds = time.perf_counter() - started
        self.char_ids = {}
        for char in ALPHABET + '01':
            ids = self.tokenizer.encode(char, add_special_tokens=False)
            if len(ids) != 1 or self.tokenizer.decode(ids) != char:
                raise ValueError(f'{char!r} is not a single round-tripping token')
            self.char_ids[char] = ids[0]
        self.eos_ids = list(self.tokenizer.eos_token_ids)
        print(json.dumps({'event':'model_loaded','load_seconds':self.load_seconds,
                          'char_ids':self.char_ids,'eos_ids':self.eos_ids,
                          'peak_memory_gb':mx.get_peak_memory()/1e9}), flush=True)

    def infer(self, messages, *, mode, temperature=0.0, seed=0, label=None):
        self.calls += 1
        mx.random.seed(seed)
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        processors = None
        max_tokens = 256 if mode == 'private_reasoning' else 64
        if mode in ('message', 'action'):
            chars = ALPHABET if mode == 'message' else '01'
            limit = 2 if mode == 'message' else 1
            processors = [ChannelMask([self.char_ids[c] for c in chars],
                self.eos_ids, limit, allow_empty=(mode == 'message'))]
            max_tokens = limit + 1
        started = time.perf_counter()
        pieces = []
        last = None
        for response in stream_generate(self.model, self.tokenizer, prompt,
                max_tokens=max_tokens,
                sampler=make_sampler(temp=temperature),
                logits_processors=processors,
                prompt_cache=None, prefill_step_size=512):
            pieces.append(response.text)
            last = response
        elapsed = time.perf_counter() - started
        text = ''.join(pieces)
        valid = (len(text) <= 2 and all(c in ALPHABET for c in text)) if mode == 'message' else text in ('0','1') if mode == 'action' else True
        entry = {'call':self.calls,'label':label,'mode':mode,'temperature':temperature,
            'seed':seed,'messages':messages,'rendered_prompt':prompt,
            'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),
            'output':text,'valid':valid,'seconds':elapsed,
            'prompt_tokens':last.prompt_tokens,'generation_tokens':last.generation_tokens,
            'prompt_tps':last.prompt_tps,'generation_tps':last.generation_tps,
            'peak_memory_gb':last.peak_memory,'finish_reason':last.finish_reason,
            'fresh_cache':True,'thinking':False}
        self.log.write(json.dumps(entry,ensure_ascii=False)+'\n')
        self.log.flush()
        self.elapsed.append(elapsed)
        self.prompt_tokens.append(last.prompt_tokens)
        if not valid:
            raise RuntimeError(f'Invalid constrained output, preserved in log: {text!r}')
        return text

    def decide(self, messages, *, mode, temperature=0.0, seed=0, label=None):
        if not self.private_reasoning:
            return self.infer(messages,mode=mode,temperature=temperature,seed=seed,label=label)
        private_messages = deepcopy(messages)
        for message in private_messages:
            message['content'] = re.sub(r'只输出[^。\n]*[。\n]?', '', message['content']).replace('不要解释。','')
        instruction = ('请先在自己的私有上下文中分析，最多120个汉字。这不是对其他主体发送的消息，其他人看不到这段分析。'
            + ('根据自己的需求与历史反馈，分析本轮应选什么符号消息；不要假定别人已经知道一个未给定的词典。'
               if mode=='message' else
               '先根据收到的消息及自己已知的协议或历史判断所需资源，再逐项核对本轮资源与槽位的对应，说明应选哪个槽位。没有确定依据时承认不确定并作出选择。'))
        private_messages.append({'role':'user','content':instruction})
        reasoning = self.infer(private_messages,mode='private_reasoning',temperature=0,seed=seed,
            label={**(label or {}),'stage':'private_reasoning'})
        final_messages = deepcopy(messages) + [
            {'role':'assistant','content':reasoning},
            {'role':'user','content':('依据你刚才的私有分析，现在只输出正式广播消息：0到2个@#%&中的字符。不要解释。'
                if mode=='message' else '依据你刚才的私有分析，现在只输出正式动作的槽位编号0或1。不要解释。')}]
        return self.infer(final_messages,mode=mode,temperature=temperature,seed=seed,
            label={**(label or {}),'stage':'final_output'})

    def stats(self):
        return {'calls':self.calls,'load_seconds':self.load_seconds,
            'inference_seconds':sum(self.elapsed),
            'max_prompt_tokens':max(self.prompt_tokens,default=0),
            'peak_mlx_memory_gb':mx.get_peak_memory()/1e9}

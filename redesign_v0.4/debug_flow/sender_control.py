"""Engineering control: train senders against explicitly scripted interpreters.

The interpreters know resource categories and a human-specified codebook. This
is not an emergence experiment. It tests the sender's reward-gradient route.
"""
from pathlib import Path
import copy, json, sys, time
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_pilot import ImageBank, make_agents, public_tensor, write_json
from agents import draw
from resource_env import sample_scenes, transition

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'debug_flow/sender_control'

def interpret(kinds, sent):
    # Codes 1=FF, 2=mixed, 3=WW; 0/4 use the mixed tie-breaking rule.
    n = len(kinds)
    actions = np.zeros((n, 2), dtype=np.int64)
    for who in range(2):
        peer_message = sent[:, 1 - who]
        preferred = np.full(n, who, dtype=np.int64)
        preferred[peer_message == 1] = 1
        preferred[peer_message == 3] = 0
        has_preferred = kinds[:, who] == preferred[:, None]
        actions[:, who] = has_preferred.argmax(1)
    return actions

@torch.no_grad()
def evaluate(agents, bank, seed, n=8192):
    rng = np.random.default_rng(seed)
    kinds = sample_scenes(rng, n)
    f, _ = bank.sample(kinds, 'test', rng)
    public = public_tensor(np.zeros((n, 2), dtype=np.int64), 1, 1)
    sent = np.column_stack([agents[i].send(agents[i].observe(f[:, i], public)[1]).argmax(-1).numpy() for i in range(2)])
    _, reward, _ = transition(np.zeros((n, 2), dtype=np.int64), kinds, interpret(kinds, sent))
    shuffled = sent.copy()
    for who in range(2):
        shuffled[:, who] = sent[rng.permutation(n), who]
    _, broken_reward, _ = transition(np.zeros((n, 2), dtype=np.int64), kinds, interpret(kinds, shuffled))
    return {'normal_success': float(reward.mean()), 'shuffled_success': float(broken_reward.mean())}

def main():
    OUT.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    config = {'seeds': [101, 202, 303], 'updates': 500, 'batch': 128,
              'lr': .0003, 'entropy': .01, 'scope': 'scripted recipient positive control; no emergence claim'}
    write_json(OUT / 'config.json', config)
    bank = ImageBank()
    results = []
    for seed in config['seeds']:
        start = time.monotonic()
        agents = make_agents(seed)
        states = torch.load(ROOT / f'results/pilot_001/prepared_s{seed}.pt', weights_only=True)
        for a, s in zip(agents, states):
            a.load_state_dict(s)
        before = evaluate(agents, bank, seed + 19000)
        optimizers = [torch.optim.Adam(a.parameters(), lr=config['lr']) for a in agents]
        rng = np.random.default_rng(seed + 20000)
        prng = [np.random.default_rng(seed + 21000 + i) for i in range(2)]
        curve = []
        for update in range(config['updates']):
            n = config['batch']
            kinds = sample_scenes(rng, n)
            features, _ = bank.sample(kinds, 'train', rng)
            public = public_tensor(np.zeros((n, 2), dtype=np.int64), 1, 1)
            outputs = [draw(agents[i].send(agents[i].observe(features[:, i], public)[1]), prng[i]) for i in range(2)]
            sent = np.column_stack([o[0].numpy() for o in outputs])
            _, reward, _ = transition(np.zeros((n, 2), dtype=np.int64), kinds, interpret(kinds, sent))
            advantage = torch.from_numpy(reward.astype(np.float32) - .7)
            for i in range(2):
                loss = -(advantage * outputs[i][1]).mean() - config['entropy'] * outputs[i][2].mean()
                optimizers[i].zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(agents[i].parameters(), 2.)
                optimizers[i].step()
            if (update + 1) % 100 == 0:
                curve.append({'update': update + 1, **evaluate(agents, bank, seed + 19000)})
        result = {'seed': seed, 'before': before, 'after': evaluate(agents, bank, seed + 29000),
                  'curve': curve, 'seconds': time.monotonic() - start}
        results.append(result)
        write_json(OUT / 'results.json', results)
        torch.save([a.state_dict() for a in agents], OUT / f'final_s{seed}.pt')
        print(json.dumps(result), flush=True)

if __name__ == '__main__':
    main()

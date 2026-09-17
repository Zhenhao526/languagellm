"""Replay original first 100 updates with more logging; assert identical weights."""
from pathlib import Path
import argparse, json, sys
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_pilot import ImageBank, make_agents, train_one, write_json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=ROOT / 'debug_flow/telemetry_replay')
    args = parser.parse_args()
    torch.set_num_threads(4)
    original = ROOT / 'results/pilot_001'
    config = json.loads((original / 'config.json').read_text())
    config.update(updates=100, checkpoint_every=10, evaluation_episodes=128)
    args.out.mkdir(exist_ok=False)
    write_json(args.out / 'config.json', config)
    bank, checks = ImageBank(), []
    for seed in config['seeds']:
        agents = make_agents(seed)
        states = torch.load(original / f'prepared_s{seed}.pt', weights_only=True)
        for agent, state in zip(agents, states):
            agent.load_state_dict(state)
        train_one(agents, bank, seed, 'communicate', args.out / f'communicate_s{seed}', config)
        expected = torch.load(original / f'communicate_s{seed}/checkpoint_0100.pt', weights_only=True)
        actual = torch.load(args.out / f'communicate_s{seed}/checkpoint_0100.pt', weights_only=True)
        same = all(torch.equal(x[k], y[k]) for x, y in zip(expected, actual) for k in x)
        checks.append({'seed': seed, 'checkpoint_100_bitwise_equal_to_original': same})
        assert same, 'Instrumentation changed the original trajectory'
        write_json(args.out / 'trajectory_checks.json', checks)

if __name__ == '__main__':
    main()

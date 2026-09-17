"""Render the first recorded interaction as an auditable task illustration."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .env import VisualWorld


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    cfg = json.loads((args.run / 'config.json').read_text())
    with (args.run / 'interactions.jsonl').open() as stream:
        trial = json.loads(next(stream))
    batch_n = min(cfg['rollout'], cfg['eval_every'], cfg['planned_steps'])
    batch = VisualWorld(cfg['seed'] * 100000 + 1001).sample(batch_n)
    assert int(batch['target_ids'][0]) == trial['target_id']
    assert batch['candidate_ids'][0].tolist() == trial['candidate_ids']
    fig, axes = plt.subplots(1, 5, figsize=(10, 2.9))
    pictures = [batch['target_images'][0, 0], *batch['candidate_images'][0, :, 0]]
    for i, (axis, pixels) in enumerate(zip(axes, pictures)):
        axis.imshow(pixels, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
        axis.set_xticks([]); axis.set_yticks([])
        axis.set_title('Sender sees' if i == 0 else f'Candidate {i}', fontsize=10)
        if i == trial['choice'] + 1:
            for spine in axis.spines.values():
                spine.set_color('#245ca6'); spine.set_linewidth(3)
            axis.set_xlabel('Receiver chose', fontsize=9, color='#245ca6')
    sender = 'A' if trial['sender'] == 0 else 'B'
    receiver = 'B' if sender == 'A' else 'A'
    fig.suptitle(f"Actual trial 1 | {cfg['task']} / {cfg['condition']} / seed {cfg['seed']} | "
                 f"{sender} sends S{trial['message']} to {receiver} | reward = {trial['reward']:.0f}", fontsize=11)
    fig.text(.5, .025, 'Agents receive pixels and a message ID; category names are never shown to them.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .055, 1, .94))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()

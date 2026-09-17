"""Frozen visual senders: matched reward and supervised receiver diagnostics."""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
import hashlib
from itertools import product
import json
from pathlib import Path
import platform
import shutil
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'redesign_v0.10'))
import run_generalization as v10
from camp import MAPS, ImageBank, remake_agents, projected_banks, scene_visual, draw, write_json
sys.path.insert(0, str(ROOT))
from receiver_metrics import evaluate

SEEDS = [29101, 29102, 29103, 29104]
ARMS = ['warm_rl24', 'warm_ce24', 'fresh_ce24', 'fresh_ce30']
TIMES = [0, 100, 300, 600, 1200, 1800, 2400]
BEHAVIOR = ('receive_embedding.', 'actor.')
CRITIC = ('receive_value.',)
SPEC = ROOT.parent / 'paper_program/receiver_baseline/固定发送者接收诊断_实施方案.md'
DATA = ROOT.parent / 'redesign_v0.4/data'
NAMESPACE = 'receiver-baseline|2026-09-15'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seed_identity(seed, partition, direction, stream_kind, step=0):
    label = f'{NAMESPACE}|{seed}|{partition}|{direction}|{stream_kind}|{step}'
    return label, int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], 'big')


def rng_for(seed, partition, direction, kind, step=0):
    return np.random.default_rng(seed_identity(seed, partition, direction, kind, step)[1])


def array_sha(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def state_sha(agent, excluded=()):
    h = hashlib.sha256()
    for key, value in agent.state_dict().items():
        if not key.startswith(excluded):
            h.update(key.encode())
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def sources():
    return [Path(__file__), ROOT / 'receiver_metrics.py', ROOT / '固定执行方案.md',
            ROOT / '测量方案.md', SPEC, DATA / 'encoder_report.json'] + v10.sources()


def photo_split(bank):
    report = json.loads((DATA / 'encoder_report.json').read_text())
    assert report['manifest_sha256'] == sha(DATA / 'manifest.json')
    assert report['features_sha256'] == sha(DATA / 'features.npz')
    result = {'fit': [], 'dev': [], 'evaluation': []}
    for kind in (0, 1):
        ordered = sorted(bank.pools['train', kind].tolist(), key=lambda i:
            hashlib.sha256((NAMESPACE + '|photo-split|' + bank.entries[i]['sha256']).encode()).hexdigest())
        assert len(ordered) == 22 and len(bank.pools['test', kind]) == 8
        for split, rows in [('fit', ordered[:16]), ('dev', ordered[16:]),
                            ('evaluation', bank.pools['test', kind].tolist())]:
            result[split].append(rows)
    detail = {split: [[dict(feature_row=i, image_id=bank.entries[i]['id'],
                    sha256=bank.entries[i]['sha256']) for i in ids] for ids in kinds]
              for split, kinds in result.items()}
    return result, detail


def load_source(seed, partition, source_root):
    folder = source_root / f's{seed}_p{partition}_base'
    prepared = torch.load(source_root / f'prepared_{seed}.pt', weights_only=True)
    agents = remake_agents(seed, prepared, 7, 2, 'identity')
    for agent, state in zip(agents, torch.load(folder / 'final.pt', weights_only=True)):
        agent.load_state_dict(state)
        agent.requires_grad_(False)
    assert v10.v8.state_sha(agents) == json.loads((folder / 'result.json').read_text())['final_sha256']
    return agents


@torch.no_grad()
def make_context(sender, projected, rows):
    photos = np.asarray(list(product(*rows)), dtype=np.int64)
    indices = np.asarray(list(product(range(30), range(len(photos)))), dtype=np.int64)
    positions = MAPS[indices[:, 0]].copy()
    photo_ids = photos[indices[:, 1]].copy()
    hidden, first, second, greedy = [], [], [], []
    for lo in range(0, len(indices), 512):
        pos, ids = positions[lo:lo+512], photo_ids[lo:lo+512]
        h = sender.observe(scene_visual(pos, ids, projected))
        n = len(h)
        state = sender.send_context(torch.cat((h, torch.zeros(n, 4)), 1))
        f = sender.send_out(state)
        token = torch.arange(7).repeat(n)
        nxt = sender.send_recur(sender.send_embedding(token), state.repeat_interleave(7, 0))
        s = sender.send_out(nxt).reshape(n, 7, 7)
        msg, _, _, _ = sender.send(h, torch.zeros(n, 2), torch.zeros(n, 2), np.random.default_rng(12), True)
        hidden.append(h.numpy()); first.append(f.numpy()); second.append(s.numpy()); greedy.append(msg.numpy())
    return dict(map_ids=indices[:, 0], positions=positions, photo_ids=photo_ids,
                h=np.concatenate(hidden), first_logits=np.concatenate(first),
                second_logits=np.concatenate(second), greedy_message=np.concatenate(greedy))


def softmax(x):
    x = np.asarray(x, dtype=np.float64)
    p = np.exp(x - x.max(-1, keepdims=True))
    return p / p.sum(-1, keepdims=True)


def probabilities(context, mode='native'):
    if mode == 'native':
        p = (softmax(context['first_logits'])[:, :, None] * softmax(context['second_logits'])).reshape(-1, 49)
    else:
        assert mode == 'greedy'
        msg = context['greedy_message']
        p = np.eye(49)[msg[:, 0] * 7 + msg[:, 1]]
    assert np.allclose(p.sum(1), 1, atol=1e-12, rtol=0)
    return p


def joint_distribution(context, pool, mode='native'):
    take = np.isin(context['map_ids'], pool)
    pos = context['positions'][take]
    p = probabilities(context, mode)[take]
    c = np.zeros((49, 6, 6), dtype=np.float64)
    for f, w in MAPS:
        c[:, f, w] = p[(pos[:, 0] == f) & (pos[:, 1] == w)].sum(0) / len(p)
    return c


@torch.no_grad()
def receiver_logits(agent):
    messages = torch.tensor(list(product(range(7), repeat=2)), dtype=torch.int64)
    out = []
    for kind in (0, 1):
        goal = torch.zeros(49, 2); goal[:, kind] = 1
        logits, _ = agent.receive(messages, goal, torch.zeros(49, 2), torch.zeros(49, 18),
                                  torch.arange(6).repeat(49, 1))
        out.append(logits.numpy())
    return np.stack(out, 1)


def compact(metrics):
    out = {k: v for k, v in metrics.items() if np.isscalar(v)}
    for key in ('stochastic', 'greedy', 'oracle_branch_ce', 'oracle_joint_map', 'oracle_mixed_reward'):
        out[key] = {k: v for k, v in metrics[key].items() if np.isscalar(v)}
    return out


def prepare_direction(seed, partition, direction, agents, bank, split, detail, args):
    folder = args.out / f'source_s{seed}_p{partition}_d{direction}'
    folder.mkdir(exist_ok=False)
    projected = projected_banks(agents, bank)[direction]
    contexts = {name: make_context(agents[direction], projected, rows) for name, rows in split.items()}
    for name, context in contexts.items():
        np.savez_compressed(folder / f'context_{name}.npz', **context)
    torch.save([a.state_dict() for a in agents], folder / 'source.pt')
    source_folder = args.source_root / f's{seed}_p{partition}_base'
    files = [source_folder / 'final.pt', source_folder / 'config.json', source_folder / 'result.json',
             args.source_root / f'prepared_{seed}.pt', DATA / 'encoder_report.json',
             DATA / 'manifest.json', DATA / 'features.npz']
    write_json(folder / 'config.json', dict(seed=seed, partition=partition, direction=direction,
        sender=direction, receiver=1-direction, source_hashes={str(p): sha(p) for p in files},
        frozen_sender_sha256=state_sha(agents[direction]), source_receiver_sha256=state_sha(agents[1-direction]),
        photo_split=detail, context_hashes={name: sha(folder / f'context_{name}.npz') for name in contexts},
        source_state_file_sha256=sha(folder / 'source.pt'), observation_cache='float32 hidden state only; native sender.send executed for every training batch'))
    return folder, contexts


def make_receiver(original, seed, partition, direction, arm):
    agent = copy.deepcopy(original)
    if arm.startswith('fresh'):
        torch.manual_seed(seed_identity(seed, partition, direction, 'receiver_initialization')[1])
        agent.receive_embedding.reset_parameters()
        for module in agent.actor.modules():
            if isinstance(module, torch.nn.Linear): module.reset_parameters()
    active = BEHAVIOR + (CRITIC if arm == 'warm_rl24' else ())
    for name, param in agent.named_parameters(): param.requires_grad_(name.startswith(active))
    return agent, active


def training_batch(sender, receiver, context, pool, seed, partition, direction, arm, step, batch):
    support = '30' if arm.endswith('30') else '24'
    streams = {k: seed_identity(seed, partition, direction, k+support, step) for k in ('world', 'send', 'action')}
    wr = np.random.default_rng(streams['world'][1]); sr = np.random.default_rng(streams['send'][1])
    rr = np.random.default_rng(streams['action'][1])
    possible = np.flatnonzero(np.isin(context['map_ids'], pool))
    ix = wr.choice(possible, batch)
    h = torch.from_numpy(context['h'][ix])
    with torch.no_grad():
        sent, _, _, _ = sender.send(h, torch.zeros(batch, 2), torch.zeros(batch, 2), sr, False)
    logits, values = [], []
    for kind in (0, 1):
        goal = torch.zeros(batch, 2); goal[:, kind] = 1
        logit, value = receiver.receive(sent, goal, torch.zeros(batch, 2), torch.zeros(batch, 18),
                                        torch.arange(6).repeat(batch, 1))
        logits.append(logit); values.append(value)
    positions = context['positions'][ix]
    truth = torch.from_numpy(positions)
    ce = F.cross_entropy(logits[0], truth[:, 0]) + F.cross_entropy(logits[1], truth[:, 1])
    arrays = dict(context_indices=ix, map_ids=context['map_ids'][ix], positions=positions,
        photo_ids=context['photo_ids'][ix], h=context['h'][ix], sent=sent.numpy(),
        receiver_logits=torch.stack(logits, 1).detach().numpy(), values=torch.stack(values, 1).detach().numpy())
    entropy_weight = .02 if step < 2100 else 0.
    if arm == 'warm_rl24':
        action, lp, ent = [], [], []
        for logit in logits:
            a, logp, entropy = draw(logit, rr, False)
            action.append(a); lp.append(logp); ent.append(entropy)
        chosen = torch.stack(action, 1)
        success = (chosen == truth).float()
        reward = .25 * success.sum(1) + .5 * success.prod(1)
        target = reward - 1
        value = torch.stack(values).mean(0)
        policy = -((lp[0]+lp[1]) * (target-value).detach()).mean()
        value_loss = .5 * F.mse_loss(value, target)
        entropy = (ent[0]+ent[1]).mean()
        loss = policy + value_loss - entropy_weight * entropy
        arrays.update(action=chosen.numpy(), successes=success.numpy(), reward=reward.numpy(), target=target.numpy())
        components = dict(policy_loss=float(policy.detach()), value_loss=float(value_loss.detach()),
                          entropy=float(entropy.detach()), entropy_weight=entropy_weight,
                          mean_reward=float(reward.mean()), J=float(success.prod(1).mean()))
    else:
        loss = ce
        components = dict(entropy_weight=0.)
    components.update(loss=float(loss.detach()), branch_ce_sample=float(ce.detach()))
    return loss, arrays, components, {k: dict(label=v[0], seed=v[1]) for k,v in streams.items()}


def fit(seed, partition, direction, arm, agents, source_folder, contexts, args):
    dest = args.out / f's{seed}_p{partition}_d{direction}_{arm}'
    dest.mkdir(exist_ok=False)
    sender = agents[direction]; sender_hash = state_sha(sender)
    receiver, active = make_receiver(agents[1-direction], seed, partition, direction, arm)
    frozen_hash = state_sha(receiver, active)
    pool = np.arange(30) if arm.endswith('30') else np.sort(np.r_[v10.partition(partition)['old'], v10.partition(partition)['added']])
    checkpoints = sorted(set([0, args.updates] + [x for x in TIMES if x < args.updates]))
    opt = torch.optim.Adam([p for p in receiver.parameters() if p.requires_grad], lr=.0007)
    cfg = dict(seed=seed, partition=partition, direction=direction, sender=direction, receiver=1-direction, arm=arm,
        updates=args.updates, batch=args.batch, training_pool=pool.tolist(), map_groups={k:v.tolist() for k,v in v10.partition(partition).items()},
        checkpoints=checkpoints, forensic_pre_update=2100, learning_rate=.0007, entropy_off_after=2100,
        source_folder=str(source_folder), source_config_sha256=sha(source_folder/'config.json'),
        source_hashes=args.source_hashes, initial_sha256=state_sha(receiver), frozen_sha256=frozen_hash,
        frozen_sender_sha256=sender_hash, trainable_names=[k for k,p in receiver.named_parameters() if p.requires_grad],
        trainable_parameters=sum(p.numel() for p in receiver.parameters() if p.requires_grad),
        behavioral_parameters=sum(p.numel() for k,p in receiver.named_parameters() if k.startswith(BEHAVIOR)),
        initialization_rng=dict(zip(('label','seed'), seed_identity(seed, partition, direction, 'receiver_initialization'))),
        extra_supervision=arm!='warm_rl24', sealed_supervised_exposure=arm=='fresh_ce30',
        label_input='loss only; receiver sees integers, private goal and zero inventory/history',
        world_sampling='uniform with replacement over full fit context Cartesian product in support',
        menu='identity', goal_order=[0,1], receiver_loss_reduction='sum of two goal losses; no sender role constant',
        rng_template=NAMESPACE+'|source_seed|partition|direction|stream_kind|step',
        selection='CE only: lowest native-send dev-support branch NLL among planned checkpoints including zero; earliest tie',
        evaluation_photos='old development photographs; not independent visual confirmation')
    write_json(dest/'config.json',cfg)
    torch.save(receiver.state_dict(),dest/'initial.pt');torch.save(opt.state_dict(),dest/'initial_optimizer.pt')
    distributions = {s:joint_distribution(contexts[s],pool) for s in ('fit','dev')}
    curve=[]; start=time.monotonic()
    with (dest/'training.jsonl').open('w') as log:
        for step in range(args.updates+1):
            if step in checkpoints:
                assert state_sha(receiver,active)==frozen_hash and state_sha(sender)==sender_hash
                logits=receiver_logits(receiver)
                scores={s:compact(evaluate(c,logits)) for s,c in distributions.items()}
                curve.append(dict(update=step,scores=scores,state_sha256=state_sha(receiver)))
                write_json(dest/'curve.json',curve)
                np.savez_compressed(dest/f'receiver_{step:04d}.npz',logits=logits)
                torch.save(receiver.state_dict(),dest/f'checkpoint_{step:04d}.pt')
                torch.save(opt.state_dict(),dest/f'optimizer_{step:04d}.pt')
                if step==args.updates or step%600==0:
                    print(json.dumps(dict(run=dest.name,update=step,fit_nll=scores['fit']['branch_nll'],
                        fit_J=scores['fit']['stochastic']['J'])),flush=True)
            if step==args.updates:break
            if step==2100:
                torch.save(receiver.state_dict(),dest/'forensic_2100.pt')
                torch.save(opt.state_dict(),dest/'forensic_optimizer_2100.pt')
            loss,arrays,components,streams=training_batch(sender,receiver,contexts['fit'],pool,
                seed,partition,direction,arm,step,args.batch)
            assert torch.isfinite(loss)
            if step in (0,2100):np.savez_compressed(dest/f'train_{step+1:04d}.npz',**arrays)
            opt.zero_grad(set_to_none=True);loss.backward()
            assert all(p.grad is None for p in sender.parameters())
            assert all(p.grad is None for p in receiver.parameters() if not p.requires_grad)
            norm=float(torch.nn.utils.clip_grad_norm_([p for p in receiver.parameters() if p.requires_grad],2.))
            assert np.isfinite(norm)
            opt.step()
            if step==0:
                torch.save(receiver.state_dict(),dest/'after_first_update.pt')
                torch.save(opt.state_dict(),dest/'after_first_optimizer.pt')
            exposure={k:int(np.isin(arrays['map_ids'],v).sum()) for k,v in v10.partition(partition).items()}
            assert arm=='fresh_ce30' or exposure['sealed']==0
            log.write(json.dumps(dict(update=step+1,streams=streams,exposure=exposure,
                world_sha256=array_sha(arrays['context_indices'],arrays['positions'],arrays['photo_ids']),
                message_sha256=array_sha(arrays['sent']),
                action_sha256=array_sha(arrays['action']) if 'action' in arrays else None,
                components=components,gradient_norm=norm))+'\n')
            if (step+1)%100==0:log.flush()
    assert state_sha(receiver,active)==frozen_hash and state_sha(sender)==sender_hash
    torch.save(receiver.state_dict(),dest/'final.pt');torch.save(opt.state_dict(),dest/'final_optimizer.pt')
    selected=min(curve,key=lambda r:(r['scores']['dev']['branch_nll'],r['update']))['update'] if arm!='warm_rl24' else None
    write_json(dest/'result.json',dict(seed=seed,partition=partition,direction=direction,arm=arm,
        updates=args.updates,complete=True,final_sha256=state_sha(receiver),frozen_verified=True,
        frozen_sender_verified=True,selected_dev_update=selected,seconds=time.monotonic()-start,
        final_support_scores=curve[-1]['scores'],formal_evaluation_not_yet_read=True))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=ROOT/'results/receiver_001')
    parser.add_argument('--source-root',type=Path,default=ROOT.parent/'redesign_v0.10/results/generalization_001')
    parser.add_argument('--seeds',nargs='+',type=int,default=SEEDS)
    parser.add_argument('--partitions',nargs='+',type=int,default=[1,2,3])
    parser.add_argument('--directions',nargs='+',type=int,default=[0,1])
    parser.add_argument('--updates',type=int,default=2400)
    parser.add_argument('--batch',type=int,default=512)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args(); torch.set_num_threads(1)
    args.out=args.out.resolve();args.source_root=args.source_root.resolve()
    if not args.smoke:
        assert (args.seeds,args.partitions,args.directions,args.updates,args.batch)==(SEEDS,[1,2,3],[0,1],2400,512)
        assert args.source_root==ROOT.parent/'redesign_v0.10/results/generalization_001'
        qa=json.loads((ROOT/'preflight_qa.json').read_text())
        assert qa['passed'] and qa['runner_sha256']==sha(__file__)
        for p,h in qa['frozen_source_hashes'].items():assert sha(p)==h
    args.out.mkdir(parents=True,exist_ok=False)
    args.source_hashes={str(p):sha(p) for p in sources()}
    snapshot=args.out/'frozen_sources';snapshot.mkdir()
    for i,p in enumerate(sources()):
        if p.suffix in ('.py','.md','.json'):shutil.copy2(p,snapshot/f'{i:02d}_{p.name}')
    bank=ImageBank();split,detail=photo_split(bank)
    write_json(args.out/'photo_split.json',detail)
    write_json(args.out/'invocation.json',dict(started_utc=datetime.now(timezone.utc).isoformat(),formal=not args.smoke,
        seeds=args.seeds,partitions=args.partitions,directions=args.directions,arms=ARMS,updates=args.updates,batch=args.batch,
        source_root=str(args.source_root),source_hashes=args.source_hashes,python=platform.python_version(),
        torch=str(torch.__version__),device='cpu',torch_threads=1))
    for seed,partition in product(args.seeds,args.partitions):
        agents=load_source(seed,partition,args.source_root)
        for direction in args.directions:
            source_folder,contexts=prepare_direction(seed,partition,direction,agents,bank,split,detail,args)
            for arm in ARMS:fit(seed,partition,direction,arm,agents,source_folder,contexts,args)
    write_json(args.out/'training_complete.json',dict(complete=True,runs=len(args.seeds)*len(args.partitions)*len(args.directions)*4,
        completed_utc=datetime.now(timezone.utc).isoformat(),source_hashes=args.source_hashes))
    print('COMPLETE',flush=True)


if __name__=='__main__':main()

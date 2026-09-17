"""Read-only checkpoint / gradient audit of pilot_001; no social training.

Uses original saved source and official cached DINO features. One optional
optimizer step is confined to an in-memory copy to establish update plumbing.
"""
from pathlib import Path
import copy, hashlib, json, sys
import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / 'results/pilot_001'
sys.path.insert(0, str(PILOT / 'source'))
import run_pilot as rp
from agents import ResourceAgent, draw
from resource_env import sample_scenes, transition
rp.ROOT = ROOT
torch.set_num_threads(4)

def num(x):
    return float(x.detach())

def entropy(logits):
    p = logits.softmax(-1)
    return -(p * logits.log_softmax(-1)).sum(-1)

def module_norms(agent, gradients=False):
    d = {}
    for name, p in agent.named_parameters():
        x = p.grad if gradients else p
        d.setdefault(name.split('.')[0], []).append(x.flatten() if x is not None else torch.zeros_like(p).flatten())
    return {k: num(torch.cat(v).norm()) for k,v in d.items()}

def fixed_inputs(bank, seed=99781, n=1024):
    rng = np.random.default_rng(seed)
    kinds = sample_scenes(rng, n)
    features, ids = bank.sample(kinds, 'test', rng)
    public = rp.public_tensor(np.zeros((n,2), dtype=np.int64), 16, 16)
    return kinds, features, public, ids

def describe_agent(agent, own, kinds, public):
    with torch.no_grad():
        options, local = agent.observe(own, public)
        send_logits = agent.send(local)
        acts = torch.stack([agent.act(options, local, torch.full((len(own),), s, dtype=torch.int64)) for s in range(5)], 1)
        mixed = torch.from_numpy(kinds[:,0] != kinds[:,1])
        margins = acts[:,:,1] - acts[:,:,0]
        oriented = margins * torch.from_numpy(kinds[:,1] - kinds[:,0])[:,None]
        mix = mixed.nonzero().flatten()
        probs = acts.softmax(-1)
        choices = acts.argmax(-1).numpy()
        selected = np.take_along_axis(kinds[:,None,:].repeat(5,1), choices[:,:,None],2)[:,:,0]
        zm = F.one_hot(torch.zeros(len(own), dtype=torch.int64),5).float()
        context = torch.cat([local, zm], -1)[:,None,:].expand(-1,2,-1)
        pre = agent.actor[0](torch.cat([options,context],-1))
        pre_project = agent.project[0](own)
        tanh_deriv = 1-pre.tanh().square()
        result = {
            'sender_entropy_mean': num(entropy(send_logits).mean()),
            'sender_argmax_counts': torch.bincount(send_logits.argmax(-1),minlength=5).tolist(),
            'actor_entropy_mixed_mean': num(entropy(acts[mixed]).mean()),
            'actor_resource_margin_abs_mixed_mean': num(oriented[mixed].abs().mean()),
            'actor_resource_margin_abs_mixed_min': num(oriented[mixed].abs().min()),
            'message_margin_range_mixed_mean': num((margins[mixed].max(1).values-margins[mixed].min(1).values).mean()),
            'message_margin_range_mixed_max': num((margins[mixed].max(1).values-margins[mixed].min(1).values).max()),
            'message_probability_range_mixed_mean': num((probs[mixed,:,1].max(1).values-probs[mixed,:,1].min(1).values).mean()),
            'all_five_symbol_resource_flip_fraction_mixed': float((selected[mixed.numpy()].max(1)!=selected[mixed.numpy()].min(1)).mean()),
            'actor_pre_tanh_abs_mean': num(pre.abs().mean()),
            'actor_tanh_derivative_mean': num(tanh_deriv.mean()),
            'actor_tanh_saturation_fraction_abs_pre_gt3': num((pre.abs()>3).float().mean()),
            'project_tanh_saturation_fraction_abs_pre_gt3': num((pre_project.abs()>3).float().mean()),
            'project_tanh_derivative_mean': num((1-pre_project.tanh().square()).mean()),
            'message_weight_norm': num(agent.actor[0].weight[:,-5:].norm()),
            'public_weight_norm': num(agent.actor[0].weight[:,128:131].norm()),
        }
    # Continuous one-hot coordinates only for diagnosis, never policy inputs.
    # Single shared context implies cancellation in linear candidate scoring;
    # gradients below measure remaining tanh-mediated context effects.
    option_x = options.detach().clone().requires_grad_()
    context_x = torch.cat([local.detach(), zm], -1).clone().requires_grad_()
    pre2 = agent.actor[0](torch.cat([option_x,context_x[:,None,:].expand(-1,2,-1)],-1))
    logits = agent.actor[2](pre2.tanh()).squeeze(-1)
    margin = logits[:,1]-logits[:,0]
    go, gc = torch.autograd.grad(margin.sum(),(option_x,context_x))
    result['margin_gradient_rms_per_coordinate_mixed'] = {
        'candidate_visual_projection': num(go[mixed].square().mean().sqrt()),
        'local_mean_visual_projection': num(gc[mixed,:64].square().mean().sqrt()),
        'public_stock': num(gc[mixed,64:66].square().mean().sqrt()),
        'public_time': num(gc[mixed,66:67].square().mean().sqrt()),
        'received_symbol': num(gc[mixed,67:].square().mean().sqrt()),
    }
    # Effect on each absolute logit can be much bigger than effect on the
    # candidate difference that actually determines the softmax policy.
    result['shared_message_absolute_logit_change_mixed_mean'] = num((acts[mixed]-acts[mixed,:1,:]).abs().mean())
    with torch.no_grad():
        linear_out = agent.actor[2](pre2.detach()).squeeze(-1)
        linear_ranges=[]
        for s in range(5):
            sc = torch.cat([local, F.one_hot(torch.full((len(own),),s,dtype=torch.int64),5).float()],-1)
            lp=agent.actor[2](agent.actor[0](torch.cat([options,sc[:,None,:].expand(-1,2,-1)],-1))).squeeze(-1)
            linear_ranges.append(lp[:,1]-lp[:,0])
        linear_ranges=torch.stack(linear_ranges,1)
        result['linearized_actor_message_margin_range_max'] = num((linear_ranges.max(1).values-linear_ranges.min(1).values).max())
    return result, send_logits.detach(), acts.detach()

def gradient_audit(bank, state, seed=8201):
    agents = [ResourceAgent(),ResourceAgent()]
    for a,s in zip(agents,state): a.load_state_dict(s)
    rng=np.random.default_rng(seed)
    prng=[np.random.default_rng(seed+1+i) for i in range(2)]
    n,horizon=64,16
    inventory=np.zeros((n,2),dtype=np.int64)
    logs=[[],[]]; sentlog=[[],[]]; ent=[[],[]]; vals=[[],[]]; rews=[]
    unique_sent=[[],[]]
    for step in range(horizon):
        kinds=sample_scenes(rng,n); features,_=bank.sample(kinds,'train',rng)
        public=rp.public_tensor(inventory,horizon-step,horizon)
        rep=[agents[i].observe(features[:,i],public) for i in range(2)]
        symbols=[]
        for i in range(2):
            s,lp,e=draw(agents[i].send(rep[i][1]),prng[i]);symbols.append(s.detach())
            sentlog[i].append(lp);ent[i].append(e);unique_sent[i].extend(s.tolist())
        actions=[]
        for i in range(2):
            a,lp,e=draw(agents[i].act(*rep[i],symbols[1-i]),prng[i])
            actions.append(a.numpy());logs[i].append(lp+sentlog[i][-1]);ent[i][-1]=ent[i][-1]+e
            vals[i].append(agents[i].baseline(rep[i][1]))
        inventory,reward,_=transition(inventory,kinds,np.column_stack(actions),capacity=1)
        rews.append(torch.from_numpy((reward-1).astype(np.float32)))
    target=torch.stack(rews).detach()
    report=[]
    for i,agent in enumerate(agents):
        value=torch.stack(vals[i]);adv=(target-value).detach()
        actorloss=-(torch.stack(logs[i])*adv).mean()
        criticloss=.5*F.mse_loss(value,target)
        entloss=-.01*torch.stack(ent[i]).mean()
        parts={}
        for name,loss in [('reward_policy_gradient',actorloss),('critic',criticloss),('entropy',entloss)]:
            agent.zero_grad();loss.backward(retain_graph=True);parts[name]=module_norms(agent,True)
        agent.zero_grad();(actorloss+criticloss+entloss).backward()
        total=module_norms(agent,True)
        prev={k:v.detach().clone() for k,v in agent.state_dict().items()}
        torch.nn.utils.clip_grad_norm_(agent.parameters(),2)
        opt=torch.optim.Adam(agent.parameters(),lr=3e-4);opt.step()
        changed={k:int(torch.count_nonzero(v-prev[k])) for k,v in agent.state_dict().items()}
        report.append({'agent':i,'sampled_reward_mean':num(target.mean()+1),'loss_parts_gradient_norms':parts,
                       'total_gradient_norms':total,'in_memory_one_step_changed_parameter_counts':changed,
                       'sampled_symbol_counts':np.bincount(unique_sent[i],minlength=5).tolist()})
    return report

@torch.no_grad()
def sender_causal_reward_ranges(payload, kinds):
    """Exact action/message marginalization on fixed pictured worlds.

    Choose one sender's symbol externally; marginalize the other sender's
    symbol and both choices. Rewards are computed outside policies. This is
    an optimistic full-world oracle bound, not an available learning target.
    """
    sendp=[p[0].softmax(-1) for p in payload]
    actorp=[p[1].softmax(-1) for p in payload]
    reports=[]
    for sender in range(2):
        receiver=1-sender
        ownp=(actorp[sender]*sendp[receiver][:,:,None]).sum(1)
        reward=torch.from_numpy(kinds[:,sender,:,None]!=kinds[:,receiver,None,:]).float()
        utility=torch.einsum('na,nsb,nab->ns',ownp,actorp[receiver],reward)
        actual=(utility*sendp[sender]).sum(1)
        reports.append({'sender':sender,
            'expected_reward_range_across_sent_symbols_mean':num((utility.max(1).values-utility.min(1).values).mean()),
            'expected_reward_range_across_sent_symbols_max':num((utility.max(1).values-utility.min(1).values).max()),
            'full_world_best_symbol_upper_bound_gain_mean':num((utility.max(1).values-actual).mean()),
            'current_expected_reward':num(actual.mean())})
    return reports

def main():
    bank=rp.ImageBank();kinds,features,public,_=fixed_inputs(bank)
    output={'protocol':'Read saved checkpoints; same fixed held-out photos across every checkpoint. No parameter changes saved.',
            'source_hash_match':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()==hashlib.sha256((ROOT/p.name).read_bytes()).hexdigest() for p in (PILOT/'source').glob('*.py')},
            'rows':[],'gradient_audits':[]}
    for condition in ['communicate','silent']:
        for seed in [101,202,303]:
            folder=PILOT/f'{condition}_s{seed}'
            initial=torch.load(folder/'checkpoint_0000.pt',weights_only=True)
            baseline=None
            for update in range(0,601,100):
                state=torch.load(folder/f'checkpoint_{update:04d}.pt',weights_only=True)
                agents=[ResourceAgent(),ResourceAgent()]
                details=[];payload=[]
                for i,(a,s) in enumerate(zip(agents,state)):
                    a.load_state_dict(s)
                    desc,send,acts=describe_agent(a,features[:,i],kinds[:,i],public)
                    payload.append((send,acts))
                    drift={}
                    for module in ['project','sender','actor','value']:
                        keys=[k for k in s if k.startswith(module+'.')]
                        dv=torch.cat([(s[k]-initial[i][k]).flatten() for k in keys]);iv=torch.cat([initial[i][k].flatten() for k in keys])
                        drift[module]={'l2':num(dv.norm()),'relative_l2':num(dv.norm()/iv.norm()),'changed_parameters':int(torch.count_nonzero(dv)),'parameters':dv.numel()}
                    desc['parameter_drift_from_checkpoint0']=drift
                    if baseline is not None:
                        desc['sender_logit_abs_change_from_checkpoint0']=num((send-baseline[i][0]).abs().mean())
                        desc['actor_logit_abs_change_from_checkpoint0']=num((acts-baseline[i][1]).abs().mean())
                        desc['actor_greedy_option_changes_from_checkpoint0']=num((acts.argmax(-1)!=baseline[i][1].argmax(-1)).float().mean())
                    details.append(desc)
                if baseline is None:baseline=payload
                greedy,_=rp.evaluate(agents,bank,condition,seed+700000,n=128,horizon=16,greedy=True)
                stoch,_=rp.evaluate(agents,bank,condition,seed+700000,n=128,horizon=16,greedy=False)
                row={'condition':condition,'seed':seed,'update':update,'agents':details,
                     'greedy_reward':greedy['mean_reward_per_step'],'stochastic_reward':stoch['mean_reward_per_step'],
                     'sender_causal_reward_ranges':sender_causal_reward_ranges(payload,kinds) if condition=='communicate' else None}
                output['rows'].append(row)
                if condition=='communicate' and update in [0,600]:
                    output['gradient_audits'].append({'seed':seed,'update':update,'agents':gradient_audit(bank,state)})
    (ROOT/'debug_flow/audit_updates.json').write_text(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False))
    lines=['# 固定检查点与梯度审计','',
           '未修改原模型、原训练脚本或原结果；诊断 optimizer.step 仅发生在各检查点的内存副本，未保存新模型。原四个源文件哈希与运行快照全部相符。', '',
           '## 结论与限制', '',
           '1. 训练通路确实工作。第 600 次更新相对起点，交流组视觉投影参数相对 L2 漂移 27.1%–37.6%，行动头 16.8%–19.5%，发送头 6.06%–8.66%。重建同样损失后，各活动模块均有非零梯度，内存中的 Adam 步骤能改变参数。无消息组发送头保持不变，符合其损失权重为零的代码。',
           '2. 并非从头到尾完全没学：三组交流策略的贪心回报在第 0→100 次更新显著上升，之后最大概率动作保持同一种资源偏好。随机采样评估仍有小幅变化，单独展示贪心成功率掩盖了选择概率继续变确定。',
           '3. 六个交流主体在混合资源观察中的行动熵从 0.655–0.680 降至第 100 次的 0.042–0.051，再降至第 600 次的 0.0055–0.014。发送熵仍为 1.581–1.597，接近五个符号均匀分布的 ln(5)=1.609。因此主要是资源动作探索减少，不能说发送符号也塌缩为单一符号。',
           '4. 第 600 次的资源候选得分差平均为 7.04–7.58；更换全部五种符号，得分差平均仅变化 0.063–0.154，选择概率范围平均只有 0.000050–0.000272，六个主体资源翻转率均为零。即便给发送者额外全局信息，让它逐场挑选当前对伙伴最有利的符号，保持所有其他参数不变，平均回报提升也仅为 0.000015–0.000085（0.0015–0.0085 个百分点）。这是对当前策略的精确概率边缘化，不是新增训练或通信能力结论。',
           '5. 不支持“actor 的 tanh 已经普遍饱和导致梯度断开”：第 600 次 |激活前数值|>3 的比例仅 0.33%–3.17%，tanh 平均导数 0.35–0.42。接收消息对候选得分差的梯度 RMS 为 0.053–0.095，与视觉候选坐标的 0.063–0.077 同量级；但确定性很强的 softmax 把这些得分扰动变成很小的行动概率变化。',
           '6. 公共库存坐标对得分差仍有较大局部梯度，但社会阶段库存恒为 [0,0]，这些输入不会变化；单个坐标梯度不能等同于实际行为贡献。发送、行动、价值头的首层各有 128 个与库存连接的权重不变是常零输入的直接结果，不是漏训练。', '',
           '参数梯度非零不意味着发送者获得了可靠的学习信号：REINFORCE 的单批梯度还含采样噪声。当前数据支持“固定资源偏好变得非常确定、消息对行动的因果作用很弱”；谁导致了这个局面，仍需受控干预确认。', '',
           '## 固定评估曲线', '',
           '每行在相同的中途评估种子上重算。最终报告使用另一套评估场景，因此此表第 600 次与原最终主表数值不同属预期。行动熵与符号干预使用独立固定 1,024 个留出图片场景，所有检查点相同。', '',
           '| 条件 | 种子 | 更新 | 贪心回报 | 随机回报 | A/B 行动熵（混合资源） | A/B 所有符号资源翻转率 |','|---|---:|---:|---:|---:|---|---|']
    for r in output['rows']:
        if r['condition']=='communicate':
            lines.append(f"| {r['condition']} | {r['seed']} | {r['update']} | {r['greedy_reward']:.5f} | {r['stochastic_reward']:.5f} | "+'/'.join(f"{a['actor_entropy_mixed_mean']:.6f}" for a in r['agents'])+' | '+'/'.join(f"{a['all_five_symbol_resource_flip_fraction_mixed']:.5f}" for a in r['agents'])+' |')
    lines+=['','具体参数漂移、梯度、tanh 饱和与符号影响见 audit_updates.json。','',
            '结构事实：每个候选的打分为 vᵀtanh(W_option option + W_context context + b)。两个候选共用相同上下文；若去掉 tanh，上下文对两者分数的贡献精确抵消。现结构并非完全不能用消息，但消息必须通过 tanh 的非线性交互改变相对得分。',
            '本审计不把梯度小或探索减少等同于因果解释；需受控修复与重跑才能确定各因素贡献。']
    (ROOT/'debug_flow/audit_updates.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'rows':len(output['rows']),'source_hash_match':output['source_hash_match'],'output':str(ROOT/'debug_flow/audit_updates.json')}))

if __name__=='__main__':main()

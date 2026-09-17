"""Independent trainable interfaces on top of cached, frozen DINO features."""
import torch
from torch import nn
from torch.nn import functional as F

class ResourceAgent(nn.Module):
    def __init__(self, feature_dim=1024, width=64):
        super().__init__()
        # This projection is part of the new interface, not pretrained DINO.
        self.project = nn.Sequential(nn.Linear(feature_dim, width), nn.Tanh())
        self.sender = nn.Sequential(nn.Linear(width + 3, width), nn.Tanh(), nn.Linear(width, 5))
        self.actor = nn.Sequential(nn.Linear(width * 2 + 3 + 5, width), nn.Tanh(), nn.Linear(width, 1))
        self.value = nn.Sequential(nn.Linear(width + 3, width), nn.Tanh(), nn.Linear(width, 1))

    def observe(self, own_features, public):
        options = self.project(own_features)
        local = torch.cat([options.mean(1), public], -1)
        return options, local

    def send(self, local):
        return self.sender(local)

    def act(self, options, local, received):
        assert received.dtype == torch.int64 and not received.requires_grad
        symbol = F.one_hot(received, 5).float()
        context = torch.cat([local, symbol], -1)[:, None, :].expand(-1, 2, -1)
        return self.actor(torch.cat([options, context], -1)).squeeze(-1)

    def baseline(self, local):
        # No partner observation or sampled message enters this baseline.
        return self.value(local).squeeze(-1)

def draw(logits, rng, greedy=False):
    logp = F.log_softmax(logits, -1)
    p = logp.exp()
    if greedy:
        action = p.argmax(-1)
    else:
        u = torch.from_numpy(rng.random((len(p), 1)).astype('float32'))
        action = (p.detach().cumsum(-1) < u).sum(-1).clamp(max=p.shape[-1] - 1)
    return action, logp.gather(1, action[:, None]).squeeze(1), -(p * logp).sum(-1)

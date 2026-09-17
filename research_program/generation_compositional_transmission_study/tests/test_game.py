from __future__ import annotations
import numpy as np
from .. import design, policy, runner

def run():
    p=policy.make_replacement(76101); ep=design.episode_stream(76101,64)
    tr=runner.rollout(p,ep,'live','worker',sample=False)
    assert tr['messages'].shape==(64,design.HORIZON)
    assert np.all(np.isfinite(tr['rewards']))
    assert len(runner.sequence(p,(0,0)))==2
    assert design.parse_condition('worker_live')==('worker','live')
if __name__=='__main__': run(); print('generation_compositional_transmission tests passed')

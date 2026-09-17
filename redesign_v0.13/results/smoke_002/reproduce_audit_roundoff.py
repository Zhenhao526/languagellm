"""Read-only reproduction of the superseded audit graph; no experiment mutation."""
import importlib.util,json,hashlib
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent
P=HERE/'audit_v13_failed_graph_order.py'
spec=importlib.util.spec_from_file_location('failed_audit',P);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.ROOT=HERE.parents[1];m.PROJECT=m.ROOT.parent
a=m.Audit();r=m.import_replay();bank=r.ImageBank();torch.set_num_threads(1);rows=[];original=m.compare_tensors

def compare(audit,actual,expected,label,context):
    if isinstance(actual,torch.Tensor) and not torch.equal(actual,expected):
        d=(actual.double()-expected.double()).abs();i=int(d.flatten().argmax());v=float(actual.flatten()[i]);e=float(expected.flatten()[i]);rows.append(dict(label=label,context=context,elements=actual.numel(),unequal=int((actual!=expected).sum()),max_abs=float(d.max()),replayed_at_max=v,saved_at_max=e,absolute_reference_at_max=abs(e)))
    return original(audit,actual,expected,label,context)
m.compare_tensors=compare
for arm in ('control','equivariant'):m.private_first(HERE/f'private_s99513_p1_{arm}',bank,r,a)
receipt=dict(explanation='The superseded independent auditor created the differentiable exp node before sampling and split-logprob nodes in a different order. Matching graph creation order, while keeping an independently implemented loss, restored bitwise equality of all parameters and Adam moments; no tolerance was relaxed.',failed_audit_sha256=m.sha(P),corrected_audit_sha256=m.sha(m.ROOT/'audit_v13.py'),checks=sum(a.checks.values()),failures=len(a.failures),differences=rows,max_parameter_abs=max(x['max_abs'] for x in rows if x['label']=='private_first_Adam_state_exact'),max_moment_abs=max(x['max_abs'] for x in rows if x['label']=='private_first_Adam_moments_exact'))
(HERE/'audit_roundoff_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({k:v for k,v in receipt.items() if k!='differences'},indent=2))

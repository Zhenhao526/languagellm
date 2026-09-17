from pathlib import Path
import hashlib,json,sys,itertools
import numpy as np
import torch
root=Path('/Users/xia/Documents/ChatGPT/语言/redesign_v0.4/results/course_control_001')
sys.path.insert(0,str(root/'source'))
from agents import ResourceAgent

torch.set_num_threads(4)
manifest=json.loads((root/'data_manifest.json').read_text())['images']
raw=np.load(root/'data_features.npz')['features'].astype(np.float32)
scaling=np.load(root/'feature_scaling.npz')
features=torch.from_numpy((raw-scaling['center'])/max(float(scaling['scale']),1e-6))
labels=np.asarray([0 if e['category']=='food' else 1 for e in manifest])
splits=np.asarray([e['split'] for e in manifest])
assert not ({e['sha256'] for e in manifest if e['split']=='train'} & {e['sha256'] for e in manifest if e['split']=='test'})
source_hash=json.loads((root/'source_hashes.json').read_text())
assert all(hashlib.sha256((root/'source'/name).read_bytes()).hexdigest()==h for name,h in source_hash.items())
scenes=np.asarray([s for s in itertools.product((0,1),repeat=4) if len(set(s))==2]).reshape(14,2,2)

def cases(seed,n=4096):
 rng=np.random.default_rng(seed)
 kinds=scenes[rng.integers(14,size=n)].copy()
 ids=np.empty(kinds.shape,dtype=np.int64)
 for kind in (0,1):
  mask=kinds==kind
  ids[mask]=rng.choice(np.flatnonzero((splits=='test')&(labels==kind)),int(mask.sum()))
 public=torch.zeros(n,3);public[:,2]=torch.tensor((16-np.arange(n)%16)/16,dtype=torch.float32)
 return kinds,features[torch.from_numpy(ids)],public

def array(rows,key):return np.asarray([r[key] for r in rows])

audits=[]
with torch.no_grad():
 for folder in sorted(root.glob('*_s*')):
  if not folder.is_dir():continue
  result=json.loads((folder/'result.json').read_text())
  agents=[ResourceAgent(),ResourceAgent()]
  for agent,state in zip(agents,torch.load(folder/'checkpoint_1200.pt',weights_only=True)):
   agent.load_state_dict(state,strict=True);agent.eval()
  checks={'run':folder.name,'traces':[],'direction_gaps':[]}
  for task in ('curriculum','full'):
   sets={}
   for mode in ('normal','shuffle','blank'):
    rows=[json.loads(line) for line in (folder/f'final_{task}_{mode}_trace.jsonl').read_text().splitlines()]
    assert len(rows)==8192
    a={key:array(rows,key) for key in rows[0]}
    kinds,ids,actions=a['kinds'],a['image_ids'],a['actions']
    assert np.all(labels[ids]==kinds) and np.all(splits[ids]=='test')
    assert np.all(a['inventory']==0)
    selected=np.take_along_axis(kinds,actions[...,None],axis=-1)[...,0]
    success=selected[:,0]!=selected[:,1]
    assert np.array_equal(a['selected_kinds'],selected)
    assert np.array_equal(a['success'],success) and np.array_equal(a['reward'],success.astype(float))
    assert success.mean()==result[task][mode]['balanced_gathering']
    pub=torch.tensor(np.column_stack([a['inventory'],a['remaining']/16]),dtype=torch.float32)
    for who in range(2):
     rep=agents[who].observe(features[torch.tensor(ids[:,who])],pub)
     inferred_sent=agents[who].send(rep[1]).argmax(-1).numpy()
     inferred_action=agents[who].act(*rep,torch.tensor(a['delivered'][:,1-who])).argmax(-1).numpy()
     assert np.array_equal(inferred_sent,a['sent'][:,who])
     assert np.array_equal(inferred_action,actions[:,who])
    if mode=='normal':assert np.array_equal(a['sent'],a['delivered'])
    if mode=='blank':assert np.all(a['delivered']==0)
    sets[mode]=a
    checks['traces'].append({'task':task,'mode':mode,'n':8192,'success':float(success.mean()),'score_independently_recomputed':True,'all_photos_test_split':True,'checkpoint_actions_and_sent_symbols_match':True})
   for mode in ('shuffle','blank'):
    for key in ('case','remaining','inventory','kinds','image_ids','sent'):
     assert np.array_equal(sets['normal'][key],sets[mode][key])
    assert result[task]['normal']['external_cases_sha256']==result[task][mode]['external_cases_sha256']
   for t in range(1,17):
    mask=sets['shuffle']['remaining']==t
    for who in range(2):
     assert np.array_equal(np.bincount(sets['shuffle']['sent'][mask,who],minlength=5),np.bincount(sets['shuffle']['delivered'][mask,who],minlength=5))
   same=sets['normal']['kinds'][...,0]==sets['normal']['kinds'][...,1]
   for sender in range(2):
    mask=same[:,sender]&~same[:,1-sender]
    checks['direction_gaps'].append({'task':task,'sender':sender,'receiver':1-sender,'n':int(mask.sum()),'normal':float(sets['normal']['success'][mask].mean()),'shuffle':float(sets['shuffle']['success'][mask].mean()),'gap':float(sets['normal']['success'][mask].mean()-sets['shuffle']['success'][mask].mean())})
  checks['paired_external_cases_and_sent_messages']=True
  checks['shuffle_counts_preserved_within_sender_and_public_clock']=True
  seed=result['seed']+900000
  kinds,feat,pub=cases(seed)
  rep=[agents[i].observe(feat[:,i],pub) for i in range(2)]
  sent=[agents[i].send(rep[i][1]).argmax(-1) for i in range(2)]
  probabilities=[agents[i].act(*rep[i],sent[1-i]).softmax(-1).numpy() for i in range(2)]
  acts=np.column_stack([p.argmax(-1) for p in probabilities])
  resources=np.take_along_axis(kinds,acts[...,None],-1)[...,0]
  success=resources[:,0]!=resources[:,1]
  _,ref_feat,ref_pub=cases(seed+33001)
  same=kinds[...,0]==kinds[...,1]
  checks['independent_fixed_observation_intervention']=[]
  for receiver in range(2):
   sender=1-receiver
   saved=next(d for d in result['intervention']['directions'] if d['receiver']==receiver)
   ref_rep=agents[sender].observe(ref_feat[:,sender],ref_pub)
   ref_symbol=agents[sender].send(ref_rep[1]).argmax(-1).numpy()
   counts=np.bincount(ref_symbol,minlength=5)
   assert counts.tolist()==saved['reference_counts']
   used=np.flatnonzero(counts>=40)
   mask=same[:,sender]&~same[:,receiver]
   table=[];foodp=[]
   maxdiff=0.
   for symbol in range(5):
    p=agents[receiver].act(*rep[receiver],torch.full((4096,),symbol,dtype=torch.int64)).softmax(-1).numpy()
    newacts=p.argmax(-1)
    newresources=kinds[np.arange(4096),receiver,newacts]
    newsuccess=newresources!=resources[:,sender]
    table.append(newresources);foodp.append((p*(kinds[:,receiver]==0)).sum(1))
    sub=mask&(sent[sender].numpy()!=symbol)
    saved_effect=next(e for e in saved['interventions'] if e['replacement_symbol']==symbol)
    checkvalues={'resource_flip_rate':(newresources[sub]!=resources[sub,receiver]).mean(),'baseline_success':success[sub].mean(),'intervened_success':newsuccess[sub].mean(),'success_change':(newsuccess[sub].astype(float)-success[sub].astype(float)).mean()}
    for key,val in checkvalues.items():
     delta=abs(float(val)-saved_effect[key]);maxdiff=max(maxdiff,delta);assert delta<1e-7
   rates=float((np.ptp(np.stack(table)[used][:,mask],axis=0)>0).mean())
   assert rates==saved['observed_symbol_sensitivity']['fraction_cases_resource_changes_for_some_used_symbol']
   checks['independent_fixed_observation_intervention'].append({'sender':sender,'receiver':receiver,'n':int(mask.sum()),'used_symbols':used.tolist(),'resource_response_fraction':rates,'max_success_metric_recomputation_difference':maxdiff,'passed':True})
  audits.append(checks)
report={'status':'passed','runs':len(audits),'trace_files':len(audits)*6,'trace_rows':len(audits)*6*8192,'source_snapshot_sha256_verified':True,'disjoint_train_test_image_bytes_verified':True,'scope':'Independent trace scoring and final-checkpoint replay, heldout images, message-condition case matching, within-public-clock symbol permutation, fixed-observation symbol interventions. No training.','audits':audits}
(root/'独立课程对照核查.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps({key:report[key] for key in ('status','runs','trace_files','trace_rows')}))

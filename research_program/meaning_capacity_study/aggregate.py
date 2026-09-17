"""Compact analysis for the eight-meaning capacity-pressure curve."""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
from . import design


def ci(values):
    x=np.asarray(values,dtype=float)
    if len(x)<2: return [float(x.mean()),float(x.mean())] if len(x) else [None,None]
    h=2.13145*float(x.std(ddof=1))/math.sqrt(len(x)); return [float(x.mean()-h),float(x.mean()+h)]

def load(path): return json.loads(Path(path).read_text())

def summary(row):
    final=row['final']; nw=final['new_worker']; new=float(nw['natural_at_train_noise']['team_return_mean']); clean=float(nw['clean_natural']['team_return_mean']); silent=float(nw['silent']['team_return_mean']); incumbent=float(final['incumbent_workers_mean']['natural_at_train_noise']['team_return_mean']); permuted=float(nw['permuted_at_train_noise']['team_return_mean'])
    cross=int(final['task_class_hamming']); within=int(final['within_class_hamming'])
    candidate=bool(new>=0.60 and cross>=2 and within==0)
    return {'seed':int(row['seed']),'condition':row['condition'],'task':row['task'],'form':row['form'],'adaptation':row['adaptation'],'noise_p':float(row['noise_p']),'noise_key':row['noise_key'],'natural':new,'clean':clean,'silent':silent,'incumbent_natural':incumbent,'permuted':permuted,'live_minus_silent':new-silent,'noise_degradation':new-clean,'natural_minus_permuted':new-permuted,'pairwise_min_hamming':int(final['pairwise_min_hamming']),'task_class_hamming':cross,'within_class_hamming':within,'sender_codebook_distance_to_parent':row['sender_codebook_distance_to_parent'],'functional':bool(new>=0.60),'candidate':candidate}

def analyze(parents_path, results_path):
    parents=load(parents_path)['results']; children=load(results_path)['results']; rows=[summary(row) for row in children]
    parent_rows=[{'seed':int(row['seed']),'task':row['task'],'form':row['form'],'natural':float(row['final']['new_worker']['natural_at_train_noise']['team_return_mean']),'pairwise_min_hamming':int(row['final']['pairwise_min_hamming']),'sender_codebook':row['final']['sender_codebook']} for row in parents]
    groups=[]
    for task in design.TASKS:
      for form in design.FORMS:
       for adaptation in design.ADAPTATIONS:
        for noise_key,noise_p in design.NOISE_KEYS.items():
         vals=[r for r in rows if r['task']==task and r['form']==form and r['adaptation']==adaptation and r['noise_key']==noise_key]
         if not vals: continue
         groups.append({'task':task,'form':form,'adaptation':adaptation,'noise_key':noise_key,'noise_p':float(noise_p),'n':len(vals),'natural_mean':float(np.mean([r['natural'] for r in vals])),'natural_ci95_t':ci([r['natural'] for r in vals]),'clean_mean':float(np.mean([r['clean'] for r in vals])),'silent_mean':float(np.mean([r['silent'] for r in vals])),'live_minus_silent_mean':float(np.mean([r['live_minus_silent'] for r in vals])),'noise_degradation_mean':float(np.mean([r['noise_degradation'] for r in vals])),'incumbent_natural_mean':float(np.mean([r['incumbent_natural'] for r in vals])),'natural_minus_permuted_mean':float(np.mean([r['natural_minus_permuted'] for r in vals])),'min_hamming_mean':float(np.mean([r['pairwise_min_hamming'] for r in vals])),'task_class_hamming_mean':float(np.mean([r['task_class_hamming'] for r in vals])),'within_class_hamming_mean':float(np.mean([r['within_class_hamming'] for r in vals])),'functional_count':int(sum(r['functional'] for r in vals)),'candidate_count':int(sum(r['candidate'] for r in vals))})
    adaptation_effects=[]
    for task in design.TASKS:
      for form in design.FORMS:
       for noise_key in design.NOISE_KEYS:
        pairs=[]; incumbent_pairs=[]
        for seed in sorted({r['seed'] for r in rows}):
         co=next((r for r in rows if r['seed']==seed and r['task']==task and r['form']==form and r['noise_key']==noise_key and r['adaptation']=='coadapt'),None); wo=next((r for r in rows if r['seed']==seed and r['task']==task and r['form']==form and r['noise_key']==noise_key and r['adaptation']=='worker_only'),None)
         if co is not None and wo is not None: pairs.append(co['natural']-wo['natural']); incumbent_pairs.append(co['incumbent_natural']-wo['incumbent_natural'])
        if pairs: adaptation_effects.append({'task':task,'form':form,'noise_key':noise_key,'coadapt_minus_worker_only_natural':float(np.mean(pairs)),'ci95_t':ci(pairs),'coadapt_minus_worker_only_incumbent':float(np.mean(incumbent_pairs)),'incumbent_ci95_t':ci(incumbent_pairs)})
    form_effects=[]
    for task in design.TASKS:
      for noise_key in design.NOISE_KEYS:
       for adaptation in design.ADAPTATIONS:
        pairs=[]
        for seed in sorted({r['seed'] for r in rows}):
         q=next((r for r in rows if r['seed']==seed and r['task']==task and r['form']=='quad2' and r['adaptation']==adaptation and r['noise_key']==noise_key),None); at=next((r for r in rows if r['seed']==seed and r['task']==task and r['form']=='atomic16' and r['adaptation']==adaptation and r['noise_key']==noise_key),None)
         if q is not None and at is not None: pairs.append(q['natural']-at['natural'])
        if pairs: form_effects.append({'task':task,'noise_key':noise_key,'adaptation':adaptation,'quad2_minus_atomic16':float(np.mean(pairs)),'ci95_t':ci(pairs)})
    task_effects=[]
    for form in design.FORMS:
      for noise_key in design.NOISE_KEYS:
       for adaptation in design.ADAPTATIONS:
        a=[]; b=[]
        for seed in sorted({r['seed'] for r in rows}):
         unique=next((r for r in rows if r['seed']==seed and r['task']=='unique8' and r['form']==form and r['noise_key']==noise_key and r['adaptation']==adaptation),None); repeat=next((r for r in rows if r['seed']==seed and r['task']=='repeat4' and r['form']==form and r['noise_key']==noise_key and r['adaptation']==adaptation),None); shared=next((r for r in rows if r['seed']==seed and r['task']=='shared8' and r['form']==form and r['noise_key']==noise_key and r['adaptation']==adaptation),None)
         if unique is not None and repeat is not None: a.append(repeat['natural']-unique['natural'])
         if unique is not None and shared is not None: b.append(shared['natural']-unique['natural'])
        if a and b: task_effects.append({'form':form,'noise_key':noise_key,'adaptation':adaptation,'repeat4_minus_unique8':float(np.mean(a)),'repeat4_ci95_t':ci(a),'shared8_minus_unique8':float(np.mean(b)),'shared8_ci95_t':ci(b)})
    return {'schema':'meaning_capacity_analysis_v1','rule':{'functional_natural_min':0.60,'candidate':'natural >= 0.60; cross-class/minimum Hamming distance >= 2; shared8 requires within-parity distance = 0'},'parents':parent_rows,'rows':rows,'groups':groups,'adaptation_effects':adaptation_effects,'form_effects':form_effects,'task_effects':task_effects}

def write_md(path,data):
    groups={(r['task'],r['form'],r['adaptation'],r['noise_key']):r for r in data['groups']}; lines=['# Meaning-capacity repetition curve','','`unique8` uses eight distinct balanced meanings, `repeat4` repeats four meanings twice each, and `shared8` repeats one three-bit parity across eight stages. `quad2` and `atomic16` both expose sixteen raw message states; `triple2` exposes eight.','', '| task | form | adaptation | noise | natural | clean | silent | live−silent | class distance | within-class | functional | candidate |','|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for task in design.TASKS:
      for form in design.FORMS:
       for adaptation in design.ADAPTATIONS:
        for noise_key in design.NOISE_KEYS:
         row=groups.get((task,form,adaptation,noise_key))
         if row: lines.append(f"| `{task}` | `{form}` | `{adaptation}` | {row['noise_p']:.2f} | {row['natural_mean']:.3f} | {row['clean_mean']:.3f} | {row['silent_mean']:.3f} | {row['live_minus_silent_mean']:.3f} | {row['task_class_hamming_mean']:.2f} | {row['within_class_hamming_mean']:.2f} | {row['functional_count']}/{row['n']} | {row['candidate_count']}/{row['n']} |")
    lines += ['', '| task | noise | adaptation | quad2−atomic16 natural |','|---|---:|---|---:|']
    for row in data['form_effects']: lines.append(f"| `{row['task']}` | {design.NOISE_KEYS[row['noise_key']]:.2f} | `{row['adaptation']}` | {row['quad2_minus_atomic16']:.3f} [{row['ci95_t'][0]:.3f},{row['ci95_t'][1]:.3f}] |")
    lines += ['', '| form | noise | adaptation | repeat4−unique8 | shared8−unique8 |','|---|---:|---|---:|---:|']
    for row in data['task_effects']: lines.append(f"| `{row['form']}` | {design.NOISE_KEYS[row['noise_key']]:.2f} | `{row['adaptation']}` | {row['repeat4_minus_unique8']:.3f} [{row['repeat4_ci95_t'][0]:.3f},{row['repeat4_ci95_t'][1]:.3f}] | {row['shared8_minus_unique8']:.3f} [{row['shared8_ci95_t'][0]:.3f},{row['shared8_ci95_t'][1]:.3f}] |")
    lines += ['', 'This matrix tests whether the low-entropy `shared4` result survives an eight-class meaning space. It remains a mechanism study of discrete protocols, not a claim of natural language.']; Path(path).write_text('\n'.join(lines)+'\n')

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--parents',required=True); ap.add_argument('--results',required=True); ap.add_argument('--out',required=True); ap.add_argument('--markdown',required=True); args=ap.parse_args(); data=analyze(args.parents,args.results); Path(args.out).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n'); write_md(args.markdown,data); print(json.dumps({'status':'written','parent_rows':len(data['parents']),'rows':len(data['rows']),'groups':len(data['groups'])},ensure_ascii=False))

"""Compact analysis for the repetition-pressure curve."""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
from . import design

def ci(values):
    x=np.asarray(values,dtype=float)
    if len(x)<2: return [float(x.mean()),float(x.mean())] if len(x) else [None,None]
    h=2.13145*float(x.std(ddof=1))/math.sqrt(len(x))
    return [float(x.mean()-h),float(x.mean()+h)]

def load(path): return json.loads(Path(path).read_text())

def summary(row):
    final=row['final']; new=float(final['new_worker']['natural_at_train_noise']['team_return_mean']); clean=float(final['new_worker']['clean_natural']['team_return_mean']); silent=float(final['new_worker']['silent']['team_return_mean']); incumbent=float(final['incumbent_workers_mean']['natural_at_train_noise']['team_return_mean']); permuted=float(final['new_worker']['permuted_at_train_noise']['team_return_mean'])
    return {'seed':int(row['seed']),'condition':row['condition'],'task':row['task'],'form':row['form'],'adaptation':row['adaptation'],'noise_p':float(row['noise_p']),'noise_key':row['noise_key'],'natural':new,'clean':clean,'silent':silent,'incumbent_natural':incumbent,'permuted':permuted,'live_minus_silent':new-silent,'noise_degradation':new-clean,'natural_minus_permuted':new-permuted,'pairwise_min_hamming':int(final['pairwise_min_hamming']),'task_class_hamming':int(final['task_class_hamming']),'within_class_hamming':int(final['within_class_hamming']),'sender_codebook_distance_to_parent':row['sender_codebook_distance_to_parent'],'functional':bool(new>=0.60),'error_correcting_candidate':bool(row['form']=='triple2' and new>=0.60 and final['task_class_hamming']>=2 and final['within_class_hamming']==0)}

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
         groups.append({'task':task,'form':form,'adaptation':adaptation,'noise_key':noise_key,'noise_p':float(noise_p),'n':len(vals),'natural_mean':float(np.mean([r['natural'] for r in vals])),'natural_ci95_t':ci([r['natural'] for r in vals]),'clean_mean':float(np.mean([r['clean'] for r in vals])),'silent_mean':float(np.mean([r['silent'] for r in vals])),'live_minus_silent_mean':float(np.mean([r['live_minus_silent'] for r in vals])),'noise_degradation_mean':float(np.mean([r['noise_degradation'] for r in vals])),'incumbent_natural_mean':float(np.mean([r['incumbent_natural'] for r in vals])),'natural_minus_permuted_mean':float(np.mean([r['natural_minus_permuted'] for r in vals])),'min_hamming_mean':float(np.mean([r['pairwise_min_hamming'] for r in vals])),'task_class_hamming_mean':float(np.mean([r['task_class_hamming'] for r in vals])),'within_class_hamming_mean':float(np.mean([r['within_class_hamming'] for r in vals])),'functional_count':int(sum(r['functional'] for r in vals)),'error_correcting_candidate_count':int(sum(r['error_correcting_candidate'] for r in vals))})
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
         tr=next((r for r in rows if r['seed']==seed and r['task']==task and r['form']=='triple2' and r['adaptation']==adaptation and r['noise_key']==noise_key),None); at=next((r for r in rows if r['seed']==seed and r['task']==task and r['form']=='atomic8' and r['adaptation']==adaptation and r['noise_key']==noise_key),None)
         if tr is not None and at is not None: pairs.append(tr['natural']-at['natural'])
        if pairs: form_effects.append({'task':task,'noise_key':noise_key,'adaptation':adaptation,'triple2_minus_atomic8':float(np.mean(pairs)),'ci95_t':ci(pairs)})
    task_effects=[]
    for form in design.FORMS:
      for noise_key in design.NOISE_KEYS:
       for adaptation in design.ADAPTATIONS:
        repeat_minus_unique=[]; shared_minus_unique=[]
        for seed in sorted({r['seed'] for r in rows}):
         unique=next((r for r in rows if r['seed']==seed and r['task']=='unique4' and r['form']==form and r['noise_key']==noise_key and r['adaptation']==adaptation),None)
         repeat=next((r for r in rows if r['seed']==seed and r['task']=='repeat2' and r['form']==form and r['noise_key']==noise_key and r['adaptation']==adaptation),None)
         shared=next((r for r in rows if r['seed']==seed and r['task']=='shared4' and r['form']==form and r['noise_key']==noise_key and r['adaptation']==adaptation),None)
         if unique is not None and repeat is not None: repeat_minus_unique.append(repeat['natural']-unique['natural'])
         if unique is not None and shared is not None: shared_minus_unique.append(shared['natural']-unique['natural'])
        if repeat_minus_unique and shared_minus_unique:
         task_effects.append({'form':form,'noise_key':noise_key,'adaptation':adaptation,'repeat2_minus_unique4':float(np.mean(repeat_minus_unique)),'repeat2_ci95_t':ci(repeat_minus_unique),'shared4_minus_unique4':float(np.mean(shared_minus_unique)),'shared4_ci95_t':ci(shared_minus_unique)})
    return {'schema':'repetition_pressure_analysis_v1','rule':{'functional_natural_min':0.60,'error_correcting_candidate':'triple2 natural >= 0.60; shared4 requires cross-parity Hamming distance >= 2 and within-parity distance = 0; unique4/repeat2 require full codebook minimum distance >= 2'},'parents':parent_rows,'rows':rows,'groups':groups,'adaptation_effects':adaptation_effects,'form_effects':form_effects,'task_effects':task_effects}

def write_md(path,data):
    groups={(r['task'],r['form'],r['adaptation'],r['noise_key']):r for r in data['groups']}
    lines=['# Repetition-pressure curve','','`unique4` exposes four distinct Boolean meanings once each, `repeat2` repeats two independent meanings twice each, and `shared4` repeats one hidden parity in all four action stages. `triple2` and `atomic8` each have eight raw message states; `dual2` has four.','', '| task | form | adaptation | noise | natural | clean | silent | live−silent | code distance | within-class | functional | candidate |','|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for task in design.TASKS:
      for form in design.FORMS:
       for adaptation in design.ADAPTATIONS:
        for noise_key in design.NOISE_KEYS:
         row=groups.get((task,form,adaptation,noise_key))
         if row is None: continue
         lines.append(f"| `{task}` | `{form}` | `{adaptation}` | {row['noise_p']:.2f} | {row['natural_mean']:.3f} | {row['clean_mean']:.3f} | {row['silent_mean']:.3f} | {row['live_minus_silent_mean']:.3f} | {row['task_class_hamming_mean']:.2f} | {row['within_class_hamming_mean']:.2f} | {row['functional_count']}/{row['n']} | {row['error_correcting_candidate_count']}/{row['n']} |")
    lines += ['', '| task | noise | adaptation | triple2−atomic8 natural |','|---|---:|---|---:|']
    for row in data['form_effects']: lines.append(f"| `{row['task']}` | {design.NOISE_KEYS[row['noise_key']]:.2f} | `{row['adaptation']}` | {row['triple2_minus_atomic8']:.3f} [{row['ci95_t'][0]:.3f},{row['ci95_t'][1]:.3f}] |")
    lines += ['', '| task | form | noise | coadapt−worker_only new | coadapt−worker_only incumbent |','|---|---|---:|---:|---:|']
    for row in data['adaptation_effects']: lines.append(f"| `{row['task']}` | `{row['form']}` | {design.NOISE_KEYS[row['noise_key']]:.2f} | {row['coadapt_minus_worker_only_natural']:.3f} [{row['ci95_t'][0]:.3f},{row['ci95_t'][1]:.3f}] | {row['coadapt_minus_worker_only_incumbent']:.3f} [{row['incumbent_ci95_t'][0]:.3f},{row['incumbent_ci95_t'][1]:.3f}] |")
    lines += ['', '| form | noise | adaptation | repeat2−unique4 | shared4−unique4 |','|---|---:|---|---:|---:|']
    for row in data['task_effects']: lines.append(f"| `{row['form']}` | {design.NOISE_KEYS[row['noise_key']]:.2f} | `{row['adaptation']}` | {row['repeat2_minus_unique4']:.3f} [{row['repeat2_ci95_t'][0]:.3f},{row['repeat2_ci95_t'][1]:.3f}] | {row['shared4_minus_unique4']:.3f} [{row['shared4_ci95_t'][0]:.3f},{row['shared4_ci95_t'][1]:.3f}] |")
    lines += ['', 'The analysis separates task repetition, code-space form, maintenance (`worker_only`) and renegotiation (`coadapt`). It is a mechanism test, not a claim that any learned code is natural language.']
    Path(path).write_text('\n'.join(lines)+'\n')

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--parents',required=True); ap.add_argument('--results',required=True); ap.add_argument('--out',required=True); ap.add_argument('--markdown',required=True); args=ap.parse_args(); data=analyze(args.parents,args.results); Path(args.out).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n'); write_md(args.markdown,data); print(json.dumps({'status':'written','parent_rows':len(data['parents']),'rows':len(data['rows']),'groups':len(data['groups'])},ensure_ascii=False))

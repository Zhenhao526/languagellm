"""Plot completed aggregate JSON only: no model, NPZ, or task-kernel imports."""
import argparse
from datetime import datetime,timezone
import hashlib,json,platform
from pathlib import Path
import numpy as np

SEEDS=(59101,59102,59103,59104)
CONDITIONS=('PL_live','PL_silent')
PARTS=('train','new_needs','new_layouts','new_needs_and_layouts')
TARGET='new_needs_and_layouts'
STEPS=(0,100,500,1500,3000,6000)
WORLD_COUNTS=dict(zip(PARTS,(419904,160704,139968,53568)))
MONITOR_COUNTS=dict(zip(PARTS,(7776,2976,7776,2976)))
AXES=('kind','length','destination')
MODES=('silent','live','closed')
METRICS=('Q','full_success_rate','executed_partner_correct_rate')
SEED_COLORS=('#256F91','#B15B39','#68854D','#8E609A')
MARKERS=('o','s','^','D')
MODE_COLORS={'live':'#226F9B','silent':'#C16843','closed':'#65656D'}


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')


def rate(value,name):
    require(type(value) in (int,float) and np.isfinite(value) and 0<=value<=1,f'Invalid probability: {name}')
    return float(value)


def extract(result):
    """Keep all four seeds, partitions, and saved checkpoints; no selection."""
    require(result.get('status')=='completed','Only completed results can be plotted')
    digest=result.get('plan_sha256','')
    require(len(digest)==64 and all(c in '0123456789abcdef' for c in digest),'Missing frozen plan hash')
    runs=result['runs'];require(len(runs)==8,'Expected exactly eight policy fits')
    by={(r['seed'],r['condition']):r for r in runs}
    require(len(by)==8 and set(by)=={(s,c) for s in SEEDS for c in CONDITIONS},'Incomplete/duplicate run grid')
    budget=result['budget']
    for key,value in dict(runs=8,training_updates=48000,checkpoints=48,natural_monitor_files=192,
        natural_final_files=32,closed_final_files=16,aliased_evaluations=0).items():
        require(budget[key]==value,f'Unexpected budget: {key}')
    rows=[]
    for seed in SEEDS:
        for condition in CONDITIONS:
            r=by[seed,condition];live=condition=='PL_live'
            require(r['rule']=='reciprocal' and r['updates']==6000,'Wrong settlement or training endpoint')
            require(set(r['final'])==set(PARTS),'Missing full final partition')
            final={}
            for part in PARTS:
                modes=r['final'][part]
                require(set(modes)==({'natural','closed'} if live else {'natural'}),'Unexpected final modes or alias')
                final[part]={}
                for mode,record in modes.items():
                    require(record['worlds']==WORLD_COUNTS[part] and record['information']=='PL','Wrong full domain/information')
                    require(record['rule']=='reciprocal' and record['mode']==mode,'Wrong final rule/mode')
                    require(record['live']==(live and mode=='natural'),'Wrong final routing')
                    require(record['intervention']==('close_cross_agent_channel_from_window_1' if mode=='closed' else None),'Wrong closure definition')
                    response=record['need_response']
                    require(response['complete_nine_strata'] is True and response['empty_strata']==[], 'Incomplete nine-stratum Q')
                    require(response['worlds']==WORLD_COUNTS[part] and response['partition']==part,'Q domain mismatch')
                    strata=response['strata'];lookup={(s['changed_person'],s['axis']):s for s in strata}
                    require(len(strata)==9 and len(lookup)==9 and set(lookup)=={(a,x) for a in range(3) for x in AXES},'Wrong Q strata')
                    layer_q=[rate(lookup[a,x]['Q'],'stratum Q') for a in range(3) for x in AXES]
                    q=rate(response['Q'],'Q');require(abs(q-float(np.mean(layer_q)))<1e-12,'Q is not equal nine-stratum mean')
                    q_shuffle=rate(response['Q_shuffle'],'Q_shuffle')
                    require(abs(response['Q_excess']-(q-q_shuffle))<1e-12,'Q excess mismatch')
                    final[part][mode]=dict(Q=q,Q_shuffle=q_shuffle,Q_excess=response['Q_excess'],
                        Q_strata=layer_q,worlds=record['worlds'],
                        full_success_rate=rate(record['full_success_rate'],'full_success_rate'),
                        executed_partner_correct_rate=rate(record['executed_partner_correct_rate'],'executed_partner_correct_rate'),
                        reward_mean=rate(record['reward_mean'],'reward_mean'))
            monitor=r['monitor'];require([m['update'] for m in monitor]==list(STEPS),'Wrong or missing checkpoints')
            collected={p:{m:[] for m in METRICS[1:]} for p in PARTS}
            for checkpoint in monitor:
                require(set(checkpoint['monitor'])==set(PARTS),'Missing monitor partition')
                for part,record in checkpoint['monitor'].items():
                    require(record['worlds']==MONITOR_COUNTS[part] and record['information']=='PL','Wrong monitor support')
                    require(record['rule']=='reciprocal' and record['mode']=='natural' and record['live']==live,'Wrong monitor condition')
                    require('need_response' not in record,'Q is not defined for these monitor subsets')
                    for metric in METRICS[1:]:collected[part][metric].append(rate(record[metric],metric))
            rows.append(dict(seed=seed,condition=condition,final=final,monitor=collected))
    rowmap={(r['seed'],r['condition']):r for r in rows};paired=[]
    for seed in SEEDS:
        cells={'silent':rowmap[seed,'PL_silent']['final'][TARGET]['natural'],
               'live':rowmap[seed,'PL_live']['final'][TARGET]['natural'],
               'closed':rowmap[seed,'PL_live']['final'][TARGET]['closed']}
        difference=cells['live']['Q']-cells['silent']['Q']
        layers=np.asarray(cells['live']['Q_strata'])-np.asarray(cells['silent']['Q_strata'])
        require(abs(float(layers.mean())-difference)<1e-12,'Q layer difference mismatch')
        paired.append(dict(seed=seed,cells=cells,Q_difference=difference,Q_strata_difference=layers.tolist()))
    mean=float(np.mean([p['Q_difference'] for p in paired]));stored=result['primary']
    require(stored['metric']=='Q' and stored['partition']==TARGET and stored['contrast']=='PL_live_minus_PL_silent','Wrong primary estimand')
    require(stored['independent_paired_seed_blocks']==4 and abs(stored['mean_difference']-mean)<1e-12,'Primary mean mismatch')
    saved={v['seed']:v for v in stored['paired_seeds']}
    require(len(stored['paired_seeds'])==4 and set(saved)==set(SEEDS),'Primary seed mismatch')
    for row in paired:
        check=saved[row['seed']]
        for key,value in [('Q_live',row['cells']['live']['Q']),('Q_silent',row['cells']['silent']['Q']),('difference',row['Q_difference'])]:
            require(abs(check[key]-value)<1e-12,f'Primary paired mismatch: {key}')
    return dict(schema='private_partner_figure_data_v1',plan_sha256=digest,rows=rows,
        primary=dict(metric='Q',partition=TARGET,mean_difference=mean,paired_seeds=paired),
        independent_paired_seeds=4,actual_policy_fits=8,closed_is_same_live_policy=True,
        full_worlds=WORLD_COUNTS,monitor_worlds=MONITOR_COUNTS,checkpoints=list(STEPS),
        scope='All aggregate data retained; displayed domain is the complete double holdout. Monitor Q is absent.')


def draw(data,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
        'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'savefig.dpi':180})
    out=Path(out);outputs={}
    def save(fig,stem):
        for suffix in ('.png','.pdf'):
            path=out/(stem+suffix);require(not path.exists(),'Never overwrite figures')
            fig.savefig(path,bbox_inches='tight',facecolor='white');outputs[path.name]={'sha256':sha(path),'bytes':path.stat().st_size}
        plt.close(fig)
    paired=data['primary']['paired_seeds'];rows={(r['seed'],r['condition']):r for r in data['rows']}
    fig,axes=plt.subplots(2,2,figsize=(15.8,11.4),gridspec_kw={'wspace':.28,'hspace':.43})
    for ax,metric,title in [(axes[0,0],'Q','Correct partner response at both endpoints (Q)'),
        (axes[1,0],'full_success_rate','Full task success'),
        (axes[1,1],'executed_partner_correct_rate','Correct actual executing pair')]:
        for row,color,marker,offset in zip(paired,SEED_COLORS,MARKERS,(np.arange(4)-1.5)*.045):
            values=[100*row['cells'][mode][metric] for mode in MODES]
            x=np.arange(3)+offset
            ax.plot(x[:2],values[:2],color=color,lw=1.8,marker=marker,markersize=6,label=f'Seed {row["seed"]}')
            ax.plot(x[1:],values[1:],color=color,lw=1.5,ls=':',marker=marker,markersize=6)
        ax.set_xticks(range(3),['Silent\ntrained policy','Live\ntrained policy','Live policy\nchannel closed'])
        ax.set_xlim(-.22,2.22);ax.set_ylim(0,100);ax.set_yticks(np.arange(0,101,20));ax.grid(axis='y',alpha=.17)
        ax.set_ylabel('Rate (%)');ax.set_title(title,fontsize=12,pad=14)
    axes[0,0].legend(loc='best',framealpha=.93,fontsize=9)
    matrix=np.asarray([p['Q_strata_difference'] for p in paired])*100
    matrix=np.vstack((matrix,matrix.mean(axis=0)));bound=max(float(np.abs(matrix).max()),1.)
    ax=axes[0,1];im=ax.imshow(matrix,cmap='RdBu',vmin=-bound,vmax=bound,aspect='auto')
    for i in range(5):
        for j in range(9):ax.text(j,i,f'{matrix[i,j]:+.1f}',ha='center',va='center',fontsize=8,
            color='white' if abs(matrix[i,j])>.64*bound else '#222222')
    ax.set_xticks(range(9),[f'{"ABC"[a]}\n{axis}' for a in range(3) for axis in ('kind','length','dest')],fontsize=8)
    ax.set_yticks(range(5),[str(s) for s in SEEDS]+['Mean'])
    ax.set_title('Nine strata: Live minus Silent Q (pp)',fontsize=12,pad=14)
    ax.set_xlabel('Changed person and changed need axis',fontsize=9)
    colorbar=fig.colorbar(im,ax=ax,fraction=.035,pad=.02);colorbar.set_label('Percentage points',fontsize=9)
    mean=data['primary']['mean_difference']*100
    fig.suptitle('Private needs and communication: complete final evaluation',fontsize=16,y=.98)
    fig.text(.5,.937,f'Primary paired Q difference: {mean:+.2f} percentage points, mean of all four fresh seed blocks.',ha='center',fontsize=11)
    fig.subplots_adjust(top=.855,bottom=.17)
    fig.text(.5,.101,'All 53,568 double-holdout worlds; fixed update 6000. Q averages the nine person-by-axis strata equally.',ha='center',fontsize=10)
    fig.text(.5,.066,'Dotted links: close the same live policy from the first message window and rerun both windows; this is not another trained seed.',ha='center',fontsize=10)
    fig.text(.5,.031,'Four independent paired seeds, eight policy fits. Small horizontal offsets reveal coincident points; no old FI results are included.',ha='center',fontsize=9.5,color='#555555')
    save(fig,'01_full_partner_response')

    fig=plt.figure(figsize=(17.5,8.7));grid=fig.add_gridspec(2,4,wspace=.3,hspace=.36)
    for col,seed in enumerate(SEEDS):
        for row,metric in enumerate(METRICS[1:]):
            inner=grid[row,col].subgridspec(1,2,width_ratios=(4,1.15),wspace=.14)
            monitor_ax=fig.add_subplot(inner[0,0]);final_ax=fig.add_subplot(inner[0,1],sharey=monitor_ax)
            for condition,mode,marker in [('PL_live','live','o'),('PL_silent','silent','s')]:
                values=np.asarray(rows[seed,condition]['monitor'][TARGET][metric])*100
                monitor_ax.plot(STEPS,values,color=MODE_COLORS[mode],marker=marker,lw=1.7,markersize=3.5)
            for x,mode,marker in zip(range(3),MODES,('s','o','D')):
                cell=next(p for p in paired if p['seed']==seed)['cells'][mode]
                final_ax.scatter(x,100*cell[metric],color=MODE_COLORS[mode],marker=marker,s=31,zorder=3)
            for ax in (monitor_ax,final_ax):ax.set_ylim(0,100);ax.set_yticks(np.arange(0,101,20));ax.grid(axis='y',alpha=.15)
            monitor_ax.set_xlim(0,6000);monitor_ax.set_xticks([0,1500,3000,6000]);monitor_ax.tick_params(axis='x',labelsize=8)
            final_ax.set_xlim(-.5,2.5);final_ax.set_xticks(range(3),['S','L','C']);final_ax.tick_params(axis='both',labelsize=8)
            final_ax.tick_params(axis='y',labelleft=False,left=False);final_ax.spines['left'].set_visible(False)
            final_ax.set_facecolor('#F3F4F5')
            if row==0:
                monitor_ax.set_title(f'Seed {seed}',fontsize=11,pad=12);final_ax.set_title('Full\nendpoint',fontsize=8,pad=6)
            if col==0:monitor_ax.set_ylabel('Full success (%)' if metric=='full_success_rate' else 'Correct executing pair (%)')
            else:monitor_ax.tick_params(axis='y',labelleft=False)
            if row==1:monitor_ax.set_xlabel('Optimization update',fontsize=9);final_ax.set_xlabel('Full',fontsize=9)
    fig.suptitle('Natural monitoring and separate complete endpoints',fontsize=16,y=.988)
    fig.legend(handles=[Line2D([0],[0],color=MODE_COLORS['live'],marker='o',label='L: Live trained policy'),
        Line2D([0],[0],color=MODE_COLORS['silent'],marker='s',label='S: Silent trained policy'),
        Line2D([0],[0],color=MODE_COLORS['closed'],marker='D',ls='None',label='C: Same live policy, channel closed (full only)')],
        loc='upper center',bbox_to_anchor=(.5,.95),ncol=3,frameon=False,fontsize=10)
    fig.subplots_adjust(top=.82,bottom=.195)
    fig.text(.5,.112,'Six natural checkpoints: 0, 100, 500, 1500, 3000, 6000. Lines use 2,976 worlds at two fixed double-holdout backgrounds.',ha='center',fontsize=10)
    fig.text(.5,.073,'Shaded endpoint axes use all 53,568 double-holdout worlds and have no connecting lines to the monitor. Q is evaluated only at full endpoints.',ha='center',fontsize=10)
    fig.text(.5,.034,'All policies use reciprocal execution. Correct executing pair is an actual-execution measure, distinct from all three proposal roles.',ha='center',fontsize=9.5,color='#555555')
    save(fig,'02_monitor_and_complete_endpoint')
    return outputs


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',required=True);parser.add_argument('--out')
    args=parser.parse_args();run=Path(args.run).resolve();path=run/'execution/results.json'
    require(path.is_file(),'Completed aggregate result missing');digest=sha(path)
    with path.open(encoding='utf-8') as stream:data=extract(json.load(stream))
    out=Path(args.out).resolve() if args.out else run/'figures_001'
    require(not out.exists(),'Never overwrite a figure directory');out.mkdir(parents=True)
    write(out/'figure_data.json',data)
    try:
        outputs=draw(data,out);require(sha(path)==digest,'Aggregate result changed during plotting')
        import matplotlib
        receipt=dict(status='generated_visual_inspection_pending',at=datetime.now(timezone.utc).isoformat(),
            source_result_path=str(path),source_result_sha256=digest,plot_source_sha256=sha(__file__),
            plan_sha256=data['plan_sha256'],figure_data_sha256=sha(out/'figure_data.json'),outputs=outputs,
            runtime=dict(python=platform.python_version(),numpy=np.__version__,matplotlib=matplotlib.__version__),
            independent_paired_seed_blocks=4,policy_fits=8,closed_channel_additional_policy_fits=0,
            additional_model_forward_samples=0,training_updates=0,visual_inspection_performed=False,
            read_scope='Only completed execution/results.json; no NPZ, checkpoint, training log, model or kernel import.')
        write(out/'figure_receipt.json',receipt)
        print(json.dumps(dict(status=receipt['status'],out=str(out),outputs=list(outputs)),ensure_ascii=False))
    except BaseException as error:
        write(out/'failure.json',dict(status='failed',at=datetime.now(timezone.utc).isoformat(),
            error_type=type(error).__name__,error=str(error),source_result_sha256=digest,
            plot_source_sha256=sha(__file__),automatic_retry=False));raise


if __name__=='__main__':main()

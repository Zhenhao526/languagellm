"""Human-facing schematic, never used as an agent observation or training input."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'redesign_v0.9/.analysis_deps'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def main():
    out=ROOT/'task_figure';out.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,3,figsize=(12.8,4.6))
    locations=[(.7,1.7),(1.9,1.7),(3.1,1.7),(.7,.5),(1.9,.5),(3.1,.5)]
    for ax,title,food,water,partial in zip(axes,
        ['First observation: complete scene','Second: immediate condition','Second: delayed condition'],
        [0,0,None],[3,4,4],[False,False,True]):
        for i,(x,y) in enumerate(locations):
            color='#eabf75' if i==food else '#8fc4df' if i==water else '#f1f2f4'
            ax.add_patch(Circle((x,y),.38,facecolor=color,edgecolor='#8b9399',lw=1.2))
            label='Food' if i==food else 'Water' if i==water else '?' if partial else ''
            ax.text(x,y,label,ha='center',va='center',fontsize=11,color='#252d33')
            ax.text(x,y-.56,str(i+1),ha='center',va='center',fontsize=10,color='#535c63')
        ax.set_title(title,fontsize=12,pad=12);ax.set(xlim=(0,3.8),ylim=(-.35,2.3));ax.set_aspect('equal');ax.axis('off')
    fig.suptitle('One resource moves; the other stays in place',fontsize=17,y=.97)
    fig.text(.5,.19,'Same event for both conditions: water moves from site 4 to site 5.',ha='center',fontsize=12)
    fig.text(.5,.12,'In the delayed observation, food is hidden. Its site must be recovered from the earlier scene.',ha='center',fontsize=11)
    fig.text(.5,.055,'Schematic only. Experimental inputs are visual feature slots; full / detach receive identical observations.',ha='center',fontsize=10,color='#424b53')
    fig.subplots_adjust(left=.025,right=.975,top=.78,bottom=.29,wspace=.15)
    for ext in ('png','pdf'):fig.savefig(out/f'00_temporal_task.{ext}',dpi=180,facecolor='white')
    plt.close(fig)

if __name__=='__main__':main()

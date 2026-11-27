"""Visualize rounded upstream aggregates; never imply raw validation or new runs."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
D=Path(__file__).resolve().parent;x=json.loads((D/'hold-sweep-reported.json').read_text());names=x['mixtures'];rows={r['setting']:r for r in x['rows']}
plt.rcParams.update({'font.size':13,'axes.titlesize':14,'axes.labelsize':13,'pdf.fonttype':42})
def heat(settings,name,figsize):
 data=np.array([[rows[s][n] for n in names] for s in settings]);fig,ax=plt.subplots(figsize=figsize)
 im=ax.imshow(data,cmap='YlGnBu',vmin=0,vmax=50,aspect='auto');ax.set_xticks(range(5),[n.replace('-','\n') for n in names]);ax.set_yticks(range(len(settings)),[s.upper().replace('-H',' h').replace('-B',' b').replace('-W',' w').replace('-T',' t') for s in settings]);ax.tick_params(length=0)
 for i in range(len(settings)):
  for j in range(5):ax.text(j,i,f'{data[i,j]:.1f}',ha='center',va='center',color='white' if data[i,j]>29 else '#152C42',fontsize=12)
 fig.colorbar(im,ax=ax,pad=.025,label='Cached prompt tokens (%)');ax.set_title('Reported larger sweep · seed 0 · rounded table',pad=12)
 fig.tight_layout();fig.savefig(D/(name+'.pdf'),bbox_inches='tight');fig.savefig(D/(name+'.png'),dpi=200,bbox_inches='tight');plt.close(fig)
heat(['lru','lfu','tfifo','hlru-h8-b1','hlru-h30-b1','rfifo-w100-t0.02','rfifo-w500-t0.05'],'hold_selected',(10.8,4.2))
heat(list(rows),'hold_full',(11.2,12.5))
settings=['lru','lfu','hlru-h30-b1','rfifo-w100-t0.02','rfifo-w500-t0.05'];ranks={r['setting']:r for r in x['derived_from_rounded_table']};fig,axes=plt.subplots(1,3,figsize=(12,4.2))
for ax,key,title in zip(axes,['macro_mean_percent','mean_regret_pp','worst_regret_pp'],['Macro mean (%) ↑','Mean regret (pp) ↓','Worst regret (pp) ↓']):
 vals=[ranks[s][key] for s in settings];ax.barh(range(len(settings)),vals,color=['#167dad','#a55b19','#15977b','#8b6eaf','#72829b']);ax.set_yticks(range(len(settings)),[s.upper() for s in settings] if ax is axes[0] else []);ax.invert_yaxis();ax.set_xlim(0,max(vals)*1.24);ax.set_title(title)
 for i,v in enumerate(vals):ax.text(v+.2,i,f'{v:.1f}',va='center',fontsize=12)
 ax.spines[['top','right']].set_visible(False)
fig.suptitle('Five mixtures equally weighted · regret to best tested setting · rounded inputs',fontsize=13);fig.tight_layout();fig.savefig(D/'hold_robustness.pdf',bbox_inches='tight');fig.savefig(D/'hold_robustness.png',dpi=200,bbox_inches='tight');plt.close(fig)
print('Generated 3 reported-aggregate figures; no experiments executed.')

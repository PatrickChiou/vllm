"""Source-based vector diagrams, independent of benchmark execution."""
from pathlib import Path
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,Rectangle
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'pdf.fonttype':42,'svg.fonttype':'none'})
def box(ax,x,y,w,h,text,fc='#E8F2F8',size=11):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.04',fc=fc,ec='#167DAD',lw=1));ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=size)
def arrow(ax,x,y,xx,yy):ax.annotate('',xy=(xx,yy),xytext=(x,y),arrowprops={'arrowstyle':'->','color':'#167DAD','lw':1.3})
def save(fig,name):
 fig.tight_layout()
 for ext in ['pdf','svg','png']:fig.savefig(a.output/(name+'.'+ext),bbox_inches='tight',dpi=320)
 plt.close(fig)
fig,ax=plt.subplots(figsize=(7.2,2.8));ax.set_xlim(0,12);ax.set_ylim(0,3.3);ax.axis('off')
for x,t in [(0.05,'HTTP API +\nclient replay'),(2.5,'v0 engine /\nscheduler'),(5.05,'Virtual blocks /\nevictor')]:box(ax,x,2.1,2.15,.7,t,size=10)
arrow(ax,2.2,2.45,2.45,2.45);arrow(ax,4.65,2.45,5,2.45)
box(ax,8,2.4,3.6,.62,'Original: GPU model /\nreal KV tensors','#F1EDE8',10);box(ax,8,1.45,3.6,.62,'Simulator: trace tokens /\nno model weights','#DEF3E8',10);arrow(ax,7.22,2.45,7.9,2.7);arrow(ax,7.22,2.4,7.9,1.76)
ax.text(.05,1.5,'Identical leading prefix; different suffix',fontsize=10,color='#167DAD')
for row,label in enumerate(['Request A','Request B']):
 y=.75-row*.62;ax.text(.05,y+.16,label,fontsize=10)
 for j,word in enumerate(['P₀','P₁', 'A₂' if row==0 else 'B₂']):
  x=1.55+j*1.05;box(ax,x,y,.9,.4,word,'#D1E8F3' if j<2 else '#F6DCBF',10)
 ax.text(5,y+.1,'P₁ chains P₀',fontsize=10)
ax.text(8.05,.65,'Retained: batching, refcounts,\nprefix lookup and completion.',fontsize=10)
save(fig,'system_prefix')
fig,ax=plt.subplots(figsize=(7.2,2.7));ax.set_xlim(0,12);ax.set_ylim(0,3);ax.axis('off')
texts=['Normalize / filter','Template /\ntokenize','Assign arrivals','Causal HTTP\nreplay','Schedule /\nprefix lookup','Allocate / evict','Trace prefill +\ndecode','Validate / export']
for i,t in enumerate(texts):
 x=.1+i%4*3;y=1.95-i//4*1.55;box(ax,x,y,2.6,.75,t,size=10)
 if i%4<3:arrow(ax,x+2.62,y+.37,x+2.95,y+.37)
arrow(ax,10.35,1.92,1.4,1.2);save(fig,'workflow')

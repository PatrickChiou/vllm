"""Validate existing completed experiments and write new figures/tables only.
Usage: python tools/comp536_final_figures.py --m2 DIR --mixed DIR --task1 DIR --output NEW_DIR
No model, server or benchmark is launched. Large raw data remain at their source.
"""
import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from comp536_prompt_recurrence import analyze as prompt_recurrence
from matplotlib.patches import FancyBboxPatch,Rectangle

COLORS={'lru':'#167DAD','lfu':'#DE8A29','fifo':'#7256A8','tlru':'#15977B',
        'tfifo':'#C84C51','hlru':'#64748B','hfifo':'#D87AA0','rfifo':'#699330'}
POLICIES=list(COLORS)
LABELS=['chat','agent','coding-agent','quality','bailian-chat','bailian-coder','mooncake-agent','code']

def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def csvout(p,rows):
 if not rows:return
 fields=list(dict.fromkeys(k for r in rows for k in r))
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def readcsv(p):return list(csv.DictReader(p.open()))
def quant(xs,q):return float(np.percentile(xs,q)) if xs else None

def validate(path):
 r=json.loads((path/'summary.json').read_text());q=[json.loads(x) for x in (path/'requests.jsonl').read_text().splitlines() if x]
 assert len(q)==r['requests'] and len({x['request_id'] for x in q})==len(q)
 assert q and all(x['success'] for x in q) and not r['failed_requests']
 assert all(0<=x['cached_tokens']<=x['prompt_tokens'] and math.isfinite(x['shared_fraction']) for x in q)
 assert sum(x['prompt_tokens'] for x in q)==r['prompt_tokens']
 assert sum(x['cached_tokens'] for x in q)==r['cached_tokens']
 assert math.isclose(r['cached_tokens']/r['prompt_tokens'],r['token_weighted_shared_fraction'],abs_tol=1e-12)
 assert math.isclose(statistics.mean(x['shared_fraction'] for x in q),r['mean_request_shared_fraction'],abs_tol=1e-12)
 assert all(x['prompt_tokens']>0 and math.isclose(x['cached_tokens']/x['prompt_tokens'],x['shared_fraction'],abs_tol=1e-12) for x in q)
 counter=path/'eviction_counts.json'
 if counter.exists():
  measured=json.loads(counter.read_text())
  assert isinstance(measured['cache_evictions'],int) and measured['cache_evictions']>=0
  if 'cache_evictions' in r:assert r['cache_evictions']==measured['cache_evictions']
  r.update(measured)
 h=readcsv(path/'block_hits.csv');g=readcsv(path/'reuse_gaps.csv')
 assert len({x['block_key'] for x in h})==len(h)==r['unique_blocks']
 assert sum(int(x['hits']) for x in h)==r['block_hits']
 assert len(g)==r['reuse_gap_samples'] and all(math.isfinite(float(x['reuse_gap_s'])) and float(x['reuse_gap_s'])>=0 for x in g)
 # Independently reconstruct block/access/hit populations from the raw event stream.
 observed={};gap_index=0;event_hash=hashlib.sha256();label_gaps=defaultdict(list);request_events=set();request_index={x['request_id']:x for x in q}
 def client_id(server_id):
  if server_id.startswith('chatcmpl-'):return server_id[len('chatcmpl-'):]
  assert server_id.startswith('cmpl-') and server_id.endswith('-0'),server_id
  return server_id[len('cmpl-'):-2]
 with (path/'blocks.jsonl').open('rb') as stream:
  for line in stream:
   event_hash.update(line)
   if not line.strip():continue
   event=json.loads(line)
   if event['event']=='metadata':assert event['block_size']==r['block_size']
   if event['event']=='cache_config':
    assert event['num_gpu_blocks']==r['num_gpu_blocks']
    assert event['eviction_policy']==r['policy']
    assert event['num_gpu_blocks']*event['block_bytes']==r['virtual_cache_bytes']
   if event['event'] in ['request','block']:
    rid=client_id(event['request_id']);assert rid in request_index
    assert event.get('workload')==request_index[rid].get('workload')
    if event['event']=='request':
     assert rid not in request_events;request_events.add(rid)
     assert event['prompt_tokens']==request_index[rid]['prompt_tokens']
     assert event['cached_tokens']==request_index[rid]['cached_tokens']
   if event['event']!='block':continue
   entry=observed.setdefault(event['block_key'],[0,0]);entry[0]+=1;entry[1]+=int(event['hit'])
   if event['reuse_gap_s'] is not None:
    exported=g[gap_index];gap_index+=1
    label_gaps[event.get('workload')].append(event['reuse_gap_s'])
    assert exported['block_key']==event['block_key'] and exported['request_id']==event['request_id']
    assert math.isclose(float(exported['reuse_gap_s']),event['reuse_gap_s'],abs_tol=1e-12)
    assert math.isclose(float(exported['time_s']),event['time_s'],abs_tol=1e-12)
 assert request_events==set(request_index)
 assert gap_index==len(g)
 if g:assert math.isclose(statistics.mean(float(x['reuse_gap_s']) for x in g),r['mean_reuse_gap_s'],abs_tol=1e-12)
 else:assert r['mean_reuse_gap_s'] is None
 assert r['successful_requests']==len(q)
 assert math.isclose(max(x['finished_s'] for x in q),r['elapsed_s'],abs_tol=1e-12)
 assert observed=={x['block_key']:[int(x['accesses']),int(x['hits'])] for x in h}
 assert sum(v[1]==0 for v in observed.values())==r['zero_hit_blocks']
 r['block_events_sha256']=event_hash.hexdigest()
 times=[x['sent_s'] for x in q]
 r.update({'median_request_shared_fraction':statistics.median(x['shared_fraction'] for x in q),
           'median_dispatch_lag_s':quant([x['dispatch_lag_s'] for x in q],50),
           'p95_dispatch_lag_s':quant([x['dispatch_lag_s'] for x in q],95),
           'achieved_dispatch_requests_per_s':(len(times)-1)/(max(times)-min(times)) if len(times)>1 else None,
           'evictions':r.get('cache_evictions'), 'eviction_count_status':('opt-in allocator counter' if 'cache_evictions' in r else 'not instrumented in historical event schema'),
           'raw_directory':str(path.resolve())})
 if r.get('by_workload'):
  for label,stats in r['by_workload'].items():
   subset=[x for x in q if x.get('workload')==label]
   if label_gaps[label]:assert math.isclose(statistics.mean(label_gaps[label]),stats['mean_reuse_gap_s'],abs_tol=1e-12)
   else:assert stats['mean_reuse_gap_s'] is None
   assert subset and math.isclose(sum(x['cached_tokens'] for x in subset)/sum(x['prompt_tokens'] for x in subset),stats['token_weighted_shared_fraction'],abs_tol=1e-12)
 return r,q

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for n in ['m2','mixed','task1','output']:p.add_argument('--'+n,type=Path,required=True)
 p.add_argument('--allow-incomplete',action='store_true',help='Preview completed runs only; never use as a final completion claim')
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 out=a.output
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':14,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'savefig.dpi':320})
 def save(name):
  plt.tight_layout();plt.savefig(out/(name+'.pdf'),bbox_inches='tight');plt.savefig(out/(name+'.png'),bbox_inches='tight');plt.close()
 families={};req={};checked=[];input_hashes={};prepared_manifests={}
 for family,root in [('m2',a.m2),('mixed',a.mixed),('task1',a.task1)]:
  paths=[x.parent for x in root.rglob('complete.json')] if family!='task1' else [x.parent for x in root.rglob('summary.json')]
  rs=[]
  for d in sorted(paths):
   if family=='mixed':assert (d/'eviction_counts.json').exists(),d
   r,q=validate(d);rs.append(r);req[str(d.resolve())]=q
   checked.append({'family':family,'directory':str(d.resolve()),'requests':r['requests'],'status':'passed'})
   if len(checked)%25==0:print('Validated',len(checked),'completed raw runs',flush=True)
   input_hashes[str((d/'blocks.jsonl').resolve())]=r['block_events_sha256']
   for f in ['run.json','summary.json','requests.jsonl','block_hits.csv','reuse_gaps.csv']+(['eviction_counts.json'] if (d/'eviction_counts.json').exists() else []):
    z=d/f;input_hashes[str(z.resolve())]=hashlib.sha256(z.read_bytes()).hexdigest()
  for r in rs:
   local=(root/'prepared'/r['workload']/f"seed-{r['seed']}") if family=='task1' else ((root/'prepared'/f"seed-{r['seed']}"/r['mode']) if family=='m2' else root/r['mixture']/f"seed-{r['seed']}"/'prepared')
   prepared=local if local.exists() else Path(r['prepared_directory'])
   for name,key in [('workload.json','workload_sha256'),('trace.jsonl','trace_sha256')]:
    z=prepared/name;namekey=str(z.resolve())
    if namekey not in input_hashes:input_hashes[namekey]=hashlib.sha256(z.read_bytes()).hexdigest()
    assert input_hashes[namekey]==r[key]
   prepared_key=str(prepared.resolve())
   if prepared_key not in prepared_manifests:prepared_manifests[prepared_key]=json.loads((prepared/'workload.json').read_text())
   manifest=prepared_manifests[prepared_key]
   assert r['model']==manifest['model'] and r['revision']==manifest['revision']
   if 'tokenizer_revision' in r:assert r['tokenizer_revision']==manifest['tokenizer_revision']
   actual={q['request_id']:q for q in req[r['raw_directory']]}
   expected={t['request_id']:t['prompt_tokens'] for c in manifest['conversations'] for t in c['turns']}
   assert {k:v['prompt_tokens'] for k,v in actual.items()}==expected
   for c in manifest['conversations']:
    previous=None
    for t in c['turns']:
     q=actual[t['request_id']]
     assert q.get('workload')==c.get('workload')
     assert all(math.isfinite(q[k]) for k in ['sent_s','finished_s','dispatch_lag_s','latency_s'])
     assert q['finished_s']>=q['sent_s']
     if previous is not None:assert q['sent_s']>=previous['finished_s']+manifest['think_time_s']-1e-6
     previous=q
  identities=[(r.get('workload') if family=='task1' else r.get('mixture',r.get('mode')),r['policy'],r['seed'],r.get('block_size'),r.get('cache_gb')) for r in rs]
  assert len(set(identities))==len(identities)
  families[family]=rs
  csvout(out/(family+'-runs.csv'),[{k:v for k,v in r.items() if not isinstance(v,(dict,list))} for r in rs]);dump(out/(family+'-runs.json'),rs)
 # Cohort/capacity compatibility within compared policies.
 cohorts={}
 for fam in families:
  for r in families[fam]:
   group=(fam,r.get('workload') if fam=='task1' else r.get('mixture',r.get('mode')),r.get('seed'),r.get('block_size'),r.get('cache_gb'),r.get('cache_ratio'))
   identity=(r['workload_sha256'],r['trace_sha256'],r['prompt_tokens'],r['requests'],r['virtual_cache_bytes'],frozenset((q['request_id'],q['prompt_tokens']) for q in req[r['raw_directory']]))
   assert cohorts.setdefault(group,identity)==identity
 # Matched first-turn cohorts across M2 modes; exact trace/tokenization equality.
 for seed in sorted({r['seed'] for r in families['m2']}):
  prep=a.m2/'prepared'/f'seed-{seed}'
  x=json.loads((prep/'single/workload.json').read_text());y=json.loads((prep/'multi/workload.json').read_text())
  assert [c['id'] for c in x['conversations']]==[c['id'] for c in y['conversations']]
  assert [c['turns'][0] for c in x['conversations']]==[c['turns'][0] for c in y['conversations']]
 # Repeated-seed summaries, not confidence intervals.
 groups=defaultdict(list)
 for r in families['m2']:groups[(r['mode'],r['cache_gb'],r['block_size'],r['policy'])].append(r)
 m2=[]
 for key,rs in sorted(groups.items()):
  vals=[r['token_weighted_shared_fraction'] for r in rs]
  m2.append(dict(zip(['mode','cache_gb','block_size','policy'],key),mean=statistics.mean(vals),sd=statistics.stdev(vals) if len(vals)>1 else None,n=len(vals),seeds=sorted(r['seed'] for r in rs)))
 dump(out/'m2-means.json',m2);csvout(out/'m2-means.csv',m2)
 tgroups=defaultdict(list)
 for r in families['task1']:tgroups[(r['workload'],r['policy'])].append(r)
 t1=[]
 for (label,policy),rs in tgroups.items():
  vals=[r['token_weighted_shared_fraction'] for r in rs]
  t1.append({'workload':label,'policy':policy,'mean':statistics.mean(vals),'sd':statistics.stdev(vals) if len(vals)>1 else None,'n':len(vals)})
 winners=[]
 for label in LABELS:
  rs=sorted([r for r in t1 if r['workload']==label],key=lambda x:-x['mean'])
  winners.append({'workload':label,'highest_observed':','.join(r['policy'] for r in rs if abs(r['mean']-rs[0]['mean'])<1e-12),'gap_to_second_pp':100*(rs[0]['mean']-rs[1]['mean']),'tie':abs(rs[0]['mean']-rs[1]['mean'])<1e-12})
 dump(out/'task1-means.json',t1);csvout(out/'task1-means.csv',t1);csvout(out/'task1-observed-best.csv',winners)
 fig,ax=plt.subplots(figsize=(10.6,3.5));xx=np.arange(8)
 for j,policy in enumerate(POLICIES[:3]):
  rs=[next(r for r in t1 if r['workload']==l and r['policy']==policy) for l in LABELS]
  ax.bar(xx+(j-1)*.24,[r['mean']*100 for r in rs],.23,yerr=[r['sd']*100 if r['sd'] is not None else np.nan for r in rs],color=COLORS[policy],label=policy.upper(),capsize=2)
 ax.set_xticks(xx,LABELS,rotation=18,ha='right');ax.set_ylabel('Cached prompt tokens (%)');ax.legend(ncol=3,frameon=False);ax.grid(axis='y',alpha=.15);save('m3_task1')
 # M2-A reuses M2-B .05 GiB block16 LRU settings.
 fig,axs=plt.subplots(1,2,figsize=(8.6,3))
 for j,cap in enumerate([.05,.1]):
  rs=[next(r for r in m2 if r['mode']==mode and r['cache_gb']==cap and r['block_size']==16 and r['policy']=='lru') for mode in ['single','multi']]
  axs[j].bar(['Single','Multi'],[r['mean']*100 for r in rs],yerr=[(r['sd'] or 0)*100 for r in rs],capsize=3,color=['#A5CDE0',COLORS['lru']]);axs[j].set_title(f'LRU · {cap:g} GiB · block 16');axs[j].set_ylabel('Cached prompt tokens (%)');axs[j].grid(axis='y',alpha=.15)
 save('m2_single_multi')
 fig,axs=plt.subplots(1,2,figsize=(10.4,3.4))
 for mode,style in [('single','--'),('multi','-')]:
  for policy in POLICIES[:3]:
   rs=[next(r for r in m2 if r['mode']==mode and r['cache_gb']==cap and r['block_size']==16 and r['policy']==policy) for cap in [.05,.1]]
   axs[0].errorbar([.05,.1],[r['mean']*100 for r in rs],yerr=[(r['sd'] or 0)*100 for r in rs],label=mode+' '+policy.upper(),color=COLORS[policy],linestyle=style,marker='o',capsize=2)
   rs=[next(r for r in m2 if r['mode']==mode and r['cache_gb']==.05 and r['block_size']==b and r['policy']==policy) for b in [8,16,32]]
   axs[1].errorbar([8,16,32],[r['mean']*100 for r in rs],yerr=[(r['sd'] or 0)*100 for r in rs],color=COLORS[policy],linestyle=style,marker='o',capsize=2)
 axs[0].set_xlabel('Virtual cache (GiB), block 16');axs[1].set_xlabel('Tokens/block, 0.05 GiB');axs[0].set_xticks([.05,.1]);axs[1].set_xticks([8,16,32]);axs[0].set_ylabel('Cached prompt tokens (%)');fig.legend(*axs[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,1.22),fontsize=13,ncol=3,frameon=False);axs[1].grid(alpha=.15);axs[0].grid(alpha=.15);save('m2_sensitivity')
 def cdf(values,ax,label,color):
  v=np.sort(values);ax.step(v,np.arange(1,len(v)+1)/len(v),where='post',label=label,color=color)
 refs={mode:next(r for r in families['m2'] if r['mode']==mode and r['cache_gb']==.05 and r['block_size']==16 and r['policy']=='lru' and r['seed']==0) for mode in ['single','multi']}
 stats={}
 for kind,field,xlabel in [('request','shared_fraction','Per-request cached fraction'),('blocks','hits','Hits per logical full prefix block'),('gaps','reuse_gap_s','Hit-associated reuse gap (seconds)')]:
  fig,ax=plt.subplots(figsize=(6.3,3.3))
  for mode,color in [('single','#DE8A29'),('multi','#167DAD')]:
   r=refs[mode];d=Path(r['raw_directory']);qs=req[r['raw_directory']] if kind=='request' else readcsv(d/('block_hits.csv' if kind=='blocks' else 'reuse_gaps.csv'))
   values=[float(q[field]) for q in qs];cdf(values,ax,mode,color);stats.setdefault(mode,{})[kind]={'n':len(values),'median':statistics.median(values) if values else None,'p95':quant(values,95)}
  if kind!='request':ax.set_xscale('symlog',linthresh=1 if kind=='blocks' else .001)
  ax.set_xlabel(xlabel);ax.set_ylabel('Empirical CDF');ax.legend(frameon=False);ax.grid(alpha=.15);save('m2_cdf_'+kind)
 dump(out/'m2-cdf-statistics.json',stats)
 planned={mode:prompt_recurrence(a.m2/'prepared'/'seed-0'/mode) for mode in ['single','multi']}
 dump(out/'m2-planned-prompt-recurrence.json',planned)
 fig,ax=plt.subplots(figsize=(6.3,3.3))
 for mode,color in [('single','#94A3B8'),('multi','#167DAD')]:
  cdf(planned[mode]['planned_recurrence_gaps_s'],ax,mode,color)
 ax.set_xlabel('Gap between planned prompt recurrences (s)');ax.set_ylabel('Empirical CDF');ax.set_title('Prompt-only recurrence · no cache/availability model',fontsize=12);ax.legend(frameon=False);ax.grid(alpha=.15);save('m2_planned_prompt_recurrence')
 # Heterogeneous request/block/gap distributions; seed0 LRU only.
 for kind,field,xlabel in [('request','shared_fraction','Per-request cached fraction'),('blocks','hits','Hits per logical full prefix block'),('gaps','reuse_gap_s','Hit-associated reuse gap (seconds)')]:
  fig,ax=plt.subplots(figsize=(6.3,3.3))
  for label,color in zip(['chat','agent','coding-agent','quality'],['#167DAD','#DE8A29','#7256A8','#15977B']):
   r=next(r for r in families['task1'] if r['workload']==label and r['policy']=='lru' and r['seed']==0);d=Path(r['raw_directory']);qs=req[r['raw_directory']] if kind=='request' else readcsv(d/('block_hits.csv' if kind=='blocks' else 'reuse_gaps.csv'));cdf([float(q[field]) for q in qs],ax,label,color)
  if kind!='request':ax.set_xscale('symlog',linthresh=1 if kind=='blocks' else .001)
  ax.set_xlabel(xlabel);ax.set_ylabel('Empirical CDF');ax.legend(frameon=False,ncol=2,fontsize=13);ax.grid(alpha=.15);save('m3_cdf_'+kind)
 # Complete source-specific CDF supplement: all eight sources, seed0, all baselines.
 supplement=out/'task1_by_source';supplement.mkdir(exist_ok=True)
 for label in LABELS:
  for kind,field,xlabel in [('request','shared_fraction','Per-request cached fraction'),('blocks','hits','Hits per logical full prefix block'),('gaps','reuse_gap_s','Hit-associated reuse gap (seconds)')]:
   fig,ax=plt.subplots(figsize=(6.3,3.3))
   for policy in POLICIES[:3]:
    r=next(r for r in families['task1'] if r['workload']==label and r['policy']==policy and r['seed']==0);d=Path(r['raw_directory'])
    population=req[r['raw_directory']] if kind=='request' else readcsv(d/('block_hits.csv' if kind=='blocks' else 'reuse_gaps.csv'))
    cdf([float(q[field]) for q in population],ax,policy.upper(),COLORS[policy])
   if kind!='request':ax.set_xscale('symlog',linthresh=1 if kind=='blocks' else .001)
   ax.set_xlabel(xlabel);ax.set_ylabel('Empirical CDF');ax.set_title(label+' · seed0',fontsize=13);ax.legend(frameon=False,fontsize=13);ax.grid(alpha=.15);save('task1_by_source/'+label+'-'+kind)
 # Complete mixed experiments are required; no documentary values substituted.
 mixed=families['mixed']
 if len(mixed)<16 and a.allow_incomplete:
  dump(out/'validation.json',{'incomplete':True,'checked_runs':checked,'totals':{k:{'runs':len(v),'requests':sum(r['requests'] for r in v),'failures':sum(r['failed_requests'] for r in v)} for k,v in families.items()},'input_sha256':input_hashes})
  print('PREVIEW ONLY: mixed evaluation incomplete; no final mixed figures or conclusions generated.')
  return
 assert len(mixed)>=16
 fig,axs=plt.subplots(1,2,figsize=(10.8,3.6),sharey=True);mixed_best=[];composition=[];per=[]
 for ax,mixture in zip(axs,['skewed','equal-token']):
  rs=[next(r for r in mixed if r['mixture']==mixture and r['policy']==p) for p in POLICIES]
  ax.bar(np.arange(8),[100*r['token_weighted_shared_fraction'] for r in rs],color=[COLORS[p] for p in POLICIES]);ax.set_xticks(np.arange(8),[p.upper() for p in POLICIES],rotation=45,ha='right');ax.set_title(mixture);ax.grid(axis='y',alpha=.15)
  for i,r in enumerate(rs):ax.text(i,100*r['token_weighted_shared_fraction']+.6,f"{100*r['token_weighted_shared_fraction']:.1f}",ha='center',fontsize=12)
  best=max(rs,key=lambda r:r['token_weighted_shared_fraction']);baseline=max(rs[:3],key=lambda r:r['token_weighted_shared_fraction']);new=best['token_weighted_shared_fraction'];base=baseline['token_weighted_shared_fraction']
  mixed_best.append({'mixture':mixture,'best_observed':best['policy'],'strongest_baseline':baseline['policy'],'best_fraction':new,'baseline_fraction':base,'gain_pp':100*(new-base),'relative_gain_percent':100*(new-base)/base if base else None,'n':1,'sd':None})
  q=req[rs[0]['raw_directory']];total=sum(x['prompt_tokens'] for x in q)
  for label in LABELS:
   sub=[x for x in q if x['workload']==label];composition.append({'mixture':mixture,'workload':label,'requests':len(sub),'request_share':len(sub)/len(q),'prompt_tokens':sum(x['prompt_tokens'] for x in sub),'token_share':sum(x['prompt_tokens'] for x in sub)/total})
  for r in rs:
   for label,v in r['by_workload'].items():per.append({'mixture':mixture,'policy':r['policy'],'workload':label,**v})
 axs[0].set_ylabel('Cached prompt tokens (%)');save('mixed_overall');dump(out/'mixed-best.json',mixed_best);csvout(out/'mixed-best.csv',mixed_best);csvout(out/'mixed-composition.csv',composition);csvout(out/'mixed-by-workload.csv',per)
 fig,axs=plt.subplots(1,2,figsize=(10.6,3.7))
 for ax,mixture in zip(axs,['skewed','equal-token']):
  mat=np.array([[100*next(v['token_weighted_shared_fraction'] for v in per if v['mixture']==mixture and v['policy']==p and v['workload']==l) for l in LABELS] for p in POLICIES]);im=ax.imshow(mat,vmin=0,vmax=100,cmap='YlGnBu',aspect='auto');ax.set_xticks(np.arange(8),LABELS,rotation=45,ha='right',fontsize=13);ax.set_yticks(np.arange(8),[p.upper() for p in POLICIES],fontsize=13);ax.set_title(mixture)
 fig.colorbar(im,ax=axs.ravel().tolist(),label='Cached prompt tokens (%)',fraction=.025,pad=.03);plt.savefig(out/'mixed_by_workload.pdf',bbox_inches='tight');plt.savefig(out/'mixed_by_workload.png',bbox_inches='tight');plt.close()
 fig,axs=plt.subplots(1,2,figsize=(9,3.1),sharey=True)
 for ax,kind in zip(axs,['request_share','token_share']):
  bottom=np.zeros(2)
  for label,c in zip(LABELS,plt.cm.tab10.colors):
   vals=[100*next(r[kind] for r in composition if r['mixture']==m and r['workload']==label) for m in ['skewed','equal-token']];ax.bar(['Skewed','Equal-token'],vals,bottom=bottom,label=label,color=c);bottom+=vals
  ax.set_title('Requests' if kind=='request_share' else 'Prompt tokens');ax.set_ylabel('Composition (%)')
 axs[1].legend(bbox_to_anchor=(1.01,1),frameon=False,fontsize=13);save('mixed_composition')
 dump(out/'validation.json',{'checked_runs':checked,'totals':{k:{'runs':len(v),'requests':sum(r['requests'] for r in v),'failures':sum(r['failed_requests'] for r in v)} for k,v in families.items()},'input_sha256':input_hashes,'mixed_uncertainty':'one seed per mixture; no SD/CI estimate','m2_controls':'original documented absolute-byte sweep, 3 seeds','eviction_counts':'mixed: opt-in successful allocator-eviction counter; M2/old Task1 unavailable (null), never inferred from misses'})
 print(json.dumps({'totals':{k:len(v) for k,v in families.items()},'mixed_best':mixed_best}))

if __name__=='__main__':main()

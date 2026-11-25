"""Fill the paper from independently validated tables; never run experiments."""
from pathlib import Path
import json,shutil,re
D=Path(__file__).resolve().parent;F=D.parent/'figures'
meta=json.loads((F/'validation.json').read_text());assert meta['totals']['m2']['runs']==108 and meta['totals']['mixed']['runs']==16
m2=json.loads((F/'m2-means.json').read_text());mix=json.loads((F/'mixed-runs.json').read_text());best=json.loads((F/'mixed-best.json').read_text());t1=json.loads((F/'task1-means.json').read_text())
def m(mode,cap=.05,block=16,policy='lru'):return next(r for r in m2 if r['mode']==mode and r['cache_gb']==cap and r['block_size']==block and r['policy']==policy)
def pct(x):return f'{100*x:.2f}\\%'
def val(r):return f"({100*r['mean']:.2f} $\\pm$ {100*r['sd']:.2f})\\%" if r['sd'] is not None else pct(r['mean'])+' (n=1)'
def mx(mixture,policy):return next(r for r in mix if r['mixture']==mixture and r['policy']==policy)
bs={r['mixture']:r for r in best}
m2text=(f"At block16 with LRU, single-turn sharing is {val(m('single'))} at0.05\\,GiB versus multi-turn {val(m('multi'))}. At0.10\\,GiB, the corresponding means are {val(m('single',.1))} and {val(m('multi',.1))} (Figure~\\ref{{fig:m2mode}}). Each mode starts from200 identical eligible conversations; single runs contain200 requests and multi runs608. All three arrival seeds use the same tokenized cohorts. These are observations under the documented64-concurrent-request load, not an assumption that multi-turn must improve sharing.")
blockstr=lambda mode:', '.join(pct(m(mode,block=b)['mean']) for b in [8,16,32])
sens=(f"At0.05\\,GiB, LRU scores at8/16/32 tokens per block are {blockstr('single')} for single-turn and {blockstr('multi')} for multi-turn. Doubling byte capacity changes block16 LRU by {100*(m('single',.1)['mean']-m('single')['mean']):+.2f}pp (single) and {100*(m('multi',.1)['mean']-m('multi')['mean']):+.2f}pp (multi). Figure~\\ref{{fig:sensitivity}} shows the other policies and seed variability; the complete36-cell mean/SD table is supplementary.")
sens+=f" In single-turn/block8 at0.05\\,GiB, FIFO reaches {val(m('single',block=8,policy='fifo'))}, versus LRU {val(m('single',block=8))} and LFU {val(m('single',block=8,policy='lfu'))}. The observed ranking therefore interacts with granularity; the study does not isolate a causal retention mechanism."
mixedtext=[]
for mixture in ['skewed','equal-token']:
 b=bs[mixture];r=mx(mixture,b['best_observed']);relative=f"{b['relative_gain_percent']:.2f}\\%" if b['relative_gain_percent'] is not None else 'undefined (zero baseline)'
 mixedtext.append(f"The {mixture} mixture contains{r['requests']:,} requests and{r['prompt_tokens']:,} prompt tokens, with{r['num_gpu_blocks']:,} virtual blocks. Its highest observed policy is {b['best_observed'].upper()} at{pct(b['best_fraction'])}, compared with strongest baseline {b['strongest_baseline'].upper()} at{pct(b['baseline_fraction'])}: {b['gain_pp']:+.2f}pp, or{relative} relative using unrounded scores. This single-run margin has no SD/CI estimate.")
tf=[mx(z,'tfifo')['token_weighted_shared_fraction'] for z in ['skewed','equal-token']];rf=[mx(z,'rfifo')['token_weighted_shared_fraction'] for z in ['skewed','equal-token']]
chat=[]
for mixture in ['skewed','equal-token']:
 b=bs[mixture];a=mx(mixture,'tfifo')['by_workload']['chat']['token_weighted_shared_fraction'];v=mx(mixture,b['strongest_baseline'])['by_workload']['chat']['token_weighted_shared_fraction'];chat.append(f"{mixture}: TFIFO chat{pct(a)} versus baseline{pct(v)}")
interpret=(f"TFIFO changes from{pct(tf[0])} on the skewed mix to{pct(tf[1])} on equal tokens; RFIFO changes from{pct(rf[0])} to{pct(rf[1])}. The per-label chat comparison is {'; '.join(chat)}. Static treatment of chat as one-shot can therefore be evaluated against measured category effects rather than inferred from the aggregate alone. Dataset composition, hold expiration and adaptive admission offer plausible mechanisms; there is no held-out tuning/ablation study establishing causation. The new raw-validated scores replace documentary historical values rather than being adjusted to reproduce them.")
for mixture in ['skewed','equal-token']:
 b=bs[mixture];winner=mx(mixture,b['best_observed']);base=mx(mixture,b['strongest_baseline'])
 deltas={l:100*(winner['by_workload'][l]['token_weighted_shared_fraction']-base['by_workload'][l]['token_weighted_shared_fraction']) for l in winner['by_workload']}
 label=min(deltas,key=deltas.get)
 if deltas[label]<0:interpret+=f" On {mixture}, the aggregate winner's largest category decline versus the strongest baseline is {label}, {deltas[label]:.2f}pp; aggregate improvement does not protect every category."

cdfstats=json.loads((F/'m2-cdf-statistics.json').read_text())
m2text+=f" Representative seed0 request-fraction medians are {pct(cdfstats['single']['request']['median'])} single and {pct(cdfstats['multi']['request']['median'])} multi; the logical-block hit median is zero in both. These populations differ from token weighting."
planned=json.loads((F/'m2-planned-prompt-recurrence.json').read_text())
m2text+=" At block16 the two capacities hold273/546 virtual blocks (4,368/8,736 tokens), small relative to64 active sequence slots."
m2text+=f" A policy-independent diagnostic finds {pct(planned['single']['planned_prompt_recurrence_fraction'])} single-turn and {pct(planned['multi']['planned_prompt_recurrence_fraction'])} multi-turn prompt tokens in full blocks seen in an earlier planned prompt. It omits decode blocks, availability and actual completion timing, so it is not a full-serving cache bound. The large multi-turn recurrence versus low realized sharing distinguishes existing overlap from retained state."
# Detailed tables use actual seed counts and do not print single-run zero SD.
pols=['lru','lfu','fifo','tlru','tfifo','hlru','hfifo','rfifo'];labels=['chat','agent','coding-agent','quality','bailian-chat','bailian-coder','mooncake-agent','code']
tex='\\begin{table*}[t]\\centering\\small\n\\caption{Task1 sharing (percent), sample SD where repeated, and observed best--second gap. Single-run uncertainty is unknown.}\n\\begin{tabular}{lrllllr}\\toprule Workload & n & LRU & LFU & FIFO & Best & Gap (pp)\\\\\\midrule\n'
for l in labels:
 qs=[next(r for r in t1 if r['workload']==l and r['policy']==p) for p in pols[:3]];order=sorted(qs,key=lambda x:-x['mean']);nums=[f"{100*r['mean']:.2f}"+(f" $\\pm$ {100*r['sd']:.2f}" if r['sd'] is not None else '') for r in qs];bestname='tie' if abs(order[0]['mean']-order[1]['mean'])<1e-12 else order[0]['policy'].upper();tex+=l+f" & {qs[0]['n']} & "+' & '.join(nums)+f" & {bestname} & {100*(order[0]['mean']-order[1]['mean']):.3f}\\\\\n"
tex+='\\bottomrule\\end{tabular}\\end{table*}\n'
tex+='\\begin{table*}[t]\\centering\\small\\caption{M2 complete repeated-seed table. All rows n=3; values are mean $\\pm$ sample SD in percent.}\n\\begin{tabular}{lrrlrr}\\toprule Mode & GiB & Block & Policy & Mean & SD\\\\\\midrule\n'
for r in m2:tex+=f"{r['mode']} & {r['cache_gb']:g} & {r['block_size']} & {r['policy'].upper()} & {100*r['mean']:.3f} & {100*r['sd']:.3f}\\\\\n"
tex+='\\bottomrule\\end{tabular}\\end{table*}\n'
tex+='\\begin{table*}[t]\\centering\\small\\caption{Mixed scores (percent) and measured eviction allocations; one seed per mixture, unknown variability.}\n\\begin{tabular}{lrrrr}\\toprule Policy & Skewed & Equal-token & Evict: skewed & Evict: equal\\\\\\midrule\n'
for p in pols:tex+=f"{p.upper()} & {100*mx('skewed',p)['token_weighted_shared_fraction']:.3f} & {100*mx('equal-token',p)['token_weighted_shared_fraction']:.3f} & {mx('skewed',p)['cache_evictions']:,} & {mx('equal-token',p)['cache_evictions']:,}\\\\\n"
tex+='\\bottomrule\\end{tabular}\\end{table*}\n'
tex+='Exact commands and input hashes are in the experiment manifests; source changes are preserved as a patch against base3d501ab. Figure generation uses existing results only. The README provides setup, smoke tests, resumable experiment commands and external raw-data locations. No raw logs or prepared traces were overwritten.\n'
replacements={'M2_RUNS':'108','ABSTRACT_RESULT':f"With LRU/block16, matched single/multi means are {pct(m('single')['mean'])}/{pct(m('multi')['mean'])} at0.05\\,GiB. Coding-agent LRU exceeds LFU by13.16pp. Highest observed mixed scores are {bs['skewed']['best_observed'].upper()} {pct(bs['skewed']['best_fraction'])} (skewed) and {bs['equal-token']['best_observed'].upper()} {pct(bs['equal-token']['best_fraction'])} (equal-token).",'MIXED_SCOPE':'We complete one seed per mixture for all eight policies before any optional repetition, totaling16 runs. The two-core/8\\,GiB environment and recorded run times favor complete required comparisons over large ablation products; no mixed confidence interval is inferred.','VALIDATION_TOTAL':f"All{sum(v['runs'] for v in meta['totals'].values())} studied configurations pass exported consistency checks; raw request totals include repeated replays of the same cohorts, not independent dataset samples.",'M2_RESULTS':m2text,'M2_SENSITIVITY':sens,'MIXED_RESULTS':'\n\n'.join(mixedtext),'MIXED_INTERPRETATION':interpret,'CONCLUSION':f"Matched chat reuse changes with block/byte controls, while heterogeneous policies range from large coding-agent differences to near ties. Highest observed mixed policies are {bs['skewed']['best_observed'].upper()} on skewed and {bs['equal-token']['best_observed'].upper()} on equal-token input, with single-seed uncertainty.",'APPENDIX_TABLES':tex}
s=(D/'main.template.tex').read_text()
for key,value in replacements.items():s=s.replace('@'+key+'@',value)
assert not re.search(r'@[A-Z_]+@',s)
# Repair prose spacing without changing filenames, equations or identifiers.
words=['FIFO','LFU','LRU','TFIFO','RFIFO','Mooncake','Milestone','Task','about','achieves','all','an','and','approximately','are','at','block','budget','by','capacity','contains','concurrency','context','contributes','costs','for','from','in','is','least','load','of','only','or','over','rate','ratio','reliable','scores','seed','seeds','seconds','source','think','time','to','totaling','uses','use','versus','with']
s=re.sub(r'\b('+'|'.join(words)+r')(?=\d)',r'\1 ',s);s=re.sub(r'(\d)(pp|s)\b',r'\1\,\2',s)
(D/'main.tex').write_text(s)
for p in F.glob('*.pdf'):shutil.copy2(p,D/'figures'/p.name)
print('Paper filled from validated tables.')

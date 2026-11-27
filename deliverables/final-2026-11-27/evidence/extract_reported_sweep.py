"""Extract rounded, source-attributed sweep tables without touching original artifacts."""
from pathlib import Path
import argparse,json,csv,hashlib
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
source=args.source.read_text();output=args.output;output.mkdir(parents=True,exist_ok=True)
assert hashlib.sha256(source.encode()).hexdigest()=='885b01ac2225665fc9682cf9702afb5b4df28ce8d8ba97b0bbb0d6be70e283ca', 'Source differs from reviewed commit; review before extraction'
commit='ad1d8265420c2515882300710d1cdf5d375515ba'
text=source.split('## Hold and admission sweep',1)[1]
def table(header):
 s=text.split(header,1)[1];rows=[]
 for line in s.splitlines():
  if not line.startswith('|'):
   if rows:break
   continue
  vals=[v.strip().replace('**','').replace('`','') for v in line.strip('|').split('|')]
  if vals[0].startswith('---'):continue
  rows.append(vals)
 return rows
scores=table('| Setting | Chat | Protected | Trace | Template | Balanced |')
assert len(scores)==25 and len({r[0] for r in scores})==25
names=['chat-heavy','protected-heavy','trace-heavy','template-heavy','balanced']
rows=[{'setting':r[0],**{n:float(v) for n,v in zip(names,r[1:])}} for r in scores]
assert all(0<=r[n]<=100 for r in rows for n in names)
maxima={n:max(r[n] for r in rows) for n in names}
rank=[]
for r in rows:
 gaps=[maxima[n]-r[n] for n in names];rank.append({'setting':r['setting'],'macro_mean_percent':sum(r[n] for n in names)/5,'mean_regret_pp':sum(gaps)/5,'worst_regret_pp':max(gaps)})
for r in rank:
 for key in ['macro_mean_percent','mean_regret_pp','worst_regret_pp']:r[key]=round(r[key],2)
rank.sort(key=lambda r:-r['macro_mean_percent'])
cohort=table('| Mix | Requests | Prompt tokens | Cache blocks | What is large |')
assert len(cohort)==5
cohorts=[]
for r in cohort:
 cap,working=map(lambda z:int(z.strip().replace(',','')),r[3].split('/'))
 cohorts.append({'mixture':r[0],'requests':int(r[1].replace(',','')),'prompt_tokens':int(r[2].replace(',','')),'cache_blocks':cap,'working_set_blocks':working,'composition':r[4]})
labels=table('| Mix | Label | Share | LRU | LFU | TFIFO | HLRU h8 b1 | HLRU h30 b1 | TLRU h8 b0.25 | RFIFO w500 t0.05 |')
assert len(labels)==11
payload={'source_commit':commit,'source':'docs/milestone3-task2.md, Hold and admission sweep, larger m3-hold-grid3 table','source_sha256':hashlib.sha256(source.encode()).hexdigest(),'evidence_level':'repository-reported rounded aggregates; original requests/prepared inputs/manifests unavailable locally and not independently raw-validated','precision':'one decimal percentage; differences/rankings approximate, sub-0.1pp margins unresolved by rounding','reported_runs':125,'settings_per_mix':25,'mixtures':names,'seeds_per_cell':1,'reported_failures':0,'reported_clamping':False,'rows':rows,'cohorts':cohorts,'maxima':maxima,'derived_from_rounded_table':rank,'selected_label_rows':labels,'derived_replayed_requests':25*sum(r['requests'] for r in cohorts),'do_not_pool_with_historical':True,'execution_order':'fixed order in hold_grid.run_specs(), not randomized','hfifo':'not included in this sweep','test_evidence':'run separately on ad1d826542; see full-tests.log'}
(output/'hold-sweep-reported.json').write_text(json.dumps(payload,indent=2)+'\n')
with (output/'hold-sweep-reported.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=['setting',*names]);w.writeheader();w.writerows(rows)
print('Extracted 25 settings x 5 mixtures; no raw validation or experiments performed.')

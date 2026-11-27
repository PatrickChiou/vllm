"""Finalize compact evidence only after completed experiments and document QA."""
from pathlib import Path
import json, hashlib, shutil, subprocess
R=Path(__file__).resolve().parents[2];repo=R/'work/review-vllm';d=repo/'deliverables/final-2026-11-24';e=d/'evidence';w=R/'work/final-2026-11-24'
def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def git(*a):return subprocess.check_output(['git',*a],cwd=repo,text=True).strip()
meta=json.loads((d/'figures/validation.json').read_text());tot=meta['totals'];assert [tot[k]['runs'] for k in ['m2','task1','mixed']]==[108,51,16];assert all(v['failures']==0 for v in tot.values())
qa=json.loads((w/'document-qa/document-checks.json').read_text());assert qa['documents']['report']['main_pages']<=6 and qa['documents']['slides']['pages']==24
runtime={}
for family in ['m2','mixed']:
 markers=[json.loads(p.read_text()) for p in (w/'experiments'/family).rglob('complete.json')];runtime[family]={'completed_runs':len(markers),'completed_lifecycle_seconds':sum(m['runtime_s'] for m in markers)}
for name in ['m2-full.log','m2-pilot.log','mixed-driver.log','mixed-resume.log','final-figures.log','report-build.log','presentation-build.log','presentation-export.log','office-version-final.log','manual-smoke.log']:
 p=w/'evidence'/name
 if p.exists():shutil.copy2(p,e/name)
shutil.copy2(w/'document-qa/document-checks.json',e/'document-checks.json')
for p in (w/'document-qa').glob('*-contact-*.png'):shutil.copy2(p,e/p.name)
shutil.copy2(w/'validate_documents.py',e/'validate_documents.py')
dump(e/'runtime.json',runtime)
newminutes=sum(x['completed_lifecycle_seconds'] for x in runtime.values())/60
counts='; '.join(f"{k}: {v['runs']} runs / {v['requests']:,} requests / {v['failures']} failures" for k,v in tot.items())
text=f"Validated studies: **{counts}**. These replay counts are repeated configurations, not independent dataset samples. New completed M2/mixed server/replay/analysis lifecycles total **{newminutes:.2f} minutes** (M2 {runtime['m2']['completed_lifecycle_seconds']/60:.2f}; mixed {runtime['mixed']['completed_lifecycle_seconds']/60:.2f}); preparation, tests, failed/interrupted attempts and document work are separate. Task1 was reused, not rerun. Mixed results use one seed per composition; no SD/CI is estimated."
s=(d/'README.final.template.md').read_text().replace('@FINAL_TOTALS@',text)
s+='\n'+git('show','HEAD:README.md')+'\n';(repo/'README.md').write_text(s)
# Patch includes implementation/tests/new tools/README. Final documents are supplied as files, not encoded as binary patches.
tracked=['README.md','tests/simulator/test_sim_api_server.py','vllm/engine/multiprocessing/__init__.py','vllm/engine/multiprocessing/client.py','vllm/engine/multiprocessing/engine.py','vllm/simulator/eviction_study.py']
new=['tests/simulator/test_rpc_workload.py','tools/comp536_final_study.py','tools/comp536_final_figures.py','tools/comp536_diagrams.py','tools/comp536_prompt_recurrence.py','tools/comp536_counting/sitecustomize.py']
patch=subprocess.check_output(['git','diff','--binary','HEAD','--',*tracked],cwd=repo)
for f in new:
 z=subprocess.run(['git','diff','--no-index','--binary','--','/dev/null',f],cwd=repo,capture_output=True);assert z.returncode==1;patch+=z.stdout
(e/'final-source-changes.patch').write_bytes(patch)
dump(e/'source-hashes.json',{f:sha(repo/f) for f in tracked+new})

before=json.loads((repo/'deliverables/evidence/implementation-before.json').read_text())
changed=[f for f,h in before.items() if (repo/f).exists() and sha(repo/f)!=h]
expected={'tests/simulator/test_sim_api_server.py','vllm/engine/multiprocessing/__init__.py','vllm/engine/multiprocessing/client.py','vllm/engine/multiprocessing/engine.py','vllm/simulator/eviction_study.py'}
assert set(changed)==expected,changed
dump(e/'implementation-change-scope.json',{'original_snapshot_files':len(before),'changed_files':changed,'executor_scheduler_policy_decisions':'unchanged; transport fix and strict zip guard only','new_tests_and_tools':'recorded separately in source patch'})
core={f:sha(repo/f) for f in subprocess.check_output(['git','ls-files','vllm'],cwd=repo,text=True).splitlines()}
current=hashlib.sha256(json.dumps(core,sort_keys=True).encode()).hexdigest()
recorded={json.loads((m.parent/'run.json').read_text())['implementation_sha256'] for family in ['m2','mixed'] for m in (w/'experiments'/family).rglob('complete.json')}
assert recorded=={current},recorded
dump(e/'new-experiment-source-consistency.json',{'current_implementation_sha256':current,'recorded_new_run_fingerprints':list(recorded),'all124_new_runs_match':True})

state={'branch':git('branch','--show-current'),'head':git('rev-parse','HEAD'),'shallow':git('rev-parse','--is-shallow-repository'),'baseline_ancestor':subprocess.run(['git','merge-base','--is-ancestor','94d545a1a18e20ea8763a6760194589b8a3c9065','HEAD'],cwd=repo).returncode==0,'status':git('status','--short'),'final_commit_created':False,'remote_push_performed':False,'patch_scope':'implementation, tests, new study/analysis tools and README; dated deliverables supplied separately'}
assert state['shallow']=='false' and state['baseline_ancestor'];dump(e/'final-repository-state.json',state)
(d/'FINAL_CHANGELOG.md').write_text(f'''# Final changelog — 2026-11-24

- Integration branch: `{state['branch']}`, base `{state['head']}` plus the included working-tree source patch. No final commit or remote push was made.
- Recovered full original history and verified the assignment baseline as an ancestor. Fetched upstream main was inspected but not merged into the course branch. No reset, rebase or history rewrite occurred.
- Repaired optional workload labels across multiprocessing client/RPC/backend submission; retained unlabeled compatibility. Added real API/block-label checks and pickle/backend regression tests.
- Resolved the heterogeneous study's B905 warning with strict zip semantics; avoided unrelated legacy refactoring.
- Executed the applicable simulator suite: **146 passed, 0 failed, 68.10 s**. Additional isolated eviction-counter checks passed; historical setup/development failures are preserved and qualified.
- Completed M2: **108 configurations, {tot['m2']['requests']:,} requests, zero failures**, three arrival seeds and matched200-conversation modes. Reused overlapping M2-A/M2-B cells and the pilot.
- Revalidated/reused Task1: **51 configurations, {tot['task1']['requests']:,} requests, zero failures**. Generated complete means/SD/gaps, representative cross-source CDFs and all-eight-source baseline CDF supplements.
- Completed new mixed evaluation: **16 configurations, {tot['mixed']['requests']:,} requests, zero failures**. Two compositions, all eight policies, one seed each, randomized order, fresh caches and measured allocator eviction counts. Preserved the interrupted first LFU attempt and retried separately.
- Added immutable numbered attempts, source/input/control fingerprints, raw-to-table checks, policy-independent planned-prompt diagnostics, portable figure regeneration and explicit historical null counters.
- Produced the official-style report ({qa['documents']['report']['main_pages']} main pages; references/appendices separate), editable24-slide presentation (15 main/9 backup), native Impress PDF, English script/Q&A and TA guide from the same validated table set.
- Preserved original reports/slides, datasets, raw studies and Git history. Large experimental evidence is supplied separately from source; model weights/caches are excluded.

## Remaining limits

Mixed repetition/optional ablations, promotion/protection counts, fresh-OS installation, GPU numerical/performance validation and completed team review/rehearsal are not claimed. Production windows are limited; labels are dataset supplied. Raw CPU replay outcomes do not establish GPU speedups or a universally optimal policy. See the requirements checklist and protocol for scope.

New completed experiment lifecycle runtime: **{newminutes:.2f} min**; exact family totals are in `evidence/runtime.json`. This excludes data preparation, tests, interrupted attempts and document rendering.
''')
(d/'REQUIREMENTS_COMPLETION.md').write_text(f'''# Requirements-to-evidence checklist

Status applies to the supplied final source and raw-validated artifacts. "Completed and verified" is scoped to the listed evidence, not every upstream GPU feature.

| Requirement | Status | Evidence and limits |
|---|---|---|
| M1 CPU vLLM v0 online simulator | Completed and verified | Active scheduler/sequence/block path, trace executor, engine/API tests; unsupported extensions are documented |
| M2 Task1 causal client simulator | Completed and verified | Preparation, exact request IDs/tokens, causal replay, timing/failure tests and new matched workloads |
| M2 Task2 single/multi ShareGPT | Completed and verified | Matched200 conversations,200/608 requests per cell; LRU/block16 comparisons reuse the sensitivity sweep |
| M2 Task3 parameter sensitivity | Completed and verified |36 cells ×3 arrival seeds =108 runs; .05/.10GiB, blocks8/16/32, LRU/LFU/FIFO; bytes approximately fixed across blocks |
| M3 Task1 separate heterogeneous sources | Completed and verified |51 reused raw runs, eight sources, {tot['task1']['requests']:,} requests; mean/SD/n/best gaps and complete source CDF supplement |
| M3 Task2 mixed policies | Completed and verified (required comparison) |16 new runs, two mixtures ×8 policies, seed0; identical within-mixture input/capacity; n=1 uncertainty unknown |
| Correctness and former multiprocessing issue | Completed and verified |146/0 applicable tests; optional label pickle/backend and both HTTP/frontend checks; B905 targeted lint passes |
| Experimental arithmetic/cohort integrity | Completed and verified |175 configurations; complete IDs, labels, causal order, hashes, counts, finite/bounded metrics and common policy controls checked |
| Reproducibility | Completed and verified within observed environment | Exact commands, versions, hashes, raw logs, resumable driver, tested smoke path and portable figure input resolution; fresh OS installation not repeated |
| Technical report | Completed and verified |Official MLSys2025 style, {qa['documents']['report']['main_pages']} main pages, confirmed authors/order/emails; compiled and all pages rendered/visually checked |
| AI disclosure and prompts | Completed and verified |Dedicated report section and appendix; code/test/script/figure assistance disclosed, no inferred human review |
| Final presentation | Completed and verified |15 main +9 backup slides; native editable PPTX and Impress PDF; all slides rendered/visually checked |
| Speaker notes/Q&A/timing | Completed and verified as generated artifacts |{qa['timing']['spoken_words']} spoken words including transitions,810-second plan,20 Q&A; timing is an estimate, not a human rehearsal |
| Team contribution/history evidence | Completed and verified within available history |Full original project history, baseline ancestry and diffs; confirmed identities/order; no effort percentages or presenter split |
| Mixed three-seed repetitions | Partially completed |One complete seed per mixture; extra repetitions not run ahead of required families; no mixed SD/CI |
| Optional robustness/ablations | Missing (optional) |No window/threshold/hold/cache-ratio ablation claimed |
| Protection/promotion counters | Missing (optional if unavailable) |New mixed evictions measured; promotion counters absent. M2/old Task1 evictions null |
| Human review and rehearsal | Not verified |Team must confirm understanding/review and rehearse in the presentation environment |

No required comparison remains blocked. Original artifacts are historical snapshots; the dated final tree supersedes their older evidence gaps only where the new logs support it. No remote push occurred.
''')
(d/'FIGURE_PROVENANCE.md').write_text('''# Figure provenance and populations

Every numerical figure comes from `tools/comp536_final_figures.py` and the raw inputs hashed in `figures/validation.json`. No chart uses the historical guide's mixed reference values as new measurements. Tables are generated from the same validated summaries.

| Figure(s) | Inputs/population |
|---|---|
| `m2_single_multi` | Matched M2 LRU/block16; .05/.10GiB; means/sample SD over arrival seeds0/1/2 |
| `m2_sensitivity` | Capacity at block16 and block size at .05GiB; both modes, three baseline policies, three seeds |
| `m2_cdf_request/blocks/gaps` | Representative M2 LRU/block16/.05GiB/seed0; requests, logical full blocks including zero hits, and successful-hit-associated wall time respectively |
| `m2_planned_prompt_recurrence` | Prepared seed0 prompts, nominal arrival order, block16; no decode blocks/capacity/availability model; not a serving upper bound |
| `m3_task1` | All eight sources; means and sample SD where repeated; n=1 uncertainty unknown |
| `m3_cdf_request/blocks/gaps` | Representative LRU/seed0 chat, agent, coding-agent and quality; source contexts/capacities differ |
| `task1_by_source/*` | Complete eight-source CDF supplement, each comparing LRU/LFU/FIFO at seed0; same three populations as above |
| `mixed_overall` | Both newly executed mixtures, all eight policies; one seed each; no SD/error bars |
| `mixed_by_workload` | Per-label token-weighted scores from the exact same complete mixed cohorts |
| `mixed_composition` | Request shares and prompt-token shares; causal balanced-prefix overshoot preserved |
| `system_prefix`, `workflow` | Implementation-grounded schematic, generated by `tools/comp536_diagrams.py`; not empirical measurements |

PDF plots retain vector paths/text; PNGs are high resolution; diagrams also include editable SVG text. Native PowerPoint diagrams/text/tables remain editable. Policy colors are consistent in policy-comparison plots. Source-specific trace transformations and real CPU wall-clock effects are documented in the report/protocol. Figure generation does not start a server or rewrite raw studies.
''')
print(json.dumps({'counts':tot,'runtime':runtime,'documents':qa['documents']}))

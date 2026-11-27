from pathlib import Path
import json,shutil,subprocess,hashlib,zipfile
R=Path(__file__).resolve().parents[2];repo=R/'work/review-vllm';D=repo/'deliverables/final-2026-11-27';W=R/'work/update-2026-11-27';e=D/'evidence';out=R/'outputs/final-2026-11-27';out.mkdir(parents=True,exist_ok=True)
qa=json.loads((W/'document-qa/document-checks.json').read_text());assert qa['documents']['report']['main_pages']==6 and qa['documents']['slides']['pages']==28 and not qa['layout_estimates'];assert qa['timing']['spoken_words']==1589
for name in ['full-tests.log','report-build.log','presentation-build.log','presentation-export.log','document-qa.log']:
 shutil.copy2(W/'evidence'/name,e/name)
shutil.copy2(W/'document-qa/document-checks.json',e/'document-checks.json')
for p in (W/'document-qa').glob('*-contact-*.png'):shutil.copy2(p,e/p.name)
shutil.copy2(W/'validate_documents.py',e/'validate_documents.py')
shutil.copy2(W/'extract_reported_sweep.py',e/'extract_reported_sweep.py')
# Store build versions without exposing environment credentials.
version=subprocess.run([str(R/'work/final-2026-11-24/office.sh'),'--version'],capture_output=True,text=True);(e/'office-version.log').write_text(version.stdout+version.stderr)
(D/'presentation/VALIDATION.md').write_text('''# November 27 presentation validation

The actual final PPTX was reopened with python-pptx. ZIP CRC and all XML/relationships parsed; confirmed author metadata and notes are present. There are 28 slides:15 timed main slides and13 backups, with766 editable text shapes. Native editable shapes/tables/text are retained; plots are images backed by vector figures and sources.

LibreOffice Impress24.2.7.2 exported the final PPTX itself to the required presentation PDF (28 pages; Creator Impress). The separately named shared-source Layout Companion is not the native export and is excluded from the compact delivery.

All28 native-export pages were rendered and visually inspected using three contact sheets, with detailed inspection of the new sweep and complete-table pages. No out-of-slide shapes or estimated text-height issues remain. Main notes contain1,589 spoken words including transitions and total810 planned seconds (13min30s); this is an estimate, not a human rehearsal.

All historical numerical results are identified separately from the new reported rounded sweep. Team PowerPoint rendering, human review and rehearsal remain unverified. Tests (153/0) apply to ad1d826542, not proof of125 completed benchmark replays. See current evidence/document-checks.json.
''')
(e/'final-document-validation.md').write_text('''# November 27 document inspection

The report uses the unchanged official supplied MLSys2025 style, confirmed author order/emails and Rice affiliation. It compiles with Tectonic0.17.0: six main pages, references beginning on page7,15 total pages including appendices. No overfull boxes or unresolved citations/references remain.

Every report page1–15 and native Impress presentation page1–28 was rendered and visually inspected. Dense tables, the main sweep heatmap, complete backup tables and the new figure/caption were also inspected in detailed page images. No clipping or overlaps were observed. The independent slide geometry/font-height checker reports zero issues. The deck contains15 main slides,13 backups,1,589 spoken words and810 planned seconds.

The 125-cell sweep is extracted from rounded upstream guide values and explicitly source-attributed. Its raw completion, datasets and runtime remain unverified. Historical175-run checks remain historical. Automated artifact inspection does not establish student review, rehearsal or GPU validation.
''')
# Record the exact source actually tested and the source-only changes reviewed.
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
(e/'current-source-review.json').write_text(json.dumps({'head':head,'branch':subprocess.check_output(['git','branch','--show-current'],cwd=repo,text=True).strip(),'reviewed_commit':'ad1d8265420c2515882300710d1cdf5d375515ba','previous_deliverables_commit':'f5d84b66cd2b522cd3c009471efc1a86497e14dd','simulator_source_changed_by_document_update':False,'benchmarks_executed_in_update':0,'test_passed':153,'test_failed':0,'new_commit_or_push_in_update':False,'review_findings':['hold_grid.run_succeeded does not verify expected request count/IDs or source fingerprints; partial successful prefixes can be resumed as complete','Fixed settings order; one seed/cell; rounded source table; no new raw logs/manifests available','Changed reply preparation means old traces are not current-code reruns'],'documents':qa['documents']},indent=2)+'\n')
(D/'AI_PROMPTS.md').write_text((D/'AI_PROMPTS.md').read_text().split('\n## November 27 update',1)[0]+'''\n## November 27 update\n\nThe user's new request, translated from Traditional Chinese, was: “There are new commits; help me review them and update the PPT and report.” Codex reviewed ad1d826542, ran the full simulator suite, extracted and source-attributed rounded guide results, regenerated plots/English report/slides/notes and performed artifact/rendering checks. No new benchmark completion or human review is inferred.\n''')
# Build inputs include both machine-readable current provenance and clearly labeled inherited evidence.
def files(root):return sorted(p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ['.pyc','.aux','.bbl','.blg'] and p.name not in ['main.pdf','main.log','Prefix_Sharing_Layout_Companion.pdf'])
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
for folder,name in [('report','Prefix_Sharing_Final_Report.pdf'),('presentation','Prefix_Sharing_Final_Presentation.pptx'),('presentation','Prefix_Sharing_Final_Presentation.pdf'),('presentation','Presentation_Speaker_Notes.md'),('presentation','Presentation_QA.md')]:shutil.copy2(D/folder/name,out/name)
for name in ['README.md','FINAL_CHANGELOG.md','FILE_INDEX.md']:shutil.copy2(D/name,out/name)
shutil.copy2(Path(__file__),e/'finalize_update.py')
(e/'current-file-hashes.json').write_text(json.dumps({str(p.relative_to(D)):sha(p) for p in files(D) if p.name!='current-file-hashes.json'},indent=2)+'\n')
def archive(path,items):
 with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for p,n in items:z.write(p,str(n))
 with zipfile.ZipFile(path) as z:assert z.testzip() is None
archive(out/'COMP536_Updated_Report_Presentation.zip',[(p,Path('final-2026-11-27')/p.relative_to(D)) for p in files(D)])
archive(out/'Prefix_Sharing_Updated_LaTeX.zip',[(p,Path('report')/p.relative_to(D/'report')) for p in files(D/'report') if p.suffix in ['.tex','.bib','.sty','.bst','.py','.md'] or 'figures' in p.parts])
# Test standalone compilation from the extracted LaTeX archive, not just the working directory.
latexcheck=W/'latex-extraction-check';latexcheck.mkdir(exist_ok=True)
with zipfile.ZipFile(out/'Prefix_Sharing_Updated_LaTeX.zip') as z:z.extractall(latexcheck)
command=[str(R/'work/report-build/tectonic'),'--keep-logs','main.tex']
import os
env=dict(os.environ,XDG_CACHE_HOME=str(R/'work/final-2026-11-24/tex-cache'))
res=subprocess.run(command,cwd=latexcheck/'report',env=env,capture_output=True,text=True);(W/'evidence/latex-extraction-check.log').write_text(res.stdout+res.stderr);assert res.returncode==0
manifest={'source_commit':head,'updated_on':'2026-11-27','new_benchmarks_executed':0,'tests':{'passed':153,'failed':0,'seconds':73.93},'documents':qa['documents'],'slides':qa['timing'],'archive_crc':'passed','standalone_extracted_latex_compile':'passed','remote_push':False,'artifacts':[{'file':p.name,'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(out.iterdir()) if p.is_file() and p.name!='ARTIFACTS.json']};(out/'ARTIFACTS.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'output':str(out),'report':qa['documents']['report'],'slides':qa['documents']['slides'],'archives':'CRC and extracted LaTeX compile passed'},indent=2))

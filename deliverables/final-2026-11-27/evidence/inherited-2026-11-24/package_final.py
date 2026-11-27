"""Package final documents, exact source with Git history, and raw evidence separately."""
from pathlib import Path
import json,hashlib,subprocess,shutil,zipfile,tarfile,io
R=Path(__file__).resolve().parents[2];repo=R/'work/review-vllm';d=repo/'deliverables/final-2026-11-24';out=R/'outputs/final-2026-11-24';out.mkdir(exist_ok=True)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def regular(root):return sorted(p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc')
def archive(path,items):
 assert not path.exists(),f'Choose a fresh archive name: {path}'
 with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
  for p,name in items:
   typ=zipfile.ZIP_STORED if p.suffix.lower() in ['.pack','.idx','.png','.pdf','.parquet','.zip'] else zipfile.ZIP_DEFLATED
   z.write(p,str(name),compress_type=typ,compresslevel=1)
 with zipfile.ZipFile(path) as z:assert z.testzip() is None
 print(path.name,path.stat().st_size,flush=True)
# All final checks must exist before packaging.
qa=json.loads((d/'evidence/document-checks.json').read_text());assert qa['documents']['report']['main_pages']<=6 and qa['documents']['slides']['pages']==24
subprocess.run(['git','diff','--check'],cwd=repo,check=True)
subprocess.run(['git','fsck','--connectivity-only'],cwd=repo,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
shutil.copy2(repo/'README.md',d/'README.md')
for family,name in [('report','Prefix_Sharing_Final_Report.pdf'),('presentation','Prefix_Sharing_Final_Presentation.pptx'),('presentation','Prefix_Sharing_Final_Presentation.pdf'),('presentation','Presentation_Speaker_Notes.md'),('presentation','Presentation_QA.md')]:shutil.copy2(d/family/name,out/name)
shutil.copy2(repo/'README.md',out/'README.md');shutil.copy2(d/'evidence/final-source-changes.patch',out/'Final_Source_Changes.patch')
# Portable, relative archive layouts; original absolute paths remain provenance only.
raw_roots=[(R/'work/rerun-2026-11-23/task1','raw/task1'),(R/'work/final-2026-11-24/experiments/m2','raw/m2'),(R/'work/final-2026-11-24/experiments/mixed','raw/mixed'),(R/'work/rerun-2026-11-23/data','raw/source-data'),(R/'work/rerun-2026-11-23/task2-skewed/prepared','raw/source-prepared/skewed'),(R/'work/final-2026-11-24/experiments/mixed-expanded','raw/source-prepared/expanded')]
raw_items=[(p,Path(prefix)/p.relative_to(root)) for root,prefix in raw_roots for p in regular(root)]
raw_index=[{'archive_path':str(name),'original_path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)} for p,name in raw_items]
(d/'evidence/raw-archive-index.json').write_text(json.dumps(raw_index,indent=2))
(d/'FILE_INDEX.md').write_text('''# Final artifact index

Primary standalone files in `outputs/final-2026-11-24`:

- `Prefix_Sharing_Final_Report.pdf`: final official-style paper.
- `Prefix_Sharing_Final_Presentation.pptx` and `.pdf`: editable slide deck and native Impress export.
- `Presentation_Speaker_Notes.md`, `Presentation_QA.md`, `README.md`: English script, Q&A and complete TA guide.
- `Final_Source_Changes.patch`: implementation/tests/tools/README patch against base3d501ab; final dated documents are supplied separately. The code archive already includes the patch's changes: do not apply twice.
- `COMP536_Final_Deliverables.zip`: compact report, slides, sources, figures/tables and validation records.
- `Prefix_Sharing_Final_LaTeX.zip`: complete source/styles/bibliography/vector figures to compile `report/main.tex`.
- `COMP536_Code_with_Git.zip`: current source and original `.git` history; no virtual environment, model cache or large dataset.
- `COMP536_Raw_Experiment_Evidence.zip`: all three study families, prepared cohorts/source inputs and retained incomplete attempt. Only completed markers count in the new studies.
- `ARTIFACTS.json`: archive/standalone SHA256, exact paths, sizes and raw-source locations.

Within the dated deliverable tree, `figures` contains validated CSV/JSON, vector PDFs/high-resolution PNGs, editable SVG diagrams and the all-eight-source CDF supplement. `evidence/validation.json` is not the data validator: use `figures/validation.json`; document/test/source checks live separately in `evidence`. Historical artifacts outside the dated tree remain preserved snapshots.

After extracting the raw archive, use `raw/m2`, `raw/mixed` and `raw/task1` as the figure generator's three inputs. Adjacent prepared cohorts resolve portably without altering absolute provenance paths. Source data are included for exact input reconstruction; model/tokenizer metadata must be downloaded separately from the recorded public revisions. Raw archive entries/hashes are listed in `evidence/raw-archive-index.json`.
''')
# Keep exact packaging method with evidence; no recursive self-hash.
shutil.copy2(Path(__file__),d/'evidence/package_final.py');shutil.copy2(R/'work/final-2026-11-24/finalize_evidence.py',d/'evidence/finalize_evidence.py')
(d/'evidence/deliverable-file-hashes.json').write_text(json.dumps({str(p.relative_to(d)):sha(p) for p in regular(d) if p.name!='deliverable-file-hashes.json'},indent=2))
archive(out/'COMP536_Raw_Experiment_Evidence.zip',raw_items)
source_names=subprocess.check_output(['git','ls-files'],cwd=repo,text=True).splitlines()
source=set(repo/n for n in source_names if (repo/n).is_file())
source.update(repo/n for n in ['tests/simulator/test_rpc_workload.py','tools/comp536_final_study.py','tools/comp536_final_figures.py','tools/comp536_diagrams.py','tools/comp536_prompt_recurrence.py','tools/comp536_counting/sitecustomize.py'])
source.update(regular(repo/'deliverables'));source.update(regular(repo/'.git'))
archive(out/'COMP536_Code_with_Git.zip',[(p,Path('vllm')/p.relative_to(repo)) for p in sorted(source)])
archive(out/'COMP536_Final_Deliverables.zip',[(p,Path('final-2026-11-24')/p.relative_to(d)) for p in regular(d)])
latex=[p for p in regular(d/'report') if p.suffix in ['.tex','.sty','.bst','.bib','.md','.py'] or 'figures' in p.parts]
archive(out/'Prefix_Sharing_Final_LaTeX.zip',[(p,Path('report')/p.relative_to(d/'report')) for p in latex])
# Verify an independently extracted code/history bundle and patch applicability.
check=R/'work/final-2026-11-24/package-check';check.mkdir(exist_ok=False)
with zipfile.ZipFile(out/'COMP536_Code_with_Git.zip') as z:z.extractall(check)
subprocess.run(['git','fsck','--connectivity-only'],cwd=check/'vllm',check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
subprocess.run(['git','apply','--reverse','--check',str(out/'Final_Source_Changes.patch')],cwd=check/'vllm',check=True)
patchcheck=R/'work/final-2026-11-24/patch-check';patchcheck.mkdir(exist_ok=False)
blob=subprocess.check_output(['git','archive','HEAD'],cwd=repo)
with tarfile.open(fileobj=io.BytesIO(blob)) as tar:tar.extractall(patchcheck,filter='data')
subprocess.run(['git','init','--quiet',str(patchcheck)],check=True)
subprocess.run(['git','apply','--check',str(out/'Final_Source_Changes.patch')],cwd=patchcheck,check=True)
shutil.copy2(d/'FILE_INDEX.md',out/'FILE_INDEX.md')
manifest={'base_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),'branch':'comp536-sharing','final_state':'base plus working-tree source patch; not committed or pushed','artifacts':[{'file':p.name,'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(out.iterdir()) if p.is_file() and p.name!='ARTIFACTS.json'],'raw_source_roots':[{'path':str(root),'archive_prefix':prefix} for root,prefix in raw_roots],'archive_crc_checks':'all passed','extracted_git_connectivity':'passed','patch_forward_on_clean_base':'passed','patch_reverse_on_submitted_source':'passed'}
(out/'ARTIFACTS.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps({'packaged':len(manifest['artifacts']),'checks':'passed'}))

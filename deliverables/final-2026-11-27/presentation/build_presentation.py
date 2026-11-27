"""Build editable slides and a companion layout preview (not a PowerPoint render).
Document-only tools: python-pptx 1.0.2, reportlab 5.1.0; numpy/matplotlib.
Run from repository root. No benchmarks or modifications to source results.
"""
from pathlib import Path
import json,textwrap,re,math
from datetime import datetime,timezone
from pptx import Presentation
from pptx.util import Inches,Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE,MSO_CONNECTOR
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE,XL_LEGEND_POSITION
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
D=Path(__file__).resolve().parent; F=D.parent/'figures'; prs=Presentation();prs.slide_width=Inches(13.333);prs.slide_height=Inches(7.5)
prs.core_properties.title='A Study of Prefix Sharing in LLM Serving'
prs.core_properties.author='Thomas Lin; Yu-Ting Chung; Patrick Chiou; Yu-Xin Lai'
prs.core_properties.subject='Rice University COMP536 final project'
prs.core_properties.keywords='prefix caching, vLLM, CPU trace simulation'
prs.core_properties.last_modified_by='COMP536 project team; AI-assisted preparation'
prs.core_properties.created=prs.core_properties.modified=datetime.now(timezone.utc)
BG='F6F8FB';NAVY='162B42';BLUE='167DAD';TEAL='15977B';ORANGE='A55B19';GRAY='52677C'
pdfmetrics.registerFont(TTFont('DejaVu','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))
pdfmetrics.registerFont(TTFont('DejaVuBold','/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'))
c=canvas.Canvas(str(D/'Prefix_Sharing_Layout_Companion.pdf'),pagesize=(960,540)); c.setTitle('A Study of Prefix Sharing in LLM Serving — Final Presentation'); c.setAuthor(prs.core_properties.author); shapes=[];notes=[]
def rgb(h):return RGBColor.from_string(h)
def text(s,x,y,w,h,size=22,color=NAVY,bold=False):
 box=sl.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h));tf=box.text_frame;tf.word_wrap=True;tf.margin_left=tf.margin_right=Inches(.015);tf.margin_top=tf.margin_bottom=0
 for i,line in enumerate(s.split('\n')):
  p=tf.paragraphs[0] if i==0 else tf.add_paragraph();p.text=line;p.font.name='DejaVu Sans';p.font.size=Pt(size);p.font.bold=bold;p.font.color.rgb=rgb(color);p.space_after=Pt(size*.3)
 # Same geometry, font and text for the companion; renderer wrapping may differ.
 yy=540-y*72-size
 c.setFillColorRGB(*[int(color[i:i+2],16)/255 for i in (0,2,4)]);font='DejaVuBold' if bold else 'DejaVu';c.setFont(font,size)
 for para in s.split('\n'):
  line=''
  for word in para.split():
   trial=(line+' '+word).strip()
   if pdfmetrics.stringWidth(trial,font,size)>w*72-4 and line:c.drawString(x*72,yy,line);yy-=size*1.18;line=word
   else:line=trial
  c.drawString(x*72,yy,line);yy-=size*1.4
 shapes.append({'slide':len(prs.slides),'kind':'text','x':x,'y':y,'w':w,'h':h,'text':s,'font':size,'bold':bold})
 return box
def rect(x,y,w,h,fill='E9F2F8'):
 sh=sl.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,Inches(x),Inches(y),Inches(w),Inches(h));sh.fill.solid();sh.fill.fore_color.rgb=rgb(fill);sh.line.fill.background()
 c.setFillColorRGB(*[int(fill[i:i+2],16)/255 for i in (0,2,4)]);c.roundRect(x*72,540-(y+h)*72,w*72,h*72,8,fill=1,stroke=0)
 return sh
def arrow(x,y,x2,y2):
 sh=sl.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,Inches(x),Inches(y),Inches(x2),Inches(y2));sh.line.color.rgb=rgb(BLUE);sh.line.width=Pt(2)
 c.setStrokeColorRGB(.086,.49,.68);c.setLineWidth(2);c.line(x*72,540-y*72,x2*72,540-y2*72)
 # Visible arrowhead as an editable triangle.
 sh=sl.shapes.add_shape(MSO_SHAPE.CHEVRON,Inches(x2-.08),Inches(y2-.08),Inches(.16),Inches(.16));sh.fill.solid();sh.fill.fore_color.rgb=rgb(BLUE);sh.line.fill.background();sh.rotation=math.degrees(math.atan2(y2-y,x2-x))
def image(path,x,y,w,h):
 from PIL import Image
 im=Image.open(path);ratio=im.width/im.height
 if w/h>ratio:
  ww=h*ratio;x+=(w-ww)/2;w=ww
 else:
  hh=w/ratio;y+=(h-hh)/2;h=hh
 sl.shapes.add_picture(str(path),Inches(x),Inches(y),width=Inches(w),height=Inches(h));c.drawImage(ImageReader(str(path)),x*72,540-(y+h)*72,w*72,h*72,mask='auto')
def start(title,kicker='PREFIX SHARING / COMP536'):
 global sl
 sl=prs.slides.add_slide(prs.slide_layouts[6]);sl.background.fill.solid();sl.background.fill.fore_color.rgb=rgb(BG)
 c.setFillColorRGB(.965,.973,.985);c.rect(0,0,960,540,fill=1,stroke=0)
 text(kicker,.55,.25,12,.3,11,BLUE,True);text(title,.55,.76,12.2,1.2,30,bold=True)
 text('Rice University · COMP536',.55,7.08,11,.22,10,GRAY);text(f'{len(prs.slides):02d}',12.15,7.03,.6,.3,13,BLUE,True)
def end(message,script,seconds,transition):
 notes.append({'slide':len(prs.slides),'title':sl.shapes[1].text,'message':message,'script':script,'seconds':seconds,'transition':transition})
 sl.notes_slide.notes_text_frame.text=f'Main message: {message}\nEstimated time: {seconds} s\n\n{script}\n\nTransition: {transition}'
 c.showPage()
def card(x,y,w,title,body,color=BLUE):
 rect(x,y,w,2.2);text(title,x+.2,y+.18,w-.4,.7,21,color,True);text(body,x+.2,y+.98,w-.4,1.1,17)
# Results are generated only after required completed-run checks succeed.
results=json.loads((F/'validation.json').read_text())
assert results['totals']['m2']['runs']==108 and results['totals']['mixed']['runs']==16
m2=json.loads((F/'m2-means.json').read_text());best=json.loads((F/'mixed-best.json').read_text())
def cell(mode,cap=.05,block=16,policy='lru'):
 return next(r for r in m2 if r['mode']==mode and r['cache_gb']==cap and r['block_size']==block and r['policy']==policy)
def score(mode,cap=.05):return f"{cell(mode,cap)['mean']*100:.2f}%"
def bestline(m):
 r=next(x for x in best if x['mixture']==m)
 return f"{r['best_observed'].upper()} {r['best_fraction']*100:.2f}% · {r['gain_pp']:+.2f} pp vs {r['strongest_baseline'].upper()}"
def bestspoken(m):
 r=next(x for x in best if x['mixture']==m)
 margin=(f"{r['gain_pp']:.2f} percentage points above {r['strongest_baseline'].upper()}" if r['gain_pp']>0 else 'matching the strongest baseline')
 return f"{r['best_observed'].upper()} reaches {r['best_fraction']*100:.2f} percent on {m}, {margin}"
def rows(items,y=1.95):
 for i,(a,b) in enumerate(items):
  yy=y+i*1.07;rect(.65,yy,12.05,.9);text(a,.85,yy+.14,3.5,.65,20,BLUE,True);text(b,4.5,yy+.14,7.95,.68,18)
start('A Study of Prefix Sharing\nin LLM Serving','RICE UNIVERSITY / COMP536')
text('Thomas Lin · Yu-Ting Chung · Patrick Chiou · Yu-Xin Lai',.65,2.65,12,.65,23,BLUE,True)
text('tl183@rice.edu · yc223@rice.edu · pc82@rice.edu · yl371@rice.edu',.65,3.45,12,.7,17,GRAY)
text('CPU trace simulation · homogeneous, heterogeneous and mixed workloads',.65,4.75,11.9,.9,24)
end('Study cache reuse through the serving control path.',"We study prefix sharing in LLM serving using a CPU trace simulator integrated with vLLM. Our team is Thomas Lin, Yu-Ting Chung, Patrick Chiou and Yu-Xin Lai at Rice University. We examine homogeneous chat, heterogeneous sources and mixed-workload eviction.",20,'First, why should workload structure affect cache management?')
start('Locality makes eviction a workload-dependent decision')
card(.65,1.95,3.85,'Growing histories','Recent conversation state may be valuable at the next turn.')
card(4.72,1.95,3.85,'Repeated instructions','Popular prefixes may recur across otherwise unrelated requests.')
card(8.79,1.95,3.85,'Document bursts','A large prefix may be useful briefly, then become cold.')
text('RQ1  Retain online serving behavior without GPUs?\nRQ2  How do capacity and block size change realized reuse?\nRQ3  Which policy helps each source and workload mixture?',.75,4.6,11.8,1.8,23)
end('Different return patterns motivate different retention rules.',"Prefill constructs attention state for an input prompt. If a later request starts with the same full prefix, it can reuse those KV blocks instead of computing them again. Limited capacity makes retention important: a conversation grows, a common instruction repeats across sessions, and a document may receive a short burst of questions. We ask whether CPU replay can preserve the serving control path, how cache capacity and block granularity affect realized sharing, and whether any baseline or custom policy is robust across workload mixtures. ",50,'The implementation and final evaluation combine several team contributions.')
start('Team contributions')
rows([('Thomas Lin','Mixed workloads/policies; new RFIFO controls, stop-token fix and hold sweep.'),('Yu-Ting Chung','Simulator integration/optimization; heterogeneous adapters and traces.'),('Patrick Chiou','Initial Task1; final report/slides, validation, bug fix and result integration.'),('Yu-Xin Lai','Client timing/templates, M2 replay/metrics, baseline policies and sweeps.')])
text('Based on complete project history and confirmed responsibilities. No contribution percentages or presenter assignments.',.75,6.45,11.8,.6,16,GRAY)
end('The final artifacts integrate implementation, experiments and validation.',"Thomas worked on mixed workloads, metadata transport and custom policies, then added admission controls and the reported hold sweep. Yu-Ting implemented and optimized the simulator and expanded heterogeneous adapters and traces. Patrick contributed the initial Task1 framework and the final AI-assisted report, presentation, testing, bug repair and result integration. Yu-Xin implemented client timing and templates, homogeneous replay and measurements, baseline policies and sweep automation.  ",40,'Here is the completed project state supported by the available evidence.')
start('Project progress: all required experiment families covered')
rows([('Milestone 1','CPU online simulator; direct and multiprocessing API tests pass.'),('Milestone 2','108 matched-mode / cache-parameter configurations, three seeds.'),('Milestone 3 Task 1','51 reused configurations across eight sources; 21,834 requests.'),('Milestone 3 Task 2','16 raw-validated runs; plus a reported 125-cell parameter sweep.')])
text('153 tests passed. New sweep: summaries available; raw logs not supplied.',.75,6.45,11.8,.6,19,BLUE,True)
end('Required experiment families are complete; uncertainty remains qualified.',"The simulator now passes both frontend modes. Milestone two contains one hundred eight configurations across three arrival seeds. We reuse the validated fifty-one-run heterogeneous study rather than rerunning it unnecessarily. The historical mixed evaluation has one seed under eight policies. A new commit reports a five-mixture parameter sweep; its raw logs are unavailable. ",30,'The architecture explains which real serving components still execute.')
start('Trace execution replaces the model, not the serving loop')
boxes=['Dataset + TraceBuilder','Causal HTTP replay','OpenAI API / engine','v0 scheduler + batching','Prefix blocks / evictor','SimExecutor + Simulator','EOS / completion','Requests + block metrics']
for i,b in enumerate(boxes):
 x=.65+i%4*3.18;y=1.95+i//4*1.75;rect(x,y,2.85,1.15);text(b,x+.12,y+.18,2.6,.9,19,BLUE,True)
 if i%4<3:arrow(x+2.85,y+.55,x+3.12,y+.55)
arrow(11.7,3.1,1.9,3.65)
text('Retained: scheduling, sequence and prefix-block management.\nReplaced: GPU execution and KV tensors → trace tokens and virtual capacity.',.75,5.6,11.8,1.2,20)
end('The standard control path surrounds a trace-driven executor.',"Read the top row from left to right. Preparation normalizes traces and applies the serving template and tokenizer. The client submits causal HTTP requests to the OpenAI-compatible API and engine. The v0 scheduler decides iteration-level work and continuous batches. The lower row shows prefix allocation and eviction, followed by the simulator executor and normal completion processing. The original model executor would load weights and run GPU kernels; our replacement returns recorded token outputs and uses virtual KV geometry. Sequence state, reference counting, prefix lookup and EOS handling remain active. Workload labels are additional metadata, so we do not claim every scheduler line is untouched. No model weights or real KV tensors are needed, but model configuration and tokenizer files are. Synthetic execution delay and real CPU overhead affect timing, so the system is a cache-behavior simulator rather than a calibrated GPU performance model.",80,'An exact-prefix example makes the sharing boundary concrete.')
start('Sharing requires the same complete preceding prefix')
for row,label in enumerate(['Request A','Request B']):
 yy=2.3+row*1.3;text(label,.75,yy+.1,2.1,.5,23,bold=True)
 for j,v in enumerate(['P₀','P₁','A₂' if row==0 else 'B₂']):
  xx=3.0+j*2.6;rect(xx,yy,2.2,.82,'D1E8F3' if j<2 else 'F6DCBF');text(v,xx+.7,yy+.15,1.1,.5,25,BLUE,True)
text('B hits retained P₀ / P₁; different suffixes are misses',.85,5.1,11.7,.65,24,BLUE,True)
text('LRU: oldest access · LFU: lowest allocator frequency · FIFO: oldest insertion\nEviction chooses unused cached blocks; active referenced blocks are excluded.',.85,6,11.7,.9,19)
end('Chained identity prevents sharing arbitrary matching chunks.',"These two requests begin with the same first two token blocks. Their logical identities include the entire preceding prefix, so those blocks can share KV state. The final blocks differ and must be computed separately. An identical local chunk after a different history would not be the same prefix block. Only full blocks are reusable; partial tails remain outside that matching unit. Under pressure, LRU selects the least recently accessed unused block, LFU the lowest allocator frequency, and FIFO the oldest insertion.  Protected custom-policy tiers also express priority rather than an absolute guarantee of residency.",50,'Preserving these semantics required several specific engineering safeguards.')
start('Engineering: problem → safeguard → validation')
rows([('Ambiguous duplicate prompts','Request-ID plus exact token matching; trace/engine tests.'),('Preemption / recomputation','Select next token from current output length; exact-response tests.'),('Multi-turn causality','Wait for prior reply and think time; replay/timing tests.'),('Multiprocessing labels','Client → pickled RPC → engine → block metadata; both API modes.')])
text('153 passed / 0 failed. New preparation fix removes model-configured stop IDs.',.75,6.45,11.8,.6,19,BLUE,True)
end('The transport fix preserves labels instead of hiding the broken frontend.',"Duplicate prompts can have different recorded responses, so exact tokens alone are insufficient; request IDs disambiguate the trace. Recomputing a sequence must not restart its answer, so next-token selection uses the existing output length. Multi-turn replay must wait for the preceding reply, even if the planned arrival is already past. The final regression involved workload metadata: handlers supplied a label that the multiprocessing client did not accept or serialize. We added it through the client, RPC object and engine-side request submission, keeping an optional default for unlabeled requests. Tests verify both HTTP endpoints and both frontend modes, including request and full-block labels. The new reply builder also strips model-configured stop IDs to prevent early completion. The suite passes one hundred fifty-three tests. ",70,'Next, the experimental controls.')
start('Methodology: common inputs, controlled cache geometry')
card(.65,1.95,3.85,'Homogeneous chat','200 conversations; 2 modes; 2 capacities; 3 block sizes; 3 policies; 3 seeds.')
card(4.72,1.95,3.85,'Separate sources','Eight workload types; block16; ratio .25; concurrency8; 51 runs.')
card(8.79,1.95,3.85,'Mixed workloads','Two compositions; eight policies; ratio .10; fresh cache; one seed.')
text('Primary score = total cached prompt tokens / total prompt tokens',.75,4.65,11.8,.8,25,BLUE,True)
text('Pinned Qwen model/tokenizer · identical within-comparison traces/capacities\nCPU simulation outcomes are not measured GPU latency or throughput.',.75,5.7,11.8,1,20)
end('The score weights long prompts more than short ones.',"We pin the Qwen model configuration and tokenizer. The homogeneous sweep uses matched ShareGPT cohorts and absolute virtual capacities, holding bytes approximately constant when block size changes. Separate sources retain their source-specific contexts and ratio-sized caches. Historical mixed policies share inputs and capacity and run in randomized order. The new sweep driver uses fixed order. Our primary score divides total cached prompt tokens by total prompt tokens; it differs from averaging request fractions. Repeated-seed whiskers are sample standard deviation, while single-run mixed results have no uncertainty estimate. ",45,'First, the matched homogeneous comparison tests whether longer histories actually remain reusable.')
start('M2: multi-turn history does not guarantee realized reuse')
image(F/'m2_single_multi.png',.55,1.7,8.7,4.75)
rect(9.45,1.95,3.25,4.45);text('LRU · block16',9.65,2.2,2.85,.6,21,BLUE,True)
text(f'0.05 GiB\nSingle {score("single")}\nMulti  {score("multi")}\n\n0.10 GiB\nSingle {score("single",.1)}\nMulti  {score("multi",.1)}',9.65,3,2.85,3.2,18)
text('Same 200 eligible conversations and first-turn targets; three arrival seeds. CDFs in backup slides.',.75,6.5,11.8,.5,17,GRAY)
end('Potential history reuse and realized token-weighted sharing are different.',f"The two modes start from the same two hundred eligible conversations and identical first-turn target arrivals. Single-turn sends only the first prompt and response; multi-turn sends the complete causal conversation. Under LRU and sixteen-token blocks, sharing is {score('single')} for single-turn and {score('multi')} for multi-turn at zero point zero five GiB. At zero point one GiB the values are {score('single',.1)} and {score('multi',.1)}. The bars summarize three arrival seeds. Longer histories create potential overlap, but the cache must retain complete blocks while many requests are active. Token weighting also gives long prompts greater influence than short ones. The request, hit-count and hit-associated gap CDFs in backup slides describe these different populations; none is an unconditional potential-reuse distribution.",65,'The parameter sweep separates capacity from full-block matching granularity.')
start('M2: capacity and granularity are separate controls')
image(F/'m2_sensitivity.png',.55,1.65,12.2,4.45)
text('Fixed block16 for capacity; approximately fixed bytes for block size.\nTwo capacity levels do not establish a general saturation curve.',.75,6.25,11.8,.8,20,BLUE,True)
end('Coarser full-block matching can lose short shared prefixes.',"The capacity panel keeps sixteen tokens per block, while the granularity panel keeps the virtual byte budget at zero point zero five GiB. Whole-block rounding means capacities are approximately, rather than bit-for-bit, equal across block sizes. Each point is a three-seed mean with sample standard deviation. A larger block requires a longer identical complete prefix and leaves a longer partial tail that cannot be reused. At eight-token blocks, the single-turn baselines nearly tie; multi-turn policy order differs. The measured cells show why capacity and block size should not be treated as interchangeable controls.  These findings apply to this cohort, load and model template.",55,'Heterogeneous workloads introduce a different range of return patterns.')
start('M3 Task 1: strong effects coexist with near ties')
image(F/'m3_task1.png',.55,1.65,12.2,4.75)
text('Coding agent: LRU 48.59% vs LFU 35.43% (+13.16 pp).\nBailian windows favor LFU; AgentBank / Mooncake leaders nearly tie.',.75,6.25,11.8,.85,20,BLUE,True)
end('Policy rankings are conditional on source and tested settings.',"The separate-source study has fifty-one configurations and twenty-one thousand eight hundred thirty-four successful requests. The largest highlighted gap is coding-agent replay: LRU reaches forty-eight point five nine percent, compared with thirty-five point four three for LFU. Growing contexts plausibly benefit from retaining recently added prefixes, while frequency can favor older popular blocks, but we have not isolated that mechanism with an ablation. LFU has the highest observed scores in the Bailian chat and coder windows, each with only one repetition. QuALITY's highest mean is FIFO. AgentBank's LRU and LFU differ by only about zero point zero seven percentage points, and Mooncake's by about zero point zero two. Code Alpaca ties. These are conditional observations, not statistically established universal winners.",70,'Custom policies try to combine workload knowledge with observed reuse.')
start('Custom policies: labels, own hits or recent outcomes')
rows([('TLRU / TFIFO','Static label tiers + shared tier; recency/8s hold vs FIFO/no expiry.'),('HLRU / HFIFO','Protect after the block’s own first hit; labels not required.'),('RFIFO','Configurable window/threshold; defaults 100 outcomes and 5% hits.'),('When tiers fill','Lower tiers first; protected unused blocks remain evictable.')])
text('Labels come from datasets, not an online classifier. RFIFO does not retroactively promote an existing low block.',.75,6.45,11.8,.6,17,GRAY)
end('Protection is a victim preference, not pinning.',"TLRU and TFIFO use static workload tiers, giving agent and Bailian content preference over one-shot labels and protecting prefixes shared across labels. TLRU uses recency with an eight-second hold; TFIFO uses insertion order without expiry. HLRU and HFIFO instead protect a block after its own reuse, without requiring labels. RFIFO defaults to one hundred resolved outcomes per label; the new commit makes its window and admission threshold configurable. Once that window is full and at least five percent are hits, future blocks can enter its protected FIFO tier. Existing low blocks are not retroactively promoted. In every case, protection changes victim priority rather than absolutely pinning content. If lower tiers are empty, protected unused blocks can still be evicted. ",65,'The two mixtures test whether those choices survive a different token composition.')
start('New sweep: the best setting depends on the mixture')
image(F/'hold_selected.png',.55,1.65,12.2,4.65)
text('5 mixtures × 25 settings · seed 0 · reported rounded aggregates',.75,6.3,11.8,.4,19,BLUE,True)
text('Source: ad1d826542 / Task2 guide; original raw logs unavailable. Not pooled with historical runs.',.75,6.75,11.8,.3,14,GRAY)
end('A broad in-sample mean and a worst-case gap favor different settings.',"The new commit reports one hundred twenty-five cells: five mixtures and twenty-five settings each. It varies TLRU and HLRU hold duration and budget, and RFIFO admission window and threshold. These numbers come from the committed rounded table; the original raw logs were not supplied, so they are not pooled with our earlier validated runs. RFIFO reaches twenty-three point six percent on chat-heavy input. HLRU hold eight, budget one leads protected-heavy at thirty-nine point three. LFU leads template-heavy at forty-seven point six. HLRU hold thirty, budget one leads trace-heavy and balanced, but its trace margin is just zero point one percentage point. Across equally weighted mixtures that HLRU setting has the highest observed mean, while RFIFO window one hundred, threshold two percent has the smallest worst gap. One seed, fixed execution order and in-sample selection limit robustness claims.",80,'The handoff separates directly validated results from reported aggregates.')
start('TA reproduction: a short path before the full sweeps')
text('1  Empty-target editable installation; model config + tokenizer only\n2  Run the simulator correctness suite: 153 passed\n3  Use the small exact-token engine test for a smoke check\n4  Resume required studies with the recorded driver / fresh outputs\n5  Generate figures from existing results without benchmarks',.75,1.95,11.8,3.2,23)
rect(.7,5.4,11.9,1.25);text('python -m pytest tests/simulator -q\npython tools/comp536_final_figures.py --help',.9,5.62,11.5,.9,20,BLUE,True)
end('The README provides tested commands and external raw-evidence paths.',"The README separates a lightweight correctness path from the expensive evaluation. Use Python three point eleven, the documented empty-target editable installation and the pinned model metadata and tokenizer.  The full applicable suite passes one hundred fifty-three tests; a small concurrent exact-token engine test provides a smoke check. The experiment driver records controls and resumes only compatible completed attempts, preserving interrupted outputs. Figure generation reads existing artifacts and does not start a server. Complete commands, expected directories and known limitations are in the README; large datasets and caches are not added to Git.",50,'We conclude with what the evidence supports and what remains limited.')
start('Three takeaways')
card(.65,1.95,3.85,'Control flow matters','Trace replay retains online scheduling and prefix-cache decisions without GPUs.')
card(4.72,1.95,3.85,'Configuration matters','History, byte capacity and full-block granularity jointly determine realized reuse.')
card(8.79,1.95,3.85,'Composition matters','Observed policy rankings vary; mixed single-seed results remain provisional.')
text('Limits: transformed traces; single-seed sweeps; missing new raw logs; no GPU speedup.\nTeam review and rehearsal remain necessary.',.75,4.8,11.8,1.3,22,GRAY)
end('Conditional findings support further controlled evaluation, not a universal winner.',"Three conclusions follow. First, CPU trace execution can retain the serving scheduler and cache-management path while bypassing model computation. Second, homogeneous history does not automatically become realized reuse: byte capacity and block granularity matter. Third, composition and hold or admission parameters change retention value; the reported sweep strengthens this conditional conclusion. The results remain limited by transformed traces, short production windows and one seed for each mixed cohort. We make no measured GPU speedup or universal-optimality claim. ",40,'Questions; backup slides provide detailed evidence.')
# Backup slides are outside the 810-second main presentation allocation.
for title,img,caption in [('Backup: M2 request sharing','m2_cdf_request.png','LRU · block16 · 0.05 GiB · seed0; per-request population.'),('Backup: M2 block hits','m2_cdf_blocks.png','Logical full blocks include zero hits; aggregate across residencies.'),('Backup: M2 realized reuse gaps','m2_cdf_gaps.png','Successful-hit-associated wall time; misses are excluded.'),('Backup: planned prompt recurrence','m2_planned_prompt_recurrence.png','Nominal prompt-only recurrence; no decode blocks or cache/availability model.'),('Backup: mixed workload composition','mixed_composition.png','Token balance is different from equal request counts.'),('Backup: per-workload mixed sharing','mixed_by_workload.png','Inspect category regressions as well as the overall score.')]:
 start(title);image(F/img,.55,1.65,12.2,4.85);text(caption,.75,6.5,11.8,.4,17,GRAY);end(caption,'Backup evidence; not included in the timed main talk.',0,'Return to questions.')
start('Backup: complete Task1 numerical results')
task=json.loads((F/'task1-means.json').read_text())
labels=['chat','agent','coding-agent','quality','bailian-chat','bailian-coder','mooncake-agent','code']
for i,values in enumerate([['Workload','n','LRU (%)','LFU (%)','FIFO (%)']]+[[l,str(next(r['n'] for r in task if r['workload']==l))]+[f"{100*next(r['mean'] for r in task if r['workload']==l and r['policy']==p):.2f}" for p in ['lru','lfu','fifo']] for l in labels]):
 y=1.85+i*.5
 for x,w,v in zip([.65,4.0,4.9,7.5,10.1],[3.25,.8,2.5,2.5,2.5],values):
  rect(x,y,w,.44,'DCECF5' if i==0 else 'ECF1F6');text(v,x+.08,y+.07,w-.16,.32,17,BLUE if i==0 else NAVY,i==0)
text('Means over available arrival seeds; sample SD and best–second gaps are in the report/table CSV.',.75,6.6,11.8,.4,16,GRAY)
end('All eight sources are reported without mixing experimental settings.','Backup table; see the report for uncertainty.',0,'Return to questions.')
start('Backup: complete mixed results and measured evictions')
mixed=json.loads((F/'mixed-runs.json').read_text());policies=['lru','lfu','fifo','tlru','tfifo','hlru','hfifo','rfifo']
values=[['Policy','Skewed (%)','Equal (%)','Evict: skewed','Evict: equal']]
for policy in policies:
 rs=[next(r for r in mixed if r['mixture']==m and r['policy']==policy) for m in ['skewed','equal-token']]
 values.append([policy.upper()]+[f"{100*r['token_weighted_shared_fraction']:.2f}" for r in rs]+[str(r['cache_evictions']) for r in rs])
for i,row in enumerate(values):
 y=1.85+i*.5
 for j,v in enumerate(row):
  x=.65+j*2.42;rect(x,y,2.32,.44,'DCECF5' if i==0 else 'ECF1F6');text(v,x+.08,y+.07,2.16,.32,16,BLUE if i==0 else NAVY,i==0)
text('One seed per mixture. Evictions count successful unused-block allocations, not cache misses.',.75,6.6,11.8,.4,16,GRAY)
end('Counters are directly measured for the new mixed trials.','Backup table; no mixed variability estimate is available.',0,'Return to questions.')
start('Backup: historical raw-validated mixed comparison')
image(F/'mixed_overall.png',.55,1.65,12.2,4.25)
text('Skewed: '+bestline('skewed')+'\nEqual-token: '+bestline('equal-token'),.75,6.1,11.8,.85,19,BLUE,True)
end('Historical 16 runs remain separate from new reported sweeps.','These are the original November 24 raw-validated results, before the new reply preparation. They are not current-commit reruns.',0,'Return to questions.')
start('Backup: robustness depends on the objective')
image(F/'hold_robustness.png',.55,1.65,12.2,4.8)
text('Rounded upstream inputs; five mixtures equally weighted; in-sample regret to best tested setting.',.75,6.6,11.8,.4,16,GRAY)
end('Macro average is not a token-pooled score or held-out validation.','HLRU h30 b1 has the highest mean; RFIFO w100 t0.02 has the smallest worst regret. Original logs and repeated seeds are unavailable.',0,'Return to questions.')
sweep=json.loads((F/'hold-sweep-reported.json').read_text())
for part,entries in enumerate([sweep['rows'][:13],sweep['rows'][13:]],1):
 start(f'Backup: reported sweep settings ({part}/2)')
 values=[['Setting','Chat','Protected','Trace','Template','Balanced']]+[[r['setting'].upper()]+[f"{r[n]:.1f}" for n in sweep['mixtures']] for r in entries]
 for i,row in enumerate(values):
  y=1.8+i*.35
  for x,w,v in zip([.65,5,6.45,7.9,9.35,10.8],[4.25,1.35,1.35,1.35,1.35,1.8],row):
   rect(x,y,w,.31,'DCECF5' if i==0 else 'ECF1F6');text(v,x+.06,y+.045,w-.12,.25,14,BLUE if i==0 else NAVY,i==0)
 text('Percent cached prompt tokens; rounded guide values; one seed per cell. Raw completion not verified.',.75,6.85,11.8,.3,14,GRAY)
 end('Complete 25-setting table from the larger reported sweep.','H and B denote hold seconds and budget; W and T denote RFIFO window and threshold. This table is source-attributed, not raw-validated.',0,'Return to questions.')
start('Backup: AI assistance and validation scope')
text('Codex assisted code changes, tests, experiment scripts, validation, figures,\nreport organization and slide preparation.\n\nAutomated checks do not establish student understanding or human review.\nHistorical AI use outside this conversation remains unconfirmed.\n\nNo GPU model-quality or performance claim is derived from trace replay.',.75,1.95,11.8,4.6,23)
end('AI assistance is disclosed beyond proofreading.', 'Backup disclosure; no presenter assignments or human-review claims are inferred.',0,'Return to questions.')
prs.save(D/'Prefix_Sharing_Final_Presentation.pptx');c.save()
(D/'layout-elements.json').write_text(json.dumps(shapes,indent=2))
main=notes[:15];seconds=sum(n['seconds'] for n in main)
spoken=' '.join(n['script']+' '+n['transition'] for n in main);words=len(re.findall(r"\b[\w’'-]+\b",spoken))
md=f'# Presentation speaker notes\n\n15 main slides, {len(notes)-15} backups. Main allocation: {seconds} s (13 min30 s), leaving90 s before15 min. Spoken words including transitions: {words}. Average planned rate: {words/(seconds/60):.1f} words/min. Estimate, not a timed human rehearsal. No presenter split assigned.\n\n'
for n in notes:md+=f"## Slide {n['slide']} — {n['title']}\n\n**Time:** {n['seconds']} s. **Main message:** {n['message']}\n\n{n['script']}\n\n**Transition:** {n['transition']}\n\n"
(D/'Presentation_Speaker_Notes.md').write_text(md)
(D/'timing-summary.json').write_text(json.dumps({'main_slides':15,'backup_slides':len(notes)-15,'seconds':seconds,'spoken_words':words,'planned_words_per_minute':words/(seconds/60),'native_pptx_render':'Native LibreOffice export is run separately; see presentation/VALIDATION.md. Shared-source layout companion retained for comparison.'},indent=2))
print(json.dumps({'slides':len(notes),'main_seconds':seconds,'spoken_words':words}))

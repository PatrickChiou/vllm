"""Inspect generated documents without changing them; render every PDF page."""
from pathlib import Path
import json, re, subprocess, argparse,zipfile
from lxml import etree
from PIL import Image, ImageOps, ImageDraw
from pptx import Presentation
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

def main():
 p=argparse.ArgumentParser();p.add_argument('deliverables',type=Path);p.add_argument('output',type=Path);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 pdfmetrics.registerFont(TTFont('check','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'));pdfmetrics.registerFont(TTFont('check-bold','/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'))
 d=a.deliverables;slides=d/'presentation/Prefix_Sharing_Final_Presentation.pptx';prs=Presentation(slides)
 assert prs.core_properties.author=='Thomas Lin; Yu-Ting Chung; Patrick Chiou; Yu-Xin Lai'
 with zipfile.ZipFile(slides) as z:
  assert z.testzip() is None
  for name in z.namelist():
   if name.endswith('.xml') or name.endswith('.rels'):etree.fromstring(z.read(name))
 timing=json.loads((d/'presentation/timing-summary.json').read_text());assert len(prs.slides)==28 and timing['main_slides']==15 and timing['seconds']==810 and 1450<=timing['spoken_words']<=1600
 issues=[]
 for i,s in enumerate(prs.slides,1):
  assert s.has_notes_slide and len(s.notes_slide.notes_text_frame.text)>50
  for sh in s.shapes:
   if sh.left<0 or sh.top<0 or sh.left+sh.width>prs.slide_width+10000 or sh.top+sh.height>prs.slide_height+10000:issues.append({'slide':i,'problem':'outside slide','text':getattr(sh,'text','')})
 # Estimate wrapping from the explicit shared layout source; native PDF is inspected separately.
 for q in json.loads((d/'presentation/layout-elements.json').read_text()):
  f=q['font'];height=0
  for para in q['text'].split('\n'):
   line='';lines=1
   for word in para.split():
    trial=(line+' '+word).strip()
    if pdfmetrics.stringWidth(trial,'check-bold' if q.get('bold') else 'check',f)>q['w']*72-3 and line:lines+=1;line=word
    else:line=trial
   height+=lines*f*1.18
  height+=max(0,len(q['text'].split('\n'))-1)*f*.3
  if height>q['h']*72+2:issues.append({'slide':q['slide'],'problem':'estimated text height','height_pt':height,'box_pt':q['h']*72,'text':q['text']})
 docs={}
 for name,path in [('report',d/'report/Prefix_Sharing_Final_Report.pdf'),('slides',d/'presentation/Prefix_Sharing_Final_Presentation.pdf')]:
  info=subprocess.check_output(['pdfinfo',str(path)],text=True);pages=int(re.search(r'Pages:\s+(\d+)',info)[1]);texts=subprocess.check_output(['pdftotext','-layout',str(path),'-'],text=True).split('\f')[:pages]
  if name=='report':
   firstref=next(i+1 for i,t in enumerate(texts) if any(re.sub(r'\s+','',line).lower().startswith('references') for line in t.splitlines()));assert firstref<=7,(firstref,'main exceeds six pages')
   for text in ['Thomas Lin','Yu-Ting Chung','Patrick Chiou','Yu-Xin Lai','tl183@rice.edu','yc223@rice.edu','pc82@rice.edu','yl371@rice.edu','Rice University']:assert text in texts[0],text
   assert not any('??' in t or 'AUTHORERR' in t for t in texts)
   log=(d/'report/main.log').read_text()
   assert 'Overfull' not in log and not re.search(r'Citation .*undefined|undefined references|undefined citations',log,re.I)
   docs[name]={'pages':pages,'main_pages':firstref-1,'first_reference_page':firstref}
  else:assert pages==len(prs.slides);docs[name]={'pages':pages,'creator':re.search(r'Creator:\s+(.+)',info)[1]}
  (a.output/(name+'-text.txt')).write_text('\n\nPAGE BREAK\n\n'.join(texts));folder=a.output/name;folder.mkdir(exist_ok=True)
  subprocess.run(['pdftoppm','-scale-to','1400','-png',str(path),str(folder/'page')],check=True)
  files=sorted(folder.glob('page-*.png'));cols=3 if name=='slides' else 2;thumbw=480 if name=='slides' else 390;thumbh=300 if name=='slides' else 535
  for start in range(0,len(files),12):
   selected=files[start:start+12];sheet=Image.new('RGB',(cols*thumbw,((len(selected)+cols-1)//cols)*thumbh),'#cbd5e1');draw=ImageDraw.Draw(sheet)
   for j,f in enumerate(selected):
    im=Image.open(f).convert('RGB');im.thumbnail((thumbw-12,thumbh-28));x=(j%cols)*thumbw+(thumbw-im.width)//2;y=(j//cols)*thumbh+22;sheet.paste(im,(x,y));draw.text(((j%cols)*thumbw+8,(j//cols)*thumbh+5),f'{name} page {start+j+1}',fill='black')
   sheet.save(a.output/f'{name}-contact-{start+1:02d}.png')
 result={'pptx_zip_crc_and_xml':'passed','pptx_author':prs.core_properties.author,'editable_text_shapes':sum(sh.has_text_frame for slide in prs.slides for sh in slide.shapes),'documents':docs,'timing':timing,'layout_estimates':issues,'scope':'All pages rendered; human-visible contact sheets and detailed page images require visual inspection.'};(a.output/'document-checks.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()

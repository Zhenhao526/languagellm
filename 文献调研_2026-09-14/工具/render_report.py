"""Render the research report with embedded Chinese fonts and page source notes."""
import json, re, html
from pathlib import Path
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.platypus import BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, Table, TableStyle, PageBreak, NextPageTemplate, CondPageBreak
from reportlab.pdfgen import canvas

BASE=Path(__file__).resolve().parents[1]
OUTPUT=BASE/'output/pdf'
OUTPUT.mkdir(parents=True,exist_ok=True)
PDF=OUTPUT/'语言诞生文献调研与选题建议.pdf'
pdfmetrics.registerFont(TTFont('CJK','/System/Library/Fonts/STHeiti Light.ttc',subfontIndex=1))
pdfmetrics.registerFont(TTFont('CJKBold','/System/Library/Fonts/STHeiti Medium.ttc',subfontIndex=1))
pdfmetrics.registerFontFamily('CJK',normal='CJK',bold='CJKBold',italic='CJK',boldItalic='CJKBold')
W,H=A4
MARGIN=49
WIDTH=W-2*MARGIN
FOOTHEIGHT=145
FOOTTOP=MARGIN+FOOTHEIGHT
GAP=16

styles={
 'title':ParagraphStyle('title',fontName='CJKBold',fontSize=21,leading=29,spaceAfter=24,wordWrap='CJK'),
 'h2':ParagraphStyle('h2',fontName='CJKBold',fontSize=14,leading=21,spaceBefore=17,spaceAfter=10,keepWithNext=True,wordWrap='CJK'),
 'h3':ParagraphStyle('h3',fontName='CJKBold',fontSize=11.3,leading=18,spaceBefore=10,spaceAfter=7,keepWithNext=True,wordWrap='CJK'),
 'body':ParagraphStyle('body',fontName='CJK',fontSize=10.3,leading=17.4,spaceAfter=9,wordWrap='CJK',allowWidows=0,allowOrphans=0),
 'cell':ParagraphStyle('cell',fontName='CJK',fontSize=9,leading=14,wordWrap='CJK'),
 'cellhead':ParagraphStyle('cellhead',fontName='CJKBold',fontSize=9.1,leading=14,wordWrap='CJK'),
 'foot':ParagraphStyle('foot',fontName='CJK',fontSize=7.1,leading=9.7,wordWrap='CJK'),
 'ref':ParagraphStyle('ref',fontName='CJK',fontSize=9.1,leading=14.5,spaceAfter=10,wordWrap='CJK',allowWidows=0,allowOrphans=0),
}
styles['h2table']=ParagraphStyle('h2table',parent=styles['h2'],keepWithNext=False)
refs=json.loads((BASE/'数据/报告引文映射.json').read_text())
by_number={int(r['number']):r for r in refs}
page_notes=[]

def clean(s):
    return s.replace('\u2011','-').replace('\u2013','-').replace('\u2014','-')

def inline(s, capture=True):
    s=html.escape(clean(s))
    s=re.sub(r'\*\*(.+?)\*\*',r'<b>\1</b>',s)
    def citation(m):
        ns=re.findall(r'\d+',m[0])
        links=','.join(f'<link href="#ref-{n}" color="#303030">{n}</link>' for n in ns)
        callbacks=''.join(f'<onDraw name="capture_citation" label="{n}"/>' for n in ns) if capture else ''
        return '<super>'+links+'</super>'+callbacks
    s=re.sub(r'(?:\[\^\d+\])+',citation,s)
    s=re.sub(r'\[([^\]]+)\]\((https?://[^)]+)\)',r'<link href="\2" color="#303030">\1</link>',s)
    return s

class ResearchCanvas(canvas.Canvas):
    def __init__(self,*a,**kw):
        super().__init__(*a,**kw)
        self.citation_numbers=[]
        self.setNamedCB('capture_citation',lambda canv,kind,label: self.citation_numbers.append(int(label)))

def begin_page(c,doc):
    c.citation_numbers=[]

def end_page(c,doc):
    c.saveState()
    c.setFont('CJK',8)
    c.setFillColor(colors.HexColor('#666666'))
    c.drawCentredString(W/2,25,str(doc.page))
    numbers=list(dict.fromkeys(c.citation_numbers))
    heights=[0,0]
    column_width=(WIDTH-20)/2
    for n in numbers:
        r=by_number[n]
        t=clean(r['title'])
        if len(t)>43: t=t[:40].rstrip()+'…'
        note=f"{n}. {r['first_author']} ({r['year']}). {t}"
        p=Paragraph(f'<link href="{html.escape(r["url"],quote=True)}">{html.escape(note)}</link>',styles['foot'])
        _,height=p.wrap(column_width,1000)
        col=0 if heights[0]<=heights[1] else 1
        p.drawOn(c,MARGIN+col*(column_width+20),FOOTTOP-heights[col]-height)
        heights[col]+=height+2
    maxheight=max(heights)
    if maxheight>FOOTHEIGHT:
        raise RuntimeError(f'Footnote overflow on page {doc.page}: {maxheight} > {FOOTHEIGHT}')
    page_notes.append({'page':doc.page,'references':numbers,'note_height':maxheight})
    c.restoreState()

class ResearchDoc(BaseDocTemplate):
    def afterFlowable(self,flowable):
        if isinstance(flowable,Paragraph) and flowable.style.name in ('h2','h3','h2table'):
            text=flowable.getPlainText()
            key='section-'+str(self.seq.nextf('section'))
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(text,key,level=1 if flowable.style.name=='h3' else 0,closed=False)

doc=ResearchDoc(str(PDF),pagesize=A4,leftMargin=MARGIN,rightMargin=MARGIN,topMargin=MARGIN,bottomMargin=MARGIN,title='语言诞生的实验研究：文献、机制与选题',author='',allowSplitting=1)
body_frame=Frame(MARGIN,FOOTTOP+GAP,WIDTH,H-MARGIN-FOOTTOP-GAP,leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0,id='body')
source_frame=Frame(MARGIN,MARGIN,WIDTH,H-2*MARGIN,leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0,id='sources')
doc.addPageTemplates([PageTemplate(id='Main',frames=[body_frame],onPage=begin_page,onPageEnd=end_page),PageTemplate(id='Sources',frames=[source_frame],onPage=begin_page,onPageEnd=end_page)])

text=(BASE/'语言诞生文献调研与选题建议.md').read_text()
lines=text.splitlines()
story=[]
i=0
while i<len(lines):
    line=lines[i].strip()
    if not line:
        i+=1; continue
    if line=='## 参考文献':
        story += [NextPageTemplate('Sources'),PageBreak(),Paragraph('参考文献',styles['h2'])]
        i+=1; continue
    if line.startswith('[^'):
        m=re.match(r'\[\^(\d+)\]:\s*(.*)',line)
        if not m: raise ValueError(line)
        n,txt=m.groups()
        story.append(Paragraph(f'<a name="ref-{n}"/>{n}. '+inline(txt,False),styles['ref']))
        i+=1; continue
    if line.startswith('|'):
        rows=[]
        while i<len(lines) and lines[i].strip().startswith('|'):
            raw=lines[i].strip()
            cells=[x.strip() for x in raw.strip('|').split('|')]
            if not all(re.fullmatch(r':?-+:?',x.replace(' ','')) for x in cells): rows.append(cells)
            i+=1
        ncols=len(rows[0])
        weights=[0.19,0.40,0.41] if ncols==3 else [1/ncols]*ncols
        data=[[Paragraph(inline(cell),styles['cellhead'] if r==0 else styles['cell']) for cell in row] for r,row in enumerate(rows)]
        table=Table(data,colWidths=[WIDTH*w for w in weights],repeatRows=1,hAlign='LEFT',spaceBefore=5,spaceAfter=13)
        table.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),8),('BOTTOMPADDING',(0,0),(-1,-1),8),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#EDEDED')),('LINEBELOW',(0,0),(-1,0),0.6,colors.HexColor('#888888')),('LINEBELOW',(0,1),(-1,-1),0.35,colors.HexColor('#DDDDDD'))]))
        story.append(table)
        continue
    style='title' if line.startswith('# ') else 'h2' if line.startswith('## ') else 'h3' if line.startswith('### ') else 'body'
    remaining=next((x.strip() for x in lines[i+1:] if x.strip()),'')
    if style=='h2' and remaining.startswith('|'):
        story.append(CondPageBreak(155))
        style='h2table'
    if line.startswith('#'): line=re.sub(r'^#+\s+','',line)
    story.append(Paragraph(inline(line),styles[style]))
    i+=1

doc.build(story,canvasmaker=ResearchCanvas)
(BASE/'数据/PDF页下注校验.json').write_text(json.dumps(page_notes,ensure_ascii=False,indent=2))
print(json.dumps({'pdf':str(PDF),'pages':len(page_notes),'max_note_height':max(x['note_height'] for x in page_notes)},ensure_ascii=False))

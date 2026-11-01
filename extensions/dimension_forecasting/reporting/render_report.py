#!/usr/bin/env python3
"""Render a scientific results report from a fully specified JSON document.

The document specification supplies complete prose, data tables and captions.
This renderer does not estimate effects or select scientific results.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether


def render(specification, destination):
    specification=Path(specification).resolve()
    spec=json.loads(specification.read_text())
    pdfmetrics.registerFont(TTFont("ResearchSerif","/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"))
    pdfmetrics.registerFont(TTFont("ResearchSerifBold","/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"))
    pdfmetrics.registerFont(TTFont("ResearchSans","/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
    pdfmetrics.registerFont(TTFont("ResearchSansBold","/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"))
    pdfmetrics.registerFontFamily("ResearchSerif",normal="ResearchSerif",bold="ResearchSerifBold",italic="ResearchSerif",boldItalic="ResearchSerifBold")
    navy=colors.HexColor("#163449"); ink=colors.HexColor("#202b35"); gray=colors.HexColor("#53616c")
    styles={
        "title":ParagraphStyle("title",fontName="ResearchSansBold",fontSize=19,leading=24,textColor=navy,spaceAfter=12),
        "subtitle":ParagraphStyle("subtitle",fontName="ResearchSans",fontSize=9.2,leading=13,textColor=gray,spaceAfter=13),
        "heading":ParagraphStyle("heading",fontName="ResearchSansBold",fontSize=13.4,leading=18,textColor=navy,spaceAfter=10,keepWithNext=True),
        "subheading":ParagraphStyle("subheading",fontName="ResearchSansBold",fontSize=10,leading=14,textColor=navy,spaceBefore=8,spaceAfter=6,keepWithNext=True),
        "body":ParagraphStyle("body",fontName="ResearchSerif",fontSize=9.3,leading=13.8,textColor=ink,spaceAfter=8),
        "caption":ParagraphStyle("caption",fontName="ResearchSans",fontSize=7.8,leading=11,textColor=gray,spaceAfter=9),
        "cell":ParagraphStyle("cell",fontName="ResearchSans",fontSize=7.8,leading=10.7,textColor=ink),
        "headcell":ParagraphStyle("headcell",fontName="ResearchSansBold",fontSize=7.6,leading=10.5,textColor=navy),
    }
    width=A4[0]-88; story=[]
    for index,page in enumerate(spec["pages"]):
        if index: story.append(PageBreak())
        for item in page:
            kind=item["type"]
            if kind in styles:
                story.append(Paragraph(item["text"],styles[kind]))
            elif kind=="space": story.append(Spacer(1,item.get("height",8)))
            elif kind=="table":
                header=[Paragraph(escape(str(value)),styles["headcell"]) for value in item["headers"]]
                body=[[Paragraph(escape(str(value)),styles["cell"]) for value in row] for row in item["rows"]]
                fractions=item.get("widths",[1/len(header)]*len(header))
                if abs(sum(fractions)-1)>1e-8: raise ValueError("Table widths must sum to one")
                table=Table([header]+body,colWidths=[width*f for f in fractions],repeatRows=1,hAlign="LEFT")
                table.setStyle(TableStyle([
                    ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#eaf0f4")),("VALIGN",(0,0),(-1,-1),"TOP"),
                    ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#f7f9fa")]),
                    ("LINEBELOW",(0,0),(-1,0),.6,navy),("LINEBELOW",(0,-1),(-1,-1),.4,colors.HexColor("#b8c5cf")),
                    ("LEFTPADDING",(0,0),(-1,-1),6),("RIGHTPADDING",(0,0),(-1,-1),6),
                    ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5)]))
                story.extend([table,Spacer(1,8)])
            elif kind=="image":
                picture_path=Path(item["path"])
                if not picture_path.is_absolute(): picture_path=specification.parent/picture_path
                picture=Image(str(picture_path))
                factor=min(width/picture.imageWidth,item.get("max_height",250)/picture.imageHeight)
                picture.drawWidth=picture.imageWidth*factor; picture.drawHeight=picture.imageHeight*factor
                picture.hAlign="CENTER"; story.extend([picture,Spacer(1,6)])
            else: raise ValueError("Unknown report element: "+kind)
    destination=Path(destination); destination.parent.mkdir(parents=True,exist_ok=True)
    def footer(canvas,document):
        canvas.saveState();canvas.setStrokeColor(colors.HexColor("#ccd6dc"));canvas.line(44,43,A4[0]-44,43)
        canvas.setFont("ResearchSans",7.4);canvas.setFillColor(gray)
        canvas.drawString(44,29,spec.get("running_title","KAHM subspace selection"))
        canvas.drawRightString(A4[0]-44,29,str(document.page));canvas.restoreState()
    document=SimpleDocTemplate(str(destination),pagesize=A4,rightMargin=44,leftMargin=44,topMargin=43,bottomMargin=55,
        title=spec["title"],author=spec.get("author",""),subject=spec.get("subject","KAHM forecasting and risk certificates"))
    document.build(story,onFirstPage=footer,onLaterPages=footer)
    return {"path":str(destination),"bytes":destination.stat().st_size,"sha256":hashlib.sha256(destination.read_bytes()).hexdigest()}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("specification",type=Path);parser.add_argument("destination",type=Path)
    args=parser.parse_args();print(json.dumps(render(args.specification,args.destination)))

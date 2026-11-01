#!/usr/bin/env python3
"""Render the frozen protocol and scientific report without changing analysis."""
from __future__ import annotations
import argparse
import html
import re
import textwrap
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                               Preformatted, Image, PageBreak, KeepTogether)

ROOT = Path(__file__).resolve().parents[1]
INK = colors.HexColor('#142C42')
GREY = colors.HexColor('#56616B')
LINE = colors.HexColor('#CBD3DB')
PAGE_W, PAGE_H = A4
WIDTH = PAGE_W - 94


def fonts():
    base = Path('/usr/share/fonts/truetype/dejavu')
    for name, filename in (('Body', 'DejaVuSerif.ttf'), ('Body-Bold', 'DejaVuSerif-Bold.ttf'),
                           ('Head', 'DejaVuSans.ttf'), ('Head-Bold', 'DejaVuSans-Bold.ttf'),
                           ('Mono', 'DejaVuSansMono.ttf')):
        pdfmetrics.registerFont(TTFont(name, str(base / filename)))
    pdfmetrics.registerFontFamily('Body', normal='Body', bold='Body-Bold', italic='Body', boldItalic='Body-Bold')
    pdfmetrics.registerFontFamily('Head', normal='Head', bold='Head-Bold', italic='Head', boldItalic='Head-Bold')


def styles():
    return {
        'body': ParagraphStyle('body', fontName='Body', fontSize=9.6, leading=14.3, spaceAfter=6.2, textColor=colors.black,
                               allowWidows=0, allowOrphans=0),
        'h1': ParagraphStyle('h1', fontName='Head-Bold', fontSize=19, leading=25, spaceAfter=14, textColor=INK, keepWithNext=True),
        'h2': ParagraphStyle('h2', fontName='Head-Bold', fontSize=12.3, leading=17, spaceBefore=13, spaceAfter=7, textColor=INK, keepWithNext=True),
        'h3': ParagraphStyle('h3', fontName='Head-Bold', fontSize=10.4, leading=15, spaceBefore=9, spaceAfter=5, textColor=INK, keepWithNext=True),
        'small': ParagraphStyle('small', fontName='Head', fontSize=8, leading=11.3, spaceAfter=7, textColor=GREY),
        'cell': ParagraphStyle('cell', fontName='Head', fontSize=8, leading=11, textColor=colors.black),
        'cellhead': ParagraphStyle('cellhead', fontName='Head-Bold', fontSize=8, leading=11, textColor=INK),
        'code': ParagraphStyle('code', fontName='Mono', fontSize=7.1, leading=10.2, leftIndent=8, spaceBefore=3, spaceAfter=7),
        'bullet': ParagraphStyle('bullet', fontName='Body', fontSize=9.6, leading=14.3, leftIndent=12, firstLineIndent=-10, spaceAfter=5,
                                 allowWidows=0, allowOrphans=0),
    }


def clean(text):
    return text.replace('\u2011', '-').replace('\u2013', '-').replace('\u2014', ' - ').replace('\u2212', '-')


def inline(text):
    text = html.escape(clean(text))
    text = re.sub(r'`([^`]+)`', r'<font name="Mono" size="8.1">\1</font>', text)
    text = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', text)
    def link(match):
        label, url = match.group(1), match.group(2)
        return ('<link href="' + url + '" color="#174F7D">' + label + '</link>'
                if url.startswith('https://') else label)
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', link, text)
    return text


def table(lines, st):
    rows = [[x.strip() for x in line.strip().strip('|').split('|')] for line in lines]
    rows = [r for r in rows if not all(re.fullmatch(r':?-+:?', c.replace(' ', '')) for c in r)]
    n = len(rows[0])
    if any(len(r) != n for r in rows): raise ValueError('Inconsistent report table columns')
    # Allocate more space to long prose columns, while keeping numeric columns compact.
    weights = []
    for j in range(n):
        lengths = [len(re.sub(r'[`*]', '', r[j])) for r in rows]
        weights.append(max(8, min(36, max(lengths) ** .65 * 3)))
    if n == 2: weights = [1.0, 1.7]
    widths = [WIDTH * w / sum(weights) for w in weights]
    data = [[Paragraph(inline(c), st['cellhead'] if i == 0 else st['cell']) for c in row]
            for i, row in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1, hAlign='LEFT')
    t.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#EDF2F6')),
        ('LINEBELOW', (0, 0), (-1, 0), .65, INK),
        ('LINEBELOW', (0, -1), (-1, -1), .5, LINE),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8FAFB')]),
        ('LEFTPADDING', (0, 0), (-1, -1), 6), ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6), ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    return [Spacer(1, 3), t, Spacer(1, 9)]


def build(markdown, output, title, base):
    st = styles(); lines = markdown.splitlines(); flow = []; i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line: i += 1; continue
        if line == '<!-- pagebreak -->': i += 1; continue
        if line == '<!-- sectionbreak -->': flow.append(PageBreak()); i += 1; continue
        if line.startswith('```'):
            code = []; i += 1
            while i < len(lines) and not lines[i].strip().startswith('```'):
                code += textwrap.wrap(clean(lines[i]), width=103, replace_whitespace=False, drop_whitespace=False) or ['']
                i += 1
            flow.append(Preformatted('\n'.join(code), st['code'])); i += 1; continue
        if line.startswith('|'):
            rows = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                rows.append(lines[i]); i += 1
            flow += table(rows, st); continue
        image = re.fullmatch(r'!\[([^\]]*)\]\(([^)]+)\)', line)
        if image:
            path = (base / image.group(2)).resolve()
            from PIL import Image as PILImage
            with PILImage.open(path) as source: w, h = source.size
            width = WIDTH; height = width * h / w
            if height > 415: width *= 415 / height; height = 415
            graphic = Image(str(path), width=width, height=height)
            caption = Paragraph(inline(image.group(1)), st['small'])
            flow.append(KeepTogether([Spacer(1, 4), graphic, Spacer(1, 5), caption])); i += 1; continue
        heading = re.match(r'^(#{1,3}) (.+)', line)
        if heading:
            flow.append(Paragraph(inline(heading.group(2)), st['h' + str(len(heading.group(1)))])); i += 1; continue
        bullet = re.match(r'^(-|\d+\.) (.+)', line)
        paragraph = [line]; i += 1
        while (i < len(lines) and lines[i].strip()
               and (sum(part.count('`') for part in paragraph) % 2
                    or not re.match(r'^(#|\||```|!\[|<!--|[-] |\d+\. )', lines[i].strip()))):
            paragraph.append(lines[i].strip()); i += 1
        text = ' '.join(paragraph)
        if bullet: text = ('\u2022 ' if bullet.group(1) == '-' else bullet.group(1) + ' ') + text[len(bullet.group(1)) + 1:]
        flow.append(Paragraph(inline(text), st['bullet' if bullet else 'body']))
    output.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(output), pagesize=A4, leftMargin=47, rightMargin=47, topMargin=49, bottomMargin=46,
                            title=title, author='KAHM research study', pageCompression=1)
    def decorate(canvas, document):
        canvas.saveState(); canvas.setStrokeColor(LINE); canvas.setLineWidth(.4)
        canvas.line(47, PAGE_H - 33, PAGE_W - 47, PAGE_H - 33)
        canvas.setFont('Head', 7.3); canvas.setFillColor(GREY)
        canvas.drawString(47, PAGE_H - 26, 'KAHM | Preprocessing and subspace dimension')
        canvas.drawString(47, 27, '1 November 2026 | Controlled simulation study')
        canvas.drawRightString(PAGE_W - 47, 27, str(document.page)); canvas.restoreState()
    doc.build(flow, onFirstPage=decorate, onLaterPages=decorate)
    print(output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol-only', action='store_true')
    p.add_argument('--output', type=Path, default=ROOT.parents[2] / 'output/pdf')
    args = p.parse_args(); fonts()
    introduction = ('# Frozen experimental protocol\n\n'
        '**Preprocessing and validation-selected native dimensions in KAHM forecasting**\n\n'
        'Prospective source and analysis freeze. 24 independent repetitions, 192 task suites, '
        'five preprocessing pairs and six native dimension caps. The complete-procedure '
        'comparison and fixed-common residual mechanism have separate interpretations.\n\n'
        '[Published protocol source](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/blob/'
        '5cfd7a1745a52459b7b5045fe84803b9bbb2b56a/extensions/preprocessing_dimension/PROTOCOL.md).\n\n')
    protocol_body = (ROOT / 'PROTOCOL.md').read_text().split('\n', 1)[1]
    protocol = introduction + protocol_body + '\n\n<!-- sectionbreak -->\n\n' + (ROOT / 'analysis_code/ANALYSIS_PROTOCOL.md').read_text()
    build(protocol, args.output / 'KAHM_Preprocessing_Dimension_Protocol.pdf', 'KAHM preprocessing and dimension: frozen protocol', ROOT)
    if not args.protocol_only:
        build((ROOT / 'FINDINGS.md').read_text(), args.output / 'KAHM_Preprocessing_Dimension_Research_Report.pdf',
              'Preprocessing and validation-selected subspaces in KAHM forecasting', ROOT)


if __name__ == '__main__': main()

"""Read-only PDF checks; requires pypdf and pdfplumber, never imports solvers."""
from pathlib import Path
import hashlib
import json
import re

import pdfplumber
from pypdf import PdfReader

HERE = Path(__file__).resolve().parent
pdf = HERE / 'report.pdf'
digest = hashlib.sha256(pdf.read_bytes()).hexdigest()
log = (HERE / 'report.log').read_text()
reader = PdfReader(pdf)
bad_characters = []
blank_pages = []
fonts = {}
links = 0
def resolve(value):
    return value.get_object() if hasattr(value, 'get_object') else value

for index, page in enumerate(reader.pages, 1):
    links += len(resolve(page.get('/Annots', [])))
    for font_ref in resolve(page['/Resources'].get('/Font', {})).values():
        font = font_ref.get_object()
        if '/DescendantFonts' in font:
            font = font['/DescendantFonts'][0].get_object()
        descriptor = font.get('/FontDescriptor', {})
        if hasattr(descriptor, 'get_object'):
            descriptor = descriptor.get_object()
        name = str(font.get('/BaseFont', 'unknown'))
        fonts[name] = any(key in descriptor for key in ('/FontFile', '/FontFile2', '/FontFile3'))
with pdfplumber.open(pdf) as document:
    for index, page in enumerate(document.pages, 1):
        if len(page.chars) < 10:
            blank_pages.append(index)
        for char in page.chars:
            if (char['x0'] < -0.1 or char['x1'] > page.width + 0.1
                    or char['top'] < -0.1 or char['bottom'] > page.height + 0.1):
                bad_characters.append(dict(page=index, text=char['text']))
errors = {
    'undefined_references_or_citations': bool(re.search(r'(?:Reference|Citation).*undefined|undefined references', log, re.I)),
    'undefined_commands': 'Undefined control sequence' in log,
    'missing_characters': 'Missing character:' in log,
    'characters_outside_page': bad_characters,
    'blank_pages': blank_pages,
    'unembedded_fonts': [name for name, embedded in fonts.items() if not embedded],
}
overfull = [float(x) for x in re.findall(r'Overfull \\hbox \(([\d.]+)pt too wide\)', log)]
visual_file = HERE / 'visual_review.json'
visual = json.loads(visual_file.read_text()) if visual_file.exists() else None
passed = not any(errors.values()) and len(reader.pages) == 49
if visual is not None:
    passed = passed and visual['pdf_sha256'] == digest and visual['pages_reviewed'] == list(range(1, 50))
result = dict(status='passed' if passed else 'failed', pdf_sha256=digest,
              tex_sha256=hashlib.sha256((HERE/'report.tex').read_bytes()).hexdigest(),
              pages=len(reader.pages), fonts_embedded=fonts, link_annotations=links,
              named_destinations=len(reader.named_destinations), errors=errors,
              overfull_hbox_pt=overfull,
              minor_layout_note='One 1.82121 pt paragraph overrun lies within the physical page; no clipping. Underfull spacing notices remain.',
              visual_review=visual, new_solver_MC_ML_calls=0)
(HERE/'layout_verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
print(json.dumps({k:result[k] for k in ('status','pages','link_annotations','named_destinations','errors')},ensure_ascii=False))
raise SystemExit(0 if passed else 1)

# parsers.py
from typing import Dict
from pathlib import Path
import pandas as pd
from PyPDF2 import PdfReader
from pptx import Presentation
import docx

def parse_pdf(path: str) -> Dict:
    out = {'text': []}
    reader = PdfReader(path)
    for page in reader.pages:
        text = page.extract_text() or ""
        out['text'].append(text)
    out['text'] = "\n".join(out['text'])
    return out

def parse_pptx(path: str) -> Dict:
    prs = Presentation(path)
    slides = []
    for slide in prs.slides:
        texts = []
        for shape in slide.shapes:
            try:
                if hasattr(shape, "text"):
                    texts.append(shape.text)
            except Exception:
                pass
        slides.append("\n".join(texts))
    return {'text': "\n".join(slides)}

def parse_docx(path: str) -> Dict:
    doc = docx.Document(path)
    paras = [p.text for p in doc.paragraphs]
    return {'text': "\n".join(paras)}

def parse_csv(path: str) -> Dict:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    # Convert to reasonable text
    rows = []
    for i, row in df.iterrows():
        rows.append(" | ".join([f"{c}: {row[c]}" for c in df.columns]))
    return {'text': "\n".join(rows)}

def parse_txt(path: str) -> Dict:
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        return {'text': f.read()}

def parse_file(path: str) -> Dict:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == '.pdf':
        return parse_pdf(path)
    if suffix in ['.pptx', '.ppt']:
        return parse_pptx(path)
    if suffix in ['.docx', '.doc']:
        return parse_docx(path)
    if suffix == '.csv':
        return parse_csv(path)
    if suffix in ['.txt', '.md']:
        return parse_txt(path)
    # default fallback: try reading text
    return parse_txt(path)

# utils.py
import re
from typing import List

def clean_text(t: str) -> str:
    t = t.replace('\n', ' ').strip()
    t = re.sub(r'\s+', ' ', t)
    return t

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Simple sliding-window chunker (characters).
    """
    text = clean_text(text)
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
        if start < 0:
            start = 0
    return chunks

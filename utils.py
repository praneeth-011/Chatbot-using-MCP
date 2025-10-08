# utils.py
import re
from typing import List

def clean_text(t: str) -> str:
    """Cleans unwanted whitespaces and newlines from text."""
    t = re.sub(r'\s+', ' ', t.replace('\n', ' ').replace('\t', ' ')).strip()
    return t

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Splits text into overlapping chunks (word-based, to preserve meaning).
    """
    text = clean_text(text)
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = max(end - overlap, 0)
    return chunks

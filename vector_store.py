# vector_store.py
import faiss
import numpy as np
import os
import json
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any

EMBED_MODEL = "all-MiniLM-L6-v2"

class VectorStore:
    def __init__(self, dim: int = 384, index_path="assets/faiss.index", meta_path="assets/meta.json"):
        os.makedirs("assets", exist_ok=True)
        self.index_path = index_path
        self.meta_path = meta_path
        self.dim = dim
        self.model = SentenceTransformer(EMBED_MODEL)
        if os.path.exists(index_path) and os.path.exists(meta_path):
            self.index = faiss.read_index(index_path)
            with open(meta_path, 'r', encoding='utf-8') as f:
                self.meta = json.load(f)
        else:
            self.index = faiss.IndexFlatL2(self.dim)
            self.meta = []  # list of dicts aligned with index
            self._save()

    def _save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, 'w', encoding='utf-8') as f:
            json.dump(self.meta, f, ensure_ascii=False, indent=2)

    def add_texts(self, texts: List[str], metadatas: List[Dict[str, Any]]):
        emb = self.model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        emb = np.array(emb).astype('float32')
        self.index.add(emb)
        # append metadata
        for m in metadatas:
            self.meta.append(m)
        self._save()

    def query(self, q: str, top_k: int = 5):
        q_emb = self.model.encode([q], show_progress_bar=False, normalize_embeddings=True).astype('float32')
        D, I = self.index.search(q_emb, top_k)
        results = []
        for idx, dist in zip(I[0], D[0]):
            if idx < 0 or idx >= len(self.meta):
                continue
            m = self.meta[idx].copy()
            m['score'] = float(dist)
            results.append(m)
        return results

    def clear(self):
        self.index = faiss.IndexFlatL2(self.dim)
        self.meta = []
        self._save()

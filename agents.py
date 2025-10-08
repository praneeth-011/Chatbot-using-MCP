# agents.py
import asyncio
import uuid
from typing import Dict, Any, List
from parsers import parse_file
from utils import chunk_text, clean_text
from vector_store import VectorStore
import os
import json
import openai
from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_API_KEY = os.getenv("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")

if not OPENAI_API_KEY:
    print("⚠️ OPENAI_API_KEY not set. LLM queries will not work.")
    
# Ensure OPENAI_API_KEY is set in environment
OPENAI_API_KEY = os.environ.get("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")
openai.api_key = OPENAI_API_KEY

# MCP message structure for typing:
def make_message(sender: str, receiver: str, type_: str, payload: Dict[str, Any], trace_id: str = None):
    return {
        "sender": sender,
        "receiver": receiver,
        "type": type_,
        "trace_id": trace_id or str(uuid.uuid4()),
        "payload": payload
    }

class IngestionAgent:
    def __init__(self, inbox: asyncio.Queue, outbox: asyncio.Queue, store: VectorStore):
        self.inbox = inbox
        self.outbox = outbox
        self.store = store

    async def run(self):
        while True:
            msg = await self.inbox.get()
            try:
                if msg['type'] == 'INGEST_FILES':
                    files = msg['payload']['files']  # list of filepaths
                    trace_id = msg['trace_id']
                    all_chunks = []
                    metas = []
                    for fpath in files:
                        parsed = parse_file(fpath)
                        text = clean_text(parsed.get('text', ''))
                        chunks = chunk_text(text, chunk_size=800, overlap=100)
                        for i, ch in enumerate(chunks):
                            meta = {
                                'source': fpath,
                                'chunk_id': f"{os.path.basename(fpath)}_chunk_{i}",
                                'text': ch
                            }
                            all_chunks.append(ch)
                            metas.append(meta)
                    if all_chunks:
                        self.store.add_texts(all_chunks, metas)
                    # Send MCP message to RetrievalAgent
                    resp = make_message(
                        sender='IngestionAgent',
                        receiver='RetrievalAgent',
                        type_='INGESTION_DONE',
                        payload={'num_chunks': len(all_chunks)},
                        trace_id=trace_id
                    )
                    await self.outbox.put(resp)
            except Exception as e:
                print("Ingestion error:", e)
            finally:
                self.inbox.task_done()

class RetrievalAgent:
    def __init__(self, inbox: asyncio.Queue, outbox: asyncio.Queue, store: VectorStore):
        self.inbox = inbox
        self.outbox = outbox
        self.store = store

    async def run(self):
        while True:
            msg = await self.inbox.get()
            try:
                if msg['type'] == 'RETRIEVE':
                    query = msg['payload']['query']
                    top_k = msg['payload'].get('top_k', 5)
                    trace_id = msg['trace_id']
                    results = self.store.query(query, top_k=top_k)
                    # Build payload with top chunks
                    top_chunks = [{'text': r['text'], 'source': r['source'], 'score': r['score']} for r in results]
                    resp = make_message(
                        sender='RetrievalAgent',
                        receiver='LLMResponseAgent',
                        type_='RETRIEVAL_RESULT',
                        payload={'retrieved_context': top_chunks, 'query': query},
                        trace_id=trace_id
                    )
                    await self.outbox.put(resp)
            except Exception as e:
                print("Retrieval error:", e)
            finally:
                self.inbox.task_done()

class LLMResponseAgent:
    def __init__(self, inbox: asyncio.Queue, outbox: asyncio.Queue):
        self.inbox = inbox
        self.outbox = outbox

    def _build_prompt(self, query: str, top_chunks: List[Dict]):
        # Compose prompt with context
        context_texts = []
        for i, c in enumerate(top_chunks):
            context_texts.append(f"Source {i+1} ({c['source']}):\n{c['text']}\n")
        context = "\n\n".join(context_texts)
        prompt = f"""
You are a helpful assistant. Use only the information in the provided sources to answer the user query. If the answer is not present, say 'I don't know from the documents provided.'.

CONTEXT:
{context}

USER QUERY:
{query}

Provide: 1) A concise answer, 2) A short list of sources used (file name + chunk id or index).
"""
        return prompt

    async def call_openai(self, prompt: str):
        # uses OpenAI ChatCompletion API
        if OPENAI_API_KEY is None:
            # fallback: simple echo
            return {"answer": "OPENAI_API_KEY not set. Can't call LLM.", "llm_raw": None}
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini" if "gpt-4o-mini" in openai.Model.list().__str__() else "gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.0
        )
        txt = resp['choices'][0]['message']['content']
        return {"answer": txt, "llm_raw": resp}

    async def run(self):
        while True:
            msg = await self.inbox.get()
            try:
                if msg['type'] == 'RETRIEVAL_RESULT':
                    trace_id = msg['trace_id']
                    query = msg['payload']['query']
                    top_chunks = msg['payload']['retrieved_context']
                    prompt = self._build_prompt(query, top_chunks)
                    llm_res = await self.call_openai(prompt)
                    resp = make_message(
                        sender='LLMResponseAgent',
                        receiver='UI',
                        type_='FINAL_ANSWER',
                        payload={'answer': llm_res['answer'], 'sources': top_chunks},
                        trace_id=trace_id
                    )
                    await self.outbox.put(resp)
            except Exception as e:
                print("LLMResponse error:", e)
            finally:
                self.inbox.task_done()

class CoordinatorAgent:
    """
    Accepts user actions (upload + ask) and dispatches MCP messages to the agent inboxes.
    """
    def __init__(self, ingestion_in: asyncio.Queue, retrieval_in: asyncio.Queue, llm_in: asyncio.Queue, ui_out: asyncio.Queue):
        self.ingestion_in = ingestion_in
        self.retrieval_in = retrieval_in
        self.llm_in = llm_in
        self.ui_out = ui_out

    async def ingest_files(self, files: List[str]):
        msg = make_message(sender='UI', receiver='IngestionAgent', type_='INGEST_FILES', payload={'files': files})
        await self.ingestion_in.put(msg)
        # Wait for ingestion done -> will be processed by RetrievalAgent through MCP path

    async def handle_query(self, query: str):
        msg = make_message(sender='UI', receiver='RetrievalAgent', type_='RETRIEVE', payload={'query': query})
        await self.retrieval_in.put(msg)

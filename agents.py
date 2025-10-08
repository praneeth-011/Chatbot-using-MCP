import os
import asyncio
from typing import List, Dict
from dotenv import load_dotenv

# Detect Streamlit environment
try:
    import streamlit as st
    STREAMLIT = True
except ImportError:
    STREAMLIT = False

# Load local .env
load_dotenv()

# Dual-source API key
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") if STREAMLIT else os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    print("⚠️ OPENAI_API_KEY not set. LLM queries will not work.")

# ---------------- Safe LLM wrapper ----------------
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

class SafeLLMAgent:
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

    async def get_response(self, prompt: str):
        if not self.client:
            return {"answer": "⚠️ LLM not available. Set OPENAI_API_KEY.", "sources": []}
        try:
            # synchronous call inside async
            from openai import ChatCompletion
            resp = ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}]
            )
            answer = resp.choices[0].message.content
            return {"answer": answer, "sources": []}
        except Exception as e:
            return {"answer": f"⚠️ Error calling LLM: {e}", "sources": []}


# ---------------- MCP Agents ----------------
class LLMResponseAgent:
    def __init__(self, inbox: asyncio.Queue, outbox: asyncio.Queue):
        self.inbox = inbox
        self.outbox = outbox
        self.safe_llm = SafeLLMAgent()

    def _build_prompt(self, query: str, top_chunks: List[Dict]):
        context_texts = [f"Source {i+1} ({c['source']}):\n{c['text']}\n" for i, c in enumerate(top_chunks)]
        context = "\n\n".join(context_texts)
        prompt = f"""
You are a helpful assistant. Use only the information in the provided sources to answer the user query.
If the answer is not present, say 'I don't know from the documents provided.'

Context:
{context}

User query: {query}
"""
        return prompt

    async def run(self):
        while True:
            item = await self.inbox.get()
            query = item['query']
            top_chunks = item.get('top_chunks', [])

            print("[LLM] Received query:", query)

            prompt = self._build_prompt(query, top_chunks)
            response = await self.safe_llm.get_response(prompt)

            await self.outbox.put({"type": "FINAL_ANSWER", "payload": response})
            self.inbox.task_done()
            print("[LLM] Response sent to UI")

# ---------------- Dummy Agents for testing ----------------
class IngestionAgent:
    def __init__(self, inbox, retrieval_in, store):
        self.inbox = inbox
        self.retrieval_in = retrieval_in
        self.store = store

    async def run(self):
        while True:
            await asyncio.sleep(1)

class RetrievalAgent:
    def __init__(self, inbox, llm_in, store):
        self.inbox = inbox
        self.llm_in = llm_in
        self.store = store

    async def run(self):
        while True:
            await asyncio.sleep(1)

class CoordinatorAgent:
    def __init__(self, ingest_in, retrieval_in, llm_in, ui_out):
        self.ingest_in = ingest_in
        self.retrieval_in = retrieval_in
        self.llm_in = llm_in
        self.ui_out = ui_out

    async def ingest_files(self, paths: list):
        print("[Coordinator] Files ingested:", paths)

    async def handle_query(self, query: str):
        print("[Coordinator] Handling query:", query)
        # Send dummy chunks to LLM
        top_chunks = [{"source": "Test Doc", "text": "This is a test document text."}]
        await self.llm_in.put({"query": query, "top_chunks": top_chunks})

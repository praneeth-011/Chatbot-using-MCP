# agent.py
import asyncio
from typing import List, Dict
from openai import OpenAI
import os

# Load API key from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


# ---------------- LLM Agent ----------------
class LLMResponseAgent:
    def __init__(self, inbox: asyncio.Queue, outbox: asyncio.Queue):
        self.inbox = inbox
        self.outbox = outbox

    def _build_prompt(self, query: str, top_chunks: List[Dict]):
        context_texts = []
        for i, c in enumerate(top_chunks):
            context_texts.append(f"Source {i+1} ({c['source']}):\n{c['text']}\n")
        context = "\n\n".join(context_texts)

        return f"""
You are a helpful assistant. Answer the user question using only the information provided in the sources.
If the answer is not present, say "I don't know from the provided documents."

Context:
{context}

User question:
{query}

Answer:
"""

    async def run(self):
        while True:
            task = await self.inbox.get()
            try:
                if task["type"] == "LLM_QUERY":
                    query = task["payload"]["query"]
                    top_chunks = task["payload"]["top_chunks"]
                    prompt = self._build_prompt(query, top_chunks)

                    if not client:
                        answer = "⚠️ LLM not available. Set OPENAI_API_KEY."
                    else:
                        # OpenAI API call
                        resp = client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[{"role": "user", "content": prompt}],
                        )
                        answer = resp.choices[0].message.content.strip()

                    await self.outbox.put({
                        "type": "FINAL_ANSWER",
                        "payload": {"answer": answer, "sources": top_chunks}
                    })

            except Exception as e:
                await self.outbox.put({"type": "ERROR", "payload": {"error": str(e)}})
            finally:
                self.inbox.task_done()


# ---------------- Dummy Ingestion & Retrieval Agents ----------------
class IngestionAgent:
    def __init__(self, inbox, retrieval_in, store):
        self.inbox = inbox
        self.retrieval_in = retrieval_in
        self.store = store

    async def run(self):
        while True:
            task = await self.inbox.get()
            # Here, you would parse and add files to vector store
            # For demo, just simulate ingestion
            await asyncio.sleep(1)
            self.inbox.task_done()


class RetrievalAgent:
    def __init__(self, inbox, llm_in, store):
        self.inbox = inbox
        self.llm_in = llm_in
        self.store = store

    async def run(self):
        while True:
            task = await self.inbox.get()
            if task["type"] == "QUERY":
                query = task["payload"]["query"]
                # For demo: fetch top chunks from store (dummy)
                top_chunks = [{"source": "DemoDoc.txt", "text": "This is a demo chunk."}]
                await self.llm_in.put({"type": "LLM_QUERY", "payload": {"query": query, "top_chunks": top_chunks}})
            await asyncio.sleep(0.1)
            self.inbox.task_done()


class CoordinatorAgent:
    def __init__(self, ingest_in, retrieval_in, llm_in, ui_out):
        self.ingest_in = ingest_in
        self.retrieval_in = retrieval_in
        self.llm_in = llm_in
        self.ui_out = ui_out

    async def ingest_files(self, paths):
        await self.ingest_in.put({"type": "INGEST_FILES", "payload": {"paths": paths}})

    async def handle_query(self, query):
        await self.retrieval_in.put({"type": "QUERY", "payload": {"query": query}})

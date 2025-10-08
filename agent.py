# agent.py
import asyncio
from typing import List, Dict
from vector_store import VectorStore

# ---------------- LLM Agent ----------------
class LLMResponseAgent:
    def __init__(self, inbox: asyncio.Queue, outbox: asyncio.Queue, client=None):
        self.inbox = inbox
        self.outbox = outbox
        self.client = client  # OpenAI client

    def _build_prompt(self, query: str, top_chunks: List[Dict]):
        context_texts = []
        for i, c in enumerate(top_chunks):
            context_texts.append(f"Source {i+1} ({c.get('source','unknown')}):\n{c.get('text','')}\n")
        context = "\n\n".join(context_texts)
        return f"""
You are a helpful AI assistant. Use only the given sources to answer the question.
If the answer is not in the sources, say "I don't know from the provided documents."

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

                    if self.client:
                        response = self.client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[{"role": "user", "content": prompt}],
                        )
                        answer = response.choices[0].message.content.strip()
                    else:
                        answer = "⚠️ LLM not available. Set OPENAI_API_KEY."

                    await self.outbox.put({
                        "type": "FINAL_ANSWER",
                        "payload": {"answer": answer, "sources": top_chunks}
                    })
            except Exception as e:
                await self.outbox.put({"type": "ERROR", "payload": {"error": str(e)}})
            finally:
                self.inbox.task_done()

# ---------------- Dummy Agents ----------------
class IngestionAgent:
    def __init__(self, inbox, retrieval_in, store: VectorStore):
        self.inbox = inbox
        self.retrieval_in = retrieval_in
        self.store = store

    async def run(self):
        while True:
            task = await self.inbox.get()
            # Here you can implement actual ingestion logic
            self.inbox.task_done()
            await asyncio.sleep(0.1)

class RetrievalAgent:
    def __init__(self, inbox, llm_in, store: VectorStore):
        self.inbox = inbox
        self.llm_in = llm_in
        self.store = store

    async def run(self):
        while True:
            task = await self.inbox.get()
            # Here you can implement retrieval and push LLM_QUERY to llm_in
            self.inbox.task_done()
            await asyncio.sleep(0.1)

class CoordinatorAgent:
    def __init__(self, ingest_in, retrieval_in, llm_in, ui_out):
        self.ingest_in = ingest_in
        self.retrieval_in = retrieval_in
        self.llm_in = llm_in
        self.ui_out = ui_out

    async def ingest_files(self, paths: list):
        await self.ingest_in.put({"type": "INGEST_FILES", "payload": {"paths": paths}})

    async def handle_query(self, query: str):
        # Example: push query to retrieval_in
        await self.retrieval_in.put({"type": "QUERY", "payload": {"query": query}})


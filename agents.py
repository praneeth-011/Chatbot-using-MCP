import os
import asyncio
from typing import List, Dict
from openai import OpenAI
from dotenv import load_dotenv

# Detect Streamlit environment
try:
    import streamlit as st
    STREAMLIT = True
except ImportError:
    STREAMLIT = False

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

    def _build_prompt(self, query: str, top_chunks: List[Dict]):
        context_texts = []
        for i, c in enumerate(top_chunks):
            context_texts.append(f"Source {i+1} ({c['source']}):\n{c['text']}\n")
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

                    # ✅ Use new OpenAI SDK call
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",  # change model if needed
                        messages=[{"role": "user", "content": prompt}],
                    )

                    answer = response.choices[0].message.content.strip()

                    await self.outbox.put({
                        "type": "FINAL_ANSWER",
                        "payload": {
                            "answer": answer,
                            "sources": top_chunks,
                        }
                    })
            except Exception as e:
                await self.outbox.put({
                    "type": "ERROR",
                    "payload": {"error": str(e)},
                })
            finally:
                self.inbox.task_done()

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

    async def handle_query(self, query: str):
        # retrieve top chunks
        await self.retrieval_in.put({
            "type": "RETRIEVE",
            "payload": {"query": query}
        })


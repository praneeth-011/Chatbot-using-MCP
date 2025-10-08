# app.py
import streamlit as st
import asyncio
import threading
import uuid
from pathlib import Path
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.warning("⚠️ OPENAI_API_KEY not set. LLM queries will not work.")

from agent import IngestionAgent, RetrievalAgent, LLMResponseAgent, CoordinatorAgent

# ---------------- Queues ----------------
ingest_in, retrieval_in, llm_in, ui_out = asyncio.Queue(), asyncio.Queue(), asyncio.Queue(), asyncio.Queue()

store = {}  # Replace with your VectorStore if available

ingestion_agent = IngestionAgent(ingest_in, retrieval_in, store)
retrieval_agent = RetrievalAgent(retrieval_in, llm_in, store)
llm_agent = LLMResponseAgent(llm_in, ui_out)
coordinator = CoordinatorAgent(ingest_in, retrieval_in, llm_in, ui_out)

# ---------------- Async event loop ----------------
def start_event_loop(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()

loop = asyncio.new_event_loop()
threading.Thread(target=start_event_loop, args=(loop,), daemon=True).start()

async def schedule_agents():
    await asyncio.gather(
        ingestion_agent.run(),
        retrieval_agent.run(),
        llm_agent.run()
    )

asyncio.run_coroutine_threadsafe(schedule_agents(), loop)

def run_async(coro):
    return asyncio.run_coroutine_threadsafe(coro, loop)

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Agentic RAG Chatbot", layout="wide")
st.title("🧠 Agentic RAG Chatbot — MCP Demo")

# Upload documents
st.sidebar.header("1️⃣ Upload Documents")
uploaded = st.sidebar.file_uploader("Upload files", accept_multiple_files=True)

if st.sidebar.button("Ingest Files"):
    if not uploaded:
        st.sidebar.warning("Upload at least one file.")
    else:
        paths = []
        Path("assets").mkdir(exist_ok=True)
        for f in uploaded:
            p = f"assets/{uuid.uuid4().hex[:6]}_{f.name}"
            with open(p, "wb") as out:
                out.write(f.getbuffer())
            paths.append(p)
        run_async(coordinator.ingest_files(paths))
        st.sidebar.success("Files queued for ingestion.")

# Ask a question
st.sidebar.header("2️⃣ Ask a Question")
query = st.sidebar.text_input("Enter your question")

if st.sidebar.button("Ask"):
    if query.strip():
        run_async(coordinator.handle_query(query))
        st.sidebar.info("Query sent! Waiting for response...")
    else:
        st.sidebar.warning("Enter a question first.")

# Display responses
import time
msgs = []
start_time = time.time()
timeout = 5  # seconds

while time.time() - start_time < timeout:
    try:
        fut = asyncio.run_coroutine_threadsafe(ui_out.get(), loop)
        msg = fut.result(timeout=0.5)
        msgs.append(msg)
        ui_out.task_done()
    except Exception:
        time.sleep(0.1)

for m in msgs:
    if m['type'] == 'FINAL_ANSWER':
        st.subheader("Answer")
        st.write(m['payload']['answer'])
        st.subheader("Sources")
        for s in m['payload']['sources'][:3]:
            st.markdown(f"- {s.get('source','unknown')} (score={s.get('score',0):.3f})")
    elif m['type'] == 'ERROR':
        st.error(m['payload']['error'])

# app.py
import streamlit as st
import asyncio
import threading
import uuid
from pathlib import Path
from dotenv import load_dotenv
import os
from agents import IngestionAgent, RetrievalAgent, LLMResponseAgent, CoordinatorAgent
from vector_store import VectorStore

# ---------------- Load OpenAI API Key ----------------
load_dotenv()
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.warning("⚠️ OPENAI_API_KEY not set. LLM queries will not work.")

# ---------------- Queues ----------------
ingest_in, retrieval_in, llm_in, ui_out = asyncio.Queue(), asyncio.Queue(), asyncio.Queue(), asyncio.Queue()

# ---------------- Vector Store ----------------
store = VectorStore()

# ---------------- Agents ----------------
ingestion_agent = IngestionAgent(ingest_in, retrieval_in, store)
retrieval_agent = RetrievalAgent(retrieval_in, llm_in, store)
llm_agent = LLMResponseAgent(llm_in, ui_out, api_key=OPENAI_API_KEY)
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

# ----- Upload Documents -----
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

# ----- Ask a Question -----
st.sidebar.header("2️⃣ Ask a Question")
query = st.sidebar.text_input("Enter your question")

if st.sidebar.button("Ask"):
    if query.strip():
        run_async(coordinator.handle_query(query))
        st.sidebar.info("Query sent! Waiting for response...")
    else:
        st.sidebar.warning("Enter a question first.")

# ----- Display responses dynamically -----
st.header("💬 Chatbot Responses")
placeholder = st.empty()

async def fetch_responses():
    msgs = []
    try:
        while True:
            msg = await ui_out.get()
            msgs.append(msg)
            ui_out.task_done()
            yield msg
    except asyncio.CancelledError:
        return

# Non-blocking Streamlit display
response_container = st.container()

def display_messages(msgs):
    for m in msgs:
        if m['type'] == 'FINAL_ANSWER':
            response_container.subheader("Answer")
            response_container.write(m['payload']['answer'])
            response_container.subheader("Sources")
            for s in m['payload']['sources'][:3]:
                response_container.markdown(f"- {s.get('source','unknown')} (score={s.get('score',0):.3f})")
        elif m['type'] == 'ERROR':
            response_container.error(m['payload']['error'])
        else:
            response_container.write(f"⚙️ Intermediate: {m.get('payload', {}).get('answer','')}")

# Poll ui_out queue every 0.5 seconds
def poll_ui():
    msgs = []
    while True:
        try:
            fut = asyncio.run_coroutine_threadsafe(ui_out.get(), loop)
            msg = fut.result(timeout=0.2)
            ui_out.task_done()
            msgs.append(msg)
        except Exception:
            break
    display_messages(msgs)

st.button("Refresh Responses", on_click=poll_ui)


        st.write(m['payload']['answer'])
        st.subheader("Sources")
        for s in m['payload']['sources'][:3]:
            st.markdown(f"- {s.get('source','unknown')} (score={s.get('score',0):.3f})")

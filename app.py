import streamlit as st, asyncio, threading, uuid
from agents import IngestionAgent, RetrievalAgent, LLMResponseAgent, CoordinatorAgent
from vector_store import VectorStore
from pathlib import Path


ingest_in, retrieval_in, llm_in, ui_out = asyncio.Queue(), asyncio.Queue(), asyncio.Queue(), asyncio.Queue()
store = VectorStore()
ingestion_agent = IngestionAgent(ingest_in, retrieval_in, store)
retrieval_agent = RetrievalAgent(retrieval_in, llm_in, store)
llm_agent = LLMResponseAgent(llm_in, ui_out)
coordinator = CoordinatorAgent(ingest_in, retrieval_in, llm_in, ui_out)

def start_event_loop(loop): asyncio.set_event_loop(loop); loop.run_forever()
loop = asyncio.new_event_loop()
threading.Thread(target=start_event_loop, args=(loop,), daemon=True).start()
async def schedule_agents(): await asyncio.gather(ingestion_agent.run(), retrieval_agent.run(), llm_agent.run())
asyncio.run_coroutine_threadsafe(schedule_agents(), loop)
def run_async(coro): return asyncio.run_coroutine_threadsafe(coro, loop)

st.set_page_config(page_title="Agentic RAG Chatbot (No-LLM)", layout="wide")
st.title("🧠 Agentic RAG Chatbot — MCP Demo (Offline Mode)")

st.sidebar.header("1️⃣ Upload Documents")
uploaded = st.sidebar.file_uploader("Upload files", accept_multiple_files=True)
if st.sidebar.button("Ingest Files"):
    if not uploaded: st.sidebar.warning("Upload at least one file.")
    else:
        paths = []
        for f in uploaded:
            p = f"assets/{uuid.uuid4().hex[:6]}_{f.name}"
            with open(p,'wb') as out: out.write(f.getbuffer())
            paths.append(p)
        run_async(coordinator.ingest_files(paths))
        st.sidebar.success("Files queued for ingestion.")

st.sidebar.header("2️⃣ Ask a Question")
query = st.sidebar.text_input("Enter your question")
if st.sidebar.button("Ask"):
    if query.strip():
        run_async(coordinator.handle_query(query))
        st.sidebar.info("Query sent! Waiting for response...")
    else:
        st.sidebar.warning("Enter a question first.")

st.header("💬 Chatbot Responses")
msgs = []
try:
    while True:
        fut = asyncio.run_coroutine_threadsafe(ui_out.get(), loop)
        msg = fut.result(timeout=0.2)
        msgs.append(msg)
        ui_out.task_done()
except Exception:
    pass

for m in msgs:
    if m['type'] == 'FINAL_ANSWER':
        st.subheader("Answer")
        st.write(m['payload']['answer'])
        st.subheader("Sources")
        for s in m['payload']['sources'][:3]:
            st.markdown(f"- {s.get('source','unknown')} (score={s.get('score',0):.3f})")

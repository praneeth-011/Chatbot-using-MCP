# Chatbot-using-MCP
# Agentic RAG Chatbot (MCP) — Prototype

## Overview
This is a prototype implementation of an **agentic RAG chatbot** using a Model Context Protocol (MCP) for message passing between agents:
- IngestionAgent (parses documents)
- RetrievalAgent (embeddings + faiss search)
- LLMResponseAgent (formats prompt and calls LLM)
- CoordinatorAgent (dispatches user actions)

Supports files: PDF, PPTX, DOCX, CSV, TXT/MD.

UI: Streamlit

Installation Steps
1️⃣ Install Python
Download and install the latest version from python.org/downloads
During installation, make sure to tick
“Add Python to PATH”

2️⃣ Extract the Project
If you downloaded a ZIP file:
Right-click → Extract All

You’ll get a folder named:
agentic_rag_chatbot/

3️⃣ Open PowerShell or CMD in the Folder
In File Explorer:
Go to the extracted folder
Click on the address bar

Type:
powershel
and press Enter

4️⃣ Create a Virtual Environment (Recommended)
python -m venv venv
venv\Scripts\activate

5️⃣ Install Dependencies
Install all required packages using:
pip install --upgrade pip setuptools wheel
pip install streamlit faiss-cpu sentence-transformers openai pandas python-docx PyPDF2 python-pptx tqdm aiofiles

6️⃣ Run the Application

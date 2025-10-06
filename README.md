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

## Setup
1. Clone repo
2. Create virtualenv and install requirements:

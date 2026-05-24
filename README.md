# AI Customer Support Bot (Enterprise Full-Stack RAG)

A professional, enterprise-grade AI customer support platform built with a high-performance **FastAPI** backend and a sleek **Next.js** frontend. The core support engine utilizes **LangChain** and **LangGraph** to orchestrate RAG (Retrieval-Augmented Generation) workflows, ensuring highly accurate, context-grounded, and stateful responses to customer inquiries.

---

## 🚀 System Architecture

```mermaid
graph TD
    A[Next.js React Client] -->|SSE Stream /chat/stream| B[FastAPI Backend]
    A -->|Static Chat /chat| B
    B -->|Query / Retrieve| C[Local Vector Database]
    B -->|Checkpointer Persistence| D[MemorySaver Checkpoints]
    B -->|LLM Bindings| E[Local LLM (Qwen 2.5)]
    B -->|SQL Queries| G[Oracle 11g Relational DB]
    C -->|Indexes| F[Multi-format Knowledge Base]
```

### Core Execution Flow
1. **RAG Ingestion**: On server start, company documentation (PDFs, TXT, DOCX, and Excel files in `data/`) are processed using **Docling** and indexed into a local vector database (Qdrant/FAISS/ChromaDB) using local embeddings (e.g. `nomic-embed-text` via Ollama).
2. **Relational Database Queries (SQL Agent)**: Integrates an **Oracle 11g** database containing transaction, customer profiles, shipping tracking, returns, and support ticket records. A LangChain SQL Agent queries this relational data using natural language via the `oracledb` thin client driver.
3. **Table Selector & Router**: Dynamically matches user queries against table descriptions to load only relevant schemas into the SQL database context, preventing LLM prompt clutter when scaling to a high number of tables.
4. **Stateful Conversation**: LangGraph maintains the support assistant's state, orchestrating conditional routing. If a customer query demands company specifics (products, shipping, returns, technical policies), the agent invokes a retrieval tool to fetch grounded knowledge.
5. **Local LLM Orchestration**: Connects fully offline to Ollama or a custom local model gateway, running Qwen 2.5 or other local models, eliminating third-party API dependencies.
6. **Real-time Streaming**: Model responses are streamed back to the Next.js client token-by-token using HTTP Server-Sent Events (SSE), reducing Time-to-First-Token (TTFT) and providing a premium, fluid user experience.

---

## 🛠️ Tech Stack

### Backend Service
* **API Framework**: [FastAPI](https://fastapi.tiangolo.com/) (using asynchronous endpoints & Lifespan managers)
* **Agentic Orchestration**: [LangGraph](https://github.com/langchain-ai/langgraph) (featuring `MemorySaver` thread checkpoints)
* **LLM Engine**: LangChain community model integrations (supporting local **Qwen 2.5** and local embeddings)
* **SQL Agent Toolkit**: LangChain `SQLDatabase` and `create_sql_agent`
* **Vector Database**: Configurable [Qdrant](https://qdrant.tech/) or [FAISS](https://github.com/facebookresearch/faiss) or [ChromaDB](https://github.com/chroma-core/chroma)
* **Relational Database**: **Oracle 11g** via `oracledb` Thin Driver and SQLAlchemy ORM
* **Document Loaders**: **Docling** for advanced, unified offline layout-aware parsing of PDF, DOCX, and Excel files

### Frontend Interface
* **Framework**: [Next.js](https://nextjs.org/) (React, TypeScript)
* **Styling**: [Tailwind CSS](https://tailwindcss.com/) & Vanilla CSS with beautiful dark-mode glassmorphic aesthetics
* **Streaming Client**: Native `fetch` with `ReadableStream` for smooth token-by-token UI rendering
* **UX/UI Highlights**: Interactive chat interface, multi-modal screenshot attachment, and an expandable **RAG Insights side panel** displaying precise document sources and text snippets.

---

## ✨ Features

* **Retrieval-Augmented Generation (RAG)**: Automatically searches and retrieves company documentation to ground replies.
* **LangChain SQL Agent**: Automatically writes, verifies, and executes SQL queries to fetch structured order details, tracking status, support tickets, and inventory levels.
* **Dynamic Table Routing**: Semantic routing filters and groups tables for the SQL Agent based on user query intent, preventing prompt bloat and schema confusion.
* **Server-Sent Events (SSE) Streaming**: Token-by-token text generation rendering in real-time.
* **Multi-Modal Support**: Allows customers to attach error screenshots alongside their questions.
* **Stateful Tool Calling**: Agent decides when to search the knowledge base or query relational tables using native LangGraph conditional edges.
* **Docker Containerization**: Multi-stage Docker configurations for both development and production deployment.

---

## ⚙️ Getting Started

### Prerequisites
* Docker & Docker Compose (Recommended)
* OR Python 3.9+ & Node.js 18+
* Local LLM runner (e.g., [Ollama](https://ollama.com/) with Qwen 2.5 model pulled)
* Oracle 11g Database (or containerized Oracle instance)

---

### Method A: Running with Docker Compose (Quickest)

1. Clone the repository:
   ```bash
   git clone https://github.com/Yashborse4/ai-customer-support-bot.git
   cd ai-customer-support-bot
   ```

2. Configure environment variables:
   Create a `.env` file in the root directory:
   ```env
   # Oracle 11g Database Configuration
   DB_USER=system
   DB_PASSWORD=oracle
   DB_HOST=localhost
   DB_PORT=1521
   DB_SERVICE_NAME=xe

   # Vector Store Configuration
   PERSIST_DIRECTORY=./chroma_db
   COLLECTION_NAME=customer_support_kb

   # Model Configuration
   MODEL_NAME=qwen2.5
   EMBEDDING_MODEL=nomic-embed-text
   LOCAL_LLM_BASE_URL=http://localhost:11434/v1
   LOCAL_EMBEDDING_BASE_URL=http://localhost:11434/v1

   NEXT_PUBLIC_API_URL=http://localhost:8000
   ```

3. Launch the complete application:
   ```bash
   docker compose up --build
   ```
   * Next.js Frontend: `http://localhost:3000`
   * FastAPI Backend: `http://localhost:8000`
   * FastAPI Swagger Docs: `http://localhost:8000/docs`

---

### Method B: Manual Local Setup

#### 1. Start the FastAPI Backend
1. Create a Python virtual environment and install packages:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Make sure your `.env` file is properly configured with your Oracle DB connection parameters and local model URLs.

3. Place company documentation (e.g., PDFs, TXT, DOCX, or Excel files) inside the `data/` directory.

4. Run the API server (will automatically initialize the Oracle 11g database tables and mock seed data if they do not exist):
   ```bash
   uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
   ```

#### 2. Start the Next.js Frontend
1. Open a new terminal in the `frontend` folder:
   ```bash
   cd frontend
   npm install
   ```

2. Start the Next.js local development server:
   ```bash
   npm run dev
   ```
   Open `http://localhost:3000` in your browser.

---

## 🧪 Testing

The codebase includes an extensive unit and integration test suite (fully typed, leveraging `pytest` and `fastapi.testclient`):

To run tests:
```bash
$env:PYTHONPATH="."
python -m pytest
```

---

## 📈 Project Roadmap

- [x] Complete Multi-modal query integration (Vision analysis for error screenshots).
- [x] Robust PDF, DOCX, and Excel indexing system using Docling.
- [x] Migrate relational database to Oracle 11g with thin client driver support.
- [x] Implement Dynamic Table Selection & Routing for large schemas.
- [x] State-of-the-art token streaming architecture (SSE) for low Time-to-First-Token (TTFT).
- [x] Premium Next.js interactive frontend with RAG insights.
- [ ] Integration with Slack and Discord channels.
- [ ] Advanced analytics and administrator dashboard for human support operators.

---
Built with ❤️ by [Yash Borse](https://github.com/Yashborse4)

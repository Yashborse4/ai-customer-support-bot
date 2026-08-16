# Enterprise AI Customer Support Bot (Full-Stack RAG & SQL Agent)

An enterprise-grade, local-first AI customer support platform built with a high-performance **FastAPI** backend and a responsive **Next.js** frontend. The system leverages **LangChain** and **LangGraph** to orchestrate stateful, multi-agent Retrieval-Augmented Generation (RAG) and structured SQL agent workflows, persisting conversation sessions natively via SQLite checkpoints.

This repository is designed to showcase production-ready LLM application patterns, focusing on **security sandboxing, PII redaction, stateful token streaming, and advanced retrieval architectures**.

---

## 🚀 System Architecture

```mermaid
graph TD
    Client["Next.js React Client"] -->|1. SSE Stream /chat/stream| API["FastAPI API Gateway"]
    
    subgraph FastAPI Backend
        API -->|2. Mask PII & Initialize Turn| PII["PII Security Guard"]
        PII -->|3. Invoke Stateful Execution| Graph["LangGraph Workflow"]
        
        subgraph LangGraph Support Engine
            Graph -->|Checkpointer| Mem["MemorySaver Checkpoints"]
            Graph -->|State Node| Agent["Support Agent Node"]
            
            Agent -->|Tool Call| SQL["SQL Agent Tool"]
            Agent -->|Tool Call| RAG["RAG Retrieval Tool"]
            
            subgraph SQL Engine Sandbox
                SQL -->|Unmask Query| SQLDB["SQL Database Engine"]
                SQLDB -->|intercept before_cursor_execute| Guard["Write-Blocking Interceptor"]
                Guard -->|Blocked| Err["PermissionError"]
                Guard -->|Allowed SELECT| DB["Oracle 11g / SQLite DB"]
            end
            
            subgraph Parent-Child RAG Pipeline
                RAG -->|Query| Vector["Vector Store (Child Chunks)"]
                Vector -->|Map parent_id| SQLite["SQLite Parent Doc Store"]
                SQLite -->|Fetch Context| Context["Context-Rich Documents"]
            end
        end
    end
    
    Context -->|Return Parent Context| Agent
    DB -->|Return Masked Records| SQL
    Agent -->|4. Mask Output & Update Map| PII
    PII -->|5. SSE Stream Buffer (Unmask)| Client
```

---

## 🌟 Core Engineering Enhancements

### 1. PII Redaction & Data Security Guardrails
To enforce compliance (GDPR, PCI-DSS) and prevent sensitive data leaks, the backend features a robust PII sanitization layer:
* **Deterministic Masking:** User queries containing emails, credit cards, phone numbers, SSNs, or IP addresses are masked with deterministic tokens (e.g. `__[MASKED_EMAIL_0]__`) before entering the LangGraph state or database checkpointers.
* **Turn Consistency:** Placeholder mappings are saved statefully in `SupportState` across multiple turns, reusing indices if a user mentions the same details again.
* **SSE Stream Buffer:** A token-by-token stream buffer in the `/chat/stream` API intercepts outgoing model streams. It caches incomplete placeholders (holding back partial `__` segments) and yields the fully unmasked string to the user once reconstructed.

```python
# Real-time token streaming unmasking buffer in main.py
parts = stream_buffer.split("__")
if len(parts) % 2 == 0:
    # Hold back partial placeholder forming at the end of the buffer
    split_idx = stream_buffer.rfind("__")
    emit_part = stream_buffer[:split_idx]
    stream_buffer = stream_buffer[split_idx:]
else:
    emit_part = stream_buffer
    stream_buffer = ""
```

### 2. Parent-Document Retrieval & Semantic Chunking
Traditional character-based chunking often dilutes context, while indexing large documents causes prompt bloat. We solved this with a hybrid Parent-Child architecture:
* **Semantic Chunking:** Text is split using sentence embeddings from our local Qwen model. It calculates the cosine similarity between consecutive sentences, dynamically computing a threshold (20th percentile) to split only where semantic shifts occur.
* **SQLite Parent Doc Store:** Large parent blocks (2,000 characters) are saved in the `parent_documents` table in SQLite. Compact, semantically split child chunks are stored in Qdrant/FAISS, mapping to their respective `parent_id`.
* **Precision Retrieval:** The custom `ParentDocumentRetriever` queries the vector store for child chunks but swaps them for the complete parent documents before prompting the LLM, ensuring the model gets the broader context.

### 3. SQL Sandbox & Engine-Level Write Interceptor
A LangChain SQL Agent dynamically queries transactions, shipping, returns, and ticket details. To prevent malicious or hallucinated database modifications:
* **SQLAlchemy Event Listener:** An event hook (`before_cursor_execute`) intercepts every query compiled by the database engine.
* **Strict Blacklisting:** The hook scans statements using regular expressions and blocks forbidden SQL commands (`INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, `CREATE`, `RENAME`, `TRUNCATE`, `REPLACE`) with a `PermissionError`.
* **Scope Segregation:** Sandboxing runs dynamically on connections returned to the LLM agent, while database creation/seeding functions run securely under a separate write-enabled context.

```python
# SQL Interceptor in sql_db.py
@event.listens_for(engine, "before_cursor_execute")
def block_write_queries(conn, cursor, statement, parameters, context, executemany):
    statement_upper = statement.strip().upper()
    forbidden_keywords = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "RENAME", "TRUNCATE", "REPLACE"]
    for keyword in forbidden_keywords:
        if re.search(rf"\b{keyword}\b", statement_upper):
            raise PermissionError(f"Security Alert: Forbidden keyword '{keyword}' is blocked.")
```

### 4. Dynamic Query Routing & Table Selection
To prevent prompt clutter when scaling schemas, a semantic router selects relevant tables (e.g. `shipping` for shipping queries, `returns` for refund queries) based on natural language query keywords and department privileges (`sales`, `technical`, `billing`, `general`).

---

## 🛠️ Tech Stack

### Backend Service
* **Orchestration**: [LangGraph](https://github.com/langchain-ai/langgraph) & [LangChain](https://www.langchain.com/) (Stateful workflows with `MemorySaver` thread persistence)
* **API Framework**: [FastAPI](https://fastapi.tiangolo.com/) (Async lifespan, StreamingResponse, Server-Sent Events)
* **LLM Engine**: Local **Qwen 2.5** and local embeddings (`nomic-embed-text`) via Ollama/Local Gateway
* **Vector Store**: Configurable [Qdrant](https://qdrant.tech/) or [FAISS](https://github.com/facebookresearch/faiss)
* **Relational Database**: **Oracle 11g** via `oracledb` Thin Driver & **SQLite** via SQLAlchemy ORM
* **Document Parsing**: **Docling** for layout-aware PDF, DOCX, and Excel structure extraction

### Frontend Client
* **Framework**: [Next.js](https://nextjs.org/) (React, TypeScript)
* **Styling**: [Tailwind CSS](https://tailwindcss.com/) (Dark-mode glassmorphic aesthetics)
* **Streaming Client**: Native `fetch` with `ReadableStream` for real-time SSE token rendering
* **Insights Panel**: Side drawer displaying precise RAG source document matches and metadata

---

## 📁 Repository Structure

```text
├── src/
│   ├── api/
│   │   └── main.py          # FastAPI application, streaming router, & endpoints
│   ├── agents/
│   │   └── support_agent.py # LangChain support agent node prompt & tool bindings
│   ├── tools/
│   │   ├── sql_tool.py      # Database agent query tool with PII unmasking
│   │   └── retrieval_tool.py# Knowledge base lookup tool
│   ├── database/
│   │   ├── sql_db.py        # Oracle/SQLite setups, mock seeding, and SQL sandbox listener
│   │   ├── vector_store.py  # SemanticChunker, ParentDocumentRetriever, & Docling indexing
│   │   └── pdf_parser.py    # Docling & PyPDF extraction pipelines
│   ├── core/
│   │   ├── config.py        # Pydantic Settings and SQLite config loaders
│   │   └── security.py      # PIISecurityGuard utility class
│   ├── schemas/
│   │   └── state.py         # SupportState dict definition
│   └── graph/
│       └── workflow.py      # LangGraph state workflow compilation
├── tests/
│   ├── test_pii_security.py # PII masking, turn persistence, and SSE buffer tests
│   ├── test_semantic_rag.py # Semantic Chunking & Parent-child retriever tests
│   ├── test_sql_sandbox.py  # Write-blocking SQL interceptor tests
│   ├── test_sql_agent.py    # Database initializations and routing tests
│   └── test_docling_vector_stores.py # Vector store mocking and Qdrant/FAISS indexing tests
└── frontend/                # Next.js UI folder
```

---

## ⚙️ Getting Started

### 1. Configure Environment Variables
Create a `.env` file in the root directory:
```env
# Relational Database Selection (sqlite or oracle)
DB_TYPE=sqlite
SQLITE_DB_PATH=data/customer_support.db

# Oracle 11g Database Configuration (only used if DB_TYPE=oracle)
DB_USER=system
DB_PASSWORD=oracle
DB_HOST=localhost
DB_PORT=1521
DB_SERVICE_NAME=xe

# Vector Store Configuration
PERSIST_DIRECTORY=./chroma_db
COLLECTION_NAME=customer_support_kb
VECTOR_DB_TYPE=qdrant
QDRANT_PATH=./qdrant_db

# Model Configuration
MODEL_NAME=qwen2.5
EMBEDDING_MODEL=nomic-embed-text
LOCAL_LLM_BASE_URL=http://localhost:11434/v1
LOCAL_EMBEDDING_BASE_URL=http://localhost:11434/v1

NEXT_PUBLIC_API_URL=http://localhost:8000
```

### 2. Method A: Running via Docker Compose (Recommended)
Launch the frontend, backend, and database in a single command:
```bash
docker compose up --build
```
* Next.js Frontend: `http://localhost:3000`
* FastAPI Backend: `http://localhost:8000`
* FastAPI Swagger Docs: `http://localhost:8000/docs`

### 3. Method B: Manual Local Setup
#### Backend
1. Initialize virtual environment and install packages:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. Start the API server:
   ```bash
   uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
   ```

#### Frontend
1. Install node dependencies:
   ```bash
   cd frontend
   npm install
   ```
2. Run development server:
   ```bash
   npm run dev
   ```
   Access `http://localhost:3000` in your web browser.

---

## 🧪 Verification & Testing

The project includes an extensive test suite verifying RAG, SQL, security boundaries, and API streaming configurations.

To run the complete test suite:
```bash
.venv\Scripts\python -m pytest
```

### Passing Test Run Summary:
```text
tests\test_basic.py ..                                                   [  6%]
tests\test_docling_vector_stores.py ....                                 [ 18%]
tests\test_improvements.py ......                                        [ 37%]
tests\test_pii_security.py ...                                           [ 46%]
tests\test_semantic_rag.py ....                                          [ 59%]
tests\test_sql_agent.py ....                                             [ 71%]
tests\test_sql_sandbox.py .........                                      [100%]

================== 32 passed, 1 warning in 140.96s (0:02:20) ==================
```

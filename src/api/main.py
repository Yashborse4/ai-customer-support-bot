from contextlib import asynccontextmanager
import json
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Any, AsyncGenerator, Dict, List, Optional
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from src.graph.workflow import support_bot_graph
from src.database.vector_store import vector_store_manager
from src.database.sql_db import (
    initialize_database,
    get_db_tables,
    get_all_table_metadata,
    save_table_metadata_db,
    get_db_credentials,
    test_db_connection
)
from src.core.config import settings
from fastapi.middleware.cors import CORSMiddleware
from src.core.security import PIISecurityGuard

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager that handles startup and shutdown operations.

    Args:
        app: The FastAPI application instance.
    """
    print("Initializing SQL Database...")
    initialize_database()
    
    # Load dynamic configurations into settings from SQLite DB
    from src.database.sql_db import load_settings_from_db
    load_settings_from_db()
    print("SQL Database and System Settings ready.")
    
    print("Indexing documents for API...")
    vector_store_manager.load_and_index_documents("data")
    print("Indexing complete.")
    yield

app = FastAPI(
    title="Acme Corp Support Bot API",
    description="REST API for the RAG-supported AI Customer Support Bot",
    version="1.0.0",
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models for Request/Response
class ChatMessage(BaseModel):
    role: str # "user" or "assistant"
    content: str
    image_url: Optional[str] = None # Base64 string or URL

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    thread_id: Optional[str] = None
    department: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    history: List[ChatMessage]

def convert_to_langchain_messages(messages: List[ChatMessage]) -> List[BaseMessage]:
    """Converts API message format to LangChain message format.

    Args:
        messages: A list of ChatMessage request objects.

    Returns:
        A list of LangChain BaseMessage objects (HumanMessage or AIMessage).
    """
    lc_messages = []
    for msg in messages:
        if msg.role == "user":
            if msg.image_url:
                content = [
                    {"type": "text", "text": msg.content},
                    {
                        "type": "image_url",
                        "image_url": {"url": msg.image_url},
                    },
                ]
            else:
                content = msg.content
            lc_messages.append(HumanMessage(content=content))
        elif msg.role == "assistant":
            lc_messages.append(AIMessage(content=msg.content))
    return lc_messages

def convert_to_api_messages(lc_messages: List[BaseMessage]) -> List[ChatMessage]:
    """Converts LangChain message format to API message format.

    Args:
        lc_messages: A list of LangChain BaseMessage objects.

    Returns:
        A list of ChatMessage response objects.
    """
    api_messages = []
    for msg in lc_messages:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        content = ""
        image_url = None
        
        if isinstance(msg.content, str):
            content = msg.content
        elif isinstance(msg.content, list):
            for block in msg.content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        content = block.get("text", "")
                    elif block.get("type") == "image_url":
                        image_url = block.get("image_url", {}).get("url")
        
        api_messages.append(ChatMessage(role=role, content=content, image_url=image_url))
    return api_messages

@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint.

    Returns:
        A dictionary containing the status of the server and the configuration used.
    """
    db_ok = test_db_connection()
    return {
        "status": "healthy",
        "model": settings.MODEL_NAME,
        "llm_model": settings.MODEL_NAME,
        "embedding_model": settings.EMBEDDING_MODEL,
        "vector_db": settings.VECTOR_DB_TYPE,
        "db_connected": db_ok
    }

async def event_generator(request: ChatRequest) -> AsyncGenerator[str, None]:
    """Generates server-sent event (SSE) chunks from LangGraph token execution.

    Args:
        request: The ChatRequest message history.

    Yields:
        JSON string representation of tokens generated or errors encountered.
    """
    try:
        guard = PIISecurityGuard()
        thread_id = request.thread_id or "default-session"
        config = {"configurable": {"thread_id": thread_id}}
        
        # Retrieve existing thread state to get the previous masking map
        thread_state = await support_bot_graph.aget_state(config)
        masking_map = thread_state.values.get("masking_map", {}) if thread_state and thread_state.values else {}
        
        # Mask all user messages in the request history
        masked_messages = []
        for msg in request.messages:
            if msg.role == "user":
                masked_content, masking_map = guard.mask(msg.content, masking_map)
                masked_messages.append(ChatMessage(role=msg.role, content=masked_content, image_url=msg.image_url))
            else:
                masked_messages.append(msg)
        
        # Convert history to LangChain messages
        messages = convert_to_langchain_messages(masked_messages)
        
        # Prepare state with the active masking map
        state = {
            "messages": messages,
            "department": request.department or "general",
            "masking_map": masking_map
        }
        
        stream_buffer = ""
        async for event in support_bot_graph.astream_events(state, config=config, version="v2"):
            # Intercept actual generated tokens from the model
            if event["event"] == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                if hasattr(chunk, "content") and chunk.content:
                    token = chunk.content
                    stream_buffer += token
                    
                    # Unmask complete placeholders present in the buffer
                    for placeholder, original in list(masking_map.items()):
                        if placeholder in stream_buffer:
                            stream_buffer = stream_buffer.replace(placeholder, original)
                            
                    # Buffer check: hold back partial placeholders to avoid rendering __[MASKED... to the user
                    parts = stream_buffer.split("__")
                    if len(parts) % 2 == 0:
                        split_idx = stream_buffer.rfind("__")
                        emit_part = stream_buffer[:split_idx]
                        stream_buffer = stream_buffer[split_idx:]
                    else:
                        emit_part = stream_buffer
                        stream_buffer = ""
                        
                    if emit_part:
                        yield f"data: {json.dumps({'token': emit_part})}\n\n"
                        
        # Flush any remaining tokens in the stream buffer
        if stream_buffer:
            for placeholder, original in list(masking_map.items()):
                stream_buffer = stream_buffer.replace(placeholder, original)
            yield f"data: {json.dumps({'token': stream_buffer})}\n\n"
                    
    except Exception as e:
        # Emit graceful error block via SSE
        yield f"data: {json.dumps({'error': str(e)})}\n\n"

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """Streaming chat endpoint. Streams bot response token-by-token using SSE.

    Args:
        request: The ChatRequest payload containing conversation history.

    Returns:
        A FastAPI StreamingResponse carrying the real-time token stream.
    """
    return StreamingResponse(event_generator(request), media_type="text/event-stream")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Main chat endpoint. Processes conversation history and returns a standard response.

    Args:
        request: The ChatRequest payload containing conversation history.

    Returns:
        A ChatResponse instance containing the assistant's response and updated history.

    Raises:
        HTTPException: Internal server error if processing fails.
    """
    try:
        guard = PIISecurityGuard()
        thread_id = request.thread_id or "default-session"
        config = {"configurable": {"thread_id": thread_id}}
        
        # Retrieve existing thread state to get the previous masking map
        thread_state = await support_bot_graph.aget_state(config)
        masking_map = thread_state.values.get("masking_map", {}) if thread_state and thread_state.values else {}
        
        # Mask all user messages in the request history
        masked_messages = []
        for msg in request.messages:
            if msg.role == "user":
                masked_content, masking_map = guard.mask(msg.content, masking_map)
                masked_messages.append(ChatMessage(role=msg.role, content=masked_content, image_url=msg.image_url))
            else:
                masked_messages.append(msg)
                
        # Convert history to LangChain messages
        messages = convert_to_langchain_messages(masked_messages)
        
        # Prepare state with the active masking map
        state = {
            "messages": messages,
            "department": request.department or "general",
            "masking_map": masking_map
        }
        
        # Invoke LangGraph
        result = await support_bot_graph.ainvoke(state, config=config)
        
        # Extract last message, masking map, and updated history
        updated_lc_messages = result["messages"]
        bot_response = updated_lc_messages[-1].content
        final_masking_map = result.get("masking_map", masking_map)
        
        # Unmask the response and the history before returning to the user
        unmasked_bot_response = guard.unmask(bot_response, final_masking_map)
        
        api_history = convert_to_api_messages(updated_lc_messages)
        unmasked_history = []
        for msg in api_history:
            unmasked_history.append(ChatMessage(
                role=msg.role,
                content=guard.unmask(msg.content, final_masking_map),
                image_url=msg.image_url
            ))
        
        return ChatResponse(
            response=unmasked_bot_response,
            history=unmasked_history
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/index")
async def reindex_documents() -> Dict[str, str]:
    """Manually triggers document re-indexing from the 'data/' directory.

    Returns:
        A success message indicating document indexing status.

    Raises:
        HTTPException: Internal server error if indexing fails.
    """
    try:
        vector_store_manager.load_and_index_documents("data")
        return {"status": "success", "message": "Documents re-indexed successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Dynamic control console API models and endpoints
class DbConfigPayload(BaseModel):
    user: str
    password: str
    host: str
    port: int
    service_name: str

class ModelConfigPayload(BaseModel):
    MODEL_NAME: str
    EMBEDDING_MODEL: str
    LOCAL_LLM_BASE_URL: str
    LOCAL_EMBEDDING_BASE_URL: str
    VECTOR_DB_TYPE: str

class TableMetadataPayload(BaseModel):
    table_name: str
    description: str

class SaveConfigPayload(BaseModel):
    db_config: Optional[DbConfigPayload] = None
    model_settings: Optional[ModelConfigPayload] = Field(None, alias="model_config")

@app.get("/api/config")
async def get_config() -> Dict[str, Any]:
    """Retrieves dynamic model and database configurations from system settings."""
    creds = get_db_credentials()
    return {
        "db_config": creds,
        "model_config": {
            "MODEL_NAME": settings.MODEL_NAME,
            "EMBEDDING_MODEL": settings.EMBEDDING_MODEL,
            "LOCAL_LLM_BASE_URL": settings.LOCAL_LLM_BASE_URL,
            "LOCAL_EMBEDDING_BASE_URL": settings.LOCAL_EMBEDDING_BASE_URL,
            "VECTOR_DB_TYPE": settings.VECTOR_DB_TYPE
        }
    }

@app.post("/api/config")
async def save_config(payload: SaveConfigPayload) -> Dict[str, str]:
    """Saves dynamic database connection configurations or model settings to SQLite system settings.

    Args:
        payload: The config settings payload containing optional db_config and/or model_config.

    Returns:
        A dictionary with a status and message.

    Raises:
        HTTPException: If saving configuration fails.
    """
    from src.database.sql_db import save_system_setting, load_settings_from_db
    
    try:
        if payload.db_config:
            save_system_setting("db_user", payload.db_config.user)
            save_system_setting("db_password", payload.db_config.password)
            save_system_setting("db_host", payload.db_config.host)
            save_system_setting("db_port", str(payload.db_config.port))
            save_system_setting("db_service_name", payload.db_config.service_name)
            
        if payload.model_settings:
            save_system_setting("model_name", payload.model_settings.MODEL_NAME)
            save_system_setting("embedding_model", payload.model_settings.EMBEDDING_MODEL)
            save_system_setting("local_llm_base_url", payload.model_settings.LOCAL_LLM_BASE_URL)
            save_system_setting("local_embedding_base_url", payload.model_settings.LOCAL_EMBEDDING_BASE_URL)
            save_system_setting("vector_db_type", payload.model_settings.VECTOR_DB_TYPE)
            
        # Synchronize dynamic settings singleton in memory
        load_settings_from_db()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update settings in database: {e}")
            
    return {"status": "success", "message": "Configuration updated successfully."}

@app.get("/api/database/tables")
async def fetch_tables() -> Dict[str, Any]:
    """Queries the database dynamically and returns user tables with descriptions."""
    tables = get_db_tables()
    metadata = get_all_table_metadata()
    return {"tables": tables, "metadata": metadata}

@app.post("/api/database/tables/metadata")
async def save_table_metadata(payload: TableMetadataPayload) -> Dict[str, str]:
    """Saves user-configured semantic description for a table in SQLite metadata DB."""
    try:
        save_table_metadata_db(payload.table_name, payload.description)
        return {"status": "success", "message": "Table metadata saved successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save metadata: {e}")

@app.post("/api/rag/upload")
async def upload_document(file: UploadFile = File(...)) -> Dict[str, str]:
    """Receives a document file, saves it to data/ directory, and triggers re-indexing."""
    import os
    import shutil
    os.makedirs("data", exist_ok=True)
    file_path = os.path.join("data", file.filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Trigger layout-aware extraction and indexing
        vector_store_manager.load_and_index_documents("data")
        return {"status": "success", "message": f"File '{file.filename}' uploaded and indexed successfully."}
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Failed to upload and index document: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

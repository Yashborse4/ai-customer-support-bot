from contextlib import asynccontextmanager
import json
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Any, AsyncGenerator, Dict, List, Optional
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from src.graph.workflow import support_bot_graph
from src.database.vector_store import vector_store_manager
from src.core.config import settings
from fastapi.middleware.cors import CORSMiddleware

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager that handles startup and shutdown operations.

    Args:
        app: The FastAPI application instance.
    """
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
async def health_check() -> Dict[str, str]:
    """Health check endpoint.

    Returns:
        A dictionary containing the status of the server and the LLM model name used.
    """
    return {"status": "healthy", "model": settings.MODEL_NAME}

async def event_generator(request: ChatRequest) -> AsyncGenerator[str, None]:
    """Generates server-sent event (SSE) chunks from LangGraph token execution.

    Args:
        request: The ChatRequest message history.

    Yields:
        JSON string representation of tokens generated or errors encountered.
    """
    try:
        # Convert history to LangChain messages
        messages = convert_to_langchain_messages(request.messages)
        
        # Prepare state
        state = {"messages": messages}
        
        async for event in support_bot_graph.astream_events(state, version="v2"):
            # Intercept actual generated tokens from the model
            if event["event"] == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                if hasattr(chunk, "content") and chunk.content:
                    # Emit chunk serialized as JSON formatted SSE data
                    yield f"data: {json.dumps({'token': chunk.content})}\n\n"
                    
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
        # Convert history to LangChain messages
        messages = convert_to_langchain_messages(request.messages)
        
        # Prepare state
        state = {"messages": messages}
        
        # Invoke LangGraph
        result = await support_bot_graph.ainvoke(state)
        
        # Extract last message and updated history
        updated_lc_messages = result["messages"]
        bot_response = updated_lc_messages[-1].content
        
        return ChatResponse(
            response=bot_response,
            history=convert_to_api_messages(updated_lc_messages)
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

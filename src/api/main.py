from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from src.graph.workflow import support_bot_graph
from src.database.vector_store import vector_store_manager
from src.core.config import settings
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Acme Corp Support Bot API",
    description="REST API for the RAG-supported AI Customer Support Bot",
    version="1.0.0"
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
    """Converts API message format to LangChain format."""
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
    """Converts LangChain message format to API format."""
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

@app.on_event("startup")
async def startup_event():
    """Initializes RAG on startup."""
    print("Indexing documents for API...")
    vector_store_manager.load_and_index_documents("data")
    print("Indexing complete.")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model": settings.MODEL_NAME}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint. Processes conversation history and returns a response.
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
async def reindex_documents():
    """Manually triggers document re-indexing."""
    try:
        vector_store_manager.load_and_index_documents("data")
        return {"status": "success", "message": "Documents re-indexed successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

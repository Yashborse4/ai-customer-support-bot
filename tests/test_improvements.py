import json
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient
from src.api.main import app
from src.graph.workflow import support_bot_graph
from langgraph.graph.state import CompiledStateGraph
from langchain_core.messages import HumanMessage, AIMessageChunk

def test_api_health() -> None:
    """Verify that the FastAPI health endpoint returns the correct status and model."""
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model" in data

def test_graph_compiled_properties() -> None:
    """Verify that the LangGraph workflow compiled successfully and has a checkpointer."""
    assert isinstance(support_bot_graph, CompiledStateGraph)
    # Verify checkpointer config
    assert hasattr(support_bot_graph, "checkpointer")
    assert support_bot_graph.checkpointer is not None

def test_message_conversion_helpers() -> None:
    """Test API helper functions for converting to and from LangChain messages."""
    from src.api.main import ChatMessage, convert_to_langchain_messages, convert_to_api_messages
    from langchain_core.messages import HumanMessage, AIMessage

    # Test conversion of text messages
    api_messages = [
        ChatMessage(role="user", content="Hello support"),
        ChatMessage(role="assistant", content="Hi there!")
    ]
    lc_messages = convert_to_langchain_messages(api_messages)
    
    assert len(lc_messages) == 2
    assert isinstance(lc_messages[0], HumanMessage)
    assert lc_messages[0].content == "Hello support"
    assert isinstance(lc_messages[1], AIMessage)
    assert lc_messages[1].content == "Hi there!"

    # Test conversion back to API messages
    api_back = convert_to_api_messages(lc_messages)
    assert len(api_back) == 2
    assert api_back[0].role == "user"
    assert api_back[0].content == "Hello support"
    assert api_back[1].role == "assistant"
    assert api_back[1].content == "Hi there!"

    # Test base64 image message conversion
    image_api_msg = [ChatMessage(role="user", content="Look at this", image_url="data:image/png;base64,123")]
    lc_image_msg = convert_to_langchain_messages(image_api_msg)
    
    assert len(lc_image_msg) == 1
    assert isinstance(lc_image_msg[0].content, list)
    assert lc_image_msg[0].content[0]["type"] == "text"
    assert lc_image_msg[0].content[0]["text"] == "Look at this"
    assert lc_image_msg[0].content[1]["type"] == "image_url"
    assert lc_image_msg[0].content[1]["image_url"]["url"] == "data:image/png;base64,123"

@pytest.mark.asyncio
async def test_graph_execution_with_mock_llm() -> None:
    """Verify that support_bot_graph executes successfully when the LLM is mocked."""
    from langchain_core.messages import AIMessage
    from langchain_openai import ChatOpenAI
    
    mock_response = AIMessage(content="This is a mock support response.")
    
    with patch.object(ChatOpenAI, "ainvoke", new_callable=AsyncMock) as mock_ainvoke:
        mock_ainvoke.return_value = mock_response
        
        state = {"messages": [HumanMessage(content="Hello Acme Corp")]}
        result = await support_bot_graph.ainvoke(state, config={"configurable": {"thread_id": "test_thread"}})
        
        assert "messages" in result
        assert len(result["messages"]) == 2
        assert result["messages"][-1].content == "This is a mock support response."

def test_chat_endpoint_with_mock_llm() -> None:
    """Verify that POST /chat endpoint responds correctly with a mocked LLM."""
    from langchain_core.messages import AIMessage
    from langchain_openai import ChatOpenAI
    
    mock_response = AIMessage(content="Hello! I am a mocked support assistant.")
    
    with patch.object(ChatOpenAI, "ainvoke", new_callable=AsyncMock) as mock_ainvoke:
        mock_ainvoke.return_value = mock_response
        
        client = TestClient(app)
        response = client.post(
            "/chat",
            json={"messages": [{"role": "user", "content": "Hi support"}]}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["response"] == "Hello! I am a mocked support assistant."
        assert len(data["history"]) == 2
        assert data["history"][0]["role"] == "user"
        assert data["history"][1]["role"] == "assistant"
        assert data["history"][1]["content"] == "Hello! I am a mocked support assistant."

@pytest.mark.asyncio
async def test_chat_endpoint_streaming_mocked() -> None:
    """Verify that POST /chat/stream responds with SSE tokens correctly."""
    async def mock_astream_events(*args, **kwargs):
        yield {
            "event": "on_chat_model_stream",
            "data": {
                "chunk": AIMessageChunk(content="Mocked ")
            }
        }
        yield {
            "event": "on_chat_model_stream",
            "data": {
                "chunk": AIMessageChunk(content="stream.")
            }
        }

    with patch.object(support_bot_graph, "astream_events", side_effect=mock_astream_events):
        client = TestClient(app)
        response = client.post(
            "/chat/stream",
            json={"messages": [{"role": "user", "content": "Hi streaming"}]}
        )
        
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        
        content_lines = response.text.split("\n")
        tokens = []
        for line in content_lines:
            if line.startswith("data: "):
                data = json.loads(line[6:])
                if "token" in data:
                    tokens.append(data["token"])
        
        assert "".join(tokens) == "Mocked stream."


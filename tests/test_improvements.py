import pytest
from fastapi.testclient import TestClient
from src.api.main import app
from src.graph.workflow import support_bot_graph
from langgraph.graph.state import CompiledStateGraph

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

import pytest
from src.core.config import settings

def test_config_loading():
    """Verify that settings are initialized (even if env is missing, defaults should exist)."""
    assert settings.MODEL_NAME == "gpt-4o"
    assert settings.COLLECTION_NAME == "customer_support_kb"

def test_retrieval_tool_docstring():
    """Verify tool metadata."""
    from src.tools.retrieval_tool import retrieve_company_info
    assert "retrieve_company_info" in str(retrieve_company_info.name)
    assert "knowledge base" in retrieve_company_info.description

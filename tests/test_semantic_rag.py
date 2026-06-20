"""Unit tests for Semantic Chunking and Parent-Document Retrieval."""

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from src.database.vector_store import SemanticChunker, ParentDocumentRetriever

def test_semantic_chunker_sentence_splitting() -> None:
    """Verify that SemanticChunker.split_sentences correctly splits text while respecting abbreviations."""
    mock_embeddings = MagicMock()
    chunker = SemanticChunker(mock_embeddings)

    text = "Hello world! This is Acme Corp. Please see e.g. section 4. Is this working?"
    sentences = chunker.split_sentences(text)
    
    assert len(sentences) == 3
    assert sentences[0] == "Hello world!"
    assert sentences[1] == "This is Acme Corp. Please see e.g. section 4."
    assert sentences[2] == "Is this working?"

def test_semantic_chunker_split_text() -> None:
    """Verify that SemanticChunker.split_text groups sentences based on embedding similarities."""
    mock_embeddings = MagicMock()
    
    # 3 sentences: s1 and s2 are very similar, s3 is completely different
    sentences = [
        "Acme support helps clients reset widget devices.",
        "To reset your widget, press and hold the power button.",
        "Baking sourdough bread requires flour, water, and salt."
    ]
    
    # Mock embeddings returning vectors
    mock_embeddings.embed_documents.return_value = [
        [1.0, 1.0, 0.0],  # s1
        [1.0, 0.9, 0.0],  # s2 (similar to s1)
        [0.0, 0.0, 1.0]   # s3 (dissimilar)
    ]
    
    chunker = SemanticChunker(mock_embeddings)
    chunks = chunker.split_text("\n".join(sentences))
    
    # Should group s1 & s2, and split s3 into a separate chunk
    assert len(chunks) == 2
    assert "Acme support" in chunks[0]
    assert "reset your widget" in chunks[0]
    assert "Baking sourdough" in chunks[1]

@patch("src.database.vector_store.get_parent_document")
def test_parent_document_retriever(mock_get_parent) -> None:
    """Verify that ParentDocumentRetriever fetches the parent document content instead of child chunks."""
    mock_get_parent.return_value = "This is the full parent document text detailing widget resets and setups."
    
    mock_vs = MagicMock()
    mock_vs_retriever = MagicMock()
    
    child_doc = Document(
        page_content="widget resets",
        metadata={"parent_id": "widget_doc_0", "source": "widget.txt", "department": "general"}
    )
    mock_vs_retriever.invoke.return_value = [child_doc]
    mock_vs.as_retriever.return_value = mock_vs_retriever
    
    retriever = ParentDocumentRetriever(
        vector_store=mock_vs,
        department="general",
        k=3
    )
    
    docs = retriever.invoke("reset widget")
    
    assert len(docs) == 1
    # Check that retrieved doc content is the parent content, not child chunk content
    assert docs[0].page_content == "This is the full parent document text detailing widget resets and setups."
    assert docs[0].metadata["parent_id"] == "widget_doc_0"
    assert docs[0].metadata["source"] == "widget.txt"
    mock_get_parent.assert_called_once_with("widget_doc_0")

@patch("src.database.vector_store.get_parent_document")
def test_parent_document_retriever_fallback(mock_get_parent) -> None:
    """Verify that ParentDocumentRetriever falls back to the child document if the parent is missing from SQLite."""
    mock_get_parent.return_value = None  # Parent not found in DB
    
    mock_vs = MagicMock()
    mock_vs_retriever = MagicMock()
    
    child_doc = Document(
        page_content="widget resets",
        metadata={"parent_id": "widget_doc_0", "source": "widget.txt", "department": "general"}
    )
    mock_vs_retriever.invoke.return_value = [child_doc]
    mock_vs.as_retriever.return_value = mock_vs_retriever
    
    retriever = ParentDocumentRetriever(
        vector_store=mock_vs,
        department="general",
        k=3
    )
    
    docs = retriever.invoke("reset widget")
    
    assert len(docs) == 1
    # Content should be the child chunk content due to fallback
    assert docs[0].page_content == "widget resets"

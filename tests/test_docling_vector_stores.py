import os
import shutil
import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document
from src.core.config import settings
from src.database.pdf_parser import parse_pdf_with_docling
from src.database.vector_store import VectorStoreManager, ParentDocumentRetriever

# Check for optional packages
try:
    import docling
    HAS_DOCLING = True
except ImportError:
    HAS_DOCLING = False

try:
    import langchain_qdrant
    HAS_QDRANT = True
except ImportError:
    HAS_QDRANT = False

# Mock OpenAI API Key for testing
os.environ["OPENAI_API_KEY"] = "mock-openai-key"

@pytest.mark.skipif(not HAS_DOCLING, reason="docling not installed")
def test_pdf_parser_docling_success() -> None:
    """Verify that Docling parses PDFs correctly when installed."""
    mock_converter = MagicMock()
    mock_result = MagicMock()
    mock_result.document.export_to_markdown.return_value = "# Header\nSome parsed PDF markdown text."
    mock_result.document.name = "test.pdf"
    mock_converter.convert.return_value = mock_result

    with patch("docling.document_converter.DocumentConverter", return_value=mock_converter, create=True):
        docs = parse_pdf_with_docling("dummy_path.pdf")
        assert len(docs) == 1
        assert "# Header" in docs[0].page_content
        assert docs[0].metadata["source"] == "dummy_path.pdf"
        assert docs[0].metadata["parser"] == "docling"

@pytest.mark.skipif(not HAS_DOCLING, reason="docling not installed")
def test_pdf_parser_docling_fallback() -> None:
    """Verify that PyPDF fallback is invoked when Docling imports/runs fail."""
    # Force ImportError
    with patch("docling.document_converter.DocumentConverter", side_effect=ImportError("No module named 'docling'"), create=True):
        mock_pypdf_loader = MagicMock()
        mock_pypdf_loader.load.return_value = [Document(page_content="Fallback page text", metadata={"source": "dummy_path.pdf"})]
        
        with patch("langchain_community.document_loaders.PyPDFLoader", return_value=mock_pypdf_loader):
            docs = parse_pdf_with_docling("dummy_path.pdf")
            assert len(docs) == 1
            assert "Fallback page text" in docs[0].page_content
            assert docs[0].metadata["parser"] == "pypdf"

@pytest.fixture
def clean_db_dirs():
    """Fixture to ensure vector db test directories are cleaned before and after tests."""
    qdrant_test_dir = "./qdrant_test_db"
    faiss_test_dir = "./faiss_test_db"
    
    for folder in [qdrant_test_dir, faiss_test_dir]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
            
    yield qdrant_test_dir, faiss_test_dir
    
    for folder in [qdrant_test_dir, faiss_test_dir]:
        if os.path.exists(folder):
            try:
                shutil.rmtree(folder)
            except Exception:
                pass

@patch("langchain_openai.OpenAIEmbeddings")
@pytest.mark.skipif(not HAS_QDRANT, reason="langchain_qdrant not installed")
def test_vector_store_manager_qdrant(mock_embeddings_cls, clean_db_dirs) -> None:
    """Verify Qdrant vector store initialization and indexing."""
    qdrant_test_dir, _ = clean_db_dirs
    mock_embeddings = MagicMock()
    mock_embeddings_cls.return_value = mock_embeddings

    # Temporarily adjust configuration to test Qdrant with local path
    settings.VECTOR_DB_TYPE = "qdrant"
    settings.QDRANT_PATH = qdrant_test_dir

    manager = VectorStoreManager()
    
    dummy_docs = [
        Document(page_content="Acme customer service returns are free within 30 days.", metadata={"source": "test.txt"})
    ]
    
    # We mock out the actual Qdrant client or langchain_qdrant to avoid hitting real embedding API/libs if not fully installed.
    mock_qdrant_vector_store = MagicMock()
    mock_retriever = MagicMock()
    mock_qdrant_vector_store.as_retriever.return_value = mock_retriever
    
    with patch("langchain_qdrant.QdrantVectorStore.from_documents", return_value=mock_qdrant_vector_store, create=True) as mock_from_docs:
        manager._initialize_store(documents=dummy_docs)
        mock_from_docs.assert_called_once()
        retriever = manager.get_retriever()
        assert isinstance(retriever, ParentDocumentRetriever)
        assert retriever.vector_store == mock_qdrant_vector_store

@patch("langchain_openai.OpenAIEmbeddings")
def test_vector_store_manager_faiss(mock_embeddings_cls, clean_db_dirs) -> None:
    """Verify FAISS vector store initialization and indexing."""
    _, faiss_test_dir = clean_db_dirs
    mock_embeddings = MagicMock()
    mock_embeddings_cls.return_value = mock_embeddings

    # Adjust config for FAISS
    settings.VECTOR_DB_TYPE = "faiss"
    settings.FAISS_INDEX_PATH = faiss_test_dir

    manager = VectorStoreManager()
    
    dummy_docs = [
        Document(page_content="Acme support is available 24/7 via chat.", metadata={"source": "test.txt"})
    ]
    
    # Mock FAISS to avoid actual native build dependencies issues during simple unit testing
    mock_faiss_store = MagicMock()
    mock_retriever = MagicMock()
    mock_faiss_store.as_retriever.return_value = mock_retriever
    
    with patch("langchain_community.vectorstores.FAISS.from_documents", return_value=mock_faiss_store) as mock_faiss_from_docs:
        manager._initialize_store(documents=dummy_docs)
        mock_faiss_from_docs.assert_called_once()
        mock_faiss_store.save_local.assert_called_with(faiss_test_dir)
        
        retriever = manager.get_retriever()
        assert isinstance(retriever, ParentDocumentRetriever)
        assert retriever.vector_store == mock_faiss_store

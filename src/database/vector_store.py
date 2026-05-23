"""Module for managing vector databases (Qdrant or FAISS) for document RAG.

Handles loading, parsing with Docling, chunking, and embedding texts into the
configured vector store.
"""

import logging
import os
from typing import List, Optional
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever

from src.core.config import settings
from src.database.pdf_parser import parse_pdf_with_docling

# Configure logging
logger = logging.getLogger(__name__)

class VectorStoreManager:
    """Manages document ingestion, chunking, and vector database indexing.

    Supports Qdrant and FAISS vector stores, configurable via settings.

    Attributes:
        embeddings: OpenAIEmbeddings instance used to embed documents.
        vector_store: Active vector store instance (Qdrant or FAISS).
    """

    def __init__(self) -> None:
        """Initializes the VectorStoreManager with the local embeddings model."""
        self.embeddings = OpenAIEmbeddings(
            api_key="local-placeholder",
            model=settings.EMBEDDING_MODEL,
            openai_api_base=settings.LOCAL_EMBEDDING_BASE_URL
        )
        self.vector_store: Optional[VectorStore] = None

    def _initialize_store(self, documents: Optional[List[Document]] = None) -> VectorStore:
        """Initializes the vector store based on the active configuration.

        Args:
            documents: Optional list of documents to build the store with.

        Returns:
            The initialized VectorStore instance.
        """
        db_type = settings.VECTOR_DB_TYPE.lower()
        logger.info("Initializing vector store of type: %s", db_type)

        if db_type == "qdrant":
            from langchain_qdrant import QdrantVectorStore
            from qdrant_client import QdrantClient

            if documents:
                # Build vector store from documents
                self.vector_store = QdrantVectorStore.from_documents(
                    documents=documents,
                    embedding=self.embeddings,
                    path=settings.QDRANT_PATH,
                    url=settings.QDRANT_URL,
                    collection_name=settings.COLLECTION_NAME
                )
            else:
                # Load existing client
                if settings.QDRANT_URL:
                    client = QdrantClient(url=settings.QDRANT_URL)
                else:
                    client = QdrantClient(path=settings.QDRANT_PATH)
                self.vector_store = QdrantVectorStore(
                    client=client,
                    collection_name=settings.COLLECTION_NAME,
                    embeddings=self.embeddings
                )
            return self.vector_store

        elif db_type == "faiss":
            from langchain_community.vectorstores import FAISS

            if documents:
                self.vector_store = FAISS.from_documents(documents, self.embeddings)
                self.vector_store.save_local(settings.FAISS_INDEX_PATH)
            else:
                if os.path.exists(os.path.join(settings.FAISS_INDEX_PATH, "index.faiss")):
                    self.vector_store = FAISS.load_local(
                        settings.FAISS_INDEX_PATH,
                        self.embeddings,
                        allow_dangerous_deserialization=True
                    )
                else:
                    logger.warning("FAISS index not found at %s. Creating empty index.", settings.FAISS_INDEX_PATH)
                    dummy_doc = Document(page_content="Initial index placeholder", metadata={"source": "system", "department": "general"})
                    self.vector_store = FAISS.from_documents([dummy_doc], self.embeddings)
                    self.vector_store.save_local(settings.FAISS_INDEX_PATH)
            return self.vector_store

        else:
            raise ValueError(f"Unsupported vector database type: {db_type}")

    def load_and_index_documents(self, data_dir: str = "data") -> None:
        """Recursively loads documents from data_dir, chunks them, and indexes them.

        PDF files are parsed using Docling (falling back to PyPDF), text/md files
        use standard loaders, and Word/CSV/Excel loaders are used for other files.
        Documents are automatically tagged with a department based on subfolders.

        Args:
            data_dir: Path to directory containing files.
        """
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            logger.info("Created data directory: %s", data_dir)
            return

        docs: List[Document] = []
        for root, _, files in os.walk(data_dir):
            # Extract department from folder path (e.g. data/sales -> sales, data -> general)
            relative_path = os.path.relpath(root, data_dir)
            if relative_path == ".":
                department = "general"
            else:
                department = relative_path.split(os.sep)[0].lower()

            for file in files:
                file_path = os.path.join(root, file)
                loaded_docs: List[Document] = []

                if file.endswith(".pdf"):
                    loaded_docs = parse_pdf_with_docling(file_path)
                elif file.endswith((".txt", ".md")):
                    try:
                        from langchain_community.document_loaders import TextLoader
                        loader = TextLoader(file_path, encoding="utf-8")
                        loaded_docs = loader.load()
                    except Exception as e:
                        logger.error("Failed to load text file %s: %s", file_path, e)
                elif file.endswith(".docx"):
                    try:
                        from langchain_community.document_loaders import Docx2txtLoader
                        loader = Docx2txtLoader(file_path)
                        loaded_docs = loader.load()
                    except Exception as e:
                        logger.error("Failed to load Word file %s: %s", file_path, e)
                elif file.endswith(".csv"):
                    try:
                        from langchain_community.document_loaders import CSVLoader
                        loader = CSVLoader(file_path, encoding="utf-8")
                        loaded_docs = loader.load()
                    except Exception as e:
                        logger.error("Failed to load CSV file %s: %s", file_path, e)
                elif file.endswith((".xlsx", ".xls")):
                    try:
                        import pandas as pd
                        df = pd.read_excel(file_path)
                        text = df.to_string(index=False)
                        loaded_docs = [Document(page_content=text, metadata={"source": file_path})]
                    except Exception as e:
                        logger.error("Failed to load Excel file %s: %s", file_path, e)

                # Attach department metadata to each document
                for doc in loaded_docs:
                    doc.metadata["department"] = department
                    if "source" not in doc.metadata:
                        doc.metadata["source"] = file_path
                
                docs.extend(loaded_docs)

        if not docs:
            logger.warning("No documents found in %s to index.", data_dir)
            return

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        splits = text_splitter.split_documents(docs)

        # Initialize and populate vector store
        self._initialize_store(documents=splits)
        logger.info("Successfully indexed %d chunks in %s vector store.", len(splits), settings.VECTOR_DB_TYPE)

    def get_retriever(self, department: Optional[str] = None) -> VectorStoreRetriever:
        """Returns a retriever interface for the configured vector store.

        Args:
            department: Optional department name to filter documents by.

        Returns:
            A LangChain VectorStoreRetriever configured to fetch the top 3 most relevant documents.
        """
        if not self.vector_store:
            self._initialize_store()
        
        search_kwargs = {"k": 3}
        if department and department.lower() != "general":
            search_kwargs["filter"] = {"department": department.lower()}

        return self.vector_store.as_retriever(search_kwargs=search_kwargs)

vector_store_manager: VectorStoreManager = VectorStoreManager()

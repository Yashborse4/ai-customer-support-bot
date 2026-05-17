import logging
import os
from typing import List, Optional
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from src.core.config import settings

# Configure logging
logger = logging.getLogger(__name__)

class VectorStoreManager:
    """Manages the RAG vector store using ChromaDB and OpenAI Embeddings.

    Attributes:
        embeddings: OpenAIEmbeddings instance used to generate document embeddings.
        vector_store: Chroma instance representing the underlying vector database.
    """

    def __init__(self) -> None:
        """Initializes the VectorStoreManager with configuration settings."""
        self.embeddings = OpenAIEmbeddings(
            api_key=settings.OPENAI_API_KEY,
            model=settings.EMBEDDING_MODEL
        )
        self.vector_store: Optional[Chroma] = None

    def load_and_index_documents(self, data_dir: str = "data") -> None:
        """Loads documents from the specified directory, splits them, and indexes them in ChromaDB.

        Args:
            data_dir: Path to the directory containing documents to load and index.
        """
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            logger.info("Created data directory: %s", data_dir)
            return

        # Loaders for different file types
        pdf_loader = DirectoryLoader(data_dir, glob="**/*.pdf", loader_cls=PyPDFLoader)
        txt_loader = DirectoryLoader(data_dir, glob="**/*.txt", loader_cls=TextLoader)

        docs: List[Document] = []
        docs.extend(pdf_loader.load())
        docs.extend(txt_loader.load())

        if not docs:
            logger.warning("No documents found to index.")
            return

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        splits = text_splitter.split_documents(docs)

        # Create/Update Vector Store
        self.vector_store = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory=settings.PERSIST_DIRECTORY,
            collection_name=settings.COLLECTION_NAME
        )
        logger.info("Successfully indexed %d document chunks.", len(splits))

    def get_retriever(self) -> VectorStoreRetriever:
        """Returns a retriever interface for the compiled vector store.

        Returns:
            A LangChain VectorStoreRetriever configured to return the top 3 most relevant documents.
        """
        if not self.vector_store:
            self.vector_store = Chroma(
                persist_directory=settings.PERSIST_DIRECTORY,
                embedding_function=self.embeddings,
                collection_name=settings.COLLECTION_NAME
            )
        return self.vector_store.as_retriever(search_kwargs={"k": 3})

vector_store_manager: VectorStoreManager = VectorStoreManager()

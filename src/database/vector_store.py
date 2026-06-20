"""Module for managing vector databases (Qdrant or FAISS) for document RAG.

Handles loading, parsing with Docling, chunking, and embedding texts into the
configured vector store.
"""

import logging
import os
import re
import math
from typing import List, Optional, Any
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun

from src.core.config import settings
from src.database.pdf_parser import parse_pdf_with_docling
from src.database.sql_db import save_parent_document, get_parent_document

# Configure logging
logger = logging.getLogger(__name__)

class SemanticChunker:
    """Splits text based on semantic shifts using sentence embeddings similarity."""
    def __init__(self, embeddings: OpenAIEmbeddings) -> None:
        self.embeddings = embeddings

    def split_sentences(self, text: str) -> List[str]:
        # Split text on sentence endings (., !, ?) ignoring common abbreviations
        raw_sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = []
        temp = ""
        for s in raw_sentences:
            if temp:
                s = temp + " " + s
                temp = ""
            
            # If the segment ends with a period, check if it's an abbreviation
            if s.endswith("."):
                abbreviations = ("dr.", "mr.", "mrs.", "ms.", "jr.", "sr.", "vs.", "e.g.", "i.e.", "acme.", "corp.", "inc.", "ltd.", "co.")
                s_lower = s.lower()
                words = s_lower.strip().split()
                last_word = words[-1] if words else ""
                
                # Merge if it ends with an abbreviation or a single letter (like middle initial)
                if any(s_lower.endswith(abbr) for abbr in abbreviations) or (len(last_word) == 2 and last_word[0].isalpha()):
                    temp = s
                    continue
                    
            sentences.append(s.strip())
            
        if temp:
            if sentences:
                sentences[-1] = sentences[-1] + " " + temp
            else:
                sentences.append(temp.strip())
        return [s for s in sentences if s]

    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        dot_product = sum(x * y for x, y in zip(v1, v2))
        norm_v1 = math.sqrt(sum(x * x for x in v1))
        norm_v2 = math.sqrt(sum(x * x for x in v2))
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        return dot_product / (norm_v1 * norm_v2)

    def split_text(self, text: str) -> List[str]:
        """Splits a single block of text into semantically cohesive chunks."""
        sentences = self.split_sentences(text)
        if not sentences:
            return []
        if len(sentences) == 1:
            return sentences

        # Batch embed all sentences to prevent multiple requests
        try:
            embeddings = self.embeddings.embed_documents(sentences)
        except Exception as e:
            logger.error("Failed to generate sentence embeddings for semantic chunking: %s. Falling back to simple split.", e)
            return sentences

        similarities = []
        for i in range(len(embeddings) - 1):
            sim = self._cosine_similarity(embeddings[i], embeddings[i+1])
            similarities.append(sim)

        # Set dynamic threshold at the 20th percentile (lowest 20% represent boundaries)
        if similarities:
            sorted_sims = sorted(similarities)
            percentile_idx = int(len(sorted_sims) * 0.20)
            threshold = sorted_sims[percentile_idx]
            # Clamp threshold to sensible limits
            threshold = max(0.65, min(0.85, threshold))
        else:
            threshold = 0.75

        chunks = []
        current_chunk = [sentences[0]]
        current_len = len(sentences[0])

        for i in range(len(sentences) - 1):
            sim = similarities[i]
            next_sentence = sentences[i+1]
            # Length guardrail: split if similarity drops or if chunk size exceeds 600 characters
            if sim < threshold or current_len + len(next_sentence) > 600:
                chunks.append(" ".join(current_chunk))
                current_chunk = [next_sentence]
                current_len = len(next_sentence)
            else:
                current_chunk.append(next_sentence)
                current_len += len(next_sentence) + 1

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

class ParentDocumentRetriever(BaseRetriever):
    """Custom retriever that searches child chunks but returns parent documents."""
    vector_store: Any
    department: Optional[str] = None
    k: int = 3

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        search_kwargs = {"k": self.k}
        if self.department and self.department.lower() != "general":
            search_kwargs["filter"] = {"department": self.department.lower()}

        retriever = self.vector_store.as_retriever(search_kwargs=search_kwargs)
        child_docs = retriever.invoke(query)

        parent_docs = []
        seen_parent_ids = set()

        for doc in child_docs:
            parent_id = doc.metadata.get("parent_id")
            if not parent_id:
                # Fallback to child chunk if no parent ID is mapped
                parent_docs.append(doc)
                continue

            if parent_id in seen_parent_ids:
                continue
            seen_parent_ids.add(parent_id)

            parent_content = get_parent_document(parent_id)
            if parent_content:
                parent_doc = Document(
                    page_content=parent_content,
                    metadata={
                        "source": doc.metadata.get("source"),
                        "department": doc.metadata.get("department"),
                        "parent_id": parent_id
                    }
                )
                parent_docs.append(parent_doc)
            else:
                # Fallback to child chunk if parent cannot be loaded
                parent_docs.append(doc)

        return parent_docs

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

        # 1. Split raw documents into large parent documents (chunk_size=2000)
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200
        )
        parent_docs = parent_splitter.split_documents(docs)

        # 2. Initialize the SemanticChunker
        semantic_chunker = SemanticChunker(self.embeddings)

        child_splits: List[Document] = []
        for idx, parent_doc in enumerate(parent_docs):
            source_file = parent_doc.metadata.get("source", "unknown")
            file_basename = os.path.basename(source_file)
            parent_id = f"{file_basename}_{idx}"
            
            # Save parent document content persistently in SQLite
            save_parent_document(
                parent_id=parent_id,
                content=parent_doc.page_content,
                source=source_file,
                department=parent_doc.metadata.get("department", "general")
            )

            # Split parent document into smaller child chunks using Semantic Chunking
            child_contents = semantic_chunker.split_text(parent_doc.page_content)
            for child_idx, child_content in enumerate(child_contents):
                child_doc = Document(
                    page_content=child_content,
                    metadata={
                        "parent_id": parent_id,
                        "source": source_file,
                        "department": parent_doc.metadata.get("department", "general"),
                        "child_index": child_idx
                    }
                )
                child_splits.append(child_doc)

        # 3. Initialize and populate vector store with the child splits
        self._initialize_store(documents=child_splits)
        logger.info("Successfully indexed %d child chunks (from %d parent documents) in %s vector store.", len(child_splits), len(parent_docs), settings.VECTOR_DB_TYPE)

    def get_retriever(self, department: Optional[str] = None) -> ParentDocumentRetriever:
        """Returns a custom parent-document retriever interface for the configured vector store.

        Args:
            department: Optional department name to filter documents by.

        Returns:
            A custom ParentDocumentRetriever configured to return parent sections for the top matches.
        """
        if not self.vector_store:
            self._initialize_store()
        
        return ParentDocumentRetriever(
            vector_store=self.vector_store,
            department=department,
            k=3
        )

vector_store_manager: VectorStoreManager = VectorStoreManager()

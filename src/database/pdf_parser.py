"""Module for parsing PDF documents using Docling with a PyPDF fallback.

Provides layout-aware conversion of PDFs to Markdown, falling back to basic text
extraction if docling is not available or fails.
"""

import logging
from typing import List
from langchain_core.documents import Document

# Configure logging
logger = logging.getLogger(__name__)

def parse_pdf_with_docling(file_path: str) -> List[Document]:
    """Parses a PDF document into LangChain Document instances.

    Tries to use Docling for a rich layout-aware Markdown extraction.
    If Docling is not installed or fails for any reason, falls back to PyPDFLoader.

    Args:
        file_path: The absolute or relative path to the PDF file.

    Returns:
        A list of LangChain Document objects containing the extracted text/markdown.
    """
    try:
        # Dynamically import docling so failures in PyTorch/Docling installation
        # do not break the startup of the application.
        from docling.document_converter import DocumentConverter

        logger.info("Attempting to parse PDF with Docling: %s", file_path)
        converter = DocumentConverter()
        result = converter.convert(file_path)
        markdown_text = result.document.export_to_markdown()

        # Build metadata with source and parser info
        metadata = {
            "source": file_path,
            "parser": "docling",
            "title": result.document.name or ""
        }

        # Wrap in a single document representing the entire parsed text.
        # It will be chunked later by the text splitter.
        return [Document(page_content=markdown_text, metadata=metadata)]

    except ImportError:
        logger.warning(
            "Docling is not installed. Falling back to PyPDFLoader for: %s",
            file_path
        )
        return _fallback_pypdf_loader(file_path)
    except Exception as e:
        logger.error(
            "Docling failed to parse %s (error: %s). Falling back to PyPDFLoader.",
            file_path,
            e
        )
        return _fallback_pypdf_loader(file_path)

def _fallback_pypdf_loader(file_path: str) -> List[Document]:
    """Fallback PDF loader using langchain_community PyPDFLoader.

    Args:
        file_path: The path to the PDF file.

    Returns:
        A list of parsed LangChain Document pages.
    """
    try:
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        for doc in docs:
            doc.metadata["parser"] = "pypdf"
        return docs
    except Exception as e:
        logger.error("Fallback PyPDFLoader also failed for %s: %s", file_path, e)
        return []

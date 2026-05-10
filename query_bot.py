import asyncio
import os
from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from src.core.config import settings
from src.database.vector_store import vector_store_manager

class RAGSource(BaseModel):
    """
    Data model representing a RAG document source chunk.

    Attributes:
        content: Content of the document chunk.
        source: Source file name or path.
    """
    content: str = Field(description="Content of the document chunk")
    source: str = Field(description="Source file name or path")

class QueryResult(BaseModel):
    """
    Data model representing the final answer along with RAG and Model metadata.

    Attributes:
        question: User's original question.
        answer: LLM's answered output.
        model_name: LLM model name used.
        embedding_model: Embedding model name used.
        sources: List of document sources retrieved.
    """
    question: str = Field(description="User's original question")
    answer: str = Field(description="LLM's answered output")
    model_name: str = Field(description="LLM model name used")
    embedding_model: str = Field(description="Embedding model name used")
    sources: List[RAGSource] = Field(description="List of document sources retrieved")

async def perform_query(question: str) -> QueryResult:
    """
    Retrieves context for a given question and gets an answer from OpenAI.

    Args:
        question: The user's query/question string.

    Returns:
        A QueryResult object containing the answer and full metadata.
    """
    # 1. Retrieve relevant document chunks
    retriever = vector_store_manager.get_retriever()
    retrieved_docs = await retriever.ainvoke(question)

    # 2. Extract sources and build context string
    sources: List[RAGSource] = []
    context_parts: List[str] = []
    for doc in retrieved_docs:
        source_name = os.path.basename(doc.metadata.get("source", "Unknown"))
        sources.append(RAGSource(content=doc.page_content, source=source_name))
        context_parts.append(f"[Source: {source_name}]\n{doc.page_content}")

    context_str = "\n\n---\n\n".join(context_parts)

    # 3. Formulate the prompt
    system_prompt = (
        "You are a helpful customer support assistant for Acme Corp.\n"
        "Use the following retrieved context from our knowledge base to answer the user's question.\n"
        "If the answer cannot be found in the context, tell the user politely that you do not know.\n\n"
        f"### Retrieved Context:\n{context_str}"
    )

    # 4. Call ChatOpenAI
    chat = ChatOpenAI(
        model_name=settings.MODEL_NAME,
        api_key=settings.OPENAI_API_KEY,
        temperature=0.2
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=question)
    ]

    response = await chat.ainvoke(messages)
    
    # Return complete result structure
    return QueryResult(
        question=question,
        answer=str(response.content),
        model_name=settings.MODEL_NAME,
        embedding_model=settings.EMBEDDING_MODEL,
        sources=sources
    )

async def main() -> None:
    """
    Main interactive entry point for the query bot script.
    """
    load_dotenv()
    
    # Ensure OPENAI_API_KEY is available
    if not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") == "your_openai_api_key_here":
        print("[WARNING] OPENAI_API_KEY is not set or is using placeholder in your .env file.")
        print("Please configure a valid OpenAI API key to get responses from the model.\n")

    print("==================================================")
    print("      Acme Corp Interactive RAG Query Bot         ")
    print("==================================================")

    # Load and index documents in data/ on startup
    print("Initializing Vector Store and indexing 'data/' directory...")
    vector_store_manager.load_and_index_documents("data")
    print("Initialization Complete.\n")

    while True:
        try:
            question = input("Enter your question (or type 'exit' to quit): ").strip()
            if not question:
                continue
            if question.lower() in ["exit", "quit", "bye"]:
                print("Goodbye!")
                break

            print("\nSearching knowledge base and generating answer...")
            result = await perform_query(question)

            print("\n------------------ MODEL & RAG INFO ------------------")
            print(f" LLM Model:        {result.model_name}")
            print(f" Embedding Model:  {result.embedding_model}")
            print(f" Retrieved Chunks: {len(result.sources)}")
            for idx, src in enumerate(result.sources, 1):
                print(f"   [{idx}] Source: {src.source}")
                # Print a small snippet of the retrieved chunk
                snippet = src.content.replace('\n', ' ')[:120] + "..."
                print(f"       Snippet: {snippet}")
            print("------------------------------------------------------")

            print("\n====================== ANSWER ======================")
            print(result.answer)
            print("====================================================\n")

        except Exception as e:
            print(f"\n[ERROR] An error occurred: {e}\n")

if __name__ == "__main__":
    asyncio.run(main())

from langchain.tools import tool
from src.database.vector_store import vector_store_manager

@tool
def retrieve_company_info(query: str, department: str = "general") -> str:
    """Retrieves relevant company information, product details, and policies from the knowledge base.

    Use this tool whenever a customer asks about products, shipping, returns, or technical support.

    Args:
        query: The semantic search query string to look up.
        department: Optional department scope (e.g., 'sales', 'support', 'general') to filter documents.

    Returns:
        A concatenated block of retrieved source documents with their file origins,
        or a message indicating that no information was found.
    """
    retriever = vector_store_manager.get_retriever(department=department)
    docs = retriever.invoke(query)
    
    if not docs:
        return f"No relevant information found in the {department} knowledge base."
    
    context = "\n\n".join([f"Source: {d.metadata.get('source')}\nContent: {d.page_content}" for d in docs])
    return context

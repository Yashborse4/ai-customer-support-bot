from langchain.tools import tool
from src.database.vector_store import vector_store_manager

@tool
def retrieve_company_info(query: str) -> str:
    """
    Retrieves relevant company information, product details, and policies from the knowledge base.
    Use this tool whenever a customer asks about products, shipping, returns, or technical support.
    """
    retriever = vector_store_manager.get_retriever()
    docs = retriever.invoke(query)
    
    if not docs:
        return "No relevant information found in the knowledge base."
    
    context = "\n\n".join([f"Source: {d.metadata.get('source')}\nContent: {d.page_content}" for d in docs])
    return context
